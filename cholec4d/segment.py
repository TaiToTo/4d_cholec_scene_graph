"""Segment each window's seed frame into class-agnostic regions with SAM.

SAM's automatic mask generator (ViT-H) runs on the seed frame's RGB image,
depth image or surface normal map, with a grid of `pps` x `pps` point
prompts. Its masks are painted into one region per pixel, the most
confident mask on top (SAM returns them roughly by predicted IoU), and the
region ids are made consecutive. The seed map and the area, predicted IoU
and stability score of every mask SAM returned are written for each window,
input and grid density.

`--quality` sets both of SAM's quality cuts (predicted IoU and stability);
the run used 0.8 everywhere, and 0.6 and 0.7 on the ends of the grid only
to count masks.

Usage:
    python -m cholec4d.segment --gpu 0 --sam-checkpoint sam_vit_h_4b8939.pth [--pps 24] [--windows W000]
    python -m cholec4d.segment --gpu 0 --sam-checkpoint ... --quality 0.7 --pps 4 48
"""
import argparse
import hashlib
import os
import random

import numpy as np

from cholec4d import settings
from cholec4d.data import Work, default_work, pick, read_windows, write_atomic
from cholec4d.geometry import load_rgb, sam_image, window_depth_range


def paint(masks, shape):
    """Paint SAM's masks into one region id per pixel, the first mask on top.

    Returns:
        (H, W) int64, ids 0..K-1 in the order of the masks' first appearance
        from the bottom, -1 where no mask lies.
    """
    regions = np.full(shape, -1, dtype=np.int64)
    for k, m in enumerate(reversed(masks)):
        regions[m["segmentation"]] = k
    used = np.unique(regions[regions >= 0])
    if used.size == 0:
        return regions
    lut = np.full(used.max() + 2, -1)
    lut[used + 1] = np.arange(used.size)
    return lut[regions + 1]


def load_window_inputs(work, clip):
    """Return a window's RGB frames, depth (float32) and intrinsics (float64), on the depth grid."""
    with np.load(work.depth(clip)) as z:
        depth = z["depth"].astype(np.float32)
        K = z["intrinsics"].astype(np.float64)
    n, h, w = depth.shape
    rgb = [load_rgb(work.frame(clip, i), h, w) for i in range(n)]
    return rgb, depth, K


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", default=default_work())
    ap.add_argument("--sam-checkpoint", default=os.environ.get("SAM_CHECKPOINT"))
    ap.add_argument("--windows", nargs="*")
    ap.add_argument("--shard")
    ap.add_argument("--pps", type=int, nargs="+", default=list(settings.POINTS_PER_SIDE))
    ap.add_argument("--inputs", nargs="+", default=list(settings.SAM_INPUTS))
    ap.add_argument("--quality", type=float, default=settings.SAM_PRED_IOU_THRESH)
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    if not args.sam_checkpoint:
        ap.error("give --sam-checkpoint (or set SAM_CHECKPOINT)")
    if sha256(args.sam_checkpoint) != settings.SAM_CHECKPOINT_SHA256:
        raise SystemExit(f"{args.sam_checkpoint} is not sam_vit_h_4b8939.pth")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    work = Work(args.work)
    jobs = [(w["clip"], pps, s) for w in pick(read_windows(), args.windows, args.shard)
            for pps in args.pps for s in args.inputs
            if not work.seed(w["clip"], s, pps, args.quality).exists()]
    print(f"{len(jobs)} seed map(s) to make", flush=True)
    if not jobs:
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[settings.SAM_MODEL_TYPE](checkpoint=args.sam_checkpoint).to(device)
    generators = {pps: SamAutomaticMaskGenerator(
        sam, points_per_side=pps, pred_iou_thresh=args.quality,
        stability_score_thresh=args.quality,
        min_mask_region_area=settings.SAM_MIN_MASK_REGION_AREA) for pps in args.pps}
    loaded = None
    for i, (clip, pps, s) in enumerate(jobs, 1):
        if loaded is None or loaded[0] != clip:
            rgb, depth, K = load_window_inputs(work, clip)
            loaded = (clip, rgb, depth, K, window_depth_range(depth))
        _, rgb, depth, K, depth01 = loaded
        f = settings.SEED_INDEX
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        masks = generators[pps].generate(sam_image(s, rgb[f], depth[f], K[f], depth01[f]))
        record = dict(
            seed_map=paint(masks, depth.shape[1:]),
            area=np.array([int(m["area"]) for m in masks], np.int64),
            predicted_iou=np.array([float(m["predicted_iou"]) for m in masks], np.float64),
            stability_score=np.array([float(m["stability_score"]) for m in masks], np.float64))
        write_atomic(work.seed(clip, s, pps, args.quality), lambda p: np.savez(p, **record))
        print(f"[{i}/{len(jobs)}] {clip} {s} pps{pps}: {len(masks)} masks", flush=True)


if __name__ == "__main__":
    main()
