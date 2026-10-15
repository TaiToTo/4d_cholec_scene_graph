"""Predict each window's depth, intrinsics and camera poses with Depth Anything 3.

DA3-LARGE takes a window's 30 cropped frames as one sequence and returns,
per frame, a depth map, its confidence, the intrinsics and the
world-to-camera pose, all at the processing resolution (longest side 504).
The world frame is the camera of DA3's reference view; it is fixed to the
middle view, which is the seed frame, and checked: that view's pose must be
the identity and stand clear of every other.

Usage:
    python -m cholec4d.depth --gpu 0 [--windows W000 ...] [--shard 0/4]
"""
import argparse
import os

import numpy as np

from cholec4d import settings
from cholec4d.data import Work, default_work, pick, read_windows, write_atomic

# The runner-up must lie this many times farther from the identity.
REFERENCE_SEPARATION = 3.0


def check_reference(extrinsics, view):
    """Raise unless `view` is the one camera at the identity.

    Raises:
        ValueError: Another camera is nearer the identity, or not clearly farther.
    """
    e = np.asarray(extrinsics, np.float64)[:, :3, :4]
    dist = np.abs(e - np.hstack([np.eye(3), np.zeros((3, 1))])).max(axis=(1, 2))
    order = np.argsort(dist)
    if order[0] != view or dist[order[1]] < REFERENCE_SEPARATION * dist[order[0]]:
        raise ValueError(f"view {view} is not the reference: distances {np.round(dist, 6)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", default=default_work())
    ap.add_argument("--windows", nargs="*")
    ap.add_argument("--shard")
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from depth_anything_3.api import DepthAnything3

    work = Work(args.work)
    todo = [w for w in pick(read_windows(), args.windows, args.shard)
            if not work.depth(w["clip"]).exists()]
    print(f"{len(todo)} window(s) to estimate", flush=True)
    if not todo:
        return
    model = DepthAnything3.from_pretrained(settings.DEPTH_MODEL).to(
        "cuda" if torch.cuda.is_available() else "cpu")
    for i, w in enumerate(todo, 1):
        frames = sorted(work.window(w["clip"]).joinpath("frames").glob("*.png"))
        if len(frames) != settings.SAMPLES:
            raise FileNotFoundError(f"{w['clip']}: {len(frames)} frames; run cholec4d.extract")
        pred = model.inference([str(p) for p in frames], process_res=settings.DEPTH_PROCESS_RES,
                               ref_view_strategy=settings.DEPTH_REFERENCE_VIEW)
        ext = np.asarray(pred.extrinsics)
        ext = ext[:, :3, :] if ext.shape[-2] == 4 else ext
        check_reference(ext, settings.SEED_INDEX)
        arrays = dict(depth=np.asarray(pred.depth, np.float32), conf=np.asarray(pred.conf, np.float32),
                      extrinsics=ext.astype(np.float32),
                      intrinsics=np.asarray(pred.intrinsics, np.float32))
        write_atomic(work.depth(w["clip"]), lambda p: np.savez(p, **arrays))
        print(f"[{i}/{len(todo)}] {w['clip']}: depth {arrays['depth'].shape}", flush=True)


if __name__ == "__main__":
    main()
