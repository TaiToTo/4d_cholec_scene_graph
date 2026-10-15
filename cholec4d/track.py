"""Track each seed region through its window with SAM 3's mask-prompted tracker.

Every region of the seed map with at least SEED_MIN_AREA pixels becomes a
mask prompt on the seed frame, with object id = region id + 1. The tracker
(SAM 3's SAM 2-style memory tracker, float32) runs on the RGB frames
whatever the seeds were segmented from, first forward from the seed frame
and then backward from it. On each frame the objects' masks are flattened
into one label map, painted in increasing order of the tracker's presence
score so the most confident object wins an overlap. A window's labels are
written as one (frames, H, W) int16 array, -1 where no object lies, ids
equal to the seed region ids.

Usage:
    python -m cholec4d.track --gpu 0 [--pps 24] [--windows W000] [--shard 0/4]
"""
import argparse
import os
import random

import numpy as np

from cholec4d import settings
from cholec4d.data import Work, default_work, pick, read_windows, write_atomic
from cholec4d.segment import load_window_inputs


def seed_prompts(seed_map):
    """Return the seed masks the tracker starts from and their object ids."""
    masks, ids = [], []
    for r in np.unique(seed_map):
        if r < 0:
            continue
        m = seed_map == r
        if int(m.sum()) >= settings.SEED_MIN_AREA:
            masks.append(m)
            ids.append(int(r) + 1)
    return masks, ids


def flatten(masks, object_ids, scores, shape):
    """Paint one frame's object masks into a label map, the highest score last.

    A score the tracker gave no value for paints first.
    """
    lab = np.full(shape, -1, dtype=np.int16)
    s = np.nan_to_num(np.asarray(scores, dtype=float), nan=-1e9)
    for k in np.argsort(s):
        lab[masks[k] > 0] = int(object_ids[k])
    return lab


class Tracker:
    """SAM 3's video tracker, loaded once and run on one window at a time."""

    def __init__(self, torch):
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Sam3TrackerVideoModel.from_pretrained(settings.TRACK_MODEL).to(
            self.device, dtype=torch.float32)
        self.processor = Sam3TrackerVideoProcessor.from_pretrained(settings.TRACK_MODEL)

    def presence(self, session, frame, object_ids):
        """Return each object's presence probability on a frame, NaN where none is stored."""
        out = self.torch.full((len(object_ids),), float("nan"), dtype=self.torch.float32)
        for i, obj in enumerate(object_ids):
            idx = session.obj_id_to_idx(obj)
            logit = None
            for conditioning in (False, True):
                try:
                    logit = session.get_output(idx, frame, "object_score_logits",
                                               is_conditioning_frame=conditioning)
                    break
                except Exception:  # noqa: BLE001 - the output is stored under one of the two
                    continue
            if logit is not None:
                out[i] = float(self.torch.as_tensor(logit).reshape(-1)[0].sigmoid())
        return out

    def run(self, video, seed_index, masks, object_ids):
        """Track the seed masks through the video; return {frame: label map}."""
        torch = self.torch
        h, w = video.shape[1:3]
        session = self.processor.init_video_session(video=video, inference_device=self.device,
                                                    dtype=torch.float32)
        self.processor.add_inputs_to_inference_session(
            inference_session=session, frame_idx=seed_index, obj_ids=object_ids,
            input_masks=[np.asarray(m, dtype=np.uint8) for m in masks], clear_old_inputs=True)
        labels = {}
        with torch.inference_mode():
            for reverse in (False, True):
                for out in self.model.propagate_in_video_iterator(
                        session, start_frame_idx=seed_index, max_frame_num_to_track=None,
                        reverse=reverse):
                    if out.frame_idx in labels:
                        continue
                    logits = out.pred_masks.unsqueeze(1) if out.pred_masks.dim() == 3 else out.pred_masks
                    full = self.processor.post_process_masks(
                        [logits.to(dtype=torch.float32).cpu()], original_sizes=[[h, w]])[0].squeeze(1)
                    ids = list(out.object_ids)
                    labels[out.frame_idx] = flatten(full.cpu().numpy(), ids,
                                                    self.presence(session, out.frame_idx, ids), (h, w))
        return labels


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", default=default_work())
    ap.add_argument("--windows", nargs="*")
    ap.add_argument("--shard")
    ap.add_argument("--pps", type=int, nargs="+", default=list(settings.POINTS_PER_SIDE))
    ap.add_argument("--inputs", nargs="+", default=list(settings.SAM_INPUTS))
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch

    work = Work(args.work)
    jobs = [(w["clip"], pps, s) for w in pick(read_windows(), args.windows, args.shard)
            for pps in args.pps for s in args.inputs if not work.track(w["clip"], s, pps).exists()]
    print(f"{len(jobs)} track(s) to run", flush=True)
    if not jobs:
        return
    tracker = Tracker(torch)
    loaded = None
    for i, (clip, pps, s) in enumerate(jobs, 1):
        seed_file = work.seed(clip, s, pps)
        if not seed_file.exists():
            raise FileNotFoundError(f"{seed_file} is missing; run cholec4d.segment first")
        if loaded is None or loaded[0] != clip:
            rgb, depth, _ = load_window_inputs(work, clip)
            loaded = (clip, np.stack(rgb, 0))
        with np.load(seed_file) as z:
            masks, ids = seed_prompts(z["seed_map"])
        if not masks:
            raise ValueError(f"{clip} {s} pps{pps}: no seed region of {settings.SEED_MIN_AREA} px")
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        # The processor consumes the id list it is given, so it gets a copy.
        labels = tracker.run(loaded[1], settings.SEED_INDEX, masks, list(ids))
        if sorted(labels) != list(range(len(loaded[1]))):
            raise RuntimeError(f"{clip} {s} pps{pps}: frames {sorted(labels)} tracked")
        stack = np.stack([labels[f] for f in range(len(loaded[1]))])
        write_atomic(work.track(clip, s, pps), lambda p: np.save(p, stack))
        print(f"[{i}/{len(jobs)}] {clip} {s} pps{pps}: {len(ids)} regions", flush=True)


if __name__ == "__main__":
    main()
