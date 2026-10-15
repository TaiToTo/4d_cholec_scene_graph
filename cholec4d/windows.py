"""Define one evaluation window per CholecSeg8k clip, with no selection by content.

A CholecSeg8k clip is a folder of 80 consecutive annotated frames. Its
window holds SAMPLES CholecSeg8k numbers STRIDE apart, with the clip's
middle annotated frame (the earlier of the two middle ones) at SEED_INDEX.
A window is excluded when one of its frames falls outside the video or when
fewer than MIN_ANNOTATED of its frames are annotated. Of CholecSeg8k's 101
clips, 86 give a window; the other 15 would start before their video does.

The repository ships the result as `windows/windows.csv` and
`windows/excluded.csv`; this command writes them again from the datasets.

Usage:
    python -m cholec4d.windows --cholecseg8k <CholecSeg8k root> --cholec80 <cholec80 videos>
"""
import argparse
import csv
import os
import re

import cv2

from cholec4d import settings
from cholec4d.data import REPO


def cholecseg8k_clips(root):
    """Return each CholecSeg8k clip as (video number, first number, annotated frames).

    Raises:
        ValueError: A clip folder is not named `video<nn>_<first number>`.
    """
    clips = []
    for vdir in sorted(d for d in os.listdir(root) if re.fullmatch(r"video\d{2}", d)):
        for cdir in sorted(os.listdir(os.path.join(root, vdir))):
            if cdir.startswith("."):
                continue
            m = re.fullmatch(r"video(\d{2})_(\d+)", cdir)
            if m is None:
                raise ValueError(f"unexpected CholecSeg8k folder {vdir}/{cdir}")
            n = sum(f.endswith("_endo_color_mask.png") for f in os.listdir(os.path.join(root, vdir, cdir)))
            clips.append((int(m.group(1)), int(m.group(2)), n))
    return clips


def frame_count(videos, video):
    """Return the number of frames of a Cholec80 video."""
    cap = cv2.VideoCapture(os.path.join(videos, f"video{video:02d}.mp4"))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video{video:02d}.mp4 under {videos}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def define_windows(clips, lengths):
    """Give every clip a window and decide whether it is kept.

    Args:
        clips: As `cholecseg8k_clips` returns them.
        lengths: {video number: frame count}.

    Returns:
        (kept, excluded): lists of dict rows; the excluded carry a reason.
    """
    annotated = {}
    for v, start, n in clips:
        annotated.setdefault(v, set()).update(range(start, start + n))
    kept, excluded = [], []
    for v, start, n in clips:
        seed = start + (n - 1) // 2
        numbers = [seed + settings.STRIDE * (k - settings.SEED_INDEX) for k in range(settings.SAMPLES)]
        n_ann = sum(x in annotated[v] for x in numbers)
        per_number = 1.0 if v in settings.VIDEOS_AT_25FPS else 25 / 30
        row = dict(clip=f"VID{v:02d}_s{settings.STRIDE}_{numbers[0]}", video=f"VID{v:02d}",
                   cholecseg8k_clip=f"video{v:02d}_{start:05d}", seed_number=seed,
                   n_annotated=n_ann, numbers=" ".join(map(str, numbers)))
        why = []
        if numbers[0] < 0:
            why.append("starts before the first frame of the video")
        if int(numbers[-1] * per_number) >= lengths[v]:
            why.append("ends after the last frame of the video")
        if n_ann < settings.MIN_ANNOTATED:
            why.append(f"fewer than {settings.MIN_ANNOTATED} annotated frames")
        if why:
            excluded.append(dict(row, reason="; ".join(why)))
        else:
            kept.append(row)
    for i, row in enumerate(kept):
        row["window_id"] = f"W{i:03d}"
    return kept, excluded


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cholecseg8k", default=os.environ.get("CHOLECSEG8K_ROOT"))
    ap.add_argument("--cholec80", default=os.environ.get("CHOLEC80_VIDEOS"))
    ap.add_argument("--out", default=str(REPO / "windows"))
    args = ap.parse_args()
    if not args.cholecseg8k or not args.cholec80:
        ap.error("give --cholecseg8k and --cholec80 (or set CHOLECSEG8K_ROOT and CHOLEC80_VIDEOS)")

    clips = cholecseg8k_clips(args.cholecseg8k)
    lengths = {v: frame_count(args.cholec80, v) for v in sorted({c[0] for c in clips})}
    kept, excluded = define_windows(clips, lengths)

    cols = ["window_id", "clip", "video", "cholecseg8k_clip", "seed_number", "n_annotated", "numbers"]
    os.makedirs(args.out, exist_ok=True)
    for name, rows, fields in (("windows.csv", kept, cols),
                               ("excluded.csv", excluded, cols[1:] + ["reason"])):
        with open(os.path.join(args.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    print(f"{len(clips)} clips: {len(kept)} windows, {len(excluded)} excluded -> {args.out}")


if __name__ == "__main__":
    main()
