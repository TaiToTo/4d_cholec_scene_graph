"""Cut each window's frames and masks out of CholecSeg8k and Cholec80, and crop them.

An annotated frame takes the image CholecSeg8k drew the mask on, so frame
and mask are aligned; its Cholec80 frame is found by matching that image
against the video. An unannotated frame is decoded from the video. Twelve
of the 17 videos are numbered by CholecSeg8k at about 30 frames a second,
so the video frame of an unannotated number is interpolated between the
window's annotated frames (or carried on at their rate past the first or
the last), rounding halves up.

Every frame and mask of a window is then cropped to one rectangle: the
largest one inside the pixels whose luminance, at the 90th percentile over
the window's frames, reaches 15, which excludes the black border outside
the endoscope's field of view.

Usage:
    python -m cholec4d.extract --cholecseg8k <root> --cholec80 <videos> [--windows W000 ...] [--workers 8]
"""
import argparse
import json
import math
import os
import shutil
from fractions import Fraction
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cholec4d import settings
from cholec4d.data import Work, default_work, pick, read_windows


def annotation_index(cholecseg8k, video):
    """Map each annotated CholecSeg8k number of a video to its (image, mask) paths.

    Raises:
        FileNotFoundError: A mask has no image beside it.
    """
    out = {}
    vdir = Path(cholecseg8k) / f"video{video:02d}"
    for chunk in sorted(p for p in vdir.iterdir() if p.is_dir()):
        for mask in chunk.glob("frame_*_endo_color_mask.png"):
            number = int(mask.name[len("frame_"):-len("_endo_color_mask.png")])
            image = chunk / f"frame_{number}_endo.png"
            if not image.is_file():
                raise FileNotFoundError(f"{mask} has no image {image.name}")
            out.setdefault(number, (image, mask))
    return out


def imread(path):
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    return img


def locate(cap, n_frames, targets):
    """Find the video frame each annotated image is, by the smallest mean absolute difference.

    Each image of CholecSeg8k number s is searched for in
    [s - s // 5 - 150, s + 60], clipped to the video. The video is decoded
    once, in order, over the union of the searches.

    Args:
        cap: The open video.
        n_frames: Its frame count.
        targets: {CholecSeg8k number: BGR image}.

    Returns:
        {CholecSeg8k number: video frame}.

    Raises:
        ValueError: An image matches no frame closely enough, or matches one
            on an edge of its search that is not an end of the video.
    """
    spans = {}
    for s in targets:
        lo = max(0, s - s // settings.SEARCH_BACK_DIVISOR - settings.SEARCH_BACK)
        hi = min(s + settings.SEARCH_AHEAD, n_frames - 1)
        if lo > hi:
            raise ValueError(f"number {s}: the search [{lo}, {hi}] lies past the video")
        spans[s] = (lo, hi)
    best = {s: (math.inf, None) for s in targets}
    start, stop = min(lo for lo, _ in spans.values()), max(hi for _, hi in spans.values())
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for i in range(start, stop + 1):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"cannot decode video frame {i}")
        for s, img in targets.items():
            lo, hi = spans[s]
            if lo <= i <= hi:
                if frame.shape != img.shape:
                    raise ValueError(f"video frame {i} is {frame.shape}, the image {img.shape}")
                d = float(np.mean(np.abs(frame.astype(np.int16) - img.astype(np.int16))))
                if d < best[s][0]:
                    best[s] = (d, i)
    out = {}
    for s, (d, i) in best.items():
        lo, hi = spans[s]
        if d > settings.MATCH_MAX_MAE:
            raise ValueError(f"number {s} matches no video frame (best {i}, difference {d:.2f})")
        if (i == lo and lo > 0) or (i == hi and hi < n_frames - 1):
            raise ValueError(f"number {s} matched frame {i} on the edge of its search [{lo}, {hi}]")
        out[s] = i
    return out


def carry_rate(numbers, located):
    """Give each unannotated number a video frame from the annotated ones around it.

    Between two annotated frames the video frame is interpolated; before the
    first or after the last it is carried on at the rate from the first
    annotated frame to the last. Halves round up.

    Raises:
        ValueError: Fewer than two frames are annotated, or they run at a
            rate CholecSeg8k does not number at.
    """
    known = sorted(located)
    if len(known) < 2:
        raise ValueError("fewer than two annotated frames give no rate")
    span = Fraction(located[known[-1]] - located[known[0]], known[-1] - known[0])
    if not any(abs(float(span) - r) <= settings.RATE_TOLERANCE * r for r in settings.NUMBER_RATES):
        raise ValueError(f"the annotated frames run at {float(span):.4f} video frames a number")
    out = dict(located)
    for s in numbers:
        if s in out:
            continue
        before = [k for k in known if k < s]
        after = [k for k in known if k > s]
        if before and after:
            a, b = before[-1], after[0]
            rate = Fraction(located[b] - located[a], b - a)
        else:
            a, rate = (before[-1] if before else after[0]), span
        out[s] = located[a] + math.floor((s - a) * rate + Fraction(1, 2))
    return out


def largest_rectangle(inside):
    """Return the largest axis-aligned rectangle of True pixels, as (y0, y1, x0, x1).

    Rows are scanned top to bottom, and the bars of each row's histogram
    left to right; the first rectangle of the largest area is kept.
    """
    h, w = inside.shape
    heights = np.zeros(w + 1, np.int64)
    best, best_area = (0, 0, 0, 0), 0
    for y in range(h):
        heights[:w] = np.where(inside[y], heights[:w] + 1, 0)
        stack = []
        for x in range(w + 1):
            while stack and heights[stack[-1]] > heights[x]:
                top = int(heights[stack.pop()])
                left = stack[-1] + 1 if stack else 0
                if top * (x - left) > best_area:
                    best_area = top * (x - left)
                    best = (y + 1 - top, y + 1, left, x)
            stack.append(x)
    return best


def crop_rectangle(frames_rgb):
    """Return the crop of a window from its frames, as (y0, y1, x0, x1)."""
    gray = np.stack([f.mean(axis=2) for f in frames_rgb])
    return largest_rectangle(np.percentile(gray, settings.CROP_PERCENTILE, axis=0)
                             >= settings.CROP_LUMINANCE)


def read_window(window, cholecseg8k, cholec80):
    """Return a window's RGB frames, its annotation paths and the video frame of each number.

    Raises:
        RuntimeError: The video cannot be opened or a frame cannot be decoded.
        ValueError: The frame numbering cannot be resolved (see `locate`, `carry_rate`).
    """
    numbers, video = window["numbers"], int(window["video"][3:])
    cap = cv2.VideoCapture(os.path.join(cholec80, f"video{video:02d}.mp4"))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video{video:02d}.mp4")
        index = annotation_index(cholecseg8k, video)
        annotated = {s: index[s] for s in numbers if s in index}
        images = {s: imread(p[0]) for s, p in annotated.items()}
        natives = carry_rate(numbers, locate(cap, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), images))
        order = [natives[s] for s in numbers]
        if any(b <= a for a, b in zip(order, order[1:])):
            raise ValueError(f"the video frames do not rise: {order}")
        for s in numbers:
            if s not in images:
                cap.set(cv2.CAP_PROP_POS_FRAMES, natives[s])
                ok, images[s] = cap.read()
                if not ok:
                    raise RuntimeError(f"cannot decode video frame {natives[s]}")
    finally:
        cap.release()
    return [np.ascontiguousarray(images[s][..., ::-1]) for s in numbers], annotated, order


def extract_window(args):
    """Extract, crop and write one window; return (clip, error message or None).

    The window is written under a temporary name and moved into place once complete.
    """
    window, cholecseg8k, cholec80, work = args
    clip, numbers = window["clip"], window["numbers"]
    out = Work(work).window(clip)
    if (out / "window.json").exists():
        return clip, None
    partial = out.with_name(f".{clip}.partial")
    shutil.rmtree(partial, ignore_errors=True)
    try:
        frames, annotated, order = read_window(window, cholecseg8k, cholec80)
        y0, y1, x0, x1 = crop_rectangle(frames)
        (partial / "frames").mkdir(parents=True)
        (partial / "masks").mkdir()
        for i, (s, rgb) in enumerate(zip(numbers, frames)):
            Image.fromarray(rgb[y0:y1, x0:x1]).save(partial / "frames" / f"{i:06d}.png")
            if s in annotated:
                mask = imread(annotated[s][1])[..., ::-1]
                Image.fromarray(np.ascontiguousarray(mask[y0:y1, x0:x1])).save(
                    partial / "masks" / f"{i:06d}.png")
        record = dict(clip=clip, window_id=window["window_id"], video=window["video"],
                      numbers=numbers, video_frames=order,
                      annotated=[s in annotated for s in numbers],
                      crop=dict(y0=y0, y1=y1, x0=x0, x1=x1, height=frames[0].shape[0],
                                width=frames[0].shape[1]))
        (partial / "window.json").write_text(json.dumps(record, indent=1))
    except Exception as e:  # noqa: BLE001 - reported per window, the others go on
        shutil.rmtree(partial, ignore_errors=True)
        return clip, f"{type(e).__name__}: {e}"
    shutil.rmtree(out, ignore_errors=True)
    partial.rename(out)
    return clip, None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cholecseg8k", default=os.environ.get("CHOLECSEG8K_ROOT"))
    ap.add_argument("--cholec80", default=os.environ.get("CHOLEC80_VIDEOS"))
    ap.add_argument("--work", default=default_work())
    ap.add_argument("--windows", nargs="*", help="window ids or clips (default: all)")
    ap.add_argument("--shard", help="k/n: every n-th window from the k-th")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if not args.cholecseg8k or not args.cholec80:
        ap.error("give --cholecseg8k and --cholec80 (or set CHOLECSEG8K_ROOT and CHOLEC80_VIDEOS)")

    windows = pick(read_windows(), args.windows, args.shard)
    jobs = [(w, args.cholecseg8k, args.cholec80, args.work) for w in windows]
    failed = []
    with Pool(args.workers) as pool:
        for i, (clip, err) in enumerate(pool.imap_unordered(extract_window, jobs), 1):
            print(f"[{i}/{len(jobs)}] {clip}: {err or 'ok'}", flush=True)
            if err:
                failed.append(clip)
    if failed:
        raise SystemExit(f"{len(failed)} window(s) failed: {' '.join(failed)}")


if __name__ == "__main__":
    main()
