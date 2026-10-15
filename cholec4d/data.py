"""CholecSeg8k annotation, the window list, and where each stage writes.

A window is named by its clip, `VID<nn>_s15_<first CholecSeg8k number>`.
Under the work directory each stage writes:

    windows/<clip>/frames/<i>.png        cropped frame i of the window
    windows/<clip>/masks/<i>.png         cropped colour mask, annotated frames only
    windows/<clip>/window.json           frame numbers and the crop
    windows/<clip>/depth.npz             depth, conf, intrinsics, extrinsics
    seeds/q<cut>/pps<N>/<clip>/<input>.npz   SAM's seed map and its masks' scores
    tracks/pps<N>/<clip>/<input>.npy     tracked labels, (frames, H, W) int16, -1 = none
"""
import csv
import os
from pathlib import Path

import cv2
import numpy as np

from cholec4d import settings

REPO = Path(__file__).resolve().parents[1]
WINDOWS_CSV = REPO / "windows" / "windows.csv"

# CholecSeg8k's colour of each class; background has two more colours that
# occur at the edges of some masks. Any other colour reads as background.
PALETTE = {
    (127, 127, 127): 0, (0, 0, 0): 0, (50, 50, 50): 0,
    (210, 140, 140): 1, (255, 114, 114): 2, (231, 70, 156): 3, (186, 183, 75): 4,
    (170, 255, 0): 5, (255, 85, 0): 6, (255, 0, 0): 7, (255, 255, 0): 8,
    (169, 255, 184): 9, (255, 160, 165): 10, (0, 255, 0): 11, (111, 74, 0): 12,
}


def class_map(path, height, width):
    """Read a colour mask as a (height, width) int32 class map.

    The mask is resized by nearest neighbour, so no colour is blended.

    Raises:
        FileNotFoundError: The mask cannot be read.
    """
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"cannot read the mask {path}")
    rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (width, height),
                     interpolation=cv2.INTER_NEAREST)
    key = (rgb[..., 0].astype(np.int64) << 16) | (rgb[..., 1].astype(np.int64) << 8) | rgb[..., 2]
    codes, inverse = np.unique(key.ravel(), return_inverse=True)
    lut = np.array([PALETTE.get((int(c) >> 16, (int(c) >> 8) & 255, int(c) & 255), 0)
                    for c in codes], dtype=np.int32)
    return lut[inverse].reshape(height, width)


def video_of(clip):
    """Return the video a window comes from, `VID<nn>`."""
    return clip.split("_")[0]


def read_windows(path=WINDOWS_CSV):
    """Return the windows as dicts, in the order of the file.

    Each has `window_id`, `clip`, `video`, `seed_number` and `numbers`, the
    window's CholecSeg8k numbers in order.
    """
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["numbers"] = [int(n) for n in r["numbers"].split()]
        r["seed_number"] = int(r["seed_number"])
    return rows


def pick(windows, names=None, shard=None):
    """Select windows by id or clip name, then keep one shard of them.

    Args:
        windows: The windows, as `read_windows` returns them.
        names: Window ids or clip names; None keeps all.
        shard: "k/n" keeps every n-th window starting at the k-th.

    Raises:
        ValueError: A name matches no window, or the shard is malformed.
    """
    if names:
        known = {w["window_id"]: w for w in windows} | {w["clip"]: w for w in windows}
        missing = [n for n in names if n not in known]
        if missing:
            raise ValueError(f"no window named {missing}")
        windows = [known[n] for n in names]
    if shard:
        k, n = (int(x) for x in shard.split("/"))
        if not 0 <= k < n:
            raise ValueError(f"shard {shard} is not k/n with 0 <= k < n")
        windows = windows[k::n]
    return windows


class Work:
    """Paths under the work directory."""

    def __init__(self, root):
        self.root = Path(root)

    def window(self, clip):
        return self.root / "windows" / clip

    def frame(self, clip, i):
        return self.window(clip) / "frames" / f"{i:06d}.png"

    def mask(self, clip, i):
        return self.window(clip) / "masks" / f"{i:06d}.png"

    def depth(self, clip):
        return self.window(clip) / "depth.npz"

    def seed(self, clip, sam_input, pps, quality=settings.SAM_PRED_IOU_THRESH):
        return self.root / "seeds" / f"q{quality:g}" / f"pps{pps}" / clip / f"{sam_input}.npz"

    def track(self, clip, sam_input, pps):
        return self.root / "tracks" / f"pps{pps}" / clip / f"{sam_input}.npy"


def default_work():
    """Return the work directory: `$CHOLEC4D_WORK`, or `work/` in the repository."""
    return os.environ.get("CHOLEC4D_WORK", str(REPO / "work"))


def write_atomic(path, write):
    """Write a file through a temporary name, so a stopped run leaves no partial file.

    Args:
        path: The final path.
        write: A function taking the temporary path; its suffix is kept, so
            `np.save` and `np.savez` add none.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.partial{path.suffix}")
    write(tmp)
    os.replace(tmp, path)
