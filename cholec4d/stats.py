"""Means over windows with 95 % bootstrap intervals over videos.

Windows of one video are not independent, so the video is the resampling
unit: each of the N_BOOT resamples draws as many videos as there are, with
replacement, and averages all windows of the drawn videos. The interval is
the 2.5th and 97.5th percentile of those averages. A difference between
two conditions is taken window by window first.

The generator is seeded from the values and the videos themselves (with
BOOT_SEED), so the same input always gives the same interval, whatever
else is computed before it. Values are visited in the order of their clip
names.
"""
import hashlib

import numpy as np
import pandas as pd

from cholec4d import settings
from cholec4d.data import video_of


def _generator(values, videos):
    h = hashlib.blake2b(np.ascontiguousarray(values, dtype=np.float64).tobytes(), digest_size=8)
    h.update(b"|")
    h.update("\x00".join(map(str, videos)).encode())
    return np.random.default_rng([settings.BOOT_SEED, int.from_bytes(h.digest(), "big")])


def video_interval(values, videos):
    """Return the 95 % interval of the mean of `values`, resampling `videos`."""
    values = np.asarray(values, dtype=np.float64)
    videos = np.asarray(videos)
    rng = _generator(values, videos)
    names = np.unique(videos)
    groups = [values[videos == v] for v in names]
    means = np.empty(settings.N_BOOT)
    for b in range(settings.N_BOOT):
        drawn = rng.integers(0, len(names), size=len(names))
        means[b] = np.concatenate([groups[i] for i in drawn]).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mean_interval(values):
    """Return (mean, low, high, windows) of a Series indexed by clip; NaN windows are left out."""
    s = values.dropna().sort_index()
    v = s.to_numpy(float)
    lo, hi = video_interval(v, [video_of(c) for c in s.index])
    return float(v.mean()), lo, hi, len(v)


def difference_interval(a, b):
    """Return (mean of a - b, low, high, windows) over the windows both Series hold."""
    both = pd.concat({"a": a, "b": b}, axis=1).dropna().sort_index()
    d = (both["a"] - both["b"]).to_numpy(float)
    lo, hi = video_interval(d, [video_of(c) for c in both.index])
    return float(d.mean()), lo, hi, len(d)
