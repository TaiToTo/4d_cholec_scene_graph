"""The bootstrap gives the same interval for the same input, whatever ran before it."""
import numpy as np
import pandas as pd

from cholec4d.stats import difference_interval, mean_interval, video_interval


def windows():
    rng = np.random.default_rng(1)
    clips = [f"VID{v:02d}_s15_{k}" for v in (1, 9, 12, 17) for k in (100, 200, 300)]
    return pd.Series(rng.normal(size=len(clips)), index=clips)


def test_interval_depends_on_the_input_only():
    s = windows()
    first = mean_interval(s)
    video_interval([1.0, 2.0, 3.0], ["a", "b", "c"])
    assert mean_interval(s) == first
    assert mean_interval(s.iloc[::-1]) == first     # windows are visited in clip order


def test_interval_brackets_the_mean():
    m, lo, hi, n = mean_interval(windows())
    assert lo < m < hi and n == 12


def test_difference_is_taken_per_window():
    s = windows()
    m, lo, hi, n = difference_interval(s + 1.0, s)
    assert np.isclose(m, 1.0) and np.isclose(lo, 1.0) and np.isclose(hi, 1.0)
