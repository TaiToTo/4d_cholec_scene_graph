"""Frame numbering and the crop rectangle."""
import numpy as np
import pytest

from cholec4d.extract import carry_rate, largest_rectangle


def test_rate_is_carried_with_halves_up():
    located = {0: 0, 30: 25}                  # numbered at 30 a second, video at 25
    out = carry_rate([0, 15, 30, 45, -15], located)
    assert out[15] == 13                      # 12.5 rounds up
    assert out[45] == 38
    assert out[-15] == -12                    # -12.5 rounds up too


def test_unknown_rate_is_refused():
    with pytest.raises(ValueError):
        carry_rate([0, 10, 20], {0: 0, 20: 10})


def test_largest_rectangle():
    inside = np.zeros((8, 10), bool)
    inside[2:7, 1:9] = True
    inside[4, 0] = True
    assert largest_rectangle(inside) == (2, 7, 1, 9)
