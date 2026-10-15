"""Every number the paper prints follows from the per-window tables shipped here."""
import pandas as pd

from cholec4d.data import REPO
from cholec4d.numbers import Tables, check, compute


def test_every_printed_number_is_reproduced():
    rows = check(pd.DataFrame(compute(Tables(REPO / "results" / "per_window"))),
                 REPO / "results" / "manuscript.csv")
    bad = rows[~rows["equal"] & (rows["note"] == "")]
    assert bad.empty, bad[["where", "printed", "manuscript"]].to_string()
