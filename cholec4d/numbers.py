"""Recompute every number the paper prints from the per-window tables, and check it.

Each number is computed from `results/per_window/`, formatted as the paper
prints it, and compared with `results/manuscript.csv`, which holds the
printed text. Intervals are 95 % bootstrap intervals over videos
(`cholec4d.stats`); a claim about an interval (it excludes 0, say) is
checked as a claim. The numbers are written to `results/numbers.csv`.

The paper's run read its window scores (instance F1, region counts, oracle
mIoU) and, for the tracked-against-static difference, its tracked shares
back from CSV files with pandas' default parser, which can change the last
bit of a value. The bootstrap seeds itself from the values' bits, so those
tables are read the same way here; the others are read exactly.

Usage:
    python -m cholec4d.numbers [--tables results/per_window]
"""
import argparse
import os

import numpy as np
import pandas as pd

from cholec4d import settings
from cholec4d.data import REPO
from cholec4d.stats import difference_interval, mean_interval

PPS0 = settings.DEFAULT_POINTS_PER_SIDE


class Tables:
    """Per-window values, one Series indexed by clip per (table, input, pps, metric)."""

    def __init__(self, root):
        def read(name, exact=True):
            return pd.read_csv(os.path.join(root, f"{name}.csv"),
                               float_precision="round_trip" if exact else None)
        self.t = {name: read(name) for name in
                  ("counts", "sam_masks", "mixing", "persistence", "identity", "relations", "gt")}
        self.t["scores"] = read("scores", exact=False)
        self.t["persistence_as_read"] = read("persistence", exact=False)

    def get(self, table, metric, sam_input=None, pps=PPS0):
        t = self.t[table]
        sel = t["metric"] == metric
        if "input" in t:
            sel &= (t["input"] == sam_input) & (t["pps"] == pps)
        s = t[sel].set_index("clip")["value"].sort_index()
        if s.index.has_duplicates:
            raise ValueError(f"{table}/{metric}/{sam_input}@{pps} holds a window twice")
        if s.empty:
            raise KeyError(f"{table} holds no {metric} for {sam_input}@{pps}")
        return s.astype(float)


def fmt(v, d, sign=False):
    return f"{v:+.{d}f}" if sign else f"{v:.{d}f}"


def interval(values, d=3, dl=2, dh=None, sign=False, scale=1.0, unit=""):
    """Format (mean, low, high, n) as `mean [low, high]`."""
    m, lo, hi, _ = values
    dh = dl if dh is None else dh
    return (f"{fmt(scale * m, d, sign)}{unit} [{fmt(scale * lo, dl, sign)}, "
            f"{fmt(scale * hi, dh, sign)}]")


def excludes_zero(values):
    return "excludes 0" if values[1] > 0 or values[2] < 0 else "includes 0"


def compute(tb):
    """Return [(key, where, value, low, high, n, printed)] for every number."""
    rows = []

    def add(key, where, values, printed):
        """Record a number; `values` is a mean, a (mean, low, high, windows) tuple or None."""
        m, lo, hi, n = values if isinstance(values, tuple) else (values, None, None, None)
        rows.append(dict(key=key, where=where, value=m, low=lo, high=hi, n=n, printed=printed))

    def mean(series):
        return float(series.dropna().mean())

    score = lambda s, m, pps=PPS0: tb.get("scores", m, s, pps)  # noqa: E731

    # Table 1 and the modality comparison.
    for label, s, pps in (("RGB", "rgb", PPS0), ("depth", "depth", PPS0), ("normal", "normal", PPS0),
                          ("RGB (pps 8)", "rgb", 8), ("RGB (pps 6)", "rgb", 6)):
        for col, metric, spec in (("F1@.5", "f1@0.5", {}), ("F1@.75", "f1@0.75", {}),
                                  ("n_reg", "n_reg", dict(d=2, dl=1)), ("mIoU oracle", "miou", {}),
                                  ("F1@.5 merged", "merged_f1@0.5", {})):
            v = mean_interval(score(s, metric, pps))
            add(f"table1/{s}@{pps}/{metric}", f"Table 1, {label}, {col}", v, interval(v, **spec))
    diff = lambda a, b, m: difference_interval(score(a[0], m, a[1]), score(b[0], m, b[1]))  # noqa: E731
    v = diff(("normal", PPS0), ("rgb", PPS0), "f1@0.5")
    add("diff/normal-rgb/f1@0.5", "Sec. 5, normal - RGB, F1@.5", v, interval(v, sign=True))
    v = diff(("normal", PPS0), ("depth", PPS0), "f1@0.5")
    add("diff/normal-depth/f1@0.5", "Sec. 5, normal - depth, F1@.5", v,
        interval(v, dl=3, dh=2, sign=True))
    v = diff(("normal", PPS0), ("depth", PPS0), "f1@0.75")
    add("diff/normal-depth/f1@0.75", "Sec. 5, normal - depth at IoU 0.75", v, excludes_zero(v))
    v = diff(("normal", PPS0), ("rgb", 6), "f1@0.5")
    add("diff/normal-rgb@6/f1@0.5", "Sec. 5, normal - RGB at pps 6, F1@.5", v, interval(v, sign=True))

    # Merging.
    for s in ("rgb", "normal"):
        v = mean(tb.get("mixing", "mixing_regions", s))
        add(f"mixing/{s}/straddle", f"Sec. 5, {s} regions straddling instances", v, f"{100 * v:.0f}%")
        v = mean(tb.get("mixing", "case_instrument_tissue", s))
        add(f"mixing/{s}/instrument_tissue", f"Sec. 5, {s} instrument + tissue", v, f"{100 * v:.0f}%")

    # Granularity against the operating point.
    v = mean(score("normal", "n_gt"))
    add("gt/instances", "Sec. 4.2 and Fig. 5, instances per frame", v, f"{v:.1f}")
    v = mean(tb.get("gt", "gt_class"))
    add("gt/classes", "Sec. 4.2, classes per frame", v, f"{v:.1f}")
    small, large = settings.EXTRA_QUALITY_POINTS_PER_SIDE
    masks = lambda m, s, pps: tb.get("sam_masks", m, s, pps)  # noqa: E731
    for s in ("rgb", "normal"):
        v = mean(masks("generated", s, large)) / mean(masks("generated", s, small))
        add(f"sam/{s}/growth", f"Sec. 5, SAM masks pps {small} -> {large}, {s}", v, f"{v:.1f}x")
    # The growth of each window's mask count, normal against RGB, at every quality cut.
    claims = []
    for metric in ([f"generated_q{q:g}" for q in settings.EXTRA_QUALITY] + ["generated"]
                   + [f"kept_q{q:g}" for q in settings.STRICTER_QUALITY[1:]]):
        g = {s: masks(metric, s, large) / masks(metric, s, small).replace(0, np.nan)
             for s in ("normal", "rgb")}
        claims.append(excludes_zero(difference_interval(g["normal"], g["rgb"])))
    add("sam/growth_gap_all_cuts", "Sec. 5, growth gap across quality cuts (not shown)", None,
        "excludes 0" if set(claims) == {"excludes 0"} else "includes 0")
    v = mean(tb.get("counts", "covered_inst", "normal"))
    add("count/normal/covered", "Sec. 5, instances covered by normal", v, f"{100 * v:.0f}%")
    m, lo, hi, n = mean_interval(tb.get("counts", "err_inst", "normal"))
    v = (-m, -hi, -lo, n)
    add("count/normal/below_instances", "Sec. 5, normal below the instance count", v,
        interval(v, d=1, dl=1))
    v = mean_interval(tb.get("counts", "err_class", "normal"))
    add("count/normal/above_classes", "Sec. 5, normal above the class count", v,
        interval(v, d=1, dl=1))
    between = [p for p in settings.POINTS_PER_SIDE
               if mean_interval(tb.get("counts", "err_class", "normal", p))[1] > 0
               and mean_interval(tb.get("counts", "err_inst", "normal", p))[2] < 0]
    add("count/normal/between_from", "Sec. 5, normal between the two counts from pps", None,
        f"pps {min(between)}" if between == [p for p in settings.POINTS_PER_SIDE if p >= min(between)]
        else "not contiguous")
    curve = {p: mean(score("rgb", "f1@0.5", p)) for p in settings.POINTS_PER_SIDE}
    best = max(curve, key=curve.get)
    add("sweep/rgb/peak", "Sec. 5, RGB peak F1@.5", curve[best], f"{curve[best]:.3f} (pps {best})")
    nf = [mean(score("normal", "f1@0.5", p)) for p in settings.POINTS_PER_SIDE]
    add("sweep/normal/range", "Sec. 5, normal F1@.5 range", None, f"{min(nf):.2f}-{max(nf):.2f}")
    v = lovo(tb)
    add("sweep/rgb/lovo", "Sec. 5, RGB tuned leave-one-video-out", v, f"{v:.3f}")

    # Table 2 and the temporal behaviour.
    for s in ("rgb", "depth", "normal"):
        kept = mean(tb.get("persistence", "share_kept", s))
        static = mean(tb.get("persistence", "static/share_kept", s))
        add(f"table2/{s}/kept", f"Table 2, {s}, kept (static)", kept, f"{kept:.3f} ({static:.3f})")
        v = mean(tb.get("identity", "class_switch_rate", s))
        add(f"table2/{s}/switch", f"Table 2, {s}, ID switch", v, f"{100 * v:.1f}%")
        v = mean(tb.get("identity", "fragmentation_mean", s))
        add(f"table2/{s}/frag", f"Table 2, {s}, frag. / GT track", v, f"{v:.3f}")
    v = mean_interval(tb.get("persistence", "share_kept", "normal"))
    add("temporal/normal/kept", "Sec. 5 and abstract, normal kept", v,
        interval(v, d=1, dl=1, scale=100, unit="%"))
    v = difference_interval(tb.get("persistence", "share_kept", "normal"),
                            tb.get("persistence", "share_kept", "rgb"))
    add("temporal/normal-rgb/kept", "Sec. 5, normal - RGB kept", v, excludes_zero(v))
    v = mean(tb.get("persistence", "static/share_kept", "normal"))
    add("temporal/normal/static", "Sec. 5 and abstract, normal static", v, f"{100 * v:.1f}%")
    tracked = tb.get("persistence_as_read", "share_kept", "normal")
    v = difference_interval(tracked, tb.get("persistence", "static/share_kept", "normal"))
    add("temporal/normal/tracked-static", "Sec. 5, tracking contributes", v, interval(v, sign=True))
    events = tb.get("identity", "reappear_events_judged", "normal")
    same = (tb.get("identity", "reappear_same_class", "normal").fillna(0) * events).round()
    add("temporal/normal/reappear", "Sec. 5, returned on the same class", None,
        f"{int(same.sum())} of {int(events.sum())}")
    v = mean(tb.get("relations", "pair_coverage", "normal"))
    add("relations/normal/pairs", "Sec. 5, pairs matched to two instances", v, f"{100 * v:.0f}%")
    v = mean(tb.get("relations", "agree_2d", "normal"))
    add("relations/normal/agree_2d", "Sec. 5 and abstract, 2D relations agree", v, f"{100 * v:.1f}%")
    p = mean(tb.get("relations", "change_precision_2d", "normal"))
    r = mean(tb.get("relations", "change_recall_2d", "normal"))
    add("relations/normal/change", "Sec. 5, relation changes P / R", None, f"{p:.2f} / {r:.2f}")
    v = mean(tb.get("relations", "vanish_precision", "normal"))
    add("relations/normal/vanish_precision", "Sec. 5, disappearance precision", v, f"{v:.2f}")
    return rows


def lovo(tb):
    """Mean F1@.5 of RGB when each video's points per side is chosen on the other videos.

    The choice is the points per side with the highest mean F1@.5 over the
    other videos' windows (the smallest on a tie).
    """
    t = tb.t["scores"]
    t = t[(t["input"] == "rgb") & (t["metric"] == "f1@0.5")].sort_values("clip")
    t = t.assign(video=t["clip"].str.split("_").str[0])
    out = []
    for video, held in t.groupby("video"):
        rest = t[t["video"] != video].groupby("pps")["value"].mean().sort_index()
        pps = int(rest.index[int(np.argmax(rest.to_numpy()))])
        out.append(held[held["pps"] == pps].set_index("clip")["value"])
    return float(pd.concat(out).sort_index().mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tables", default=str(REPO / "results" / "per_window"))
    ap.add_argument("--manuscript", default=str(REPO / "results" / "manuscript.csv"))
    ap.add_argument("--out", default=str(REPO / "results" / "numbers.csv"))
    args = ap.parse_args()

    rows = check(pd.DataFrame(compute(Tables(args.tables))), args.manuscript)
    rows.to_csv(args.out, index=False)
    width = rows["where"].str.len().max()
    for _, r in rows.iterrows():
        tail = "" if r["equal"] else f"   (paper: {r['manuscript']}; {r['note'] or 'no note'})"
        print(f"{'ok  ' if r['equal'] else 'note' if r['note'] else 'DIFF'} "
              f"{r['where']:<{width}}  {r['printed']}{tail}")
    bad = int((~rows["equal"] & (rows["note"] == "")).sum())
    print(f"{int(rows['equal'].sum())} of {len(rows)} numbers equal the paper's, "
          f"{int((~rows['equal']).sum()) - bad} differ as noted, {bad} differ")
    if bad:
        raise SystemExit(1)


def check(rows, manuscript):
    """Add the paper's printed text and note to each number, and whether they are equal.

    Raises:
        ValueError: A number and the manuscript's list do not name the same keys.
    """
    paper = pd.read_csv(manuscript, dtype=str, keep_default_na=False).set_index("key")
    if set(rows["key"]) != set(paper.index):
        raise ValueError(f"keys differ: {sorted(set(rows['key']) ^ set(paper.index))}")
    rows["manuscript"] = rows["key"].map(paper["printed"])
    rows["note"] = rows["key"].map(paper["note"])
    rows["equal"] = rows["printed"] == rows["manuscript"]
    return rows


if __name__ == "__main__":
    main()
