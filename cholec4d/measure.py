"""Measure every window from its tracks and write the per-window tables.

Each table under `--out` (default `results/per_window/`) is long: one value
per window, SAM input, points per side and metric.

- `scores.csv`: per annotated frame, then averaged over the window (F1
  over the frames with an instance): instance F1, region and instance
  counts, oracle mIoU and F1 after merging per instance (`merged_*`);
  every points per side;
- `counts.csv`: the same way, the region count minus the frame's instance
  and class counts (`err_*`) and the share of instances covered;
- `sam_masks.csv`: masks SAM returned on the seed frame (`generated`), at
  the run's quality cut and the others (`q<cut>`);
- `mixing.csv`, `persistence.csv` (`static/` rows: the seed mask left in
  place), `identity.csv`, `relations.csv`: at the default points per side;
- `gt.csv`: the frames' instance and class counts.

Usage:
    python -m cholec4d.measure --workers 16
"""
import argparse
import os
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

from cholec4d import settings
from cholec4d.data import REPO, Work, class_map, default_work, read_windows
from cholec4d.geometry import valid_depth
from cholec4d.metrics import (frame_scores, gt_instances, index_map, merge_per_instance, present,
                              regions)
from cholec4d.relations import relations
from cholec4d.temporal import identity, persistence

PPS0 = settings.DEFAULT_POINTS_PER_SIDE


class WindowData:
    """Read one window's depth, annotation, seeds and tracks from the work directory."""

    def __init__(self, work, clip):
        self.work, self.clip = Work(work), clip

    def depth(self):
        with np.load(self.work.depth(self.clip)) as z:
            return z["depth"].astype(np.float32), z["intrinsics"].astype(np.float64)

    def gts(self, n, h, w):
        return {f: class_map(self.work.mask(self.clip, f), h, w)
                for f in range(n) if self.work.mask(self.clip, f).exists()}

    def labels(self, sam_input, pps):
        stack = np.load(self.work.track(self.clip, sam_input, pps)).astype(np.int32)
        return {f: stack[f] for f in range(len(stack))}

    def masks(self, sam_input, pps, quality):
        with np.load(self.work.seed(self.clip, sam_input, pps, quality)) as z:
            return z["predicted_iou"], z["stability_score"]


def mixing_case(counts):
    """Name what a region mixing instances covers, from its pixels' class counts."""
    order = np.argsort(counts)[::-1]
    a, b = int(order[0]), int(order[1])
    if counts[a] >= settings.PURE_FRACTION * counts.sum() or counts[b] == 0:
        return "same_class"
    kind = [("background" if c == settings.BACKGROUND else
             "instrument" if c in settings.INSTRUMENTS else "tissue") for c in (a, b)]
    if "background" in kind:
        return "background"
    return "instrument_tissue" if kind[0] != kind[1] else f"{kind[0]}_{kind[1]}"


def mixing(labels, gts, valids, insts):
    """Share of regions mixing instances, and the share in each mixing case."""
    total, mixed, cases = 0, 0, {}
    for f, gt in gts.items():
        if not insts[f]:
            continue
        inst = index_map([m for m, _ in insts[f]], gt.shape) - 1
        for m, _ in regions(labels[f], valids[f]):
            total += 1
            if np.bincount(inst[m & (inst >= 0)], minlength=len(insts[f])).max() \
                    >= settings.PURE_FRACTION * int(m.sum()):
                continue
            mixed += 1
            case = mixing_case(np.bincount(gt[m], minlength=settings.N_CLASSES))
            cases[case] = cases.get(case, 0) + 1
    if not total:
        return {}
    out = dict(mixing_regions=mixed / total)
    for case in ("same_class", "background", "instrument_tissue", "tissue_tissue",
                 "instrument_instrument"):
        out[f"case_{case}"] = cases.get(case, 0) / total
    return out


def measure_window(args):
    """Measure one window; return {table: rows}."""
    data, inputs, all_pps = args
    cv2.setNumThreads(1)
    clip = data.clip
    depth, K = data.depth()
    n, h, w = depth.shape
    valids = {f: valid_depth(depth[f]) for f in range(n)}
    gts = data.gts(n, h, w)
    insts = {f: gt_instances(g, valids[f]) for f, g in gts.items()}
    out = {k: [] for k in ("frames", "sam_masks", "mixing", "persistence", "identity",
                           "relations", "gt")}
    n_class = {f: sum(1 for c in present(g[valids[f]]) if c != settings.BACKGROUND)
               for f, g in gts.items()}
    for f in gts:
        out["gt"].append(dict(clip=clip, frame=f, gt_inst=len(insts[f]), gt_class=n_class[f]))

    for pps in all_pps:
        for s in inputs:
            base = dict(clip=clip, input=s, pps=pps)
            labels = data.labels(s, pps)
            for f, g in gts.items():
                gm = [m for m, _ in insts[f]]
                row = frame_scores(labels[f], g, valids[f], gm)
                merged = frame_scores(merge_per_instance(labels[f], valids[f], gm), g, valids[f], gm)
                row.update({f"merged_{k}": merged[k] for k in merged if k.startswith("f1@")})
                held = labels[f] >= 0
                covered = sum(1 for m in gm if (m & held).sum() >= settings.COVERED_FRACTION * m.sum())
                row.update(err_inst=row["n_pred"] - len(gm), err_class=row["n_pred"] - n_class[f],
                           covered_inst=covered / len(gm) if gm else np.nan)
                out["frames"].append(dict(base, frame=f, **row))
            # Masks SAM returned at the run's cut, those of them passing a stricter
            # cut, and the masks it returned when run again at a looser one.
            iou, stab = data.masks(s, pps, settings.SAM_PRED_IOU_THRESH)
            counts = dict(generated=len(iou))
            if pps in settings.EXTRA_QUALITY_POINTS_PER_SIDE:
                counts.update({f"kept_q{q:g}": int(((iou >= q) & (stab >= q)).sum())
                               for q in settings.STRICTER_QUALITY})
                counts.update({f"generated_q{q:g}": len(data.masks(s, pps, q)[0])
                               for q in settings.EXTRA_QUALITY})
            out["sam_masks"] += [dict(base, metric=k, value=v) for k, v in counts.items()]
            if pps != PPS0:
                continue
            static = {f: labels[settings.SEED_INDEX] for f in labels}
            tables = (("mixing", mixing(labels, gts, valids, insts)),
                      ("persistence", persistence(labels, gts, valids, settings.SEED_INDEX)),
                      ("persistence", {f"static/{k}": v for k, v in
                                       persistence(static, gts, valids, settings.SEED_INDEX).items()}),
                      ("identity", identity(labels, gts, valids, insts, settings.SEED_INDEX)),
                      ("relations", relations(labels, gts, valids, insts, depth, K)))
            for table, values in tables:
                out[table] += [dict(base, metric=k, value=v) for k, v in values.items()]
    return out


def window_means(frames):
    """Average frame rows into window values, as the paper's run did (pandas' group mean)."""
    keys = ["clip", "input", "pps"]
    cols = [c for c in frames.columns if c not in keys + ["frame"]]
    return frames.groupby(keys)[cols].mean().reset_index().melt(
        id_vars=keys, var_name="metric", value_name="value")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", default=default_work())
    ap.add_argument("--out", default=str(REPO / "results" / "per_window"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    run(args.work, args.out, args.workers, WindowData)


def run(work, out_dir, workers, source):
    """Measure every window with `source(work, clip)` and write the tables."""
    windows = read_windows()
    ids = {w["clip"]: w["window_id"] for w in windows}
    jobs = [(source(work, w["clip"]), settings.SAM_INPUTS, settings.POINTS_PER_SIDE) for w in windows]
    with Pool(workers) as pool:
        parts = pool.map(measure_window, jobs, chunksize=1)
    rows = {k: [r for p in parts for r in p[k]] for k in parts[0]}

    means = window_means(pd.DataFrame(rows.pop("frames")))
    count = means["metric"].isin(["err_inst", "err_class", "covered_inst"])
    tables = {"scores": means[~count], "counts": means[count],
              "gt": pd.DataFrame(rows.pop("gt")).groupby("clip")[["gt_inst", "gt_class"]].mean()
              .reset_index().melt(id_vars="clip", var_name="metric", value_name="value")}
    tables.update({k: pd.DataFrame(v) for k, v in rows.items()})
    os.makedirs(out_dir, exist_ok=True)
    for name, t in tables.items():
        t.insert(0, "window_id", t["clip"].map(ids))
        cols = [c for c in ("window_id", "clip", "input", "pps", "metric", "value") if c in t]
        t[cols].sort_values(cols[:-1]).to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    print(f"wrote {len(tables)} tables for {len(windows)} windows -> {out_dir}")


if __name__ == "__main__":
    main()
