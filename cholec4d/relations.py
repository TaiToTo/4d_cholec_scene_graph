"""Whether the graph's relations and events agree with the annotation.

The graph relates two nodes by the direction between their centroids
(`geometry.relation`). The same rule is applied to the annotated instances
the nodes are matched to, on the same depth, so what is checked is whether
the node stands for the right structure:

- 2D relations: over the pairs of regions matched to two different
  instances (IoU >= MATCH_IOU, one to one, as instance F1 matches), the
  share whose relation from the image-plane centroids (right, left, above,
  below) equals the instances'; `pair_coverage` is the share of all region
  pairs on a frame that are matched pairs with a 3D relation on both sides;
- relation changes: for a pair matched on two consecutive annotated frames
  to the same two ground-truth tracks, a change is a relation that differs
  from the earlier frame's; precision is the share of the regions' changes
  the instances also make, recall the converse;
- disappearances: between consecutive annotated frames, a region named by
  a class (the class most of its pixels lie on) disappears; precision is
  the share of those whose class has left the frame too.

Shares pool the counts over a window's frames.
"""
import numpy as np

from cholec4d import settings
from cholec4d.geometry import RELATIONS_2D, RELATIONS_3D, centroids_2d, centroids_3d, relation
from cholec4d.metrics import index_map, majority_class
from cholec4d.temporal import link_instances, matched_instances


def _rel(pos, a, b, axes):
    return relation(pos[a], pos[b], axes) if a in pos and b in pos else None


def _ratio(num, den):
    return num / den if den else None


def relations(labels, gts, valids, insts, depth, K):
    """Measure relation agreement, relation changes and disappearances in one window.

    Args:
        labels: {frame: (H, W) int32 region ids}.
        gts: {annotated frame: class map}.
        valids: {frame: valid mask}.
        insts: {annotated frame: [(mask, class)]}.
        depth: (N, H, W) float32.
        K: (N, 3, 3) float64.

    Returns:
        A dict of shares and the counts behind them.
    """
    frames = sorted(gts)
    tracks = link_instances(insts, frames)
    h, w = depth.shape[1:]
    rel, named, alive, shown = {}, {}, {}, {}
    pairs3, pairs2, n_pairs = [], [], 0
    for f in frames:
        lab, valid = labels[f], valids[f]
        inst_map = index_map([m for m, _ in insts[f]], (h, w)).astype(np.int32)
        node_map = np.where(lab >= 0, lab + 1, 0).astype(np.int32)
        g3, g2 = centroids_3d(depth[f], K[f], inst_map), centroids_2d(inst_map, valid)
        p3, p2 = centroids_3d(depth[f], K[f], node_map), centroids_2d(node_map, valid)
        match = matched_instances(lab, valid, insts[f])
        named[f] = {r: c for r, c in majority_class(lab, gts[f], valid).items()
                    if c != settings.BACKGROUND}
        alive[f] = {int(r) for r in np.unique(lab) if r >= 0}
        shown[f] = {int(c) for c in np.unique(gts[f][valid]) if c != settings.BACKGROUND}
        present = [int(r) for r in np.unique(lab[valid]) if r >= 0]
        n_pairs += len(present) * (len(present) - 1) // 2
        rel[f] = {}
        matched = sorted(match)
        for i, a in enumerate(matched):
            for b in matched[i + 1:]:
                ga, gb = match[a] + 1, match[b] + 1
                r3 = (_rel(p3, a + 1, b + 1, RELATIONS_3D), _rel(g3, ga, gb, RELATIONS_3D))
                r2 = (_rel(p2, a + 1, b + 1, RELATIONS_2D), _rel(g2, ga, gb, RELATIONS_2D))
                rel[f][(a, b)] = (r2, tracks[f][ga - 1], tracks[f][gb - 1])
                if None not in r3:
                    pairs3.append(r3)
                if None not in r2:
                    pairs2.append(r2)

    changes = dict(pred=0, gt=0, both=0)
    vanish = dict(pred=0, right=0)
    for a, b in zip(frames, frames[1:]):
        for pair in set(rel[a]) & set(rel[b]):
            (p0, g0), ta, tb = rel[a][pair]
            (p1, g1), ua, ub = rel[b][pair]
            if (ta, tb) != (ua, ub) or None in (p0, p1, g0, g1):
                continue
            changes["pred"] += p0 != p1
            changes["gt"] += g0 != g1
            changes["both"] += p0 != p1 and g0 != g1
        for r, c in named[a].items():
            gone = r not in alive[b]
            vanish["pred"] += gone
            vanish["right"] += gone and c not in shown[b]

    return dict(agree_2d=float(np.mean([p == g for p, g in pairs2])) if pairs2 else None,
                pairs_2d=len(pairs2), pairs_3d=len(pairs3), pairs_all=n_pairs,
                pair_coverage=_ratio(len(pairs3), n_pairs),
                change_precision_2d=_ratio(changes["both"], changes["pred"]),
                change_recall_2d=_ratio(changes["both"], changes["gt"]),
                changes_pred_2d=changes["pred"], changes_gt_2d=changes["gt"],
                changes_both_2d=changes["both"],
                vanish_precision=_ratio(vanish["right"], vanish["pred"]),
                vanish_pred=vanish["pred"], vanish_right=vanish["right"])
