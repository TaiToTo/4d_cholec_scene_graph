"""Score class-agnostic regions against CholecSeg8k's semantic masks, one frame at a time.

A frame's evaluation domain is its valid-depth pixels. Ground-truth
instances are the connected components (8-connected) of each
non-background class. Regions and instances are matched by IoU alone,
greedily and one to one: pairs in decreasing order of (IoU, instance index,
region index). No area cut applies to either side.
"""
import cv2
import numpy as np

from cholec4d import settings


def present(values):
    """Return the distinct non-negative ints of an array, sorted."""
    values = values.ravel()
    values = values[values >= 0]
    if values.size == 0:
        return []
    return np.flatnonzero(np.bincount(values)).tolist()


def gt_instances(gt, valid):
    """Split a class map into instances: [(mask, class)], background left out."""
    out = []
    for c in present(gt[valid]):
        if c == settings.BACKGROUND:
            continue
        n, comp = cv2.connectedComponents(((gt == c) & valid).astype(np.uint8))
        out += [(comp == i, int(c)) for i in range(1, n)]
    return out


def regions(labels, valid):
    """Return each region with a valid pixel: [(mask, region id)]."""
    return [((labels == r) & valid, int(r)) for r in present(labels[valid])]


def index_map(masks, shape):
    """Number disjoint masks 1..n in one map, 0 where none lies."""
    out = np.zeros(shape, np.int64)
    for k, m in enumerate(masks, 1):
        out[m] = k
    return out


def overlaps(gts, preds):
    """Return (IoU, instance index, region index) for every pair sharing a pixel.

    Both lists hold disjoint masks; one joint histogram of the two index maps
    gives every intersection and, in its margins, every area.
    """
    if not gts or not preds:
        return []
    g = index_map(gts, gts[0].shape).ravel()
    p = index_map(preds, gts[0].shape).ravel()
    joint = np.bincount(g * (len(preds) + 1) + p,
                        minlength=(len(gts) + 1) * (len(preds) + 1)).reshape(len(gts) + 1, -1)
    inter = joint[1:, 1:]
    union = joint[1:].sum(1)[:, None] + joint[:, 1:].sum(0)[None, :] - inter
    return [(float(inter[a, b] / union[a, b]), int(a), int(b)) for a, b in zip(*np.nonzero(inter))]


def match(gts, preds):
    """Match instances to regions greedily, one to one; return [(IoU, instance, region)]."""
    used_g, used_p, out = set(), set(), []
    for iou, i, j in sorted(overlaps(gts, preds), reverse=True):
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        out.append((iou, i, j))
    return out


def majority_class(labels, gt, valid):
    """Give each region the class most of its valid pixels lie on (the smaller id on a tie)."""
    m = valid & (labels >= 0)
    if not m.any():
        return {}
    n_cls = max(settings.N_CLASSES, int(gt.max()) + 1)
    counts = np.bincount(labels[m].astype(np.int64) * n_cls + gt[m],
                         minlength=(int(labels.max()) + 1) * n_cls).reshape(-1, n_cls)
    return {int(r): int(counts[r].argmax()) for r in np.flatnonzero(counts.sum(1)).tolist()}


def oracle_miou(labels, gt, valid):
    """Label each region with its majority class and score the class map by mIoU.

    The mean runs over the non-background classes in the annotation or the
    prediction. Splitting a region changes no label, so this does not
    penalize fragmentation.
    """
    assign = majority_class(labels, gt, valid)
    pred = np.zeros_like(gt)
    if assign:
        ids = np.array(sorted(assign))
        cls = np.array([assign[r] for r in ids])
        hit = np.isin(labels, ids)
        pred[hit] = cls[np.searchsorted(ids, labels[hit])]
    classes = set(present(gt[valid])) | set(present(pred[valid]))
    classes.discard(settings.BACKGROUND)
    ious = []
    for c in classes:
        p = (pred == c) & valid
        g = (gt == c) & valid
        union = int((p | g).sum())
        ious.append(int((p & g).sum()) / union if union else 0.0)
    return float(np.mean(ious)) if ious else 0.0


def merge_per_instance(labels, valid, gts):
    """Merge the regions whose largest overlap is one instance into one region.

    Each region goes to the instance it shares most pixels with (the first
    on a tie) and takes the smallest id among that instance's regions; a
    region on no instance keeps its id.
    """
    inst = index_map(gts, labels.shape) - 1
    on = valid & (labels >= 0) & (inst >= 0)
    out = labels.copy()
    if not on.any():
        return out
    k = len(gts)
    shared = np.bincount(labels[on].astype(np.int64) * k + inst[on],
                         minlength=(int(labels.max()) + 1) * k).reshape(-1, k)
    owner = {r: int(shared[r].argmax()) for r in np.flatnonzero(shared.sum(1)).tolist()}
    first = {}
    for r, j in owner.items():
        first[j] = min(first.get(j, r), r)
    for r, j in owner.items():
        if r != first[j]:
            out[labels == r] = first[j]
    return out


def frame_scores(labels, gt, valid, gts):
    """Score one annotated frame.

    Args:
        labels: (H, W) int32 region ids, -1 for none.
        gt: (H, W) class map.
        valid: (H, W) evaluation domain.
        gts: The frame's instance masks.

    Returns:
        n_reg (regions anywhere in the frame), n_pred (regions in the
        domain), n_gt, oracle mIoU and F1 at each IoU threshold (None when
        the frame has no instance).
    """
    preds = [m for m, _ in regions(labels, valid)]
    ious = [p[0] for p in match(gts, preds)]
    out = dict(n_reg=len(present(labels)), n_pred=len(preds), n_gt=len(gts),
               miou=oracle_miou(labels, gt, valid))
    for t in settings.IOU_THRESHOLDS:
        tp = sum(1 for i in ious if i >= t)
        denom = 2 * tp + (len(preds) - tp) + (len(gts) - tp)
        out[f"f1@{t}"] = (2 * tp / denom if denom else 0.0) if gts else None
    return out
