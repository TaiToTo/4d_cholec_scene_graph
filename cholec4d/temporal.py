"""How tracked regions behave over a window: persistence, identity switches, fragmentation.

`labels` is {frame: (H, W) int32 region ids}, `gts` {annotated frame:
class map} and `valids` {frame: valid-depth mask}. Region ids are the
tracker's, so one id is one tracked region over the window.
"""
import numpy as np

from cholec4d import settings
from cholec4d.metrics import match, regions

CASES = ("kept", "drifted", "lingered", "lost", "vanished")


# --- Persistence -------------------------------------------------------------

def reference(region, labels, gts, valids, seed):
    """Return a region's reference frame, class and purity.

    The reference frame is the annotated frame nearest the seed where the
    region is present (the seed frame itself in every window of the paper);
    the class is the one most of the region's valid pixels lie on there, and
    the purity that class's share.

    Returns:
        (frame, class, purity), or (None, None, None) when the region is on
        no annotated frame.
    """
    alive = [f for f in sorted(gts) if (labels[f] == region).any()]
    if not alive:
        return None, None, None
    f = min(alive, key=lambda a: (abs(a - seed), a))
    m = (labels[f] == region) & valids[f]
    n = int(m.sum())
    if n == 0:
        return f, None, 0.0
    counts = np.bincount(gts[f][m].ravel(), minlength=settings.N_CLASSES)
    c = int(counts.argmax())
    return f, c, counts[c] / n


def frame_case(mask, gt, valid, ref):
    """Classify a region on one annotated frame against its reference class.

    kept: present, at least KEPT_FRACTION of it on the class, which is on
    the frame; drifted: present, the class on the frame, the region mostly
    elsewhere; lingered: present, the class gone; lost: absent, the class on
    the frame; vanished: both absent. A class is on the frame when it covers
    a valid pixel.
    """
    shown = int(((gt == ref) & valid).sum()) >= 1
    if not mask.any():
        return "lost" if shown else "vanished"
    if not shown:
        return "lingered"
    on = (mask & valid).sum()
    share = ((mask & valid & (gt == ref)).sum() / on) if on else 0.0
    return "kept" if share >= settings.KEPT_FRACTION else "drifted"


def persistence(labels, gts, valids, seed):
    """Count each seeded region's cases over the annotated frames other than its reference.

    Returns:
        A dict: the share of each case, the number of observations judged,
        the regions seeded, those whose reference class is background (left
        out), and for each SEED_PURITY cut the share of drifted observations
        that come from regions under it.
    """
    ids = [int(r) for r in np.unique(labels[seed]) if r >= 0]
    total = dict.fromkeys(CASES, 0)
    drifted_mixed = dict.fromkeys(settings.SEED_PURITY, 0)
    background = 0
    for r in ids:
        f_ref, ref, purity = reference(r, labels, gts, valids, seed)
        if f_ref is None:
            continue
        if ref is None or ref == settings.BACKGROUND:
            background += 1
            continue
        for f in sorted(gts):
            if f != f_ref:
                case = frame_case(labels[f] == r, gts[f], valids[f], ref)
                total[case] += 1
                for cut in settings.SEED_PURITY:
                    drifted_mixed[cut] += case == "drifted" and purity < cut
    judged = sum(total.values())
    out = {f"share_{k}": (v / judged if judged else None) for k, v in total.items()}
    out.update(judged=judged, seeded_regions=len(ids), background_regions=background)
    for cut in settings.SEED_PURITY:
        out[f"drifted_from_mixed@{cut}"] = (drifted_mixed[cut] / total["drifted"]
                                            if total["drifted"] else None)
    return out


# --- Identity ----------------------------------------------------------------

def link_instances(insts, frames):
    """Link ground-truth instances into tracks across consecutive annotated frames.

    A pair of instances of one class on consecutive annotated frames is
    linked when it overlaps, largest IoU first, one to one.

    Args:
        insts: {frame: [(mask, class), ...]}.
        frames: The annotated frames, sorted.

    Returns:
        {frame: [track id of each instance]}.
    """
    tracks, next_id = {}, 0
    for k, f in enumerate(frames):
        ids = [None] * len(insts[f])
        if k:
            prev = frames[k - 1]
            pairs = []
            for i, (mi, ci) in enumerate(insts[f]):
                for j, (mj, cj) in enumerate(insts[prev]):
                    if ci == cj:
                        u = int((mi | mj).sum())
                        iou = int((mi & mj).sum()) / u if u else 0.0
                        if iou > 0.0:
                            pairs.append((iou, i, j))
            used_i, used_j = set(), set()
            for iou, i, j in sorted(pairs, reverse=True):
                if i not in used_i and j not in used_j:
                    used_i.add(i)
                    used_j.add(j)
                    ids[i] = tracks[prev][j]
        for i in range(len(ids)):
            if ids[i] is None:
                ids[i] = next_id
                next_id += 1
        tracks[f] = ids
    return tracks


def matched_instances(labels, valid, insts):
    """Return {region id: instance index} for the matches with IoU >= MATCH_IOU."""
    regs = regions(labels, valid)
    pairs = match([m for m, _ in insts], [m for m, _ in regs])
    return {regs[j][1]: i for iou, i, j in pairs if iou >= settings.MATCH_IOU}


def reappearances(present, seed):
    """Return (vanish frame, return frame, direction) for each time a region comes back.

    Starting at the seed, a region that was present and is then absent on a
    sampled frame reappears when it is present again further out.
    """
    n = len(present)
    events = []
    for direction, frames in (("forward", range(seed + 1, n)), ("backward", range(seed - 1, -1, -1))):
        gone, seen = None, present[seed]
        for f in frames:
            if not present[f] and seen and gone is None:
                gone = f
            if present[f] and gone is not None:
                events.append((gone, f, direction))
                gone = None
            seen = seen or present[f]
    return events


def identity(labels, gts, valids, insts, seed):
    """Measure class switches, fragmentation and reappearance in one window.

    Returns:
        class_switch_rate (switches over the transitions between consecutive
        annotated frames on which a region is matched on both),
        matched_pairs, fragmentation_mean (regions matched to one GT track),
        gt_tracks_matched, reappear_same_class and reappear_events_judged.
    """
    frames = sorted(gts)
    tracks = link_instances(insts, frames)
    matches = {f: matched_instances(labels[f], valids[f], insts[f]) for f in frames}
    track_of = {f: {r: tracks[f][i] for r, i in matches[f].items()} for f in frames}
    class_of = {f: {r: insts[f][i][1] for r, i in matches[f].items()} for f in frames}

    pairs = switches = 0
    for a, b in zip(frames, frames[1:]):
        for r in set(track_of[a]) & set(track_of[b]):
            pairs += 1
            switches += class_of[a][r] != class_of[b][r]

    per_track = {}
    for f in frames:
        for r, t in track_of[f].items():
            per_track.setdefault(t, set()).add(r)
    frag = [len(s) for s in per_track.values()]

    n = len(labels)
    same = []
    for r in [int(x) for x in np.unique(labels[seed]) if x >= 0]:
        here = [bool((labels[f] == r).any()) for f in range(n)]
        for gone, back, direction in reappearances(here, seed):
            step, end = (1, n) if direction == "forward" else (-1, -1)
            before = [f for f in range(seed, gone, step) if r in track_of.get(f, {})]
            after = [f for f in range(back, end, step) if r in track_of.get(f, {})]
            if before and after:
                same.append(class_of[before[-1]][r] == class_of[after[0]][r])
    return dict(class_switch_rate=switches / pairs if pairs else None, matched_pairs=pairs,
                fragmentation_mean=float(np.mean(frag)) if frag else None,
                gt_tracks_matched=len(frag),
                reappear_same_class=float(np.mean(same)) if same else None,
                reappear_events_judged=len(same))
