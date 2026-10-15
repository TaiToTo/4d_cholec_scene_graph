"""Assemble a window's tracked regions into a 3D time-series scene graph.

Each tracked region is a node. On every frame its mask selects the frame's
back-projected pixels (boundary pixels between regions left out); the node
records the centroid of those points in the window's frame (the camera of
the seed frame), their second-moment ellipsoid (axes and standard
deviations along them) and its area fraction (mask pixels over valid
pixels). For each ordered pair of nodes on a frame, the edge stores the 3D
offset between the centroids in that frame's camera (x right, y up, z
towards the camera) and the dominant of its components as the relation.

Events mark the frames at which the record changes discretely: a node
enters or leaves the view, or the relation of a pair changes (recorded once
per pair, from the smaller id to the larger).

Usage:
    python -m cholec4d.graph --windows W000 [--input normal] [--pps 24]
"""
import argparse
import json

import numpy as np

from cholec4d import settings
from cholec4d.data import Work, default_work, pick, read_windows
from cholec4d.geometry import RELATIONS_3D, region_points, relation, to_world, valid_depth


def frame_graph(labels, depth, K, extrinsic):
    """Return one frame's nodes and edges."""
    ids = np.where(labels >= 0, labels + 1, 0).astype(np.int32)
    points, _ = region_points(depth, K, ids)
    valid = valid_depth(depth)
    nodes, centre = {}, {}
    for i, p in points.items():
        world = to_world(p, extrinsic.astype(np.float64))
        mean = world.mean(axis=0)
        var, axes = np.linalg.eigh(np.cov(world.T)) if len(world) >= 3 else (np.zeros(3), np.eye(3))
        order = np.argsort(var)[::-1]
        nodes[i - 1] = dict(id=i - 1, centroid=mean.round(6).tolist(),
                            axes=axes[:, order].T.round(6).tolist(),
                            sd=np.sqrt(np.clip(var[order], 0, None)).round(6).tolist(),
                            area_fraction=round(float((ids == i).sum() / valid.sum()), 6))
        c = p.mean(axis=0)
        centre[i - 1] = np.array([c[0], -c[1], -c[2]])
    edges = []
    for a in sorted(centre):
        for b in sorted(centre):
            if a != b:
                off = centre[a] - centre[b]
                edges.append(dict(src=a, dst=b, offset=off.round(6).tolist(),
                                  relation=relation(centre[a], centre[b], RELATIONS_3D)))
    return list(nodes.values()), edges


def window_graph(labels, depth, K, extrinsics):
    """Return the per-frame graphs of a window and its events."""
    frames, events = [], []
    for t in range(len(labels)):
        nodes, edges = frame_graph(labels[t], depth[t], K[t], extrinsics[t])
        frames.append(dict(frame=t, nodes=nodes, edges=edges))
    for prev, cur in zip(frames, frames[1:]):
        t = cur["frame"]
        before = {n["id"] for n in prev["nodes"]}
        after = {n["id"] for n in cur["nodes"]}
        events += [dict(frame=t, kind="enter", node=i) for i in sorted(after - before)]
        events += [dict(frame=t, kind="leave", node=i) for i in sorted(before - after)]
        was = {(e["src"], e["dst"]): e["relation"] for e in prev["edges"]}
        for e in cur["edges"]:
            old = was.get((e["src"], e["dst"]))
            if e["src"] < e["dst"] and old is not None and e["relation"] is not None \
                    and old != e["relation"]:
                events.append(dict(frame=t, kind="relation", src=e["src"], dst=e["dst"],
                                   before=old, after=e["relation"]))
    return dict(frames=frames, events=events)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", default=default_work())
    ap.add_argument("--windows", nargs="*")
    ap.add_argument("--input", default="normal", choices=settings.SAM_INPUTS)
    ap.add_argument("--pps", type=int, default=settings.DEFAULT_POINTS_PER_SIDE)
    args = ap.parse_args()
    work = Work(args.work)
    for w in pick(read_windows(), args.windows):
        with np.load(work.depth(w["clip"])) as z:
            depth, K, ext = z["depth"].astype(np.float32), z["intrinsics"].astype(np.float64), z["extrinsics"]
        labels = np.load(work.track(w["clip"], args.input, args.pps)).astype(np.int32)
        out = work.root / "graphs" / f"pps{args.pps}" / w["clip"] / f"{args.input}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(window_graph(labels, depth, K, ext)))
        print(f"{w['clip']}: {out}")


if __name__ == "__main__":
    main()
