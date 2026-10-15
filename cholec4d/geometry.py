"""Images SAM segments, and the 3D geometry of regions.

Depth is z-depth on DA3's processing grid, `K` the matching pinhole
intrinsics, and a pose maps world to camera (x_cam = R x_world + t). A pixel
is valid where its depth is finite and positive.

Relations are read in the camera frame with y and z flipped, so that +x is
right, +y is up in the image and +z points towards the camera (front).
"""
import matplotlib
import numpy as np
from PIL import Image
import cv2

# The six directions a relation can take, in that flipped camera frame. On a
# tie the first wins.
RELATIONS_3D = (
    ("right", np.array([1, 0, 0])), ("left", np.array([-1, 0, 0])),
    ("above", np.array([0, 1, 0])), ("below", np.array([0, -1, 0])),
    ("front", np.array([0, 0, 1])), ("behind", np.array([0, 0, -1])),
)
RELATIONS_2D = RELATIONS_3D[:4]

_NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def valid_depth(depth):
    """Return the pixels whose depth is finite and positive."""
    return np.isfinite(depth) & (depth > 1e-6)


def load_rgb(path, height, width):
    """Read a frame as RGB uint8 on the depth grid (area-averaged when resized)."""
    im = np.array(Image.open(path).convert("RGB"))
    if im.shape[:2] != (height, width):
        im = cv2.resize(im, (width, height), interpolation=cv2.INTER_AREA)
    return im


# --- What SAM is given -------------------------------------------------------

def window_depth_range(depth):
    """Scale a window's depth to [0, 1] between its 1st and 99th percentiles.

    One scale for the whole window keeps the depth image of a frame
    comparable with its neighbours'.
    """
    finite = depth[np.isfinite(depth) & (depth > 1e-6)]
    lo, hi = np.percentile(finite, 1), np.percentile(finite, 99)
    return np.clip((depth - lo) / (hi - lo + 1e-6), 0, 1)


def depth_image(depth01):
    """Render scaled depth with the viridis colour map, as RGB uint8."""
    return (matplotlib.colormaps["viridis"](depth01)[:, :, :3] * 255).astype(np.uint8)


def surface_normals(depth, K):
    """Return unit camera-frame normals (H, W, 3) and the valid pixels.

    Each pixel is back-projected, and its normal is the cross product of the
    central differences along the row and along the column. Scaling the
    depth leaves the normals unchanged.
    """
    h, w = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    ys, xs = np.mgrid[0:h, 0:w]
    ok = np.isfinite(depth) & (depth > 1e-6)
    z = np.where(ok, depth, np.nan)
    P = np.stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z], -1)
    du = np.zeros_like(P)
    dv = np.zeros_like(P)
    du[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dv[1:-1, :] = P[2:, :] - P[:-2, :]
    n = np.cross(du, dv)
    n /= (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8)
    return n, ok


def normal_image(depth, K):
    """Encode surface normals as RGB uint8, [-1, 1] to [0, 255]; invalid pixels are black."""
    n, ok = surface_normals(depth, K)
    img = (n * 0.5 + 0.5) * 255
    img[~ok] = 0
    return np.nan_to_num(img).astype(np.uint8)


def sam_image(sam_input, rgb, depth, K, depth01):
    """Return the image SAM segments for one frame: the RGB frame, its depth or its normals."""
    if sam_input == "rgb":
        return rgb
    if sam_input == "normal":
        return normal_image(depth, K)
    if sam_input == "depth":
        return depth_image(depth01)
    raise ValueError(f"unknown SAM input {sam_input!r}")


# --- Region geometry ---------------------------------------------------------

def seam(id_map, valid, dilate=1):
    """Pixels on a boundary between two labels, widened by `dilate` pixels.

    A valid pixel is on the boundary when a valid 8-neighbour carries
    another label. Depth blends across such a boundary, so its pixels float
    between the two surfaces once back-projected.
    """
    h, w = id_map.shape
    ids = id_map.astype(np.int64)
    off = int(ids.min()) - 1
    ids = np.where(valid, ids, off)
    pad = np.full((h + 2, w + 2), off - 1, np.int64)
    pad[1:-1, 1:-1] = ids
    out = np.zeros((h, w), bool)
    for dy, dx in _NEIGHBOURS:
        nb = pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        out |= valid & (ids != off) & (nb != off - 1) & (nb != off) & (nb != ids)
    for _ in range(dilate):
        pad = np.zeros((h + 2, w + 2), bool)
        pad[1:-1, 1:-1] = out
        grown = out.copy()
        for dy, dx in _NEIGHBOURS:
            grown |= pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        out = grown
    return out & valid


def region_points(depth, K, id_map):
    """Back-project the valid pixels off the seams, grouped by label (0 = none).

    Returns:
        ({label: (n, 3) camera-frame points}, number of pixels kept).
    """
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    base = (depth > 0) & np.isfinite(depth)
    keep = base & ~seam(id_map, base)
    z = depth[keep]
    x = (u[keep] - K[0, 2]) * z / K[0, 0]
    y = (v[keep] - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z], axis=-1)
    ids = id_map[keep]
    return {int(i): pts[ids == i] for i in np.unique(ids) if i != 0}, int(keep.sum())


def centroids_3d(depth, K, id_map):
    """Return {label: centroid in the flipped camera frame} of every label but 0."""
    return {i: [float(c) * s for c, s in zip(p.mean(axis=0), (1, -1, -1))]
            for i, p in region_points(depth, K, id_map)[0].items()}


def centroids_2d(id_map, valid):
    """Return {label: image-plane centroid as [x right, y up, 0]} over the valid pixels."""
    v, u = np.nonzero(valid & (id_map != 0))
    ids = id_map[v, u]
    return {int(i): [float(u[ids == i].mean()), -float(v[ids == i].mean()), 0.0]
            for i in np.unique(ids)}


def relation(src, dst, axes=RELATIONS_3D):
    """Return the direction of `src` seen from `dst`: the axis closest to dst -> src.

    Returns None when the two points coincide.
    """
    d = np.asarray(src) - np.asarray(dst)
    n = np.linalg.norm(d)
    if n < 1e-8:
        return None
    d = d / n
    best, best_dot = None, -1.0
    for name, axis in axes:
        dot = float(np.dot(d, axis))
        if dot > best_dot:
            best, best_dot = name, dot
    return best


def to_world(points_cam, extrinsic):
    """Map camera-frame points to the world frame of a world-to-camera pose (3, 4)."""
    R, t = extrinsic[:, :3], extrinsic[:, 3]
    return (R.T @ (np.asarray(points_cam).T - t[:, None])).T
