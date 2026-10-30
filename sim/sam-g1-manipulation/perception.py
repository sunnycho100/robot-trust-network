"""Object and surface geometry from rendered RGB-D, selected by image boxes.

Inputs are camera images, depth and calibration only. The vision-language model
chooses WHICH object (a 2D box); this module measures WHERE it is and how big.
No simulator object poses, names or identities are read here.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

FLOOR_MAX_Z = 0.03


def capture(console, camera="head"):
    """Render a fresh RGB-D frame from one camera at a physics-step boundary."""
    previous = console.agent_camera
    console.agent_camera = camera
    try:
        console.publish(force=True, depth=True)
    finally:
        console.agent_camera = previous
    sensor = dict(console.visual_sensor)
    if "rgb" not in sensor:
        sensor["rgb"] = np.asarray(Image.open(io.BytesIO(sensor["image"])).convert("RGB"))
    return sensor


def point_cloud(sensor, stride=1):
    """World points and their pixels; the robot's own silhouette is excluded."""
    depth = np.asarray(sensor["depth"])
    v, u = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = depth[v, u]
    c = sensor["calibration"]
    valid = np.isfinite(z) & (z > 0) & (z < c["far"] * 0.99)
    if "robot_mask" in sensor:
        valid &= ~sensor["robot_mask"][v, u]
    v, u, z = v[valid], u[valid], z[valid]
    forward, up = np.asarray(c["forward"]), np.asarray(c["up"])
    right = np.cross(forward, up)
    height = c["top"] - c["bottom"]
    yy = c["top"] - (v + 0.5) / depth.shape[0] * height
    xx = ((u + 0.5) / depth.shape[1] - 0.5) * height * depth.shape[1] / depth.shape[
        0
    ] + c.get("center", 0)
    xyz = np.asarray(c["position"]) + z[:, None] * (
        forward + (xx[:, None] * right + yy[:, None] * up) / c["near"]
    )
    return xyz, np.column_stack((u, v))


def project(sensor, points):
    """World points to pixel (u, v) in this frame; None where behind the camera."""
    c = sensor["calibration"]
    h, w = sensor["depth"].shape
    forward, up = np.asarray(c["forward"]), np.asarray(c["up"])
    right = np.cross(forward, up)
    relative = np.atleast_2d(points) - np.asarray(c["position"])
    depth = relative @ forward
    height = c["top"] - c["bottom"]
    xx = (relative @ right) / np.maximum(depth, 1e-6) * c["near"] - c.get("center", 0)
    yy = (relative @ up) / np.maximum(depth, 1e-6) * c["near"]
    u = (xx / (height * w / h) + 0.5) * w - 0.5
    v = (c["top"] - yy) / height * h - 0.5
    pixels = np.column_stack((u, v))
    pixels[depth <= 0] = np.nan
    return pixels


def normalized_box(box):
    box = np.asarray(box, dtype=float)
    if (
        box.shape != (4,)
        or not np.isfinite(box).all()
        or np.any(box < 0)
        or np.any(box > 1000)
        or box[2] <= box[0]
        or box[3] <= box[1]
    ):
        raise ValueError("Boxes are [ymin, xmin, ymax, xmax] normalized 0..1000")
    return box


def pixel_from_normalized(sensor, point):
    """[y, x] normalized 0..1000 to pixel (u, v)."""
    point = np.asarray(point, dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all() or np.any(point < 0) or np.any(point > 1000):
        raise ValueError("Points are [y, x] normalized 0..1000")
    h, w = sensor["depth"].shape
    return np.array([point[1] * w / 1000, point[0] * h / 1000])


def _horizontal_mask(grid):
    normal = np.cross(grid[1:-1, 2:] - grid[1:-1, :-2], grid[2:, 1:-1] - grid[:-2, 1:-1])
    mask = np.zeros(grid.shape[:2], dtype=bool)
    with np.errstate(invalid="ignore"):
        mask[1:-1, 1:-1] = np.abs(normal[:, :, 2]) > 0.95 * np.linalg.norm(normal, axis=2)
    return mask


def _support_height(cloud, uv, region_box, below):
    """Highest well-populated horizontal plane under `below` near the box."""
    x0, y0, x1, y1 = region_box
    ring = (uv[:, 0] >= x0) & (uv[:, 0] <= x1) & (uv[:, 1] >= y0) & (uv[:, 1] <= y1)
    points, pixels = cloud[ring], uv[ring]
    if len(points) < 30:
        return None
    lo, hi = pixels.min(axis=0), pixels.max(axis=0)
    grid = np.full((hi[1] - lo[1] + 1, hi[0] - lo[0] + 1, 3), np.nan)
    grid[pixels[:, 1] - lo[1], pixels[:, 0] - lo[0]] = points
    horizontal = _horizontal_mask(grid)[pixels[:, 1] - lo[1], pixels[:, 0] - lo[0]]
    samples = points[horizontal, 2]
    samples = samples[samples < below]
    if len(samples) < 20:
        return None
    counts, edges = np.histogram(
        samples, bins=np.arange(float(samples.min()) - 0.002, below + 0.004, 0.004)
    )
    dense = np.flatnonzero(counts >= max(6, 0.01 * len(samples)))
    if not len(dense):
        return None
    index = dense[-1]
    level = float(np.median(samples[(samples >= edges[index]) & (samples <= edges[index + 1])]))
    return float(np.median(samples[np.abs(samples - level) < 0.006]))


def _footprint(xy):
    """Oriented rectangle around points: centre, half extents (long, short), yaw."""
    center = np.median(xy, axis=0)
    centred = xy - center
    if len(xy) < 5:
        return center, np.array([0.01, 0.01]), 0.0
    _, eigenvectors = np.linalg.eigh(np.cov(centred.T))
    best = None
    principal = np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])
    # PCA picks a box axis for elongated shapes; refine by area for squares.
    for angle in principal + np.radians(np.arange(-45, 46, 3)):
        axis = np.array([np.cos(angle), np.sin(angle)])
        normal = np.array([-axis[1], axis[0]])
        a = centred @ axis
        b = centred @ normal
        lo_a, hi_a = np.percentile(a, [1, 99])
        lo_b, hi_b = np.percentile(b, [1, 99])
        area = (hi_a - lo_a) * (hi_b - lo_b)
        if best is None or area < best[0]:
            mid = center + axis * (lo_a + hi_a) / 2 + normal * (lo_b + hi_b) / 2
            best = (area, mid, np.array([(hi_a - lo_a) / 2, (hi_b - lo_b) / 2]), angle)
    _, mid, half, angle = best
    if half[1] > half[0]:
        half = half[::-1]
        angle += np.pi / 2
    return mid, half, float(np.arctan2(np.sin(angle), np.cos(angle)))


def object_geometry(sensor, box, ignore=(), chroma=None):
    """Measure the individual object inside a [ymin, xmin, ymax, xmax] box.

    Returns centre, oriented footprint, top/bottom heights and the supporting
    surface. Colour separates an object from a differently coloured object it
    sits on, so one block of a stack is measured, not the whole stack.
    `ignore` lists (centre, radius) spheres to leave out, e.g. an object held
    in the robot's own hand. `chroma` re-measures a known object by its colour
    instead of the box's dominant colour.
    """
    box = normalized_box(box)
    h, w = sensor["depth"].shape
    y0, x0, y1, x1 = box * [h, w, h, w] / 1000
    width, height = x1 - x0, y1 - y0
    cloud, uv = point_cloud(sensor)
    for centre, radius in ignore:
        keep = np.linalg.norm(cloud - np.asarray(centre), axis=1) > radius
        cloud, uv = cloud[keep], uv[keep]
    pad = 0.15
    inside = (
        (uv[:, 0] >= x0 - width * pad)
        & (uv[:, 0] <= x1 + width * pad)
        & (uv[:, 1] >= y0 - height * pad)
        & (uv[:, 1] <= y1 + height * pad)
    )
    points, pixels = cloud[inside], uv[inside]
    if len(points) < 15:
        raise ValueError("The selected box has too little visible depth; choose a closer or clearer view")
    image = np.asarray(sensor["rgb"], dtype=float)
    chroma_image = image / np.maximum(image.sum(axis=2, keepdims=True), 1)
    roi = image[int(max(0, y0)) : int(min(h, y1)) + 1, int(max(0, x0)) : int(min(w, x1)) + 1]
    saturation = (roi.max(axis=2) - roi.min(axis=2)) / np.maximum(roi.max(axis=2), 1)
    chroma = chroma_image[pixels[:, 1], pixels[:, 0]]
    if chroma is not None:
        target_chroma = np.asarray(chroma, dtype=float)
        foreground = np.linalg.norm(chroma_image[pixels[:, 1], pixels[:, 0]] - target_chroma, axis=1) < 0.12
    elif np.percentile(saturation, 85) > 0.25:
        # The selected object is the dominant saturated colour inside the box.
        core = roi[saturation >= max(0.25, np.percentile(saturation, 60))]
        target_chroma = np.median(core / np.maximum(core.sum(axis=1, keepdims=True), 1), axis=0)
        foreground = np.linalg.norm(chroma - target_chroma, axis=1) < 0.12
    else:
        target_chroma = np.median(roi.reshape(-1, 3) / np.maximum(roi.reshape(-1, 3).sum(axis=1, keepdims=True), 1), axis=0)
        foreground = np.ones(len(points), dtype=bool)
    if np.count_nonzero(foreground) < 10:
        raise ValueError("No consistent object colour inside the selected box")
    top_guess = float(np.percentile(points[foreground, 2], 98))
    low_guess = float(np.percentile(points[foreground, 2], 3))
    support = _support_height(
        cloud,
        uv,
        (x0 - width * 1.2, y0 - height * 1.2, x1 + width * 1.2, y1 + height * 1.2),
        below=min(top_guess - 0.012, low_guess + 0.012),
    )
    if support is None:
        support = low_guess - 0.002
    valid = foreground & (points[:, 2] > support + 0.005)
    mask = np.zeros((h, w), dtype=bool)
    mask[pixels[valid, 1], pixels[valid, 0]] = True
    labels, count = ndimage.label(ndimage.binary_closing(mask, iterations=2) & mask)
    if not count:
        raise ValueError("Nothing in the box rises above its supporting surface")
    ids = labels[pixels[:, 1], pixels[:, 0]]
    centre_px = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    choices = []
    for label in range(1, count + 1):
        selected = (ids == label) & valid
        if np.count_nonzero(selected) >= 8:
            choices.append((np.linalg.norm(pixels[selected].mean(axis=0) - centre_px), label))
    if not choices:
        raise ValueError("The selected object is too small or hidden in this view")
    selected = (ids == min(choices)[1]) & valid
    object_points = points[selected]
    top = float(np.percentile(object_points[:, 2], 98))
    visible_low = float(np.percentile(object_points[:, 2], 2))
    # Resting on the measured support unless its visible sides end clearly higher:
    # then it stands on something hidden, e.g. the lower block of a stack. With
    # only the top face in view the bottom is unknown; assume it rests and say so.
    if visible_low - support < 0.02:
        bottom, bottom_certain = support, True
    elif top - visible_low >= 0.015:
        bottom, bottom_certain = visible_low, True
    else:
        bottom, bottom_certain = support, False
    upper = object_points[object_points[:, 2] > top - 0.006]
    footprint_points = object_points[:, :2] if len(upper) < 25 else np.vstack(
        (upper[:, :2], object_points[:, :2])
    )
    centre_xy, half_xy, yaw = _footprint(footprint_points)
    top_xy = upper[:, :2] if len(upper) >= 12 else object_points[:, :2]
    # The top face is seen whole from above; its centre is not pulled toward the
    # camera by visible side faces, so things are stacked on it.
    top_centre = _footprint(upper[:, :2])[0] if len(upper) >= 25 else centre_xy
    round_top = False
    if len(top_xy) >= 30:
        # A circle fills pi/4 of its bounding square; a rectangle fills all of it.
        centred = top_xy - np.median(top_xy, axis=0)
        radius = np.percentile(np.linalg.norm(centred, axis=1), 95)
        corners = np.abs(centred) / max(radius, 1e-6)
        round_top = bool(np.mean(np.linalg.norm(corners, axis=1) > 1.05) < 0.02) and abs(
            half_xy[0] - half_xy[1]
        ) < 0.006
    colour = image[pixels[selected, 1], pixels[selected, 0]].mean(axis=0)
    return {
        "center_world_m": [float(centre_xy[0]), float(centre_xy[1]), float((top + bottom) / 2)],
        "top_center_m": [float(top_centre[0]), float(top_centre[1]), top],
        "half_size_m": [float(half_xy[0]), float(half_xy[1]), float((top - bottom) / 2)],
        "yaw_rad": yaw,
        "top_z_m": top,
        "bottom_z_m": float(bottom),
        "bottom_certain": bool(bottom_certain),
        "support_z_m": float(support),
        "shape": "round" if round_top else "box",
        "colour_rgb": colour.round(1).tolist(),
        "chroma": target_chroma.round(4).tolist(),
        "visible_points": len(object_points),
        "frame_id": sensor.get("frame_id"),
        "camera": sensor.get("camera"),
    }


def surface_at(sensor, point_norm):
    """Horizontal surface under a [y, x] normalized point: world xy and height."""
    u, v = pixel_from_normalized(sensor, point_norm)
    cloud, uv = point_cloud(sensor)
    distance = np.hypot(uv[:, 0] - u, uv[:, 1] - v)
    near = distance < 18
    if np.count_nonzero(near) < 10:
        raise ValueError("No visible surface at that point")
    points = cloud[near]
    centre = cloud[np.argmin(distance)]
    level = float(np.median(points[np.abs(points[:, 2] - centre[2]) < 0.01, 2]))
    return np.array([centre[0], centre[1], level])


def obstacle_points(sensor, stride=3, min_z=FLOOR_MAX_Z, thickness=0.08):
    """Everything above the floor that the robot's body must not walk into.

    A camera sees the top of a table but not how thick it is underneath, so
    each seen point is also assumed to extend `thickness` below itself.
    """
    cloud, _ = point_cloud(sensor, stride=stride)
    cloud = cloud[cloud[:, 2] > min_z]
    layers = [cloud]
    for drop in np.arange(0.03, thickness + 1e-6, 0.03):
        lowered = cloud - [0.0, 0.0, drop]
        layers.append(lowered[lowered[:, 2] > min_z])
    return np.vstack(layers)


def remove_near(points, center, half_size, yaw, margin=0.015):
    """Drop points inside an oriented box (the object being handled)."""
    if not len(points):
        return points
    c, s = np.cos(yaw), np.sin(yaw)
    local = points - np.asarray(center)
    a = local[:, 0] * c + local[:, 1] * s
    b = -local[:, 0] * s + local[:, 1] * c
    inside = (
        (np.abs(a) <= half_size[0] + margin)
        & (np.abs(b) <= half_size[1] + margin)
        & (np.abs(local[:, 2]) <= half_size[2] + margin)
    )
    return points[~inside]


def surface_free(points, center_xy, half_xy, yaw, support_z, clearance=0.012, cell=0.015):
    """Is a footprint fully on a support and free of other objects?

    `points` are camera points (not extruded obstacles). Returns the fraction of
    footprint cells with the support seen at its height, and the number of
    points standing above it. A spot hanging over an edge has low coverage.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    local = points[:, :2] - np.asarray(center_xy)
    a = local[:, 0] * c + local[:, 1] * s
    b = -local[:, 0] * s + local[:, 1] * c
    inside = (np.abs(a) <= half_xy[0]) & (np.abs(b) <= half_xy[1])
    on_support = inside & (np.abs(points[:, 2] - support_z) < 0.008)
    blocking = int(np.count_nonzero(inside & (points[:, 2] > support_z + clearance)))
    shape = np.maximum(1, np.ceil(2 * np.asarray(half_xy) / cell).astype(int))
    cells = np.floor((np.column_stack((a, b))[on_support] + half_xy) / cell).astype(int)
    cells = np.clip(cells, 0, shape - 1)
    covered = len(np.unique(cells[:, 0] * shape[1] + cells[:, 1])) if len(cells) else 0
    return covered / float(shape[0] * shape[1]), blocking


def nearest_clearance(points, query):
    if not len(points):
        return np.full(len(np.atleast_2d(query)), np.inf)
    return cKDTree(points).query(np.atleast_2d(query))[0]
