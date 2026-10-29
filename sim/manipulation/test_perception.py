"""Object and surface measurement on a ray-cast RGB-D frame with known geometry."""

import numpy as np
import pytest

from perception import (
    object_geometry,
    obstacle_points,
    point_cloud,
    project,
    surface_at,
)

TABLE_Z = 0.73


def render(objects, height=160, width=200):
    """Pinhole RGB-D of a table slab, the floor and axis-aligned boxes.

    `objects` lists (centre, half size, rgb). Depth is along the camera axis,
    as the robot's renderer provides it.
    """
    position = np.array([0.0, 0.0, 1.25])
    forward = np.array([1.0, 0.0, -1.0]) / np.sqrt(2)
    up = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)
    right = np.cross(forward, up)
    near = 0.05
    half = near * np.tan(np.radians(30))
    v, u = np.mgrid[0:height, 0:width]
    yy = half - (v + 0.5) / height * 2 * half
    xx = ((u + 0.5) / width - 0.5) * 2 * half * width / height
    rays = forward + (xx[..., None] * right + yy[..., None] * up) / near
    depth = np.full((height, width), np.inf)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)

    def hit(t, colour):
        closer = (t > 0) & (t < depth)
        depth[closer] = t[closer]
        rgb[closer] = colour

    def slab(centre, size, colour):
        lo, hi = np.asarray(centre) - size, np.asarray(centre) + size
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo - position) / rays
            t2 = (hi - position) / rays
        enter = np.nanmax(np.minimum(t1, t2), axis=-1)
        leave = np.nanmin(np.maximum(t1, t2), axis=-1)
        hit(np.where(enter <= leave, enter, np.inf), colour)

    with np.errstate(divide="ignore"):
        hit(-position[2] / rays[..., 2], (40, 50, 60))  # floor
    slab([0.75, 0.0, TABLE_Z - 0.03], np.array([0.35, 0.45, 0.03]), (120, 120, 126))
    for centre, size, colour in objects:
        slab(centre, np.asarray(size), colour)
    calibration = {
        "position": position.tolist(),
        "forward": forward.tolist(),
        "up": up.tolist(),
        "near": near,
        "far": 10.0,
        "top": half,
        "bottom": -half,
    }
    depth[~np.isfinite(depth)] = 0
    return {"depth": depth, "rgb": rgb, "calibration": calibration, "frame_id": "test"}


def box_around(sensor, colour):
    mask = np.all(sensor["rgb"] == colour, axis=-1)
    v, u = np.nonzero(mask)
    h, w = mask.shape
    return [v.min() / h * 1000, u.min() / w * 1000, (v.max() + 1) / h * 1000, (u.max() + 1) / w * 1000]


def test_point_cloud_and_projection_agree():
    sensor = render([])
    cloud, pixels = point_cloud(sensor, stride=7)
    projected = project(sensor, cloud)
    np.testing.assert_allclose(projected, pixels, atol=0.6)


def test_cube_on_table_is_measured_to_a_few_millimetres():
    red = (220, 30, 25)
    sensor = render([([0.70, 0.05, 0.755], [0.025, 0.025, 0.025], red)])
    geometry = object_geometry(sensor, box_around(sensor, red))
    np.testing.assert_allclose(geometry["center_world_m"][:2], [0.70, 0.05], atol=0.008)
    assert geometry["top_z_m"] == pytest.approx(0.78, abs=0.006)
    assert geometry["support_z_m"] == pytest.approx(TABLE_Z, abs=0.005)
    assert geometry["bottom_certain"]
    np.testing.assert_allclose(geometry["half_size_m"], [0.025, 0.025, 0.025], atol=0.007)


def test_upper_block_of_a_stack_is_measured_alone():
    red, blue = (220, 30, 25), (30, 60, 220)
    sensor = render(
        [
            ([0.70, 0.0, 0.755], [0.03, 0.03, 0.025], red),
            ([0.70, 0.0, 0.805], [0.025, 0.025, 0.025], blue),
        ]
    )
    geometry = object_geometry(sensor, box_around(sensor, blue))
    assert geometry["top_z_m"] == pytest.approx(0.83, abs=0.006)
    assert geometry["bottom_z_m"] == pytest.approx(0.78, abs=0.012)


def test_surface_point_and_obstacles():
    sensor = render([])
    h, w = sensor["depth"].shape
    table_pixel = np.argwhere(np.all(sensor["rgb"] == (120, 120, 126), axis=-1))[len(sensor["rgb"]) // 2]
    point = [table_pixel[0] / h * 1000, table_pixel[1] / w * 1000]
    assert surface_at(sensor, point)[2] == pytest.approx(TABLE_Z, abs=0.004)
    points = obstacle_points(sensor, stride=4)
    assert points[:, 2].min() > 0.03  # the floor is not an obstacle
    # A table seen from above extends below its top surface.
    assert points[:, 2].min() < TABLE_Z - 0.05


def test_empty_box_is_rejected():
    sensor = render([])
    with pytest.raises(ValueError):
        object_geometry(sensor, [0, 0, 5, 5])
