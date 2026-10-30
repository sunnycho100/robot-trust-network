from types import SimpleNamespace

import numpy as np
import pytest

from sensor_renderer import render_rgb_exposure


def test_sensor_rgb_exposure_is_temporary_and_restores_on_failure():
    light = SimpleNamespace(
        ambient=np.array([0.2, 0.2, 0.2]),
        diffuse=np.array([0.8, 0.6, 0.4]),
        specular=np.array([0.1, 0.1, 0.1]),
    )
    saved = {k: getattr(light, k).copy() for k in ("ambient", "diffuse", "specular")}
    renderer = SimpleNamespace(scene=SimpleNamespace(lights=[light], nlight=1))

    def acquire():
        np.testing.assert_allclose(light.diffuse, saved["diffuse"] * 0.5)
        raise RuntimeError("Render interrupted")

    renderer.render = acquire
    with pytest.raises(RuntimeError, match="Render interrupted"):
        render_rgb_exposure(renderer, 0.5)
    for key, value in saved.items():
        np.testing.assert_array_equal(getattr(light, key), value)
    for bad in (0, 2, np.nan):
        with pytest.raises(ValueError, match="exposure"):
            render_rgb_exposure(renderer, bad)
