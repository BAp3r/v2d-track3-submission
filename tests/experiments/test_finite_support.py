"""Physical projection and finite-support contract tests, without simulator dependencies."""

import numpy as np
import pytest

from experiments.nvidia_v2d.finite_support import fit_circle, intersect_plane, support_usda


def test_tilted_camera_recovers_metric_table_circle():
    angle = .35
    rotation = np.array([[1, 0, 0], [0, -np.cos(angle), np.sin(angle)],
                         [0, -np.sin(angle), -np.cos(angle)]])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = [.1, -.2, 1.0]
    theta = np.linspace(0, 2*np.pi, 80, endpoint=False)
    points = np.column_stack((.04 + .31*np.cos(theta), .08 + .31*np.sin(theta), np.zeros(80)))
    camera = (points - transform[:3, 3]) @ rotation
    pixels = camera[:, :2] / camera[:, 2, None] * [800, 850] + [320, 240]
    recovered = intersect_plane(pixels, np.array([800, 850, 320, 240]), transform, 0)
    np.testing.assert_allclose(recovered, points, atol=1e-12)
    fit = fit_circle(recovered[:, :2])
    np.testing.assert_allclose(fit["center_xy_m"], [.04, .08], atol=1e-12)
    assert fit["radius_m"] == pytest.approx(.31)


def test_rejects_ray_behind_camera_and_degenerate_boundary():
    with pytest.raises(ValueError, match="behind"):
        intersect_plane(np.zeros((8, 2)), np.array([1, 1, 0, 0]), np.eye(4), -1)
    with pytest.raises(ValueError, match="Degenerate"):
        fit_circle(np.column_stack((np.arange(8), np.zeros(8))))


def test_support_top_is_preserved_and_dimensions_are_validated():
    text = support_usda([.1, -.2], .3, 0, .04)
    assert "double height = 0.0400000000" in text
    assert "(0.1000000000, -0.2000000000, -0.0200000000)" in text
    assert 'axis = "Z"' in text
    with pytest.raises(ValueError, match="positive"):
        support_usda([0, 0], -.3, 0, .04)
