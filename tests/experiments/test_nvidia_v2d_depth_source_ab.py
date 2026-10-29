import numpy as np
import pytest

from experiments.nvidia_v2d.depth_source_ab import (
    CameraK,
    depth_fit,
    rasterize_depth,
    reproject_depth,
)


def camera(fx=1, cx=0, width=4, height=3):
    return CameraK(width=width, height=height, fx=fx, fy=1, cx=cx, cy=0)


def test_reprojection_respects_changed_rays_and_holes():
    depth = np.full((3, 4), np.nan)
    depth[1, 2] = 2
    result = reproject_depth(depth, camera(fx=2), camera())
    assert result[1, 1] == 2
    assert np.isnan(result[1, 2])


def test_z_buffer_keeps_nearest_depth_when_pixels_collide():
    depth = np.full((3, 4), np.nan)
    depth[1, 0:2] = [3, 1]
    result = reproject_depth(depth, camera(fx=4), camera())
    assert result[1, 0] == 1


def test_paired_fit_rejects_different_valid_support():
    rendered = np.ones((2, 2))
    a = np.array([[2, 2], [2, np.nan]])
    b = np.array([[3, np.nan], [3, 3]])
    shared = np.isfinite(a) & np.isfinite(b)
    assert depth_fit(rendered, a, shared)["scale"] == 2
    assert depth_fit(rendered, b, shared)["scale"] == 3
    assert depth_fit(rendered, a, np.zeros((2, 2), bool))["scale"] is None


def test_triangle_depth_uses_perspective_correct_interpolation():
    # Projected triangle (0,0), (2,0), (0,2), with z 1,2,2.
    v = np.array([[0, 0, 1], [4, 0, 2], [0, 4, 2]], float)
    result = rasterize_depth(v, np.array([[0, 1, 2]]), camera(width=3, height=3))
    assert result[0, 1] == pytest.approx(4 / 3)
    assert np.isnan(result[2, 2])
