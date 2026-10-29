import numpy as np
import pytest

from experiments.nvidia_v2d.static_support_depth import (
    plane_ray_scale,
    project_scene,
    robust_plane,
    translate_hand_on_wrist_ray,
)


def test_plane_recovers_static_surface_with_foreground_occlusion():
    rng = np.random.default_rng(3)
    xy = rng.uniform(-0.3, 0.3, (2000, 2))
    z = 0.02 * xy[:, 0] - 0.01 * xy[:, 1] - 0.08 + rng.normal(0, 0.001, len(xy))
    z[:500] += 0.15
    result = robust_plane(np.column_stack((xy, z)))
    np.testing.assert_allclose(result["coefficients"], [0.02, -0.01, -0.08], atol=0.001)
    assert 0.74 < result["inlier_fraction"] < 0.76


def test_camera_ray_scaling_maps_plane_to_anchor_height():
    plane = np.array([0.03, -0.01, -0.12])
    camera = np.array([0.02, -0.4, 0.7])
    anchor = np.array([0, 0.05])
    s = plane_ray_scale(plane, camera, anchor, -0.06)
    # A point mapping to anchor xy lies on the transformed plane at target z.
    transformed_c = (1 - s) * np.dot([-plane[0], -plane[1], 1], camera) + s * plane[2]
    assert plane[0] * anchor[0] + plane[1] * anchor[1] + transformed_c == pytest.approx(
        -0.06
    )


def test_rigid_candidate_preserves_anatomy_and_wrist_projection_but_not_all_pixels():
    camera = np.array([0.0, 0.0, 0.0])
    wrist = np.array([0.1, 0.2, 1.0])
    vertices = np.array([wrist, wrist + [0.04, 0, 0], wrist + [0, 0.06, 0.03]])
    new = translate_hand_on_wrist_ray(vertices, wrist, camera, 0.9)
    np.testing.assert_allclose(new - new[0], vertices - vertices[0])
    before = project_scene(vertices, np.eye(4), np.array([500, 500, 100, 100]))
    after = project_scene(new, np.eye(4), np.array([500, 500, 100, 100]))
    np.testing.assert_allclose(before[0], after[0])
    assert np.max(abs(before[1:] - after[1:])) > 1


def test_unreliable_or_degenerate_support_rejected():
    with pytest.raises(ValueError):
        robust_plane(np.zeros((99, 3)))
    with pytest.raises(ValueError):
        plane_ray_scale([0, 0, 1], [0, 0, 1], [0, 0], 0)
