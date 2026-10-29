import numpy as np
import pytest

wp = pytest.importorskip("warp")
from experiments.nvidia_v2d.semantic_contact_metrics import (
    record_roles,
    summarize_transition,
)


def arr(values, dtype=wp.float32):
    return wp.array(values, dtype=dtype, device="cpu")


def test_transition_metrics_ignore_unknown_geometry_and_measure_each_side():
    # World0 has a correct left role, distant/opposed right role. World1's
    # inactive frame intentionally contains invalid bodies and NaN geometry.
    poses = np.tile([0., 0., 0., 0., 0., 0., 1.], (6, 1))
    poses[2, 0] = .115
    poses[2, 3:] = [0., 0., 1., 0.]
    normals = np.array([[1., 0., 0.]] * 2 + [[np.nan] * 3] * 2)
    points = np.array([[0., 0., 0.]] * 2 + [[np.nan] * 3] * 2)
    object_points = points.copy(); object_points[:2, 0] = .001
    body = arr(poses, wp.transform)
    values = arr(np.full(24, 99.))
    wp.launch(record_roles, dim=(2, 2), inputs=[body, arr([0, 1], wp.int32),
        arr([1, 0, 0, 0], wp.int32), arr([1, 1, 0, 0], wp.int32),
        arr([1, 2, 999, 999], wp.int32), arr([0, 1, 999, 999], wp.int32),
        arr(points, wp.vec3), arr(normals, wp.vec3), arr(object_points, wp.vec3), arr(normals, wp.vec3),
        2, 2, 3, 0, 2], outputs=[values], device="cpu")
    records = values.numpy().reshape(2, 2, 6)
    np.testing.assert_array_equal(records[1], np.zeros((2, 6)))
    np.testing.assert_allclose(records[0, 1], [1., .115, -1., 0., 3.18, 0.], atol=1e-6)
    np.testing.assert_array_equal(body.numpy(), poses.astype(np.float32))
    metrics = summarize_transition(records, [2., 4.], [.5, 1.], [1., 0.], [.75, 4.], .25)
    assert metrics["contact/raw_wrench_support_mean"] == 3.
    assert metrics["contact/matched_wrench_support_mean"] == 2.5
    assert metrics["contact/alignment_penalty_mean"] == .125
    assert metrics["contact/mixed_support_mean"] == 2.375
    assert metrics["contact/right/active_worlds"] == 1.
    assert metrics["contact/active_worlds"] == 1.
    assert metrics["contact/role_match_active_mean"] == .5
    assert metrics["contact/alignment_cost_active_mean"] == 1.
    assert metrics["contact/left/aligned_pair_fraction"] == 1.
    assert metrics["contact/right/aligned_pair_fraction"] == 0.
    metrics = summarize_transition(np.zeros((2, 2, 6)), [0., 0.], [1., 1.], [0., 0.], [0., 0.], 0.)
    assert metrics["contact/right/active_worlds"] == 0.
    assert "contact/role_match_active_mean" not in metrics
    assert "contact/right/target_distance_m" not in metrics
