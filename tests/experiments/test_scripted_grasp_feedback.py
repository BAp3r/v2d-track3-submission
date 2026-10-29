from types import SimpleNamespace

import numpy as np

from experiments.nvidia_v2d.scripted_grasp_feedback import encode_residual, smooth_gate, wrist_residuals


def test_live_feedback_preserves_baseline_lift_and_opposing_sides():
    wrists = np.array([[[-0.2, 0, 0.3], [0.2, 0, 0.3]]] * 2)
    ref = np.array([[0, 0, 0.2, 0, 0, 0, 1]] * 2)
    live = np.array([[0, 0.1, 0.05, 0, 0, 0, 1]] * 2)
    result = wrist_residuals(wrists, ref, live, np.array([0, 0.04]), np.ones(2), 0.03)
    np.testing.assert_allclose(result[0], 0)
    np.testing.assert_allclose(result[1], [[0.04, 0.03, 0], [-0.04, 0.03, 0]])


def test_inward_vectors_follow_live_object_rotation():
    wrists = np.array([[[-0.2, 0, 0.3], [0.2, 0, 0.3]]])
    ref = np.array([[0, 0, 0.2, 0, 0, 0, 1]])
    live = np.array([[0, 0, 0.2, 0, 0, np.sqrt(0.5), np.sqrt(0.5)]])
    result = wrist_residuals(wrists, ref, live, np.array([0.04]), np.ones(1), 0)
    np.testing.assert_allclose(result, [[[0, 0.04, 0], [0, -0.04, 0]]], atol=1e-8)


def test_smooth_gate_uses_source_phase_and_returns_to_zero():
    result = smooth_gate(np.array([0, 2.5, 2.9, 3.3, 8.5, 9.0, 9.5, 12]), (2.5, 3.3, 8.5, 9.5))
    np.testing.assert_allclose(result, [0, 0, 0.5, 1, 1, 0.5, 0, 0])


def test_encode_residual_respects_raw_and_normalized_mappings():
    physical = np.array([[0.04, -0.03, 0]])
    scale = np.full(3, 0.05)
    for mapping, bound in [("identity", 1), ("linear", 4), ("rational", 4 / 0.7)]:
        buffers = SimpleNamespace(
            scale=SimpleNamespace(numpy=lambda: scale),
            clip=SimpleNamespace(numpy=lambda: np.full(3, 0.2)),
            input_mapping=mapping, input_scale_values=np.full(3, bound),
        )
        encoded = encode_residual(physical, buffers)
        if mapping == "identity":
            raw = encoded
        elif mapping == "linear":
            raw = encoded * bound
        else:
            raw = encoded / (1 - np.abs(encoded) + np.abs(encoded) / bound)
        np.testing.assert_allclose(raw * scale, physical, atol=1e-8)
