import numpy as np

from experiments.nvidia_v2d.scripted_finger_grid import bounded_closure, closure_mask, grid_parameters


def test_grid_has_exact_zero_control_and_independent_axes():
    wrist, finger = grid_parameters()
    np.testing.assert_allclose(wrist[:4], 0)
    np.testing.assert_allclose(finger[:4], [0, 0.15, 0.3, 0.45])
    np.testing.assert_allclose(wrist[-4:], 0.04)
    assert len(set(zip(wrist, finger))) == 16


def test_closure_leaves_thumb_abduction_and_carpal_joints_unchanged():
    names = ("left_thumb_MCP_FE", "left_index_MCP_FE", "left_index_MCP_AA",
             "left_index_PIP", "left_index_DIP", "left_pinky_CMC")
    np.testing.assert_array_equal(closure_mask(names), [False, True, False, True, True, False])


def test_closure_respects_joint_limit_and_never_reverses_flexion():
    reference = np.array([[0.2, 1.5, 1.7]])
    result = bounded_closure(reference, np.array([0.45]), np.ones(3, dtype=bool), np.full((1, 3), 1.6))
    np.testing.assert_allclose(result, [[0.45, 0.1, 0]])
