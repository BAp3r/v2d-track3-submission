from types import SimpleNamespace

import numpy as np
import pytest

wp = pytest.importorskip("warp")
pytest.importorskip("flash_chord")

from flash_chord.runtime.actions.hand_pose import ResidualHandPoseAction

from experiments.nvidia_v2d.right_wrist_command_objective import (
    RightWristCommandCostConfig,
    bind_right_wrist_command_penalty,
    penalize_right_wrist_command,
)


def arr(value, dtype=wp.float32):
    return wp.array(value, dtype=dtype, device="cpu")


def test_kernel_uses_meters_world_dof_offsets_and_skips_unknown_release():
    # Explicit noncontiguous wrist DOFs, two role slots, six worlds.
    target = np.zeros((6, 10), np.float32)
    target[:, [5, 2, 7]] = [2., -3., 4.]
    target[:, 5] += [.14, .01, .8, np.nan, np.nan, .14]
    refs = np.tile([2., -3., 4.], (6, 1)); refs[2:] = np.nan
    args = [arr([0, 0, 0, 1, 2, 0], wp.int32),
            arr([0, 1, 1, 0, 0, 1], wp.int32),  # release active deliberately corrupt
            arr(refs, wp.vec3), arr(target.reshape(-1)),
            3, 2, 1, 2, 2, 1, 10, 5, 2, 7, .02, .05, np.sqrt(3)*.2, .1]
    distance, cost, support = arr(np.zeros(6)), arr(np.zeros(6)), arr(np.ones(6))
    before = args[3].numpy().copy()
    wp.launch(penalize_right_wrist_command, dim=6, inputs=args,
              outputs=[distance, cost, support], device="cpu")
    expected = [2.4, 0, (np.sqrt(3)*.2-.02)/.05, 0, 0, 2.4]
    np.testing.assert_allclose(cost.numpy(), expected, atol=5e-6)
    np.testing.assert_allclose(support.numpy(), 1-.1*np.asarray(expected), atol=1e-6)
    np.testing.assert_allclose(distance.numpy(), [.14, .01, .8, 0, 0, .14], atol=3e-7)
    np.testing.assert_array_equal(args[3].numpy(), before)


def fixture_action():
    # Action order is right/left, while semantic role order is left/right.
    layout = SimpleNamespace(sides=("right", "left"), num_hands=2,
        num_joint_dof=10, wrist_pos_dof_ids=(5, 2, 7, 0, 1, 3),
        hand_action_starts=(0, 3))
    reference = SimpleNamespace(sides=layout.sides, num_frames=3,
        wrist_pos_w=arr(np.zeros((6, 3)), wp.vec3))
    buffers = SimpleNamespace(layout=layout, reference=reference, world_count=2,
        clip=arr([.2]*6), scale=arr([.05]*6), joint_target=arr(np.zeros(20)))
    return ResidualHandPoseAction(buffers, None)


def test_binding_follows_action_reference_and_explicit_hand_names():
    action = fixture_action()
    active = arr([0, 1, 0, 1, 0, 0], wp.int32)
    bound = bind_right_wrist_command_penalty(action, active,
        role_sides=("left", "right"), release_start_frame=2)
    assert bound.right_hand == 0 and bound.right_role == 1
    assert bound.wrist_dofs == (5, 2, 7)
    assert bound.reference_wrist is action.buffers.reference.wrist_pos_w
    assert bound.joint_target is action.joint_target
    assert bound.radial_cap_m == pytest.approx(np.sqrt(3)*.2)
    assert bound.scale_m == pytest.approx(.05)
    support = arr([.3, .4]); bound.command_cost.assign([9., 9.])
    bound.apply(arr([0, 0], wp.int32), support)  # zero default launches nothing
    np.testing.assert_array_equal(support.numpy(), np.array([.3, .4], np.float32))
    np.testing.assert_array_equal(bound.command_cost.numpy(), [9., 9.])
    for bad in [-1., float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            bound.apply(arr([0, 0], wp.int32), support, weight=bad)


@pytest.mark.parametrize("bad", ["release", "shape", "sides", "delay", "scale", "deadzone"])
def test_binding_rejects_unsupported_or_untrusted_contract(bad):
    action = fixture_action(); active = arr([0, 1, 0, 1, 0, 0], wp.int32)
    sides = ("left", "right"); config = RightWristCommandCostConfig()
    if bad == "release": active = arr([0, 1, 0, 1, 0, 1], wp.int32)
    if bad == "shape": active = arr([0, 1], wp.int32)
    if bad == "sides": sides = ("right", "right")
    if bad == "delay": action.delay = object()
    if bad == "scale": action.buffers.scale = arr([.05, .1, .05, .05, .05, .05])
    if bad == "deadzone": config = RightWristCommandCostConfig(deadzone_m=.35)
    with pytest.raises(ValueError):
        bind_right_wrist_command_penalty(action, active, role_sides=sides,
            release_start_frame=2, config=config)


def test_cost_is_translation_invariant_and_only_penalizes_trusted_right_role():
    action = fixture_action();active = arr([1, 0, 0, 1, 0, 0], wp.int32)
    bound = bind_right_wrist_command_penalty(action, active,
        role_sides=("left", "right"), release_start_frame=2,
        config=RightWristCommandCostConfig(deadzone_m=0))
    target = np.zeros((2, 10), np.float32);target[:, 5] = .1
    action.joint_target.assign(target.reshape(-1))
    initial = action.joint_target.numpy().copy()
    support = arr(np.zeros(2))
    bound.apply(arr([0, 1], wp.int32), support, weight=.1)
    np.testing.assert_allclose(support.numpy(), [0., -.2], atol=1e-6)
    np.testing.assert_array_equal(action.joint_target.numpy(), initial)
    for axis, dof in enumerate(bound.wrist_dofs): target[:, dof] += [3., -4., 5.][axis]
    action.joint_target.assign(target.reshape(-1))
    action.buffers.reference.wrist_pos_w.assign(np.tile([3., -4., 5.], (6, 1)))
    support.zero_();bound.apply(arr([0, 1], wp.int32), support, weight=.1)
    np.testing.assert_allclose(support.numpy(), [0., -.2], atol=1e-6)


@pytest.mark.parametrize("deadzone", [-.01, float("nan"), float("inf")])
def test_config_rejects_invalid_deadzone(deadzone):
    with pytest.raises(ValueError):
        RightWristCommandCostConfig(deadzone_m=deadzone)


def test_nonfinite_active_commands_remain_visible_in_cost_and_reward():
    action = fixture_action();active = arr([0, 1, 0, 0, 0, 0], wp.int32)
    bound = bind_right_wrist_command_penalty(action, active,
        role_sides=("left", "right"), release_start_frame=2)
    targets = np.zeros((2, 10));targets[:, 5] = [np.nan, np.inf]
    action.joint_target.assign(targets.reshape(-1));support = arr(np.ones(2))
    bound.apply(arr([0, 0], wp.int32), support, weight=.1)
    assert not np.isfinite(bound.command_cost.numpy()).any()
    assert not np.isfinite(support.numpy()).any()
