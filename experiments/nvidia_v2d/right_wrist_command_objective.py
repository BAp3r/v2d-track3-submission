"""Optional reward-only prior on trusted right-hand wrist translation commands.

This module does not alter the actor, action mapping, joint targets, controller,
forces, or termination. Integrating its cost changes stored objective terms and
therefore requires fresh replay and an explicit checkpoint/config contract.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp


@wp.kernel
def penalize_right_wrist_command(
    timestep: wp.array(dtype=wp.int32),
    active: wp.array(dtype=wp.int32),
    reference_wrist: wp.array(dtype=wp.vec3),
    joint_target: wp.array(dtype=wp.float32),
    frames: int, roles: int, right_role: int, release_start: int,
    hands: int, right_hand: int, dofs: int,
    wrist_x: int, wrist_y: int, wrist_z: int,
    deadzone_m: float, scale_m: float, radial_cap_m: float, weight: float,
    command_distance: wp.array(dtype=wp.float32),
    command_cost: wp.array(dtype=wp.float32),
    contact_support: wp.array(dtype=wp.float32),
):
    world = wp.tid()
    frame = wp.clamp(timestep[world], 0, frames - 1)
    distance = float(0.0)  # noqa: UP018 - mutable Warp branch variable
    cost = float(0.0)  # noqa: UP018 - mutable Warp branch variable
    # Branch before reading any reference/target value. Unknown and release
    # frames are excluded supervision, not zero-contact ground truth.
    if timestep[world] < release_start and active[frame * roles + right_role] != 0:
        base = world * dofs
        target = wp.vec3(joint_target[base + wrist_x],
                         joint_target[base + wrist_y],
                         joint_target[base + wrist_z])
        reference = reference_wrist[frame * hands + right_hand]
        distance = wp.length(target - reference)
        if wp.isfinite(distance):
            cost = wp.max(wp.min(distance, radial_cap_m) - deadzone_m, 0.0) / scale_m
        else:
            cost = distance  # Preserve invalid active data for the metric failure path.
    command_distance[world] = distance
    command_cost[world] = cost
    contact_support[world] -= weight * cost


@dataclass(frozen=True)
class RightWristCommandCostConfig:
    """Meters; weight is supplied separately and deliberately defaults to zero."""

    deadzone_m: float = 0.02

    def __post_init__(self):
        if not math.isfinite(self.deadzone_m) or self.deadzone_m < 0:
            raise ValueError("wrist command deadzone must be finite and nonnegative")


@dataclass
class RightWristCommandPenalty:
    """Bound references to the existing residual-action buffers; owns only diagnostics."""

    active: object
    reference_wrist: object
    joint_target: object
    worlds: int
    frames: int
    roles: int
    right_role: int
    release_start: int
    hands: int
    right_hand: int
    dofs: int
    wrist_dofs: tuple[int, int, int]
    config: RightWristCommandCostConfig
    radial_cap_m: float
    scale_m: float
    command_distance: object
    command_cost: object

    def apply(self, timestep, contact_support, *, weight: float = 0.0):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("wrist command weight must be finite and nonnegative")
        if weight == 0:
            return  # Preserve the old compute path; no kernel or buffer write.
        wp.launch(penalize_right_wrist_command, dim=self.worlds,
            inputs=[timestep, self.active, self.reference_wrist, self.joint_target,
                    self.frames, self.roles, self.right_role, self.release_start,
                    self.hands, self.right_hand, self.dofs, *self.wrist_dofs,
                    self.config.deadzone_m, self.scale_m, self.radial_cap_m, weight],
            outputs=[self.command_distance, self.command_cost, contact_support],
            device=self.command_cost.device)


def bind_right_wrist_command_penalty(
    action, active, *, role_sides: tuple[str, ...], release_start_frame: int,
    config: RightWristCommandCostConfig | None = None,
) -> RightWristCommandPenalty:
    """Bind the exact reference used by ResidualHandPoseAction.process.

``active`` is the already SHA-validated, runtime-resampled semantic map's
flattened int32 [frames * roles] array. Explicit side names prevent confusion
between semantic left/right order and policy right/left order. Delayed or other
action implementations are deliberately unsupported by this narrow adapter.
"""
    from flash_chord.runtime.actions.hand_pose import ResidualHandPoseAction

    if not isinstance(action, ResidualHandPoseAction) or action.delay is not None:
        raise ValueError("right wrist command cost requires undelayed ResidualHandPoseAction")
    cfg = config or RightWristCommandCostConfig()
    buffers = action.buffers
    layout, reference = buffers.layout, buffers.reference
    if len(set(role_sides)) != len(role_sides) or role_sides.count("right") != 1:
        raise ValueError("semantic role order must contain exactly one right hand")
    if tuple(reference.sides) != tuple(layout.sides) or layout.sides.count("right") != 1:
        raise ValueError("action and reference hand orders disagree")
    frames = reference.num_frames
    if type(release_start_frame) is not int or not 0 <= release_start_frame < frames:
        raise ValueError("release boundary must lie in the runtime reference")
    if active.dtype != wp.int32 or active.shape != (frames * len(role_sides),):
        raise ValueError("semantic active array has the wrong dtype or runtime shape")
    mask = active.numpy().reshape(frames, len(role_sides))
    if not np.isin(mask, [0, 1]).all() or mask[release_start_frame:].any():
        raise ValueError("semantic active array must be boolean-valued with zero release suffix")
    right = layout.sides.index("right")
    dofs = tuple(int(x) for x in layout.wrist_pos_dof_ids[right * 3:right * 3 + 3])
    if len(set(dofs)) != 3 or any(x < 0 or x >= layout.num_joint_dof for x in dofs):
        raise ValueError("right wrist needs three unique valid translation DOFs")
    if reference.wrist_pos_w.shape != (frames * layout.num_hands,):
        raise ValueError("action wrist reference shape disagrees with its layout")
    if action.joint_target.shape != (buffers.world_count * layout.num_joint_dof,):
        raise ValueError("action target shape disagrees with its world/DOF layout")
    action_start = layout.hand_action_starts[right]
    clip = buffers.clip.numpy()[action_start:action_start + 3]
    if clip.shape != (3,) or not np.isfinite(clip).all() or np.any(clip <= 0):
        raise ValueError("action translation clips must be finite and positive")
    scale = buffers.scale.numpy()[action_start:action_start + 3]
    if (scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0)
            or not np.all(scale == scale[0])):
        raise ValueError("right wrist command cost requires equal positive translation scales")
    scale_m = float(scale[0])
    radial_cap = float(np.linalg.norm(clip.astype(float)))
    if cfg.deadzone_m >= radial_cap:
        raise ValueError("deadzone must be smaller than the reachable radial action bound")
    device = action.joint_target.device
    if active.device != device or reference.wrist_pos_w.device != device:
        raise ValueError("action, reference and active buffers must use the same device")
    return RightWristCommandPenalty(active, reference.wrist_pos_w, action.joint_target,
        buffers.world_count, frames, len(role_sides), role_sides.index("right"),
        release_start_frame, layout.num_hands, right, layout.num_joint_dof, dofs,
        cfg, radial_cap, scale_m, wp.zeros(buffers.world_count, dtype=wp.float32, device=device),
        wp.zeros(buffers.world_count, dtype=wp.float32, device=device))
