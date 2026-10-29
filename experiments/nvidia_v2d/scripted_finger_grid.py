"""Optional finger-closure ablation; intentionally not wired into evaluation.

The recorded human keeps the four non-thumb fingertips extended. Thumb-only
pinching already lifts the pot in simulation, so this alternative grasp must not
be presented as a reconstruction correction or as learned-policy success.
"""
from __future__ import annotations

import numpy as np

from experiments.nvidia_v2d.scripted_grasp_feedback import (
    ScriptedGraspFeedback,
    encode_residual,
    smooth_gate,
)


def grid_parameters() -> tuple[np.ndarray, np.ndarray]:
    """Four inward offsets crossed with four positive flexion residuals."""
    return np.repeat(np.linspace(0, 0.04, 4), 4), np.tile([0, 0.15, 0.30, 0.45], 4)


def closure_mask(names: tuple[str, ...]) -> np.ndarray:
    return np.array([
        "thumb" not in name and name.endswith(("_MCP_FE", "_PIP", "_DIP"))
        for name in names
    ], dtype=bool)


def bounded_closure(reference: np.ndarray, amount: np.ndarray, mask: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Only add positive flexion, bounded by each simulated joint's limit."""
    return np.minimum(amount[:, None], np.maximum(upper - reference, 0)) * mask[None]


class ScriptedFingerGrid(ScriptedGraspFeedback):
    """Drop-in optional controller for an explicitly named 16-world ablation.

    The parent diagnostic would instantiate this class instead of
    ScriptedGraspFeedback; no other integration or GPU execution is performed
    by this module. Actions still pass through normal scale/EMA/PD/force limits.
    """

    def __init__(self, env, reference, **kwargs):
        if env.world_count != 16:
            raise ValueError("finger grid requires exactly 16 worlds")
        super().__init__(env, reference, **kwargs)
        self.offsets, self.closures = grid_parameters()
        self.finger_residuals = []
        self.hand_finger_names = {}
        self.finger_reference = {}
        self.finger_upper = {}
        self.masks = {}
        upper = env.scene.model.joint_limit_upper.numpy().reshape(env.world_count, -1)
        for side in self.layout.sides:
            hand = env.scene.layout.hand(side)
            self.hand_finger_names[side] = tuple(hand.finger_joint_names)
            self.finger_reference[side] = np.asarray(env.scene.robot_reference.hand(side).finger_joint_pos)
            self.finger_upper[side] = upper[:, hand.finger_dof_ids]
            self.masks[side] = closure_mask(self.hand_finger_names[side])
            if self.masks[side].sum() != 12:
                raise ValueError("expected 12 non-thumb MCP/PIP/DIP flexion joints")

    def action(self, *, motion_speed: float) -> np.ndarray:
        super().action(motion_speed=motion_speed)
        frame = self.steps[-1]
        gate = smooth_gate(frame * motion_speed / self.fps, self.phases)
        physical = np.zeros((self.env.world_count, self.env.action.action_dim))
        per_hand = []
        for hand_index, (side, start) in enumerate(zip(self.layout.sides, self.layout.hand_action_starts, strict=True)):
            physical[:, start:start + 3] = self.residuals[-1][:, hand_index]
            fingers = bounded_closure(
                self.finger_reference[side][frame], self.closures * gate,
                self.masks[side], self.finger_upper[side],
            )
            physical[:, start + 6:start + 6 + fingers.shape[1]] = fingers
            per_hand.append(fingers)
        encoded = encode_residual(physical, self.buffers)
        self.actions[-1] = encoded
        self.finger_residuals.append(np.stack(per_hand, axis=1))
        return encoded

    def metadata(self) -> dict:
        metadata = super().metadata()
        metadata.update(
            controller="scripted_alternative_finger_grasp_grid",
            claim="alternative grasp feasibility; not original human contact reconstruction or learned success",
            finger_closure_rad_by_world=self.closures.tolist(),
            finger_joint_names_by_side=self.hand_finger_names,
            finger_closure_rule="positive non-thumb MCP_FE/PIP/DIP only; clipped to simulation upper limits",
        )
        metadata.pop("finger_residual")
        return metadata

    def traces(self) -> dict:
        return {**super().traces(), "feedback_finger_residual": np.asarray(self.finger_residuals)}
