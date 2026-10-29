"""Bounded wrist-only feedback for a physical grasp-feasibility sweep.

This is a scripted controller, never a trained-policy success result. It sends
ordinary residual actions through the existing EMA, collision solver and wrist
force limits. It does not write object state, forces, gains or reference data.
"""
from __future__ import annotations

import numpy as np


def rotate_xyzw(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    v = np.asarray(vector, dtype=float)
    uv = np.cross(q[..., :3], v)
    return v + 2 * (q[..., 3:4] * uv + np.cross(q[..., :3], uv))


def smooth_gate(seconds: np.ndarray, phases: tuple[float, float, float, float]) -> np.ndarray:
    start, full, release, end = phases
    if not 0 <= start < full <= release < end:
        raise ValueError("feedback phases must satisfy 0 <= start < full <= release < end")
    up = np.clip((seconds - start) / (full - start), 0, 1)
    down = np.clip((end - seconds) / (end - release), 0, 1)
    return (up * up * (3 - 2 * up)) * (down * down * (3 - 2 * down))


def wrist_residuals(
    reference_wrists: np.ndarray,
    reference_object_pose: np.ndarray,
    live_object_pose: np.ndarray,
    offsets: np.ndarray,
    gate: np.ndarray,
    center_feedback_max: float,
) -> np.ndarray:
    """Return [world, hand, xyz] residuals; lift z remains reference commanded.

    The opposing inward vectors are computed in the reference object's frame,
    then rotated with the LIVE object's orientation. A bounded LIVE horizontal
    center-error term follows incidental pot translation. World 0 with offset 0
    remains exactly the unmodified reference baseline.
    """
    reference_wrists = np.asarray(reference_wrists, dtype=float)
    ref = np.asarray(reference_object_pose, dtype=float)
    live = np.asarray(live_object_pose, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    if reference_wrists.shape != (len(offsets), 2, 3):
        raise ValueError("feedback requires two reference wrists per world")
    if not np.isfinite(reference_wrists).all() or not np.isfinite(ref).all() or not np.isfinite(live).all():
        raise ValueError("nonfinite feedback inputs")
    inverse_ref = ref[:, 3:7].copy()
    inverse_ref[:, :3] *= -1
    inward_local = rotate_xyzw(inverse_ref[:, None], ref[:, None, :3] - reference_wrists)
    inward = rotate_xyzw(live[:, None, 3:7], inward_local)
    inward[..., 2] = 0
    norm = np.linalg.norm(inward, axis=-1, keepdims=True)
    inward = np.divide(inward, norm, out=np.zeros_like(inward), where=norm > 1e-8)
    center_error = live[:, :3] - ref[:, :3]
    center_error[:, 2] = 0
    distance = np.linalg.norm(center_error, axis=-1, keepdims=True)
    center_error *= np.minimum(1, center_feedback_max / np.maximum(distance, 1e-8))
    # Scale the correction continuously with the offset, preserving zero control.
    center_gain = offsets / max(float(np.max(offsets)), 1e-8)
    residual = offsets[:, None, None] * inward + center_gain[:, None, None] * center_error[:, None]
    residual *= gate[:, None, None]
    return residual


def encode_residual(physical: np.ndarray, buffers) -> np.ndarray:
    """Invert the configured input map without bypassing its scale or EMA."""
    scale = np.asarray(buffers.scale.numpy())
    clip = np.asarray(buffers.clip.numpy())
    if np.any(np.abs(physical) > clip + 1e-7):
        raise ValueError("scripted residual exceeds configured physical action bounds")
    raw = physical / scale
    if buffers.input_mapping == "identity":
        return raw.astype(np.float32)
    bound = np.asarray(buffers.input_scale_values)
    if buffers.input_mapping == "linear":
        return (raw / bound).astype(np.float32)
    if buffers.input_mapping == "rational":
        return (raw / (1 + np.abs(raw) * (1 - 1 / bound))).astype(np.float32)
    raise ValueError(f"unsupported residual input mapping {buffers.input_mapping!r}")


class ScriptedGraspFeedback:
    """Read live state on every control step and produce normal policy inputs."""

    def __init__(
        self, env, reference, *, max_offset: float = 0.04, center_feedback_max: float = 0.03,
        phases: tuple[float, float, float, float] = (2.5, 3.3, 8.5, 9.5),
    ):
        if not 0 < max_offset <= 0.04 or not 0 <= center_feedback_max <= 0.04:
            raise ValueError("diagnostic wrist and feedback offsets must be bounded by 4 cm")
        if env.world_count < 2:
            raise ValueError("the sweep requires at least 2 worlds including a zero baseline")
        self.env = env
        self.buffers = env.action.buffers
        self.layout = self.buffers.layout
        if set(self.layout.sides) != {"left", "right"}:
            raise ValueError("expected Sharpa left/right hand action layout")
        if env.command.layout.num_bodies != 1:
            raise ValueError("grasp feedback currently requires one rigid object body")
        self.body_ids = env.command.body_ids_w.numpy().reshape(env.world_count)
        self.offsets = np.linspace(0, max_offset, env.world_count)
        self.center_feedback_max = center_feedback_max
        self.phases = phases
        smooth_gate(np.zeros(1), phases)
        self.reference_wrists = np.stack([reference.wrist_pos_w(s) for s in self.layout.sides], axis=1)
        obj_pos = np.asarray(reference.object_body_pos_w()).reshape(-1, 1, 3)[:, 0]
        obj_quat = np.asarray(reference.object_body_quat_w()).reshape(-1, 1, 4)[:, 0]
        # Reference API is wxyz; Newton live body transforms use xyzw.
        self.reference_objects = np.concatenate((obj_pos, obj_quat[:, [1, 2, 3, 0]]), axis=1)
        self.source_speed = float(env.scene.motion_speed) if hasattr(env.scene, "motion_speed") else None
        self.fps = float(env.config.sim.fps)
        self.motion_speed = float(env.config.motion_speed) if hasattr(env.config, "motion_speed") else None
        # The caller supplies speed because scene/reference FPS describe resampled time.
        self.steps = []
        self.live_objects = []
        self.residuals = []
        self.actions = []

    def action(self, *, motion_speed: float) -> np.ndarray:
        frame = np.clip(self.env.timestep.numpy(), 0, len(self.reference_objects) - 1).astype(int)
        live = self.env.state_0.body_q.numpy()[self.body_ids]
        seconds = frame * motion_speed / self.fps
        residual = wrist_residuals(
            self.reference_wrists[frame], self.reference_objects[frame], live,
            self.offsets, smooth_gate(seconds, self.phases), self.center_feedback_max,
        )
        physical = np.zeros((self.env.world_count, self.env.action.action_dim))
        for hand, start in enumerate(self.layout.hand_action_starts):
            physical[:, start:start + 3] = residual[:, hand]
        action = encode_residual(physical, self.buffers)
        self.steps.append(frame.copy())
        self.live_objects.append(live.copy())
        self.residuals.append(residual)
        self.actions.append(action)
        return action

    def metadata(self) -> dict:
        return {
            "controller": "scripted_live_object_wrist_feedback",
            "claim": "physical feasibility only; not a learned-policy success",
            "offsets_m_by_world": self.offsets.tolist(),
            "center_feedback_max_m": self.center_feedback_max,
            "phase_source_seconds": list(self.phases),
            "action_sides": list(self.layout.sides),
            "action_starts": list(self.layout.hand_action_starts),
            "finger_residual": 0.0,
            "vertical_residual": 0.0,
            "orientation_residual": 0.0,
            "physics_and_effort_limits_modified": False,
        }

    def traces(self) -> dict:
        return {
            "feedback_reference_frame": np.asarray(self.steps),
            "feedback_live_object_pose": np.asarray(self.live_objects),
            "feedback_wrist_residual": np.asarray(self.residuals),
            "feedback_policy_input": np.asarray(self.actions),
        }
