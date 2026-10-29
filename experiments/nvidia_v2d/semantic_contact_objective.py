"""Match trusted robot contact surfaces to object-local reference contact roles.

This extends the release objective and explicitly changes the meaning of the
existing contact_wrench_support term, so old replay must not be reused. Wrench
support is multiplied by mean trusted role agreement. An optional independent,
bounded alignment cost supplies a signal outside the Gaussian's narrow basin.
No action, state, force limit, reference or success criterion is changed here.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

from experiments.nvidia_v2d.release_contact_objective import (
    ReleaseContactObjective,
    ReleaseContactObjectiveConfig,
)


@wp.kernel
def match_semantic_contacts(
    body_pose: wp.array(dtype=wp.transform), timestep: wp.array(dtype=wp.int32),
    raw_contact_count: wp.array(dtype=wp.int32),
    active: wp.array(dtype=wp.int32), body_ids: wp.array(dtype=wp.int32), slot_ids: wp.array(dtype=wp.int32),
    robot_point: wp.array(dtype=wp.vec3), robot_normal: wp.array(dtype=wp.vec3),
    object_point: wp.array(dtype=wp.vec3), object_normal: wp.array(dtype=wp.vec3),
    frames: int, roles: int, bodies_per_world: int, object_body: int, slots_per_world: int,
    inverse_variance: float, role_match: wp.array(dtype=wp.float32),
    contact_support: wp.array(dtype=wp.float32),
):
    world = wp.tid()
    frame = wp.clamp(timestep[world], 0, frames - 1)
    total = float(0.0)  # noqa: UP018 - mutable Warp loop variable
    used = int(0)  # noqa: UP018, RUF046 - mutable Warp loop variable
    object_pose = body_pose[world * bodies_per_world + object_body]
    for role in range(roles):
        i = frame * roles + role
        if active[i] != 0:
            robot_pose = body_pose[world * bodies_per_world + body_ids[i]]
            delta = wp.transform_point(robot_pose, robot_point[i]) - wp.transform_point(object_pose, object_point[i])
            nr = wp.transform_vector(robot_pose, robot_normal[i])
            no = wp.transform_vector(object_pose, object_normal[i])
            alignment = wp.clamp(wp.dot(nr, no), 0.0, 1.0)
            geometry = wp.exp(-wp.dot(delta, delta) * inverse_variance) * alignment
            presence = float(0.0)  # noqa: UP018 - Warp branch variable
            if raw_contact_count[world * slots_per_world + slot_ids[i]] > 0:
                presence = 1.0
            # Dense point/normal agreement remains useful before touching;
            # full reward additionally requires the chosen physical role.
            total += geometry * (0.25 + 0.75 * presence)
            used += 1
    match = float(1.0)  # noqa: UP018 - Warp branch variable
    if used > 0:
        match = total / float(used)
    role_match[world] = match
    contact_support[world] *= match


@wp.kernel
def penalize_role_alignment(
    body_pose: wp.array(dtype=wp.transform), timestep: wp.array(dtype=wp.int32),
    active: wp.array(dtype=wp.int32), body_ids: wp.array(dtype=wp.int32),
    robot_point: wp.array(dtype=wp.vec3), robot_normal: wp.array(dtype=wp.vec3),
    object_point: wp.array(dtype=wp.vec3), object_normal: wp.array(dtype=wp.vec3),
    frames: int, roles: int, bodies_per_world: int, object_body: int, weight: float,
    alignment_cost: wp.array(dtype=wp.float32), contact_support: wp.array(dtype=wp.float32),
):
    world = wp.tid()
    frame = wp.clamp(timestep[world], 0, frames - 1)
    total = float(0.0)  # noqa: UP018 - mutable Warp loop variable
    used = int(0)  # noqa: UP018, RUF046 - mutable Warp loop variable
    object_pose = body_pose[world * bodies_per_world + object_body]
    for role in range(roles):
        i = frame * roles + role
        if active[i] != 0:
            robot_pose = body_pose[world * bodies_per_world + body_ids[i]]
            # Match the independently accepted IK target: 1mm outside the
            # object surface (stored normals point inward).
            target = object_point[i] - 0.001 * object_normal[i]
            distance = wp.length(wp.transform_point(robot_pose, robot_point[i])
                                 - wp.transform_point(object_pose, target))
            dot = wp.clamp(wp.dot(wp.transform_vector(robot_pose, robot_normal[i]),
                                 wp.transform_vector(object_pose, object_normal[i])), -1.0, 1.0)
            # Linear signal between the accepted 6mm tolerance and 30cm cap.
            # The cap bounds failed/free-fall trajectories; normals remain
            # learnable even when opposed. Good roles contribute zero cost.
            total += wp.max(wp.min(distance, 0.3) - 0.006, 0.0) / 0.05
            total += wp.max(0.5 - dot, 0.0) / 1.5
            used += 1
    cost = float(0.0)  # noqa: UP018 - Warp branch variable
    if used > 0:
        cost = total / float(used)
    alignment_cost[world] = cost
    contact_support[world] -= weight * cost


def bind_role_map(path, expected_sha, reference, reference_sha, motion_speed, model, embodiment, contact, worlds):
    from flash_chord.data.resampling import nearest, playback_times

    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError("semantic role map hash changed")
    if hashlib.sha256(Path(reference.metadata.source_path).read_bytes()).hexdigest() != reference_sha:
        raise ValueError("semantic role map belongs to a different reference")
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if data.get("reference_sha256", np.asarray("")).item() != reference_sha:
        raise ValueError("semantic map is not bound to the accepted candidate reference")
    if data["side_names"].tolist() != ["left", "right"]:
        raise ValueError("role map side order must be explicit left/right")
    active = data["active"]
    n = len(active)
    if active.shape != (n, 2) or active.dtype.kind != "b" or not active.any():
        raise ValueError("role map needs nonempty trusted contact evidence")
    feasible = np.asarray(data.get("feasible"))
    status = np.asarray(data.get("source_status"))
    confidence = np.asarray(data.get("confidence"))
    source_active = np.asarray(data.get("source_contact_active"))
    if (feasible.shape != active.shape or feasible.dtype.kind != "b" or not feasible[active].all()):
        raise ValueError("cannot train with infeasible active contact targets")
    if (status.shape != active.shape or not np.isin(status[active], ["consistent", "repaired"]).all()
            or source_active.shape != active.shape or source_active.dtype.kind != "b" or not source_active[active].all()
            or confidence.shape != active.shape or not np.isfinite(confidence).all()
            or not ((confidence[active] > 0) & (confidence[active] <= 1)).all()):
        raise ValueError("active semantic targets lack trusted source evidence")
    for key in ("robot_point_local", "robot_normal_local", "object_point_local", "object_normal_local"):
        values = np.asarray(data[key])
        if values.shape != (n, 2, 3) or not np.isfinite(values).all():
            raise ValueError(f"invalid semantic role field {key}")
        if "normal" in key and not np.allclose(np.linalg.norm(values[active], axis=-1), 1., atol=1e-4):
            raise ValueError(f"active semantic role {key} must have unit normals")
    source_times, target_times = playback_times(n, reference.metadata.source_fps, reference.fps, motion_speed)
    if (len(target_times) != reference.num_frames or not np.allclose(data["source_time_s"], source_times, atol=1e-9, rtol=0)):
        raise ValueError("role map timeline differs from full runtime reference")
    if contact is None or contact.layout.num_objects != 1:
        raise ValueError("semantic contact mapping requires one object and a contact tracker")
    body_count = model.body_count // worlds
    labels = list(model.body_label[:body_count])
    shape_bodies = np.asarray(model.shape_body.numpy())
    body_ids = np.zeros((n, 2), dtype=np.int32)
    slot_ids = np.zeros_like(body_ids)
    for side_index, side in enumerate(("left", "right")):
        hand_index = contact.layout.sides.index(side)
        links = embodiment.hand(side).contact_links
        for f in np.flatnonzero(active[:, side_index]):
            name = str(data["robot_body_name"][f, side_index]).rsplit("/", 1)[-1]
            if name not in (f"{side}_thumb_PP", f"{side}_thumb_DP"):
                raise ValueError("only verified thumb PP/DP roles are supported")
            matches = [i for i, label in enumerate(labels) if label.rsplit("/", 1)[-1] == name]
            if len(matches) != 1:
                raise ValueError(f"semantic robot body is ambiguous: {name}")
            body_id = matches[0]
            slots = [i for i, link in enumerate(links) if body_id in shape_bodies[list(link.shape_ids)]]
            if len(slots) != 1:
                raise ValueError(f"semantic body has no unique physical contact slot: {name}")
            body_ids[f, side_index] = body_id
            slot_ids[f, side_index] = contact.layout.hand_slot_starts[hand_index] + slots[0]
    arrays = {"active": active.astype(np.int32), "body_ids": body_ids, "slot_ids": slot_ids,
              **{key: data[key] for key in ("robot_point_local", "robot_normal_local", "object_point_local", "object_normal_local")}}
    return {key: nearest(values, source_times, target_times) for key, values in arrays.items()}


@dataclass(frozen=True)
class SemanticContactObjectiveConfig(ReleaseContactObjectiveConfig):
    semantic_map_path: str = ""
    semantic_map_sha256: str = ""
    semantic_reference_sha256: str = ""
    semantic_motion_speed: float = .5
    semantic_sigma_m: float = .02
    semantic_alignment_weight: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if not self.semantic_map_path or any(len(value) != 64 for value in (self.semantic_map_sha256, self.semantic_reference_sha256)):
            raise ValueError("semantic contact objective requires hashed map and reference")
        if not all(math.isfinite(x) and x > 0 for x in (self.semantic_motion_speed, self.semantic_sigma_m)):
            raise ValueError("semantic motion speed and sigma must be finite and positive")
        if not math.isfinite(self.semantic_alignment_weight) or not 0 <= self.semantic_alignment_weight <= 1:
            raise ValueError("semantic alignment weight must be finite in [0, 1]")

    def build(self, model, embodiment, robot_reference, reference, command, contact, action,
              world_count, frame_dt, device=None, reference_joint_q_offset=None, episode_start_frame=None):
        inner = super().build(model, embodiment, robot_reference, reference, command, contact, action,
            world_count, frame_dt, device, reference_joint_q_offset, episode_start_frame)
        arrays = bind_role_map(self.semantic_map_path, self.semantic_map_sha256, reference,
            self.semantic_reference_sha256, self.semantic_motion_speed, model, embodiment, contact, world_count)
        if arrays["active"][self.release_start_frame:].any():
            raise ValueError("semantic grasp attraction must be disabled during release")
        device_arrays = {key: wp.array(values.reshape(-1) if values.ndim == 2 else values.reshape(-1, 3),
            dtype=wp.int32 if values.ndim == 2 else wp.vec3, device=device) for key, values in arrays.items()}
        return SemanticContactObjective(inner, self.release_start_frame, device_arrays,
            wp.zeros(world_count, dtype=wp.float32, device=device), model.body_count // world_count,
            contact.layout.object_body_ids[0], self.semantic_sigma_m,
            self.semantic_alignment_weight, wp.zeros(world_count, dtype=wp.float32, device=device),
            wp.zeros(world_count, dtype=wp.float32, device=device),
            wp.zeros(world_count, dtype=wp.float32, device=device),
            wp.zeros(world_count * 2 * 6, dtype=wp.float32, device=device))


@dataclass
class SemanticContactObjective(ReleaseContactObjective):
    role_arrays: dict
    role_match: object
    bodies_per_world: int
    object_body: int
    sigma_m: float
    alignment_weight: float
    alignment_cost: object
    raw_support_snapshot: object
    mixed_support_snapshot: object
    role_diagnostics: object

    def transition_metrics(self):
        from experiments.nvidia_v2d.semantic_contact_metrics import summarize_transition
        return summarize_transition(self.role_diagnostics.numpy(), self.raw_support_snapshot.numpy(),
            self.role_match.numpy(), self.alignment_cost.numpy(), self.mixed_support_snapshot.numpy(),
            self.alignment_weight)

    def evaluate(self, state, timestep, terminated):
        self.inner.evaluate(state, timestep, terminated)
        raw = self.inner.inner
        a = self.role_arrays
        wp.copy(self.raw_support_snapshot, raw.buffers.contact_wrench_support)
        wp.launch(match_semantic_contacts, dim=raw.world_count, inputs=[state.body_q, timestep,
            raw.contact.raw_contact_count, a["active"], a["body_ids"], a["slot_ids"],
            a["robot_point_local"], a["robot_normal_local"], a["object_point_local"], a["object_normal_local"],
            raw.num_frames, 2, self.bodies_per_world, self.object_body, raw.contact.layout.slots_per_world,
            1 / (2 * self.sigma_m**2)], outputs=[self.role_match, raw.buffers.contact_wrench_support])
        if self.alignment_weight > 0:
            wp.launch(penalize_role_alignment, dim=raw.world_count, inputs=[state.body_q, timestep,
                a["active"], a["body_ids"], a["robot_point_local"], a["robot_normal_local"],
                a["object_point_local"], a["object_normal_local"], raw.num_frames, 2,
                self.bodies_per_world, self.object_body, self.alignment_weight],
                outputs=[self.alignment_cost, raw.buffers.contact_wrench_support])
        wp.copy(self.mixed_support_snapshot, raw.buffers.contact_wrench_support)
        from experiments.nvidia_v2d.semantic_contact_metrics import record_roles
        wp.launch(record_roles, dim=(raw.world_count, 2), inputs=[state.body_q, timestep,
            raw.contact.raw_contact_count, a["active"], a["body_ids"], a["slot_ids"],
            a["robot_point_local"], a["robot_normal_local"], a["object_point_local"], a["object_normal_local"],
            raw.num_frames, 2, self.bodies_per_world, self.object_body, raw.contact.layout.slots_per_world],
            outputs=[self.role_diagnostics])
        raw.buffers.compose(terminated, raw.action_rate_l2, raw.action_l2)


def make_semantic_contact_objective(**kwargs):
    return SemanticContactObjectiveConfig(**kwargs)
