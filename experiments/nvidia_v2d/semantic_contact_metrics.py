"""Read-only diagnostics captured at objective evaluation, before auto-reset."""
from __future__ import annotations

import numpy as np
import warp as wp


@wp.kernel
def record_roles(
    body_pose: wp.array(dtype=wp.transform), timestep: wp.array(dtype=wp.int32),
    raw_contact_count: wp.array(dtype=wp.int32), active: wp.array(dtype=wp.int32),
    body_ids: wp.array(dtype=wp.int32), slot_ids: wp.array(dtype=wp.int32),
    robot_point: wp.array(dtype=wp.vec3), robot_normal: wp.array(dtype=wp.vec3),
    object_point: wp.array(dtype=wp.vec3), object_normal: wp.array(dtype=wp.vec3),
    frames: int, roles: int, bodies_per_world: int, object_body: int, slots_per_world: int,
    values: wp.array(dtype=wp.float32),
):
    world, role = wp.tid()
    frame = wp.clamp(timestep[world], 0, frames - 1)
    i = frame * roles + role
    out = (world * roles + role) * 6
    for column in range(6):
        values[out + column] = 0.0
    if active[i] != 0:
        robot_pose = body_pose[world * bodies_per_world + body_ids[i]]
        object_pose = body_pose[world * bodies_per_world + object_body]
        target = object_point[i] - 0.001 * object_normal[i]
        distance = wp.length(wp.transform_point(robot_pose, robot_point[i])
                             - wp.transform_point(object_pose, target))
        dot = wp.clamp(wp.dot(wp.transform_vector(robot_pose, robot_normal[i]),
                             wp.transform_vector(object_pose, object_normal[i])), -1.0, 1.0)
        values[out] = 1.0
        values[out + 1] = distance
        values[out + 2] = dot
        if raw_contact_count[world * slots_per_world + slot_ids[i]] > 0:
            values[out + 3] = 1.0
        values[out + 4] = wp.max(wp.min(distance, 0.3) - 0.006, 0.0) / 0.05 + wp.max(0.5 - dot, 0.0) / 1.5
        if distance >= 0.3:
            values[out + 5] = 1.0


def summarize_transition(role_values, base_support, role_match, alignment_cost, mixed_support, weight):
    """Means over worlds; side geometry/presence means use active roles only.

    These are the last collected transition, not optimizer update means or
    evaluation success. Empty side cohorts emit a count of zero and no geometry
    value, so absent evidence is not misreported as zero error.
    """
    roles = np.asarray(role_values).reshape(-1, 2, 6)
    result = {
        "contact/raw_wrench_support_mean": float(np.mean(base_support)),
        "contact/role_match_mean": float(np.mean(role_match)),
        "contact/matched_wrench_support_mean": float(np.mean(np.asarray(base_support) * np.asarray(role_match))),
        "contact/alignment_cost_mean": float(np.mean(alignment_cost)),
        "contact/alignment_penalty_mean": float(weight * np.mean(alignment_cost)),
        "contact/mixed_support_mean": float(np.mean(mixed_support)),
    }
    active_worlds = np.any(roles[:, :, 0] == 1, axis=1)
    result["contact/active_worlds"] = float(active_worlds.sum())
    if active_worlds.any():
        result["contact/role_match_active_mean"] = float(np.asarray(role_match)[active_worlds].mean())
        result["contact/alignment_cost_active_mean"] = float(np.asarray(alignment_cost)[active_worlds].mean())
    names = ("target_distance_m", "normal_dot", "role_pair_fraction", "geometry_cost", "distance_cap_fraction")
    for side_index, side in enumerate(("left", "right")):
        rows = roles[:, side_index]
        mask = rows[:, 0] == 1
        result[f"contact/{side}/active_worlds"] = float(mask.sum())
        if mask.any():
            for column, name in enumerate(names, 1):
                result[f"contact/{side}/{name}"] = float(np.mean(rows[mask, column]))
            strict = (rows[mask, 1] <= .006) & (rows[mask, 2] >= .5) & (rows[mask, 3] == 1)
            result[f"contact/{side}/aligned_pair_fraction"] = float(strict.mean())
    return result
