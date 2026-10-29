"""Isolated reference-field ablations; no contact-perfection or policy-quality gate.

Normal changes exactly the two reviewed world normals. Support applies the
existing common scene gauge and object support trajectory after source203.
Finger coordinates, source IDs/unknowns and all physics assets stay untouched.
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation

from experiments.nvidia_v2d.support_reference_repair import apply_support_repair

FACTORS = ('normal', 'support', 'normal-support')
EXPECTED_NORMAL_SLOTS = {'left': [(143, 3), (201, 3)], 'right': []}


def value_hash(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), allow_nan=True).encode()).hexdigest()


def apply_normal_factor(cells, patch):
    """Change normals only; retain the original points, IDs and unknown evidence."""
    result = copy.deepcopy(cells)
    for key in ('object_body_position', 'object_body_wxyz'):
        np.testing.assert_array_equal(cells[key], patch['source_' + key])
    rotation = Rotation.from_quat(np.asarray(cells['object_body_wxyz'])[:, 0], scalar_first=True).as_matrix()
    for side in ('left', 'right'):
        np.testing.assert_array_equal(cells[f'mano_{side}_object_contact_part_ids'], patch[f'{side}_part_ids'])
        changed = np.argwhere(patch[f'{side}_status'] == 2)
        if changed.tolist() != [list(x) for x in EXPECTED_NORMAL_SLOTS[side]]:
            raise ValueError('Normal factor must contain only reviewed source143/201 left slot3')
        key = f'mano_{side}_object_contact_normals'
        values = np.asarray(cells[key]).copy()
        local = patch[f'{side}_object_local_normals']
        for frame, slot in changed:
            values[frame, slot] = rotation[frame] @ local[frame, slot]
        result[key] = values.tolist()
    return result


def compose_factor(cells, factor, *, patch=None, support_solution=None):
    if factor not in FACTORS:
        raise ValueError('Unknown geometry factor')
    result = copy.deepcopy(cells)
    if factor in ('normal', 'normal-support'):
        if patch is None:
            raise ValueError('Normal patch required')
        result = apply_normal_factor(result, patch)
    if factor in ('support', 'normal-support'):
        if support_solution is None:
            raise ValueError('Support solution required')
        result = apply_support_repair(result, support_solution)
    return result


def audit_factor(original, candidate, factor, *, patch, support_solution):
    """Read back serialized fields and independently check declared isolation.

    This is a numerical/factor audit, not collision, contact, natural-grasp or
    learned-success acceptance. Matrix comparisons tolerate float32 serialization.
    """
    if factor not in FACTORS or original.keys() != candidate.keys():
        raise ValueError('Invalid factor or schema fields')
    has_normal = factor in ('normal', 'normal-support')
    has_support = factor in ('support', 'normal-support')
    first = support_solution['manifest']['config']['last_unchanged_frame'] + 1
    if first != 204:
        raise ValueError('Unexpected support prefix')
    checks = dict.fromkeys(('finite_arrays', 'frame_count', 'factor_scope',
                            'unchanged_joint_trajectories', 'unchanged_physics'), False)
    details = {}
    changed = {}
    for key in original:
        equal = original[key] == candidate[key]
        if not equal:
            before, after = np.asarray(original[key]), np.asarray(candidate[key])
            indices = np.argwhere(np.any(before != after, axis=tuple(range(1, before.ndim))) if before.ndim > 1 else before != after)
            changed[key] = {'original_field_sha256': value_hash(original[key]),
                            'candidate_field_sha256': value_hash(candidate[key]),
                            'changed_source_frames': indices[:, 0].tolist()}
    for key, value in candidate.items():
        array = np.asarray(value)
        if array.dtype.kind in 'fiu' and not np.isfinite(array).all():
            raise ValueError(f'Nonfinite field: {key}')
    checks['finite_arrays'] = True
    if len(candidate['object_body_position']) != 249 or candidate['fps'] != 20 or original['fps'] != candidate['fps']:
        raise ValueError('Source time axis changed')
    checks['frame_count'] = True
    normal_fields = {f'mano_{s}_object_contact_normals' for s in ('left', 'right')}
    support_fields = {'object_body_position', 'object_body_wxyz', 'object_root_position', 'object_root_axis_angle'}
    for side in ('left', 'right'):
        support_fields |= {f'robot_{side}_{s}' for s in ('wrist_position', 'wrist_wxyz', 'frames')}
        support_fields |= {f'mano_{side}_{s}' for s in ('joints', 'joints_wxyz', 'trans', 'global_orient', 'tips_distance',
                           'link_contact_positions', 'object_contact_positions', 'link_contact_normals', 'object_contact_normals')}
    allowed = (normal_fields if has_normal else set()) | (support_fields if has_support else set())
    if not set(changed) <= allowed:
        raise ValueError(f'Undeclared field changes: {set(changed) - allowed}')
    gauge = support_solution['scene_gauge_delta_world']
    gr, gp = gauge[:, :3, :3], gauge[:, :3, 3]
    max_fk_position_delta = 0.
    for side in ('left', 'right'):
        for suffix in ('finger_joints',):
            np.testing.assert_array_equal(candidate[f'robot_{side}_{suffix}'], original[f'robot_{side}_{suffix}'])
        for suffix in ('object_contact_part_ids', 'finger_pose', 'betas'):
            key = f'mano_{side}_{suffix}'
            np.testing.assert_array_equal(candidate[key], original[key])
        ids = np.asarray(original[f'mano_{side}_object_contact_part_ids'])
        np.testing.assert_array_equal(ids, patch[f'{side}_part_ids'])
        for key in original:
            if key not in allowed:
                np.testing.assert_array_equal(candidate[key], original[key])
        # The two normal replacements are before the support gauge begins.
        key = f'mano_{side}_object_contact_normals'
        expected = np.asarray(original[key]).copy()
        if has_normal:
            rotations = Rotation.from_quat(np.asarray(original['object_body_wxyz'])[:, 0], scalar_first=True).as_matrix()
            for frame, slot in EXPECTED_NORMAL_SLOTS[side]:
                expected[frame, slot] = rotations[frame] @ patch[f'{side}_object_local_normals'][frame, slot]
        if has_support:
            expected[first:] = np.einsum('tij,tkj->tki', gr[first:], expected[first:])
        np.testing.assert_allclose(candidate[key], expected, atol=2e-7, rtol=0)
        # Point coordinates never change in normal-only; inactive support contacts remain zero.
        if not has_support:
            np.testing.assert_array_equal(candidate[f'mano_{side}_object_contact_positions'], original[f'mano_{side}_object_contact_positions'])
        else:
            for key in support_fields:
                if key in candidate and key not in normal_fields:
                    np.testing.assert_array_equal(candidate[key][:first], original[key][:first])
            for suffix in ('wrist_position', 'frames'):
                key = f'robot_{side}_{suffix}';a = np.asarray(original[key]);b = np.asarray(candidate[key])
                positions = a if suffix == 'wrist_position' else a[..., :3]
                expected_p = (np.einsum('tij,tj->ti', gr, positions) + gp if suffix == 'wrist_position'
                              else np.einsum('tij,tkj->tki', gr, positions) + gp[:, None])
                expected_p[:first] = positions[:first]
                if suffix == 'frames':
                    universe = original[f'{side}_robot_frame_names'].index('universe')
                    expected_p[:, universe] = positions[:, universe]
                actual_p = b if suffix == 'wrist_position' else b[..., :3]
                np.testing.assert_allclose(actual_p, expected_p, atol=2e-7, rtol=0)
                max_fk_position_delta = max(max_fk_position_delta, float(np.max(abs(actual_p - expected_p))))
            for suffix in ('wrist_wxyz', 'frames'):
                key = f'robot_{side}_{suffix}';a = np.asarray(original[key]);b = np.asarray(candidate[key])
                a = a if suffix == 'wrist_wxyz' else a[..., 3:];b = b if suffix == 'wrist_wxyz' else b[..., 3:]
                ar = Rotation.from_quat(a.reshape(-1, 4), scalar_first=True).as_matrix().reshape(a.shape[:-1] + (3, 3))
                br = Rotation.from_quat(b.reshape(-1, 4), scalar_first=True).as_matrix().reshape(b.shape[:-1] + (3, 3))
                expected_r = gr @ ar if suffix == 'wrist_wxyz' else gr[:, None] @ ar
                expected_r[:first] = ar[:first]
                if suffix == 'frames':
                    expected_r[:, universe] = ar[:, universe]
                np.testing.assert_allclose(br, expected_r, atol=3e-7, rtol=0)
            # Independently verify every MANO world-field affected by the common gauge.
            for suffix in ('joints', 'link_contact_positions', 'object_contact_positions'):
                key = f'mano_{side}_{suffix}';a = np.asarray(original[key])
                expected_p = np.einsum('tij,tkj->tki', gr, a) + gp[:, None]
                if 'contact' in suffix:
                    expected_p[ids == 0] = 0
                expected_p[:first] = a[:first]
                np.testing.assert_allclose(candidate[key], expected_p, atol=2e-7, rtol=0)
            key = f'mano_{side}_link_contact_normals';a = np.asarray(original[key])
            expected_n = np.einsum('tij,tkj->tki', gr, a);expected_n[:first] = a[:first]
            np.testing.assert_allclose(candidate[key], expected_n, atol=2e-7, rtol=0)
            key = f'mano_{side}_trans';a = np.asarray(original[key]);rest = np.asarray(support_solution['manifest']['mano_rest_roots'][side])
            expected_t = np.einsum('tij,tj->ti', gr, a + rest) + gp - rest;expected_t[:first] = a[:first]
            np.testing.assert_allclose(candidate[key], expected_t, atol=2e-7, rtol=0)
            for suffix in ('global_orient', 'joints_wxyz'):
                key = f'mano_{side}_{suffix}';a = np.asarray(original[key]);b = np.asarray(candidate[key])
                if suffix == 'global_orient':
                    ar = Rotation.from_rotvec(a).as_matrix();br = Rotation.from_rotvec(b).as_matrix();expected_r = gr @ ar
                else:
                    ar = Rotation.from_quat(a.reshape(-1,4), scalar_first=True).as_matrix().reshape(a.shape[:-1]+(3,3))
                    br = Rotation.from_quat(b.reshape(-1,4), scalar_first=True).as_matrix().reshape(b.shape[:-1]+(3,3));expected_r = gr[:, None] @ ar
                expected_r[:first] = ar[:first]
                np.testing.assert_allclose(br, expected_r, atol=3e-7, rtol=0)
            np.testing.assert_allclose(candidate[f'mano_{side}_tips_distance'], support_solution[f'repaired_{side}_tips_distance'], atol=2e-7, rtol=0)
    if has_support:
        repaired = support_solution['repaired_object_to_world'][:, 0]
        np.testing.assert_allclose(np.asarray(candidate['object_body_position'])[first:, 0], repaired[first:, :3, 3], atol=2e-7, rtol=0)
        actual_r = Rotation.from_quat(np.asarray(candidate['object_body_wxyz'])[first:, 0], scalar_first=True).as_matrix()
        np.testing.assert_allclose(actual_r, repaired[first:, :3, :3], atol=3e-7, rtol=0)
        np.testing.assert_allclose(np.asarray(candidate['object_root_position'])[first:], repaired[first:, :3, 3], atol=2e-7, rtol=0)
        np.testing.assert_allclose(Rotation.from_rotvec(np.asarray(candidate['object_root_axis_angle'])[first:]).as_matrix(), repaired[first:, :3, :3], atol=3e-7, rtol=0)
    checks['factor_scope'] = checks['unchanged_joint_trajectories'] = True
    # Only reference-field changes are allowed above; physical assets remain identical paths.
    for key in ('object_mesh_paths', 'object_urdf_paths'):
        np.testing.assert_array_equal(candidate[key], original[key])
    checks['unchanged_physics'] = True
    details.update(changed_fields=changed, normal_repaired_slots=EXPECTED_NORMAL_SLOTS if has_normal else {},
                   support_first_changed_frame=204 if has_support else None,
                   robot_world_fields_follow_common_gauge=has_support, max_robot_position_gauge_residual_m=max_fk_position_delta,
                   all_finger_q_exactly_unchanged=True, all_source_contact_ids_and_unknowns_preserved=True,
                   contactaware_ik=False, q_smoothing=False, semantic_reward=False,
                   meaning='Reference factor isolation only; no policy, natural-contact or collision-quality success claim.')
    return {'factor': factor, 'checks': checks, 'details': details}
