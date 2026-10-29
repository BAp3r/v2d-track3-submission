import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from experiments.nvidia_v2d.static_support_depth import project_scene
from experiments.nvidia_v2d.support_reference_repair import (
    RepairConfig,
    apply_scene_gauge,
    apply_support_repair,
    quintic_blend,
    recover_mano_rest_root,
    static_support_path,
    support_metrics,
    transform_points,
)


def box_vertices(half=.05):
    return np.array([[x, y, z] for x in (-half, half) for y in (-half, half)
                     for z in (-half, half)])


def toy_row(n=4):
    identity = np.tile([1., 0, 0, 0], (n, 1))
    row = {"object_body_position": np.zeros((n, 1, 3)).tolist(),
           "object_body_wxyz": identity[:, None].tolist(),
           "object_root_position": np.zeros((n, 3)).tolist(),
           "object_root_axis_angle": np.zeros((n, 3)).tolist()}
    for side in ("left", "right"):
        prefix = f"mano_{side}_"
        row[prefix + "trans"] = np.tile([.1, .2, .3], (n, 1)).tolist()
        row[prefix + "global_orient"] = np.zeros((n, 3)).tolist()
        row[prefix + "joints"] = np.tile([.1, .2, .3], (n, 21, 1)).tolist()
        row[prefix + "joints_wxyz"] = np.tile(identity[:, None], (1, 21, 1)).tolist()
        row[prefix + "tips_distance"] = np.zeros((n, 5)).tolist()
        row[prefix + "object_contact_part_ids"] = np.tile([1, 0], (n, 1)).tolist()
        for suffix in ("link_contact_positions", "object_contact_positions"):
            row[prefix + suffix] = np.tile([[.1, .2, .3], [0, 0, 0]], (n, 1, 1)).tolist()
        for suffix in ("link_contact_normals", "object_contact_normals"):
            row[prefix + suffix] = np.tile([[1., 0, 0], [0, 0, 0]], (n, 1, 1)).tolist()
        row[f"robot_{side}_wrist_position"] = np.tile([.12, .2, .3], (n, 1)).tolist()
        row[f"robot_{side}_wrist_wxyz"] = identity.tolist()
        row[f"robot_{side}_finger_joints"] = np.tile([.2, .4], (n, 1)).tolist()
        row[f"{side}_robot_frame_names"] = ["universe", "root_joint", "tip"]
        row[f"robot_{side}_frames"] = np.tile([[0, 0, 0, 1, 0, 0, 0],
                                              [.12, .2, .3, 1, 0, 0, 0],
                                              [.15, .2, .3, 1, 0, 0, 0]], (n, 1, 1)).tolist()
    return row


def test_quintic_has_stationary_endpoint_derivatives():
    assert quintic_blend(0) == 0 and quintic_blend(1) == 1
    h = 1e-4
    assert quintic_blend(h) / h < 1e-6
    assert (1 - quintic_blend(1-h)) / h < 1e-6
    assert (quintic_blend(2*h) - 2*quintic_blend(h)) / h**2 < .01


def test_supported_path_preserves_prefix_and_becomes_static():
    config = RepairConfig(last_unchanged_frame=2, static_start_frame=7, fit_frames=(7, 8))
    original = np.tile(np.eye(4), (10, 1, 1))
    original[:, 2, 3] = .15
    gauge = np.tile(np.eye(4), (10, 1, 1))
    fixed = np.eye(4)
    fixed[2, 3] = .05
    result = static_support_path(original, gauge, fixed, box_vertices(), config)
    np.testing.assert_array_equal(result[:3], original[:3])
    np.testing.assert_array_equal(result[7:], np.broadcast_to(fixed, (3, 4, 4)))
    assert all(np.min(box_vertices() @ pose[2, :3] + pose[2, 3]) >= -1e-12 for pose in result)
    assert support_metrics(box_vertices(), fixed)["com_inside_contact_hull"]
    assert not support_metrics(box_vertices(), fixed, com_local=np.array([.2, 0, 0]))["com_inside_contact_hull"]


def test_finite_support_allows_noncontact_overhang_but_never_invents_support():
    vertices = np.r_[box_vertices(), [[.4, 0, .1]]]
    fixed = np.eye(4)
    fixed[2, 3] = .05
    valid = support_metrics(vertices, fixed, table_center_xy=np.zeros(2), table_radius=.08)
    assert valid["com_inside_contact_hull"]
    assert valid["supported_contact_band_vertex_count"] == 4
    missing = support_metrics(vertices, fixed, table_center_xy=np.array([.1, 0]), table_radius=.08)
    assert not missing["com_inside_contact_hull"]
    assert missing["contact_hull_xy_m"] == []
    with pytest.raises(ValueError, match="both"):
        support_metrics(vertices, fixed, table_radius=.08)


def test_common_gauge_preserves_image_coordinates():
    points = np.array([[[.1, .2, 2.0], [-.1, -.2, 2.1]]])
    camera = np.eye(4)
    delta = np.eye(4)
    delta[:3, :3] = Rotation.from_rotvec([.1, .05, -.03]).as_matrix()
    delta[:3, 3] = [.03, .04, .05]
    new = transform_points(points, delta[None])[0]
    np.testing.assert_allclose(project_scene(points[0], camera, np.array([100, 100, 50, 50])),
                               project_scene(new, delta @ camera, np.array([100, 100, 50, 50])), atol=1e-12)


def test_recover_root_and_centered_translation_formula():
    camera = np.tile(np.eye(4), (5, 1, 1))
    camera[:, :3, :3] = Rotation.from_rotvec(np.arange(15).reshape(5, 3) * .03).as_matrix()
    camera[:, :3, 3] = [.1, .2, .3]
    root = np.array([.09, -.02, -.008])
    ct = np.tile([.2, .3, .5], (5, 1))
    trans = transform_points(ct + root, camera) - root
    recovered, proof = recover_mano_rest_root(trans, camera, ct)
    np.testing.assert_allclose(recovered, root, atol=1e-12)
    assert proof["max_equation_error_m"] < 1e-12


def test_gauge_keeps_universe_inactive_sentinels_and_external_finger_edits():
    row = toy_row()
    old = copy.deepcopy(row)
    gauge = np.tile(np.eye(4), (4, 1, 1))
    gauge[:, 0, 3] = .1
    apply_scene_gauge(row, gauge, first_changed_frame=2,
                      mano_rest_roots={"left": [0, 0, 0], "right": [0, 0, 0]})
    np.testing.assert_array_equal(row["robot_left_finger_joints"], old["robot_left_finger_joints"])
    np.testing.assert_array_equal(np.asarray(row["robot_left_frames"])[:, 0], np.asarray(old["robot_left_frames"])[:, 0])
    np.testing.assert_array_equal(np.asarray(row["mano_left_object_contact_positions"])[:, 1], np.zeros((4, 3)))
    np.testing.assert_array_equal(row["robot_left_wrist_position"][:2], old["robot_left_wrist_position"][:2])
    np.testing.assert_allclose(np.asarray(row["robot_left_wrist_position"])[2:, 0], .22)


def test_composable_solution_preserves_external_ik_and_rejects_double_application():
    cells = toy_row()
    original = copy.deepcopy(cells)
    poses = np.tile(np.eye(4), (4, 1, 1, 1))
    repaired = poses.copy()
    repaired[2:, 0, 2, 3] = .1
    gauge = np.tile(np.eye(4), (4, 1, 1))
    gauge[2:, 0, 3] = .02
    solution = {"manifest": {"config": {"last_unchanged_frame": 1},
                              "mano_rest_roots": {"left": [0, 0, 0], "right": [0, 0, 0]}},
                "original_object_to_world": poses, "repaired_object_to_world": repaired,
                "scene_gauge_delta_world": gauge,
                "repaired_left_tips_distance": np.zeros((4, 5)),
                "repaired_right_tips_distance": np.zeros((4, 5))}
    result = apply_support_repair(cells, solution)
    assert cells == original  # Caller-owned cells remain untouched.
    np.testing.assert_array_equal(result["robot_left_finger_joints"], cells["robot_left_finger_joints"])
    np.testing.assert_allclose(np.asarray(result["robot_left_wrist_position"])[2:, 0], .14)
    with pytest.raises(ValueError, match="double application"):
        apply_support_repair(result, solution)
