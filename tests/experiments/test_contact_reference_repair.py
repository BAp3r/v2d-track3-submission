from __future__ import annotations

import copy

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from experiments.nvidia_v2d.contact_reference_repair import (
    STATUS_CODES,
    ContactRepairPolicy,
    apply_object_local_patch,
    candidate_faces,
    repair_contact_reference,
    training_contact_candidate,
    validate_repair,
)


def thin_wall():
    # Non-watertight, consistently wound opposed surface sheets, 4 mm apart.
    def vertices(z):
        return [[-.01, -.01, z], [.01, -.01, z], [.01, .01, z], [-.01, .01, z]]
    return trimesh.Trimesh(vertices=np.array(vertices(0) + vertices(-.004)),
                          faces=[[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6]], process=False)


def source_cells():
    cells = {"fps": 20., "mano_link_names": ["link_thumb3"],
             "object_body_position": np.zeros((5, 1, 3)).tolist(),
             "object_body_wxyz": np.tile([1., 0, 0, 0], (5, 1, 1)).tolist(),
             "robot_joint_pos": [[1, 2, 3]] * 5}
    for side in ("left", "right"):
        active = side == "left"
        hand = np.tile([0., 0., .003], (5, 1, 1)) if active else np.zeros((5, 1, 3))
        point = np.zeros((5, 1, 3))
        normal = np.tile([0., 0., -1.], (5, 1, 1)) if active else np.zeros((5, 1, 3))
        if active:
            point[2, 0, 2] = -.004
            normal[2, 0, 2] = 1.
        cells.update({f"mano_{side}_link_contact_positions": hand.tolist(),
                      f"mano_{side}_object_contact_positions": point.tolist(),
                      f"mano_{side}_object_contact_normals": normal.tolist(),
                      f"mano_{side}_object_contact_part_ids": np.full((5, 1), int(active)).tolist()})
    return cells


def test_opposite_surface_jump_moves_point_to_supported_face_not_just_flips_normal():
    mesh, cells = thin_wall(), source_cells()
    original = copy.deepcopy(cells)
    repaired, evidence, patch = repair_contact_reference(cells, mesh)
    assert cells == original
    assert evidence["counts"] == {"consistent": 4, "repaired": 1}
    point = np.array(repaired["mano_left_object_contact_positions"])[2, 0]
    normal = np.array(repaired["mano_left_object_contact_normals"])[2, 0]
    np.testing.assert_allclose(point, [0, 0, 0], atol=1e-12)
    np.testing.assert_array_equal(normal, [0, 0, -1])
    assert patch["left_status"][2, 0] == STATUS_CODES["repaired"]
    assert patch["left_confidence"][2, 0] == pytest.approx(1)
    assert patch["left_original_world_positions"][2, 0, 2] == -.004
    assert repaired["robot_joint_pos"] == cells["robot_joint_pos"]
    assert validate_repair(cells, repaired, patch, mesh)["all_checks_passed"]


def test_physically_different_hand_side_is_not_smoothed_away():
    cells = source_cells()
    cells["mano_left_link_contact_positions"][2][0][2] = -.007
    repaired, evidence, patch = repair_contact_reference(cells, thin_wall())
    assert repaired == cells
    assert evidence["counts"] == {"consistent": 5}
    assert patch["left_status"][2, 0] == STATUS_CODES["consistent"]


@pytest.mark.parametrize("inactive_frame", [1, 3])
def test_temporal_evidence_never_bridges_an_inactive_contact(inactive_frame):
    cells = source_cells()
    cells["mano_left_object_contact_part_ids"][inactive_frame][0] = 0
    repaired, evidence, patch = repair_contact_reference(cells, thin_wall())
    assert repaired == cells
    assert patch["left_status"][2, 0] == STATUS_CODES["unknown"]
    assert not patch["left_trusted"][2, 0] and patch["left_confidence"][2, 0] == 0
    assert evidence["discrete_contact_ids_unchanged"]


def test_persistent_conflicting_side_remains_unknown_instead_of_global_flip():
    cells = source_cells()
    for frame in range(5):
        cells["mano_left_object_contact_positions"][frame][0][2] = -.004
        cells["mano_left_object_contact_normals"][frame][0][2] = 1
    repaired, evidence, patch = repair_contact_reference(cells, thin_wall())
    assert repaired == cells
    assert evidence["counts"] == {"unknown": 5}
    assert not patch["left_trusted"].any()


def test_near_tied_spatially_distinct_surfaces_are_ambiguous():
    vertices = [[-.006, -.01, 0], [-.002, -.01, 0], [-.002, .01, 0], [-.006, .01, 0],
                [.002, -.01, 0], [.006, -.01, 0], [.006, .01, 0], [.002, .01, 0]]
    mesh = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], process=False)
    best, ambiguous, *_ = candidate_faces(mesh, np.array([0, 0, .003]), np.array([0, 0, -.004]),
                                          np.array([0, 0, 1]), np.array([[0, 0, -1], [0, 0, -1]]), ContactRepairPolicy())
    assert best is not None and ambiguous


def test_candidate_requires_a_face_facing_the_hand():
    mesh = thin_wall()
    mesh.update_faces([False, False, True, True])
    best, _ambiguous, *_ = candidate_faces(mesh, np.array([0, 0, .003]), np.array([0, 0, -.004]),
                                          np.array([0, 0, 1]), np.array([[0, 0, -1], [0, 0, -1]]), ContactRepairPolicy())
    assert best is None


def test_object_pose_composition_transforms_active_world_fields_and_keeps_inactive_and_ids():
    cells = source_cells()
    cells["mano_left_object_contact_part_ids"][0][0] = 0
    _repaired, _evidence, patch = repair_contact_reference(cells, thin_wall())
    target = copy.deepcopy(cells)
    rotation = Rotation.from_euler("x", 90, degrees=True)
    translation = np.array([.4, -.2, .1])
    target["object_body_position"] = np.tile(translation, (5, 1, 1)).tolist()
    target["object_body_wxyz"] = np.tile(rotation.as_quat(scalar_first=True), (5, 1, 1)).tolist()
    merged = apply_object_local_patch(target, patch)
    active = patch["left_part_ids"] > 0
    np.testing.assert_allclose(np.array(merged["mano_left_object_contact_positions"])[active],
                               rotation.apply(patch["left_object_local_positions"][active]) + translation, atol=1e-12)
    np.testing.assert_allclose(np.array(merged["mano_left_object_contact_normals"])[active],
                               rotation.apply(patch["left_object_local_normals"][active]), atol=1e-12)
    assert merged["mano_left_object_contact_part_ids"] == cells["mano_left_object_contact_part_ids"]
    assert merged["mano_right_object_contact_positions"] == cells["mano_right_object_contact_positions"]
    assert merged["mano_left_object_contact_positions"][0] == cells["mano_left_object_contact_positions"][0]
    assert merged["mano_left_link_contact_positions"] == cells["mano_left_link_contact_positions"]
    with pytest.raises(ValueError, match="original object poses"):
        apply_object_local_patch(target, patch, only_changed=True)


def test_patch_cannot_change_contact_timing():
    cells = source_cells()
    _repaired, _evidence, patch = repair_contact_reference(cells, thin_wall())
    cells["mano_left_object_contact_part_ids"][4][0] = 0
    with pytest.raises(ValueError, match="timing"):
        apply_object_local_patch(cells, patch)


def test_independent_validator_rejects_normal_flip_on_original_wrong_face():
    cells, mesh = source_cells(), thin_wall()
    repaired, _evidence, patch = repair_contact_reference(cells, mesh)
    repaired["mano_left_object_contact_positions"][2][0][2] = -.004
    with pytest.raises(ValueError, match="mesh surface"):
        validate_repair(cells, repaired, patch, mesh)


def test_independent_validator_rejects_moving_point_off_surface():
    cells, mesh = source_cells(), thin_wall()
    repaired, _evidence, patch = repair_contact_reference(cells, mesh)
    repaired["mano_left_object_contact_positions"][2][0][2] = .001
    with pytest.raises(ValueError, match="mesh surface"):
        validate_repair(cells, repaired, patch, mesh)


def test_unknown_training_exclusion_preserves_source_activity_and_does_not_claim_absence():
    cells = source_cells()
    for frame in range(5):
        cells["mano_left_object_contact_positions"][frame][0][2] = -.004
        cells["mano_left_object_contact_normals"][frame][0][2] = 1
    repaired, _evidence, patch = repair_contact_reference(cells, thin_wall())
    training, receipt = training_contact_candidate(repaired, patch)
    assert np.array(training["mano_left_object_contact_part_ids"]).sum() == 0
    assert patch["left_source_active"].all()
    assert patch["left_source_object_contact_part_ids"].sum() == 5
    assert patch["left_ignored_unknown"].all()
    assert receipt["source_no_contact_spans"] == []
    assert receipt["any_source_contact_but_no_reliable_target_spans"] == [
        {"first_frame": 0, "last_frame": 4, "first_time_s": 0, "last_time_s": .2}]
    assert receipt["requires_fresh_replay"]
    assert "unintended_contact penalty must be zero" in receipt["required_training_contract"]
    for key in cells:
        if not key.endswith("object_contact_part_ids"):
            assert training[key] == cells[key]
    assert repaired == cells


def test_confirmed_repaired_contact_remains_supervised_despite_unknown_source_status():
    repaired, _evidence, patch = repair_contact_reference(source_cells(), thin_wall())
    assert patch["left_source_status"][2, 0] == STATUS_CODES["unknown"]
    assert patch["left_status"][2, 0] == STATUS_CODES["repaired"]
    training, receipt = training_contact_candidate(repaired, patch)
    assert training["mano_left_object_contact_part_ids"] == [[1]] * 5
    assert receipt["hands"]["left"]["ignored_unknown_slots"] == 0


def test_training_mask_rejects_conflicting_confidence_masks():
    repaired, _evidence, patch = repair_contact_reference(source_cells(), thin_wall())
    patch["left_trusted"][2, 0] = False
    with pytest.raises(ValueError, match="partition"):
        training_contact_candidate(repaired, patch)


def test_training_mask_must_follow_object_pose_composition():
    cells = source_cells()
    cells["mano_left_object_contact_part_ids"][1][0] = 0
    repaired, _evidence, patch = repair_contact_reference(cells, thin_wall())
    training, _receipt = training_contact_candidate(repaired, patch)
    assert training["mano_left_object_contact_part_ids"][2][0] == 0
    with pytest.raises(ValueError, match="timing"):
        apply_object_local_patch(training, patch)
