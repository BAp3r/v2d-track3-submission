import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from experiments.nvidia_v2d.contact_aware_reference import (
    SharpaCPUModel,
    apply_delta,
    project_motion,
    release_decay,
    select_temporal_roles,
    smooth_corrections,
    smooth_role_wrist_trajectory,
    trusted_thumb_evidence,
)


def test_explicit_hinge_fk_and_site_include_root_and_joint_rotation(tmp_path):
    xml = tmp_path / "model.xml"
    xml.write_text("""<mujoco><compiler angle="radian"/><worldbody>
    <body name="palm"><body name="thumb" pos="1 0 0">
    <joint name="thumb_hinge" axis="0 0 1" range="-2 2"/>
    <site name="tip" pos="1 0 0"/></body></body></worldbody></mujoco>""")
    model = SharpaCPUModel(
        xml,
        ["universe", "root_joint", "palm", "thumb_hinge", "thumb", "tip"],
        ["thumb_hinge"],
    )
    frames = model.frames([np.pi / 2], [2, 3, 0], [1, 0, 0, 0])
    np.testing.assert_allclose(frames[5, :3], [3, 4, 0], atol=1e-12)
    np.testing.assert_allclose(frames[3], frames[4])
    np.testing.assert_allclose(frames[0], [0, 0, 0, 1, 0, 0, 0])


def test_unsupported_mjcf_units_and_joint_names_rejected(tmp_path):
    xml = tmp_path / "model.xml"
    xml.write_text(
        '<mujoco><compiler angle="degree"/><worldbody><body name="palm"/></worldbody></mujoco>'
    )
    with pytest.raises(ValueError, match="Radian"):
        SharpaCPUModel(xml, [], [])


def evidence_fixture():
    source = {
        "mano_link_names": ["link_thumb3", "link_index3"],
        "mano_left_object_contact_part_ids": [[1, 1]] * 3,
    }
    record = {
        "side": "left",
        "frame": 1,
        "slot": 0,
        "link_name": "link_thumb3",
        "part_id": 1,
        "trusted": True,
        "confidence": 0.8,
        "object_local_position": [0, 0, 0],
        "object_local_normal": [1, 0, 0],
    }
    return source, record


def test_unknown_and_other_fingers_are_never_promoted_to_hard_contact():
    source, record = evidence_fixture()
    unknown = copy.deepcopy(record)
    unknown.update(frame=0, trusted=False)
    index = copy.deepcopy(record)
    index.update(frame=2, slot=1, link_name="link_index3")
    result = trusted_thumb_evidence([unknown, record, index], 3, source)
    assert list(result) == [(1, "left")]


@pytest.mark.parametrize(
    "key,value",
    [
        ("slot", 1),
        ("part_id", 2),
        ("confidence", 0),
        ("object_local_normal", [0, 0, 0]),
        ("object_local_position", [np.nan, 0, 0]),
    ],
)
def test_evidence_mapping_or_invalid_geometry_is_rejected(key, value):
    source, record = evidence_fixture()
    record[key] = value
    with pytest.raises(ValueError):
        trusted_thumb_evidence([record], 3, source)


def test_apply_delta_only_changes_named_thumb_dofs_and_uses_three_rotation_values():
    q = np.zeros(22)
    delta = np.r_[[0.01, 0, 0], [0, 0, 0.2], np.ones(5) * 0.1]
    cq, cp, cr = apply_delta(q, np.zeros(3), [1, 0, 0, 0], delta, np.arange(5))
    np.testing.assert_array_equal(cq[5:], q[5:])
    np.testing.assert_allclose(cq[:5], 0.1)
    np.testing.assert_allclose(cp, [0.01, 0, 0])
    assert Rotation.from_quat(cr, scalar_first=True).magnitude() == pytest.approx(0.2)
    np.testing.assert_array_equal(q, np.zeros(22))


def test_unknown_correction_is_softly_interpolated_and_release_decays_to_original():
    raw = np.zeros((10, 11))
    raw[2, 0] = 0.01
    raw[5, 0] = 0.02
    active = np.zeros(10, bool)
    active[[2, 5]] = True
    source = np.zeros(10, bool)
    source[2:7] = True
    result = smooth_corrections(raw, active, source)
    assert result[3, 0] > 0.005
    assert 0 < result[8, 0] < result[7, 0] < result[6, 0]
    np.testing.assert_array_equal(result[9], 0)
    np.testing.assert_array_equal(result[0], 0)
    assert abs(result[2, 0] - 0.01) < 0.001


def test_no_source_contact_has_no_candidate_correction():
    np.testing.assert_array_equal(
        smooth_corrections(np.ones((8, 11)), np.zeros(8, bool), np.zeros(8, bool)), 0
    )


def test_release_decay_keeps_contact_solution_and_avoids_one_frame_reset():
    correction = np.zeros((20, 11))
    correction[10] = 0.3
    result = release_decay(correction, 10, 7)
    np.testing.assert_array_equal(result[:11], correction[:11])
    assert result[11, 0] > 0.29
    assert np.max(np.abs(np.diff(result[10:, 0]))) < 0.1
    np.testing.assert_array_equal(result[17:], 0)


def test_complete_trajectory_speed_projection_removes_source_jump_and_preserves_limits():
    values = np.zeros((40, 2))
    values[20:, 0] = 1.6
    values[:, 1] = 0.2
    projected, diagnostics = project_motion(values, [-0.2, 0], [1.7, 0.5], 0.3)
    assert np.abs(np.diff(projected, axis=0)).max() <= 0.300002
    assert projected[:, 0].min() >= -0.2
    np.testing.assert_allclose(projected[:, 1], 0.2, atol=1e-6)
    assert diagnostics["primal_residual"] < 1e-6


def test_vector_motion_projection_limits_norm_not_individual_axes():
    values = np.zeros((30, 3))
    values[15:] = [1, 1, 1]
    projected, _ = project_motion(values, -2, 2, 0.1, vector_step=True)
    assert np.linalg.norm(np.diff(projected, axis=0), axis=1).max() <= 0.100002


def test_wrist_smoothing_constrains_actual_so3_and_retains_fixed_q_contact_point():
    count = 30
    original = np.zeros((count, 3))
    rv = np.zeros((count, 3))
    rv[12, 2] = 0.8
    quat = Rotation.from_rotvec(rv).as_quat(scalar_first=True)
    active = np.zeros(count, bool)
    active[8:17] = True
    source_active = active.copy()
    source_active[17:19] = True
    local = np.tile([0.02, 0.0, 0.0], (count, 1))
    target = local.copy()
    p, wq, diagnostics = smooth_role_wrist_trajectory(
        original,
        original,
        quat,
        active,
        source_active,
        local,
        target,
        max_rotation_step=0.15,
    )
    r = Rotation.from_quat(wq, scalar_first=True)
    np.testing.assert_allclose((p + r.apply(local))[active], target[active], atol=1e-9)
    assert (r[:-1].inv() * r[1:]).magnitude().max() <= 0.150002
    assert diagnostics["actual_rotation_step_max_rad"] <= 0.150002
    np.testing.assert_array_equal(p[25:], original[25:])


def test_wrist_smoothing_rejects_invented_contact_or_rotation_branch_crossing():
    p = np.zeros((10, 3))
    active = np.zeros(10, bool)
    active[3] = True
    with pytest.raises(ValueError, match="masks"):
        smooth_role_wrist_trajectory(
            p, p, np.tile([1, 0, 0, 0], (10, 1)), active, np.zeros(10, bool), p, p
        )
    rv = np.zeros((10, 3))
    rv[5:8, 2] = [3.1, -3.1, -3.0]
    with pytest.raises(ValueError, match="branch"):
        smooth_role_wrist_trajectory(
            p,
            p,
            Rotation.from_rotvec(rv).as_quat(scalar_first=True),
            active,
            active,
            p,
            p,
        )


def test_wrist_projection_preserves_trusted_contacts_at_sequence_endpoints():
    p = np.zeros((10, 3))
    active = np.ones(10, bool)
    target = np.tile([0.01, 0.0, 0.0], (10, 1))
    result, _, _ = smooth_role_wrist_trajectory(
        p,
        p,
        np.tile([1, 0, 0, 0], (10, 1)),
        active,
        active,
        p,
        target,
    )
    np.testing.assert_allclose(result, target, atol=1e-9)


def test_temporal_role_cost_avoids_isolated_switch_without_erasing_trusted_frames():
    def role(name, cost):
        return {"body_name": name, "geometric_cost": cost, "feasible": True}

    candidates = {
        0: [role("PP", 0.0)],
        1: [role("PP", 0.1), role("DP", 0.0)],
        3: [role("PP", 0.0)],  # Frame 2 is unknown, not promoted to an active target.
        4: [role("DP", 0.0)],  # Sole feasible role is retained despite switching cost.
    }
    chosen, diagnostic = select_temporal_roles(candidates, switch_cost=0.25)
    assert set(chosen) == {0, 1, 3, 4}
    assert candidates[1][chosen[1]]["body_name"] == "PP"
    assert candidates[4][chosen[4]]["body_name"] == "DP"
    assert diagnostic["switch_count"] == 1
    assert diagnostic["total_cost"] == pytest.approx(0.35)
    candidates[4][0]["feasible"] = False
    with pytest.raises(ValueError, match="feasible"):
        select_temporal_roles(candidates)
