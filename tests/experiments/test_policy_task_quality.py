import numpy as np
import pytest

from experiments.nvidia_v2d.policy_task_quality import physical_quality


def fixture():
    labels = [f"{s}_sharpa_wave/{s}_thumb_{p}" for s in ("left", "right") for p in ("MC", "PP", "DP")]
    labels += ["white_pot/object", "support", "right_sharpa_wave/right_index_MCP_VL"]
    frames, worlds, bodies = 21, 2, len(labels)
    pose = np.zeros((frames, worlds, 1, 7)); pose[..., 2] = .06; pose[..., 6] = 1
    data = {"tracking_error": np.zeros((frames, worlds, 4)), "auxiliary_scales": np.zeros((frames, 2)),
            "contact_frame_offsets": np.arange(frames + 1) * 2,
            "contacts_body0_ids": np.tile([7, 7 + bodies], frames),
            "contacts_body1_ids": np.tile([6, 6 + bodies], frames),
            "contacts_contact_normal_w": np.tile([0., 0., 1.], (frames * worlds, 1)),
            "contacts_contact_pos_w": np.zeros((frames * worlds, 3)),
            "achieved_pose_w": pose, "actual_body_velocities": np.zeros((frames, worlds, bodies, 6))}
    manifest = {"body_labels": labels, "fps": 20.}
    support = {"center_xy_m": [0., 0.], "radius_m": .3, "top_m": 0.}
    return data, manifest, support, np.linspace(4, 8, frames), [{"success": True}, {"success": False}]


def test_settling_does_not_turn_failed_whole_task_into_success():
    rows, summary, _ = physical_quality(*fixture())
    assert summary["sampled_settled_count"] == 2
    assert summary["chord_release_and_settled_count"] == 1
    assert rows[1]["sampled_settled"] and not rows[1]["chord_success"]


@pytest.mark.parametrize("reason", ["hidden_hand_shape", "unsupported_sample", "outside_table", "angular_motion", "linear_motion", "drift"])
def test_visible_release_does_not_hide_contact_or_residual_motion(reason):
    args = fixture(); data = args[0]
    if reason == "hidden_hand_shape":
        data["contacts_body0_ids"][-2] = 8  # Not one of the seventeen reward slots.
    elif reason == "unsupported_sample":
        data["contacts_contact_normal_w"][-2] = [1., 0., 0.]
    elif reason == "outside_table":
        data["achieved_pose_w"][-1, 0, 0, 0] = .31
    elif reason == "angular_motion":
        data["actual_body_velocities"][-1, 0, 6, 3] = .26
    elif reason == "linear_motion":
        data["actual_body_velocities"][-1, 0, 6, 0] = .021
    else:
        data["achieved_pose_w"][-1, 0, 0, 0] = .006
    rows, _, _ = physical_quality(*args)
    assert rows[0]["chord_success"] and not rows[0]["sampled_settled"]


def test_auxiliary_or_cross_world_contacts_reject_evidence():
    args = fixture(); args[0]["auxiliary_scales"][0, 1] = .01
    with pytest.raises(ValueError, match="zero auxiliary"):
        physical_quality(*args)
    args = fixture(); args[0]["contacts_body0_ids"][0] = 16
    with pytest.raises(ValueError, match="crosses worlds"):
        physical_quality(*args)
