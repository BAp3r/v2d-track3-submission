from copy import deepcopy

import numpy as np
import pytest

from experiments.nvidia_v2d.initial_wrist_reference import (
    apply_initial_wrist_delta,
    project_world,
    ray_smooth_candidate,
)


def fixture():
    wrist = np.tile([0.1, 0.2, 1.0], (16, 1))
    wrist[1:11] *= 0.7
    wrist[11:] *= 0.7
    camera = np.tile(np.eye(4), (16, 1, 1))
    active = np.arange(16) >= 12
    return wrist, camera, active


def test_ray_constraint_and_fixed_contact_suffix_hold_while_spike_decreases():
    wrist, camera, active = fixture()
    saved = wrist.copy()
    candidate, delta = ray_smooth_candidate(wrist, camera, active, fidelity=0)
    assert np.linalg.norm(candidate[1] - candidate[0]) < np.linalg.norm(
        wrist[1] - wrist[0]
    )
    np.testing.assert_array_equal(wrist, saved)
    np.testing.assert_array_equal(candidate[0], wrist[0])
    np.testing.assert_array_equal(candidate[10:], wrist[10:])
    np.testing.assert_array_equal(delta[active], 0)
    np.testing.assert_allclose(
        project_world(candidate, camera, [100, 100, 0, 0]),
        project_world(wrist, camera, [100, 100, 0, 0]),
        atol=1e-12,
    )
    assert np.linalg.norm(delta, axis=1).max() <= 0.1 + 1e-10


def test_fixed_wrist_ray_does_not_claim_whole_hand_projection_invariance():
    wrist, camera, active = fixture()
    _, delta = ray_smooth_candidate(wrist, camera, active, fidelity=0)
    finger = wrist + np.array([0.15, 0, 0])
    assert (
        np.abs(
            project_world(finger + delta, camera, [100, 100, 0, 0])
            - project_world(finger, camera, [100, 100, 0, 0])
        ).max()
        > 0.1
    )


@pytest.mark.parametrize(
    "fault", ["contact", "nonrigid", "negative_depth", "nonfinite"]
)
def test_invalid_or_contact_inputs_fail_closed(fault):
    wrist, camera, active = fixture()
    if fault == "contact":
        active[3] = True
    elif fault == "nonrigid":
        camera[2, 0, 0] = 2
    elif fault == "negative_depth":
        wrist[0, 2] = -1
    else:
        wrist[2, 0] = np.nan
    with pytest.raises(ValueError):
        ray_smooth_candidate(wrist, camera, active)


def test_composition_moves_only_robot_translations_and_refuses_contact_edit():
    wrist, camera, active = fixture()
    _, delta = ray_smooth_candidate(wrist, camera, active)
    row = {
        "object_body_position": [[[1, 2, 3]]] * 16,
        "mano_left_joints": [[[4, 5, 6]]] * 16,
    }
    for side in ["left", "right"]:
        row[f"robot_{side}_wrist_position"] = wrist.tolist()
        row[f"robot_{side}_frames"] = np.tile(
            [0, 0, 1, 1, 0, 0, 0], (16, 2, 1)
        ).tolist()
        row[f"{side}_robot_frame_names"] = ["universe", f"{side}_hand_C_MC"]
        row[f"mano_{side}_object_contact_part_ids"] = (
            active[:, None].astype(int).tolist()
        )
    saved = deepcopy(row)
    result = apply_initial_wrist_delta(row, {"left": delta, "right": delta})
    assert row == saved
    assert result["object_body_position"] == row["object_body_position"]
    assert result["mano_left_joints"] == row["mano_left_joints"]
    np.testing.assert_array_equal(
        np.asarray(result["robot_left_frames"])[:, 0],
        np.asarray(row["robot_left_frames"])[:, 0],
    )
    np.testing.assert_array_equal(
        np.asarray(result["robot_left_frames"])[..., 3:],
        np.asarray(row["robot_left_frames"])[..., 3:],
    )
    delta[12, 0] = 0.1
    with pytest.raises(ValueError, match="contact"):
        apply_initial_wrist_delta(row, {"left": delta, "right": delta})
