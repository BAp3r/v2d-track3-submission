import copy

import numpy as np
import pytest

from experiments.nvidia_v2d.policy_diagnostic_visuals import (
    ERROR_NAMES,
    controller_label,
    per_world_outcomes,
)


def cohort():
    errors = np.zeros((5, 2, 4))
    errors[3, 1, 2] = .2
    data = {"tracking_error": errors, "completion_step": np.array([4, 4]),
            "completion_terminated": np.array([False, False]),
            "completion_truncated": np.array([True, True]),
            "completion_reference_progress": np.array([1., 1.])}
    manifest = {"error_names": list(ERROR_NAMES), "action_mode": "policy", "world_count": 2, "steps": 5,
                "reference_tracking_criteria": {"error_names": list(ERROR_NAMES),
                    "thresholds": [.15, None, .1, .5], "active_after_step": [-1] * 4,
                    "causes": [["wrist", [0, 1]], ["object", [2, 3]]]}}
    report = {"world_count": 2, "step_count": 5, "reset_mode": "explicit", "start_frame": 0,
              "motion_start_frame": 0, "motion_end_frame": -1,
              "object_position_threshold_m": .1, "object_orientation_threshold_rad": .5,
              "non_finite_worlds": [], "metrics": {"chord_sr": .5},
              "reference_tracking": {"success_rate": .5, "reference_end_fraction": 1.}}
    return data, manifest, report


def test_cohort_half_success_does_not_label_every_world_successful():
    data, manifest, report = cohort()
    outcomes, _ = per_world_outcomes(data, manifest, report)
    assert [row["success"] for row in outcomes] == [True, False]
    assert [row["status"] for row in outcomes] == ["SUCCESS", "FAILED"]
    assert outcomes[1]["first_failure_step"] == 3


def test_missing_completion_or_criteria_cannot_infer_success_from_cohort_sr():
    data, manifest, report = cohort()
    del data["completion_step"]
    outcomes, _ = per_world_outcomes(data, manifest, report)
    assert [row["success"] for row in outcomes] == [None, False]
    data, manifest, report = cohort()
    del manifest["reference_tracking_criteria"]
    outcomes, _ = per_world_outcomes(data, manifest, report)
    assert [row["success"] for row in outcomes] == [None, False]


def test_completed_reference_end_and_finite_world_are_required():
    data, manifest, report = cohort()
    data["completion_truncated"][0] = False
    report["reference_tracking"].update(success_rate=0, reference_end_fraction=.5)
    report["metrics"]["chord_sr"] = 0
    outcomes, _ = per_world_outcomes(data, manifest, report)
    assert not outcomes[0]["success"]
    data, manifest, report = cohort()
    report["non_finite_worlds"] = [0]
    report["reference_tracking"]["success_rate"] = report["metrics"]["chord_sr"] = 0
    outcomes, _ = per_world_outcomes(data, manifest, report)
    assert not outcomes[0]["success"] and outcomes[0]["invalid"]


def test_threshold_activation_and_disabled_terms_match_native_semantics():
    data, manifest, report = cohort()
    data["tracking_error"][0, 0, 2] = .2
    data["tracking_error"][:, 0, 1] = 3.
    manifest["reference_tracking_criteria"]["active_after_step"][2] = 0
    outcomes, _ = per_world_outcomes(data, manifest, report)
    assert [row["success"] for row in outcomes] == [True, False]


def test_wrong_aggregate_and_partial_sequence_are_rejected():
    data, manifest, report = cohort()
    wrong = copy.deepcopy(report)
    wrong["metrics"]["chord_sr"] = 1.
    with pytest.raises(ValueError, match="disagree"):
        per_world_outcomes(data, manifest, wrong)
    report["motion_start_frame"] = 70
    with pytest.raises(ValueError, match="complete sequence"):
        per_world_outcomes(data, manifest, report)


def test_scripted_controller_label_never_becomes_learned_policy():
    assert controller_label({"action_mode": "scripted-grasp"})[2] is False
    assert "Scripted diagnostic" in controller_label({"action_mode": "scripted-grasp"})[1]
    assert controller_label({"action_mode": "policy"})[2] is True
