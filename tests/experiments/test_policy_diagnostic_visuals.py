import copy
import json

import numpy as np
import pytest

from experiments.nvidia_v2d.policy_diagnostic_visuals import (
    ERROR_NAMES,
    PLAYBACK_METHOD,
    controller_label,
    load_cases,
    per_world_outcomes,
    playback_reference_times,
    reference_timeline,
    validate_timeline_record,
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


def captured_timing():
    data = {"reference_pose_w": np.zeros((496, 1, 7)),
            "reference_source_time_s": np.linspace(0, 12.4, 496)}
    manifest = {"fps": 20, "motion_speed_override": .5,
                "reference_timing": {"verified": True, "method": PLAYBACK_METHOD,
                    "source_path": "/input/data.parquet", "source_fps": 20, "source_frame_count": 249,
                    "control_fps": 20, "motion_speed": .5}}
    return data, manifest


def test_runtime_time_array_uses_real_endpoint_and_bound_source_metadata(tmp_path):
    data, manifest = captured_timing()
    times, provenance, hashes = reference_timeline(data, manifest, {}, tmp_path)
    assert times[-1] == 12.4
    assert times[407] == pytest.approx(10.195555555555556)
    assert times[-1] != 495 * .025
    assert provenance["origin"] == "captured_runtime_array" and provenance["verified"]
    assert hashes == {}


@pytest.mark.parametrize("invalid", ["nominal_array", "missing_provenance", "nan", "wrong_count"])
def test_claimed_runtime_times_reject_corruption_or_wrong_metadata(tmp_path, invalid):
    data, manifest = captured_timing()
    if invalid == "nominal_array":
        data["reference_source_time_s"] = np.arange(496) * .025
    elif invalid == "missing_provenance":
        del manifest["reference_timing"]["verified"]
    elif invalid == "nan":
        data["reference_source_time_s"][100] = np.nan
    else:
        # Parquet has one nested-array row; its row count is not source frame count.
        manifest["reference_timing"]["source_frame_count"] = 1
    with pytest.raises(ValueError):
        reference_timeline(data, manifest, {}, tmp_path)


def test_speed_one_uses_runtime_arange_clip_not_unconditional_linspace():
    np.testing.assert_array_equal(playback_reference_times(3, 3, 4, 1), [0, .25, .5, 2/3])
    np.testing.assert_array_equal(playback_reference_times(249, 20, 20, 1), np.arange(249)/20)
    assert len(playback_reference_times(249, 20, 20, .5)) == 496


def test_old_capture_time_requires_source_position_match(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "reference.parquet"
    src = np.arange(5) / 2
    target = np.linspace(0, 2, 8)
    def positions(times, scale):
        return np.column_stack((times * scale, times * 0, times * 2))
    cells = {"fps": 2., "object_body_position": positions(src, 1)[:, None].tolist(),
             "robot_left_wrist_position": positions(src, 3).tolist(),
             "robot_right_wrist_position": positions(src, -3).tolist()}
    pq.write_table(pa.Table.from_pylist([cells]), source)
    assert pq.ParquetFile(source).metadata.num_rows == 1
    data = {"reference_pose_w": np.zeros((8, 1, 7)),
            "reference_left_wrist": positions(target, 3), "reference_right_wrist": positions(target, -3)}
    data["reference_pose_w"][..., :3] = positions(target, 1)[:, None]
    manifest = {"fps": 2., "motion_speed_override": .5}
    times, provenance, hashes = reference_timeline(data, manifest, {"parquet": str(source)}, tmp_path)
    np.testing.assert_array_equal(times, target)
    assert provenance["verified"] and provenance["source_frame_count"] == 5
    assert str(source) in hashes
    data["reference_left_wrist"][3, 0] += .01
    times, provenance, _ = reference_timeline(data, manifest, {"parquet": str(source)}, tmp_path)
    assert not provenance["verified"]
    assert "does not match" in provenance["reason"]
    np.testing.assert_array_equal(times, np.arange(8) * .25)


def test_missing_timing_evidence_is_explicitly_nominal(tmp_path):
    data = {"reference_pose_w": np.zeros((496, 1, 7))}
    times, provenance, _ = reference_timeline(data, {"fps": 20, "motion_speed_override": .5}, {}, tmp_path)
    assert times[-1] == 12.375
    assert not provenance["verified"]
    assert "not verified source time" in provenance["label"]


def test_plot_timeline_is_bound_to_capture_and_keeps_exact_times():
    record = {"timestamps_s": np.linspace(0, 12.4, 496).tolist(), "capture_npz_sha256": "capture-a",
              "provenance": {"verified": True, "method": PLAYBACK_METHOD}}
    times, provenance = validate_timeline_record(record, "capture-a", 496)
    assert times[-1] == 12.4 and provenance["verified"]
    with pytest.raises(ValueError, match="different capture"):
        validate_timeline_record(record, "capture-b", 496)


def write_position_capture(folder, distance, recorded_error):
    data, manifest, report = cohort()
    actual = np.zeros((5, 2, 1, 7), dtype=np.float32)
    reference = np.zeros((5, 1, 7), dtype=np.float32)
    actual[..., 6] = reference[..., 6] = 1
    actual[3, 1, 0, 0] = distance
    data.update(achieved_pose_w=actual, reference_pose_w=reference,
                actual_bodies=actual.copy(), auxiliary_scales=np.zeros((5, 2), dtype=np.float32))
    data["tracking_error"][3, 1, 2] = recorded_error
    manifest.update(body_labels=["white_pot/object"], fps=20, checkpoint_sha256="checkpoint")
    report["checkpoint_sha256"] = "checkpoint"
    np.savez(folder / "diagnostic.npz", **data)
    (folder / "diagnostic.manifest.json").write_text(json.dumps(manifest))
    (folder / "diagnostic.metrics.json").write_text(json.dumps(report))
    return data["tracking_error"].copy()


@pytest.mark.parametrize("distance", [259.0, 1378.0])
def test_large_failed_world_float32_roundoff_loads_without_changing_outcome(tmp_path, distance):
    # Two float32 rounding units reproduce cross-kernel reduction differences
    # that exceed the old default relative tolerance at these large distances.
    recorded = np.nextafter(np.nextafter(np.float32(distance), np.float32(np.inf)), np.float32(np.inf))
    assert abs(float(recorded) - distance) > 2e-6 + 1e-7 * float(recorded)
    original_errors = write_position_capture(tmp_path, distance, float(recorded))
    cases, _ = load_cases(tmp_path, diagnostic=tmp_path, world=1)
    np.testing.assert_array_equal(cases[0]["data"]["tracking_error"], original_errors)
    assert [row["success"] for row in cases[0]["outcomes"]] == [True, False]
    assert cases[0]["outcomes"][1]["first_failure_step"] == 3
    assert "FAILED" in cases[0]["title"]


@pytest.mark.parametrize("distance", [259.0, 1378.0])
def test_large_coordinate_capture_still_rejects_real_error_mismatch(tmp_path, distance):
    write_position_capture(tmp_path, distance, distance + 0.01)
    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        load_cases(tmp_path, diagnostic=tmp_path, world=1)


def test_recheck_tolerance_never_relaxes_recorded_success_threshold(tmp_path):
    # The pose recheck may accept this tiny discrepancy, but the authoritative
    # recorded value is strictly above the 0.1 m success threshold.
    recorded = float(np.nextafter(np.float32(0.1), np.float32(np.inf)))
    write_position_capture(tmp_path, 0.1, recorded)
    cases, _ = load_cases(tmp_path, diagnostic=tmp_path, world=1)
    assert cases[0]["data"]["tracking_error"][3, 1, 2] == recorded
    assert cases[0]["criteria"]["thresholds"][2] == 0.1
    assert cases[0]["outcomes"][1]["success"] is False
