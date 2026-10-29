import json
from dataclasses import replace

import numpy as np
import pytest

from experiments.nvidia_v2d.policy_threeway_metrics import (
    ArmSpec,
    compare_arms,
    load_trajectory,
    measure,
    pose_errors,
    sha256,
    summarize_arm,
)


def fixture(tmp_path, name="official_ik", *, fps=20.0, reference_shift=0.0, negate=False):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    directory = tmp_path / name
    directory.mkdir()
    steps, worlds = 4, 2
    data = {
        "step": np.repeat(np.arange(steps), worlds),
        "environment_id": np.tile(np.arange(worlds), steps),
        "reference_frame": np.repeat(np.arange(steps), worlds),
        "time_s": np.repeat(np.arange(steps) / fps, worlds),
        "object_id": np.zeros(steps * worlds, dtype=int),
        "object_body_id": np.zeros(steps * worlds, dtype=int),
        "object_body_name": ["pot"] * (steps * worlds),
        "environment_valid": np.ones(steps * worlds, dtype=bool),
    }
    for prefix in ["reference", "achieved"]:
        for axis in "xyz":
            data[f"{prefix}_position_{axis}"] = np.zeros(steps * worlds)
            data[f"{prefix}_quaternion_{axis}"] = np.zeros(steps * worlds)
        data[f"{prefix}_quaternion_w"] = np.ones(steps * worlds)
    data["achieved_position_x"][::2] = [0, .05, .11, .03]
    data["achieved_position_z"][1::2] = [0, .02, .04, .03]
    data["reference_position_x"] += reference_shift
    if negate:
        data["reference_quaternion_w"] *= -1
    meta = {"flash_chord." + k: str(v) for k, v in {
        "schema": "flash_chord_object_trajectories_v1", "position_unit": "meter",
        "quaternion_order": "xyzw", "row_order": "step,environment_id,object_body_id",
        "start_frame": 0, "step_count": steps, "world_count": worlds,
        "object_body_count": 1, "pose_width": 7, "control_fps": fps,
    }.items()}
    trajectory = directory / "unassisted.object_trajectories.parquet"
    pq.write_table(pa.table(data).replace_schema_metadata(meta), trajectory)
    official = directory / "unassisted.metrics.json"
    official.write_text(json.dumps({"world_count": worlds, "step_count": steps,
                                    "environment_steps": 2097152, "non_finite_worlds": [],
                                    "object_position_threshold_m": .1, "object_orientation_threshold_rad": .5,
                                    "metrics": {"chord_sr": 0.0, "maniptrans_sr": .5, "add_auc": .7}}))
    spec = ArmSpec(name, trajectory, official, "baseline_geometry", {"object_collision_sha256": "a" * 64, "mass_kg": .3})
    return spec, data, meta


def rewrite(spec, data, meta):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table(data).replace_schema_metadata(meta), spec.trajectory)


def verified(spec):
    evidence = spec.trajectory.with_name("evaluation-evidence.json")
    evidence.write_text(json.dumps({"trajectory_sha256": sha256(spec.trajectory), "assistance": "none",
        "observed_control_scales": [{"voc_scale_max_abs": 0, "applied_object_control_scale_max_abs": 0}] * 4}))
    return replace(spec, evaluation_evidence=evidence)


def test_full_horizon_includes_failure_then_recovery_and_actual_lift(tmp_path):
    spec, _, _ = fixture(tmp_path)
    summary = summarize_arm(spec, tmp_path / "result")
    assert summary["object_only_success_count"] == 1
    assert summary["world_count"] == 2
    assert summary["worlds"][0]["first_failure_step"] == 2
    assert summary["worlds"][0]["first_failure_control_time_s"] == .1
    assert summary["worlds"][0]["first_failure_source_time_s"] is None
    assert summary["bodies"][1]["max_actual_lift_m"] == .04
    assert summary["bodies"][1]["peak_lift_step"] == 2
    assert summary["official_harness_metrics"]["metrics"]["chord_sr"] == 0
    assert summary["training_gate"] is False
    assert len((tmp_path / "result/frames.csv").read_text().splitlines()) == 9
    assert (tmp_path / "result/official-harness-metrics.json").read_bytes() == spec.official_metrics.read_bytes()


def test_quaternion_sign_and_known_shortest_angle():
    reference = np.array([[0, 0, 0, 0, 0, 0, 1.]])
    achieved = reference.copy(); achieved[:, 3:] *= -1
    _, angle, valid = pose_errors(achieved, reference)
    assert angle[0] == 0 and valid[0]
    achieved[0, 3:] = [0, 0, np.sin(.3), np.cos(.3)]
    _, angle, _ = pose_errors(achieved, reference)
    assert angle[0] == pytest.approx(.6)


def test_nan_world_retained_with_failure_time(tmp_path):
    spec, data, meta = fixture(tmp_path)
    data["achieved_position_x"][5] = np.nan
    rewrite(spec, data, meta)
    result = measure(load_trajectory(spec.trajectory))
    assert result["summary"]["world_count"] == 2
    assert result["summary"]["object_only_success_count"] == 0
    assert result["worlds"][1]["first_invalid_pose_step"] == 2
    assert result["worlds"][1]["mean_position_error_m"] is None


def test_zero_filled_official_nonfinite_world_is_not_success(tmp_path):
    spec, data, meta = fixture(tmp_path)
    for axis in "xyz":
        data[f"achieved_position_{axis}"][1::2] = 0
    rewrite(spec, data, meta)
    official = json.loads(spec.official_metrics.read_text()); official["non_finite_worlds"] = [1]
    spec.official_metrics.write_text(json.dumps(official))
    summary = summarize_arm(spec, tmp_path / "result")
    assert summary["object_only_success_count"] == 0 and summary["world_count"] == 2
    world = summary["worlds"][1]
    assert world["environment_flagged_invalid"]
    assert world["first_failure_step"] is None and not world["failure_timing_known"]
    assert world["max_position_error_m"] is None and world["max_actual_lift_m"] is None
    traces = np.load(tmp_path / "result/traces.npz")
    assert np.isnan(traces["position_error_m"][:, 1]).all()
    assert (traces["achieved_pose_xyzw"][:, 1, :, :3] == 0).all()  # raw evidence preserved


@pytest.mark.parametrize("change", ["duplicate", "missing", "wrong_reference", "bad_reference_quat"])
def test_incomplete_or_invalid_reference_rows_rejected(tmp_path, change):
    spec, data, meta = fixture(tmp_path)
    if change == "duplicate": data["step"][-1] = 2
    if change == "missing": data = {k: v[:-1] for k, v in data.items()}
    if change == "wrong_reference": data["reference_position_x"][-1] = .05
    if change == "bad_reference_quat": data["reference_quaternion_w"][:] = 0
    rewrite(spec, data, meta)
    with pytest.raises(ValueError): load_trajectory(spec.trajectory)


def test_threeway_accepts_same_physics_and_quaternion_sign(tmp_path):
    specs = [verified(fixture(tmp_path, n, negate=n == "multi_tip")[0]) for n in ["official_ik", "dexpilot_chain", "multi_tip"]]
    result = compare_arms(specs, tmp_path / "comparison")
    assert result["same_reference_and_time_axes_verified"]
    assert result["environment_steps"] == 2097152
    assert all(r["world_count"] == 2 for r in result["arms"])


@pytest.mark.parametrize("change", ["factor", "identity", "reference", "time", "budget", "source_time"])
def test_unfair_comparison_rejected_before_output(tmp_path, change):
    names = ["official_ik", "dexpilot_chain", "multi_tip"]
    specs = [verified(fixture(tmp_path, n, fps=10 if change == "time" and n == "multi_tip" else 20,
                     reference_shift=.01 if change == "reference" and n == "multi_tip" else 0)[0]) for n in names]
    if change == "factor": specs[2] = replace(specs[2], geometry_factor="another_geometry")
    if change == "identity": specs[2] = replace(specs[2], geometry_identity={"mass_kg": .6})
    if change == "budget":
        d = json.loads(specs[2].official_metrics.read_text()); d["environment_steps"] = 1048576
        specs[2].official_metrics.write_text(json.dumps(d))
    if change == "source_time":
        p = tmp_path / "source.npy"; np.save(p, np.arange(4) / 40)
        specs[2] = replace(specs[2], source_times=p)
    with pytest.raises(ValueError): compare_arms(specs, tmp_path / "comparison")
    assert not (tmp_path / "comparison").exists()


def test_formal_comparison_requires_evidence_but_single_historical_may_be_unknown(tmp_path):
    specs = [verified(fixture(tmp_path, n)[0]) for n in ["official_ik", "dexpilot_chain", "multi_tip"]]
    specs[1] = replace(specs[1], evaluation_evidence=None)
    with pytest.raises(ValueError, match="zero-assistance evidence"):
        compare_arms(specs, tmp_path / "comparison")
    assert not (tmp_path / "comparison").exists()
    assert summarize_arm(specs[1], tmp_path / "historical")["assistance"] == "unknown"


def test_auxiliary_cohort_binding_and_world_ids(tmp_path):
    spec, _, _ = fixture(tmp_path)
    quality = tmp_path / "quality.json"
    quality.write_text(json.dumps({"trajectory_sha256": "wrong", "worlds": []}))
    with pytest.raises(ValueError, match="different cohort"):
        summarize_arm(replace(spec, auxiliary_quality=quality), tmp_path / "bad")
    quality.write_text(json.dumps({"trajectory_sha256": sha256(spec.trajectory),
                                   "worlds": [{"environment_id": 1, "settled": True, "right_dp_fraction": 0.0}]}))
    summary = summarize_arm(replace(spec, auxiliary_quality=quality), tmp_path / "good")
    assert "aux_settled" not in summary["worlds"][0]
    assert summary["worlds"][1]["aux_settled"] is True
    assert summary["auxiliary_world_count"] == 1
