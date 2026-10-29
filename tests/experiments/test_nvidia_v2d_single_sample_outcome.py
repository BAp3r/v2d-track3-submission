"""Run bookkeeping must never confuse successful execution with a successful task."""

import json
from pathlib import Path

import pytest

from experiments.nvidia_v2d import policy_single_sample as runner


@pytest.mark.parametrize(
    ("rate", "tamper", "expected_task", "expected_count", "expected_status"),
    [(0.0, False, False, 0, "observed_failure"),
     (0.5, False, True, 2, "observed_success"),
     (0.5, True, None, 0, "unverified")],
)
def test_pipeline_and_verified_task_outcome_are_distinct(
    tmp_path, monkeypatch, rate, tamper, expected_task, expected_count, expected_status,
):
    support, reference = tmp_path / "support.usda", tmp_path / "reference.parquet"
    support.write_bytes(b"support")
    reference.write_bytes(b"reference")
    spec = runner.SingleSampleSpec(support_usda=support, support_sha256=runner.sha256(support),
                                  parquet=reference, evaluation_worlds=4)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(spec.model_dump_json())
    output = tmp_path / "run"
    monkeypatch.setattr(runner, "require_experiment_gate", lambda spec: {"verified_files": {}})
    monkeypatch.setattr(runner, "require_update_audit", lambda *args: {"verified": True})

    class Process:
        returncode = 0

        def __init__(self, argv, **kwargs):
            if "train" in argv:
                train = output / "train"
                train.mkdir()
                (train / f"policy_{spec.budget.total_steps}.safetensors").write_bytes(b"policy")
                (train / f"state_{spec.budget.total_steps}.safetensors").write_bytes(b"state")
            elif "evaluate" in argv:
                target = Path(argv[argv.index("--output") + 1])
                target.mkdir(parents=True)
                checkpoint = Path(argv[argv.index("--checkpoint") + 1])
                trajectory = target / "unassisted.object_trajectories.parquet"
                trajectory.write_bytes(b"actual rollout fixture")
                common = {"checkpoint_sha256": runner.sha256(checkpoint), "world_count": 4,
                          "step_count": 2, "reset_mode": "explicit", "start_frame": 0,
                          "motion_start_frame": 0, "motion_end_frame": -1}
                report = {**common, "non_finite_worlds": [], "metrics": {"chord_sr": rate},
                          "reference_tracking": {"success_rate": rate}}
                evidence = {**common, "trajectory_sha256": runner.sha256(trajectory),
                            "reference_parquet_sha256": runner.sha256(reference),
                            "scene_context": {"support_sha256": spec.support_sha256},
                            "assistance": "none", "policy_deterministic": True,
                            "action_step_count": 2, "observed_policy_action_count": 2,
                            "voc_scale": 0, "applied_object_control_scale": 0,
                            "voc_scale_max": 0, "applied_control_scale_max": 0,
                            "observed_control_scales": [
                                {"voc_scale_max_abs": 0, "applied_object_control_scale_max_abs": 0}
                                for _ in range(2)]}
                (target / "unassisted.metrics.json").write_text(json.dumps(report))
                (target / "evaluation-evidence.json").write_text(json.dumps(evidence))
                if tamper:
                    trajectory.write_bytes(b"different rollout")

        def poll(self):
            return self.returncode

    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    result = runner.run(spec, spec_path, output)
    assert result["status"] == result["pipeline_status"] == "succeeded"
    assert result["task_success"] is expected_task
    assert result["task_success_evidence"]["successful_attempts"] == expected_count
    assert result["task_success_evidence"]["status"] == expected_status
    assert result["task_success_evidence"]["evaluated_attempts"] == (0 if tamper else 4)
    assert bool(result["task_success_evidence"]["rejected_checkpoints"]) is tamper
    assert result["policy_quality_claim_allowed"] is False
    assert result["geometry_quality_accepted"] is False


def test_prepared_manifest_changes_fail_before_any_stage(tmp_path, monkeypatch):
    support = tmp_path / "support.usda"
    support.write_bytes(b"support")
    spec = runner.SingleSampleSpec(support_usda=support, support_sha256=runner.sha256(support))
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(spec.model_dump_json())
    output = tmp_path / "run"
    output.mkdir()
    frozen = {"source_sha256": {name: runner.sha256(Path(runner.__file__).parent / name)
                                for name in runner.SOURCE_FILES},
              "spec_sha256": "0" * 64, "support_sha256": spec.support_sha256}
    (output / "launch-manifest.json").write_text(json.dumps(frozen))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch"))
    result = runner.run(spec, spec_path, output)
    assert result["status"] == result["pipeline_status"] == "failed"
    assert "changed before execution" in result["failure_reason"]
    assert result["stages"] == []
    assert result["task_success"] is None
