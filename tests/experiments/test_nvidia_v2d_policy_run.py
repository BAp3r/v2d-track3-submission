import json
import sys
from pathlib import Path

import pytest

from experiments.nvidia_v2d.policy_budget import TrainingBudget, scaled_curriculum
from experiments.nvidia_v2d.policy_run import RunSpec, run_job


def test_curriculum_scales_every_stage_and_retains_unassisted_tail():
    original = [
        19660800,
        34406400,
        49152000,
        63897600,
        78643200,
        93388800,
        108134400,
        122880000,
        137625600,
        200007680,
        250003456,
    ]
    stages = [1, 0.75, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0, 0, 0]
    result = scaled_curriculum(original, stages, total_steps=100352, quantum=256)
    assert result[-1] == 100352
    assert all(a < b for a, b in zip([0, *result[:-1]], result, strict=True))
    assert all(x % 256 == 0 for x in result)
    assert result[7] < 0.6 * result[-1]
    with pytest.raises(ValueError):
        scaled_curriculum(original, stages, total_steps=512, quantum=256)
    with pytest.raises(ValueError):
        scaled_curriculum(original, [1] * 11, total_steps=100352, quantum=256)


def test_budget_rejects_assisted_or_unbounded_comparison():
    budget = TrainingBudget(total_steps=100352, world_count=256)
    assert budget.seeds == (42, 43, 44)
    assert budget.motion_speed == 1
    for changes in ({"wall_limit_s": 7201}, {"motion_speed": 0.5}, {"total_steps": 100353}):
        with pytest.raises(ValueError):
            TrainingBudget.model_validate({**budget.model_dump(), **changes})


def spec(tmp_path: Path, code: str, **kwargs):
    source = tmp_path / "input.json"
    if not source.exists():
        source.write_text('{"training_ready": true}')
    return RunSpec(
        phase="validation",
        episode_id="public-44",
        backend="official",
        argv=[sys.executable, "-c", code],
        cwd=tmp_path,
        inputs=[source],
        **kwargs,
    )


def test_success_requires_expected_artifacts_and_unchanged_inputs(tmp_path):
    run = spec(
        tmp_path,
        "import os; from pathlib import Path; "
        "(Path(os.environ['FMC3_RUN_OUTPUT'])/'result.json').write_text('{}')",
        required_outputs=["result.json"],
    )
    # Results must exist in the new output directory, independently of process cwd.
    outcome = run_job(run, tmp_path / "job")
    assert outcome.status == "succeeded"
    assert outcome.outputs[0].sha256
    assert outcome.input_unchanged
    assert Path(outcome.output_manifest_path).exists()
    assert Path(outcome.log_path).exists()
    with pytest.raises(FileExistsError):
        run_job(run, tmp_path / "job")


def test_missing_outputs_or_nonzero_exit_are_failures(tmp_path):
    missing = run_job(spec(tmp_path, "pass", required_outputs=["missing.json"]), tmp_path / "a")
    assert missing.status == "failed" and "missing.json" in missing.failure_reason
    failed = run_job(spec(tmp_path, "raise SystemExit(7)"), tmp_path / "b")
    assert failed.status == "failed" and failed.exit_code == 7


def test_changed_input_is_reported_not_accepted(tmp_path):
    source = tmp_path / "input.json"
    run = spec(tmp_path, f"from pathlib import Path; Path({str(source)!r}).write_text('changed')")
    result = run_job(run, tmp_path / "job")
    assert result.status == "failed" and result.input_unchanged is False


def test_timeout_kills_only_own_process_and_keeps_receipt(tmp_path):
    run = spec(tmp_path, "import time; time.sleep(20)", wall_limit_s=0.15)
    result = run_job(run, tmp_path / "timeout")
    assert result.status == "timed_out"
    assert result.finished_at and result.failure_reason
    assert json.loads((tmp_path / "timeout/run.json").read_text())["status"] == "timed_out"


def test_paths_and_training_gate_fail_closed(tmp_path):
    run = spec(tmp_path, "pass")
    for output in ["../escape", "/tmp/absolute"]:
        with pytest.raises(ValueError):
            RunSpec.model_validate({**run.model_dump(), "required_outputs": [output]})
    (tmp_path / "alias").symlink_to(tmp_path / "input.json")
    with pytest.raises(ValueError):
        run_job(
            RunSpec.model_validate({**run.model_dump(), "inputs": [tmp_path / "alias"]}),
            tmp_path / "unsafe",
        )
    training = RunSpec.model_validate({**run.model_dump(), "phase": "training"})
    with pytest.raises(ValueError, match="gate"):
        run_job(training, tmp_path / "ungated")
