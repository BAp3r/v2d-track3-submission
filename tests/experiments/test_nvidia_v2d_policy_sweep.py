import json

from experiments.nvidia_v2d.policy_budget import TrainingBudget
from experiments.nvidia_v2d.policy_experiment import ExperimentSpec
from experiments.nvidia_v2d.policy_experiment_sweep import verified_completed_run
from experiments.nvidia_v2d.policy_run import RunRecord, RunSpec, stamp


def completed_fixture(tmp_path):
    source, gate = tmp_path / "data.parquet", tmp_path / "gate.json"
    source.write_bytes(b"source")
    gate.write_text("{}")
    spec = ExperimentSpec(
        backend="official",
        parquet=source,
        gate=gate,
        budget=TrainingBudget(total_steps=1024, world_count=16, wall_limit_s=1200),
        seed=42,
        replay_capacity=4096,
        replay_minimum=128,
        replay_batch=128,
    )
    output = tmp_path / "output"
    output.mkdir()
    result = {
        "status": "succeeded",
        "adapter_source_unchanged": True,
        "spec": spec.model_dump(mode="json"),
        "evaluation_evidence": {"assistance": "none"},
    }
    (output / "experiment.json").write_text(json.dumps(result))
    (output / "unassisted.metrics.json").write_text("{}")
    (output / "unassisted.object_trajectories.parquet").write_bytes(b"trajectory")
    names = ["experiment.json", "unassisted.metrics.json", "unassisted.object_trajectories.parquet"]
    run = RunSpec(
        phase="training",
        episode_id="episode",
        backend="official",
        argv=["runner"],
        cwd=tmp_path,
        inputs=[source],
        gate=gate,
        retarget_parquet=source,
        required_outputs=names,
        wall_limit_s=1200,
    )
    record = RunRecord(
        status="succeeded",
        spec=run,
        input_manifest_path=str(output / "input-manifest.json"),
        output_manifest_path=str(output / "output-manifest.json"),
        log_path=str(output / "run.log"),
        started_at="2026-10-27T00:00:00Z",
        input_unchanged=True,
        outputs=[stamp(output / name) for name in names],
    )
    (output / "run.json").write_text(record.model_dump_json())
    (output / "input-manifest.json").write_text(
        json.dumps([stamp(path).model_dump() for path in [source, gate]])
    )
    return output, spec


def test_resume_skips_only_unchanged_successful_seed(tmp_path):
    output, spec = completed_fixture(tmp_path)
    assert verified_completed_run(output, spec)
    assert not verified_completed_run(output, spec.model_copy(update={"seed": 43}))
    (output / "unassisted.object_trajectories.parquet").write_bytes(b"changed")
    assert not verified_completed_run(output, spec)


def test_changed_input_and_failed_runs_are_not_accepted(tmp_path):
    output, spec = completed_fixture(tmp_path)
    spec.parquet.write_bytes(b"changed source")
    assert not verified_completed_run(output, spec)
    raw = json.loads((output / "run.json").read_text())
    raw["status"] = "failed"
    (output / "run.json").write_text(json.dumps(raw))
    assert not verified_completed_run(output, spec)
