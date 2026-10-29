import hashlib
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from experiments.nvidia_v2d.policy_experiment_evaluation import evaluate_with_evidence


def fake_harness(monkeypatch, tmp_path, applied):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    package, evaluation, harness = (
        ModuleType(name)
        for name in ["flash_chord", "flash_chord.evaluation", "flash_chord.evaluation.harness"]
    )
    harness.__file__ = str(tmp_path / "harness.py")
    (tmp_path / "harness.py").write_text("fixture")
    (tmp_path / "reference.parquet").write_bytes(b"reference fixture")
    env = SimpleNamespace(
        voc_scale=SimpleNamespace(numpy=lambda: np.array([0.0])),
        object_control_scale=SimpleNamespace(numpy=lambda: np.array(applied)),
    )

    def rollout(env, jax, action, *args, **kwargs):
        for _ in range(2):
            action(None, None, None)

    harness.record_object_pose_rollout = rollout

    def evaluate(checkpoint, *, object_trajectories_output, eval_script, **kwargs):
        harness.record_object_pose_rollout(env, None, lambda *args: (None, None))
        metadata = {
            "flash_chord.checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "flash_chord.eval_code_sha256": "a" * 64,
            "flash_chord.eval_script_sha256": hashlib.sha256(eval_script.read_bytes()).hexdigest(),
        }
        pq.write_table(
            pa.table({"row": [0, 1]}).replace_schema_metadata(metadata), object_trajectories_output
        )
        return SimpleNamespace(
            step_count=2, parquet=str(tmp_path / "reference.parquet"), environment_steps=64,
            object_position_threshold_m=0.1, object_orientation_threshold_rad=0.5,
            reference_tracking=SimpleNamespace(success_rate=0.0, reference_end_fraction=1.0),
            non_finite_worlds=(),
        )

    harness.evaluate_checkpoint = evaluate
    harness.write_evaluation_report = lambda path, outcome: path.write_text("{}")
    harness.evaluation_scalars = lambda outcome: {"evaluation/chord_sr": 0.0}
    evaluation.harness = harness
    for module in [package, evaluation, harness]:
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return harness, rollout


def test_evidence_comes_from_each_actual_control_state_and_binds_files(monkeypatch, tmp_path):
    harness, original = fake_harness(monkeypatch, tmp_path, [0.0, 0.0])
    checkpoint = tmp_path / "policy.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    result = evaluate_with_evidence(checkpoint, 2, tmp_path)
    assert result["assistance"] == "none"
    assert result["action_step_count"] == 2
    assert result["applied_control_scale_max"] == 0
    assert result["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert (
        result["trajectory_sha256"]
        == hashlib.sha256(
            (tmp_path / "unassisted.object_trajectories.parquet").read_bytes()
        ).hexdigest()
    )
    assert json.loads((tmp_path / "evaluation-evidence.json").read_text()) == result
    assert harness.record_object_pose_rollout is original


def test_evaluation_curves_append_only_verified_cohorts_with_provenance(monkeypatch, tmp_path):
    fake_harness(monkeypatch, tmp_path, [0.0, 0.0])
    checkpoint = tmp_path / "policy.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    history = tmp_path / "evaluation-metrics.jsonl"
    evaluate_with_evidence(checkpoint, 2, tmp_path, metrics_history=history)
    evaluate_with_evidence(checkpoint, 2, tmp_path, metrics_history=history)
    records = [json.loads(line) for line in history.read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["context"]["series_id"] == records[1]["context"]["series_id"]
    assert records[0]["context"]["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert records[0]["metrics"]["evaluation/reference_tracking_success_rate"] == 0.0
    assert records[0]["environment_steps"] == 64
    assert history.with_suffix(".csv").exists()
    assert (tmp_path / "evaluation-metrics.evaluation.svg").exists()


def test_curve_series_changes_for_reference_content_and_external_scene(monkeypatch, tmp_path):
    fake_harness(monkeypatch, tmp_path, [0.0, 0.0])
    checkpoint = tmp_path / "policy.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    history = tmp_path / "evaluation-metrics.jsonl"
    evaluate_with_evidence(checkpoint, 2, tmp_path, metrics_history=history,
                           scene_context={"support_sha256": "b" * 64})
    (tmp_path / "reference.parquet").write_bytes(b"corrected reference")
    evaluate_with_evidence(checkpoint, 2, tmp_path, metrics_history=history,
                           scene_context={"support_sha256": "b" * 64})
    evaluate_with_evidence(checkpoint, 2, tmp_path, metrics_history=history,
                           scene_context={"support_sha256": "c" * 64})
    records = [json.loads(line) for line in history.read_text().splitlines()]
    assert len({row["context"]["series_id"] for row in records}) == 3
    assert records[1]["context"]["reference_parquet_sha256"] == hashlib.sha256(b"corrected reference").hexdigest()
    assert records[2]["context"]["scene_context"] == {"support_sha256": "c" * 64}


@pytest.mark.parametrize("applied", [[0.0, 0.1], [0.0, float("nan")]])
def test_nonzero_or_nonfinite_applied_assistance_cannot_be_labelled_none(
    monkeypatch, tmp_path, applied
):
    harness, original = fake_harness(monkeypatch, tmp_path, applied)
    checkpoint = tmp_path / "policy.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    with pytest.raises(ValueError, match="辅助或非有限"):
        evaluate_with_evidence(checkpoint, 2, tmp_path, metrics_history=tmp_path / "eval.jsonl")
    assert not (tmp_path / "evaluation-evidence.json").exists()
    assert not (tmp_path / "eval.jsonl").exists()
    assert harness.record_object_pose_rollout is original
