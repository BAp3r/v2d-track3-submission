import csv
import json
import xml.etree.ElementTree as ET

import pytest

from experiments.nvidia_v2d.policy_metric_logging import JsonMetricLogger


def test_loss_logs_are_actual_finite_samples_with_step_and_nonfinite_provenance(tmp_path):
    path = tmp_path / "metrics.jsonl"
    logger = JsonMetricLogger(path)
    logger.log({"actor_loss": -0.5, "critic_loss": float("nan")}, 640)
    logger.log({"actor_loss": -0.7, "critic_loss": 1.2}, 1280)
    logger.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["metrics"] == {"actor_loss": -0.5, "critic_loss": None}
    assert rows[0]["nonfinite_metric_names"] == ["critic_loss"]
    assert rows[1]["environment_steps"] == 1280


def test_metrics_cannot_overwrite_or_restart_step_numbers(tmp_path):
    path = tmp_path / "metrics.jsonl"
    logger = JsonMetricLogger(path)
    logger.log({"loss": 1.0}, 100)
    with pytest.raises(ValueError):
        logger.log({"loss": 0.0}, 100)
    logger.close()
    with pytest.raises(FileExistsError):
        JsonMetricLogger(path)


def test_explicit_resume_keeps_repeated_steps_and_separate_sessions(tmp_path):
    path = tmp_path / "metrics.jsonl"
    logger = JsonMetricLogger(path)
    logger.log({"actor_loss": 1.0}, 100)
    logger.close()
    resumed = JsonMetricLogger(path, resume=True)
    resumed.log({"actor_loss": 0.4, "new_metric": float("inf")}, 100)
    resumed.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["environment_steps"] for row in rows] == [100, 100]
    assert rows[0]["session_id"] != rows[1]["session_id"]
    assert rows[1]["nonfinite_metric_values"] == {"new_metric": "inf"}
    with path.with_suffix(".csv").open() as stream:
        values = list(csv.DictReader(stream))
    assert len(values) == 3 and values[-1]["value"] == "inf"
    assert ET.parse(tmp_path / "metrics.loss.svg").getroot().tag.endswith("svg")


def test_logs_are_flushed_before_close_and_delegate_failure(tmp_path):
    class FailingDelegate:
        closed = False

        def log(self, metrics, step):
            raise RuntimeError("remote sink unavailable")

        def close(self):
            self.closed = True

    delegate = FailingDelegate()
    path = tmp_path / "metrics.jsonl"
    logger = JsonMetricLogger(path, delegate=delegate)
    with pytest.raises(RuntimeError, match="remote"):
        logger.log({"actor_loss": -1.0}, 64)
    assert json.loads(path.read_text())["environment_steps"] == 64
    assert "actor_loss" in path.with_suffix(".csv").read_text()
    logger.close()
    logger.close()
    assert delegate.closed


def test_curve_export_preserves_but_skips_partial_last_record(tmp_path):
    from experiments.nvidia_v2d.policy_metric_curves import export_curves

    path = tmp_path / "evaluation.jsonl"
    logger = JsonMetricLogger(path)
    logger.log({"evaluation/task_success_rate": 0.0}, 64)
    logger.close()
    with path.open("a") as stream:
        stream.write('{"environment_steps": 128, "metrics":')
    before = path.read_bytes()
    paths = export_curves(path)
    assert paths == [tmp_path / "evaluation.evaluation.svg"]
    assert path.read_bytes() == before
    with pytest.raises(ValueError):
        JsonMetricLogger(path, resume=True)


def test_large_history_curve_sampling_is_bounded_and_preserves_raw_endpoints(tmp_path):
    from experiments.nvidia_v2d.policy_metric_curves import read_records

    path = tmp_path / "metrics.jsonl"
    rows = [{"environment_steps": index, "metrics": {"loss": index / 100}} for index in range(1000)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    selected = read_records(path, max_plot_points=20)
    assert len(selected) <= 21
    assert selected[0] == rows[0] and selected[-1] == rows[-1]
    assert len(path.read_text().splitlines()) == 1000


def test_wrapper_records_every_collection_and_closes_after_training_failure(tmp_path, monkeypatch):
    import sys

    from experiments.nvidia_v2d.policy_metric_logging import main

    script = tmp_path / "train.py"
    script.write_text(
        'def run_training(*args, **kwargs):\n'
        '    assert kwargs["log_interval"] == 1\n'
        '    kwargs["logger"].log({"actor_loss": -0.8}, 64)\n'
        '    raise RuntimeError("training failed")\n'
        'def main():\n'
        '    run_training(logger=None, log_interval=10)\n'
    )
    path = tmp_path / "train-metrics.jsonl"
    monkeypatch.setattr(sys, "argv", ["driver", str(script), str(path)])
    with pytest.raises(RuntimeError, match="training failed"):
        main()
    row = json.loads(path.read_text())
    assert row["context"]["log_interval"] == 1
    assert row["metrics"]["actor_loss"] == -0.8
    assert (tmp_path / "train-metrics.loss.svg").exists()


def test_transition_diagnostics_are_live_preserved_and_cannot_replace_losses(tmp_path):
    state = {"contact/right/target_distance_m": .11}
    path = tmp_path / "metrics.jsonl"
    logger = JsonMetricLogger(path, metric_provider=lambda: state.copy())
    logger.log({"actor_loss": -1.}, 256)
    state["contact/right/target_distance_m"] = .06
    logger.log({"actor_loss": -2.}, 512)
    state["actor_loss"] = 99.
    with pytest.raises(ValueError, match="覆盖"):
        logger.log({"actor_loss": -3.}, 768)
    logger.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["metrics"]["contact/right/target_distance_m"] for r in rows] == [.11, .06]
    assert [r["metrics"]["actor_loss"] for r in rows] == [-1., -2.]
    assert (tmp_path / "metrics.contact.svg").exists()
