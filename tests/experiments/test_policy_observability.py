import argparse
import hashlib
import json
from types import SimpleNamespace

import pytest

from experiments.nvidia_v2d import policy_observability as module


def row(step=256, session="one", *, series=None, invalid=False):
    return {"environment_steps": step, "wall_time_unix": 1234567.125 + step,
            "session_id": session, "metrics": {"loss": None if invalid else 1 / step},
            "nonfinite_metric_names": ["loss"] if invalid else [],
            "nonfinite_metric_values": {"loss": "nan"} if invalid else {},
            "context": {"series_id": series} if series else {}}


def write_rows(path, rows):
    path.write_text("".join(json.dumps(value) + "\n" for value in rows))


class MemorySink:
    instances = []

    def __init__(self, settings):
        self.settings = settings
        self.records = []
        self.closed = False
        self.process = SimpleNamespace(is_alive=lambda: False, join=lambda timeout: None)
        self.instances.append(self)

    def log(self, value):
        if self.records and value["step"] <= self.records[-1]["step"]:
            raise ValueError("steps must strictly increase")
        self.records.append(value)

    def close(self):
        self.closed = True

    def status(self):
        return {"rows": len(self.records), "last_step": self.records[-1]["step"] if self.records else None,
                "sdk_initialized": True, "closed": self.closed}


def args(tmp_path, **changes):
    values = dict(run_dir=tmp_path, output_dir=None, status_file=None, ready_file=None,
                  stop_file=None, swanlog_root=None, tensorboard_root=None, mode="local",
                  project="sharpa-unassisted-repair", swanlab_url=None, tensorboard_url=None,
                  poll_interval=0.001, drain_seconds=0, once=True)
    values.update(changes)
    return argparse.Namespace(**values)


@pytest.fixture(autouse=True)
def isolated_sinks(monkeypatch):
    MemorySink.instances = []
    monkeypatch.setattr(module, "check_dependencies", lambda: {"swanlab": "test", "tensorboardX": "test"})


def test_sessions_and_eval_series_remain_separate_with_original_steps_and_walltime(tmp_path):
    train = tmp_path / "train-metrics.jsonl"
    evaluation = tmp_path / "evaluation-metrics.jsonl"
    write_rows(train, [row(512, "old"), row(256, "resumed"), row(512, "resumed")])
    write_rows(evaluation, [row(256, "eval1", series="sceneA"), row(512, "eval2", series="sceneA")])
    before = hashlib.sha256(train.read_bytes()).hexdigest()
    result = module.observe(args(tmp_path), sink_factory=MemorySink)
    assert result["status"] == "completed"
    assert len(result["runs"]) == 3
    assert [len(s.records) for s in MemorySink.instances] == [1, 2, 2]
    assert MemorySink.instances[1].records[0]["step"] == 256
    assert MemorySink.instances[1].records[0]["wall_time_unix"] == 1234567.125 + 256
    assert all(s.closed for s in MemorySink.instances)
    assert hashlib.sha256(train.read_bytes()).hexdigest() == before


def test_partial_line_waits_and_is_rejected_if_producer_finishes(tmp_path):
    source = tmp_path / "train-metrics.jsonl"
    content = json.dumps(row())
    source.write_text(content[:20])
    tail = module.JsonlTail(source, "train")
    assert list(tail.read()) == []
    with pytest.raises(ValueError, match="incomplete final"):
        list(tail.read(final=True))
    with source.open("a") as stream:
        stream.write(content[20:] + "\n")
    assert list(tail.read())[0]["step"] == 256


def test_truncated_source_is_not_silently_restarted(tmp_path):
    source = tmp_path / "train-metrics.jsonl"
    write_rows(source, [row()])
    tail = module.JsonlTail(source, "train")
    assert len(list(tail.read())) == 1
    source.write_text("")
    with pytest.raises(ValueError, match="truncated"):
        list(tail.read())


def test_nonfinite_is_audited_and_never_replaced_with_zero(tmp_path):
    write_rows(tmp_path / "train-metrics.jsonl", [row(invalid=True)])
    result = module.observe(args(tmp_path), sink_factory=MemorySink)
    assert result["status"] == "completed_with_nonfinite"
    assert result["nonfinite_count"] == 1
    record = MemorySink.instances[0].records[0]
    assert record["metrics"] == {} and record["nonfinite"] == {"loss": "nan"}
    saved = json.loads((tmp_path / "observability/nonfinite-metrics.jsonl").read_text())
    assert saved["nonfinite"] == {"loss": "nan"}


def test_duplicate_step_in_same_session_is_failure(tmp_path):
    write_rows(tmp_path / "train-metrics.jsonl", [row(), row()])
    with pytest.raises(ValueError, match="strictly increase"):
        module.observe(args(tmp_path), sink_factory=MemorySink)
    assert json.loads((tmp_path / "observability/status.json").read_text())["status"] == "failed"


def test_unexplained_null_and_bad_walltime_are_errors():
    bad = row()
    bad["metrics"]["loss"] = None
    with pytest.raises(ValueError, match="declared nonfinite"):
        module.validate_row(bad, "train")
    bad = row()
    bad["wall_time_unix"] = float("nan")
    with pytest.raises(ValueError, match="finite wall time"):
        module.validate_row(bad, "train")


def test_tail_drains_new_metrics_after_execution_marker(tmp_path, monkeypatch):
    source = tmp_path / "train-metrics.jsonl"
    write_rows(source, [row()])
    (tmp_path / "execution.json").write_text('{"exit_code":0}')
    calls = []
    def sleep(_):
        if not calls:
            with source.open("a") as stream:
                stream.write(json.dumps(row(512)) + "\n")
        calls.append(1)
    monkeypatch.setattr(module.time, "sleep", sleep)
    result = module.observe(args(tmp_path, once=False, drain_seconds=0.001), sink_factory=MemorySink)
    assert result["sources"]["train"]["rows"] == 2
    assert result["stop_file_seen"] is True


def test_missing_all_logs_cannot_report_completed(tmp_path):
    with pytest.raises(ValueError, match="no metric records"):
        module.observe(args(tmp_path), sink_factory=MemorySink)
