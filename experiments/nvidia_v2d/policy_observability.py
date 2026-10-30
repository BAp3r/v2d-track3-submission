"""Read-only JSONL sidecar: complete SwanLab and TensorBoard metric histories.

Each training session and each evaluation series has an independent SDK process.
This avoids SwanLab's process-global run and preserves overlapping resume steps.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
import signal
import struct
import sys
import time
from pathlib import Path
from typing import Any


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def check_dependencies() -> dict[str, str]:
    import swanlab
    from tensorboardX import SummaryWriter

    if not callable(swanlab.init) or not callable(SummaryWriter):
        raise RuntimeError("SwanLab/TensorBoard writer interfaces unavailable")
    return {name: importlib.metadata.version(name) for name in ("swanlab", "tensorboardX", "tensorboard")}


def _digest_update(digests: dict, key: str, step: int, value: float, wall: float | None = None) -> None:
    if key not in digests:
        digests[key] = {"count": 0, "hash": hashlib.sha256()}
    item = digests[key]
    item["hash"].update(struct.pack("<qd", step, value) if wall is None else struct.pack("<qfd", step, value, wall))
    item["count"] += 1


def verify_persisted(info: dict, expected_swan: dict, expected_tb: dict) -> dict:
    """Read back both SDK stores; detect swallowed SDK errors or dropped steps.

    SwanLab 0.10's datastore reader is version-specific; an incompatible SDK
    fails explicitly instead of reducing this to a best-effort count.
    """
    from swanlab.proto.swanlab.record.v1.record_pb2 import Record
    from swanlab.sdk.internal.core_python.store import DataStoreReader
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    actual_swan, actual_tb = {}, {}
    reader = DataStoreReader()
    reader.open(Path(info["swanlab_directory"]) / f"run-{info['swanlab_run_id']}.swanlab")
    try:
        while (raw := reader.scan()) is not None:
            record = Record()
            record.ParseFromString(raw)
            if record.HasField("scalar") and record.scalar.key in expected_swan:
                scalar = record.scalar
                _digest_update(actual_swan, scalar.key, scalar.step, scalar.value.number)
    finally:
        reader.close()
    events = EventAccumulator(info["tensorboard_directory"], size_guidance={"scalars": 0})
    events.Reload()
    for key in expected_tb:
        for event in events.Scalars(key):
            _digest_update(actual_tb, key, event.step, event.value, event.wall_time)
    def signatures(values):
        return {key: {"count": value["count"], "sha256": value["hash"].hexdigest()}
                for key, value in values.items()}
    swan, tb = signatures(actual_swan), signatures(actual_tb)
    if swan != signatures(expected_swan) or tb != signatures(expected_tb):
        raise ValueError("SwanLab/TensorBoard persisted scalars differ from submitted source values or steps")
    return {"swanlab": swan, "tensorboard": tb, "all_scalars_verified": True}


def validate_row(row: dict, kind: str) -> dict:
    if not isinstance(row, dict) or not isinstance(row.get("metrics"), dict):
        raise ValueError("metric record must contain a metrics mapping")
    step, wall = row.get("environment_steps"), row.get("wall_time_unix")
    if type(step) is not int or step < 0 or type(wall) not in (int, float) or not math.isfinite(wall):
        raise ValueError("metric record requires finite wall time and nonnegative integer step")
    session = row.get("session_id")
    if not isinstance(session, str) or not session:
        raise ValueError("metric record requires a session_id")
    context = row.get("context", {})
    identity = context.get("series_id", session) if kind == "evaluation" else session
    if not isinstance(identity, str) or not identity:
        raise ValueError("metric series identity must be a nonempty string")
    declared = row.get("nonfinite_metric_names", [])
    if not isinstance(declared, list) or any(not isinstance(name, str) for name in declared):
        raise ValueError("invalid nonfinite metric names")
    finite, invalid = {}, {}
    originals = row.get("nonfinite_metric_values", {})
    for name, value in row["metrics"].items():
        if not isinstance(name, str) or not name or name.startswith("observability/"):
            raise ValueError("invalid or reserved metric name")
        if value is None and name in declared:
            invalid[name] = str(originals.get(name, "nonfinite (source null)"))
        elif type(value) in (int, float) and math.isfinite(value):
            if name in declared:
                raise ValueError("metric declared nonfinite contains a finite value")
            finite[name] = float(value)
        elif type(value) in (int, float):
            invalid[name] = str(value)
        else:
            raise ValueError(f"metric {name!r} is neither numeric nor declared nonfinite")
    if set(declared) - set(invalid):
        raise ValueError("nonfinite declaration does not match metrics")
    return {"kind": kind, "identity": identity, "session_id": session, "step": step,
            "wall_time_unix": float(wall), "metrics": finite, "nonfinite": invalid}


class JsonlTail:
    """Only consume newline-complete records; reject replaced or truncated logs."""

    def __init__(self, path: Path, kind: str):
        self.path, self.kind = path, kind
        self.offset = 0
        self.lines = 0
        self.inode = None

    def read(self, *, final: bool = False):
        if not self.path.exists():
            return
        stat = self.path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if self.inode not in (None, identity) or stat.st_size < self.offset:
            raise ValueError(f"metric source was replaced or truncated: {self.path}")
        self.inode = identity
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            while raw := stream.readline():
                if not raw.endswith(b"\n"):
                    if final:
                        raise ValueError(f"incomplete final JSONL record: {self.path}:{self.lines + 1}")
                    break
                try:
                    record = validate_row(json.loads(raw), self.kind)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"invalid metric record: {self.path}:{self.lines + 1}: {exc}") from exc
                self.offset = stream.tell()
                self.lines += 1
                record.update(source_path=str(self.path), source_line=self.lines)
                yield record


def _sink_worker(pipe, settings: dict) -> None:
    """One SDK singleton per process. Never access or print credential contents."""
    writer = run = None
    try:
        import swanlab
        from tensorboardX import SummaryWriter

        log = Path(settings["worker_log"]).open("a", buffering=1)
        sys.stdout = sys.stderr = log
        run = swanlab.init(
            project=settings["project"], experiment_name=settings["name"],
            mode=settings["mode"], public=False, logdir=settings["swanlog_root"],
            config={"source_directory": settings["run_dir"], "source_kind": settings["kind"],
                    "source_identity": settings["identity"],
                    "wall_time_semantics": "original time is observability/source_wall_time_unix; SDK timestamp is import time"},
            settings=swanlab.Settings(interactive=False),
        )
        if run is None:
            raise RuntimeError("SwanLab init returned no run")
        writer = SummaryWriter(logdir=settings["tensorboard_dir"], flush_secs=1)
        info = {"swanlab_url": run.url if settings["mode"] == "online" else None,
                "swanlab_run_id": run.id, "swanlab_directory": str(run.dir),
                "tensorboard_directory": settings["tensorboard_dir"]}
        pipe.send({"ok": True, "info": info})
        expected_swan, expected_tb = {}, {}
        while True:
            record = pipe.recv()
            if record is None:
                writer.flush()
                writer.close()
                writer = None
                run.finish()
                run = None
                verified = verify_persisted(info, expected_swan, expected_tb)
                pipe.send({"ok": True, "finished": True, "verification": verified})
                return
            values = {**record["metrics"],
                      "observability/source_wall_time_unix": record["wall_time_unix"],
                      "observability/nonfinite_metric_count": float(len(record["nonfinite"]))}
            run.log(values, step=record["step"])
            for name, value in values.items():
                writer.add_scalar(name, value, global_step=record["step"], walltime=record["wall_time_unix"])
                _digest_update(expected_swan, name, record["step"], value)
                _digest_update(expected_tb, name, record["step"], value, record["wall_time_unix"])
            if record["nonfinite"]:
                message = json.dumps(record["nonfinite"], ensure_ascii=False)
                writer.add_text("observability/nonfinite_values", message,
                                global_step=record["step"], walltime=record["wall_time_unix"])
            writer.flush()
            pipe.send({"ok": True})
    except BaseException as exc:  # noqa: BLE001 - propagate child failure, never silently drop metrics
        pipe.send({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        if writer is not None:
            writer.close()
        if run is not None:
            run.finish(state="crashed", error="metric sidecar failure; see worker log")
        pipe.close()


class Sink:
    def __init__(self, settings: dict):
        context = mp.get_context("spawn")
        self.pipe, child = context.Pipe()
        self.process = context.Process(target=_sink_worker, args=(child, settings))
        self.process.start()
        child.close()
        try:
            self.info = self.receive()["info"]
        except BaseException:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=5)
            self.pipe.close()
            raise
        self.rows = self.scalar_count = self.nonfinite_count = 0
        self.last_step = -1
        self.first_step = None

    def receive(self) -> dict:
        deadline = time.monotonic() + 120
        while not self.pipe.poll(0.2):
            if not self.process.is_alive() or time.monotonic() >= deadline:
                raise RuntimeError("metric SDK worker exited or did not acknowledge within 120 seconds")
        result = self.pipe.recv()
        if not result["ok"]:
            raise RuntimeError(f"metric SDK worker failed: {result['error_type']}: {result['error']}")
        return result

    def log(self, row: dict) -> None:
        if row["step"] <= self.last_step:
            raise ValueError("metric steps must strictly increase within a source session/series")
        self.pipe.send(row)
        self.receive()
        self.rows += 1
        self.scalar_count += len(row["metrics"])
        self.nonfinite_count += len(row["nonfinite"])
        self.first_step = row["step"] if self.first_step is None else self.first_step
        self.last_step = row["step"]

    def close(self) -> None:
        if self.process.is_alive():
            self.pipe.send(None)
            self.info["persistence_verification"] = self.receive()["verification"]
        self.process.join(timeout=10)
        if self.process.exitcode != 0:
            raise RuntimeError(f"metric SDK worker exit code {self.process.exitcode}")
        self.pipe.close()

    def status(self) -> dict:
        return {**self.info, "rows": self.rows, "scalar_count": self.scalar_count,
                "nonfinite_count": self.nonfinite_count, "first_step": self.first_step,
                "last_step": self.last_step, "sdk_initialized": True}


def observe(args, *, sink_factory=Sink) -> dict:
    run_dir = args.run_dir.resolve()
    output = (args.output_dir or run_dir / "observability").resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = args.status_file or output / "status.json"
    if status_path.exists():
        raise FileExistsError("observability status already exists; use a new output directory for a separate import")
    swanlog = (args.swanlog_root or output / "swanlog").resolve()
    tensorboard = (args.tensorboard_root or output / "tensorboard").resolve()
    swanlog.mkdir(parents=True, exist_ok=True)
    tensorboard.mkdir(parents=True, exist_ok=True)
    stop = args.stop_file or run_dir / "execution.json"
    state = {"schema": "nvidia_v2d.observability.v1", "status": "starting", "mode": args.mode,
             "project": args.project, "requested_visibility": "private", "run_directory": str(run_dir),
             "swanlog_root": str(swanlog), "tensorboard_root": str(tensorboard),
             "swanlab_dashboard_url": args.swanlab_url, "tensorboard_url": args.tensorboard_url,
             "sources": {}, "runs": {}, "nonfinite_count": 0,
             "swanlab_wall_time": "original timestamp scalar; event timestamp is import time",
             "tensorboard_wall_time": "original source timestamp"}
    sinks = {}
    tails = [JsonlTail(run_dir / f"{kind}-metrics.jsonl", kind) for kind in ("train", "evaluation")]
    # Names include the import directory so a deliberate re-import never overwrites prior TensorBoard events.
    import_id = hashlib.sha256(str(output).encode()).hexdigest()[:10]
    nonfinite_path = output / "nonfinite-metrics.jsonl"

    def save():
        state["runs"] = {key: value.status() for key, value in sinks.items()}
        state["sources"] = {tail.kind: {"path": str(tail.path), "consumed_bytes": tail.offset,
                                          "rows": tail.lines} for tail in tails}
        state["updated_at_unix"] = time.time()
        atomic_json(status_path, state)

    def drain(final=False):
        count = 0
        for tail in tails:
            for row in tail.read(final=final):
                key = row["kind"] + "-" + hashlib.sha256(row["identity"].encode()).hexdigest()[:16]
                if key not in sinks:
                    name = f"{run_dir.name}-{key}-{import_id}"
                    settings = {"project": args.project, "mode": args.mode, "name": name,
                                "run_dir": str(run_dir), "kind": row["kind"], "identity": row["identity"],
                                "swanlog_root": str(swanlog), "tensorboard_dir": str(tensorboard / name),
                                "worker_log": str(output / f"{key}.log")}
                    sinks[key] = sink_factory(settings)
                sinks[key].log(row)
                if row["nonfinite"]:
                    with nonfinite_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    state["nonfinite_count"] += len(row["nonfinite"])
                count += 1
                if count % 100 == 0:
                    save()
        return count

    try:
        state["versions"] = check_dependencies()
        state["status"] = "ready"
        save()
        if args.ready_file is not None:
            atomic_json(args.ready_file, {"status": "ready", "sdk_run_initialization": "on first metric",
                                          "status_file": str(status_path)})
        quiet_since = None
        while True:
            count = drain()
            state["status"] = "following"
            save()
            if args.once:
                drain(final=True)
                break
            if stop.exists():
                if count or quiet_since is None:
                    quiet_since = time.monotonic()
                if time.monotonic() - quiet_since >= args.drain_seconds:
                    drain(final=True)
                    break
            time.sleep(args.poll_interval)
        for sink in sinks.values():
            sink.close()
        if not sinks:
            raise ValueError("no metric records were found before the producer stopped")
        state["status"] = "completed_with_nonfinite" if state["nonfinite_count"] else "completed"
        state["stop_file_seen"] = stop.exists()
        save()
        return state
    except BaseException as exc:  # noqa: BLE001 - persist and propagate any observer failure
        state.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        save()
        for sink in sinks.values():
            if sink.process.is_alive():
                sink.process.terminate()
            sink.process.join(timeout=5)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "observe"))
    parser.add_argument("--mode", "--swanlab-mode", choices=("local", "online"), default="local")
    parser.add_argument("--project", default="sharpa-unassisted-repair")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--swanlog-root", type=Path)
    parser.add_argument("--tensorboard-root", type=Path)
    parser.add_argument("--swanlab-url")
    parser.add_argument("--tensorboard-url")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--drain-seconds", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.command == "check":
        print(json.dumps({"status": "dependencies_available", "versions": check_dependencies(),
                          "mode": args.mode, "authentication_checked": False}))
        return
    if args.run_dir is None or args.poll_interval <= 0 or args.drain_seconds < 0:
        parser.error("observe requires --run-dir, positive polling and nonnegative drain time")
    def interrupted(signum, frame):
        raise InterruptedError(f"metric observer received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    state = observe(args)
    print(json.dumps({"status": state["status"], "run_count": len(state["runs"]),
                      "nonfinite_count": state["nonfinite_count"]}))
    if state["nonfinite_count"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
