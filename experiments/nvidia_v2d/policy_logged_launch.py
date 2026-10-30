"""Host launcher: require SwanLab and TensorBoard observation before training.

The observer runs in its own SDK environment and tails the existing fsynced
JSONL files. The reviewed Docker argv and numerical training code are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

HOST_SOURCE_FILES = ("policy_logged_launch.py", "policy_observability.py")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def host_source_hashes(module_root: Path) -> dict[str, str]:
    return {name: sha256(module_root / "experiments/nvidia_v2d" / name)
            for name in HOST_SOURCE_FILES}


def _write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _stop(process: subprocess.Popen | None, *, timeout: float = 20) -> list[str]:
    if process is None or process.poll() is not None:
        return []
    process.terminate()
    try:
        process.wait(timeout=timeout)
        return ["SIGTERM"]
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
        return ["SIGTERM", "SIGKILL"]


def _stop_training(process: subprocess.Popen | None, command: list[str]) -> list[str]:
    actions = _stop(process)
    # Docker's CLI proxies SIGTERM. If its grace period was exceeded, explicitly
    # stop this launch's named container before draining the observer's tail.
    if "SIGKILL" in actions and Path(command[0]).name == "docker" and "--name" in command:
        name = command[command.index("--name") + 1]
        stopped = subprocess.run([command[0], "stop", "--time", "20", name],
                                 capture_output=True, text=True, timeout=30, check=False)
        actions.append(f"docker_stop_exit={stopped.returncode}")
        if stopped.returncode and "No such container" not in stopped.stderr:
            raise RuntimeError("Could not confirm stopped training container")
    return actions


def _exit_code(training_code: int | None, failure: str | None, interrupted: int | None) -> int:
    if interrupted:
        return 128 + interrupted
    if not failure:
        return 0
    if training_code:
        return 128 - training_code if training_code < 0 else training_code
    return 1


def launch(manifest_path: Path, *, observer_python: Path, observer_module_root: Path,
           swanlab_mode: str = "local", ready_timeout_s: float = 30,
           drain_timeout_s: float = 45, poll_s: float = 0.1) -> dict:
    """Run once in a prepared output directory, preserving every failure receipt."""
    manifest_path = manifest_path.resolve()
    output = manifest_path.parent
    observer_module_root = observer_module_root.resolve()
    observer_python = observer_python.absolute()
    if swanlab_mode not in ("local", "online"):
        raise ValueError("SwanLab mode must be explicitly local or online")
    if ready_timeout_s <= 0 or drain_timeout_s <= 0 or poll_s <= 0:
        raise ValueError("Timeouts and polling interval must be positive")
    for name in ("execution.json", "runtime.log", "observability-ready.json", ".launcher.lock"):
        if (output / name).exists():
            raise FileExistsError(f"Refusing to overwrite a prior launch: {output / name}")
    lock = output / ".launcher.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()) + "\n")
    started = time.monotonic()
    record = {
        "schema": "nvidia_v2d.observed_launch.v1", "status": "starting",
        "started_at": datetime.now(UTC).isoformat(), "exit_code": None,
        "training_exit_code": None, "observer_exit_code": None, "preflight_exit_code": None,
        "swanlab_mode": swanlab_mode, "tensorboard_enabled": True, "swanlab_enabled": True,
        "observer_python": str(observer_python), "observer_module_root": str(observer_module_root),
        "manifest": str(manifest_path), "signal": None,
        "training_started": False, "observer_ready": False,
        "observability_status_file": str(output / "observability-status.json"),
    }
    interrupted = None

    def on_signal(number, _frame):
        nonlocal interrupted
        interrupted = number

    previous_handlers = {number: signal.signal(number, on_signal)
                         for number in (signal.SIGINT, signal.SIGTERM)}
    observer = training = preflight = None
    command = []
    streams = []
    failure = None
    try:
        manifest = json.loads(manifest_path.read_text())
        command = manifest["argv"]
        if not isinstance(command, list) or not command or any(not isinstance(x, str) or not x for x in command):
            raise ValueError("Launch manifest argv must be a nonempty list of strings")
        expected = manifest["host_source_sha256"]
        actual = host_source_hashes(observer_module_root)
        if expected != actual or actual["policy_logged_launch.py"] != sha256(Path(__file__)):
            raise ValueError("Prepared host launcher/observer source changed")
        contract = manifest.get("observability", {})
        if (contract.get("mode", "local") != swanlab_mode
                or Path(contract.get("python", observer_python)).absolute() != observer_python
                or Path(contract.get("module_root", observer_module_root)).resolve() != observer_module_root):
            raise ValueError("Observer arguments differ from prepared launch manifest")
        record.update(manifest_sha256=sha256(manifest_path), host_source_sha256=actual,
                      argv=command, status="checking_observability")
        _write(output / "launcher-status.json", record)
        environment = dict(os.environ, PYTHONPATH=str(observer_module_root), PYTHONUNBUFFERED="1",
                           CUDA_VISIBLE_DEVICES="")
        entry = [str(observer_python), "-m", "experiments.nvidia_v2d.policy_observability"]
        preflight_log = (output / "observability-preflight.log").open("xb")
        streams.append(preflight_log)
        preflight = subprocess.Popen([*entry, "check", "--mode", swanlab_mode],
                                     cwd=observer_module_root, env=environment,
                                     stdout=preflight_log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + ready_timeout_s
        while preflight.poll() is None:
            if interrupted:
                raise InterruptedError("Interrupted before observer dependency check completed")
            if time.monotonic() > deadline:
                raise TimeoutError("Observer dependency check timed out; training was not started")
            time.sleep(poll_s)
        record["preflight_exit_code"] = preflight.returncode
        if preflight.returncode:
            raise RuntimeError("SwanLab/TensorBoard dependency check failed; see observability-preflight.log")
        observer_log = (output / "observability.log").open("xb")
        streams.append(observer_log)
        observer_command = [*entry, "observe", "--run-dir", str(output),
                            "--status-file", str(output / "observability-status.json"),
                            "--ready-file", str(output / "observability-ready.json"),
                            "--stop-file", str(output / "execution.json"), "--mode", swanlab_mode]
        for key in ("swanlog_root", "tensorboard_root", "swanlab_url", "tensorboard_url"):
            if contract.get(key) is not None:
                observer_command.extend(["--" + key.replace("_", "-"), str(contract[key])])
        record["observer_argv"] = observer_command
        observer = subprocess.Popen(observer_command, cwd=observer_module_root, env=environment,
                                    stdout=observer_log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + ready_timeout_s
        while not (output / "observability-ready.json").exists():
            if interrupted:
                raise InterruptedError("Interrupted before observation was ready")
            if observer.poll() is not None:
                raise RuntimeError("Observer exited before becoming ready; training was not started")
            if time.monotonic() > deadline:
                raise TimeoutError("Observer did not become ready; training was not started")
            time.sleep(poll_s)
        ready = json.loads((output / "observability-ready.json").read_text())
        if ready.get("status") != "ready" or observer.poll() is not None:
            raise RuntimeError("Observer readiness is invalid or process already exited")
        if interrupted:
            raise InterruptedError("Interrupted before training started")
        record.update(observer_ready=True, observer_ready_receipt=ready, status="training")
        runtime_log = (output / "runtime.log").open("xb")
        streams.append(runtime_log)
        training = subprocess.Popen(command, stdout=runtime_log, stderr=subprocess.STDOUT, start_new_session=True)
        record["training_started"] = True
        _write(output / "launcher-status.json", record)
        while training.poll() is None:
            if interrupted:
                raise InterruptedError("Training interrupted; stopping process before observer drain")
            if observer.poll() is not None:
                raise RuntimeError(f"Observer exited during training (exit {observer.returncode}); stopping training")
            time.sleep(poll_s)
        if observer.poll() is not None:
            raise RuntimeError("Observer exited before the training stop/drain handshake")
        record["training_exit_code"] = training.returncode
        if training.returncode:
            failure = f"Training exited with code {training.returncode}"
    except Exception as exc:  # noqa: BLE001 - every launch failure must have an execution receipt
        failure = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    finally:
        try:
            _stop(preflight)
            record["training_stop_actions"] = _stop_training(training, command) if command else []
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            failure = f"{failure or ''}; stopping training failed: {exc}"
        record["training_exit_code"] = training.poll() if training is not None else None
        record["preflight_exit_code"] = preflight.poll() if preflight is not None else None
        record.update(signal=interrupted, failure_reason=failure,
                      status="interrupted" if interrupted else "failed" if failure else "draining_observability",
                      exit_code=_exit_code(record["training_exit_code"], failure, interrupted),
                      seconds=time.monotonic() - started)
        # This is the observer's stop handshake. All final fsynced JSONL remains
        # available; it drains the tail and closes both SDKs before we finish.
        _write(output / "execution.json", record)
        if observer is not None:
            try:
                observer.wait(timeout=drain_timeout_s)
            except subprocess.TimeoutExpired:
                _stop(observer, timeout=5)
                failure = f"{failure or ''}; observer did not finish draining in time"
            record["observer_exit_code"] = observer.poll()
            if observer.returncode:
                failure = f"{failure or ''}; observer failed with exit {observer.returncode}; see observability.log"
        if "host_source_sha256" in record:
            try:
                record["host_source_unchanged"] = record["host_source_sha256"] == host_source_hashes(observer_module_root)
                if not record["host_source_unchanged"]:
                    failure = f"{failure or ''}; host observation source changed during execution"
            except OSError as exc:
                failure = f"{failure or ''}; host source verification failed: {exc}"
                record["host_source_unchanged"] = False
        record.update(status="interrupted" if interrupted else "failed" if failure else "succeeded",
                      failure_reason=failure, seconds=time.monotonic() - started,
                      exit_code=_exit_code(record["training_exit_code"], failure, interrupted))
        for stream in streams:
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        _write(output / "execution.json", record)
        _write(output / "launcher-status.json", record)
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)
        lock.unlink(missing_ok=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--observer-python", type=Path, required=True)
    parser.add_argument("--observer-module-root", type=Path, required=True)
    parser.add_argument("--swanlab-mode", choices=("local", "online"), default="local")
    args = parser.parse_args()
    result = launch(args.manifest, observer_python=args.observer_python,
                    observer_module_root=args.observer_module_root, swanlab_mode=args.swanlab_mode)
    print(json.dumps({key: result[key] for key in ("status", "exit_code", "training_exit_code",
                                                   "observer_exit_code", "failure_reason")}))
    raise SystemExit(result["exit_code"])


if __name__ == "__main__":
    main()
