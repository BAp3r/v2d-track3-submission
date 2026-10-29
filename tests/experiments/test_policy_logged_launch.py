import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from experiments.nvidia_v2d import policy_logged_launch as launcher


def prepared(tmp_path, *, preflight_exit=0, fail_observer=False, training_code=None):
    root = tmp_path / "observer-source"
    code = root / "experiments/nvidia_v2d"
    code.mkdir(parents=True)
    (code / "policy_logged_launch.py").write_bytes(Path(launcher.__file__).read_bytes())
    (code / "policy_observability.py").write_text(f'''
import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("mode")
p.add_argument("--mode",dest="swanlab_mode")
for name in ("run-dir","status-file","ready-file","stop-file"):
 p.add_argument("--"+name)
a=p.parse_args()
if a.mode=="check": raise SystemExit({preflight_exit})
Path(a.ready_file).write_text(json.dumps({{"status":"ready"}}))
if {fail_observer}:
 time.sleep(.2)
 raise SystemExit(7)
while not Path(a.stop_file).exists(): time.sleep(.01)
Path(a.run_dir,"observer-drained").write_text("both sinks flushed")
''')
    output = tmp_path / "run"
    output.mkdir()
    if training_code is None:
        training_code = f"from pathlib import Path; Path({str(output / 'training-ran')!r}).write_text('yes'); print('trained')"
    manifest = output / "launch-manifest.json"
    manifest.write_text(json.dumps({
        "argv": [sys.executable, "-u", "-c", training_code],
        "host_source_sha256": launcher.host_source_hashes(root),
        "observability": {"mode": "local", "python": sys.executable, "module_root": str(root)},
    }))
    return root, output, manifest


def run(manifest, root):
    return launcher.launch(manifest, observer_python=Path(sys.executable), observer_module_root=root,
                           ready_timeout_s=3, drain_timeout_s=3, poll_s=.01)


def test_both_observer_readiness_and_final_drain_surround_training(tmp_path):
    root, output, manifest = prepared(tmp_path)
    result = run(manifest, root)
    assert result["status"] == "succeeded"
    assert result["observer_ready"] and result["training_started"]
    assert result["training_exit_code"] == result["observer_exit_code"] == 0
    assert (output / "training-ran").exists() and (output / "observer-drained").exists()
    assert result == json.loads((output / "execution.json").read_text())
    assert "trained" in (output / "runtime.log").read_text()
    with pytest.raises(FileExistsError):
        run(manifest, root)


def test_missing_dependency_stops_before_any_training_and_writes_failure(tmp_path):
    root, output, manifest = prepared(tmp_path, preflight_exit=2)
    result = run(manifest, root)
    assert result["status"] == "failed" and result["preflight_exit_code"] == 2
    assert not result["training_started"] and not (output / "training-ran").exists()
    assert (output / "execution.json").is_file()


def test_observer_failure_terminates_training_and_cannot_silently_pass(tmp_path):
    root, _output, manifest = prepared(tmp_path, fail_observer=True, training_code="import time; time.sleep(10)")
    result = run(manifest, root)
    assert result["status"] == "failed" and result["observer_exit_code"] == 7
    assert result["training_started"] and result["training_exit_code"] != 0
    assert "Observer exited during training" in result["failure_reason"]
    assert result["exit_code"] > 0


def test_modified_observer_is_rejected_before_training(tmp_path):
    root, output, manifest = prepared(tmp_path)
    (root / "experiments/nvidia_v2d/policy_observability.py").write_text("raise RuntimeError('changed')")
    result = run(manifest, root)
    assert result["status"] == "failed" and "source changed" in result["failure_reason"]
    assert not result["training_started"] and not (output / "training-ran").exists()


def test_sigterm_stops_training_then_drains_observer_and_records_interruption(tmp_path):
    root, output, manifest = prepared(tmp_path, training_code="import time; time.sleep(20)")
    process = subprocess.Popen([sys.executable, str(root / "experiments/nvidia_v2d/policy_logged_launch.py"),
        "--manifest", str(manifest), "--observer-python", sys.executable,
        "--observer-module-root", str(root), "--swanlab-mode", "local"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = output / "launcher-status.json"
            if status.exists() and json.loads(status.read_text()).get("training_started"):
                break
            if process.poll() is not None:
                raise AssertionError(process.communicate())
            time.sleep(.02)
        else:
            raise AssertionError("Test launch did not start training")
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=5)
        result = json.loads((output / "execution.json").read_text())
        assert process.returncode == 143 and result["status"] == "interrupted"
        assert result["signal"] == signal.SIGTERM and result["observer_exit_code"] == 0
        assert (output / "observer-drained").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
