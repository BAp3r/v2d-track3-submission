"""可恢复的单卡成对扫参；仅跳过哈希有效的成功结果，失败组不追加预算。"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from experiments.nvidia_v2d.policy_budget import TrainingBudget
from experiments.nvidia_v2d.policy_experiment import ExperimentSpec, _write, launch, training_argv
from experiments.nvidia_v2d.policy_run import RunRecord, stamp


class SweepRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    backend: Literal["official", "multitip"]
    seed: int
    spec_path: Path
    spec_sha256: str
    output: Path
    frozen_training_argv: list[str]


class SweepPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["nvidia_v2d.policy_sweep.v1"] = "nvidia_v2d.policy_sweep.v1"
    created_at: str
    purpose: Literal["mechanism_validation"] = "mechanism_validation"
    budget: TrainingBudget
    image: str
    repository: Path
    input_root: Path
    cache: Path
    source_sha256: dict[str, str]
    runs: list[SweepRun]

    @model_validator(mode="after")
    def paired_order(self) -> SweepPlan:
        expected = [(backend, seed) for seed in (42, 43, 44) for backend in ("official", "multitip")]
        if [(run.backend, run.seed) for run in self.runs] != expected:
            raise ValueError("正式对照必须按42/43/44种子成对交替串行")
        if len({str(run.output.resolve()) for run in self.runs}) != 6:
            raise ValueError("每次训练必须使用不同输出目录")
        if not self.image.startswith("sha256:"):
            raise ValueError("正式对照必须冻结实际镜像ID")
        return self


def verified_completed_run(output: Path, expected: ExperimentSpec) -> bool:
    """成功标志不足以跳过；检查所有原输入与已登记输出的内容哈希。"""
    try:
        record = RunRecord.model_validate_json((output / "run.json").read_text())
        if record.status != "succeeded" or record.input_unchanged is not True:
            return False
        if (
            record.spec.phase != "training"
            or record.spec.backend != expected.backend
            or record.spec.retarget_parquet != expected.parquet
            or record.spec.gate != expected.gate
        ):
            return False
        required = {
            "experiment.json",
            "unassisted.metrics.json",
            "unassisted.object_trajectories.parquet",
        }
        if required != set(record.spec.required_outputs) or len(record.outputs) != len(required):
            return False
        if {item.path for item in record.outputs} != {
            str((output / name).absolute()) for name in required
        }:
            return False
        if any(stamp(Path(item.path)) != item for item in record.outputs):
            return False
        inputs = json.loads((output / "input-manifest.json").read_text())
        if not {str(expected.parquet), str(expected.gate)}.issubset(
            {item["path"] for item in inputs}
        ):
            return False
        if any(stamp(Path(item["path"])).model_dump() != item for item in inputs):
            return False
        result = json.loads((output / "experiment.json").read_text())
        return (
            result.get("status") == "succeeded"
            and result.get("adapter_source_unchanged") is True
            and result.get("spec") == expected.model_dump(mode="json")
            and result.get("evaluation_evidence", {}).get("assistance") == "none"
        )
    except (OSError, ValueError, KeyError):
        return False


def _wait_existing_container(output: Path) -> None:
    name = f"fmc3-{output.name}"
    check = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", name], capture_output=True, text=True
    )
    if check.returncode == 0 and check.stdout.strip() == "true":
        # The container owns an internal hard timeout. A resumed sweep must not
        # allocate the GPU concurrently just because the earlier host exited.
        subprocess.run(["docker", "wait", name], check=False, timeout=172800, stdout=subprocess.DEVNULL)


def run_sweep(plan_path: Path) -> dict[str, Any]:
    plan_bytes = plan_path.read_bytes()
    plan = SweepPlan.model_validate_json(plan_bytes)
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    for path, digest in plan.source_sha256.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError("冻结的运行源码不一致")
    specs = []
    for run in plan.runs:
        content = run.spec_path.read_bytes()
        spec = ExperimentSpec.model_validate_json(content)
        if (
            hashlib.sha256(content).hexdigest() != run.spec_sha256
            or spec.backend != run.backend
            or spec.seed != run.seed
            or spec.budget != plan.budget
            or spec.image != plan.image
            or training_argv(spec) != run.frozen_training_argv
        ):
            raise ValueError("冻结的成对训练配置不一致")
        specs.append(spec)
    state: dict[str, Any] = {
        "schema": "nvidia_v2d.policy_sweep_state.v1",
        "plan_sha256": plan_sha,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "runs": [],
        "purpose": "mechanism_validation",
        "geometry_quality_accepted": False,
    }
    lock_path = plan_path.parent / "sweep.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for run, spec in zip(plan.runs, specs, strict=True):
            if hashlib.sha256(plan_path.read_bytes()).hexdigest() != plan_sha:
                raise ValueError("运行期间冻结计划发生变化")
            row: dict[str, Any] = {
                "backend": run.backend,
                "seed": run.seed,
                "output": str(run.output),
                "status": "running",
            }
            state["runs"].append(row)
            _write(plan_path.parent / "sweep-state.json", state)
            if run.output.exists():
                _wait_existing_container(run.output)
                row["status"] = (
                    "succeeded" if verified_completed_run(run.output, spec) else "failed_preserved"
                )
                row["resumed_without_retraining"] = True
            else:
                print(f"开始 {run.backend} seed={run.seed}", flush=True)
                exit_code = launch(
                    run.spec_path, run.output, plan.cache, plan.repository, plan.input_root
                )
                _wait_existing_container(run.output)
                row["status"] = (
                    "succeeded"
                    if exit_code == 0 and verified_completed_run(run.output, spec)
                    else "failed"
                )
            row["finished_at"] = datetime.now(UTC).isoformat()
            if row["status"] == "succeeded":
                result = json.loads((run.output / "experiment.json").read_text())
                row["metrics"] = result["metrics"]["metrics"]
                row["elapsed_s"] = result["elapsed_s"]
                row["trajectory_sha256"] = result["trajectory_validation"]["sha256"]
            print(f"结束 {run.backend} seed={run.seed}: {row['status']}", flush=True)
            _write(plan_path.parent / "sweep-state.json", state)
        state.update(
            status="completed"
            if all(row["status"] == "succeeded" for row in state["runs"])
            else "completed_with_failures",
            finished_at=datetime.now(UTC).isoformat(),
        )
        _write(plan_path.parent / "sweep-state.json", state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    result = run_sweep(parser.parse_args().plan)
    raise SystemExit(0 if result["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
