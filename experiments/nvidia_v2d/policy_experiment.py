"""单GPU成对机制实验：独立容器时限、真实训练/恢复与无辅助全序列轨迹。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from experiments.nvidia_v2d.policy_budget import TrainingBudget, training_overrides
from experiments.nvidia_v2d.policy_trajectory import validate_trajectory

THRESHOLDS = [
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
VOC_SCALES = [1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.0, 0.0, 0.0]
ADAPTER_METADATA = {
    "name": "nvidia-v2d-sharpa-flashchord-policy-experiment",
    "version": "1.2",
    "input_requirements": [
        "processed ManoSharpaData parquet",
        "verified mechanism-validation gate",
        "Sharpa assets and FlashCHORD container",
    ],
    "supported_input_scopes": ["single_episode"],
    "supported_modes": ["training", "checkpoint_resume", "unassisted_full_sequence_evaluation"],
    "output_schema": "nvidia_v2d.policy_experiment.v1",
}
CODE_FILES = (
    "policy_experiment_sweep.py",
    "policy_experiment.py",
    "policy_experiment_evaluation.py",
    "policy_checkpoint_audit.py",
    "policy_trajectory.py",
    "policy_budget.py",
    "policy_gate.py",
    "policy_run.py",
    "policy_metric_logging.py",
    "policy_metric_curves.py",
)


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    backend: Literal["official", "multitip"]
    parquet: Path
    gate: Path
    budget: TrainingBudget
    seed: int
    purpose: Literal["mechanism_validation"] = "mechanism_validation"
    verify_resume: bool = False
    replay_capacity: int = Field(gt=0)
    replay_minimum: int = Field(gt=0)
    replay_batch: int = Field(gt=0)
    updates_per_collection: float = Field(default=8.0, gt=0, allow_inf_nan=False)
    evaluation_worlds: int = Field(default=16, gt=0, le=256)
    image: str = "fmc3-flashchord:20261027-001"

    @model_validator(mode="after")
    def consistent_budget(self) -> ExperimentSpec:
        if self.budget.wall_limit_s > 7100:
            raise ValueError("需为终止容器预留时间，内部时限不得超过7100秒")
        if self.seed not in self.budget.seeds:
            raise ValueError("种子不属于冻结预算")
        if self.replay_minimum > self.replay_capacity or self.replay_batch > self.replay_minimum:
            raise ValueError("replay容量、预热与batch大小不一致")
        if self.budget.total_steps <= self.replay_minimum:
            raise ValueError("预算必须包含真实参数更新")
        if self.verify_resume and (self.budget.total_steps // 2) % self.budget.world_count:
            raise ValueError("恢复校验的中点必须为完整采集批次")
        return self


def training_argv(spec: ExperimentSpec) -> list[str]:
    args = training_overrides(
        spec.budget,
        seed=spec.seed,
        parquet=str(spec.parquet),
        output="/run/train",
        thresholds=THRESHOLDS,
        voc_scales=VOC_SCALES,
    )
    args = [arg for arg in args if not arg.startswith("training.checkpoint.save_interval")]
    interval = (
        spec.budget.total_steps // 2
        if spec.verify_resume
        else max(
            spec.budget.world_count,
            spec.budget.total_steps // 10 // spec.budget.world_count * spec.budget.world_count,
        )
    )
    return [
        *args,
        f"training.replay.capacity={spec.replay_capacity}",
        f"training.replay.minimum_size={spec.replay_minimum}",
        f"training.replay.batch_size={spec.replay_batch}",
        f"training.updates_per_collection={spec.updates_per_collection}",
        "scene.support=false",
        # The upstream default boosts VOC back to 1 after every random reset,
        # even when the curriculum target is 0. Future runs disable that boost.
        "reset.voc_decay_steps=0",
        "sim.njmax=2048",
        "sim.nconmax=512",
        f"training.checkpoint.save_interval_environment_steps={interval}",
        "training.checkpoint.save_replay=true",
        "logging.log_interval=1",
    ]


def logged_training_argv(metrics_path: Path, overrides: list[str]) -> list[str]:
    """将训练和恢复统一接到每采集批次落盘的本地指标入口。"""
    return [
        sys.executable, "-m", "experiments.nvidia_v2d.policy_metric_logging",
        "scripts/train_flash_sac.py", str(metrics_path), *overrides,
    ]


def evaluation_checkpoints(train_directory: Path, final: Path) -> list[Path]:
    """逐个评估实际保存的checkpoint；恢复后的最终权重替换同一步旧权重。"""
    policies = {}
    for path in train_directory.glob("policy_*.safetensors"):
        match = re.fullmatch(r"policy_(\d+)\.safetensors", path.name)
        if match is not None:
            policies[int(match.group(1))] = path
    match = re.fullmatch(r"policy_(\d+)\.safetensors", final.name)
    if match is None or not final.is_file():
        raise ValueError("最终策略checkpoint不存在或文件名不包含训练步数")
    policies[int(match.group(1))] = final
    return [policies[step] for step in sorted(policies)]


def docker_argv(
    spec: ExperimentSpec,
    spec_path: Path,
    output: Path,
    cache: Path,
    repository: Path,
    input_root: Path,
) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--init",
        "--name",
        f"fmc3-{output.name}",
        "--gpus",
        "device=0",
        "--network",
        "none",
        "--memory",
        "20g",
        "--memory-swap",
        "20g",
        "--cpus",
        "8",
        "--shm-size",
        "2g",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{input_root}:{input_root}:ro",
        "-v",
        f"{repository}:/adapter:ro",
        "-v",
        f"{output}:/run",
        "-v",
        f"{cache}:/cache",
        "-v",
        f"{spec_path}:/experiment-spec.json:ro",
        "-e",
        "PYTHONPATH=/adapter",
        "-e",
        "PYTHONUNBUFFERED=1",
        "-e",
        "XLA_PYTHON_CLIENT_PREALLOCATE=false",
        "-e",
        "FLASH_CHORD_CACHE_DIR=/cache/assets",
        "-e",
        "JAX_COMPILATION_CACHE_DIR=/cache/jax",
        "-e",
        "XDG_CACHE_HOME=/cache/xdg",
        "-e",
        "WARP_CACHE_PATH=/cache/warp",
        spec.image,
        "timeout",
        "--signal=TERM",
        "--kill-after=20s",
        f"{spec.budget.wall_limit_s}s",
        "python",
        "-m",
        "experiments.nvidia_v2d.policy_experiment",
        "container",
        "--spec",
        "/experiment-spec.json",
    ]


def _write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def require_experiment_gate(spec: ExperimentSpec) -> dict[str, Any]:
    from experiments.nvidia_v2d.policy_gate import verify_gate

    return verify_gate(
        spec.gate,
        backend=spec.backend,
        parquet=spec.parquet,
        episode_id="track_3-public-episode-000044",
    )


def numerical_log_problem(text: str) -> str | None:
    match = re.search(
        r"(?:nefc|ncon)\s+overflow[^\n]*|CUDA error[^\n]*|out of memory[^\n]*", text, re.IGNORECASE
    )
    return match.group(0) if match else None


def require_update_audit(
    audit_path: Path,
    checkpoint: Path,
    *,
    previous_updates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """绑定只读CPU审计；训练和恢复不能只凭尝试计数判定参数更新。"""
    from experiments.nvidia_v2d import policy_checkpoint_audit

    audit: dict[str, Any] = json.loads(audit_path.read_text())
    auditor_sha = hashlib.sha256(Path(policy_checkpoint_audit.__file__).read_bytes()).hexdigest()
    if (
        audit.get("checkpoint_sha256") != hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        or audit.get("auditor_sha256") != auditor_sha
        or audit.get("source_unchanged") is not True
        or audit.get("template_all_shapes_dtypes_checked") is not True
    ):
        raise ValueError("Adam审计与实际训练状态或审计源码绑定失败")
    updates = policy_checkpoint_audit.summarize_gradient_updates(audit["scalar_state"])
    if updates != audit.get("updates") or not updates["effective_update_gate_passed"]:
        raise ValueError("actor或critic没有实际生效更新，或审计计数不一致")
    if previous_updates is not None and any(
        updates[name]["applied"] <= previous_updates[name]["applied"] for name in ("actor", "critic")
    ):
        raise ValueError("恢复后actor或critic没有继续生效更新")
    return audit


def run_inside_container(spec: ExperimentSpec, output: Path = Path("/run")) -> dict[str, Any]:
    from flash_chord.data import load_reference
    from flash_chord.training.checkpoint import read_checkpoint_metadata

    started = time.monotonic()
    source_hashes = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in CODE_FILES
    }
    record: dict[str, Any] = {
        "schema": "nvidia_v2d.policy_experiment.v1",
        "adapter": ADAPTER_METADATA,
        "adapter_code_sha256": source_hashes,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "spec": spec.model_dump(mode="json"),
        "purpose": "mechanism_validation",
        "geometry_quality_accepted": False,
        "penetration_status": "unknown",
        "policy_quality_claim_allowed": False,
        "training_reset_voc_boost_disabled": True,
        "metric_logging": {
            "every_collection": True,
            "loss_aggregation": "mean_over_actual_optimizer_updates_in_collection",
            "evaluation": "every_saved_policy_checkpoint_unassisted_full_sequence",
        },
        "notes_zh": [
            "开放物体网格无法可靠量穿透；训练可运行不代表抓持合格。",
            "输入旧拟合误差0代表未知，不用于质量验收。",
            "官方IK优化腕位，自有multi-tip固定重建腕位，此差异保留。",
            "缺少重建支撑面；共同禁用support USDA，保留官方z=0地面。PCA落地不是实测桌面。",
            "v1.1共同关闭随机重置VOC扶持；历史policy-code-002六组仍保留20步扶持，不能混为同一实验。",
        ],
        "stages": [],
    }

    def save() -> None:
        _write(output / "experiment.json", record)

    def stage(name: str, args: list[str]) -> None:
        row: dict[str, Any] = {
            "name": name,
            "argv": args,
            "started_at": datetime.now(UTC).isoformat(),
            "status": "running",
        }
        record["stages"].append(row)
        save()
        print(f"阶段 {name} 开始", flush=True)
        before = time.monotonic()
        log_path = output / f"{name}.log"
        failure = None
        with log_path.open("w") as log:
            process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                failure = numerical_log_problem(log_path.read_text(errors="replace"))
                if failure or time.monotonic() - started >= spec.budget.wall_limit_s - 20:
                    failure = failure or "容器内部阶段达到作业时限"
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                    break
                time.sleep(2)
        failure = failure or numerical_log_problem(log_path.read_text(errors="replace"))
        row.update(
            exit_code=process.returncode,
            elapsed_s=time.monotonic() - before,
            status="succeeded" if process.returncode == 0 and failure is None else "failed",
            failure_reason=failure,
        )
        save()
        if process.returncode or failure:
            raise RuntimeError(f"{name}失败：{failure or process.returncode}")

    def update_audit(
        name: str, checkpoint: Path, previous_updates: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        audit_path = output / f"{name}.json"
        stage(
            name,
            [
                "env",
                "JAX_PLATFORMS=cpu",
                "CUDA_VISIBLE_DEVICES=",
                sys.executable,
                "-m",
                "experiments.nvidia_v2d.policy_checkpoint_audit",
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(audit_path),
            ],
        )
        audit = require_update_audit(audit_path, checkpoint, previous_updates=previous_updates)
        record[name] = audit
        save()
        return audit

    try:
        require_experiment_gate(spec)
        reference = load_reference(str(spec.parquet), control_fps=20.0, motion_speed=1.0)
        if reference.num_frames != 249 or reference.fps != 20:
            raise ValueError("public44时间轴被隐式修改")
        record["reference"] = {
            "frames": reference.num_frames,
            "fps": reference.fps,
            "object_names": list(reference.object_body_names()),
        }
        save()
        args = training_argv(spec)
        stage("train", logged_training_argv(output / "train-metrics.jsonl", args))
        final = output / f"train/policy_{spec.budget.total_steps}.safetensors"
        trained = read_checkpoint_metadata(final)
        if int(trained["global_update_step"]) <= 0 or float(trained["voc_scale"]) != 0:
            raise ValueError("训练未记录更新尝试或未进入目标VOC=0的课程阶段")
        resolved = json.loads(trained["resolved_config_json"])
        if resolved["reset"]["voc_decay_steps"] != 0:
            raise ValueError("训练仍启用了随机重置VOC扶持")
        record["train_checkpoint"] = trained
        update_audit(
            "train_update_audit", output / f"train/state_{spec.budget.total_steps}.safetensors"
        )
        if spec.verify_resume:
            half = spec.budget.total_steps // 2
            prior_audit = update_audit(
                "resume_start_update_audit", output / f"train/state_{half}.safetensors"
            )
            resume = [arg for arg in args if not arg.startswith(("output_dir=", "hydra.run.dir="))]
            resume.extend(
                [
                    "output_dir='/run/resume'",
                    "hydra.run.dir=/run/hydra/resume",
                    "training.resume=true",
                    f"training.checkpoint_path=/run/train/state_{half}.safetensors",
                    f"training.checkpoint.load_replay_path=/run/train/replay_{half}.safetensors",
                ]
            )
            stage("resume", logged_training_argv(output / "resume-metrics.jsonl", resume))
            final = output / f"resume/policy_{spec.budget.total_steps}.safetensors"
            resumed = read_checkpoint_metadata(final)
            prior = read_checkpoint_metadata(output / f"train/policy_{half}.safetensors")
            if int(resumed["global_update_step"]) <= int(prior["global_update_step"]):
                raise ValueError("恢复后没有继续尝试更新参数")
            update_audit(
                "resume_update_audit",
                output / f"resume/state_{spec.budget.total_steps}.safetensors",
                previous_updates=prior_audit["updates"],
            )
            record["resume_checkpoint"] = resumed
        record["evaluation_checkpoints"] = []
        for checkpoint in evaluation_checkpoints(output / "train", final):
            checkpoint_step = int(checkpoint.stem.removeprefix("policy_"))
            evaluation_output = output if checkpoint == final else output / "evaluations" / str(checkpoint_step)
            evaluation_output.mkdir(parents=True, exist_ok=True)
            stage(
                "evaluate" if checkpoint == final else f"evaluate_{checkpoint_step}",
                [
                    sys.executable, "-m", "experiments.nvidia_v2d.policy_experiment", "evaluate",
                    "--spec", "/experiment-spec.json", "--checkpoint", str(checkpoint),
                    "--output", str(evaluation_output),
                    "--metrics-history", str(output / "evaluation-metrics.jsonl"),
                ],
            )
            record["evaluation_checkpoints"].append({
                "checkpoint": str(checkpoint), "environment_steps": checkpoint_step,
                "report": str(evaluation_output / "unassisted.metrics.json"),
                "evidence": str(evaluation_output / "evaluation-evidence.json"),
            })
            save()
        record["metrics"] = json.loads((output / "unassisted.metrics.json").read_text())
        record["trajectory_validation"] = validate_trajectory(
            output / "unassisted.object_trajectories.parquet",
            reference.num_frames,
            spec.evaluation_worlds,
        )
        record["evaluation_evidence"] = json.loads((output / "evaluation-evidence.json").read_text())
        record["status"] = "succeeded"
    except Exception as exc:
        record.update(
            status="failed",
            failure_reason=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        print(record["traceback"], flush=True)
    record.update(finished_at=datetime.now(UTC).isoformat(), elapsed_s=time.monotonic() - started)
    record["adapter_source_unchanged"] = all(
        hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() == digest
        for name, digest in source_hashes.items()
    )
    if not record["adapter_source_unchanged"]:
        record.update(status="failed", failure_reason="运行期间适配器源码发生变化")
    save()
    return record


def launch(spec_path: Path, output: Path, cache: Path, repository: Path, input_root: Path) -> int:
    from experiments.nvidia_v2d.policy_run import RunSpec, run_job

    spec = ExperimentSpec.model_validate_json(spec_path.read_text())
    gate = require_experiment_gate(spec)
    cache.mkdir(parents=True, exist_ok=True)
    args = docker_argv(spec, spec_path, output, cache, repository, input_root)
    run = RunSpec(
        phase="training",
        episode_id="track_3-public-episode-000044",
        backend=spec.backend,
        argv=args,
        cwd=repository,
        inputs=[
            spec_path,
            spec.parquet,
            Path(gate["retarget_manifest_path"]),
            *[repository / "experiments/nvidia_v2d" / name for name in CODE_FILES],
        ],
        gate=spec.gate,
        retarget_parquet=spec.parquet,
        required_outputs=[
            "experiment.json",
            "unassisted.metrics.json",
            "unassisted.object_trajectories.parquet",
            "train-metrics.jsonl",
            "train-metrics.csv",
            "train-metrics.loss.svg",
            "evaluation-metrics.jsonl",
            "evaluation-metrics.csv",
            "evaluation-metrics.evaluation.svg",
        ],
        wall_limit_s=min(7200, spec.budget.wall_limit_s + 30),
    )
    result = run_job(run, output)
    print(result.model_dump_json(indent=2))
    return 0 if result.status == "succeeded" else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["launch", "container", "evaluate"])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--metrics-history", type=Path)
    args = parser.parse_args()
    if args.mode == "evaluate":
        from experiments.nvidia_v2d.policy_experiment_evaluation import evaluate_with_evidence

        if args.checkpoint is None:
            parser.error("evaluate需指定checkpoint")
        spec = ExperimentSpec.model_validate_json(args.spec.read_text())
        evaluate_with_evidence(
            args.checkpoint, spec.evaluation_worlds, args.output or Path("/run"),
            metrics_history=args.metrics_history,
        )
        return
    if args.mode == "container":
        result = run_inside_container(ExperimentSpec.model_validate_json(args.spec.read_text()))
        raise SystemExit(0 if result["status"] == "succeeded" else 1)
    if not all([args.output, args.cache, args.repository, args.input_root]):
        parser.error("launch需指定输出、缓存、代码与只读输入根目录")
    raise SystemExit(launch(args.spec, args.output, args.cache, args.repository, args.input_root))


if __name__ == "__main__":
    main()
