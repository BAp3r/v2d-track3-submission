"""单样本物理修复训练：有限支撑、完整 loss 和每个保存 checkpoint 的无辅助评估。

The explicit support adapter changes scene construction only. It never assigns
robot/object state, changes actions, or enables auxiliary object control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from experiments.nvidia_v2d.policy_budget import TrainingBudget
from experiments.nvidia_v2d.policy_experiment import (
    CODE_FILES,
    THRESHOLDS,
    ExperimentSpec,
    _write,
    numerical_log_problem,
    require_experiment_gate,
    require_update_audit,
    training_argv,
)

LEGACY_ROOT = Path("/var/tmp/fmc3-sharpa-validation-20261027-001")
FROZEN_IMAGE = "sha256:7fdce3ad594fbad8b5983fde7846908be2ed6179ba4ebcfd1edefbcbcf4c628c"
SOURCE_FILES = (*CODE_FILES, "policy_single_sample.py", "physical_wrist_config.py")


class SingleSampleSpec(ExperimentSpec):
    backend: str = "official"
    parquet: Path = LEGACY_ROOT / "retarget/public44-official-002/data.parquet"
    gate: Path = LEGACY_ROOT / "geometry-audit/training-gates/public44-official-002.json"
    budget: TrainingBudget = Field(default_factory=lambda: TrainingBudget(
        total_steps=2_097_152, world_count=256, wall_limit_s=7000, seeds=(42,),
    ))
    seed: int = 42
    replay_capacity: int = 65536
    replay_minimum: int = 1024
    replay_batch: int = 512
    image: str = FROZEN_IMAGE
    support_usda: Path
    support_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    motion_speed: float = Field(default=0.5, gt=0, le=1, allow_inf_nan=False)
    checkpoint_interval: int | None = Field(default=None, gt=0)
    wrist_kp: float | None = Field(default=None, ge=25, le=200, allow_inf_nan=False)
    wrist_kd: float | None = Field(default=None, ge=6, le=25, allow_inf_nan=False)
    unintended_contact_weight: float = Field(default=-10.0, ge=-10, le=0, allow_inf_nan=False)
    always_reset_to_first_frame: bool = False
    resume_checkpoint: Path | None = None
    resume_replay: Path | None = None

    @model_validator(mode="after")
    def physical_contract(self) -> SingleSampleSpec:
        if (self.wrist_kp is None) != (self.wrist_kd is None):
            raise ValueError("wrist gain ablation requires both kp and kd")
        if self.backend not in ("official", "multitip"):
            raise ValueError("backend must be official or multitip")
        if self.verify_resume:
            raise ValueError("use explicit resume_checkpoint and resume_replay")
        if (self.resume_checkpoint is None) != (self.resume_replay is None):
            raise ValueError("resume requires both a training-state checkpoint and replay")
        if self.resume_checkpoint is not None and not self.resume_checkpoint.name.startswith("state_"):
            raise ValueError("resuming optimization requires state_*.safetensors, not policy weights")
        if self.checkpoint_interval is not None and self.checkpoint_interval % self.budget.world_count:
            raise ValueError("checkpoint interval must contain complete collection batches")
        if not self.image.startswith("sha256:"):
            raise ValueError("use a content-addressed frozen container image")
        return self


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def physical_training_overrides(spec: SingleSampleSpec, output: Path) -> list[str]:
    replaced = ("scene.support=", "collision.ground=", "collision.robot_support_collision=",
                "output_dir=", "hydra.run.dir=", "training.checkpoint.save_interval_environment_steps=",
                "task.motion_speed=")
    args = [arg for arg in training_argv(spec) if not arg.startswith(replaced)]
    interval = spec.checkpoint_interval or max(spec.budget.world_count,
        spec.budget.total_steps // 10 // spec.budget.world_count * spec.budget.world_count)
    args.extend([
        "scene.support=true", "collision.ground=false", "collision.robot_support_collision=true",
        f"task.motion_speed={spec.motion_speed}",
        f"reset.always_reset_to_first_frame={str(spec.always_reset_to_first_frame).lower()}",
        f"output_dir='{output / 'train'}'", f"hydra.run.dir='{output / 'hydra/train'}'",
        f"training.checkpoint.save_interval_environment_steps={interval}",
    ])
    if spec.wrist_kp is not None:
        args.extend([
            "embodiment.config._target_=experiments.nvidia_v2d.physical_wrist_config.make_sharpa_config",
            f"+embodiment.config.kp={spec.wrist_kp}", f"+embodiment.config.kd={spec.wrist_kd}",
        ])
    if spec.unintended_contact_weight != -10:
        args.append(f"objective.unintended_contact.weight={spec.unintended_contact_weight}")
        args.extend(f"curriculum.stages.{i}.objective_weights.unintended_contact={spec.unintended_contact_weight}"
                    for i in range(len(THRESHOLDS)))
    if spec.resume_checkpoint is not None:
        args.extend(["training.resume=true", f"training.checkpoint_path='{spec.resume_checkpoint}'",
                     f"training.checkpoint.load_replay_path='{spec.resume_replay}'"])
    return args


@contextmanager
def explicit_support(spec: SingleSampleSpec) -> Iterator[dict[str, Any]]:
    """Bind one exact reference to one hashed finite support file, before scene build."""
    from flash_chord.scene import setup

    if sha256(spec.support_usda) != spec.support_sha256:
        raise ValueError("support USDA hash differs from the frozen spec")
    original = setup.support_usda_for_reference
    receipt: dict[str, Any] = {
        "support_usda": str(spec.support_usda), "support_sha256": spec.support_sha256,
        "ground": False, "robot_support_collision": True, "resolver_calls": 0,
        "state_writes_after_reset": False,
    }

    def resolve(reference: Any) -> Path:
        if Path(reference.metadata.source_path).resolve() != spec.parquet.resolve():
            raise ValueError("explicit support may only bind the frozen sample reference")
        receipt["resolver_calls"] += 1
        return spec.support_usda

    setup.support_usda_for_reference = resolve
    try:
        yield receipt
        if receipt["resolver_calls"] == 0:
            raise ValueError("scene never requested the explicit finite support")
    finally:
        setup.support_usda_for_reference = original
        if sha256(spec.support_usda) != spec.support_sha256:
            raise ValueError("support changed during scene execution")


def train(spec: SingleSampleSpec, output: Path) -> None:
    from experiments.nvidia_v2d import policy_metric_logging

    require_experiment_gate(spec)
    prior_argv = sys.argv
    with explicit_support(spec) as receipt:
        try:
            sys.argv = ["policy_metric_logging", "scripts/train_flash_sac.py",
                        str(output / "train-metrics.jsonl"), *physical_training_overrides(spec, output)]
            policy_metric_logging.main()
        finally:
            sys.argv = prior_argv
            _write(output / "training-scene.json", receipt)


def evaluate(spec: SingleSampleSpec, checkpoint: Path, output: Path,
             metrics_history: Path | None = None) -> dict[str, Any]:
    from flash_chord.training.checkpoint import read_checkpoint_metadata

    from experiments.nvidia_v2d.policy_experiment_evaluation import (
        evaluate_with_evidence,
    )

    metadata = read_checkpoint_metadata(checkpoint)
    config = json.loads(metadata["resolved_config_json"])
    if (not config["scene"]["support"] or config["collision"]["ground"]
            or not config["collision"]["robot_support_collision"]
            or config["reset"]["voc_decay_steps"] != 0):
        raise ValueError("checkpoint was not trained with the declared physical scene and reset contract")
    if Path(config["task"]["parquet"]).resolve() != spec.parquet.resolve():
        raise ValueError("checkpoint reference differs from the frozen sample")
    if float(config["task"]["motion_speed"]) != spec.motion_speed:
        raise ValueError("checkpoint motion speed differs from the frozen sample")
    if config["reset"]["always_reset_to_first_frame"] != spec.always_reset_to_first_frame:
        raise ValueError("checkpoint training reset distribution differs from the frozen spec")
    embodiment = config["embodiment"]["config"]
    if spec.wrist_kp is not None:
        if (embodiment["_target_"] != "experiments.nvidia_v2d.physical_wrist_config.make_sharpa_config"
                or embodiment["kp"] != spec.wrist_kp or embodiment["kd"] != spec.wrist_kd):
            raise ValueError("checkpoint wrist gains differ from the frozen spec")
    elif embodiment["_target_"] != "flash_chord.embodiments.sharpa_hands.SharpaHandsConfig":
        raise ValueError("checkpoint does not use the original wrist configuration")
    if any(stage["objective_weights"]["unintended_contact"] != spec.unintended_contact_weight
           for stage in config["curriculum"]["stages"]):
        raise ValueError("checkpoint contact penalty differs from the frozen spec")
    output.mkdir(parents=True, exist_ok=True)
    with explicit_support(spec) as receipt:
        evidence = evaluate_with_evidence(checkpoint, spec.evaluation_worlds, output,
                                         metrics_history=metrics_history,
                                         scene_context={"support_sha256": spec.support_sha256,
                                                        "ground": False,
                                                        "robot_support_collision": True,
                                                        "motion_speed": spec.motion_speed,
                                                        "wrist_kp": spec.wrist_kp,
                                                        "wrist_kd": spec.wrist_kd,
                                                        "training_reset_always_first": spec.always_reset_to_first_frame,
                                                        "unintended_contact_weight": spec.unintended_contact_weight})
    _write(output / "evaluation-scene.json", receipt)
    return evidence


def run(spec: SingleSampleSpec, spec_path: Path, output: Path) -> dict[str, Any]:
    """Train once, then evaluate every checkpoint, including those saved before failure."""
    started = time.monotonic()
    source = {name: sha256(Path(__file__).parent / name) for name in SOURCE_FILES}
    record: dict[str, Any] = {
        "schema": "nvidia_v2d.single_sample_physical.v1", "status": "running",
        "started_at": datetime.now(UTC).isoformat(), "spec": spec.model_dump(mode="json"),
        "source_sha256": source, "support_sha256": spec.support_sha256,
        "policy_quality_claim_allowed": False, "geometry_quality_accepted": False,
        "stages": [], "evaluation_checkpoints": [],
        "training_reset_voc_boost_disabled": True,
        "metric_logging": "each collection; each saved policy checkpoint; fsynced JSONL/CSV and SVG curves",
    }
    output.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        _write(output / "single-sample.json", record)

    def stage(name: str, argv: list[str], timeout: float) -> bool:
        row: dict[str, Any] = {"name": name, "argv": argv, "status": "running"}
        record["stages"].append(row)
        save()
        before = time.monotonic()
        log_path = output / f"{name}.log"
        failure = None
        with log_path.open("x") as log:
            process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                failure = numerical_log_problem(log_path.read_text(errors="replace"))
                if failure or time.monotonic() - before > timeout:
                    failure = failure or "stage wall limit reached"
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                    break
                time.sleep(2)
        failure = failure or numerical_log_problem(log_path.read_text(errors="replace"))
        row.update(exit_code=process.returncode, elapsed_s=time.monotonic() - before,
                   status="succeeded" if process.returncode == 0 and not failure else "failed",
                   failure_reason=failure)
        save()
        return row["status"] == "succeeded"

    entry = [sys.executable, "-m", "experiments.nvidia_v2d.policy_single_sample"]
    try:
        launch_manifest = output / "launch-manifest.json"
        if launch_manifest.exists():
            frozen = json.loads(launch_manifest.read_text())
            if (frozen["source_sha256"] != source
                    or frozen["spec_sha256"] != sha256(spec_path)
                    or frozen["support_sha256"] != sha256(spec.support_usda)):
                raise ValueError("prepared source, spec or support changed before execution")
        gate = require_experiment_gate(spec)
        record["verified_input_sha256"] = gate["verified_files"]
        save()
        # Reserve time to evaluate saved weights if the training budget is interrupted.
        trained = stage("train", [*entry, "train", "--spec", str(spec_path), "--output", str(output)],
                        max(1, spec.budget.wall_limit_s - 900))
        checkpoints = sorted((output / "train").glob("policy_*.safetensors"),
                             key=lambda path: int(path.stem.removeprefix("policy_")))
        evaluated = True
        for checkpoint in checkpoints:
            step = int(checkpoint.stem.removeprefix("policy_"))
            target = output / "evaluations" / str(step)
            remaining = spec.budget.wall_limit_s - (time.monotonic() - started) - 20
            if remaining <= 0:
                evaluated = False
                record["unevaluated_checkpoints"] = [str(p) for p in checkpoints
                    if int(p.stem.removeprefix("policy_")) >= step]
                break
            ok = stage(f"evaluate_{step}", [*entry, "evaluate", "--spec", str(spec_path),
                       "--checkpoint", str(checkpoint), "--output", str(target),
                       "--metrics-history", str(output / "evaluation-metrics.jsonl")], remaining)
            evaluated &= ok
            record["evaluation_checkpoints"].append({"checkpoint": str(checkpoint),
                "environment_steps": step, "status": "succeeded" if ok else "failed",
                "output": str(target), "checkpoint_sha256": sha256(checkpoint)})
            save()
        final_state = output / "train" / f"state_{spec.budget.total_steps}.safetensors"
        audited = False
        remaining = spec.budget.wall_limit_s - (time.monotonic() - started) - 20
        if final_state.exists() and remaining > 0:
            audit_path = output / "train-update-audit.json"
            if stage("audit", ["env", "JAX_PLATFORMS=cpu", "CUDA_VISIBLE_DEVICES=", sys.executable,
                     "-m", "experiments.nvidia_v2d.policy_checkpoint_audit", "--checkpoint",
                     str(final_state), "--output", str(audit_path)], remaining):
                record["update_audit"] = require_update_audit(audit_path, final_state)
                audited = True
        final_policy = output / "train" / f"policy_{spec.budget.total_steps}.safetensors"
        record["status"] = "succeeded" if trained and evaluated and audited and final_policy.exists() else "failed"
        record["training_completed"] = trained and final_policy.exists()
    except Exception as exc:  # noqa: BLE001 - persist evidence for any driver failure
        record.update(status="failed", failure_reason=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
    record.update(finished_at=datetime.now(UTC).isoformat(), elapsed_s=time.monotonic() - started,
                  source_unchanged=all(sha256(Path(__file__).parent / name) == digest
                                       for name, digest in source.items()))
    if not record["source_unchanged"]:
        record["status"] = "failed"
    record["inputs_unchanged"] = all(
        sha256(Path(path)) == digest for path, digest in record.get("verified_input_sha256", {}).items()
    )
    if not record["inputs_unchanged"]:
        record.update(status="failed", failure_reason="input changed during execution")
    # Process completion and measured task outcomes are distinct: a clean run can
    # legitimately finish with zero successful full-sequence attempts.
    record["pipeline_status"] = record["status"]
    task_evidence: dict[str, Any] = {
        "criterion": "official_chord_full_sequence_reference_tracking",
        "assistance": "none", "sample_count": 1,
        "count_method": "verified reported success_rate times world_count; integral count required",
        "scope": "measured attempts for this reconstructed sample; not geometry validation or generalization",
        "verified_checkpoints": [], "rejected_checkpoints": [],
        "successful_attempts": 0, "evaluated_attempts": 0,
    }
    for checkpoint_row in record["evaluation_checkpoints"]:
        if checkpoint_row["status"] != "succeeded":
            continue
        target = Path(checkpoint_row["output"])
        try:
            report_path = target / "unassisted.metrics.json"
            evidence_path = target / "evaluation-evidence.json"
            report = json.loads(report_path.read_text())
            evidence = json.loads(evidence_path.read_text())
            digest = sha256(Path(checkpoint_row["checkpoint"]))
            worlds, steps = report["world_count"], report["step_count"]
            if type(worlds) is not int or worlds <= 0 or type(steps) is not int or steps <= 0:
                raise ValueError("invalid evaluation dimensions")
            if any(value != digest for value in (
                report["checkpoint_sha256"], evidence["checkpoint_sha256"],
                checkpoint_row["checkpoint_sha256"],
            )) or evidence["trajectory_sha256"] != sha256(target / "unassisted.object_trajectories.parquet"):
                raise ValueError("evaluation artifact hash mismatch")
            if (evidence["reference_parquet_sha256"] != sha256(spec.parquet)
                    or evidence["scene_context"]["support_sha256"] != spec.support_sha256):
                raise ValueError("evaluation reference or finite support differs from the spec")
            for document in (report, evidence):
                expected = {"world_count": worlds, "step_count": steps, "reset_mode": "explicit",
                            "start_frame": 0, "motion_start_frame": 0, "motion_end_frame": -1}
                if any(document[key] != value for key, value in expected.items()):
                    raise ValueError("evaluation is not the complete explicit-reset sequence")
            scales = evidence["observed_control_scales"]
            if (evidence["assistance"] != "none" or not evidence["policy_deterministic"]
                    or evidence["action_step_count"] != steps
                    or evidence["observed_policy_action_count"] != steps or len(scales) != steps
                    or any(evidence[key] != 0 for key in (
                        "voc_scale", "applied_object_control_scale", "voc_scale_max", "applied_control_scale_max"))
                    or any(row[key] != 0 for row in scales for key in (
                        "voc_scale_max_abs", "applied_object_control_scale_max_abs"))
                    or report["non_finite_worlds"]):
                raise ValueError("unassisted finite rollout evidence is incomplete")
            rate = float(report["reference_tracking"]["success_rate"])
            if not 0 <= rate <= 1 or report["metrics"]["chord_sr"] != rate:
                raise ValueError("inconsistent task success metrics")
            successes = round(rate * worlds)
            if abs(successes - rate * worlds) > 1e-7:
                raise ValueError("reported rate does not represent a whole number of attempts")
            task_evidence["verified_checkpoints"].append({
                "environment_steps": checkpoint_row["environment_steps"], "checkpoint_sha256": digest,
                "successful_attempts": successes, "evaluated_attempts": worlds, "success_rate": rate,
                "report": str(report_path), "report_sha256": sha256(report_path),
                "evidence": str(evidence_path), "evidence_sha256": sha256(evidence_path),
            })
            task_evidence["successful_attempts"] += successes
            task_evidence["evaluated_attempts"] += worlds
        except (OSError, ValueError, KeyError, TypeError) as exc:
            task_evidence["rejected_checkpoints"].append({
                "checkpoint": checkpoint_row["checkpoint"], "reason": f"{type(exc).__name__}: {exc}"})
    record["task_success"] = (task_evidence["successful_attempts"] > 0
                              if task_evidence["evaluated_attempts"] else None)
    task_evidence["status"] = ("observed_success" if record["task_success"] else
                               "observed_failure" if record["task_success"] is False else "unverified")
    record["task_success_evidence"] = task_evidence
    save()
    return record


def docker_command(spec: SingleSampleSpec, spec_path: Path, output: Path,
                   repository: Path, legacy_inputs: Path, cache: Path) -> list[str]:
    mounts = [(repository, "/adapter", "ro"), (repository, str(repository), "ro"),
              (legacy_inputs, str(LEGACY_ROOT), "ro"), (output, "/run", "rw"),
              (cache, "/cache", "rw"), (spec_path, "/single-sample-spec.json", "ro")]
    # Inputs outside the repository (e.g. an archived state/replay) stay read-only.
    for path in (spec.support_usda, spec.resume_checkpoint, spec.resume_replay):
        if path is not None and not path.is_relative_to(repository) and not path.is_relative_to(LEGACY_ROOT):
            mounts.append((path, str(path), "ro"))
    args = ["docker", "run", "--rm", "--init", "--name", f"fmc3-{output.name}", "--gpus", "device=0",
            "--network", "none", "--memory", "20g", "--memory-swap", "20g", "--cpus", "8",
            "--shm-size", "2g", "--user", f"{os.getuid()}:{os.getgid()}"]
    for source, target, mode in mounts:
        args.extend(["-v", f"{source}:{target}:{mode}"])
    for value in ("PYTHONPATH=/adapter", "PYTHONUNBUFFERED=1", "XLA_PYTHON_CLIENT_PREALLOCATE=false",
                  "FLASH_CHORD_CACHE_DIR=/cache/assets", "JAX_COMPILATION_CACHE_DIR=/cache/jax",
                  "XDG_CACHE_HOME=/cache/xdg", "WARP_CACHE_PATH=/cache/warp"):
        args.extend(["-e", value])
    return [*args, spec.image, "timeout", "--signal=TERM", "--kill-after=20s",
            f"{spec.budget.wall_limit_s}s", "python", "-m", "experiments.nvidia_v2d.policy_single_sample",
            "run", "--spec", "/single-sample-spec.json", "--output", "/run"]


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.nvidia_v2d.policy_logged_launch import host_source_hashes

    repository = args.repository.resolve()
    output = args.output.resolve()
    support = args.support_usda.resolve()
    fields = json.loads(args.spec.read_text()) if args.spec else {}
    fields.update(support_usda=str(support), support_sha256=sha256(support))
    if args.total_steps is not None or args.world_count is not None:
        fields["budget"] = {**SingleSampleSpec.model_fields["budget"].default_factory().model_dump(),
                            **fields.get("budget", {})}
        if args.total_steps is not None:
            fields["budget"]["total_steps"] = args.total_steps
        if args.world_count is not None:
            fields["budget"]["world_count"] = args.world_count
    if args.motion_speed is not None:
        fields["motion_speed"] = args.motion_speed
    spec = SingleSampleSpec.model_validate(fields)
    observer_python = Path(getattr(args, "observer_python", None) or sys.executable).absolute()
    observer_module_root = Path(getattr(args, "observer_module_root", None)
                                or Path(__file__).resolve().parents[2]).resolve()
    swanlab_mode = getattr(args, "swanlab_mode", "local")
    host_hashes = host_source_hashes(observer_module_root)
    observer_options = {}
    for key in ("swanlog_root", "tensorboard_root", "swanlab_url", "tensorboard_url"):
        value = getattr(args, key, None)
        if value is not None:
            observer_options[key] = str(Path(value).resolve()) if key.endswith("_root") else str(value)
    output.mkdir(parents=True, exist_ok=False)
    args.cache.mkdir(parents=True, exist_ok=True)
    spec_path = output / "spec.json"
    _write(spec_path, spec.model_dump(mode="json"))
    command = docker_command(spec, spec_path, output, repository, args.legacy_inputs.resolve(), args.cache.resolve())
    manifest = {"schema": "nvidia_v2d.single_sample_launch.v1", "status": "prepared",
                "spec_sha256": sha256(spec_path), "support_sha256": spec.support_sha256,
                "source_sha256": {name: sha256(repository / "experiments/nvidia_v2d" / name) for name in SOURCE_FILES},
                "host_source_sha256": host_hashes,
                "observability": {"required": True, "swanlab": True, "tensorboard": True,
                                  "mode": swanlab_mode, "python": str(observer_python),
                                  "module_root": str(observer_module_root), **observer_options},
                "legacy_inputs": str(args.legacy_inputs.resolve()), "argv": command,
                "training_overrides": physical_training_overrides(spec, Path("/run"))}
    _write(output / "launch-manifest.json", manifest)
    launch_command = [sys.executable,
        str(observer_module_root / "experiments/nvidia_v2d/policy_logged_launch.py"),
        "--manifest", str(output / "launch-manifest.json"),
        "--observer-python", str(observer_python),
        "--observer-module-root", str(observer_module_root), "--swanlab-mode", swanlab_mode]
    (output / "run.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\nexec "
                                   + shlex.join(launch_command) + "\n")
    (output / "run.sh").chmod(0o755)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "train", "evaluate"))
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support-usda", type=Path)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--legacy-inputs", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--world-count", type=int)
    parser.add_argument("--motion-speed", type=float)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--metrics-history", type=Path)
    parser.add_argument("--observer-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--observer-module-root", type=Path)
    parser.add_argument("--swanlab-mode", choices=("local", "online"), default="local")
    parser.add_argument("--swanlog-root", type=Path)
    parser.add_argument("--tensorboard-root", type=Path)
    parser.add_argument("--swanlab-url")
    parser.add_argument("--tensorboard-url")
    args = parser.parse_args()
    if args.mode == "prepare":
        if not all((args.support_usda, args.legacy_inputs, args.cache)):
            parser.error("prepare requires --support-usda, --legacy-inputs and --cache")
        print(json.dumps(prepare(args), ensure_ascii=False, indent=2))
        return
    if args.spec is None:
        parser.error("--spec is required")
    spec = SingleSampleSpec.model_validate_json(args.spec.read_text())
    if args.mode == "run":
        raise SystemExit(0 if run(spec, args.spec, args.output)["status"] == "succeeded" else 1)
    if args.mode == "train":
        train(spec, args.output)
    elif args.mode == "evaluate":
        if args.checkpoint is None:
            parser.error("evaluate requires --checkpoint")
        evaluate(spec, args.checkpoint, args.output, args.metrics_history)


if __name__ == "__main__":
    main()
