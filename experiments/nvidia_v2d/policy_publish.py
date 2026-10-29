"""只读核对正式六组作业，幂等发布环境0实际执行；从不启动训练。"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from experiments.nvidia_v2d.policy_experiment import ExperimentSpec, training_argv
from experiments.nvidia_v2d.policy_experiment_sweep import SweepPlan, SweepRun
from experiments.nvidia_v2d.policy_gate import verify_gate
from experiments.nvidia_v2d.policy_rollout_export import export_rollout
from experiments.nvidia_v2d.policy_run import FileStamp, RunRecord, stamp
from experiments.nvidia_v2d.policy_view_data import PolicyEpisode, Scene

EPISODE = "track_3-public-episode-000044"
BACKENDS = ("official", "multitip")
SEEDS = (42, 43, 44)
# Immutable v1.0 module from commit 72b4043, used by the six frozen runs.
# Only this verified source is allowed to omit the later reset override.
LEGACY_EXPERIMENT_SHA256 = "bbf2bd44e6ace7ea5e3ae273e101ca9d079c03844b95e1fcdc70974db13f3a45"
RESET_OVERRIDE = "reset.voc_decay_steps=0"
REQUIRED = {"experiment.json", "unassisted.metrics.json", "unassisted.object_trajectories.parquet"}
STATUS_ZH = {
    "pending": "待运行",
    "running": "运行中",
    "succeeded": "已验证完成",
    "failed": "失败",
    "timed_out": "超时",
    "blocked": "凭据核对未通过",
}


def _training_source_sha(plan: SweepPlan) -> str:
    expected = plan.repository / "experiments/nvidia_v2d/policy_experiment.py"
    matches = [path for path in plan.source_sha256 if Path(path).name == expected.name]
    if matches != [str(expected)]:
        raise ValueError("冻结训练源码必须唯一绑定到运行仓库中的适配器")
    return plan.source_sha256[str(expected)]


def _versioned_training_argv(spec: ExperimentSpec, source_sha: str) -> list[str]:
    """Compare full argv; the registered v1.0 exception is exactly one argument.

    The caller has already verified all frozen source bytes against the plan.
    Never execute historical source, trust a claimed version, or filter arbitrary
    reset/seed/budget/reward overrides from the recorded argv.
    """
    expected = training_argv(spec)
    if expected.count(RESET_OVERRIDE) != 1:
        raise ValueError("当前发布器参数契约必须明确禁用重置后的辅助回升")
    if source_sha == LEGACY_EXPERIMENT_SHA256:
        expected.remove(RESET_OVERRIDE)
    return expected


def _plain(path: Path) -> Path:
    path = path.absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("发布路径不能经过符号链接")
    return path


def _snapshot(path: Path, bound: dict[str, str]) -> Any:
    snapshot = stamp(_plain(path))
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != snapshot.sha256:
        raise ValueError("读取时凭据发生变化，请重新发布")
    bound[str(path.absolute())] = snapshot.sha256
    return json.loads(content)


def _sha(path: Path, bound: dict[str, str], expected: str | None = None) -> str:
    actual = stamp(_plain(path)).sha256
    if expected is not None and actual != expected:
        raise ValueError(f"文件SHA不一致：{path.name}")
    bound[str(path.absolute())] = actual
    return actual


def _unchanged(bound: dict[str, str]) -> None:
    if any(stamp(Path(path)).sha256 != expected for path, expected in bound.items()):
        raise ValueError("发布期间来源变化；旧索引保持不动，请重新发布")


def _verify_stamps(rows: Any, bound: dict[str, str]) -> dict[str, FileStamp]:
    if not isinstance(rows, list):
        raise ValueError("文件清单必须为数组")
    result = {}
    for raw in rows:
        item = FileStamp.model_validate(raw)
        path = _plain(Path(item.path))
        current = stamp(path)
        if str(path) != item.path or current != item:
            raise ValueError(f"清单文件的SHA、大小或时间戳不符：{path.name}")
        if item.path in result:
            raise ValueError("文件清单重复条目")
        result[item.path] = item
        bound[item.path] = item.sha256
    return result


def _verified_success(
    run: SweepRun, spec: ExperimentSpec, plan: SweepPlan, record: RunRecord, bound: dict[str, str]
) -> tuple[dict[str, Any], dict[str, float | None]]:
    output = run.output
    if (
        record.status != "succeeded"
        or record.exit_code != 0
        or not record.finished_at
        or record.input_unchanged is not True
    ):
        raise ValueError("主作业未完整成功或未确认输入不变")
    if (
        record.spec.phase != "training"
        or record.spec.episode_id != EPISODE
        or record.spec.backend != run.backend
        or record.spec.retarget_parquet != spec.parquet
        or record.spec.gate != spec.gate
        or record.spec.cwd != plan.repository
    ):
        raise ValueError("作业规格与冻结的视频、后端或输入不一致")
    if record.input_manifest_path != str(
        output / "input-manifest.json"
    ) or record.output_manifest_path != str(output / "output-manifest.json"):
        raise ValueError("输入/输出清单不属于本作业")
    inputs = _verify_stamps(_snapshot(output / "input-manifest.json", bound), bound)
    outputs = _verify_stamps(_snapshot(output / "output-manifest.json", bound), bound)
    expected_outputs = {str(output / name) for name in REQUIRED}
    if (
        set(record.spec.required_outputs) != REQUIRED
        or set(outputs) != expected_outputs
        or len(record.outputs) != len(REQUIRED)
    ):
        raise ValueError("正式作业输出清单缺项或越界")
    if {item.path: item for item in record.outputs} != outputs:
        raise ValueError("run与输出清单的SHA绑定不同")
    required_inputs = {
        str(run.spec_path),
        str(spec.parquet),
        str(spec.gate),
        *map(str, record.spec.inputs),
        *plan.source_sha256,
    }
    if not required_inputs.issubset(inputs):
        raise ValueError("输入清单未绑定冻结配置或运行源码")
    if inputs[str(run.spec_path)].sha256 != run.spec_sha256:
        raise ValueError("作业输入并非冻结配置")
    for path, expected in plan.source_sha256.items():
        if inputs[path].sha256 != expected:
            raise ValueError("运行源码与冻结清单不同")
    gate = verify_gate(spec.gate, backend=run.backend, parquet=spec.parquet, episode_id=EPISODE)
    if gate["frame_count"] != 249:
        raise ValueError("只允许249帧public44正式对照")
    if not set(gate["verified_files"]).issubset(inputs):
        raise ValueError("输入清单缺少gate依赖")
    for path, sha in gate["verified_files"].items():
        if inputs[path].sha256 != sha:
            raise ValueError("gate与输入清单的SHA不同")
    experiment = _snapshot(output / "experiment.json", bound)
    if (
        experiment.get("schema") != "nvidia_v2d.policy_experiment.v1"
        or experiment.get("status") != "succeeded"
        or experiment.get("adapter_source_unchanged") is not True
        or experiment.get("spec") != spec.model_dump(mode="json")
    ):
        raise ValueError("实验凭据不是冻结配置的完整成功结果")
    if experiment.get("reference") != {"frames": 249, "fps": 20.0, "object_names": ["white_pot"]}:
        raise ValueError("实验来源不是public44完整时间轴及物体")
    expected_code = {Path(path).name: sha for path, sha in plan.source_sha256.items()}
    if experiment.get("adapter_code_sha256") != expected_code:
        raise ValueError("实验源码SHA与冻结清单不同")
    stages = experiment.get("stages", [])
    train = [row for row in stages if row.get("name") == "train"]
    evaluate = [row for row in stages if row.get("name") == "evaluate"]
    if (
        len(train) != 1
        or len(evaluate) != 1
        or any(row.get("status") != "succeeded" or row.get("exit_code") != 0 for row in stages)
    ):
        raise ValueError("训练/评估阶段未完整成功")
    if train[0].get("argv", [])[2:] != run.frozen_training_argv:
        raise ValueError("实际训练参数与冻结参数不同")
    checkpoint_meta = (
        experiment.get("resume_checkpoint")
        if spec.verify_resume
        else experiment.get("train_checkpoint")
    )
    if (
        not isinstance(checkpoint_meta, dict)
        or int(checkpoint_meta.get("environment_steps", -1)) != plan.budget.total_steps
        or int(checkpoint_meta.get("global_update_step", 0)) <= 0
        or float(checkpoint_meta.get("voc_scale", 1)) != 0
    ):
        raise ValueError("没有完成冻结步数、更新尝试与目标VOC=0课程阶段")
    metrics = _snapshot(output / "unassisted.metrics.json", bound)
    if experiment.get("metrics") != metrics:
        raise ValueError("实验内指标与已绑定指标文件不同")
    if metrics.get("environment_steps") != plan.budget.total_steps or metrics.get("parquet") != str(
        spec.parquet
    ):
        raise ValueError("指标不是完整预算或同一重定向输入")
    for key, value in {
        "step_count": 249,
        "world_count": spec.evaluation_worlds,
        "sim_control_fps": 20.0,
        "start_frame": 0,
        "motion_start_frame": 0,
        "motion_end_frame": -1,
    }.items():
        if metrics.get(key) != value:
            raise ValueError("评估指标存在截断、变速或环境数量不符")
    checkpoint = Path(metrics["checkpoint"])
    if not checkpoint.is_relative_to("/run") or ".." in checkpoint.parts:
        raise ValueError("评估策略路径越界")
    actual_checkpoint = output / checkpoint.relative_to("/run")
    if not isinstance(metrics.get("checkpoint_sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", metrics["checkpoint_sha256"]
    ):
        raise ValueError("指标缺少完整策略SHA")
    checkpoint_sha = _sha(actual_checkpoint, bound, metrics["checkpoint_sha256"])
    trajectory_sha = outputs[str(output / "unassisted.object_trajectories.parquet")].sha256
    if (
        experiment.get("trajectory_validation", {}).get("sha256") != trajectory_sha
        or experiment.get("evaluation_evidence", {}).get("trajectory_sha256") != trajectory_sha
    ):
        raise ValueError("轨迹与成功评估凭据的SHA不同")
    evidence = experiment["evaluation_evidence"]
    if evidence.get("checkpoint_sha256") != checkpoint_sha or evidence.get(
        "eval_script_sha256"
    ) != expected_code.get("policy_experiment_evaluation.py"):
        raise ValueError("评估策略或评估适配器来源不一致")
    flat = {
        key: float(value)
        for key, value in metrics.get("metrics", {}).items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }
    flat.update(
        training_environment_steps=float(metrics["environment_steps"]),
        evaluation_world_count=float(spec.evaluation_worlds),
    )
    for key, value in metrics.get("reference_tracking", {}).items():
        if isinstance(value, int | float) and not isinstance(value, bool):
            flat["reference_tracking_" + key] = float(value)
    if any(not math.isfinite(value) for value in flat.values()):
        raise ValueError("实际指标包含非有限数值")
    return experiment, flat


def _reuse_export(
    output: Path, trajectory: Path, reference: Path, receipt: Path, bound: dict[str, str]
) -> dict[str, Any]:
    manifest = _snapshot(output / "manifest.json", bound)
    expected = {str(path.resolve()): _sha(path, bound) for path in (trajectory, reference, receipt)}
    if any(
        manifest.get(key) != value
        for key, value in {
            "schema": "nvidia_v2d.policy_rollout_scene.v1",
            "status": "succeeded",
            "environment_id": 0,
            "kind": "policy_rollout",
            "assistance": "none",
            "frame_count": 249,
            "selection": "explicit_environment_not_best_result",
            "pose_source": "achieved_only",
            "inputs_unchanged": True,
            "scene_scale_changed": False,
        }.items()
    ):
        raise ValueError("已有执行导出不符合固定环境0、无辅助及完整序列约定")
    if manifest.get("input_sha256") != expected or manifest.get("output_path") != str(
        (output / "scene.json").resolve()
    ):
        raise ValueError("已有执行导出绑定的是不同输入")
    if not isinstance(manifest.get("output_sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", manifest["output_sha256"]
    ):
        raise ValueError("已有执行导出缺少场景SHA")
    _sha(output / "scene.json", bound, manifest["output_sha256"])
    scene = Scene.model_validate_json((output / "scene.json").read_bytes())
    if scene.kind != "policy_rollout" or scene.assistance != "none" or len(scene.timestamps_s) != 249:
        raise ValueError("已有执行场景类型不一致")
    return manifest


def _reference_variants(
    viewer: Path, old_entry: dict[str, Any], bound: dict[str, str]
) -> list[dict[str, Any]]:
    variants = []
    for backend in BACKENDS:
        path = viewer / "scenes-002" / f"{backend}.json"
        sha = _sha(path, bound)
        scene = Scene.model_validate_json(path.read_bytes())
        if (
            scene.kind != "kinematic_reference"
            or len(scene.timestamps_s) != 249
            or any(abs(t - i / 20) > 1e-7 for i, t in enumerate(scene.timestamps_s))
            or {obj.name for obj in scene.objects} != {"white_pot"}
        ):
            raise ValueError("参考场景不是public44米制完整时间轴")
        old = [row for row in old_entry.get("variants", []) if row.get("id") == backend]
        if len(old) > 1:
            raise ValueError("参考变体重复")
        if old:
            row = dict(old[0])
            if (
                row.get("kind") != "kinematic_reference"
                or row.get("scene_path") != f"scenes-002/{backend}.json"
                or row.get("sha256") != sha
            ):
                raise ValueError("既有参考索引的SHA或路径不一致")
        else:
            row = {"id": backend, "backend": backend, "metrics": {}}
        row.update(
            status="succeeded",
            kind="kinematic_reference",
            scene_path=f"scenes-002/{backend}.json",
            sha256=sha,
        )
        variants.append(row)
    return variants


def _atomic_index(path: Path, value: dict[str, Any]) -> bool:
    content = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    if path.exists() and path.read_bytes() == content:
        return False
    descriptor, name = tempfile.mkstemp(prefix=".policy-index-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def publish(run_root: Path, viewer_root: Path, sweep_root: Path | None = None) -> dict[str, Any]:
    """一次发布。保留六组状态、失败与所有评估环境指标，不等待或启动任何作业。"""
    runs = _plain(run_root)
    viewer = _plain(viewer_root)
    sweep = _plain(sweep_root or runs.parent / "policy-sweep-001")
    if (
        not runs.is_dir()
        or not viewer.is_dir()
        or viewer.is_relative_to(runs)
        or viewer.is_relative_to(sweep)
    ):
        raise ValueError("viewer必须独立于只读作业/计划目录")
    with (viewer / ".policy-publish.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        bound = {}
        plan_path = sweep / "frozen-plan.json"
        plan = SweepPlan.model_validate(_snapshot(plan_path, bound))
        state = _snapshot(sweep / "sweep-state.json", bound)
        if (
            state.get("schema") != "nvidia_v2d.policy_sweep_state.v1"
            or state.get("plan_sha256") != bound[str(plan_path)]
        ):
            raise ValueError("扫参状态没有绑定当前冻结计划")
        if plan.budget.total_steps != 2097152 or plan.budget.world_count != 64:
            raise ValueError("正式对照必须是冻结的2,097,152步、64环境预算")
        if tuple(plan.budget.seeds) != SEEDS:
            raise ValueError("必须包含42、43、44全部种子")
        for path, sha in plan.source_sha256.items():
            _sha(Path(path), bound, sha)
        training_source_sha = _training_source_sha(plan)
        legacy_training = training_source_sha == LEGACY_EXPERIMENT_SHA256
        states = {}
        for row in state.get("runs", []):
            key = (row["backend"], row["seed"])
            if key in states or key not in {(b, s) for b in BACKENDS for s in SEEDS}:
                raise ValueError("扫参状态重复或包含计划外组合")
            states[key] = row
        index_path = viewer / "policy-index.json"
        old = _snapshot(index_path, bound) if index_path.exists() else {"episodes": {}}
        entry = old.get("episodes", {}).get(EPISODE, {})
        variants = _reference_variants(viewer, entry, bound)
        for run in plan.runs:
            expected_output = runs / f"ab-{run.backend}-seed{run.seed}-001"
            if run.output != expected_output or run.spec_path.parent != sweep / "specs":
                raise ValueError("冻结作业路径与指定发布根不符")
            spec = ExperimentSpec.model_validate(_snapshot(run.spec_path, bound))
            if (
                bound[str(run.spec_path)] != run.spec_sha256
                or spec.backend != run.backend
                or spec.seed != run.seed
                or spec.budget != plan.budget
                or spec.image != plan.image
                or _versioned_training_argv(spec, training_source_sha) != run.frozen_training_argv
            ):
                raise ValueError("冻结配置/预算/种子已改变")
            progress = states.get((run.backend, run.seed))
            variant = {
                "id": f"rollout-{run.backend}-{run.seed}",
                "backend": run.backend,
                "seed": run.seed,
                "kind": "policy_rollout",
                "assistance": "unknown",
                "status": "pending",
                "metrics": {},
            }
            if progress and progress.get("output") != str(run.output):
                raise ValueError("状态指向计划外作业目录")
            try:
                path = run.output / "run.json"
                record = RunRecord.model_validate(_snapshot(path, bound)) if path.exists() else None
                if record and (
                    record.spec.episode_id != EPISODE or record.spec.backend != run.backend
                ):
                    raise ValueError("主作业的视频或后端与冻结组不一致")
                status = progress.get("status") if progress else "pending"
                if record and record.status in ("failed", "timed_out", "interrupted"):
                    variant.update(
                        status="timed_out" if record.status == "timed_out" else "failed",
                        failure_reason=record.failure_reason or "作业未成功完成",
                    )
                elif status in ("failed", "failed_preserved"):
                    variant.update(
                        status="failed", failure_reason="扫参记录该组失败；保留既有结果且不追加预算"
                    )
                elif status == "succeeded":
                    if record is None:
                        raise ValueError("扫参宣称成功但没有主作业凭据")
                    _, metrics = _verified_success(run, spec, plan, record, bound)
                    trajectory = run.output / "unassisted.object_trajectories.parquet"
                    reference = viewer / "scenes-002" / f"{run.backend}.json"
                    receipt = run.output / "experiment.json"
                    export = viewer / f"rollout-ab-{run.backend}-{run.seed}-001"
                    if not export.exists():
                        export_rollout(
                            trajectory, reference, export, environment=0, evaluation_receipt=receipt
                        )
                    exported = _reuse_export(export, trajectory, reference, receipt, bound)
                    variant.update(
                        status="succeeded",
                        assistance="none",
                        scene_path=str((export / "scene.json").relative_to(viewer)),
                        sha256=exported["output_sha256"],
                        metrics=metrics,
                        publication={
                            "run_sha256": bound[str(path)],
                            "input_manifest_sha256": bound[str(run.output / "input-manifest.json")],
                            "output_manifest_sha256": bound[str(run.output / "output-manifest.json")],
                            "experiment_sha256": bound[str(receipt)],
                            "environment_id": 0,
                            "metrics_scope": "all_evaluation_worlds",
                        },
                    )
                elif status == "running" or (record and record.status == "running"):
                    variant["status"] = "running"
                elif status not in ("pending", None):
                    raise ValueError("未知扫参状态")
                elif record is not None:
                    raise ValueError("存在作业但扫参状态未绑定该组")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                variant.update(status="blocked", failure_reason=f"发布核对失败：{exc}")
            variants.append(variant)
        baseline = entry.get("publication", {}).get("preserved_reference_notes")
        if baseline is None:
            baseline = [
                note
                for note in entry.get("notes", [])
                if not any(word in note for word in ("短测", "1024", "正式三种子", "当前执行视图"))
            ]
        formal = variants[2:]
        incomplete = [
            f"{row['backend']} / 种子{row['seed']}：{STATUS_ZH[row['status']]}"
            for row in formal
            if row["status"] != "succeeded"
        ]
        notes = [
            f"正式完整预算：每组 {plan.budget.total_steps:,} 环境步（约{plan.budget.total_steps / 10000:.0f}万步），{plan.budget.world_count}个训练环境；两后端均保留42、43、44三个种子。",
            "执行视图固定展示环境0，不选择最优环境；指标来自全部评估环境。",
            ("尚未完成：" + "；".join(incomplete))
            if incomplete
            else "六组均已通过输入/输出SHA、完整预算及无辅助完整序列核验。",
            "这是机制验证对照；运动学参考、实际策略执行与抓持质量验收分别记录。",
            (
                "旧版1.0冻结训练未禁用重置后的辅助回升，不能称为严格无辅助训练阶段；最终完整评估另有逐动作零辅助证据。"
                if legacy_training
                else "当前训练参数明确冻结reset.voc_decay_steps=0；最终无辅助执行仍须逐动作凭据核验。"
            ),
            *baseline,
        ]
        updated = {
            **old,
            "episodes": {
                **old.get("episodes", {}),
                EPISODE: {
                    **entry,
                    "variants": variants,
                    "notes": notes,
                    "publication": {
                        "schema": "nvidia_v2d.policy_publication.v1",
                        "plan_sha256": bound[str(plan_path)],
                        "state_sha256": bound[str(sweep / "sweep-state.json")],
                        "training_argv_contract": (
                            "frozen-v1.0-default-reset" if legacy_training else "reset-disabled-v1.1"
                        ),
                        "training_source_sha256": training_source_sha,
                        "preserved_reference_notes": baseline,
                    },
                },
            },
        }
        PolicyEpisode.model_validate(updated["episodes"][EPISODE])
        _unchanged(bound)
        changed = _atomic_index(index_path, updated)
        return {
            "status": "published" if changed else "unchanged",
            "index": str(index_path),
            "completed_runs": sum(row["status"] == "succeeded" for row in formal),
            "variants": [
                {key: row[key] for key in ("id", "backend", "seed", "status")} for row in formal
            ],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--viewer-root", type=Path, required=True)
    parser.add_argument("--sweep-root", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            publish(args.run_root, args.viewer_root, args.sweep_root), ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
