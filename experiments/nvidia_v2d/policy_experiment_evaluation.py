"""调用官方评估器，并在每次策略动作前读取真实VOC状态作为无辅助证据。"""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def evaluate_with_evidence(
    checkpoint: Path, worlds: int, output: Path, *, metrics_history: Path | None = None,
    scene_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    import numpy as np
    import pyarrow.parquet as pq
    from flash_chord.evaluation import harness

    original = harness.record_object_pose_rollout
    observed: list[dict[str, float]] = []

    def audited_rollout(env: Any, jax_env: Any, policy_action: Any, *args: Any, **kwargs: Any) -> Any:
        def audited_action(*action_args: Any, **action_kwargs: Any) -> Any:
            voc = np.asarray(env.voc_scale.numpy())
            applied = np.asarray(env.object_control_scale.numpy())
            if (
                not np.isfinite(voc).all()
                or not np.isfinite(applied).all()
                or np.any(voc != 0)
                or np.any(applied != 0)
            ):
                raise ValueError("评估策略动作前仍存在物体辅助或非有限控制尺度")
            observed.append(
                {
                    "voc_scale_max_abs": float(np.max(np.abs(voc))),
                    "applied_object_control_scale_max_abs": float(np.max(np.abs(applied))),
                }
            )
            return policy_action(*action_args, **action_kwargs)

        return original(env, jax_env, audited_action, *args, **kwargs)

    trajectory = output / "unassisted.object_trajectories.parquet"
    harness.record_object_pose_rollout = audited_rollout
    try:
        outcome = harness.evaluate_checkpoint(
            checkpoint,
            world_count=worlds,
            start_frame=0,
            reset_mode="explicit",
            motion_start_frame=0,
            motion_end_frame=-1,
            object_trajectories_output=trajectory,
            eval_script=Path(__file__).resolve(),
        )
    finally:
        harness.record_object_pose_rollout = original
    harness.write_evaluation_report(output / "unassisted.metrics.json", outcome)
    if len(observed) != outcome.step_count:
        raise ValueError("未对完整序列的每次策略动作核查VOC")
    metadata = {
        k.decode(): v.decode() for k, v in pq.ParquetFile(trajectory).schema_arrow.metadata.items()
    }
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint_sha != metadata["flash_chord.checkpoint_sha256"]:
        raise ValueError("轨迹引用的checkpoint与实际评估不一致")
    reference_sha = hashlib.sha256(Path(outcome.parquet).read_bytes()).hexdigest()
    # Explicit scene provenance includes inputs outside the checkpoint (e.g. a
    # finite support USDA resolver). Keep it nested so callers cannot override
    # measured evaluator/reset/assistance fields.
    scene = dict(scene_context or {})
    json.dumps(scene, allow_nan=False)
    evidence = {
        "schema": "nvidia_v2d.policy_evaluation_evidence.v1",
        "assistance": "none",
        "voc_scale": 0.0,
        "applied_object_control_scale": 0.0,
        "voc_scale_max": max(row["voc_scale_max_abs"] for row in observed),
        "applied_control_scale_max": max(
            row["applied_object_control_scale_max_abs"] for row in observed
        ),
        "action_step_count": len(observed),
        "checkpoint_sha256": checkpoint_sha,
        "reference_parquet_sha256": reference_sha,
        "scene_context": scene,
        "trajectory_sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
        "eval_code_sha256": metadata["flash_chord.eval_code_sha256"],
        "eval_script_sha256": metadata["flash_chord.eval_script_sha256"],
        "eval_script_path": str(Path(__file__).resolve()),
        "official_harness_sha256": hashlib.sha256(Path(harness.__file__).read_bytes()).hexdigest(),
        "motion_start_frame": 0,
        "motion_end_frame": -1,
        "start_frame": 0,
        "step_count": outcome.step_count,
        "world_count": worlds,
        "observed_policy_action_count": len(observed),
        "observed_control_scales": observed,
        "reset_mode": "explicit",
        "policy_deterministic": True,
    }
    (output / "evaluation-evidence.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    if metrics_history is not None:
        from experiments.nvidia_v2d.policy_metric_logging import JsonMetricLogger

        # Curves only join checkpoints evaluated under the same source/criteria/reset.
        series = {
            "reference_parquet": outcome.parquet,
            "reference_parquet_sha256": reference_sha,
            "scene_context": scene,
            "world_count": worlds,
            "reset_mode": "explicit",
            "assistance": "none",
            "step_count": outcome.step_count,
            "object_position_threshold_m": outcome.object_position_threshold_m,
            "object_orientation_threshold_rad": outcome.object_orientation_threshold_rad,
            "eval_code_sha256": evidence["eval_code_sha256"],
            "eval_script_sha256": evidence["eval_script_sha256"],
        }
        context = {
            **series,
            "series_id": hashlib.sha256(json.dumps(series, sort_keys=True).encode()).hexdigest(),
            "checkpoint_sha256": checkpoint_sha,
            "evidence_path": str(output / "evaluation-evidence.json"),
            "report_path": str(output / "unassisted.metrics.json"),
        }
        scalars = harness.evaluation_scalars(outcome)
        scalars.update({
            "evaluation/reference_tracking_success_rate": outcome.reference_tracking.success_rate,
            "evaluation/reference_end_fraction": outcome.reference_tracking.reference_end_fraction,
            "evaluation/non_finite_worlds": float(len(outcome.non_finite_worlds)),
        })
        logger = JsonMetricLogger(metrics_history, resume=metrics_history.exists(), context=context)
        try:
            logger.log(scalars, step=outcome.environment_steps)
        finally:
            logger.close()
    return evidence
