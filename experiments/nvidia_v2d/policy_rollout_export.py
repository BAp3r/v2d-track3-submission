"""把 FlashCHORD 的实际物体轨迹导出到查看器；从不以参考位姿冒充执行。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from experiments.nvidia_v2d.policy_trajectory import validate_trajectory
from experiments.nvidia_v2d.policy_view_data import Scene


def verified_assistance(
    receipt: Path | None, metadata: dict[str, str], trajectory_sha256: str, frame_count: int
) -> Literal["none", "unknown"]:
    if receipt is None:
        return "unknown"
    record = json.loads(receipt.read_text())
    evidence = record.get("evaluation_evidence", {})
    observations = evidence.get("observed_control_scales", [])
    if len(observations) != frame_count or any(
        row.get("voc_scale_max_abs") != 0.0 or row.get("applied_object_control_scale_max_abs") != 0.0
        for row in observations
    ):
        raise ValueError("缺少逐动作无辅助控制量观测")
    # The first receipt predates the summary aliases; derive only from all actual observations.
    evidence.setdefault("voc_scale_max", 0.0)
    evidence.setdefault("applied_control_scale_max", 0.0)
    evidence.setdefault("action_step_count", len(observations))
    expected = {
        "assistance": "none",
        "voc_scale_max": 0.0,
        "applied_control_scale_max": 0.0,
        "action_step_count": frame_count,
        "trajectory_sha256": trajectory_sha256,
        "start_frame": 0,
        "motion_end_frame": -1,
    }
    if record.get("status") != "succeeded" or any(evidence.get(k) != v for k, v in expected.items()):
        raise ValueError("缺少完整且成功的无辅助实际执行凭证")
    for key in ("checkpoint_sha256", "eval_code_sha256", "eval_script_sha256"):
        value = evidence.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("执行凭证缺少策略或评估代码SHA")
        if value != metadata.get("flash_chord." + key):
            raise ValueError("执行凭证与轨迹的策略或评估代码SHA不一致")
    return "none"


def export_rollout(
    trajectory: Path,
    reference_scene: Path,
    output: Path,
    environment: int = 0,
    evaluation_receipt: Path | None = None,
) -> dict:
    """固定展示环境编号，不挑选最优环境；保留无效帧标记和全时间轴。"""
    if output.exists():
        raise FileExistsError("导出目录必须为新目录")
    hashes = {
        str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (trajectory, reference_scene)
    }
    if evaluation_receipt:
        hashes[str(evaluation_receipt.resolve())] = hashlib.sha256(
            evaluation_receipt.read_bytes()
        ).hexdigest()
    template = Scene.model_validate_json(reference_scene.read_bytes())
    if template.kind != "kinematic_reference":
        raise ValueError("必须提供运动学参考网格和时间轴")
    table = pq.read_table(trajectory)
    meta = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    expected = {
        "flash_chord.schema": "flash_chord_object_trajectories_v1",
        "flash_chord.position_unit": "meter",
        "flash_chord.quaternion_order": "xyzw",
        "flash_chord.start_frame": "0",
        "flash_chord.step_count": str(len(template.timestamps_s)),
    }
    if any(meta.get(k) != value for k, value in expected.items()):
        raise ValueError("执行轨迹的单位、格式或完整序列契约不符")
    fps = float(meta["flash_chord.control_fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("执行频率无效")
    validation = validate_trajectory(
        trajectory, len(template.timestamps_s), int(meta["flash_chord.world_count"]), fps
    )
    assistance = verified_assistance(
        evaluation_receipt, meta, validation["sha256"], len(template.timestamps_s)
    )
    rows = [r for r in table.to_pylist() if r["environment_id"] == environment]
    names = {obj.name for obj in template.objects}
    if {r["object_body_name"] for r in rows} != names:
        raise ValueError("执行与参考物体不一致")
    frame_count = len(template.timestamps_s)
    valid = np.ones(frame_count, dtype=bool)
    objects = []
    for obj in template.objects:
        selected = sorted(
            (r for r in rows if r["object_body_name"] == obj.name), key=lambda row: row["step"]
        )
        if [r["step"] for r in selected] != list(range(frame_count)):
            raise ValueError("执行包含缺帧、重复帧或截短序列")
        if [r["reference_frame"] for r in selected] != list(range(frame_count)):
            raise ValueError("执行不是从首帧开始的连续完整序列")
        if not np.allclose([r["time_s"] for r in selected], template.timestamps_s, atol=1e-7):
            raise ValueError("执行时间与原视频不同步")
        if not np.allclose(np.arange(frame_count) / fps, template.timestamps_s, atol=1e-7):
            raise ValueError("执行频率改变了原视频时间轴")
        reference_positions = np.array(
            [[r[f"reference_position_{axis}"] for axis in "xyz"] for r in selected]
        )
        reference_quaternions = np.array(
            [[r[f"reference_quaternion_{axis}"] for axis in "xyzw"] for r in selected]
        )
        reference_poses = np.asarray(obj.poses).reshape(frame_count, 4, 4)
        if not np.allclose(
            reference_positions, reference_poses[:, :3, 3], atol=1e-5
        ) or not np.allclose(
            Rotation.from_quat(reference_quaternions).as_matrix(),
            reference_poses[:, :3, :3],
            atol=1e-5,
        ):
            raise ValueError("仿真与显示场景的世界坐标不一致，必须先提供显式变换")
        positions = np.array([[r[f"achieved_position_{axis}"] for axis in "xyz"] for r in selected])
        quaternions = np.array(
            [[r[f"achieved_quaternion_{axis}"] for axis in "xyzw"] for r in selected]
        )
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(quaternions)):
            raise ValueError("执行位姿包含非有限值，保留原轨迹但拒绝发布场景")
        if not np.allclose(np.linalg.norm(quaternions, axis=1), 1, atol=1e-4):
            raise ValueError("执行四元数未归一化")
        poses = np.tile(np.eye(4), (frame_count, 1, 1))
        poses[:, :3, :3] = Rotation.from_quat(quaternions).as_matrix()
        poses[:, :3, 3] = positions
        valid &= np.array([r["environment_valid"] for r in selected], dtype=bool)
        objects.append(obj.model_copy(update={"poses": poses.reshape(frame_count, 16).tolist()}))
    scene = Scene(
        kind="policy_rollout",
        timestamps_s=template.timestamps_s,
        hands=[],
        objects=objects,
        valid_frames=valid.tolist(),
        assistance=assistance,
    )
    payload = scene.model_dump_json().encode()
    if len(payload) > 24 * 1024**2:
        raise ValueError("场景超过查看器上限")
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha for p, sha in hashes.items()):
        raise ValueError("导出期间源文件发生变化")
    manifest = {
        "schema": "nvidia_v2d.policy_rollout_scene.v1",
        "status": "succeeded",
        "created_at": datetime.now(UTC).isoformat(),
        "environment_id": environment,
        "selection": "explicit_environment_not_best_result",
        "kind": "policy_rollout",
        "assistance": assistance,
        "frame_count": frame_count,
        "invalid_frame_count": int(np.count_nonzero(~valid)),
        "input_sha256": hashes,
        "inputs_unchanged": True,
        "trajectory_metadata": meta,
        "all_worlds_validation": validation,
        "pose_source": "achieved_only",
        "hands": "not_recorded_not_synthesized",
        "scene_scale_changed": False,
        "output_path": str((output / "scene.json").resolve()),
        "output_sha256": hashlib.sha256(payload).hexdigest(),
    }
    output.mkdir(parents=True)
    (output / "scene.json").write_bytes(payload)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--reference-scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment", type=int, default=0)
    parser.add_argument("--evaluation-receipt", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            export_rollout(
                args.trajectory,
                args.reference_scene,
                args.output,
                args.environment,
                args.evaluation_receipt,
            )
        )
    )


if __name__ == "__main__":
    main()
