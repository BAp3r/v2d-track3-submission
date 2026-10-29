"""Offline official-IK / user multi-tip comparison; inputs and hardware are read-only."""

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from experiments.nvidia_v2d.sharpa_calibration import (
    CHAIN_PAIRS,
    TIPS,
    calibrate_open_hand,
    require_timeline,
    validate_source_hand,
)
from experiments.nvidia_v2d.sharpa_multitip_kinematics import (
    SharpaLocalRobot,
    build_optimizer,
    reference_from_targets,
    solve_frame,
)
from experiments.nvidia_v2d.sharpa_multitip_surface import CollisionSurfaceSampler

BACKENDS = ("official-ik", "unidex-inspired-ik-v2")
ADAPTER_METADATA = {
    "name": "nvidia-v2d-sharpa-offline-retarget",
    "version": "1.0",
    "input_requirements": [
        "loaded ManoSharpaData parquet",
        "可信MANO形状、规范解剖腕轴与逐帧来源",
        "Sharpa MJCF和mesh",
    ],
    "supported_input_scopes": ["single_episode"],
    "supported_modes": list(BACKENDS),
    "output_schema": "nvidia_v2d.sharpa_retarget.v1",
}


class HandCalibrationInput(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True, allow_inf_nan=False)
    open_joints_m: list[list[float]]
    open_wrist_wxyz: list[float]
    joint_names: list[str]
    source: Literal["median_accepted_mano_shape_open_pose"]
    shape_betas: list[float]
    metric_hand_scale: float = Field(gt=0)
    valid: list[bool]
    source_code: list[int]


class CalibrationInput(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True, allow_inf_nan=False)
    schema_version: str = "1"
    source_frame_indices: list[int]
    timestamps_s: list[float]
    hands: dict[Literal["left", "right"], HandCalibrationInput]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _official(upstream: Path, side: str, xml: Path) -> Any:
    sys.path.insert(0, str(upstream / "robotic_grounding/source/robotic_grounding"))
    from robotic_grounding.retarget.hand_kinematics import SharpaHandKinematics

    return SharpaHandKinematics(
        side=side, robot_asset_path=str(xml), source_model="mano", frequency=200, max_iter=100
    )


def _retarget_side(
    row: dict[str, Any],
    hand: HandCalibrationInput,
    times: np.ndarray,
    side: str,
    backend: str,
    upstream: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    from scipy.spatial.transform import Rotation

    joints = np.asarray(row[f"mano_{side}_joints"], dtype=float)
    quats = np.asarray(row[f"mano_{side}_joints_wxyz"], dtype=float)
    validate_source_hand(joints, quats, hand.valid, hand.source_code, hand.joint_names)
    xml = (
        upstream
        / f"robotic_grounding/source/robotic_grounding/robotic_grounding/assets/xmls/sharpawave/{side}_sharpawave.xml"
    )
    robot = SharpaLocalRobot(xml, side)
    surface = CollisionSurfaceSampler(xml, robot.link_names)
    open_wrist = np.asarray(hand.open_wrist_wxyz, dtype=float)
    if open_wrist.shape != (4,) or not np.isclose(np.linalg.norm(open_wrist), 1, atol=1e-6):
        raise ValueError("规范展开手必须提供有效的 MANO 解剖腕轴")
    calibration = calibrate_open_hand(
        hand.open_joints_m,
        robot.points(robot.q0),
        side,
        source_open_wrist_rotation=Rotation.from_quat(open_wrist, scalar_first=True).as_matrix(),
    )
    optimizer = build_optimizer(robot, side) if backend == BACKENDS[1] else None
    official = _official(upstream, side, xml) if backend == BACKENDS[0] else None
    if (
        official is not None
        and list(official.robot_finger_joint_names.values()) != robot.dof_joint_names
    ):
        raise ValueError("官方 IK 与 multi-tip 的关节顺序不一致")
    task_ids = [0, 2, 4, 5, 8, 9, 12, 13, 16, 17, 20]
    task_names = [f"{side}_hand_C_MC", f"{side}_thumb_MCP_VL_site", f"{side}_thumb_tip_site"]
    for finger in ("index", "middle", "ring", "pinky"):
        task_names.extend([f"{side}_{finger}_MP_site", f"{side}_{finger}_tip_site"])
    fields: dict[str, Any] = {
        f"{side}_robot_finger_joint_names": robot.dof_joint_names,
        f"{side}_robot_frame_names": robot.link_names,
        f"{side}_robot_frame_task_names": task_names,
    }
    arrays: dict[str, list[Any]] = {
        key: []
        for key in (
            "raw_finger_joints",
            "robot_points_world",
            "target_points_world",
            "tip_errors_m",
            "pair_errors_m",
            "wrist_error_m",
            "solve_time_s",
            "world_surface_samples_m",
            "optimizer_status",
        )
    }
    for suffix in (
        "wrist_position",
        "wrist_wxyz",
        "finger_joints",
        "frames",
        "frame_task_errors",
        "num_optimization_iterations",
    ):
        fields[f"robot_{side}_{suffix}"] = []
    previous = robot.q0.copy()
    official_previous = None
    pair_indices = [(a, b) for a in range(5) for b in range(a + 1, 5)]
    for index in range(len(times)):
        started = time.perf_counter()
        wrist_rotation = Rotation.from_quat(quats[index, 0], scalar_first=True).as_matrix()
        targets, root, root_rotation = calibration.transform(joints[index], wrist_rotation)
        targets_world = targets @ root_rotation.T + root
        dt = times[index] - times[index - 1] if index else times[1] - times[0]
        if optimizer is not None:
            q, iterations = solve_frame(optimizer, reference_from_targets(targets), previous, dt)
            frames = robot.world_frames(q, root, root_rotation)
            status = int(optimizer.opt.last_optimize_result())
        else:
            assert official is not None
            if official_previous is None:
                official_previous = robot.full_q(previous)
                official_previous[:3] = root
                official_previous[3:7] = Rotation.from_matrix(root_rotation).as_quat()
            result = official.compute(
                joints[index],
                quats[index],
                source_to_robot_scale=calibration.scale,
                qpos=official_previous,
            )
            official_previous = result["q"]
            q = np.asarray(official_previous[7:], dtype=float)
            root = np.asarray(official_previous[:3], dtype=float)
            root_rotation = Rotation.from_quat(official_previous[3:7]).as_matrix()
            frames = robot.world_frames(q, root, root_rotation)
            iterations = result["num_optimization_iterations"]
            status = 5 if iterations >= 100 else 1
        limits = robot.joint_limits
        if (
            not np.isfinite(q).all()
            or np.any(q < limits[:, 0] - 1e-6)
            or np.any(q > limits[:, 1] + 1e-6)
        ):
            raise ValueError(f"{side} 第 {index} 帧无效或超关节限位")
        actual = frames[robot.point_indices, :3]
        tip_errors = np.linalg.norm(actual[TIPS] - targets_world[TIPS], axis=1)
        pair_errors = [
            abs(
                np.linalg.norm(actual[TIPS[b]] - actual[TIPS[a]])
                - np.linalg.norm(targets_world[TIPS[b]] - targets_world[TIPS[a]])
            )
            for a, b in pair_indices
        ]
        for key, value in {
            "raw_finger_joints": q,
            "robot_points_world": actual,
            "target_points_world": targets_world,
            "tip_errors_m": tip_errors,
            "pair_errors_m": pair_errors,
            "wrist_error_m": np.linalg.norm(actual[0] - joints[index, 0]),
            "solve_time_s": time.perf_counter() - started,
            "world_surface_samples_m": surface.world_points(frames),
            "optimizer_status": status,
        }.items():
            arrays[key].append(value)
        results = {
            "wrist_position": root.tolist(),
            "wrist_wxyz": Rotation.from_matrix(root_rotation).as_quat(scalar_first=True).tolist(),
            "finger_joints": q.tolist(),
            "frames": frames.tolist(),
            "frame_task_errors": [
                float(np.linalg.norm(frames[robot.get_link_index(name), :3] - targets_world[j]))
                for name, j in zip(task_names, task_ids, strict=True)
            ],
            "num_optimization_iterations": int(iterations),
        }
        for suffix, value in results.items():
            fields[f"robot_{side}_{suffix}"].append(value)
        previous = q
    converted = {key: np.asarray(value) for key, value in arrays.items()}
    converted["valid"] = np.asarray(hand.valid, dtype=bool)
    converted["source_code"] = np.asarray(hand.source_code, dtype=np.int16)
    converted["speeds_rad_s"] = (
        np.diff(converted["raw_finger_joints"], axis=0) / np.diff(times)[:, None]
    )
    summary = {
        "calibration": calibration.as_dict(),
        "xml_sha256": sha256(xml),
        "surface_sampling": surface.metadata,
        "tip_error_p95_m": float(np.percentile(converted["tip_errors_m"], 95)),
        "tip_error_mean_m": float(np.mean(converted["tip_errors_m"])),
        "pair_error_p95_m": float(np.percentile(converted["pair_errors_m"], 95)),
        "wrist_error_max_m": float(np.max(converted["wrist_error_m"])),
        "solve_time_p95_s": float(np.percentile(converted["solve_time_s"], 95)),
        "speed_p95_rad_s": float(np.percentile(np.abs(converted["speeds_rad_s"]), 95)),
        "joint_limit_violations": 0,
        "optimization_budget_exhausted_count": int(np.sum(converted["optimizer_status"] == 5)),
        "frame_count": len(times),
    }
    return fields, converted, summary


def run_retarget(
    input_parquet: Path, calibration_path: Path, upstream: Path, output: Path, backend: str
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scipy.spatial.transform import Rotation

    if backend not in BACKENDS:
        raise ValueError("未知重定向后端")
    if output.exists():
        raise FileExistsError("重定向输出必须为新目录")
    source_hash, calibration_hash = sha256(input_parquet), sha256(calibration_path)
    source_table = pq.ParquetFile(input_parquet).read()
    if source_table.num_rows != 1:
        raise ValueError("一次只允许一个完整 episode")
    row = source_table.to_pylist()[0]
    required = {
        f"robot_{side}_{suffix}"
        for side in ("left", "right")
        for suffix in (
            "wrist_position",
            "wrist_wxyz",
            "finger_joints",
            "frames",
            "frame_task_errors",
            "num_optimization_iterations",
        )
    }
    required.update(
        f"{side}_robot_{suffix}"
        for side in ("left", "right")
        for suffix in ("finger_joint_names", "frame_names", "frame_task_names")
    )
    if not required.issubset(source_table.schema.names):
        raise ValueError("输入缺少 ManoSharpaData 的机器人字段，不能静默丢弃重定向结果")
    calibration = CalibrationInput.model_validate_json(calibration_path.read_text())
    if set(calibration.hands) != {"left", "right"}:
        raise ValueError("必须提供两侧显式来源与规范手")
    count = len(row["mano_left_joints"])
    times = require_timeline(
        calibration.timestamps_s, calibration.source_frame_indices, count, row["fps"]
    )
    output.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema": "nvidia_v2d.sharpa_retarget.v1",
        "adapter": ADAPTER_METADATA,
        "backend": backend,
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "log_path": str(output / "run.log"),
        "output_manifest_path": str(output / "manifest.json"),
        "input_manifest_path": str(calibration_path),
        "backend_version": "sharpa-offline-v1",
        "user_solver_provenance": json.loads(
            (Path(__file__).parent / "vendor_dex_multitip/PROVENANCE.json").read_text()
        ),
        "input_parquet": str(input_parquet),
        "input_sha256": source_hash,
        "calibration_sha256": calibration_hash,
        "fps": row["fps"],
        "frame_count": count,
        "timestamps_s": times.tolist(),
        "kind": "kinematic_reference",
        "physical_rollout": False,
        "post_filter": None,
        "velocity_limiter": None,
        "hands": {},
    }
    _json(output / "manifest.json", manifest)
    log_path = output / "run.log"
    log_path.write_text(f"{manifest['created_at']} 开始 {backend}，{count} 帧\n")
    try:
        diagnostics: dict[str, np.ndarray] = {"timestamps_s": times}
        preview: dict[str, Any] = {
            "schema": "nvidia_v2d.sharpa_retarget_preview.v1",
            "backend": backend,
            "kind": "kinematic_reference",
            "timestamps_s": times.tolist(),
            "edges": list(CHAIN_PAIRS) + [(0, x) for x in (5, 9, 13, 17)],
        }
        for side in ("left", "right"):
            fields, arrays, summary = _retarget_side(
                row, calibration.hands[side], times, side, backend, upstream
            )
            row.update(fields)
            diagnostics.update({f"{side}_{key}": value for key, value in arrays.items()})
            manifest["hands"][side] = summary
            preview[f"hands_{side}"] = arrays["robot_points_world"].tolist()
            preview[f"target_{side}"] = arrays["target_points_world"].tolist()
            preview[f"valid_{side}"] = arrays["valid"].tolist()
            preview[f"source_code_{side}"] = arrays["source_code"].tolist()
            with log_path.open("a") as log:
                log.write(f"{datetime.now(UTC).isoformat()} {side} 完成 {count} 帧\n")
        positions = np.asarray(row["object_body_position"], dtype=float)
        quats = np.asarray(row["object_body_wxyz"], dtype=float)
        poses = np.tile(np.eye(4), (*positions.shape[:2], 1, 1))
        poses[..., :3, 3] = positions
        poses[..., :3, :3] = (
            Rotation.from_quat(quats.reshape(-1, 4), scalar_first=True)
            .as_matrix()
            .reshape(*positions.shape[:2], 3, 3)
        )
        preview.update(
            object_names=row["object_body_names"],
            object_poses_world=poses.tolist(),
            object_mesh_paths=row["object_mesh_paths"],
        )
        if sha256(input_parquet) != source_hash or sha256(calibration_path) != calibration_hash:
            raise ValueError("读取过程中输入发生变化")
        pq.write_table(
            pa.Table.from_pylist([row], schema=source_table.schema),
            output / "data.parquet",
            compression="zstd",
        )
        np.savez_compressed(output / "diagnostics.npz", **diagnostics)  # type: ignore[arg-type]
        _json(output / "preview.json", preview)
        manifest.update(
            status="completed",
            result_sha256=sha256(output / "data.parquet"),
            source_unchanged=True,
            failure_reason=None,
            completed_at=datetime.now(UTC).isoformat(),
        )
    except Exception as exc:
        manifest.update(
            status="failed",
            failure_reason=f"{type(exc).__name__}: {exc}",
            completed_at=datetime.now(UTC).isoformat(),
        )
        with log_path.open("a") as log:
            log.write(f"{manifest['completed_at']} 失败：{manifest['failure_reason']}\n")
        _json(output / "manifest.json", manifest)
        raise
    _json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Sharpa 离线重定向对照，不连接硬件")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    args = parser.parse_args()
    result = run_retarget(args.input, args.calibration, args.upstream, args.output, args.backend)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
