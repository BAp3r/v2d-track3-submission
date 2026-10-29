"""Serialization for the independent rebuilt DexPilot-chain reference.

The row/diagnostic format deliberately matches the frozen two-route adapter;
only the solve step differs. Existing official/multi-tip source is untouched.
"""

import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from experiments.nvidia_v2d.sharpa_calibration import (
    CHAIN_PAIRS,
    TIPS,
    calibrate_open_hand,
    require_timeline,
    validate_source_hand,
)
from experiments.nvidia_v2d.sharpa_dexpilot import (
    BACKEND,
    DexPilotChainConfig,
    DexPilotChainOptimizer,
    primary_provenance,
)
from experiments.nvidia_v2d.sharpa_multitip import (
    CalibrationInput,
    HandCalibrationInput,
    _json,
    sha256,
)
from experiments.nvidia_v2d.sharpa_multitip_kinematics import SharpaLocalRobot
from experiments.nvidia_v2d.sharpa_multitip_surface import CollisionSurfaceSampler

ADAPTER_METADATA = {
    "name": "nvidia-v2d-sharpa-dexpilot-chain-rebuilt",
    "version": "1.0",
    "input_requirements": [
        "loaded ManoSharpaData",
        "same frozen MANO calibration",
        "Sharpa MJCF",
    ],
    "supported_input_scopes": ["single_episode"],
    "supported_modes": [BACKEND],
    "output_schema": "nvidia_v2d.sharpa_retarget.v1",
}


def _retarget_side(
    row: dict[str, Any],
    hand: HandCalibrationInput,
    times: np.ndarray,
    side: str,
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
    if open_wrist.shape != (4,) or not np.isclose(
        np.linalg.norm(open_wrist), 1, atol=1e-6
    ):
        raise ValueError("规范展开手必须提供有效的 MANO 解剖腕轴")
    calibration = calibrate_open_hand(
        hand.open_joints_m,
        robot.points(robot.q0),
        side,
        source_open_wrist_rotation=Rotation.from_quat(
            open_wrist, scalar_first=True
        ).as_matrix(),
    )
    optimizer = DexPilotChainOptimizer(robot)
    task_ids = [0, 2, 4, 5, 8, 9, 12, 13, 16, 17, 20]
    task_names = [
        f"{side}_hand_C_MC",
        f"{side}_thumb_MCP_VL_site",
        f"{side}_thumb_tip_site",
    ]
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
            "dexpilot_projected",
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
    pair_indices = [(a, b) for a in range(5) for b in range(a + 1, 5)]
    for index in range(len(times)):
        started = time.perf_counter()
        wrist_rotation = Rotation.from_quat(
            quats[index, 0], scalar_first=True
        ).as_matrix()
        targets, root, root_rotation = calibration.transform(
            joints[index], wrist_rotation
        )
        targets_world = targets @ root_rotation.T + root
        dt = times[index] - times[index - 1] if index else times[1] - times[0]
        q, iterations = optimizer.solve(targets, previous, dt)
        frames = robot.world_frames(q, root, root_rotation)
        status = int(optimizer.opt.last_optimize_result())
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
            "dexpilot_projected": optimizer.primary.projected.copy(),
        }.items():
            arrays[key].append(value)
        results = {
            "wrist_position": root.tolist(),
            "wrist_wxyz": Rotation.from_matrix(root_rotation)
            .as_quat(scalar_first=True)
            .tolist(),
            "finger_joints": q.tolist(),
            "frames": frames.tolist(),
            "frame_task_errors": [
                float(
                    np.linalg.norm(
                        frames[robot.get_link_index(name), :3] - targets_world[j]
                    )
                )
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
        "optimization_budget_exhausted_count": int(
            np.sum(converted["optimizer_status"] == 5)
        ),
        "frame_count": len(times),
    }
    return fields, converted, summary


def run_retarget(
    input_parquet: Path, calibration_path: Path, upstream: Path, output: Path
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scipy.spatial.transform import Rotation

    input_parquet = input_parquet.resolve()
    calibration_path = calibration_path.resolve()
    upstream, output = upstream.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("重定向输出必须为新目录")
    source_hash, calibration_hash = sha256(input_parquet), sha256(calibration_path)
    source_table = pq.ParquetFile(input_parquet).read()
    if source_table.num_rows != 1:
        raise ValueError("一次只允许一个完整 episode")
    original_row = source_table.to_pylist()[0]
    row = dict(original_row)
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
        "backend": BACKEND,
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "log_path": str(output / "run.log"),
        "output_manifest_path": str(output / "manifest.json"),
        "input_manifest_path": str(calibration_path),
        "backend_version": "sharpa-dexpilot-chain-rebuilt-v1",
        "solver_provenance": primary_provenance(),
        "recovery_status": "rebuilt_from_installed_primary_code",
        "optimizer_config": asdict(DexPilotChainConfig()),
        "source_sha256": {
            name: sha256(Path(__file__).parent / name)
            for name in (
                "sharpa_dexpilot.py",
                "sharpa_dexpilot_reference.py",
                "sharpa_calibration.py",
                "sharpa_multitip.py",
                "sharpa_multitip_kinematics.py",
                "sharpa_multitip_surface.py",
            )
        },
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
    log_path.write_text(f"{manifest['created_at']} 开始 {BACKEND}，{count} 帧\n")
    try:
        diagnostics: dict[str, np.ndarray] = {"timestamps_s": times}
        preview: dict[str, Any] = {
            "schema": "nvidia_v2d.sharpa_retarget_preview.v1",
            "backend": BACKEND,
            "kind": "kinematic_reference",
            "timestamps_s": times.tolist(),
            "edges": list(CHAIN_PAIRS) + [(0, x) for x in (5, 9, 13, 17)],
        }
        for side in ("left", "right"):
            fields, arrays, summary = _retarget_side(
                row, calibration.hands[side], times, side, upstream
            )
            row.update(fields)
            diagnostics.update(
                {f"{side}_{key}": value for key, value in arrays.items()}
            )
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
        if (
            sha256(input_parquet) != source_hash
            or sha256(calibration_path) != calibration_hash
        ):
            raise ValueError("读取过程中输入发生变化")
        changed = [key for key in row if row[key] != original_row[key]]
        if not all(
            key.startswith(
                ("robot_left_", "robot_right_", "left_robot_", "right_robot_")
            )
            for key in changed
        ):
            raise ValueError("DexPilot changed nonrobot source fields")
        manifest["changed_fields"] = changed
        manifest["nonrobot_fields_exactly_preserved"] = True
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
            log.write(
                f"{manifest['completed_at']} 失败：{manifest['failure_reason']}\n"
            )
        _json(output / "manifest.json", manifest)
        raise
    _json(output / "manifest.json", manifest)
    return manifest
