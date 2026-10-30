"""无辅助实际物体轨迹合同；逐环境、逐物体验证完整序列，不掩盖失败环境。"""

import hashlib
from pathlib import Path
from typing import Any


def validate_trajectory(
    path: Path, expected_frames: int, world_count: int, expected_fps: float = 20.0
) -> dict[str, Any]:
    import numpy as np
    import pyarrow.parquet as pq

    if expected_frames < 1 or world_count < 1 or not np.isfinite(expected_fps) or expected_fps <= 0:
        raise ValueError("预期轨迹大小或帧率无效")
    table = pq.ParquetFile(path).read()
    metadata = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    fixed = {
        "schema": "flash_chord_object_trajectories_v1",
        "position_unit": "meter",
        "quaternion_order": "xyzw",
        "row_order": "step,environment_id,object_body_id",
        "start_frame": "0",
        "step_count": str(expected_frames),
        "world_count": str(world_count),
        "pose_width": "7",
    }
    if any(metadata.get("flash_chord." + key) != value for key, value in fixed.items()):
        raise ValueError("轨迹metadata与完整序列、坐标或单位合同不符")
    try:
        fps = float(metadata["flash_chord.control_fps"])
        body_count = int(metadata["flash_chord.object_body_count"])
    except (KeyError, ValueError) as exc:
        raise ValueError("轨迹metadata缺少帧率或物体数量") from exc
    if body_count < 1 or not np.isfinite(fps) or not np.isclose(fps, expected_fps, rtol=0, atol=1e-8):
        raise ValueError("轨迹metadata帧率或物体数量无效")
    columns = table.to_pydict()
    required = {
        "environment_id",
        "step",
        "reference_frame",
        "time_s",
        "object_id",
        "object_body_id",
        "object_body_name",
        "environment_valid",
    }
    pose_columns = [
        f"{prefix}_{kind}_{axis}"
        for prefix in ("achieved", "reference")
        for kind, axes in (("position", "xyz"), ("quaternion", "xyzw"))
        for axis in axes
    ]
    required.update(pose_columns)
    if not required.issubset(columns) or any(table[name].null_count for name in required):
        raise ValueError("轨迹缺少实际/参考位姿或包含空值")
    if len(table) != expected_frames * world_count * body_count:
        raise ValueError("轨迹并非所有环境物体的完整序列")
    arrays = {name: np.asarray(value) for name, value in columns.items()}
    for name in ("environment_id", "step", "reference_frame", "object_id", "object_body_id"):
        if not np.issubdtype(arrays[name].dtype, np.integer):
            raise ValueError("轨迹环境、物体与帧索引必须为整数")
    world, body, step = (arrays[name] for name in ("environment_id", "object_body_id", "step"))
    if not np.array_equal(np.unique(world), np.arange(world_count)) or not np.array_equal(
        np.unique(body), np.arange(body_count)
    ):
        raise ValueError("环境或物体ID不符合metadata数量")
    if np.any(arrays["object_id"] < 0) or not np.array_equal(
        np.unique(arrays["object_id"]), np.arange(len(np.unique(arrays["object_id"])))
    ):
        raise ValueError("物体组件ID必须非负连续")
    if arrays["environment_valid"].dtype != np.bool_:
        raise ValueError("环境有效性必须为显式布尔值")
    if not np.array_equal(np.lexsort((body, world, step)), np.arange(len(table))):
        raise ValueError("实际行顺序与声明不符或存在重复键")
    expected_steps = np.arange(expected_frames)
    body_names = []
    for body_id in range(body_count):
        selection = body == body_id
        names = np.unique(arrays["object_body_name"][selection])
        if (
            len(names) != 1
            or not str(names[0])
            or len(np.unique(arrays["object_id"][selection])) != 1
        ):
            raise ValueError("物体身份或组件绑定在帧间变化")
        body_names.append(str(names[0]))
        for world_id in range(world_count):
            group = selection & (world == world_id)
            if not np.array_equal(step[group], expected_steps) or not np.array_equal(
                arrays["reference_frame"][group], expected_steps
            ):
                raise ValueError("某环境物体缺帧、重复帧或参考帧不一致")
            if not np.allclose(
                arrays["time_s"][group], expected_steps / expected_fps, rtol=0, atol=1e-8
            ):
                raise ValueError("某环境物体时间轴与真实帧率不一致")
    if len(set(body_names)) != body_count:
        raise ValueError("不同物体body不能使用相同名称")
    values = np.column_stack([arrays[name] for name in pose_columns])
    if not np.isfinite(values).all():
        raise ValueError("无辅助实际或参考轨迹包含非有限数值")
    quaternion_errors = {}
    for prefix in ("achieved", "reference"):
        quats = np.column_stack([arrays[f"{prefix}_quaternion_{axis}"] for axis in "xyzw"])
        quaternion_errors[prefix] = float(np.max(np.abs(np.linalg.norm(quats, axis=1) - 1)))
        if quaternion_errors[prefix] > 1e-3:
            raise ValueError("实际或参考物体四元数未保持归一化")
    valid = arrays["environment_valid"]
    return {
        "rows": len(table),
        "world_count": world_count,
        "frame_count": expected_frames,
        "object_count": body_count,
        "object_body_names": body_names,
        "time_range_s": [float(np.min(arrays["time_s"])), float(np.max(arrays["time_s"]))],
        "all_pose_values_finite": True,
        "max_achieved_quaternion_norm_error": quaternion_errors["achieved"],
        "max_reference_quaternion_norm_error": quaternion_errors["reference"],
        "environment_valid_fraction": float(np.mean(valid)),
        "invalid_row_count": int((~valid).sum()),
        "invalid_worlds": np.unique(world[~valid]).tolist(),
        "all_worlds_valid": bool(valid.all()),
        "columns": table.column_names,
        "metadata": metadata,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
