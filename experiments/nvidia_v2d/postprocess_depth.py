"""全时间轴 CPU 深度源 A/B：固定旋转、共享像素，重估固定形态尺度与逐帧平移。"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from scipy import ndimage
from scipy.spatial.transform import Rotation

from experiments.nvidia_v2d.depth_source_ab import (
    CameraK,
    decode_depth,
    rasterize_depth,
    reproject_depth,
)
from experiments.nvidia_v2d.sharpa_bridge import digest, write_json
from experiments.nvidia_v2d.sharpa_episode import require_new_output


class DepthRefitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    stride: int = Field(default=4, ge=1, le=8)
    workers: int = Field(default=4, ge=1, le=8)
    min_overlap_pixels: int = Field(default=128, ge=16)
    min_mask_coverage: float = Field(default=0.25, gt=0, le=1)
    min_depth_coverage: float = Field(default=0.5, gt=0, le=1)
    min_largest_component_fraction: float = Field(default=0.85, gt=0, le=1)
    max_fit_rmse_m: float = Field(default=0.03, gt=0)
    min_scale: float = Field(default=0.5, gt=0)
    max_scale: float = Field(default=2.0, gt=0)
    max_translation_change_m: float = Field(default=0.5, gt=0)


def fit_scale_translation(rotated_local: np.ndarray, observed_camera: np.ndarray) -> dict[str, Any]:
    """对应关系已知且旋转固定的正尺度/平移最小二乘，三轮稳健截尾。"""
    p, q = np.asarray(rotated_local, float), np.asarray(observed_camera, float)
    if (
        p.shape != q.shape
        or p.ndim != 2
        or p.shape[1] != 3
        or len(p) < 16
        or not np.isfinite(p).all()
        or not np.isfinite(q).all()
    ):
        raise ValueError("至少需要16组有限三维对应点")
    keep = np.ones(len(p), bool)
    for _ in range(3):
        pc, qc = p[keep].mean(0), q[keep].mean(0)
        dp, dq = p[keep] - pc, q[keep] - qc
        denominator = float(np.sum(dp * dp))
        if denominator <= 1e-10:
            raise ValueError("点云退化，尺度不可辨识")
        scale = float(np.sum(dp * dq) / denominator)
        translation = qc - scale * pc
        errors = np.linalg.norm(q - (scale * p + translation), axis=1)
        keep = errors <= max(float(np.quantile(errors, 0.8)), 1e-5)
    pc, qc = p[keep].mean(0), q[keep].mean(0)
    dp, dq = p[keep] - pc, q[keep] - qc
    pp, pq, qq = float(np.sum(dp * dp)), float(np.sum(dp * dq)), float(np.sum(dq * dq))
    if pp <= 1e-10:
        raise ValueError("稳健截尾后点云退化，尺度不可辨识")
    scale = pq / pp
    translation = qc - scale * pc
    return {
        "scale": scale,
        "translation_camera_m": translation.tolist(),
        "inliers": int(keep.sum()),
        "input_points": len(p),
        "rmse_m": float(
            np.sqrt(np.mean(np.sum((q[keep] - scale * p[keep] - translation) ** 2, axis=1)))
        ),
        "p_mean": pc.tolist(),
        "q_mean": qc.tolist(),
        "pp": pp,
        "pq": pq,
        "qq": qq,
    }


def fixed_scale_refit(fit: dict[str, Any], scale: float) -> dict[str, Any]:
    translation = np.asarray(fit["q_mean"]) - scale * np.asarray(fit["p_mean"])
    rss = max(0.0, fit["qq"] - 2 * scale * fit["pq"] + scale * scale * fit["pp"])
    return {
        "scale": scale,
        "translation_camera_m": translation.tolist(),
        "rmse_m": float(np.sqrt(rss / fit["inliers"])),
    }


def fit_gate(
    fit: dict[str, Any], config: DepthRefitConfig, original_translation: np.ndarray
) -> list[str]:
    reasons = []
    if not config.min_scale <= fit["scale"] <= config.max_scale:
        reasons.append("尺度超出预设范围")
    if fit["rmse_m"] > config.max_fit_rmse_m:
        reasons.append("三维对应残差过大")
    if (
        np.linalg.norm(np.asarray(fit["translation_camera_m"]) - original_translation)
        > config.max_translation_change_m
    ):
        reasons.append("相对旧位姿平移变化过大")
    return reasons


def _camera_small(camera: CameraK, stride: int) -> CameraK:
    return CameraK(
        width=(camera.width + stride - 1) // stride,
        height=(camera.height + stride - 1) // stride,
        fx=camera.fx / stride,
        fy=camera.fy / stride,
        cx=camera.cx / stride,
        cy=camera.cy / stride,
    )


def _mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        values = line.split()
        if not values:
            continue
        if values[0] == "v":
            vertices.append([float(value) for value in values[1:4]])
        if values[0] == "f":
            indices = [int(value.split("/")[0]) - 1 for value in values[1:]]
            for i in range(1, len(indices) - 1):
                faces.append([indices[0], indices[i], indices[i + 1]])
    if not vertices or not faces:
        raise ValueError("无可用三角形网格")
    return np.asarray(vertices), np.asarray(faces)


def _load_frame(
    frame: int, reconstruction: Path, sensor: Path, target: CameraK, stride: int
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    paths = {
        "predicted": reconstruction / "depth" / f"{frame:06d}.png",
        "sensor": sensor / "depth" / f"{frame:06d}.png",
        "k": reconstruction / "intrinsics" / f"{frame:06d}.json",
        "mask": reconstruction / "masks/1" / f"{frame:06d}.png",
        "pose": reconstruction / "poses_smoothed" / f"{frame:06d}.json",
    }
    hashes = {str(path): digest(path) for path in paths.values()}
    source_k = CameraK.model_validate_json(paths["k"].read_text())
    small = _camera_small(target, stride)
    source_small = _camera_small(source_k, stride)
    depths = {
        "predicted": reproject_depth(
            decode_depth(paths["predicted"])[::stride, ::stride], source_small, small
        ),
        "sensor": decode_depth(paths["sensor"])[::stride, ::stride],
    }
    with Image.open(paths["mask"]) as image:
        mask = np.asarray(image)[::stride, ::stride] > 0
    pose = json.loads(paths["pose"].read_text())
    quat = np.asarray(pose["rotation"])
    rotation = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    translation = np.asarray(pose["translation"])
    if not np.allclose(pose["scale"], [1, 1, 1]):
        raise ValueError("旧位姿包含隐式尺度")
    return depths, mask, rotation, translation, hashes


def _initial_frame(
    frame: int,
    *,
    reconstruction: Path,
    sensor: Path,
    target: CameraK,
    vertices: np.ndarray,
    faces: np.ndarray,
    config: DepthRefitConfig,
) -> dict[str, Any]:
    try:
        depths, mask, rotation, translation, hashes = _load_frame(
            frame, reconstruction, sensor, target, config.stride
        )
        camera = _camera_small(target, config.stride)
        rendered = rasterize_depth(vertices @ rotation.T + translation, faces, camera)
        labels, _ = ndimage.label(mask)
        sizes = np.bincount(labels.ravel())[1:]
        mask_count = int(mask.sum())
        largest = float(sizes.max() / mask_count) if mask_count else 0
        both = mask & np.isfinite(depths["predicted"]) & np.isfinite(depths["sensor"])
        shared = both & np.isfinite(rendered)
        overlap = int(shared.sum())
        reasons = []
        if overlap < config.min_overlap_pixels:
            reasons.append("共同网格重叠像素不足")
        if overlap / max(mask_count, 1) < config.min_mask_coverage:
            reasons.append("网格对目标掩膜覆盖不足")
        if int(both.sum()) / max(mask_count, 1) < config.min_depth_coverage:
            reasons.append("两路共同有效深度覆盖不足")
        if largest < config.min_largest_component_fraction:
            reasons.append("掩膜明显碎裂或为空")
        result = {
            "frame_index": frame,
            "input_sha256": hashes,
            "mask_pixels": mask_count,
            "overlap_pixels": overlap,
            "largest_component_fraction": largest,
            "original_translation_camera_m": translation.tolist(),
            "input_rejection_reasons_zh": reasons,
            "fits": {},
        }
        if reasons:
            return result
        y, x = np.nonzero(shared)
        rays = np.column_stack(
            ((x - camera.cx) / camera.fx, (y - camera.cy) / camera.fy, np.ones(len(x)))
        )
        p = rays * rendered[shared, None] - translation
        for name, depth in depths.items():
            fit = fit_scale_translation(p, rays * depth[shared, None])
            fit["rejection_reasons_zh"] = fit_gate(fit, config, translation)
            result["fits"][name] = fit
        return result
    except (ValueError, OSError, KeyError) as exc:
        return {
            "frame_index": frame,
            "input_rejection_reasons_zh": [f"输入或数值错误：{exc}"],
            "fits": {},
        }


def _final_frame(
    row: dict[str, Any],
    *,
    scales: dict[str, float],
    reconstruction: Path,
    sensor: Path,
    target: CameraK,
    vertices: np.ndarray,
    faces: np.ndarray,
    config: DepthRefitConfig,
) -> dict[str, Any]:
    depths, mask, rotation, original, hashes = _load_frame(
        row["frame_index"], reconstruction, sensor, target, config.stride
    )
    if hashes != row["input_sha256"]:
        raise ValueError("第二遍读取时来源改变")
    camera = _camera_small(target, config.stride)
    result = {"frame_index": row["frame_index"], "branches": {}}
    for name, scale in scales.items():
        fit = fixed_scale_refit(row["fits"][name], scale)
        reasons = fit_gate(fit, config, original)
        translation = np.asarray(fit["translation_camera_m"])
        rendered = rasterize_depth(scale * (vertices @ rotation.T) + translation, faces, camera)
        valid = mask & np.isfinite(rendered) & np.isfinite(depths[name])
        count = int(valid.sum())
        coverage = count / max(int(mask.sum()), 1)
        residual = (
            float(np.sqrt(np.mean((depths[name][valid] - rendered[valid]) ** 2))) if count else None
        )
        if count < config.min_overlap_pixels or coverage < config.min_mask_coverage:
            reasons.append("重投影后目标支持不足")
        if residual is None or residual > config.max_fit_rmse_m:
            reasons.append("重投影后深度残差过大")
        fit.update(
            rotation_matrix=rotation.tolist(),
            accepted=not reasons,
            rejection_reasons_zh=reasons,
            final_overlap_pixels=count,
            final_mask_coverage=coverage,
            final_depth_rmse_m=residual,
        )
        result["branches"][name] = fit
    result["paired_accepted"] = all(branch["accepted"] for branch in result["branches"].values())
    return result


def run_refit(
    reconstruction: Path,
    sensor: Path,
    output: Path,
    config: DepthRefitConfig,
    frames: list[int] | None = None,
) -> dict[str, Any]:
    source_manifest = json.loads((sensor / "manifest.json").read_text())
    raw = Path(source_manifest["recording"])
    require_new_output(output, [reconstruction, sensor, raw])
    if (
        source_manifest["status"] != "succeeded"
        or source_manifest["frame_index_basis"] != "authoritative_raw_container_zero_based"
    ):
        raise ValueError("实测深度时间轴来源不明")
    source_rows = {row["frame_index"]: row for row in source_manifest["frames"]}
    selected = sorted(source_rows) if frames is None else frames
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(f not in source_rows for f in selected)
    ):
        raise ValueError("帧编号必须唯一且有真实时间来源")
    target_path = sensor / "camera_intrinsics.json"
    target = CameraK.model_validate_json(target_path.read_text())
    mesh_path = reconstruction / "mesh_pretransformed.obj"
    vertices, faces = _mesh(mesh_path)
    fixed = {
        str(path): digest(path)
        for path in [sensor / "manifest.json", target_path, sensor / "calibration.json", mesh_path]
    }
    for item in source_manifest["source_files_before"]:
        if Path(item["path"]).name in {"meta.json", "index.jsonl"}:
            if digest(Path(item["path"])) != item["sha256"]:
                raise ValueError("原始时间/标定来源改变")
            fixed[item["path"]] = item["sha256"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    started = datetime.now(UTC).isoformat()
    write_json(
        output / "input.json",
        {
            "schema": "nvidia_v2d.depth_refit_input.v1",
            "reconstruction": str(reconstruction),
            "sensor": str(sensor),
            "config": config.model_dump(),
            "frames": selected,
            "input_sha256": fixed,
        },
    )
    common = dict(
        reconstruction=reconstruction,
        sensor=sensor,
        target=target,
        vertices=vertices,
        faces=faces,
        config=config,
    )
    rows = []
    with (
        (output / "run.log").open("x") as log,
        ProcessPoolExecutor(max_workers=config.workers) as pool,
    ):
        for row in pool.map(partial(_initial_frame, **common), selected, chunksize=8):
            frame = row["frame_index"]
            row["timestamp_ns"] = source_rows[frame]["timestamp_ns"]
            row["elapsed_s"] = (
                source_rows[frame]["timestamp_ns"] - source_rows[0]["timestamp_ns"]
            ) / 1e9
            sensor_path = str(sensor / "depth" / f"{frame:06d}.png")
            if row.get("input_sha256", {}).get(sensor_path) != source_rows[frame]["sha256"]:
                row["input_rejection_reasons_zh"].append("实测深度凭据不符")
                row["fits"] = {}
            write_json(output / "checkpoints" / f"{frame:06d}.json", row)
            rows.append(row)
            if len(rows) % 100 == 0:
                log.write(f"第一遍完成 {len(rows)}/{len(selected)} 帧\n")
                log.flush()
                print(f"first_pass {len(rows)}/{len(selected)}", flush=True)
        eligible = [
            row
            for row in rows
            if not row["input_rejection_reasons_zh"]
            and len(row["fits"]) == 2
            and all(not fit["rejection_reasons_zh"] for fit in row["fits"].values())
        ]
        scales = (
            {
                name: float(np.median([row["fits"][name]["scale"] for row in eligible]))
                for name in ("predicted", "sensor")
            }
            if eligible
            else {}
        )
        final = (
            list(pool.map(partial(_final_frame, scales=scales, **common), eligible, chunksize=8))
            if eligible
            else []
        )
        log.write(f"第二遍完成 {len(final)} 帧；固定尺度 {scales}\n")
    # Emit complete source time axes; rejected frames remain NaN, never filled.
    final_by_frame = {row["frame_index"]: row for row in final}
    arrays = {
        "source_frame_indices": np.asarray(selected),
        "timestamps_ns": np.asarray(
            [source_rows[f]["timestamp_ns"] for f in selected], dtype=np.int64
        ),
    }
    for name in ("predicted", "sensor"):
        transforms = np.full((len(selected), 4, 4), np.nan)
        valid = np.zeros(len(selected), bool)
        for i, frame in enumerate(selected):
            row = final_by_frame.get(frame)
            if row and row["paired_accepted"]:
                fit = row["branches"][name]
                transforms[i] = np.eye(4)
                transforms[i, :3, :3] = fit["rotation_matrix"]
                transforms[i, :3, 3] = fit["translation_camera_m"]
                valid[i] = True
        arrays[name + "_object_to_camera_transform"] = transforms
        arrays[name + "_candidate_valid"] = valid
        arrays[name + "_mesh_scale"] = np.asarray(scales.get(name, np.nan))
    np.savez_compressed(output / "refit.npz", **arrays)
    all_hashes = dict(fixed)
    for row in rows:
        all_hashes.update(row.get("input_sha256", {}))
    unchanged = all(digest(Path(path)) == sha for path, sha in all_hashes.items())
    if not unchanged:
        raise ValueError("处理期间输入改变，拒绝发布")
    result = {
        "schema": "nvidia_v2d.depth_refit.v1",
        "status": "completed_candidates" if eligible else "blocked_no_supported_frames",
        "started_at": started,
        "finished_at": datetime.now(UTC).isoformat(),
        "source_recording": str(raw),
        "config": config.model_dump(),
        "frame_count": len(selected),
        "initial_paired_eligible_frames": len(eligible),
        "final_paired_candidate_frames": sum(row["paired_accepted"] for row in final),
        "fixed_mesh_scales": scales,
        "final_frames": final,
        "source_inputs_unchanged": unchanged,
        "geometry_quality_accepted": False,
        "full_training_accepted": False,
        "camera_world_reestimated": False,
        "semantic_mask_identity_verified": False,
        "说明": [
            "全时间轴保留真实时间与失败帧。两路使用同一原K到设备K重投影、固定旋转、同一网格和共同有效像素。",
            "第一遍估计每帧尺度/平移，仅两路同时通过门控的帧用于各自整段固定尺度中值；第二遍固定尺度重估平移，并重新渲染验收。",
            "模型无GPU依赖，没有ICP；这里估计相机坐标中的物体，不冒称重新标定了SLAM尺度或世界手轨迹。",
            "自动掩膜连通性、覆盖和残差门控不能证明语义身份正确；本产物是候选，禁止直接进入策略训练。",
        ],
    }
    write_json(output / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--sensor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = run_refit(
        args.reconstruction,
        args.sensor,
        args.output,
        DepthRefitConfig(workers=args.workers),
        args.frames,
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "frame_count",
                    "initial_paired_eligible_frames",
                    "final_paired_candidate_frames",
                    "fixed_mesh_scales",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
