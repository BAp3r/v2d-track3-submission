"""固定相机、掩膜、网格和位姿的 CPU 深度源诊断；不是完整重建重跑。"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from experiments.nvidia_v2d.sharpa_bridge import digest, write_json
from experiments.nvidia_v2d.sharpa_episode import require_new_output


class CameraK(BaseModel):
    model_config = ConfigDict(extra="forbid")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fx: float = Field(gt=0, allow_inf_nan=False)
    fy: float = Field(gt=0, allow_inf_nan=False)
    cx: float = Field(allow_inf_nan=False)
    cy: float = Field(allow_inf_nan=False)


def decode_depth(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        encoded = np.asarray(image)
    if encoded.dtype != np.uint16 or encoded.ndim != 2:
        raise ValueError("深度必须为官方 uint16 逆深度编码")
    valid = (encoded > 0) & (encoded < 65535)
    output = np.full(encoded.shape, np.nan)
    output[valid] = 65535.0 / encoded[valid] - 1.0
    return output


def reproject_depth(depth: np.ndarray, source: CameraK, target: CameraK) -> np.ndarray:
    """先反投影到同一相机 XYZ，再投影至固定 K；最近点 z-buffer 保留空洞。"""
    if depth.shape != (source.height, source.width):
        raise ValueError("深度尺寸与来源 K 不一致")
    y, x = np.nonzero(np.isfinite(depth) & (depth > 0))
    z = depth[y, x]
    tx = np.rint((x - source.cx) / source.fx * target.fx + target.cx).astype(int)
    ty = np.rint((y - source.cy) / source.fy * target.fy + target.cy).astype(int)
    valid = (tx >= 0) & (tx < target.width) & (ty >= 0) & (ty < target.height)
    output = np.full((target.height, target.width), np.inf)
    np.minimum.at(output, (ty[valid], tx[valid]), z[valid])
    output[~np.isfinite(output)] = np.nan
    return output


def rasterize_depth(vertices_camera: np.ndarray, faces: np.ndarray, camera: CameraK) -> np.ndarray:
    """CPU 三角形深度缓冲；采用透视正确的逆深度插值。"""
    vertices = np.asarray(vertices_camera, float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("无效网格顶点")
    image = np.full((camera.height, camera.width), np.inf)
    for face in faces:
        triangle = vertices[face]
        if np.any(triangle[:, 2] <= 1e-6):
            continue
        uv = triangle[:, :2] / triangle[:, 2:3]
        uv = uv * [camera.fx, camera.fy] + [camera.cx, camera.cy]
        low = np.maximum(np.floor(uv.min(0)).astype(int), [0, 0])
        high = np.minimum(np.ceil(uv.max(0)).astype(int), [camera.width - 1, camera.height - 1])
        if np.any(high < low):
            continue
        x, y = np.meshgrid(np.arange(low[0], high[0] + 1), np.arange(low[1], high[1] + 1))
        (x0, y0), (x1, y1), (x2, y2) = uv
        divisor = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(divisor) < 1e-12:
            continue
        a = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / divisor
        b = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / divisor
        c = 1 - a - b
        inside = (a >= -1e-9) & (b >= -1e-9) & (c >= -1e-9)
        denominator = a / triangle[0, 2] + b / triangle[1, 2] + c / triangle[2, 2]
        z = np.divide(1.0, denominator, out=np.full_like(a, np.inf), where=inside & (denominator > 0))
        section = image[low[1] : high[1] + 1, low[0] : high[0] + 1]
        np.minimum(section, z, out=section)
    image[~np.isfinite(image)] = np.nan
    return image


def depth_fit(rendered: np.ndarray, observed: np.ndarray, shared: np.ndarray) -> dict[str, Any]:
    valid = shared & np.isfinite(rendered) & (rendered > 0) & np.isfinite(observed) & (observed > 0)
    reference, values = rendered[valid], observed[valid]
    if not len(values):
        return {"count": 0, "scale": None, "residual_median_m": None, "fit_rmse_m": None}
    scale = float(np.dot(reference, values) / np.dot(reference, reference))
    return {
        "count": len(values),
        "scale": scale,
        "residual_median_m": float(np.median(values - reference)),
        "fit_rmse_m": float(np.sqrt(np.mean((values - scale * reference) ** 2))),
    }


def _summary(depth: np.ndarray, mask: np.ndarray, camera: CameraK) -> dict[str, Any]:
    y, x = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
    z = depth[y, x]
    if not len(z):
        return {"count": 0, "median_m": None, "p10_m": None, "p90_m": None}
    xyz = np.column_stack(((x - camera.cx) / camera.fx * z, (y - camera.cy) / camera.fy * z, z))
    return {
        "count": len(z),
        "median_m": float(np.median(z)),
        "p10_m": float(np.quantile(z, 0.1)),
        "p90_m": float(np.quantile(z, 0.9)),
        "visible_cloud_p90_minus_p10_xyz_m": (
            np.quantile(xyz, 0.9, axis=0) - np.quantile(xyz, 0.1, axis=0)
        ).tolist(),
    }


def run_depth_ab(
    reconstruction: Path, sensor: Path, output: Path, frames: list[int]
) -> dict[str, Any]:
    import trimesh
    from scipy.spatial.transform import Rotation

    require_new_output(output, [reconstruction, sensor])
    manifest_path = sensor / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["status"] != "succeeded"
        or manifest["frame_index_basis"] != "authoritative_raw_container_zero_based"
    ):
        raise ValueError("实测深度来源未成功或帧编号不明确")
    source_rows = {item["frame_index"]: item for item in manifest["frames"]}
    require_new_output(output, [reconstruction, sensor, Path(manifest["recording"])])
    if (
        not frames
        or len(set(frames)) != len(frames)
        or any(frame not in source_rows for frame in frames)
    ):
        raise ValueError("需要唯一且有真实时间来源的帧编号")
    target_path = sensor / "camera_intrinsics.json"
    target = CameraK.model_validate_json(target_path.read_text())
    # This mesh is before the depth-derived scale.json; its original orientation
    # and the previous MoGe reconstruction's poses are frozen for both arms.
    mesh_path = reconstruction / "mesh_pretransformed.obj"
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    fixed = [manifest_path, target_path, sensor / "calibration.json", mesh_path]
    # Verify the small authoritative time/index sources again. The raw depth
    # movie is not read by this derived-data diagnostic.
    for item in manifest["source_files_before"]:
        path = Path(item["path"])
        if path.name in {"meta.json", "index.jsonl"}:
            if digest(path) != item["sha256"]:
                raise ValueError("原始时间或标定来源哈希发生变化")
            fixed.append(path)
    inputs = {str(path): digest(path) for path in fixed}
    rows = []
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(UTC).isoformat()
    for frame in frames:
        item = source_rows[frame]
        sensor_path = sensor / item["path"]
        predicted_path = reconstruction / "depth" / f"{frame:06d}.png"
        intrinsic_path = reconstruction / "intrinsics" / f"{frame:06d}.json"
        mask_path = reconstruction / "masks/1" / f"{frame:06d}.png"
        pose_path = reconstruction / "poses_smoothed" / f"{frame:06d}.json"
        for path in (sensor_path, predicted_path, intrinsic_path, mask_path, pose_path):
            inputs[str(path)] = digest(path)
        if inputs[str(sensor_path)] != item["sha256"]:
            raise ValueError(f"实测深度凭据不匹配：{frame}")
        original_k = CameraK.model_validate_json(intrinsic_path.read_text())
        predicted = reproject_depth(decode_depth(predicted_path), original_k, target)
        measured = decode_depth(sensor_path)
        with Image.open(mask_path) as image:
            mask = np.asarray(image) > 0
        if mask.shape != measured.shape or mask.shape != predicted.shape:
            raise ValueError("固定 RGB 掩膜与深度尺寸不一致")
        shared = mask & np.isfinite(predicted) & np.isfinite(measured)
        pose = json.loads(pose_path.read_text())
        q = np.asarray(pose["rotation"])
        rotation = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        camera_vertices = mesh.vertices @ rotation.T + np.asarray(pose["translation"])
        rendered = rasterize_depth(camera_vertices, mesh.faces, target)
        row = {
            "frame_index": frame,
            "timestamp_ns": item["timestamp_ns"],
            "elapsed_s": (item["timestamp_ns"] - source_rows[0]["timestamp_ns"]) / 1e9,
            "mask_pixels": int(mask.sum()),
            "shared_valid_pixels": int(shared.sum()),
            "predicted_source_intrinsics": original_k.model_dump(),
            "predicted_all_mask": _summary(predicted, mask, target),
            "sensor_all_mask": _summary(measured, mask, target),
            "predicted_shared": _summary(predicted, shared, target),
            "sensor_shared": _summary(measured, shared, target),
            "predicted_camera_depth_scale_ls": depth_fit(rendered, predicted, shared),
            "sensor_camera_depth_scale_ls": depth_fit(rendered, measured, shared),
            "sensor_minus_predicted_median_m": float(np.median(measured[shared] - predicted[shared]))
            if shared.any()
            else None,
        }
        overlap = int((shared & np.isfinite(rendered)).sum())
        row["shared_mesh_overlap_pixels"] = overlap
        row["shared_mesh_overlap_fraction_of_fixed_mask"] = overlap / max(int(mask.sum()), 1)
        row["diagnostic_fit_eligible"] = overlap >= 512 and overlap / max(int(mask.sum()), 1) >= 0.05
        rows.append(row)
        np.savez_compressed(
            output / f"frame_{frame:06d}.npz",
            predicted_depth_m=predicted,
            sensor_depth_m=measured,
            rendered_mesh_depth_m=rendered,
            fixed_mask=mask,
            shared_valid=shared,
        )
    unchanged = all(digest(Path(path)) == sha for path, sha in inputs.items())
    if not unchanged:
        raise ValueError("输入在诊断期间变化，拒绝发布")
    result = {
        "schema": "nvidia_v2d.depth_source_ab.v1",
        "status": "succeeded",
        "started_at": started,
        "finished_at": datetime.now(UTC).isoformat(),
        "source_recording": manifest["recording"],
        "source_inputs_sha256": inputs,
        "source_inputs_unchanged": unchanged,
        "fixed_target_intrinsics": target.model_dump(),
        "depth_reprojection": "source K backprojection then target K nearest z-buffer; no hole filling",
        "mesh": str(mesh_path),
        "mesh_depth_scale_applied": False,
        "poses": "frozen previous MoGe-derived poses, not re-estimated",
        "full_reconstruction_rerun": False,
        "geometry_quality_accepted": False,
        "frames": rows,
        "说明": [
            "只改变深度源；K、帧、掩膜、未作深度缩放的网格、旧位姿和拟合算法保持相同。",
            "MoGe先经逐帧原K反投影，再投影到设备标定K；两路在相同有效像素上配对比较。",
            "camera_depth_scale_ls为旧位姿下投影深度的诊断比例，不是重新求出的物体尺度或位姿。",
            "原掩膜与旧位姿可能有误；本结果不认证真实物体尺寸、接触、完整重建或策略执行。",
        ],
    }
    write_json(output / "manifest.json", result)
    lines = ["NAS 箱子录制：固定输入的深度源诊断", "", *result["说明"], ""]
    for row in rows:
        lines.append(
            f"第{row['frame_index']}帧（{row['elapsed_s']:.3f}秒）：共有有效像素{row['shared_valid_pixels']}；预测中值{row['predicted_shared']['median_m']}米，实测中值{row['sensor_shared']['median_m']}米。"
        )
        if not row["diagnostic_fit_eligible"]:
            lines.append("  网格重叠支持不足，拒绝把拟合比例用于后续尺度标定。")
    (output / "report_zh.txt").write_text("\n".join(lines), encoding="utf-8")
    (output / "run.log").write_text(
        f"开始：{started}\n完成：{result['finished_at']}\n状态：成功，{len(rows)}帧。未运行GPU模型。\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--sensor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[2400, 3750, 4200, 4422, 4580])
    args = parser.parse_args()
    result = run_depth_ab(args.reconstruction, args.sensor, args.output, args.frames)
    print(
        json.dumps(
            {"status": result["status"], "frames": len(result["frames"]), "output": str(args.output)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
