"""只读官方重建，在全新目录导出带观测来源的手部质量副本。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from experiments.nvidia_v2d.hand_quality import (
    SOURCE_CODES,
    HandQualityPolicy,
    Record,
    axisangle_to_matrix,
    process_track,
    support_bounds,
)


class ReadLedger:
    """读取前后核对身份；结束时重新核对哈希，不能把改变中的输入当稳定证据。"""

    def __init__(self, root: Path):
        self.root = root
        self.entries: dict[str, Record] = {}

    def read(self, path: Path) -> bytes:
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise ValueError(f"输入必须是重建目录内的常规文件：{path}")
        before = path.stat()
        with path.open("rb") as stream:
            value = stream.read()
        after = path.stat()
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError(f"读取过程中输入改变：{path}")
        self.entries[str(path.relative_to(self.root))] = {
            "size": before.st_size,
            "mtime_ns": before.st_mtime_ns,
            "sha256": hashlib.sha256(value).hexdigest(),
        }
        return value

    def json(self, path: Path) -> Record:
        value = json.loads(self.read(path))
        if not isinstance(value, dict):
            raise ValueError(f"输入 JSON 必须为对象：{path}")
        return value

    def image(self, path: Path) -> np.ndarray:
        import io

        with Image.open(io.BytesIO(self.read(path))) as image:
            return np.asarray(image)

    def verify(self) -> None:
        for relative, expected in list(self.entries.items()):
            path = self.root / relative
            actual = self.read(path)
            current = self.entries[relative]
            if current != expected or hashlib.sha256(actual).hexdigest() != expected["sha256"]:
                raise ValueError(f"处理过程中输入改变：{path}")


def _confined(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError("手部来源目录必须在本次重建目录内")
    return path


def _records(
    directory: Path, ledger: ReadLedger, n_frames: int, expected_right: bool
) -> dict[int, Record]:
    records: dict[int, Record] = {}
    for path in sorted(directory.glob("*.json")):
        if not path.stem.isdigit():
            continue
        record = ledger.json(path)
        frame = int(path.stem)
        if record.get("frame_idx") != frame or not 0 <= frame < n_frames:
            raise ValueError(f"记录帧号不匹配或越界：{path}")
        if record.get("is_right") is not expected_right:
            raise ValueError(f"手部侧别与清单不匹配，禁止静默换手：{path}")
        records[frame] = record
    return records


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _export_hand_arrays(
    arrays: dict[str, np.ndarray], side: str, accepted: dict[int, Record], rows: list[Record]
) -> None:
    prefix = f"hand_{side}"
    n_frames = len(rows)
    expected_suffixes = {
        "betas",
        "is_valid",
        "wrist_orient_in_camera",
        "wrist_trans_in_camera",
        "finger_pose",
        "scale",
        "wrist_to_camera_transform",
        "wrist_to_world_transform",
    }
    unexpected = [
        key
        for key in arrays
        if key.startswith(prefix + "_") and key.removeprefix(prefix + "_") not in expected_suffixes
    ]
    if unexpected:
        raise ValueError(f"未知手部数组不能未经同步混入质量副本：{unexpected}")
    # 官方束的 betas 是每侧一个常量；同步 JSON 使用同一常量，避免形状两套解释。
    betas = np.asarray(arrays.get(prefix + "_betas", np.zeros(10)), dtype=np.float32).reshape(10)
    if not np.isfinite(betas).all():
        raise ValueError("官方束手形参数包含非有限值")
    arrays[prefix + "_betas"] = betas
    arrays[prefix + "_is_valid"] = np.zeros(n_frames, dtype=bool)
    arrays[prefix + "_wrist_orient_in_camera"] = np.zeros((n_frames, 3), dtype=np.float32)
    arrays[prefix + "_wrist_trans_in_camera"] = np.zeros((n_frames, 3), dtype=np.float32)
    arrays[prefix + "_finger_pose"] = np.zeros((n_frames, 15, 3), dtype=np.float32)
    arrays[prefix + "_scale"] = np.ones(n_frames, dtype=np.float32)
    for suffix in ("_wrist_to_camera_transform", "_wrist_to_world_transform"):
        arrays[prefix + suffix] = np.tile(np.eye(4, dtype=np.float32), (n_frames, 1, 1))
    for frame, record in accepted.items():
        orient = np.asarray(record["mano"]["global_orient"], dtype=np.float32)
        translation = np.asarray(record["camera"]["cam_t"], dtype=np.float32)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = axisangle_to_matrix(orient)
        pose[:3, 3] = translation
        record["mano"]["betas"] = betas.tolist()
        arrays[prefix + "_is_valid"][frame] = True
        arrays[prefix + "_wrist_orient_in_camera"][frame] = orient
        arrays[prefix + "_wrist_trans_in_camera"][frame] = translation
        arrays[prefix + "_finger_pose"][frame] = np.asarray(record["mano"]["hand_pose"]).reshape(
            15, 3
        )
        arrays[prefix + "_scale"][frame] = float(record.get("hand_scale", 1.0))
        arrays[prefix + "_wrist_to_camera_transform"][frame] = pose
        arrays[prefix + "_wrist_to_world_transform"][frame] = (
            arrays["camera_to_world_transform"][frame] @ pose
        )
    for key in ("source", "original_source"):
        arrays[f"{prefix}_{key}_code"] = np.asarray(
            [SOURCE_CODES[row[key]] for row in rows], dtype=np.uint8
        )
    arrays[prefix + "_is_observed"] = np.asarray([row["is_observed"] for row in rows], dtype=bool)
    arrays[prefix + "_original_observation_rejected"] = np.asarray(
        [row["observation_status"] == "rejected" for row in rows], dtype=bool
    )
    arrays[prefix + "_positive_depth_support_lower_bound"] = np.asarray(
        [row["support"]["positive_support_lower_bound"] if row["support"] else -1 for row in rows],
        dtype=np.int32,
    )


def export_quality(
    root: Path, output: Path, policy: HandQualityPolicy | None = None, bundle_name: str = "result"
) -> Record:
    """输入无需原始录制目录；所有逐帧侧别、变换与姿态参数一起重建。"""
    import io

    policy = policy or HandQualityPolicy()
    root, output = root.resolve(), output.resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("输出必须与输入重建目录隔离，且不能包含输入目录")
    if output.exists():
        raise FileExistsError("输出目录已存在；禁止覆盖历史产物")
    ledger = ReadLedger(root)
    manifest = ledger.json(root / bundle_name / "manifest.json")
    sources = manifest.get("sources", {})
    with np.load(
        io.BytesIO(ledger.read(root / bundle_name / "result.npz")), allow_pickle=False
    ) as data:
        arrays = {key: data[key].copy() for key in data.files}
    n_frames = int(manifest["n_frames"])
    camera = arrays["camera_to_world_transform"]
    if camera.shape != (n_frames, 4, 4):
        raise ValueError("相机矩阵帧数与清单不符")
    camera_good = np.asarray(arrays["camera_is_valid"], dtype=bool).copy()
    camera_good &= np.isfinite(camera).all(axis=(1, 2))
    camera_good &= np.max(np.abs(camera[:, 3] - [0, 0, 0, 1]), axis=1) < 1e-5
    rotations = camera[:, :3, :3].astype(np.float64)
    camera_good &= (
        np.max(np.abs(np.swapaxes(rotations, 1, 2) @ rotations - np.eye(3)), axis=(1, 2)) < 1e-4
    )
    camera_good &= np.abs(np.linalg.det(rotations) - 1) < 1e-4
    tracks: dict[str, Record] = {}
    for side in ("left", "right"):
        final_value = sources.get(f"{side}_hand_dir")
        if not final_value:
            tracks[side] = {"id": None, "observed": {}, "filled": {}, "supports": {}}
            continue
        final = _confined(root, str(final_value))
        track_id = final.name
        if not track_id.isdigit():
            raise ValueError("清单中的手部轨迹 ID 必须为数字目录")
        source_name = str(sources.get("hand_pose_source", "hamer"))
        aligned = root / f"{source_name}_aligned" / track_id
        tracks[side] = {
            "id": track_id,
            "observed": _records(aligned, ledger, n_frames, side == "right"),
            "filled": _records(final, ledger, n_frames, side == "right"),
            "supports": {},
        }
    object_id = int(sources.get("object_track_id", 1))
    for frame in range(n_frames):
        depth_path, object_path = (
            root / "depth" / f"{frame:06d}.png",
            root / "masks" / str(object_id) / f"{frame:06d}.png",
        )
        available = depth_path.is_file() and object_path.is_file()
        depth = ledger.image(depth_path) if available else None
        obj = ledger.image(object_path) > 0 if available else None
        for track in tracks.values():
            track_id = track["id"]
            mask_path = root / "masks" / str(track_id) / f"{frame:06d}.png"
            if depth is None or obj is None or track_id is None or not mask_path.is_file():
                track["supports"][frame] = None
                continue
            record = track["observed"].get(frame, {})
            if record and record.get("image_size") != [int(depth.shape[1]), int(depth.shape[0])]:
                track["supports"][frame] = None
                continue
            pixels = record.get("diagnostics", {}).get("n_pixels", 0)
            if isinstance(pixels, bool) or not isinstance(pixels, int):
                track["supports"][frame] = None
                continue
            try:
                track["supports"][frame] = support_bounds(
                    depth, ledger.image(mask_path) > 0, obj, pixels
                )
            except ValueError:
                track["supports"][frame] = None
    rows_by_side: dict[str, list[Record]] = {}
    accepted_by_side: dict[str, dict[int, Record]] = {}
    for side, track in tracks.items():
        accepted, rows = process_track(
            track["observed"],
            track["filled"],
            track["supports"],
            n_frames,
            policy,
            camera_good,
            camera,
        )
        _export_hand_arrays(arrays, side, accepted, rows)
        rows_by_side[side], accepted_by_side[side] = rows, accepted
    ledger.verify()
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "quality_result.npz", **arrays)
    for side, accepted in accepted_by_side.items():
        directory = output / "hands" / side
        directory.mkdir(parents=True)
        for frame, record in accepted.items():
            _write_json(directory / f"{frame:06d}.json", record)
    provenance = {
        "schema_version": 1,
        "source_codes": SOURCE_CODES,
        "n_frames": n_frames,
        "hands": rows_by_side,
    }
    _write_json(output / "provenance.json", provenance)
    _write_json(output / "input-manifest.json", {"root": str(root), "files": ledger.entries})
    summary = {
        side: {
            "original_sources": dict(Counter(row["original_source"] for row in rows)),
            "output_sources": dict(Counter(row["source"] for row in rows)),
            "accepted_observations": sum(row["is_observed"] for row in rows),
            "rejected_observations": sum(row["observation_status"] == "rejected" for row in rows),
            "reasons": dict(Counter(reason for row in rows for reason in row["rejection_reasons"])),
        }
        for side, rows in rows_by_side.items()
    }
    receipt: Record = {
        "schema_version": 1,
        "adapter": "nvidia_v2d_hand_quality",
        "version": "1.1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "source_reconstruction": str(root),
        "source_manifest": str(root / bundle_name / "manifest.json"),
        "implementation": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(__file__).with_name("hand_quality.py"))
        },
        "policy": asdict(policy),
        "source_codes": SOURCE_CODES,
        "source_labels": {
            "observed": "真实帧模型观测",
            "interpolated": "双端短缺口插值",
            "extrapolated": "单端外推",
            "rejected": "拒绝或未通过复核",
            "missing": "缺失",
        },
        "reason_labels": {
            "geometry_support_insufficient": "渲染几何交集不足",
            "positive_depth_support_not_guaranteed": "有限正深度支持下界不足",
            "geometry_support_count_inconsistent": "交集计数不符合掩膜子集契约",
            "depth_support_unavailable": "缺少可复核的深度或掩膜证据",
            "alignment_ratio_outside_fit_domain": "对齐尺度比超出官方拟合域",
            "alignment_ratio_missing": "缺少对齐尺度比",
            "camera_translation_not_finite": "相机坐标手部平移不是有限三维向量",
            "camera_depth_not_positive": "手部相机深度非正",
            "hand_scale_not_finite_positive": "手形尺度不是有限正数",
            "camera_transform_not_valid_rigid": "相机世界变换无效或不刚性",
            "isolated_temporal_depth_ratio_outlier": "相邻可信观测之间出现孤立深度比异常",
            "upstream_fill_requires_revalidation": "既有插值或外推不能直接复用",
            "interpolation_camera_invalid": "缺口中存在无效相机变换",
            "interpolation_camera_discontinuity": "缺口中相机运动超过保守插值连续性条件",
            "interpolation_endpoint_motion_too_large": "两端手部位置或深度差异过大",
            "interpolation_visible_depth_support_missing": "整个缺口未能证明可见深度充分",
            "temporal_depth_change_camera_motion_confounded": "相机运动混淆深度尖峰判定，仅提示不拒绝",
            "mano_betas_not_finite": "手形参数不完整或非有限",
            "mano_global_orient_not_finite": "手腕朝向参数不完整或非有限",
            "mano_hand_pose_not_finite": "手指姿态参数不完整或非有限",
        },
        "n_frames": n_frames,
        "summary": summary,
        "input_files_unchanged": True,
        "input_file_count": len(ledger.entries),
        "geometry_quality_accepted": False,
        "说明": [
            "observed 指真实视频帧上的模型回归观测，不是物理真值。",
            "缺失的精确渲染交集不能补造；有限正深度数量使用可证明的保守下界。",
            "下界适用于官方对齐已启用同一手掩膜及物掩膜相交/排除的调用；换用其他对齐器必须重新验证此契约。",
            "仅在两端通过策略、整个缺口可见深度充分且相机连续的短缺口插值，绝不新增外推；原拒绝原因仍保留。",
            "插值采用相机坐标短线段及旋转 SLERP，未使用 SLAM 修复接触或估计误差。",
            "相机和物体数组保持原值，其有效标志不因本手部过滤而得到质量认证。",
            "每侧 betas 沿用官方束常量，并同步写入手部 JSON；所有逐帧手部参数及变换重新导出。",
            "插值 valid 表示可供预览使用，is_observed 才表示通过策略的直接观测；训练筛选应分别处理。",
        ],
        "artifacts": [],
    }
    for path in sorted(output.rglob("*")):
        if path.is_file():
            receipt["artifacts"].append(
                {
                    "path": str(path.relative_to(output)),
                    "size": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    _write_json(output / "manifest.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bundle-name", choices=("result", "result_slam"), default="result")
    parser.add_argument("--max-gap-frames", type=int, default=5)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    receipt = export_quality(
        args.reconstruction_root,
        args.output_dir,
        HandQualityPolicy(max_gap_frames=args.max_gap_frames, fps=args.fps),
        args.bundle_name,
    )
    print(json.dumps({"输出": str(args.output_dir), "摘要": receipt["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
