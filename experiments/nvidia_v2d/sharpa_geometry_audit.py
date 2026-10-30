"""Read-only sampled hand/object surface audit; never a physical contact oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

SOURCE_LABELS = ("missing", "observed", "interpolated", "extrapolated", "rejected")


def build_training_gate(
    audit_report: Mapping[str, Any],
    *,
    numerical_contract_passed: bool,
    purpose: Literal["mechanism_validation", "policy_quality"] = "policy_quality",
) -> dict[str, Any]:
    """Separate runnable numerical inputs from unproven geometric/policy quality.

    An explicit mechanism-validation purpose permits bounded training only when
    every input frame is numerically usable. Sampled distances never establish
    geometric quality, even for a closed mesh or zero measured penetration.
    """
    if purpose not in ("mechanism_validation", "policy_quality"):
        raise ValueError("训练用途必须显式区分机制验证与策略质量验证")
    if not isinstance(numerical_contract_passed, bool):
        raise ValueError("数值契约结果必须为布尔值")
    reasons: list[str] = []
    if not numerical_contract_passed:
        reasons.append("上游输入数值契约未通过")
    if audit_report.get("status") != "succeeded" or audit_report.get("source_unchanged") is not True:
        reasons.append("几何诊断未完成或原输入哈希未确认不变")
    frame_count = audit_report.get("frame_count", 0)
    if not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count < 1:
        reasons.append("有效序列长度缺失")
    for side in ("left", "right"):
        hand = audit_report.get("hands", {}).get(side)
        if (
            not isinstance(hand, dict)
            or hand.get("valid_frames") != frame_count
            or hand.get("excluded_summary_frames") != []
            or hand.get("nonfinite_frames") != []
        ):
            reasons.append(f"{side} 存在无效、非有限、拒绝、外推或缺失帧，不能整段作为监督")
    objects = audit_report.get("objects", [])
    if not objects:
        reasons.append("物体几何信息缺失")
    penetration_unknown = not objects or any(
        obj.get("sampled_penetration_measurable") is not True for obj in objects
    )
    numeric_ok = not reasons
    # Geometry quality is reported, never a training blocker; only numerical
    # validity decides readiness for either purpose.
    mechanism = True
    limitations = ["有限表面采样不能证明正确接触、摩擦、完整表面无穿透或抓持稳定性"]
    if penetration_unknown:
        limitations.append("开放或无效物体网格导致穿透未知，几何质量未验收（不阻断训练）")
    return {
        "schema_version": "nvidia_v2d.sharpa_training_gate.v1",
        "purpose": purpose,
        "numerical_input_status": "passed" if numeric_ok else "failed",
        "input_contract_passed": numeric_ok,
        "geometry_quality_status": "unknown",
        "geometry_quality_accepted": False,
        "penetration_status": "unknown" if penetration_unknown else "sampled_only",
        "mechanism_validation": mechanism,
        "training_ready": numeric_ok and mechanism,
        "policy_quality_claim_allowed": False,
        "blockers": reasons,
        "limitations_zh": limitations,
        "allowed_claim_zh": "仅验证训练、保存、恢复、无辅助评估和实际物体轨迹导出机制；不能宣称策略质量或抓持可靠性。",
        "frame_count": frame_count,
        "retarget_dir": audit_report.get("retarget_dir"),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_poses(poses: NDArray[np.float64]) -> None:
    """Reject scaling, reflections, and invalid homogeneous world transforms."""
    if poses.ndim != 4 or poses.shape[-2:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError("物体姿态必须是有限的 [帧, 物体, 4, 4] 世界变换")
    if not np.allclose(poses[..., 3, :], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("物体姿态齐次矩阵末行错误")
    rotations = poses[..., :3, :3]
    if not np.allclose(rotations.swapaxes(-1, -2) @ rotations, np.eye(3), atol=1e-5):
        raise ValueError("物体姿态必须为刚体变换；米制 mesh 不得再次缩放")
    if not np.allclose(np.linalg.det(rotations), 1, atol=1e-5):
        raise ValueError("物体姿态包含反射或错误旋转")


def sampled_surface_distances(
    mesh: Any, points_world: NDArray[np.float64], pose_world: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Return unsigned distances and sampled penetration depths in metres.

    Points are transformed into the object's local frame, leaving the original
    mesh and its cached spatial index unchanged. Open surfaces have no inside.
    """
    import trimesh

    points = np.asarray(points_world, dtype=np.float64)
    pose = np.asarray(pose_world, dtype=np.float64)
    validate_poses(pose[None, None])
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("手部表面点必须是有限的 [采样点, 3] 世界坐标")
    local = (points - pose[:3, 3]) @ pose[:3, :3]
    # Far-away points can select many triangles; bound temporary query memory.
    distances = np.concatenate(
        [
            trimesh.proximity.closest_point(mesh, local[start : start + 32])[1]
            for start in range(0, len(local), 32)
        ]
    )
    if not np.isfinite(distances).all():
        raise ValueError("最近表面距离含非有限值")
    penetration = None
    if mesh.is_watertight and mesh.is_winding_consistent:
        penetration = np.where(mesh.contains(local), distances, 0.0)
    return distances, penetration


def _stats(values: NDArray[np.float64]) -> dict[str, float | int | None]:
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        return {"count": 0, "min_m": None, "median_m": None, "p95_m": None, "max_m": None}
    return {
        "count": int(finite.size),
        "min_m": float(finite.min()),
        "median_m": float(np.median(finite)),
        "p95_m": float(np.quantile(finite, 0.95)),
        "max_m": float(finite.max()),
    }


def _load_inputs(run_dir: Path) -> tuple[dict[str, Any], dict[str, NDArray[Any]], list[Path]]:
    preview_path, data_path = run_dir / "preview.json", run_dir / "diagnostics.npz"
    preview = json.loads(preview_path.read_text())
    with np.load(data_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    time = np.asarray(arrays["timestamps_s"], dtype=np.float64)
    if time.ndim != 1 or not len(time) or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("时间戳必须有限且严格递增")
    other_time = np.asarray(preview["timestamps_s"], dtype=np.float64)
    if time.shape != other_time.shape or not np.allclose(time, other_time, atol=1e-9, rtol=0):
        raise ValueError("预览和诊断时间轴不一致")
    names = preview["object_names"]
    paths = [Path(p).expanduser() for p in preview["object_mesh_paths"]]
    paths = [(path if path.is_absolute() else run_dir / path).resolve() for path in paths]
    poses = np.asarray(preview["object_poses_world"], dtype=np.float64)
    validate_poses(poses)
    if poses.shape[:2] != (len(time), len(paths)) or len(paths) != len(names) or not len(paths):
        raise ValueError("物体名称、网格和姿态数量不一致")
    if len(set(names)) != len(names):
        raise ValueError("物体名称必须唯一")
    for side in ("left", "right"):
        samples = arrays[f"{side}_world_surface_samples_m"]
        valid, codes = arrays[f"{side}_valid"], arrays[f"{side}_source_code"]
        if (
            samples.ndim != 3
            or samples.shape[0] != len(time)
            or samples.shape[2] != 3
            or samples.shape[1] < 1
        ):
            raise ValueError("表面采样形状必须为 [帧, 采样点, 3]")
        if valid.shape != (len(time),) or valid.dtype.kind != "b":
            raise ValueError("逐帧有效标记必须为布尔值")
        if (
            codes.shape != (len(time),)
            or codes.dtype.kind not in "iu"
            or not np.isin(codes, range(5)).all()
        ):
            raise ValueError("观测来源标记不符合契约")
        if (
            preview.get(f"valid_{side}") != valid.tolist()
            or preview.get(f"source_code_{side}") != codes.tolist()
        ):
            raise ValueError("预览和诊断质量标记不一致")
    return preview, arrays, [preview_path, data_path, *paths]


def audit_retarget(run_dir: Path, output_dir: Path, near_surface_m: float = 0.005) -> dict[str, Any]:
    """Write new diagnostics without repairing, scaling, or overwriting inputs."""
    import trimesh

    if not np.isfinite(near_surface_m) or near_surface_m <= 0:
        raise ValueError("邻近表面阈值必须为有限正数")
    run_dir, output_dir = run_dir.resolve(), output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("几何审计只允许写入新的输出目录")
    if output_dir == run_dir or run_dir in output_dir.parents:
        raise ValueError("几何审计不得写入已有重定向产物目录")
    preview, arrays, input_paths = _load_inputs(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest_path.exists():
        input_paths.append(manifest_path)
    before = {str(path): sha256(path) for path in input_paths}
    meshes = [
        trimesh.load_mesh(path, process=False)
        for path in input_paths[2 : 2 + len(preview["object_names"])]
    ]
    for mesh in meshes:
        if (
            not isinstance(mesh, trimesh.Trimesh)
            or not len(mesh.faces)
            or not np.isfinite(mesh.vertices).all()
        ):
            raise ValueError("物体资产必须为有限且非空的单一三角网格")
    poses = np.asarray(preview["object_poses_world"], dtype=np.float64)
    time = arrays["timestamps_s"]
    details: dict[str, NDArray[Any]] = {"timestamps_s": time}
    report: dict[str, Any] = {
        "schema": "nvidia_v2d.sharpa_geometry_audit.v1",
        "status": "succeeded",
        "created_at": datetime.now(UTC).isoformat(),
        "retarget_dir": str(run_dir),
        "backend": preview["backend"],
        "frame_count": len(time),
        "duration_s": float(time[-1] - time[0]),
        "near_surface_threshold_m": near_surface_m,
        "mesh_scale_applied": 1.0,
        "physical_rollout": False,
        "contact_truth": False,
        "exact_collision_test": False,
        "limitations_zh": [
            "固定有限手表面采样的几何距离，不是精确最小距离、仿真接触、抓持稳定性或真实接触标签。",
            "采样遗漏可能漏报穿透；只对封闭且面朝向一致网格计算采样点在内部的深度。",
            "开放网格不修补，不以凸包替代；穿透标为不可测，而不是零。",
            "物体与手已在共享米制坐标系中；不再次应用形态比例，也不调整世界腕轨迹。",
            "全部有限帧保留诊断，但正常汇总只纳入有效的真实观测或插值，来源分别统计。",
        ],
        "objects": [],
        "hands": {},
        "input_sha256": before,
    }
    for index, mesh in enumerate(meshes):
        measurable = bool(mesh.is_watertight and mesh.is_winding_consistent)
        report["objects"].append(
            {
                "index": index,
                "name": preview["object_names"][index],
                "mesh_path": preview["object_mesh_paths"][index],
                "vertex_count": len(mesh.vertices),
                "face_count": len(mesh.faces),
                "watertight": bool(mesh.is_watertight),
                "winding_consistent": bool(mesh.is_winding_consistent),
                "sampled_penetration_measurable": measurable,
                "penetration_unavailable_reason": None
                if measurable
                else "网格开放或面朝向不一致，内部定义不可靠",
            }
        )
    for side in ("left", "right"):
        samples = arrays[f"{side}_world_surface_samples_m"]
        finite = np.isfinite(samples).all(axis=(1, 2))
        valid, codes = arrays[f"{side}_valid"], arrays[f"{side}_source_code"]
        accepted = valid & finite & np.isin(codes, [1, 2])
        distance = np.full((len(time), len(meshes), samples.shape[1]), np.nan)
        penetration = np.full_like(distance, np.nan)
        for frame in np.flatnonzero(finite):
            for index, mesh in enumerate(meshes):
                d, p = sampled_surface_distances(mesh, samples[frame], poses[frame, index])
                distance[frame, index] = d
                if p is not None:
                    penetration[frame, index] = p
        nearest = distance.min(axis=1)
        details.update(
            {
                f"{side}_object_surface_distance_m": distance,
                f"{side}_object_sampled_penetration_depth_m": penetration,
                f"{side}_nearest_object_distance_m": nearest,
                f"{side}_finite_samples": finite,
                f"{side}_valid": valid,
                f"{side}_source_code": codes,
                f"{side}_accepted_summary_frame": accepted,
            }
        )
        hand: dict[str, Any] = {
            "sample_count": samples.shape[1],
            "source_sampling": manifest.get("hands", {}).get(side, {}).get("surface_sampling"),
            "valid_frames": int(valid.sum()),
            "nonfinite_frames": np.flatnonzero(~finite).tolist(),
            "excluded_summary_frames": np.flatnonzero(~accepted).tolist(),
            "nearest_object_sample_distance": _stats(nearest[accepted]),
            "frame_minimum_sample_distance": _stats(nearest[accepted].min(axis=-1)),
            "sources": {},
            "objects": [],
        }
        for code, label in enumerate(SOURCE_LABELS):
            mask = finite & (codes == code)
            hand["sources"][label] = {
                "frames": int((codes == code).sum()),
                "valid_frames": int((mask & valid).sum()),
                "nearest_object_sample_distance": _stats(nearest[mask]),
            }
        for index, obj in enumerate(report["objects"]):
            d, p = distance[accepted, index], penetration[accepted, index]
            finite_p = p[np.isfinite(p)]
            measurable = obj["sampled_penetration_measurable"]
            hand["objects"].append(
                {
                    "name": obj["name"],
                    "sample_distance": _stats(d),
                    "frame_minimum_sample_distance": _stats(d.min(axis=-1)),
                    "frames_with_sample_within_threshold": int(
                        np.any(d <= near_surface_m, axis=-1).sum()
                    ),
                    "sampled_penetration": {
                        "measurable": measurable,
                        "inside_sample_fraction": float(np.mean(finite_p > 1e-8))
                        if len(finite_p)
                        else None,
                        "max_depth_m": float(finite_p.max()) if len(finite_p) else None,
                        "frames_with_inside_samples": int(np.any(p > 1e-8, axis=-1).sum())
                        if measurable
                        else None,
                    },
                }
            )
        report["hands"][side] = hand
    after = {str(path): sha256(path) for path in input_paths}
    if before != after:
        raise RuntimeError("审计期间输入文件发生变化，拒绝交付混合快照")
    report["source_unchanged"] = True
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output_dir / "surface-audit.npz", **details)
    report["diagnostics_sha256"] = sha256(output_dir / "surface-audit.npz")
    (output_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# Sharpa 手物表面几何审计",
        "",
        f"后端：{report['backend']}；{len(time)} 帧；时长 {report['duration_s']:.3f} 秒。",
        "",
    ]
    for obj in report["objects"]:
        lines.append(
            f"- {obj['name']}：{obj['face_count']} 个三角形；封闭网格={obj['watertight']}；采样穿透可测={obj['sampled_penetration_measurable']}。"
        )
    for side, hand in report["hands"].items():
        stats = hand["frame_minimum_sample_distance"]
        lines.extend(
            [
                "",
                f"{'左手' if side == 'left' else '右手'}：每帧 {hand['sample_count']} 个固定表面采样点。有效帧 {hand['valid_frames']}，汇总排除 {len(hand['excluded_summary_frames'])} 帧。",
                f"每帧最近采样点距离（米）：{json.dumps(stats, ensure_ascii=False)}。",
            ]
        )
    lines.extend(
        [
            "",
            "局限：",
            *[f"- {s}" for s in report["limitations_zh"]],
            "",
            "输入与 mesh 文件 SHA256 审计前后完全相同。逐帧数值、来源和有效性均保留在 surface-audit.npz。",
        ]
    )
    (output_dir / "审查报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="只读 Sharpa 手物采样表面距离审计")
    parser.add_argument("--retarget-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--near-surface-m", type=float, default=0.005)
    args = parser.parse_args()
    report = audit_retarget(args.retarget_dir, args.output_dir, args.near_surface_m)
    print(
        json.dumps(
            {"status": report["status"], "output_dir": str(args.output_dir)}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
