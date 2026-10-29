"""手部观测拒绝、来源标记与保守短缺口补齐；不提供接触或几何真值认证。"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[Any]
Record = dict[str, Any]
SOURCE_CODES = {"missing": 0, "observed": 1, "interpolated": 2, "extrapolated": 3, "rejected": 4}


@dataclass(frozen=True)
class HandQualityPolicy:
    """可审计的实验策略；尺度域沿用官方拟合，时序阈值为无量纲启发式。"""

    min_support_pixels: int = 256
    min_scale_ratio: float = 0.2
    max_scale_ratio: float = 5.0
    max_gap_frames: int = 5
    max_gap_seconds: float = 0.25
    fps: float = 30.0
    temporal_depth_ratio: float = 4.0
    temporal_neighbor_ratio: float = 1.25
    max_interpolation_depth_ratio: float = 1.5
    max_interpolation_translation_over_depth: float = 0.25
    max_interpolation_camera_translation_over_depth: float = 0.25
    max_interpolation_camera_rotation_rad: float = math.pi / 4

    def __post_init__(self) -> None:
        if self.min_support_pixels < 1 or self.max_gap_frames < 0:
            raise ValueError("支持像素必须为正，缺口帧数不得为负")
        numbers = [
            getattr(self, key)
            for key in asdict(self)
            if key not in {"min_support_pixels", "max_gap_frames"}
        ]
        if not all(math.isfinite(value) and value > 0 for value in numbers):
            raise ValueError("策略参数必须为有限正数")
        if self.min_scale_ratio >= self.max_scale_ratio:
            raise ValueError("尺度比下界必须小于上界")


def support_bounds(
    encoded_depth: Array, hand_mask: Array, object_mask: Array, geometry_pixels: int
) -> Record:
    """利用交集⊆手掩膜∖物掩膜，求真实有限正深度交集的严格上下界。

    官方未保存渲染交集坐标。大区域有 I 个无效深度、交集大小为 n 时，
    有效交集至少 max(0,n-I)，最多 min(n,V)。不把大区域 V 当实际交集。
    """
    depth = np.asarray(encoded_depth)
    hand, obj = np.asarray(hand_mask, dtype=bool), np.asarray(object_mask, dtype=bool)
    if (
        depth.ndim != 2
        or depth.dtype != np.uint16
        or hand.shape != depth.shape
        or obj.shape != depth.shape
    ):
        raise ValueError("需要同尺寸二维手/物掩膜和官方 uint16 逆深度")
    if isinstance(geometry_pixels, bool) or geometry_pixels < 0:
        raise ValueError("几何交集计数必须为非负整数")
    region = hand & ~obj
    selected = depth[region]
    valid = (selected > 0) & (selected < 65535)
    invalid = int((~valid).sum())
    consistent = geometry_pixels <= len(selected)
    return {
        "region_pixels": int(len(selected)),
        "region_positive_depth_pixels": int(valid.sum()),
        "region_invalid_depth_pixels": invalid,
        "geometry_pixels": int(geometry_pixels),
        "counts_consistent": consistent,
        "exact_intersection_saved": False,
        "positive_support_lower_bound": max(0, geometry_pixels - invalid) if consistent else 0,
        "positive_support_upper_bound": min(geometry_pixels, int(valid.sum())) if consistent else 0,
    }


def axisangle_to_quaternion(value: Array) -> Array:
    """任意批次轴角转 wxyz 四元数。"""
    vector = np.asarray(value, dtype=np.float64)
    theta = np.linalg.norm(vector, axis=-1, keepdims=True)
    denominator = np.where(theta < 1e-12, 1, theta)
    factor = np.where(theta < 1e-12, 0.5, np.sin(theta / 2) / denominator)
    return np.concatenate([np.cos(theta / 2), vector * factor], axis=-1)


def quaternion_to_axisangle(value: Array) -> Array:
    """四元数转主值轴角，统一半球，避免相同旋转绕远路。"""
    q = np.asarray(value, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    q = np.where(q[..., :1] < 0, -q, q)
    sine = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    factor = np.where(
        sine < 1e-12, 2, 2 * np.arctan2(sine, q[..., :1]) / np.where(sine < 1e-12, 1, sine)
    )
    return np.asarray(q[..., 1:] * factor)


def slerp_axisangle(first: Array, last: Array, weight: float) -> Array:
    """按最短旋转路径插值，不对轴角坐标直接求均值。"""
    q0, q1 = axisangle_to_quaternion(first), axisangle_to_quaternion(last)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    omega = np.arccos(np.clip(np.abs(dot), 0, 1))
    sine = np.sin(omega)
    near = sine < 1e-8
    denominator = np.where(near, 1, sine)
    a = np.where(near, 1 - weight, np.sin((1 - weight) * omega) / denominator)
    b = np.where(near, weight, np.sin(weight * omega) / denominator)
    return quaternion_to_axisangle(a * q0 + b * q1)


def axisangle_to_matrix(value: Array) -> Array:
    """Rodrigues 旋转；仅用于单个三维轴角。"""
    vector = np.asarray(value, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = vector / angle
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.asarray(np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew))


def _finite_vector(record: Record, section: str, key: str, size: int) -> Array | None:
    try:
        vector = np.asarray(record[section][key], dtype=np.float64)
        return vector if vector.shape == (size,) and np.isfinite(vector).all() else None
    except (KeyError, ValueError, TypeError):
        return None


def reject_observation(
    record: Record, support: Record | None, policy: HandQualityPolicy
) -> list[str]:
    """只接受真实回归记录；有限正数只是必要条件，不等于几何可信。"""
    reasons: list[str] = []
    cam = _finite_vector(record, "camera", "cam_t", 3)
    if cam is None:
        reasons.append("camera_translation_not_finite")
    elif cam[2] <= 0:
        reasons.append("camera_depth_not_positive")
    for key, size in (("betas", 10), ("global_orient", 3), ("hand_pose", 45)):
        if _finite_vector(record, "mano", key, size) is None:
            reasons.append(f"mano_{key}_not_finite")
    scale = record.get("hand_scale", 1.0)
    if not isinstance(scale, int | float) or not math.isfinite(scale) or scale <= 0:
        reasons.append("hand_scale_not_finite_positive")
    try:
        ratio = float(record["diagnostics"]["scale"])
        if not math.isfinite(ratio) or not policy.min_scale_ratio < ratio < policy.max_scale_ratio:
            reasons.append("alignment_ratio_outside_fit_domain")
    except (KeyError, TypeError, ValueError):
        reasons.append("alignment_ratio_missing")
    if support is None:
        reasons.append("depth_support_unavailable")
    else:
        if not support["counts_consistent"]:
            reasons.append("geometry_support_count_inconsistent")
        if support["geometry_pixels"] < policy.min_support_pixels:
            reasons.append("geometry_support_insufficient")
        if support["positive_support_lower_bound"] < policy.min_support_pixels:
            reasons.append("positive_depth_support_not_guaranteed")
    return reasons


def original_source(frame: int, observed: dict[int, Record], filled: dict[int, Record]) -> str:
    """用原始文件存在性及插值端点判来源；不信任单个 interpolated 布尔量。"""
    if frame in observed:
        return "observed"
    record = filled.get(frame)
    if record is None:
        return "missing"
    info = record.get("interpolation", {})
    previous, following = info.get("prev"), info.get("next")
    if (
        isinstance(previous, int)
        and isinstance(following, int)
        and previous < frame < following
        and previous in observed
        and following in observed
    ):
        return "interpolated"
    if observed and (frame < min(observed) or frame > max(observed)):
        return "extrapolated"
    return "rejected"


def _interpolation_compatible(first: Record, last: Record, policy: HandQualityPolicy) -> bool:
    first_t, last_t = (np.asarray(r["camera"]["cam_t"], dtype=np.float64) for r in (first, last))
    depth = min(float(first_t[2]), float(last_t[2]))
    ratio = max(first_t[2], last_t[2]) / depth
    return bool(
        ratio <= policy.max_interpolation_depth_ratio
        and np.linalg.norm(last_t - first_t) / depth
        <= policy.max_interpolation_translation_over_depth
    )


def _interpolate(first: Record, last: Record, frame: int) -> Record:
    lo, hi = int(first["frame_idx"]), int(last["frame_idx"])
    weight = (frame - lo) / (hi - lo)
    result = copy.deepcopy(first)
    result["frame_idx"] = frame
    result["interpolated"] = True
    result["interpolation"] = {"prev": lo, "next": hi, "weight": weight}
    for key, size in (("global_orient", 3), ("hand_pose", 45)):
        a, b = (np.asarray(r["mano"][key], dtype=np.float64).reshape(-1, 3) for r in (first, last))
        result["mano"][key] = slerp_axisangle(a, b, weight).reshape(size).tolist()
    for section, key in (("camera", "cam_t"), ("mano", "betas")):
        a, b = (np.asarray(r[section][key]) for r in (first, last))
        result[section][key] = ((1 - weight) * a + weight * b).tolist()
    result["bbox"] = {
        key: (1 - weight) * float(first["bbox"][key]) + weight * float(last["bbox"][key])
        for key in ("x0", "y0", "x1", "y1")
    }
    result["diagnostics"] = {"source": "bounded_interpolation", "depth_alignment_performed": False}
    return result


def _visible_support_good(support: Record | None, policy: HandQualityPolicy) -> bool:
    return (
        support is not None and support["region_positive_depth_pixels"] >= policy.min_support_pixels
    )


def _camera_segment_continuous(
    cameras: Array, reference_depth: float, policy: HandQualityPolicy
) -> bool:
    values = np.asarray(cameras, dtype=np.float64)
    normalized_steps = np.linalg.norm(np.diff(values[:, :3, 3], axis=0), axis=1) / reference_depth
    relative = np.swapaxes(values[:-1, :3, :3], 1, 2) @ values[1:, :3, :3]
    angles = np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1))
    return bool(
        np.isfinite(values).all()
        and np.all(normalized_steps <= policy.max_interpolation_camera_translation_over_depth)
        and np.all(angles <= policy.max_interpolation_camera_rotation_rad)
    )


def process_track(
    observed: dict[int, Record],
    filled: dict[int, Record],
    supports: dict[int, Record | None],
    n_frames: int,
    policy: HandQualityPolicy,
    camera_valid: Array | None = None,
    camera_to_world: Array | None = None,
) -> tuple[dict[int, Record], list[Record]]:
    """过滤真实观测后重新插值；原拒绝原因永久保留，既有补齐结果从不直接放行。"""
    if n_frames < 1 or any(not 0 <= frame < n_frames for frame in observed):
        raise ValueError("真实观测帧号必须在给定序列内")
    if camera_to_world is not None and np.asarray(camera_to_world).shape != (n_frames, 4, 4):
        raise ValueError("相机矩阵必须为逐帧4×4数组")
    reasons = {
        frame: reject_observation(record, supports.get(frame), policy)
        for frame, record in observed.items()
    }
    if camera_valid is None:
        camera_valid = np.ones(n_frames, dtype=bool)
    camera_valid = np.asarray(camera_valid, dtype=bool)
    if camera_valid.shape != (n_frames,):
        raise ValueError("相机有效性必须与逐帧长度一致")
    for frame in observed:
        if not camera_valid[frame]:
            reasons[frame].append("camera_transform_not_valid_rigid")
    # 只拒绝两端可信且相互接近的孤立深度尖峰；不使用固定米制真值阈值。
    initially_good = {frame for frame, value in reasons.items() if not value}
    warnings: dict[int, list[str]] = {}
    for frame in sorted(initially_good):
        if frame - 1 not in initially_good or frame + 1 not in initially_good:
            continue
        a, b, c = [
            np.asarray(observed[i]["camera"]["cam_t"], dtype=np.float64)
            for i in (frame - 1, frame, frame + 1)
        ]
        baseline = math.sqrt(float(a[2] * c[2]))
        neighbors_close = max(a[2], c[2]) / min(a[2], c[2]) <= policy.temporal_neighbor_ratio
        neighbors_close &= np.linalg.norm(a - c) / baseline <= 0.25
        ratio = max(float(b[2]) / baseline, baseline / float(b[2]))
        if neighbors_close and ratio > policy.temporal_depth_ratio:
            if camera_to_world is not None and not _camera_segment_continuous(
                camera_to_world[frame - 1 : frame + 2], baseline, policy
            ):
                warnings.setdefault(frame, []).append(
                    "temporal_depth_change_camera_motion_confounded"
                )
            else:
                reasons[frame].append("isolated_temporal_depth_ratio_outlier")
    accepted = {frame: copy.deepcopy(observed[frame]) for frame, why in reasons.items() if not why}
    rows: list[Record] = []
    for frame in range(n_frames):
        source = original_source(frame, observed, filled)
        why = list(reasons.get(frame, []))
        if source in {"interpolated", "extrapolated", "rejected"}:
            why.append("upstream_fill_requires_revalidation")
        rows.append(
            {
                "frame": frame,
                "original_source": source,
                "observation_status": "accepted"
                if frame in accepted
                else "rejected"
                if frame in observed
                else "absent",
                "source": "observed"
                if frame in accepted
                else "rejected"
                if frame in observed or source != "missing"
                else "missing",
                "valid": frame in accepted,
                "is_observed": frame in accepted,
                "rejection_reasons": why,
                "warnings": warnings.get(frame, []),
                "support": supports.get(frame),
                "interpolation": None,
            }
        )
    endpoints = sorted(accepted)
    for lo, hi in zip(endpoints, endpoints[1:], strict=False):
        gap = hi - lo - 1
        if gap < 1 or gap > policy.max_gap_frames or gap / policy.fps > policy.max_gap_seconds:
            continue
        blocked_reason = None
        if not camera_valid[lo : hi + 1].all():
            blocked_reason = "interpolation_camera_invalid"
        elif not _interpolation_compatible(accepted[lo], accepted[hi], policy):
            blocked_reason = "interpolation_endpoint_motion_too_large"
        elif any(
            not _visible_support_good(supports.get(frame), policy) for frame in range(lo + 1, hi)
        ):
            blocked_reason = "interpolation_visible_depth_support_missing"
        if blocked_reason is None and camera_to_world is not None:
            reference_depth = min(
                accepted[lo]["camera"]["cam_t"][2], accepted[hi]["camera"]["cam_t"][2]
            )
            if not _camera_segment_continuous(camera_to_world[lo : hi + 1], reference_depth, policy):
                blocked_reason = "interpolation_camera_discontinuity"
        if blocked_reason:
            for frame in range(lo + 1, hi):
                rows[frame]["rejection_reasons"].append(blocked_reason)
            continue
        for frame in range(lo + 1, hi):
            accepted[frame] = _interpolate(accepted[lo], accepted[hi], frame)
            rows[frame].update(
                source="interpolated",
                valid=True,
                is_observed=False,
                interpolation=accepted[frame]["interpolation"],
            )
    for frame, record in accepted.items():
        record["interpolated"] = rows[frame]["source"] == "interpolated"
        record["quality"] = {
            key: rows[frame][key]
            for key in (
                "source",
                "original_source",
                "is_observed",
                "observation_status",
                "rejection_reasons",
            )
        }
    return accepted, rows


def main() -> None:
    """命令入口；延后导入文件层，核心策略便于独立单测。"""
    from experiments.nvidia_v2d.hand_quality_io import main as io_main

    io_main()


if __name__ == "__main__":
    main()
