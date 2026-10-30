"""Sharpa 离线输入契约：真实时间、有效性及缩放几何；不认证物理精度。"""

from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

SOURCE_LABELS = ("missing", "observed", "interpolated", "extrapolated", "rejected")


class EpisodeSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sequence_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    object_names: list[str] = Field(min_length=1)
    fps: float = Field(gt=0, le=240, allow_inf_nan=False)
    frame_count: int = Field(gt=1, le=1_000_000)
    source_video: Path
    source_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class HandAudit(BaseModel):
    valid: list[bool]
    source_code: list[int]
    source_counts: dict[str, int]
    metric_scale_min: float | None = None
    metric_scale_max: float | None = None


class EpisodeAudit(BaseModel):
    schema_version: Literal["nvidia_v2d.sharpa_episode_gate.v1"] = "nvidia_v2d.sharpa_episode_gate.v1"
    source: EpisodeSource
    input_contract_passed: bool
    geometry_quality_accepted: Literal[False] = False
    blockers: list[str]
    source_frame_indices: list[int]
    timestamps_s: list[float]
    last_frame_time_s: float
    video_duration_s: float
    timestamp_source: Literal["source_video_cfr"] = "source_video_cfr"
    hands: dict[str, HandAudit]


def _rigid(poses: NDArray) -> bool:
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():
        return False
    rotations = poses[:, :3, :3]
    return bool(
        np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-5, rtol=0)
        and np.allclose(np.einsum("nji,njk->nik", rotations, rotations), np.eye(3), atol=1e-4)
        and np.allclose(np.linalg.det(rotations), 1, atol=1e-4)
    )


def _runs(mask: NDArray) -> list[tuple[int, int]]:
    ends = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(
        zip(np.flatnonzero(ends == 1).tolist(), np.flatnonzero(ends == -1).tolist(), strict=False)
    )


def audit_bundle(source: EpisodeSource, arrays: dict[str, NDArray], *, allow_inferred_hands: bool = False) -> EpisodeAudit:
    """检查完整片段资格；不删除帧、不把单端填充改称观测。"""
    n = source.frame_count
    blockers: list[str] = []
    hands: dict[str, HandAudit] = {}
    if len(source.object_names) != 1:
        blockers.append("multi_object_scene_not_assembled")
    for kind in ("camera", "object"):
        pose = arrays.get(f"{kind}_to_world_transform", np.empty(0))
        valid = arrays.get(f"{kind}_is_valid", np.empty(0))
        if pose.shape != (n, 4, 4) or not _rigid(pose):
            blockers.append(f"{kind}:nonrigid_transform")
        if valid.shape != (n,) or valid.dtype.kind != "b" or not valid.all():
            blockers.append(f"{kind}:unavailable_frames")
    object_scale = arrays.get("object_scale", np.empty(0))
    if object_scale.size != 1 or not np.isfinite(object_scale).all() or object_scale.item() <= 0:
        blockers.append("object:invalid_scale")
    for side in ("left", "right"):
        prefix = "hand_" + side
        valid = arrays.get(prefix + "_is_valid", np.zeros(n, dtype=bool))
        observed = arrays.get(prefix + "_is_observed", np.zeros(n, dtype=bool))
        codes = arrays.get(prefix + "_source_code", np.zeros(n, dtype=np.uint8))
        if (
            valid.shape != (n,)
            or valid.dtype.kind != "b"
            or observed.shape != (n,)
            or observed.dtype.kind != "b"
            or codes.shape != (n,)
            or codes.dtype.kind not in "iu"
            or not np.isin(codes, range(5)).all()
        ):
            blockers.append(f"{side}:invalid_source_arrays")
            continue
        if not np.array_equal(observed, codes == 1) or not np.array_equal(
            valid, np.isin(codes, [1, 2, 3])
        ):
            blockers.append(f"{side}:source_flags_disagree")
        if not valid.all():
            blockers.append(f"{side}:unavailable_hand_frames")
        if np.any(codes == 3) and not allow_inferred_hands:
            blockers.append(f"{side}:extrapolated_frames")
        for start, end in _runs(codes == 2):
            if (
                start == 0
                or end == n
                or end - start > 5
                or (end - start) / source.fps > 0.25 + 1e-9
                or codes[start - 1] != 1
                or codes[end] != 1
            ):
                if not allow_inferred_hands:
                    blockers.append(f"{side}:unsupported_interpolation")
                break
        expected = {
            "_betas": (10,),
            "_wrist_orient_in_camera": (n, 3),
            "_wrist_trans_in_camera": (n, 3),
            "_scale": (n,),
        }
        for suffix, shape in expected.items():
            value = arrays.get(prefix + suffix, np.empty(0))
            if value.shape != shape or not np.isfinite(value).all():
                blockers.append(f"{side}:invalid{suffix}")
        finger = arrays.get(prefix + "_finger_pose", np.empty(0))
        if finger.shape not in ((n, 15, 3), (n, 45)) or not np.isfinite(finger).all():
            blockers.append(f"{side}:invalid_finger_pose")
        scale = arrays.get(prefix + "_scale", np.empty(0))
        scale_ok = scale.shape == (n,) and np.isfinite(scale).all() and (scale > 0).all()
        if not scale_ok:
            blockers.append(f"{side}:invalid_metric_scale")
        translation = arrays.get(prefix + "_wrist_trans_in_camera", np.empty(0))
        if translation.shape == (n, 3) and np.any(translation[valid, 2] <= 0):
            blockers.append(f"{side}:nonpositive_observed_depth")
        hands[side] = HandAudit(
            valid=valid.tolist(),
            source_code=codes.tolist(),
            source_counts={
                name: int((codes == code).sum()) for code, name in enumerate(SOURCE_LABELS)
            },
            metric_scale_min=float(scale.min()) if scale_ok else None,
            metric_scale_max=float(scale.max()) if scale_ok else None,
        )
    times = (np.arange(n, dtype=float) / source.fps).tolist()
    return EpisodeAudit(
        source=source,
        input_contract_passed=not blockers,
        blockers=list(dict.fromkeys(blockers)),
        hands=hands,
        source_frame_indices=list(range(n)),
        timestamps_s=times,
        last_frame_time_s=times[-1],
        video_duration_s=n / source.fps,
    )


def require_new_output(output: Path, input_roots: list[Path]) -> None:
    """输出不能已有，不能进入任何源目录，不能经过符号链接。"""
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"输出必须全新：{output}")
    if not output.is_absolute() or any(p.is_symlink() for p in output.parents):
        raise ValueError("输出必须为不经过符号链接的绝对路径")
    resolved = output.resolve()
    if any(resolved.is_relative_to(p.resolve()) for p in input_roots):
        raise ValueError("输出不能位于输入目录内")


def scale_about_centroid(
    vertices: NDArray,
    joints: NDArray,
    scales: NDArray,
) -> tuple[NDArray, NDArray]:
    """精确复现 HaMeR renderer：围绕网格质心缩放，随后才添加 cam_t。"""
    center = np.asarray(vertices, dtype=np.float64).mean(axis=1, keepdims=True)
    factors = np.asarray(scales, dtype=np.float64)[:, None, None]
    return (vertices - center) * factors + center, (joints - center) * factors + center


def transform_points(poses: NDArray, points: NDArray) -> NDArray:
    return np.einsum("tij,tkj->tki", poses[:, :3, :3], points) + poses[:, None, :3, 3]
