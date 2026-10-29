"""遮挡支持 A/B 的数值服务：固定旋转，保留拒绝，不把残差当接触真值。"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import Field
from scipy import ndimage

from experiments.nvidia_v2d.depth_source_ab import CameraK
from experiments.nvidia_v2d.postprocess_depth import (
    DepthRefitConfig,
    fit_gate,
    fit_scale_translation,
)
from experiments.nvidia_v2d.postprocess_occlusion import visible_depth_support


class SupportRefitConfig(DepthRefitConfig):
    min_visible_fraction: float = Field(default=0.1, gt=0, le=1)


def validate_support_bundle(
    original: np.ndarray, visible: np.ndarray, ambiguous: np.ndarray, membership: np.ndarray
) -> None:
    expected = visible_depth_support(original)
    for actual, reference in zip((visible, ambiguous, membership), expected, strict=True):
        if actual.dtype != bool or not np.array_equal(actual, reference):
            raise ValueError("支持图与原掩膜不守恒或包含歧义污染")


def fit_supported_frame(
    rendered: np.ndarray,
    observed: np.ndarray,
    support: np.ndarray,
    original: np.ndarray,
    camera: CameraK,
    translation: np.ndarray,
    config: SupportRefitConfig,
) -> dict[str, Any]:
    """由支持像素形成对应，过度遮挡帧保持拒绝；绝不扩张或补洞。"""
    shape = (camera.height, camera.width)
    if any(a.shape != shape for a in (rendered, observed, support, original)):
        raise ValueError("深度、支持与内参尺寸不一致")
    if support.dtype != bool or original.dtype != bool or np.any(support & ~original):
        raise ValueError("支持必须是原布尔掩膜的子集")
    shared = support & np.isfinite(observed) & (observed > 0)
    shared &= np.isfinite(rendered) & (rendered > 0)
    count, support_count, original_count = int(shared.sum()), int(support.sum()), int(original.sum())
    reasons = []
    labels, _ = ndimage.label(support)
    sizes = np.bincount(labels.ravel())[1:]
    largest_fraction = float(sizes.max() / support_count) if support_count else 0.0
    if largest_fraction < config.min_largest_component_fraction:
        reasons.append("支持明显碎裂或为空")
    if support_count / max(original_count, 1) < config.min_visible_fraction:
        reasons.append("相对原目标可见支持不足")
    if count < config.min_overlap_pixels:
        reasons.append("深度与网格共同支持不足")
    if count / max(support_count, 1) < config.min_mask_coverage:
        reasons.append("网格对支持覆盖不足")
    if (
        int((support & np.isfinite(observed) & (observed > 0)).sum()) / max(support_count, 1)
        < config.min_depth_coverage
    ):
        reasons.append("有效深度支持不足")
    result: dict[str, Any] = {
        "original_pixels": original_count,
        "support_pixels": support_count,
        "shared_pixels": count,
        "largest_component_fraction": largest_fraction,
        "rejection_reasons_zh": reasons,
        "fit": None,
    }
    if reasons:
        return result
    y, x = np.nonzero(shared)
    rays = np.column_stack(
        ((x - camera.cx) / camera.fx, (y - camera.cy) / camera.fy, np.ones(len(x)))
    )
    try:
        fit = fit_scale_translation(
            rays * rendered[shared, None] - translation, rays * observed[shared, None]
        )
        reasons.extend(fit_gate(fit, config, translation))
        result["fit"] = fit
    except ValueError as exc:
        reasons.append(str(exc))
    return result


def shared_evaluation(
    rendered: dict[str, np.ndarray], observed: np.ndarray, visible: np.ndarray
) -> dict[str, Any]:
    """所有候选与旧模型用完全同一组像素评价，避免换支持集造成假改善。"""
    if not rendered or visible.dtype != bool or observed.shape != visible.shape:
        raise ValueError("共同评估输入无效")
    shared = visible & np.isfinite(observed) & (observed > 0)
    for value in rendered.values():
        if value.shape != observed.shape:
            raise ValueError("候选深度尺寸不一致")
        shared &= np.isfinite(value) & (value > 0)
    count = int(shared.sum())
    return {
        "shared_pixels": count,
        "visible_coverage": count / max(int(visible.sum()), 1),
        "rmse_m": {
            name: float(np.sqrt(np.mean((value[shared] - observed[shared]) ** 2))) if count else None
            for name, value in rendered.items()
        },
        "independent_ground_truth": False,
    }
