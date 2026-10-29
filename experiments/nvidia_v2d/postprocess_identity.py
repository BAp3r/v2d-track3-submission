"""两个已命名目标的只读身份异常诊断；候选警报不自动交换标签。"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from experiments.nvidia_v2d.postprocess_scene import SceneInput
from experiments.nvidia_v2d.sharpa_bridge import digest, write_json
from experiments.nvidia_v2d.sharpa_episode import require_new_output


def mask_features(mask: np.ndarray) -> dict[str, Any]:
    y, x = np.nonzero(mask)
    labels, count = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())[1:]
    return {
        "pixels": len(x),
        "centroid_normalized": [float(x.mean() / mask.shape[1]), float(y.mean() / mask.shape[0])]
        if len(x)
        else None,
        "largest_component_fraction": float(sizes.max() / len(x)) if len(x) else 0.0,
        "components": count,
    }


def cross_assignment_candidate(
    previous: list[dict[str, Any]], current: list[dict[str, Any]]
) -> dict[str, Any]:
    if (
        len(previous) != 2
        or len(current) != 2
        or any(row["centroid_normalized"] is None for row in previous + current)
    ):
        return {"possible_swap": False, "same_cost": None, "cross_cost": None}
    a = np.asarray([row["centroid_normalized"] for row in previous])
    b = np.asarray([row["centroid_normalized"] for row in current])
    same = float(np.linalg.norm(a - b, axis=1).sum())
    cross = float(np.linalg.norm(a - b[::-1], axis=1).sum())
    return {
        "possible_swap": same > 0.04 and cross < same * 0.5,
        "same_cost": same,
        "cross_cost": cross,
    }


def audit_identity(spec: SceneInput, output: Path) -> dict[str, Any]:
    require_new_output(
        output, [spec.source.source_video.parent, *(item.job for item in spec.objects)]
    )
    if len(spec.objects) != 2:
        raise ValueError("身份交叉诊断当前只支持两个显式命名目标")
    records, inputs, previous = [], {}, None
    for frame in range(spec.source.frame_count):
        masks, features = [], []
        for item in spec.objects:
            path = item.job / "reconstruction/masks/1" / f"{frame:06d}.png"
            if not path.is_file():
                raise ValueError(f"缺少原始传播掩膜：{path}")
            inputs[str(path)] = digest(path)
            with Image.open(path) as image:
                image = image.resize((320, 240), Image.Resampling.NEAREST)
                mask = np.asarray(image) > 0
            masks.append(mask)
            features.append(mask_features(mask))
        intersection = int((masks[0] & masks[1]).sum())
        union = int((masks[0] | masks[1]).sum())
        smaller = min(row["pixels"] for row in features)
        overlap = intersection / max(smaller, 1)
        assignment = (
            cross_assignment_candidate(previous, features) if previous else {"possible_swap": False}
        )
        flags = []
        if overlap > 0.25:
            flags.append("目标传播掩膜相互覆盖超过较小目标的25%，可能存在遮挡或污染")
        if any(row["largest_component_fraction"] < 0.8 for row in features):
            flags.append("目标为空或明显碎裂")
        if assignment["possible_swap"]:
            flags.append("质心跨标签匹配优于同标签，需人工核对身份交换")
        if previous:
            ratios = [
                row["pixels"] / max(old["pixels"], 1)
                for row, old in zip(features, previous, strict=True)
            ]
            if any(ratio > 2 or ratio < 0.5 for ratio in ratios):
                flags.append("相邻帧目标面积突变超过两倍，需检查遮挡或串轨")
        records.append(
            {
                "frame_index": frame,
                "time_s": frame / spec.source.fps,
                "objects": {
                    item.name: feat for item, feat in zip(spec.objects, features, strict=True)
                },
                "intersection_over_union": intersection / max(union, 1),
                "overlap_fraction_of_smaller": overlap,
                "assignment_diagnostic": assignment,
                "flags_zh": flags,
            }
        )
        previous = features
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "nvidia_v2d.identity_audit.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "diagnosed",
        "source": spec.source.model_dump(mode="json"),
        "source_masks_sha256": inputs,
        "frames": records,
        "possible_swap_frames": [
            row["frame_index"] for row in records if row["assignment_diagnostic"]["possible_swap"]
        ],
        "overlap_warning_frames": [
            row["frame_index"] for row in records if row["overlap_fraction_of_smaller"] > 0.25
        ],
        "identity_verified": False,
        "automatic_relabel_applied": False,
        "说明": [
            "这里检查传播掩膜的交叠、碎裂、面积与质心连续性；它们是审查线索，不是物体身份真值。",
            "遮挡、真实接触和投影穿越也会触发警报；没有自动交换标签或改变旧产物。",
        ],
    }
    if not all(digest(Path(path)) == sha for path, sha in inputs.items()):
        raise ValueError("原掩膜在诊断期间变化")
    result["source_inputs_unchanged"] = True
    write_json(output / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_identity(SceneInput.model_validate_json(args.input.read_text()), args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "possible_swap_frames": result["possible_swap_frames"],
                "overlap_warning_count": len(result["overlap_warning_frames"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
