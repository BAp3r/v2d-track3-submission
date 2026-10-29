"""从原传播掩膜派生互斥的可见深度支持；保留歧义区域与每对象来源。"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from experiments.nvidia_v2d.postprocess_scene import SceneInput
from experiments.nvidia_v2d.sharpa_bridge import digest, write_json
from experiments.nvidia_v2d.sharpa_episode import require_new_output


def visible_depth_support(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = np.asarray(masks)
    if source.dtype != bool or source.ndim != 3 or source.shape[0] < 2:
        raise ValueError("需要至少两个同图像坐标的布尔掩膜")
    ambiguous = source.sum(axis=0) > 1
    visible = source & ~ambiguous[None]
    membership = source & ambiguous[None]
    if np.any(visible.sum(axis=0) > 1) or not np.array_equal(visible | membership, source):
        raise ValueError("遮挡来源重建不守恒")
    return visible, ambiguous, membership


def export_visible_support(scene_manifest: Path, output: Path) -> dict[str, Any]:
    scene = json.loads(scene_manifest.read_text())
    if (
        not scene["proof"]["shared_coordinates_proven"]
        or scene["status"] != "assembled_for_inspection"
    ):
        raise ValueError("需要已证明同源相机/图像坐标的场景")
    spec = SceneInput.model_validate_json((scene_manifest.parent / "input.json").read_text())
    require_new_output(
        output,
        [
            scene_manifest.parent,
            spec.source.source_video.parent,
            *(item.job for item in spec.objects),
        ],
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "frames").mkdir()
    started = datetime.now(UTC).isoformat()
    hashes = {
        str(scene_manifest): digest(scene_manifest),
        str(scene_manifest.parent / "input.json"): digest(scene_manifest.parent / "input.json"),
    }
    rows = []
    for frame in range(spec.source.frame_count):
        arrays = []
        sources = []
        for item in spec.objects:
            path = item.job / "reconstruction/masks/1" / f"{frame:06d}.png"
            sha = digest(path)
            hashes[str(path)] = sha
            with Image.open(path) as image:
                arrays.append(np.asarray(image) > 0)
            sources.append({"object_name": item.name, "path": str(path), "sha256": sha})
        original = np.stack(arrays)
        visible, ambiguous, membership = visible_depth_support(original)
        target = output / "frames" / f"{frame:06d}.npz"
        np.savez_compressed(
            target,
            visible_depth_support=visible,
            ambiguous=ambiguous,
            ambiguous_membership=membership,
            source_frame_index=np.asarray(frame),
        )
        rows.append(
            {
                "frame_index": frame,
                "timestamp_s": frame / spec.source.fps,
                "path": str(target.relative_to(output)),
                "sha256": digest(target),
                "sources": sources,
                "original_pixels": original.sum(axis=(1, 2)).tolist(),
                "visible_support_pixels": visible.sum(axis=(1, 2)).tolist(),
                "excluded_per_object_pixels": membership.sum(axis=(1, 2)).tolist(),
                "ambiguous_pixels": int(ambiguous.sum()),
                "mutually_exclusive": bool(np.all(visible.sum(axis=0) <= 1)),
                "original_masks_reconstructed": bool(np.array_equal(visible | membership, original)),
            }
        )
    unchanged = all(digest(Path(path)) == sha for path, sha in hashes.items())
    if not unchanged:
        raise ValueError("处理期间原掩膜或场景凭据变化")
    result = {
        "schema": "nvidia_v2d.visible_depth_support.v1",
        "status": "succeeded",
        "started_at": started,
        "finished_at": datetime.now(UTC).isoformat(),
        "source": spec.source.model_dump(mode="json"),
        "object_names": [item.name for item in spec.objects],
        "source_inputs_sha256": hashes,
        "source_inputs_unchanged": unchanged,
        "frames": rows,
        "ambiguous_frame_count": sum(row["ambiguous_pixels"] > 0 for row in rows),
        "excluded_pixel_observations": sum(sum(row["excluded_per_object_pixels"]) for row in rows),
        "semantic_identity_modified": False,
        "original_masks_modified": False,
        "inpainting_applied": False,
        "occlusion_depth_order_inferred": False,
        "usage": "scale/contact observation filtering only; not replacement semantic masks",
        "说明": [
            "两目标重叠区域标为歧义；逐对象可见深度支持是原掩膜减去歧义区域。",
            "歧义区域保留原对象归属位图，可以逐像素还原每个原掩膜；没有补出被遮挡区域。",
            "该支持图只筛选尺度和接触观测，不改变语义标签、旧掩膜、轨迹或深度前后遮挡顺序。",
        ],
    }
    write_json(output / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export_visible_support(args.scene, args.output)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "ambiguous_frame_count", "excluded_pixel_observations")
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
