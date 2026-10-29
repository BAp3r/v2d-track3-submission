"""将已保存的互斥可见支持实际接入尺度/平移重估，与原掩膜做同像素对照。"""

from __future__ import annotations

import argparse
import json
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from experiments.nvidia_v2d.depth_source_ab import (
    CameraK,
    decode_depth,
    rasterize_depth,
    reproject_depth,
)
from experiments.nvidia_v2d.postprocess_depth import _camera_small, _mesh, fit_gate, fixed_scale_refit
from experiments.nvidia_v2d.postprocess_scene import SceneInput
from experiments.nvidia_v2d.postprocess_support_fit import (
    SupportRefitConfig,
    fit_supported_frame,
    shared_evaluation,
    validate_support_bundle,
)
from experiments.nvidia_v2d.sharpa_bridge import digest, write_json
from experiments.nvidia_v2d.sharpa_episode import _rigid, require_new_output

BRANCHES = ("original_mask", "visible_support")
ADAPTER_METADATA = {
    "name": "nvidia-v2d-visible-support-refit",
    "version": "1.0",
    "input_requirements": ["proven shared scene", "hash-bound visible support", "depth and mesh"],
    "supported_input_scopes": ["single_episode"],
    "supported_modes": ["fixed_rotation_scale_translation_ab"],
    "output_schema": "nvidia_v2d.visible_support_refit.v1",
}


def _finish(output: Path, result: dict[str, Any]) -> None:
    write_json(output / "manifest.json", result)
    temporary = output / "run.json.tmp"
    write_json(temporary, result)
    temporary.replace(output / "run.json")


def _read_frame(
    frame: int, context: dict[str, Any]
) -> tuple[np.ndarray, dict[str, str], list[np.ndarray]]:
    row = context["rows"][frame]
    path = context["support_root"] / row["path"]
    hashes = {str(path): digest(path)}
    if hashes[str(path)] != row["sha256"]:
        raise ValueError("逐帧支持凭据不匹配")
    originals = []
    for item, expected in zip(context["objects"], row["sources"], strict=True):
        mask = Path(item["job"]) / "reconstruction/masks/1" / f"{frame:06d}.png"
        hashes[str(mask)] = digest(mask)
        if (str(mask), hashes[str(mask)], item["name"]) != (
            expected["path"],
            expected["sha256"],
            expected["object_name"],
        ):
            raise ValueError("原掩膜身份或SHA改变")
        with Image.open(mask) as image:
            originals.append(np.asarray(image) > 0)
    original = np.stack(originals)
    with np.load(path, allow_pickle=False) as bundle:
        if bundle["source_frame_index"].item() != frame:
            raise ValueError("支持图帧编号错位")
        visible = bundle["visible_depth_support"]
        validate_support_bundle(
            original, visible, bundle["ambiguous"], bundle["ambiguous_membership"]
        )
    stride = context["config"].stride
    depths = []
    for item in context["objects"]:
        root = Path(item["job"]) / "reconstruction"
        depth_path, k_path = (
            root / "depth" / f"{frame:06d}.png",
            root / "intrinsics" / f"{frame:06d}.json",
        )
        for value in (depth_path, k_path):
            hashes[str(value)] = digest(value)
        source_k = CameraK.model_validate_json(k_path.read_text())
        depths.append(
            reproject_depth(
                decode_depth(depth_path)[::stride, ::stride],
                _camera_small(source_k, stride),
                context["camera"],
            )
        )
    return np.stack((original[:, ::stride, ::stride], visible[:, ::stride, ::stride])), hashes, depths


def _initial(frame: int, *, context: dict[str, Any]) -> dict[str, Any]:
    masks, hashes, depths = _read_frame(frame, context)
    items = []
    for i, mesh in enumerate(context["meshes"]):
        pose = context["poses"][frame, i]
        if not context["object_valid"][frame, i] or not context["camera_valid"][frame]:
            items.append(
                {
                    "branches": {
                        name: {"fit": None, "rejection_reasons_zh": ["原相机或物体帧无效"]}
                        for name in BRANCHES
                    }
                }
            )
            continue
        rendered = rasterize_depth(mesh[0] @ pose[:3, :3].T + pose[:3, 3], mesh[1], context["camera"])
        branches = {
            name: fit_supported_frame(
                rendered,
                depths[i],
                masks[b, i],
                masks[0, i],
                context["camera"],
                pose[:3, 3],
                context["config"],
            )
            for b, name in enumerate(BRANCHES)
        }
        items.append({"branches": branches})
    return {"frame_index": frame, "input_sha256": hashes, "objects": items}


def _final(
    row: dict[str, Any], *, scales: list[dict[str, float]], context: dict[str, Any]
) -> dict[str, Any]:
    frame = row["frame_index"]
    masks, hashes, depths = _read_frame(frame, context)
    if hashes != row["input_sha256"]:
        raise ValueError("重估两遍间来源改变")
    objects = []
    config = context["config"]
    for i, mesh in enumerate(context["meshes"]):
        initial = row["objects"][i]["branches"]
        if not scales[i] or any(value["rejection_reasons_zh"] for value in initial.values()):
            objects.append({"paired_accepted": False, "branches": {}, "evaluation": None})
            continue
        pose = context["poses"][frame, i]
        local = mesh[0] @ pose[:3, :3].T
        rendered = {
            "unchanged_baseline": rasterize_depth(local + pose[:3, 3], mesh[1], context["camera"])
        }
        branches = {}
        for name in BRANCHES:
            fit = fixed_scale_refit(initial[name]["fit"], scales[i][name])
            fit["rejection_reasons_zh"] = fit_gate(fit, config, pose[:3, 3])
            branches[name] = fit
            rendered[name] = rasterize_depth(
                local * fit["scale"] + fit["translation_camera_m"], mesh[1], context["camera"]
            )
        evaluation = shared_evaluation(rendered, depths[i], masks[1, i])
        for name, fit in branches.items():
            if (
                evaluation["shared_pixels"] < config.min_overlap_pixels
                or evaluation["visible_coverage"] < config.min_mask_coverage
            ):
                fit["rejection_reasons_zh"].append("共同可见重投影支持不足")
            rmse = evaluation["rmse_m"][name]
            if rmse is None or rmse > config.max_fit_rmse_m:
                fit["rejection_reasons_zh"].append("共同可见重投影残差过大")
            fit["accepted"] = not fit["rejection_reasons_zh"]
        objects.append(
            {
                "paired_accepted": all(f["accepted"] for f in branches.values()),
                "branches": branches,
                "evaluation": evaluation,
            }
        )
    return {"frame_index": frame, "objects": objects}


def prepare_inputs(
    scene_manifest: Path, support_manifest: Path, config: SupportRefitConfig
) -> tuple[dict[str, Any], dict[str, str], SceneInput]:
    scene, support = json.loads(scene_manifest.read_text()), json.loads(support_manifest.read_text())
    spec_path = scene_manifest.parent / "input.json"
    spec = SceneInput.model_validate_json(spec_path.read_text())
    if (
        scene["status"] != "assembled_for_inspection"
        or not scene["proof"]["shared_coordinates_proven"]
        or support["status"] != "succeeded"
    ):
        raise ValueError("场景或可见支持未通过原来源证明")
    names = [o.name for o in spec.objects]
    if (
        names != support["object_names"]
        or names != scene["objects"]
        or support["source"] != spec.source.model_dump(mode="json")
    ):
        raise ValueError("支持、场景的对象或时轴身份不一致")
    hashes = {
        str(p): digest(p)
        for p in (scene_manifest, spec_path, support_manifest, spec.source.source_video)
    }
    if hashes[str(spec.source.source_video)] != spec.source.source_video_sha256:
        raise ValueError("源视频SHA改变")
    for p in (scene_manifest, spec_path):
        if support["source_inputs_sha256"].get(str(p)) != hashes[str(p)]:
            raise ValueError("支持清单未绑定当前场景")
    for raw, expected in scene["input_sha256"].items():
        hashes[raw] = digest(Path(raw))
        if hashes[raw] != expected:
            raise ValueError("场景原始输入改变")
    rows = support["frames"]
    if [r["frame_index"] for r in rows] != list(range(spec.source.frame_count)) or any(
        abs(r["timestamp_s"] - r["frame_index"] / spec.source.fps) > 1e-9 for r in rows
    ):
        raise ValueError("支持帧编号或时间轴不完整")
    for row in rows:
        path = Path(row["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("支持路径越界")
    # Read the verified original quality bundles; never silently trust an edited scene.npz.
    bundles, meshes, camera = [], [], None
    for item in spec.objects:
        root = item.job / "reconstruction"
        quality_path = item.job / "hand-quality/quality_result.npz"
        mesh_path = root / "result/mesh.obj"
        for required_path in (quality_path, mesh_path):
            if str(required_path) not in scene["input_sha256"]:
                raise ValueError("场景凭据缺少实际读取的质量束或网格")
        with np.load(quality_path, allow_pickle=False) as data:
            bundles.append(
                {
                    k: data[k]
                    for k in (
                        "object_to_camera_transform",
                        "object_is_valid",
                        "camera_to_world_transform",
                        "camera_is_valid",
                        "camera_intrinsics",
                    )
                }
            )
        k_path = root / "intrinsics_stable.json"
        hashes[str(k_path)] = digest(k_path)
        k = CameraK.model_validate_json(k_path.read_text())
        if camera is not None and camera != k:
            raise ValueError("对象稳定内参不同")
        camera = k
        if not np.allclose(
            bundles[-1]["camera_intrinsics"], [k.fx, k.fy, k.cx, k.cy], rtol=0, atol=1e-5
        ):
            raise ValueError("质量束与稳定内参不符")
        meshes.append(_mesh(mesh_path))
    assert camera is not None
    reference = bundles[0]
    for b in bundles:
        for key in ("object_to_camera_transform", "camera_to_world_transform"):
            if b[key].shape != (spec.source.frame_count, 4, 4):
                raise ValueError("位姿必须精确覆盖完整时间轴")
        for key in ("object_is_valid", "camera_is_valid"):
            if b[key].shape != (spec.source.frame_count,) or b[key].dtype != bool:
                raise ValueError("有效性必须是完整时间轴的布尔数组")
        if (
            not _rigid(b["object_to_camera_transform"])
            or not _rigid(b["camera_to_world_transform"])
            or len(b["object_is_valid"]) != spec.source.frame_count
        ):
            raise ValueError("原坐标契约不成立")
        if not np.array_equal(
            b["camera_to_world_transform"], reference["camera_to_world_transform"]
        ) or not np.array_equal(b["camera_is_valid"], reference["camera_is_valid"]):
            raise ValueError("原始相机不在同一世界")
    context = {
        "rows": rows,
        "support_root": support_manifest.parent,
        "objects": [o.model_dump(mode="json") for o in spec.objects],
        "config": config,
        "camera": _camera_small(camera, config.stride),
        "meshes": meshes,
        "poses": np.stack([b["object_to_camera_transform"] for b in bundles], axis=1),
        "object_valid": np.stack([b["object_is_valid"] for b in bundles], axis=1),
        "camera_valid": reference["camera_is_valid"],
        "camera_to_world": reference["camera_to_world_transform"],
    }
    return context, hashes, spec


def run_support_refit(
    scene_manifest: Path, support_manifest: Path, output: Path, config: SupportRefitConfig
) -> dict[str, Any]:
    context, hashes, spec = prepare_inputs(scene_manifest, support_manifest, config)
    code_root = Path(__file__).parent
    code_hashes = {
        str(code_root / name): digest(code_root / name)
        for name in (
            "postprocess_support_refit.py",
            "postprocess_support_fit.py",
            "postprocess_depth.py",
            "postprocess_scene.py",
            "postprocess_occlusion.py",
            "depth_source_ab.py",
            "sharpa_episode.py",
            "sharpa_bridge.py",
        )
    }
    require_new_output(
        output,
        [
            scene_manifest.parent,
            support_manifest.parent,
            spec.source.source_video.parent,
            *(o.job for o in spec.objects),
        ],
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    input_path, manifest_path = output / "input.json", output / "manifest.json"
    write_json(
        input_path,
        {
            "scene_manifest": str(scene_manifest),
            "support_manifest": str(support_manifest),
            "source": spec.source.model_dump(mode="json"),
            "config": config.model_dump(),
            "input_sha256": hashes,
            "implementation_sha256": code_hashes,
        },
    )
    result: dict[str, Any] = {
        "schema": ADAPTER_METADATA["output_schema"],
        "adapter": ADAPTER_METADATA,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "finished_at": None,
        "input_manifest_path": str(input_path),
        "output_manifest_path": str(manifest_path),
        "log_path": str(output / "run.log"),
        "failure_reason": None,
        "config": config.model_dump(),
        "implementation_sha256": code_hashes,
        "source": spec.source.model_dump(mode="json"),
        "geometry_quality_accepted": False,
        "full_training_accepted": False,
        "world_reestimated": False,
        "contact_ground_truth_available": False,
    }
    write_json(output / "run.json", result)
    try:
        with (
            (output / "run.log").open("x") as log,
            ProcessPoolExecutor(max_workers=config.workers) as pool,
        ):
            rows = []
            for row in pool.map(
                partial(_initial, context=context), range(spec.source.frame_count), chunksize=4
            ):
                rows.append(row)
                hashes.update(row["input_sha256"])
                write_json(output / "checkpoints" / f"{row['frame_index']:06d}.json", row)
                if len(rows) % 25 == 0:
                    message = f"初始重估 {len(rows)}/{spec.source.frame_count} 帧"
                    log.write(message + "\n")
                    log.flush()
                    print(message, flush=True)
            scales = []
            for i in range(len(spec.objects)):
                paired = [
                    r["objects"][i]["branches"]
                    for r in rows
                    if all(
                        not v["rejection_reasons_zh"] for v in r["objects"][i]["branches"].values()
                    )
                ]
                scales.append(
                    {
                        name: float(np.median([b[name]["fit"]["scale"] for b in paired]))
                        for name in BRANCHES
                    }
                    if paired
                    else {}
                )
            final = []
            for row in pool.map(partial(_final, scales=scales, context=context), rows, chunksize=4):
                final.append(row)
                if len(final) % 25 == 0:
                    message = f"固定尺度与共同像素验收 {len(final)}/{spec.source.frame_count} 帧"
                    log.write(message + "\n")
                    log.flush()
                    print(message, flush=True)
        n, m = spec.source.frame_count, len(spec.objects)
        arrays = {
            "source_frame_indices": np.arange(n),
            "timestamps_s": np.arange(n) / spec.source.fps,
            "camera_to_world_transform": context["camera_to_world"],
        }
        for name in BRANCHES:
            poses = np.full((n, m, 4, 4), np.nan)
            valid = np.zeros((n, m), bool)
            for row in final:
                frame = row["frame_index"]
                for i, item in enumerate(row["objects"]):
                    if item["paired_accepted"]:
                        poses[frame, i] = context["poses"][frame, i]
                        poses[frame, i, :3, 3] = item["branches"][name]["translation_camera_m"]
                        valid[frame, i] = True
            arrays[name + "_object_to_camera_transform"] = poses
            arrays[name + "_object_to_world_transform"] = context["camera_to_world"][:, None] @ poses
            arrays[name + "_candidate_valid"] = valid
            arrays[name + "_mesh_scale"] = np.asarray([s.get(name, np.nan) for s in scales])
        np.savez_compressed(output / "refit.npz", **arrays)
        if not all(digest(Path(path)) == sha for path, sha in hashes.items()):
            raise ValueError("源输入在执行期间改变")
        if not all(digest(Path(path)) == sha for path, sha in code_hashes.items()):
            raise ValueError("运行期间实现源码改变")
        write_json(output / "input-hashes.json", hashes)
        result.update(
            status="completed_candidates",
            finished_at=datetime.now(UTC).isoformat(),
            fixed_mesh_scales=dict(zip([o.name for o in spec.objects], scales, strict=True)),
            initial_eligible_frames={
                o.name: {
                    name: sum(
                        not r["objects"][i]["branches"][name]["rejection_reasons_zh"] for r in rows
                    )
                    for name in BRANCHES
                }
                for i, o in enumerate(spec.objects)
            },
            paired_candidate_frames={
                o.name: sum(r["objects"][i]["paired_accepted"] for r in final)
                for i, o in enumerate(spec.objects)
            },
            final_frames=final,
            source_inputs_unchanged=True,
            outputs_sha256={
                p.name: digest(p)
                for p in (output / "refit.npz", output / "input-hashes.json", input_path)
            },
            notes_zh=[
                "固定原旋转、同一预测深度与共享相机。尺度使用两路线同时通过首轮的同一帧集；只输出配对通过的帧，其余NaN。",
                "每帧旧姿态及两路线在完全相同可见像素上评价，仍是拟合内残差，不是独立真值。",
                "缩放仅作用局部物体网格；沿用原世界复合，不重新估计相机/手或把候选称为物理米制真值。",
                "未改语义身份、原掩膜、手或旧轨迹；没有接触标签，不允许直接用于策略训练。",
            ],
        )
    except Exception as exc:
        result.update(
            status="failed", failure_reason=str(exc), finished_at=datetime.now(UTC).isoformat()
        )
        with (output / "run.log").open("a") as log:
            log.write(traceback.format_exc())
        _finish(output, result)
        raise
    _finish(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = run_support_refit(
        args.scene, args.support, args.output, SupportRefitConfig(workers=args.workers)
    )
    print(
        json.dumps(
            {k: result[k] for k in ("status", "fixed_mesh_scales", "paired_candidate_frames")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
