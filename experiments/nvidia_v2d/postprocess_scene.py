"""通过同源相机与逐帧哈希证明组装多物体；不估计或猜测世界对齐。"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from experiments.nvidia_v2d.sharpa_bridge import digest, write_json
from experiments.nvidia_v2d.sharpa_episode import EpisodeSource, _rigid, require_new_output


class ObjectInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    name: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    job: Path
    source_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fps: float = Field(gt=0)
    frame_count: int = Field(gt=1)


class SceneInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["nvidia_v2d.shared_scene_input.v1"] = "nvidia_v2d.shared_scene_input.v1"
    source: EpisodeSource
    objects: list[ObjectInput] = Field(min_length=2)
    hand_reference_object: str


def prove_shared_coordinates(
    arrays: list[dict[str, np.ndarray]], fingerprints: list[dict[str, str]]
) -> dict[str, Any]:
    """严格相等的相机/内参和来源凭据，外加每个物体的坐标复合检查。"""
    if len(arrays) < 2 or len(arrays) != len(fingerprints):
        raise ValueError("至少需要两个来源完整的物体")
    blockers, composition = [], []
    first = arrays[0]
    for index, data in enumerate(arrays):
        for key in ("camera_to_world_transform", "camera_is_valid", "camera_intrinsics"):
            if key not in data or key not in first or not np.array_equal(data[key], first[key]):
                blockers.append(f"object_{index}:different_{key}")
        for key in ("source_video", "rgb_frames", "slam_trajectory", "slam_poses", "intrinsics"):
            if not fingerprints[index].get(key) or fingerprints[index].get(key) != fingerprints[
                0
            ].get(key):
                blockers.append(f"object_{index}:different_{key}_fingerprint")
        for key in (
            "camera_to_world_transform",
            "object_to_camera_transform",
            "object_to_world_transform",
        ):
            if key not in data or not _rigid(data[key]):
                blockers.append(f"object_{index}:invalid_{key}")
        if any(
            key not in data
            for key in (
                "camera_to_world_transform",
                "object_to_camera_transform",
                "object_to_world_transform",
            )
        ):
            composition.append(None)
            continue
        if (
            data["object_to_camera_transform"].shape != data["camera_to_world_transform"].shape
            or data["object_to_world_transform"].shape != data["camera_to_world_transform"].shape
        ):
            blockers.append(f"object_{index}:different_frame_count")
            composition.append(None)
            continue
        error = float(
            np.max(
                np.abs(
                    data["camera_to_world_transform"] @ data["object_to_camera_transform"]
                    - data["object_to_world_transform"]
                )
            )
        )
        composition.append(error)
        if error > 1e-5:
            blockers.append(f"object_{index}:world_composition_mismatch")
    return {
        "shared_coordinates_proven": not blockers,
        "blockers": blockers,
        "composition_max_abs_errors": composition,
        "alignment": "identity; original common SLAM world is preserved",
        "icp_used": False,
    }


def _tree_digest(directory: Path, expected_frames: int | None = None) -> str:
    import hashlib

    files = sorted(path for path in directory.iterdir() if path.is_file())
    if not files:
        raise ValueError(f"缺少来源文件：{directory}")
    if expected_frames is not None and [path.stem for path in files] != [
        f"{index:06d}" for index in range(expected_frames)
    ]:
        raise ValueError(f"逐帧来源必须完整覆盖原始帧编号：{directory}")
    value = hashlib.sha256()
    for path in files:
        value.update(path.name.encode())
        value.update(digest(path).encode())
    return value.hexdigest()


def assemble_scene(spec: SceneInput, output: Path) -> dict[str, Any]:
    require_new_output(
        output, [spec.source.source_video.parent, *(item.job for item in spec.objects)]
    )
    names = [item.name for item in spec.objects]
    if len(set(names)) != len(names) or set(names) != set(spec.source.object_names):
        raise ValueError("物体名称需唯一且与来源清单完全一致")
    if spec.hand_reference_object not in names:
        raise ValueError("必须明确选择一个已记录的手部参考来源")
    if digest(spec.source.source_video) != spec.source.source_video_sha256:
        raise ValueError("源视频哈希不匹配")
    arrays, fingerprints, hashes, source_manifests = [], [], {}, []
    for item in spec.objects:
        if (item.source_video_sha256, item.fps, item.frame_count) != (
            spec.source.source_video_sha256,
            spec.source.fps,
            spec.source.frame_count,
        ):
            raise ValueError("视频或时间轴来源不同，拒绝组装")
        root = item.job / "reconstruction"
        path = item.job / "hand-quality/quality_result.npz"
        quality_manifest = json.loads((path.parent / "manifest.json").read_text())
        expected = [
            row["sha256"] for row in quality_manifest["artifacts"] if row["path"] == path.name
        ]
        if expected != [digest(path)]:
            raise ValueError("质量束哈希不匹配")
        with np.load(path, allow_pickle=False) as loaded:
            data = {key: loaded[key] for key in loaded.files}
        if len(data["camera_to_world_transform"]) != spec.source.frame_count:
            raise ValueError("质量束帧数与源视频不同")
        manifest_path = root / "result/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest["sources"].get("object_transform_convention") != "rigid_pose_no_scale"
            or manifest["camera_pose_convention"] != "camera_to_world"
        ):
            raise ValueError("上游坐标约定不明确")
        if data["object_scale"].item() != 1:
            raise ValueError("需要先显式烘焙物体尺度，不能在组装时隐式换尺度")
        fingerprint = {
            "source_video": item.source_video_sha256,
            "rgb_frames": _tree_digest(root / "frames", item.frame_count),
            "slam_trajectory": digest(root / "slam_trajectory.txt"),
            "slam_poses": _tree_digest(root / "slam_poses", item.frame_count),
            "intrinsics": digest(root / "intrinsics_stable.json"),
        }
        arrays.append(data)
        fingerprints.append(fingerprint)
        source_manifests.append(manifest)
        for input_path in (
            path,
            path.parent / "manifest.json",
            manifest_path,
            root / "result/mesh.obj",
            root / "slam_trajectory.txt",
        ):
            hashes[str(input_path)] = digest(input_path)
    proof = prove_shared_coordinates(arrays, fingerprints)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "input.json", spec.model_dump(mode="json"))
    result = {
        "schema": "nvidia_v2d.shared_scene.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "source": spec.source.model_dump(mode="json"),
        "hand_reference_object": spec.hand_reference_object,
        "objects": names,
        "proof": proof,
        "source_fingerprints": dict(zip(names, fingerprints, strict=True)),
        "input_sha256": hashes,
        "geometry_quality_accepted": False,
        "full_training_accepted": False,
        "说明": [
            "直接保留相同的原始SLAM世界；没有ICP、独立PCA居中或猜测对齐。",
            "对象身份与接触精度仍需审查；共同坐标证明不等于重建正确。",
            "手部只来自显式指定的一套质量束；其它来源的差异单独列出，未平均。",
        ],
    }
    if not proof["shared_coordinates_proven"]:
        result.update(status="blocked", scene_npz=None)
        write_json(output / "manifest.json", result)
        return result
    reference = arrays[names.index(spec.hand_reference_object)]
    output_arrays = {
        key: value
        for key, value in reference.items()
        if key.startswith("hand_") or key.startswith("camera_")
    }
    output_arrays.update(
        object_to_world_transform=np.stack(
            [data["object_to_world_transform"] for data in arrays], axis=1
        ),
        object_to_camera_transform=np.stack(
            [data["object_to_camera_transform"] for data in arrays], axis=1
        ),
        object_is_valid=np.stack([data["object_is_valid"] for data in arrays], axis=1),
        source_frame_indices=np.arange(spec.source.frame_count),
        timestamps_s=np.arange(spec.source.frame_count) / spec.source.fps,
    )
    differences = {}
    for name, data in zip(names, arrays, strict=True):
        differences[name] = {
            key: float(np.max(np.abs(data[key].astype(float) - value.astype(float))))
            for key, value in reference.items()
            if key.startswith("hand_") and key in data and data[key].shape == value.shape
        }
    assets = []
    for item, manifest in zip(spec.objects, source_manifests, strict=True):
        source_dir = item.job / "reconstruction/result"
        directory = output / "assets" / item.name
        directory.mkdir(parents=True)
        for name in [
            "mesh.obj",
            *manifest["mesh"].get("materials", []),
            *manifest["mesh"].get("textures", []),
        ]:
            if Path(name).name != name or (source_dir / name).is_symlink():
                raise ValueError("资产只允许同级普通文件")
            hashes[str(source_dir / name)] = digest(source_dir / name)
            shutil.copyfile(source_dir / name, directory / name)
            if digest(directory / name) != hashes[str(source_dir / name)]:
                raise ValueError("复制后的场景资产哈希不一致")
        assets.append(
            {"name": item.name, "mesh": str(directory / "mesh.obj"), "scale_already_baked": True}
        )
    np.savez_compressed(output / "scene.npz", **output_arrays)
    unchanged = all(digest(Path(path)) == value for path, value in hashes.items())
    if not unchanged:
        raise ValueError("读取期间源输入改变")
    result.update(
        status="assembled_for_inspection",
        scene_npz=str(output / "scene.npz"),
        assets=assets,
        hand_alternative_max_abs_differences=differences,
        source_inputs_unchanged=True,
        hand_invalid_frames={
            side: np.flatnonzero(~reference[f"hand_{side}_is_valid"]).tolist()
            for side in ("left", "right")
        },
        object_pose_provenance="tracker estimates; no per-frame observation/extrapolation evidence available",
    )
    write_json(output / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assemble_scene(SceneInput.model_validate_json(args.input.read_text()), args.output)
    print(json.dumps({"status": result["status"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
