"""将离线重定向结果导出为只读查看器场景，保留米制原网格和真实时间轴。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from experiments.nvidia_v2d.policy_view_data import Scene


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hand(preview: dict[str, Any], side: str, frames: int) -> dict[str, Any]:
    import numpy as np

    validity = preview.get(f"valid_{side}")
    codes = preview.get(f"source_code_{side}")
    for quality in (validity, codes):
        if quality is not None and (not isinstance(quality, list) or len(quality) != frames):
            raise ValueError("逐手质量标记与时间轴长度不符")
    points = []
    for index, frame in enumerate(preview[f"hands_{side}"]):
        if frame is None:
            points.append(None)
            continue
        array = np.asarray(frame, dtype=float)
        if array.ndim != 2 or array.shape[1] != 3 or not len(array):
            raise ValueError("手部几何形状错误")
        if np.isfinite(array).all():
            points.append(frame)
        elif validity is not None and index < frames and validity[index] is True:
            raise ValueError("有效手帧必须包含有限几何")
        else:
            # Preserve the source file and quality flags; never fabricate coordinates.
            points.append(None)
    return {
        "side": side,
        "points": points,
        "edges": preview["edges"],
        "valid_frames": validity,
        "source_codes": codes,
    }


def _combined_validity(hands: list[dict[str, Any]], frames: int) -> list[bool | None]:
    result = []
    for index in range(frames):
        values = [
            hand["valid_frames"][index] if hand["valid_frames"] is not None else None
            for hand in hands
        ]
        result.append(False if False in values else None if None in values else True)
    return result


def _convert(preview_path: Path) -> tuple[Scene, dict[str, str]]:
    import numpy as np
    import trimesh

    hashes = {str(preview_path.resolve()): _sha(preview_path)}
    preview = json.loads(preview_path.read_text())
    if (
        preview.get("schema") != "nvidia_v2d.sharpa_retarget_preview.v1"
        or preview.get("kind") != "kinematic_reference"
    ):
        raise ValueError("此入口只接受运动学重定向参考，不能标为策略执行")
    times = preview["timestamps_s"]
    names, paths = preview["object_names"], preview["object_mesh_paths"]
    poses = np.asarray(preview["object_poses_world"], dtype=float)
    if len(names) != len(paths) or poses.shape != (len(times), len(names), 4, 4):
        raise ValueError("物体变换数量与时间轴不符")
    if not np.allclose(poses[..., 3, :], [0, 0, 0, 1], atol=1e-8):
        raise ValueError("物体齐次变换末行无效")
    rotations = poses[..., :3, :3]
    if not np.allclose(
        rotations @ rotations.swapaxes(-1, -2), np.eye(3), atol=1e-5
    ) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-5):
        raise ValueError("物体变换不是刚体旋转，不能隐式缩放场景")
    objects = []
    for index, (name, path) in enumerate(zip(names, paths, strict=True)):
        mesh_path = Path(path)
        if not mesh_path.is_absolute():
            mesh_path = preview_path.parent / mesh_path
        hashes[str(mesh_path.resolve())] = _sha(mesh_path)
        mesh = trimesh.load_mesh(mesh_path, process=False, skip_materials=True)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("物体资产必须为单个三角网格")
        objects.append(
            {
                "name": name,
                "vertices": mesh.vertices.tolist(),
                "faces": mesh.faces.tolist(),
                "poses": poses[:, index].reshape(len(times), 16).tolist(),
            }
        )
    hands = [_hand(preview, side, len(times)) for side in ("left", "right")]
    scene = Scene.model_validate(
        {
            "kind": "kinematic_reference",
            "assistance": "unknown",
            "timestamps_s": times,
            "hands": hands,
            "valid_frames": _combined_validity(hands, len(times)),
            "objects": objects,
        }
    )
    for path, before in hashes.items():
        if _sha(Path(path)) != before:
            raise ValueError("读取期间输入发生变化")
    return scene, hashes


def export_scenes(previews: dict[str, Path], output: Path) -> dict[str, dict[str, Any]]:
    """一次导出一组对照到全新目录；完成校验后才创建任何输出。"""
    if output.exists():
        raise FileExistsError("查看器场景目录必须为新目录")
    if not previews or any(not re.fullmatch(r"[a-z0-9-]+", key) for key in previews):
        raise ValueError("后端名称必须为非空安全标识")
    converted = {name: _convert(path) for name, path in previews.items()}
    payloads = {name: scene.model_dump_json().encode() for name, (scene, _) in converted.items()}
    if any(len(payload) > 24 * 1024**2 for payload in payloads.values()):
        raise ValueError("原始网格场景超过查看器24MiB上限，需要显式批准的简化方案")
    output.mkdir(parents=True)
    manifests = {}
    for name, (scene, hashes) in converted.items():
        path = output / f"{name}.json"
        path.write_bytes(payloads[name])
        manifest = {
            "schema": "nvidia_v2d.policy_scene_export.v1",
            "status": "completed",
            "created_at": datetime.now(UTC).isoformat(),
            "kind": "kinematic_reference",
            "assistance": scene.assistance,
            "quality_flags_preserved": True,
            "hand_representation": "fk_skeleton_not_collision_mesh",
            "physical_rollout": False,
            "input_sha256": hashes,
            "input_unchanged": True,
            "output_path": str(path.resolve()),
            "output_sha256": _sha(path),
            "output_manifest_path": str((output / f"{name}.manifest.json").resolve()),
            "mesh_simplification": None,
            "scene_scale_changed": False,
            "frame_count": len(scene.timestamps_s),
            "objects": [
                {"name": obj.name, "vertices": len(obj.vertices), "faces": len(obj.faces)}
                for obj in scene.objects
            ],
        }
        (output / f"{name}.manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        manifests[name] = manifest
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="导出官方与自有重定向的米制运动学场景")
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--multitip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            export_scenes({"official": args.official, "multitip": args.multitip}, args.output),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
