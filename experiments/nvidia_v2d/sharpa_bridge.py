"""把已审查的重建束导出到独立 ManoSharpaData；源目录从不写入。"""

import argparse
import hashlib
import importlib.util
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from experiments.nvidia_v2d.sharpa_bridge_geometry import (
    load_upstream_module,
    mano_world_geometry,
    scene_alignment,
)
from experiments.nvidia_v2d.sharpa_episode import EpisodeSource, audit_bundle, require_new_output


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def read_source(plan: Path, sequence_id: str) -> EpisodeSource:
    rows = json.loads(plan.read_text())["episodes"]
    matches = [row for row in rows if row["run_key"] == sequence_id]
    if len(matches) != 1:
        raise ValueError("批准清单须唯一对应片段")
    row = matches[0]
    return EpisodeSource(
        sequence_id=sequence_id,
        object_names=row["objects"],
        fps=row["fps"],
        frame_count=row["frame_count"],
        source_video=row["source_video"],
        source_video_sha256=row["source_video_sha256"],
    )


def accepted_shape_provenance(job: Path, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """验证常量 beta 确实来自被接受观测；不能凭常量字段冒称可信中值。"""
    records: dict[str, dict[int, tuple[np.ndarray, Path]]] = {"left": {}, "right": {}}
    for path in sorted((job / "reconstruction/hamer_aligned").glob("*/*.json")):
        data = json.loads(path.read_text())
        side = "right" if data["is_right"] else "left"
        index = int(data["frame_idx"])
        codes = arrays[f"hand_{side}_source_code"]
        if not 0 <= index < len(codes) or codes[index] != 1:
            continue
        if index in records[side]:
            raise ValueError(f"{side}: 重复的形态观测帧 {index}")
        betas = np.asarray(data["mano"]["betas"], dtype=float)
        if betas.shape != (10,) or not np.isfinite(betas).all():
            raise ValueError(f"{side}: 无效形态观测")
        records[side][index] = (betas, path)
    result = {}
    for side in ("left", "right"):
        expected = np.flatnonzero(arrays[f"hand_{side}_source_code"] == 1).tolist()
        if sorted(records[side]) != expected or not expected:
            raise ValueError(f"{side}: 缺少可信观测的原始形态来源")
        median = np.median([records[side][i][0] for i in expected], axis=0)
        if not np.allclose(median, arrays[f"hand_{side}_betas"], atol=1e-7, rtol=0):
            raise ValueError(f"{side}: 质量束常量形态不是可信观测中值，需要先重新标定")
        result[side] = {
            "method": "median_original_aligned_accepted_observations",
            "accepted_frame_indices": expected,
            "shape_betas": median.tolist(),
            "inputs": [
                {"path": str(records[side][i][1]), "sha256": digest(records[side][i][1])}
                for i in expected
            ],
        }
    return result


def _contacts(
    geometry: dict[str, Any],
    mesh: Any,
    poses: np.ndarray,
    upstream: Path,
) -> dict[str, list]:
    """复用官方 1 cm 接触启发式，固定表面采样；不是实测接触。"""
    import torch
    import trimesh

    contact = load_upstream_module(upstream, "contact_utils")
    from robotic_grounding.retarget.params import NUM_MANO_LINKS

    state = np.random.get_state()
    np.random.seed(42)
    try:
        points, face_ids = trimesh.sample.sample_surface_even(mesh, 4096)
    finally:
        np.random.set_state(state)
    normals = -mesh.face_normals[face_ids]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    fields = {
        name: []
        for name in (
            "tips_distance",
            "link_contact_positions",
            "link_contact_normals",
            "object_contact_positions",
            "object_contact_normals",
            "object_contact_part_ids",
        )
    }
    for index, pose in enumerate(poses):
        world_points = torch.tensor(points @ pose[:3, :3].T + pose[:3, 3], dtype=torch.float32)
        world_normals = torch.tensor(normals @ pose[:3, :3].T, dtype=torch.float32)
        verts = geometry["vertices"][index]
        hand_normals = trimesh.Trimesh(verts, geometry["faces"], process=False).vertex_normals
        joints = torch.tensor(geometry["joints"][index], dtype=torch.float32)
        result = contact.approximate_contact_with_id(
            world_points,
            world_normals,
            torch.ones(len(points), dtype=torch.long),
            torch.tensor(verts, dtype=torch.float32),
            torch.tensor(hand_normals.copy(), dtype=torch.float32),
            threshold=0.01,
        )
        part_ids, hand_points, hand_n, distances = result[2:]
        hp = torch.zeros((NUM_MANO_LINKS, 3))
        hn, op, on = hp.clone(), hp.clone(), hp.clone()
        ids = torch.zeros(NUM_MANO_LINKS)
        if len(distances):
            hp, hn, ids = contact.compute_hand_link_contact_positions(
                joints,
                part_ids,
                hand_points,
                hand_n,
                distances,
            )
            op, on = contact.find_object_contact_positions(hp, world_points, world_normals)
        values = (hp, hn, op, on, ids)
        if not all(torch.isfinite(value).all() for value in values):
            raise ValueError(f"接触估计出现非有限值，帧 {index}")
        fields["tips_distance"].append(
            torch.cdist(joints[[4, 8, 12, 16, 20]], world_points).amin(-1).tolist()
        )
        for name, value in zip(tuple(fields)[1:], values, strict=True):
            fields[name].append(value.int().tolist() if name.endswith("ids") else value.tolist())
    return fields


def _assets(
    source: Path, output: Path, upstream: Path, object_name: str, scale: float
) -> tuple[Any, Path, Path]:
    import trimesh

    directory = output / "assets" / object_name
    directory.mkdir(parents=True)
    manifest = json.loads((source / "manifest.json").read_text())
    names = (
        ["mesh.obj"] + manifest["mesh"].get("materials", []) + manifest["mesh"].get("textures", [])
    )
    for name in names:
        if Path(name).name != name or (source / name).is_symlink():
            raise ValueError("网格资产只允许源束中的普通同级文件")
        shutil.copyfile(source / name, directory / name)
    mesh = trimesh.load(directory / "mesh.obj", force="mesh", process=False)
    if not np.isfinite(mesh.vertices).all() or not len(mesh.faces):
        raise ValueError("无效物体网格")
    if scale != 1:
        # 原始复制保持可追溯；米制版本单独写入，不重复缩放物体姿态。
        mesh.vertices *= scale
        mesh.export(directory / "metric_mesh.obj")
    mesh_path = directory / ("metric_mesh.obj" if scale != 1 else "mesh.obj")
    mesh.export(directory / "collision.stl")
    script = upstream / "robotic_grounding/scripts/generate_rigid_urdfs.py"
    spec = importlib.util.spec_from_file_location("_sharpa_official_urdf", script)
    if spec is None or spec.loader is None:
        raise ImportError(str(script))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    urdf = directory / "object.urdf"
    urdf.write_text(module._generate_urdf(object_name, "./", "collision.stl"))
    return mesh, mesh_path, urdf


def _export_episode(
    source: EpisodeSource,
    job: Path,
    upstream: Path,
    mano_root: Path,
    output: Path,
    *,
    audit_only: bool = False,
    delivery: bool = False,
) -> dict[str, Any]:
    """写新实例及 sidecar；非法完整片段只输出 gate，不输出可训练 Parquet。"""
    quality = job / "hand-quality"
    bundle = job / "reconstruction/result"
    npz_path = quality / "quality_result.npz"
    require_new_output(output, [job, source.source_video.parent, upstream, mano_root])
    if npz_path.is_symlink() or not npz_path.is_file():
        raise ValueError("质量束必须为普通文件")
    if digest(source.source_video) != source.source_video_sha256:
        raise ValueError("源视频不匹配批准清单")
    manifest = json.loads((quality / "manifest.json").read_text())
    rows = [x for x in manifest["artifacts"] if x["path"] == "quality_result.npz"]
    if len(rows) != 1 or digest(npz_path) != rows[0]["sha256"]:
        raise ValueError("质量束哈希与已发布凭据不一致")
    with np.load(npz_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    audit = audit_bundle(source, arrays, allow_inferred_hands=delivery)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "input-gate.json", audit.model_dump(mode="json"))
    if audit_only or not audit.input_contract_passed:
        result = {
            "schema": "nvidia_v2d.sharpa_loaded_episode.v1",
            "status": "audited" if audit.input_contract_passed else "blocked",
            "finished_at": datetime.now(UTC).isoformat(),
            "input_contract_passed": audit.input_contract_passed,
            "blockers": audit.blockers,
            "source": source.model_dump(mode="json"),
            "output": str(output),
            "parquet": None,
        }
        write_json(output / "manifest.json", result)
        return result

    started = datetime.now(UTC).isoformat()
    shape_provenance = (json.loads((quality / "provenance.json").read_text())["shape_provenance"]
                        if delivery else accepted_shape_provenance(job, arrays))
    inputs = [
        npz_path,
        bundle / "mesh.obj",
        bundle / "manifest.json",
        source.source_video,
        quality / "manifest.json",
        quality / "provenance.json",
        mano_root / "models/MANO_LEFT.pkl",
        mano_root / "models/MANO_RIGHT.pkl",
    ]
    before = {str(p): digest(p) for p in inputs}
    for side in ("left", "right"):
        before.update({item["path"]: item["sha256"] for item in shape_provenance[side]["inputs"]})
    load_upstream_module(upstream, "read_mano")
    import pyarrow as pa
    import pyarrow.parquet as pq
    from robotic_grounding.retarget.data_logger import ManoSharpaData
    from robotic_grounding.retarget.params import MANO_HAND_LINKS, MANO_JOINTS_ORDER
    from scipy.spatial.transform import Rotation

    mesh, mesh_path, urdf = _assets(
        bundle, output, upstream, source.object_names[0], float(arrays["object_scale"])
    )
    alignment = scene_alignment(np.asarray(mesh.vertices), arrays["object_to_world_transform"][0])
    objects = alignment[None] @ arrays["object_to_world_transform"]
    geometry, fk_diagnostics = mano_world_geometry(arrays, upstream, mano_root, alignment,
                                                 allow_inferred_hands=delivery)
    row: dict[str, Any] = ManoSharpaData(
        **{name: None for name, _, _, is_ts in ManoSharpaData._field_specs if not is_ts}
    ).to_dict()
    for side in ("left", "right"):
        for suffix in ("finger_joint_names", "frame_names", "frame_task_names"):
            row[f"{side}_robot_{suffix}"] = []
    n = source.frame_count
    row.update(
        sequence_id=source.sequence_id,
        raw_motion_file=str(npz_path),
        robot_name="sharpa_wave",
        fps=source.fps,
        mano_flat_hand_mean=True,
        mano_center_idx=None,
        mano_to_robot_scale=None,
        object_name=source.object_names[0],
        safe_object_name=source.object_names[0],
        object_body_names=source.object_names,
        safe_object_body_names=source.object_names,
        object_mesh_paths=[str(mesh_path)],
        object_urdf_paths=[str(urdf)],
        object_mesh_radius=[
            float(np.linalg.norm(mesh.vertices - mesh.vertices.mean(axis=0), axis=1).max())
        ],
        object_articulation=[0.0] * n,
        object_root_position=objects[:, :3, 3].tolist(),
        object_root_axis_angle=Rotation.from_matrix(objects[:, :3, :3]).as_rotvec().tolist(),
        object_body_position=objects[:, None, :3, 3].tolist(),
        object_body_wxyz=Rotation.from_matrix(objects[:, :3, :3])
        .as_quat()[:, None, [3, 0, 1, 2]]
        .tolist(),
        mano_link_names=list(MANO_HAND_LINKS),
    )
    calibration: dict[str, Any] = {
        "schema": "nvidia_v2d.sharpa_calibration_input.v1",
        "fps": source.fps,
        "source_frame_indices": audit.source_frame_indices,
        "timestamps_s": audit.timestamps_s,
        "timestamp_source": audit.timestamp_source,
        "hands": {},
        "source_to_scene_transform": alignment.tolist(),
        "alignment_source": "official_single_object_first_frame_pca_heuristic",
        "gravity_or_support_truth": False,
        "geometry_quality_accepted": False,
        "authoritative_mano_fields": ["joints", "joints_wxyz"],
        "parameter_fk_requires_metric_scale_sidecar": True,
        "mano_fitting_error_source": "not_available_in_reconstruction_bundle",
    }
    for side in ("left", "right"):
        hand = geometry[side]
        prefix = "mano_" + side + "_"
        for key in ("betas", "trans", "global_orient", "finger_pose", "joints", "joints_wxyz"):
            row[prefix + key] = hand[key].tolist()
        # The reconstruction bundle supplies no measured fitting residual.
        # Arrow nullable floats preserve unknown instead of claiming a perfect fit.
        row[prefix + "fitting_err"] = [None] * n
        for key, value in _contacts(hand, mesh, objects, upstream).items():
            row[prefix + key] = value
        calibration["hands"][side] = {
            "open_joints_m": hand["open_joints_m"].tolist(),
            "open_wrist_wxyz": hand["open_wrist_wxyz"].tolist(),
            "open_coordinate_frame": "native_mano_zero_global_orientation",
            "joint_names": MANO_JOINTS_ORDER,
            "source": "submission_inferred_mano_open_pose" if delivery else "median_accepted_mano_shape_open_pose",
            "allow_inferred_hands": delivery,
            "shape_betas": hand["betas"].tolist(),
            "shape_provenance": shape_provenance[side],
            "left_shape_mirror_corrected_in_memory": fk_diagnostics[side][
                "left_shape_basis_x_mirror_corrected_in_memory"
            ],
            "middle_tip_vertex": 445 if side == "left" else 444,
            "metric_hand_scale": hand["metric_hand_scale"],
            "metric_scale_per_frame": arrays[f"hand_{side}_scale"].tolist(),
            "scale_pivot": "posed_mesh_centroid_before_camera_translation",
            "valid": audit.hands[side].valid,
            "source_code": audit.hands[side].source_code,
        }
    parquet = output / "loaded.parquet"
    pq.write_table(
        pa.Table.from_pylist([row], schema=ManoSharpaData._schema), parquet, compression="zstd"
    )
    write_json(output / "calibration.json", calibration)
    np.savez_compressed(
        output / "geometry.npz",
        source_to_scene_transform=alignment,
        object_to_world_transform=objects,
        **{
            f"{side}_{key}": geometry[side][key]
            for side in ("left", "right")
            for key in ("joints", "joints_wxyz", "vertices", "faces")
        },
    )
    unchanged = all(digest(Path(path)) == value for path, value in before.items())
    if not unchanged:
        raise ValueError("生成期间源输入发生变化，拒绝发布")
    receipt = {
        "schema": "nvidia_v2d.sharpa_loaded_episode.v1",
        "status": "succeeded",
        "started_at": started,
        "finished_at": datetime.now(UTC).isoformat(),
        "source": source.model_dump(mode="json"),
        "input_sha256": before,
        "input_files_unchanged": unchanged,
        "input_contract_passed": True,
        "geometry_quality_accepted": False,
        "parquet": str(parquet),
        "calibration": str(output / "calibration.json"),
        "fk_diagnostics": fk_diagnostics,
        "physical_assumptions": {
            "support": "first-frame object PCA heuristic, requires visual review",
            "contact": "official sampled geometry within 0.01 m, not measured contact",
            "object_mass_kg": 0.3,
            "object_inertia": "official rigid URDF default",
            "collision": "original reconstructed mesh converted to STL; simulator treatment must be reviewed",
        },
        "artifacts": [
            {"path": str(p.relative_to(output)), "sha256": digest(p), "size": p.stat().st_size}
            for p in sorted(output.rglob("*"))
            if p.is_file()
        ],
    }
    write_json(output / "manifest.json", receipt)
    return receipt


def export_episode(
    source: EpisodeSource,
    job: Path,
    upstream: Path,
    mano_root: Path,
    output: Path,
    *,
    audit_only: bool = False,
    delivery: bool = False,
) -> dict[str, Any]:
    require_new_output(output, [job, source.source_video.parent, upstream, mano_root])
    started = datetime.now(UTC).isoformat()
    try:
        return _export_episode(source, job, upstream, mano_root, output, audit_only=audit_only, delivery=delivery)
    except Exception as exc:
        if output.is_dir() and not (output / "manifest.json").exists():
            write_json(
                output / "manifest.json",
                {
                    "schema": "nvidia_v2d.sharpa_loaded_episode.v1",
                    "status": "failed",
                    "started_at": started,
                    "finished_at": datetime.now(UTC).isoformat(),
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "source": source.model_dump(mode="json"),
                    "input_contract_passed": False,
                    "parquet": None,
                },
            )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--mano-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    result = export_episode(
        read_source(args.plan, args.sequence_id),
        args.job,
        args.upstream,
        args.mano_root,
        args.output,
        audit_only=args.audit_only,
    )
    print(
        json.dumps(
            {k: result.get(k) for k in ("status", "input_contract_passed", "blockers", "parquet")},
            ensure_ascii=False,
        )
    )
    if not result["input_contract_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
