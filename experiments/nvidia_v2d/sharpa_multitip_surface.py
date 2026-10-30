"""Deterministic samples of original MJCF collision STL surfaces, not simulator contacts."""

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


class CollisionSurfaceSampler:
    """Bind fixed area-weighted collision surface samples to official robot body frames."""

    def __init__(
        self, xml_path: Path, frame_names: list[str], count: int = 512, seed: int = 42
    ) -> None:
        import trimesh
        from scipy.spatial.transform import Rotation

        if count < 1:
            raise ValueError("表面采样数量必须为正")
        root = ET.parse(xml_path).getroot()
        compiler = root.find("compiler")
        mesh_dir = xml_path.parent / (compiler.get("meshdir", "") if compiler is not None else "")
        mesh_assets = {mesh.attrib["name"]: mesh for mesh in root.findall("asset/mesh")}
        triangles: list[NDArray[np.float64]] = []
        body_indices: list[NDArray[np.int64]] = []
        asset_hashes: dict[str, str] = {}
        geometry_count = 0
        for body in root.findall(".//worldbody//body"):
            for geom in body.findall("geom"):
                if int(geom.get("contype", "1")) == int(geom.get("conaffinity", "1")) == 0:
                    continue
                if geom.get("type") != "mesh" or "mesh" not in geom.attrib:
                    raise ValueError("当前 Sharpa 表面审计只接受显式 mesh 碰撞几何")
                if set(geom.attrib) & {"euler", "axisangle", "xyaxes", "zaxis", "fromto"}:
                    raise ValueError("碰撞几何使用未支持的变换表达")
                frame_name = body.attrib["name"]
                if frame_name not in frame_names:
                    raise ValueError(f"碰撞 body 不在 FK 帧中: {frame_name}")
                mesh_asset = mesh_assets[geom.attrib["mesh"]]
                if set(mesh_asset.attrib) & {"refpos", "refquat"}:
                    raise ValueError("碰撞 mesh 引用坐标变换尚未支持")
                mesh_path = (mesh_dir / mesh_asset.attrib["file"]).resolve()
                mesh = trimesh.load_mesh(mesh_path, process=False)
                if not isinstance(mesh, trimesh.Trimesh):
                    raise ValueError("碰撞资产必须为单一三角网格")
                vertices = np.asarray(mesh.vertices, dtype=float)
                vertices *= np.fromstring(mesh_asset.get("scale", "1 1 1"), sep=" ")
                rotation = Rotation.from_quat(
                    np.fromstring(geom.get("quat", "1 0 0 0"), sep=" "), scalar_first=True
                ).as_matrix()
                vertices = vertices @ rotation.T + np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
                faces = vertices[np.asarray(mesh.faces)]
                triangles.append(faces)
                body_indices.append(
                    np.full(len(faces), frame_names.index(frame_name), dtype=np.int64)
                )
                asset_hashes[str(mesh_path)] = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
                geometry_count += 1
        if not triangles:
            raise ValueError("Sharpa 模型没有可审计碰撞表面")
        faces = np.concatenate(triangles)
        area = (
            np.linalg.norm(np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]), axis=1) / 2
        )
        if not np.isfinite(area).all() or area.sum() <= 0:
            raise ValueError("碰撞表面面积无效")
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(faces), count, p=area / area.sum())
        uv = rng.random((count, 2))
        uv[uv.sum(axis=1) > 1] = 1 - uv[uv.sum(axis=1) > 1]
        selected_faces = faces[selected]
        self.local_points = (
            selected_faces[:, 0]
            + uv[:, :1] * (selected_faces[:, 1] - selected_faces[:, 0])
            + uv[:, 1:] * (selected_faces[:, 2] - selected_faces[:, 0])
        )
        self.frame_indices = np.concatenate(body_indices)[selected]
        self.metadata: dict[str, Any] = {
            "kind": "original_mjcf_collision_stl_surface",
            "simulator_convex_hulls": False,
            "count": count,
            "seed": seed,
            "sampling": "fixed_triangle_area_weighted_barycentric",
            "triangle_count": len(faces),
            "collision_geometry_count": geometry_count,
            "asset_sha256": asset_hashes,
        }

    def world_points(self, frames: NDArray[np.float64]) -> NDArray[np.float64]:
        from scipy.spatial.transform import Rotation

        selected = np.asarray(frames)[self.frame_indices]
        rotations = Rotation.from_quat(selected[:, 3:], scalar_first=True).as_matrix()
        return np.asarray(
            np.einsum("nij,nj->ni", rotations, self.local_points) + selected[:, :3],
            dtype=np.float64,
        )
