"""隔离运行时中的 MANO FK：复用官方轴系，保留 HaMeR 质心缩放。"""

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

from experiments.nvidia_v2d.sharpa_episode import scale_about_centroid, transform_points


def load_upstream_module(upstream: Path, name: str) -> Any:
    """仅加载固定上游 task-library-loader 内的指定模块。"""
    if name not in {"read_mano", "contact_utils", "distance_utils"}:
        raise ValueError("不支持的上游模块")
    package = upstream / "robotic_grounding/source/robotic_grounding"
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
    path = upstream / f"reconstruction/modules/v2d_task_library_loader/lib/{name}.py"
    spec = importlib.util.spec_from_file_location(f"_sharpa_upstream_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scene_alignment(vertices: np.ndarray, object_first_pose: np.ndarray) -> np.ndarray:
    """官方单目标 PCA 落地启发式；不是重力或支撑面真值。"""
    from scipy.spatial.transform import Rotation

    world = vertices @ object_first_pose[:3, :3].T + object_first_pose[:3, 3]
    center = world.mean(axis=0)
    centered = world - center
    _, axes = np.linalg.eigh(centered.T @ centered)
    up = axes[:, int(np.argmin(np.ptp(centered @ axes, axis=0)))]
    if np.dot(up, center) > 0:
        up = -up
    rotation = Rotation.align_vectors(np.array([[0, 0, 1]]), up[None])[0].as_matrix()
    rotated = world @ rotation.T
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = [-rotated[:, 0].mean(), -rotated[:, 1].mean(), -rotated[:, 2].min()]
    return result


def mano_world_geometry(
    arrays: dict[str, np.ndarray],
    upstream: Path,
    mano_root: Path,
    alignment: np.ndarray,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """权威几何由 native-side FK＋质心缩放＋共同 SE(3) 得到。"""
    import torch
    from scipy.spatial.transform import Rotation

    module = load_upstream_module(upstream, "read_mano")
    mano = module.MANO(
        str(mano_root),
        device=torch.device("cpu"),
        center_idx=None,
        flat_hand_mean=True,
    )
    # Some distributed MANO_LEFT assets duplicate RIGHT's x shape basis. HaMeR
    # renders the right model then mirrors its vertices; correct only this runtime
    # tensor, never the licensed asset, and record the decision below.
    left_x = mano.left_mano_layer.th_shapedirs[:, 0, :]
    right_x = mano.right_mano_layer.th_shapedirs[:, 0, :]
    reference_norm = float(torch.linalg.norm(right_x))
    same_ratio = float(torch.linalg.norm(left_x - right_x)) / reference_norm
    mirrored_ratio = float(torch.linalg.norm(left_x + right_x)) / reference_norm
    shape_mirror_corrected = same_ratio < 0.01 and mirrored_ratio > 1.0
    if shape_mirror_corrected:
        mano.left_mano_layer.th_shapedirs[:, 0, :] *= -1
    torch.set_num_threads(4)
    n = len(arrays["camera_to_world_transform"])
    poses = alignment[None] @ arrays["camera_to_world_transform"].astype(np.float64)
    output: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, Any] = {}

    def tensor(value: np.ndarray) -> Any:
        return torch.from_numpy(np.asarray(value, dtype=np.float32))

    def fk(
        side: str, betas: np.ndarray, orient: np.ndarray, finger: np.ndarray, translation: np.ndarray
    ) -> dict[str, np.ndarray]:
        with torch.inference_mode():
            result = mano.forward(
                side, tensor(betas), tensor(orient), tensor(translation), tensor(finger)
            )
        return {key: value.detach().cpu().numpy() for key, value in result.items()}

    for side in ("left", "right"):
        prefix = f"hand_{side}"
        betas = arrays[prefix + "_betas"].astype(np.float64)
        original_orient = arrays[prefix + "_wrist_orient_in_camera"].astype(np.float64)
        original_finger = arrays[prefix + "_finger_pose"].reshape(n, 45).astype(np.float64)
        orient, finger = original_orient.copy(), original_finger.copy()
        cam_t = arrays[prefix + "_wrist_trans_in_camera"].astype(np.float64)
        if side == "left":
            orient[:, 1:] *= -1
            finger.reshape(n, 15, 3)[:, :, 1:] *= -1
        native = fk(side, betas, orient, finger, np.zeros((n, 3)))
        scales = arrays[prefix + "_scale"].astype(np.float64)
        vertices, joints = scale_about_centroid(native["vertices"], native["joints"], scales)
        world_vertices = transform_points(poses, vertices + cam_t[:, None])
        world_joints = transform_points(poses, joints + cam_t[:, None])
        native_r = Rotation.from_quat(native["joints_wxyz"].reshape(-1, 4)[:, [1, 2, 3, 0]])
        world_r = np.einsum(
            "tij,tkjl->tkil", poses[:, :3, :3], native_r.as_matrix().reshape(n, 21, 3, 3)
        )
        world_wxyz = (
            Rotation.from_matrix(world_r.reshape(-1, 3, 3))
            .as_quat()[:, [3, 0, 1, 2]]
            .reshape(n, 21, 4)
        )
        rest = fk(side, betas, np.zeros((1, 3)), np.zeros((1, 45)), np.zeros((1, 3)))
        j0 = rest["joints"][0, 0].astype(np.float64)
        orient_world = Rotation.from_matrix(
            poses[:, :3, :3] @ Rotation.from_rotvec(orient).as_matrix()
        ).as_rotvec()
        trans_world = np.einsum("tij,tj->ti", poses[:, :3, :3], cam_t + j0) + poses[:, :3, 3] - j0
        repose = fk(side, betas, orient_world, finger, trans_world)
        reposed_v, reposed_j = scale_about_centroid(repose["vertices"], repose["joints"], scales)
        error = float(
            max(np.abs(reposed_v - world_vertices).max(), np.abs(reposed_j - world_joints).max())
        )
        if error > 1e-4:
            raise ValueError(f"{side} world FK composition error {error} m")
        # Check the exact reconstruction renderer's RIGHT-layer + mirror-X convention.
        mirrored_error = 0.0
        mirrored_joint_error = 0.0
        middle_tip_convention_error = 0.0
        if side == "left":
            original = fk("right", betas, original_orient, original_finger, np.zeros((n, 3)))
            original["vertices"][:, :, 0] *= -1
            original["joints"][:, :, 0] *= -1
            original_v, original_j = scale_about_centroid(
                original["vertices"], original["joints"], scales
            )
            mirrored_error = float(np.abs(original_v - vertices).max())
            # manotorch intentionally selects vertex 445 for the native-left
            # middle tip and 444 for right. Keep the official native-side joint
            # convention; renderer equivalence is a surface test, not a demand
            # that these two different vertex landmarks be identical.
            joint_ids = [index for index in range(21) if index != 12]
            mirrored_joint_error = float(
                np.abs(original_j[:, joint_ids] - joints[:, joint_ids]).max()
            )
            middle_tip_convention_error = float(
                np.linalg.norm(original_j[:, 12] - joints[:, 12], axis=1).max()
            )
            if max(mirrored_error, mirrored_joint_error) > 1e-4:
                raise ValueError(
                    "native-left versus reconstruction mirror mismatch: "
                    f"surface {mirrored_error} m, matching joints {mirrored_joint_error} m"
                )
        accepted = arrays[prefix + "_source_code"] == 1
        if not accepted.any():
            raise ValueError(f"{side} has no accepted shape observation")
        metric_scale = float(np.median(scales[accepted]))
        _, open_joints = scale_about_centroid(
            rest["vertices"], rest["joints"], np.array([metric_scale])
        )
        output[side] = {
            "joints": world_joints,
            "joints_wxyz": world_wxyz,
            "vertices": world_vertices,
            "faces": native["faces"],
            "global_orient": orient_world,
            "trans": trans_world,
            "finger_pose": finger,
            "betas": betas,
            "open_joints_m": open_joints[0],
            "open_wrist_wxyz": rest["joints_wxyz"][0, 0],
            "metric_hand_scale": metric_scale,
        }
        diagnostics[side] = {
            "world_repose_max_abs_error_m": error,
            "native_left_vs_right_mirror_surface_max_abs_error_m": mirrored_error,
            "native_left_vs_right_matching_joints_max_abs_error_m": mirrored_joint_error,
            "left_middle_tip_445_vs_mirrored_right_444_max_distance_m": middle_tip_convention_error,
            "left_shape_basis_x_mirror_corrected_in_memory": shape_mirror_corrected
            if side == "left"
            else False,
            "left_shape_basis_same_relative_error_before": same_ratio if side == "left" else None,
            "left_shape_basis_mirror_relative_error_before": mirrored_ratio
            if side == "left"
            else None,
            "metric_scale_min": float(scales.min()),
            "metric_scale_max": float(scales.max()),
            "ignored_scale_max_joint_error_m": float(np.abs(repose["joints"] - world_joints).max()),
        }
    return output, diagnostics
