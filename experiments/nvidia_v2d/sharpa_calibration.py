"""Fixed MANO-shape calibration, with scene scale and wrist motion kept separate."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

MANO_JOINT_NAMES = ("wrist",) + tuple(
    f"{finger}{joint}"
    for finger in ("thumb", "index", "middle", "ring", "pinky")
    for joint in range(1, 5)
)
TIPS = np.array([4, 8, 12, 16, 20])
CHAIN_PAIRS = tuple(
    [(0, 1), (1, 2), (2, 3), (3, 4)]
    + [(start + offset, start + offset + 1) for start in (5, 9, 13, 17) for offset in range(3)]
)
CHAIN_NAMES = tuple(f"{MANO_JOINT_NAMES[a]}->{MANO_JOINT_NAMES[b]}" for a, b in CHAIN_PAIRS)


def _points(value: ArrayLike) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (21, 3) or not np.isfinite(result).all():
        raise ValueError("手关节必须为有限的 (21,3) 米制坐标")
    return result


def palm_frame(points: ArrayLike) -> NDArray[np.float64]:
    """Dex Teleop palm axes, rejecting degenerate geometry instead of guessing."""
    points = _points(points)
    forward = points[[5, 9, 13]].mean(axis=0) - points[0]
    lateral = points[5] - points[17]
    normal = np.cross(forward, lateral)
    if np.linalg.norm(forward) < 1e-8 or np.linalg.norm(normal) < 1e-10:
        raise ValueError("掌坐标退化，不能标定")
    forward /= np.linalg.norm(forward)
    normal /= np.linalg.norm(normal)
    return np.column_stack((forward, np.cross(normal, forward), normal))


def require_rotation(rotation: ArrayLike) -> NDArray[np.float64]:
    result = np.asarray(rotation, dtype=np.float64)
    if (
        result.shape != (3, 3)
        or not np.isfinite(result).all()
        or not np.allclose(result.T @ result, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(result), 1, atol=1e-5)
    ):
        raise ValueError("腕姿必须为有效的右手旋转矩阵")
    return result


@dataclass(frozen=True)
class ShapeCalibration:
    side: str
    scale: float
    source_to_robot_rotation: NDArray[np.float64]
    robot_wrist: NDArray[np.float64]

    def transform(
        self,
        world_joints: ArrayLike,
        world_wrist_rotation: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return local robot targets and a world root pose without scaling translation."""
        points = _points(world_joints)
        wrist_rotation = require_rotation(world_wrist_rotation)
        local = (points - points[0]) @ wrist_rotation
        targets = self.robot_wrist + (local @ self.source_to_robot_rotation.T) * self.scale
        root_rotation = wrist_rotation @ self.source_to_robot_rotation.T
        root_position = points[0] - root_rotation @ self.robot_wrist
        return targets, root_position, root_rotation

    def as_dict(self) -> dict[str, object]:
        return {
            "side": self.side,
            "scale": self.scale,
            "source_to_robot_rotation": self.source_to_robot_rotation.tolist(),
            "robot_wrist_m": self.robot_wrist.tolist(),
            "scope": "hand_local_morphology_only",
            "scene_scale_changed": False,
        }


def calibrate_open_hand(
    source_open_joints: ArrayLike,
    robot_open_joints: ArrayLike,
    side: str,
    *,
    source_open_wrist_rotation: ArrayLike | None = None,
) -> ShapeCalibration:
    """Use the original median five wrist-to-tip ratio, from a fixed open shape."""
    if side not in {"left", "right"}:
        raise ValueError("必须明确左右手")
    source, robot = _points(source_open_joints), _points(robot_open_joints)
    # MANO's native zero-pose model axes and anatomical wrist axes differ for
    # the left hand. The same J0 basis must be used here and at every frame.
    if source_open_wrist_rotation is not None:
        source = (source - source[0]) @ require_rotation(source_open_wrist_rotation)
    source_frame, robot_frame = palm_frame(source), palm_frame(robot)
    source_lengths = np.linalg.norm(source[TIPS] - source[0], axis=1)
    robot_lengths = np.linalg.norm(robot[TIPS] - robot[0], axis=1)
    if np.any(source_lengths < 1e-6) or np.any(robot_lengths < 1e-6):
        raise ValueError("展开手指长度无效")
    scale = float(np.median(robot_lengths / source_lengths))
    rotation = robot_frame @ source_frame.T
    return ShapeCalibration(side, scale, rotation, robot[0].copy())


def require_timeline(
    timestamps: ArrayLike,
    source_indices: ArrayLike,
    count: int,
    fps: float,
) -> NDArray[np.float64]:
    times = np.asarray(timestamps, dtype=np.float64)
    indices = np.asarray(source_indices)
    if (
        times.shape != (count,)
        or count < 2
        or not np.isfinite(times).all()
        or indices.shape != (count,)
        or not np.issubdtype(indices.dtype, np.integer)
        or np.any(np.diff(times) <= 0)
        or np.any(np.diff(indices) != 1)
        or not np.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError("时间轴或源帧映射无效，不能隐式补帧")
    if not np.allclose(np.diff(times), 1 / fps, rtol=1e-3, atol=1e-6):
        raise ValueError("Parquet FPS 与实际时间轴不一致；需显式重采样")
    return times.copy()


def validate_source_hand(
    joints: ArrayLike,
    wxyz: ArrayLike,
    valid: ArrayLike,
    source_code: ArrayLike,
    joint_names: list[str],
    *, allow_inferred_hands: bool = False,
) -> None:
    points = np.asarray(joints, dtype=np.float64)
    quats = np.asarray(wxyz, dtype=np.float64)
    validity, codes = np.asarray(valid), np.asarray(source_code)
    count = len(points)
    if tuple(joint_names) != MANO_JOINT_NAMES:
        raise ValueError("MANO 关节拓扑与官方顺序不一致")
    if (
        points.shape != (count, 21, 3)
        or quats.shape != (count, 21, 4)
        or not np.isfinite(points).all()
        or not np.isfinite(quats).all()
        or not np.allclose(np.linalg.norm(quats, axis=-1), 1, atol=1e-4)
    ):
        raise ValueError("手姿、旋转包含无效数值")
    if (
        validity.shape != (count,)
        or validity.dtype != np.bool_
        or not validity.all()
        or codes.shape != (count,)
        or not np.isin(codes, [1, 2, 3]).all()
    ):
        raise ValueError("完整训练序列包含缺失、拒绝或外推手帧")
