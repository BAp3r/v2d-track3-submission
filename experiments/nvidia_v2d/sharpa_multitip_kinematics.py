"""Sharpa MJCF adapter for the pinned Dex Teleop optimizer (CPU, no hardware)."""

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from experiments.nvidia_v2d.sharpa_calibration import CHAIN_NAMES, CHAIN_PAIRS, TIPS


def point_frame_names(side: str) -> tuple[str, ...]:
    if side not in {"left", "right"}:
        raise ValueError("必须指定左右手")
    suffixes = [
        "hand_C_MC",
        "thumb_CMC_VL_site",
        "thumb_MCP_VL_site",
        "thumb_DP_site",
        "thumb_tip_site",
    ]
    for finger in ("index", "middle", "ring", "pinky"):
        suffixes.extend(
            [
                f"{finger}_MCP_VL" if finger == "pinky" else f"{finger}_MCP_VL_site",
                f"{finger}_MP_site",
                f"{finger}_DP_site",
                f"{finger}_tip_site",
            ]
        )
    return tuple(f"{side}_{suffix}" for suffix in suffixes)


class SharpaLocalRobot:
    """Keep the official 67-frame free-root MJCF, exposing only 22 finger DOFs."""

    def __init__(self, xml_path: Path, side: str) -> None:
        import pinocchio as pin

        self.pin = pin
        result = pin.buildModelFromMJCF(str(xml_path), pin.JointModelFreeFlyer(), "root_joint")
        self.model = result[0] if isinstance(result, tuple) else result
        if isinstance(result, tuple) and len(result) > 1 and len(result[1]):
            raise ValueError("不支持带额外闭链约束的机器人模型")
        self.data = self.model.createData()
        self.dof_joint_names = list(self.model.names)[2:]
        self.dof = self.model.nq - 7
        self.q0 = np.asarray(pin.neutral(self.model))[7:].copy()
        self.link_names = [frame.name for frame in self.model.frames]
        self.joint_limits = np.column_stack(
            (self.model.lowerPositionLimit[7:], self.model.upperPositionLimit[7:])
        )
        if self.dof != 22 or self.model.nv != 28 or len(self.link_names) != 67:
            raise ValueError("Sharpa 模型与官方 22 关节、67 帧合同不匹配")
        if len(set(self.dof_joint_names)) != 22 or not all(
            name.startswith(side + "_") for name in self.dof_joint_names
        ):
            raise ValueError("Sharpa 关节侧别或名称不匹配")
        self.point_names = point_frame_names(side)
        self.point_indices = [self.get_link_index(name) for name in self.point_names]
        self.compute_forward_kinematics(self.q0)

    def get_link_index(self, name: str) -> int:
        if name not in self.link_names:
            raise ValueError(f"Sharpa 缺少帧: {name}")
        return self.link_names.index(name)

    def full_q(self, fingers: NDArray[np.float64]) -> NDArray[np.float64]:
        fingers = np.asarray(fingers, dtype=float)
        if fingers.shape != (22,) or not np.isfinite(fingers).all():
            raise ValueError("手指关节必须是 22 个有限弧度值")
        q = np.asarray(self.pin.neutral(self.model)).copy()
        q[7:] = fingers
        return q

    def compute_forward_kinematics(self, qpos: NDArray[np.float64]) -> None:
        self.pin.forwardKinematics(self.model, self.data, self.full_q(qpos))
        self.pin.updateFramePlacements(self.model, self.data)

    def get_link_pose(self, index: int) -> NDArray[np.float64]:
        return np.asarray(self.data.oMf[index].homogeneous).copy()

    def compute_single_link_local_jacobian(
        self, qpos: NDArray[np.float64], index: int
    ) -> NDArray[np.float64]:
        jac = self.pin.computeFrameJacobian(
            self.model,
            self.data,
            self.full_q(qpos),
            index,
            self.pin.ReferenceFrame.LOCAL,
        )
        return np.asarray(jac)[:, 6:]

    def points(self, qpos: NDArray[np.float64]) -> NDArray[np.float64]:
        self.compute_forward_kinematics(qpos)
        return np.asarray([self.get_link_pose(index)[:3, 3] for index in self.point_indices])

    def world_frames(
        self,
        qpos: NDArray[np.float64],
        root_position: NDArray[np.float64],
        root_rotation: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        from scipy.spatial.transform import Rotation

        q = self.full_q(qpos)
        q[:3] = root_position
        q[3:7] = Rotation.from_matrix(root_rotation).as_quat()
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        return np.asarray(
            [
                np.r_[
                    self.data.oMf[i].translation,
                    Rotation.from_matrix(self.data.oMf[i].rotation).as_quat(scalar_first=True),
                ]
                for i in range(len(self.link_names))
            ]
        )


def build_optimizer(robot: SharpaLocalRobot, side: str) -> Any:
    from experiments.nvidia_v2d.vendor_dex_multitip.multitip_optimizer import MultiTipIkOptimizer

    optimizer = MultiTipIkOptimizer(
        robot=robot,
        target_joint_names=robot.dof_joint_names,
        finger_tip_link_names=[robot.point_names[i] for i in TIPS],
        chain_origin_link_names=[robot.point_names[a] for a, _ in CHAIN_PAIRS],
        chain_task_link_names=[robot.point_names[b] for _, b in CHAIN_PAIRS],
        chain_names=CHAIN_NAMES,
        extension_joint_names=[
            f"{side}_{finger}_PIP" for finger in ("index", "middle", "ring", "pinky")
        ],
        extension_finger_names=("index", "middle", "ring", "pinky"),
    )
    optimizer.set_joint_limit(robot.joint_limits, epsilon=0.0)
    optimizer.opt.set_maxeval(200)
    return optimizer


def reference_from_targets(targets: NDArray[np.float64]) -> Any:
    from experiments.nvidia_v2d.vendor_dex_multitip.multitip_optimizer import MultiTipIkReference

    straightness = []
    for start in (5, 9, 13, 17):
        chain = targets[start : start + 4]
        length = np.linalg.norm(np.diff(chain, axis=0), axis=1).sum()
        if length < 1e-8:
            raise ValueError("退化手指链不能进入重定向")
        straightness.append(float(np.clip(np.linalg.norm(chain[-1] - chain[0]) / length, 0, 1)))
    return MultiTipIkReference(
        fingertip_targets=targets[TIPS],
        chain_vectors=np.asarray([targets[b] - targets[a] for a, b in CHAIN_PAIRS]),
        chain_names=CHAIN_NAMES,
        finger_straightness=np.asarray(straightness),
        finger_straightness_names=("index", "middle", "ring", "pinky"),
    )


def solve_frame(
    optimizer: Any, reference: Any, previous: NDArray[np.float64], dt: float
) -> tuple[NDArray[np.float64], int]:
    """Do not use upstream Optimizer.retarget: it silently returns the old pose on error."""
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("时间步长无效")
    optimizer.temporal_weight = 4e-3 * ((1 / 30) / dt) ** 2
    objective = optimizer.get_objective_function(reference, np.empty(0), previous)
    optimizer.opt.set_min_objective(objective)
    result = np.asarray(optimizer.opt.optimize(previous), dtype=float)
    status = int(optimizer.opt.last_optimize_result())
    if status < 1 or not np.isfinite(result).all():
        raise ValueError(f"multi-tip 求解失败: {status}")
    limits = optimizer.robot.joint_limits
    if np.any(result < limits[:, 0] - 1e-8) or np.any(result > limits[:, 1] + 1e-8):
        raise ValueError("multi-tip 结果超出关节限位")
    return result, int(optimizer.opt.get_numevals())
