from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from dex_retargeting.optimizer import Optimizer


def _immutable_array(value: np.ndarray, *, dtype: Any | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class MultiTipIkReference:
    fingertip_targets: np.ndarray
    chain_vectors: np.ndarray
    chain_names: tuple[str, ...]
    pair_distance_targets: np.ndarray | None = None
    finger_straightness: np.ndarray | None = None
    finger_straightness_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        tips = np.asarray(self.fingertip_targets)
        chains = np.asarray(self.chain_vectors)
        names = tuple(self.chain_names)
        pair_targets = self.pair_distance_targets
        straightness = self.finger_straightness
        straightness_names = tuple(self.finger_straightness_names)
        if tips.shape != (5, 3) or not np.all(np.isfinite(tips)):
            raise ValueError("fingertip targets must have shape (5, 3) and be finite")
        if chains.ndim != 2 or chains.shape[1:] != (3,) or not np.all(np.isfinite(chains)):
            raise ValueError("chain vectors must have shape (N, 3) and be finite")
        if chains.shape[0] == 0:
            raise ValueError("chain vectors must not be empty")
        if len(names) != chains.shape[0]:
            raise ValueError("chain names length must match chain vectors")
        if any(not name for name in names):
            raise ValueError("chain names must not be empty")
        pairs = fingertip_pair_indices(tips.shape[0])
        if pair_targets is None:
            pair_targets = np.asarray(
                [np.linalg.norm(tips[second] - tips[first]) for first, second in pairs],
                dtype=tips.dtype,
            )
        else:
            pair_targets = np.asarray(pair_targets)
        if pair_targets.shape != (len(pairs),) or not np.all(np.isfinite(pair_targets)):
            raise ValueError(f"pair distance targets must have shape ({len(pairs)},) and be finite")
        if np.any(pair_targets < 0.0):
            raise ValueError("pair distance targets must be non-negative")
        if straightness is None:
            straightness = np.asarray([], dtype=tips.dtype)
        else:
            straightness = np.asarray(straightness)
        if straightness.ndim != 1 or not np.all(np.isfinite(straightness)):
            raise ValueError("finger straightness must be a finite 1D array")
        if np.any((straightness < 0.0) | (straightness > 1.0)):
            raise ValueError("finger straightness values must be in [0, 1]")
        if len(straightness_names) != straightness.shape[0]:
            raise ValueError("finger straightness names must match values")
        object.__setattr__(self, "fingertip_targets", _immutable_array(tips))
        object.__setattr__(self, "chain_vectors", _immutable_array(chains))
        object.__setattr__(self, "chain_names", names)
        object.__setattr__(self, "pair_distance_targets", _immutable_array(pair_targets))
        object.__setattr__(self, "finger_straightness", _immutable_array(straightness))
        object.__setattr__(self, "finger_straightness_names", straightness_names)

    def astype(self, dtype: Any) -> MultiTipIkReference:
        return MultiTipIkReference(
            fingertip_targets=self.fingertip_targets.astype(dtype),
            chain_vectors=self.chain_vectors.astype(dtype),
            chain_names=self.chain_names,
            pair_distance_targets=self.pair_distance_targets.astype(dtype),
            finger_straightness=self.finger_straightness.astype(dtype),
            finger_straightness_names=self.finger_straightness_names,
        )


def fingertip_pair_indices(count: int) -> tuple[tuple[int, int], ...]:
    if count < 2:
        raise ValueError("at least two fingertips are required")
    return tuple((first, second) for first in range(count) for second in range(first + 1, count))


def _relative_pair_weights(
    target: torch.Tensor,
    *,
    gap_epsilon: float = 0.003,
) -> torch.Tensor:
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("target fingertips must have shape (N, 3)")
    if not np.isfinite(gap_epsilon) or gap_epsilon <= 0.0:
        raise ValueError("relative gap epsilon must be positive and finite")
    pairs = fingertip_pair_indices(target.shape[0])
    target_vectors = torch.stack([target[second] - target[first] for first, second in pairs])
    target_gaps = torch.linalg.vector_norm(target_vectors, dim=1)
    weights = torch.rsqrt(target_gaps.square() + gap_epsilon**2)
    return weights / torch.clamp(weights.mean(), min=torch.finfo(weights.dtype).eps)


def _relative_geometry_loss(
    robot: torch.Tensor,
    target: torch.Tensor,
    *,
    beta: float = 0.1,
    gap_epsilon: float = 0.003,
) -> torch.Tensor:
    if robot.shape != target.shape or robot.ndim != 2 or robot.shape[1] != 3:
        raise ValueError("robot and target fingertips must have matching shape (N, 3)")
    pairs = fingertip_pair_indices(robot.shape[0])
    robot_vectors = torch.stack([robot[second] - robot[first] for first, second in pairs])
    target_vectors = torch.stack([target[second] - target[first] for first, second in pairs])
    pair_errors = torch.linalg.vector_norm(robot_vectors - target_vectors, dim=1)
    robust_errors = torch.nn.functional.smooth_l1_loss(
        pair_errors,
        torch.zeros_like(pair_errors),
        beta=beta,
        reduction="none",
    )
    weights = _relative_pair_weights(target, gap_epsilon=gap_epsilon)
    return torch.mean(robust_errors * weights)


def _pair_distance_loss(
    robot: torch.Tensor,
    target_gaps: torch.Tensor,
    *,
    beta: float = 0.1,
    gap_epsilon: float = 0.003,
) -> torch.Tensor:
    if robot.ndim != 2 or robot.shape[1] != 3:
        raise ValueError("robot fingertips must have shape (N, 3)")
    pairs = fingertip_pair_indices(robot.shape[0])
    if target_gaps.shape != (len(pairs),):
        raise ValueError(f"target pair gaps must have shape ({len(pairs)},)")
    robot_gaps = torch.stack(
        [torch.linalg.vector_norm(robot[second] - robot[first]) for first, second in pairs]
    )
    gap_errors = torch.abs(robot_gaps - target_gaps)
    robust_errors = torch.nn.functional.smooth_l1_loss(
        gap_errors,
        torch.zeros_like(gap_errors),
        beta=beta,
        reduction="none",
    )
    weights = torch.rsqrt(target_gaps.square() + gap_epsilon**2)
    weights = weights / torch.clamp(weights.mean(), min=torch.finfo(weights.dtype).eps)
    return torch.mean(robust_errors * weights)


def _chain_direction_loss(
    robot: torch.Tensor,
    target: torch.Tensor,
    *,
    beta: float = 0.1,
) -> torch.Tensor:
    robot_unit = robot / torch.clamp(
        torch.linalg.vector_norm(robot, dim=1, keepdim=True),
        min=1e-9,
    )
    target_unit = target / torch.clamp(
        torch.linalg.vector_norm(target, dim=1, keepdim=True),
        min=1e-9,
    )
    return torch.nn.functional.smooth_l1_loss(robot_unit, target_unit, beta=beta)


class MultiTipIkOptimizer(Optimizer):
    retargeting_type = "MULTITIP_IK_V2"

    def __init__(
        self,
        robot: Any,
        target_joint_names: list[str],
        finger_tip_link_names: list[str],
        chain_origin_link_names: list[str],
        chain_task_link_names: list[str],
        chain_names: tuple[str, ...],
        *,
        absolute_weight: float = 1.0,
        relative_weight: float = 1.0,
        pair_distance_weight: float = 1.0,
        chain_weight: float = 0.1,
        temporal_weight: float = 4e-3,
        huber_beta: float = 0.1,
        relative_gap_epsilon: float = 0.003,
        normalization_length: float | None = None,
        extension_joint_names: list[str] | None = None,
        extension_finger_names: tuple[str, ...] = (),
        extension_activation_start: float = 0.88,
        extension_activation_full: float = 0.92,
        extension_weight: float = 0.045,
    ) -> None:
        if len(finger_tip_link_names) != 5:
            raise ValueError("multi-tip IK requires exactly five fingertip links")
        if len(set(finger_tip_link_names)) != 5:
            raise ValueError("multi-tip IK fingertip links must be unique")
        if not (len(chain_origin_link_names) == len(chain_task_link_names) == len(chain_names)):
            raise ValueError("chain link and name counts must match")
        if not chain_names:
            raise ValueError("multi-tip IK requires at least one chain vector")
        weights = (
            absolute_weight,
            relative_weight,
            pair_distance_weight,
            chain_weight,
            temporal_weight,
            extension_weight,
        )
        if any(not np.isfinite(weight) or weight < 0.0 for weight in weights):
            raise ValueError("multi-tip IK weights must be finite and non-negative")
        if not np.isfinite(huber_beta) or huber_beta <= 0.0:
            raise ValueError("multi-tip IK Huber beta must be positive and finite")
        if not np.isfinite(relative_gap_epsilon) or relative_gap_epsilon <= 0.0:
            raise ValueError("relative gap epsilon must be positive and finite")
        if not (
            np.isfinite(extension_activation_start)
            and np.isfinite(extension_activation_full)
            and 0.0 <= extension_activation_start < extension_activation_full <= 1.0
        ):
            raise ValueError("extension activation thresholds must satisfy 0 <= start < full <= 1")

        super().__init__(
            robot,
            target_joint_names,
            np.arange(len(finger_tip_link_names), dtype=int),
        )
        self.tip_link_names = tuple(finger_tip_link_names)
        self.chain_origin_link_names = tuple(chain_origin_link_names)
        self.chain_task_link_names = tuple(chain_task_link_names)
        self.chain_names = tuple(chain_names)
        self.pair_indices = fingertip_pair_indices(len(finger_tip_link_names))
        self.absolute_weight = float(absolute_weight)
        self.relative_weight = float(relative_weight)
        self.pair_distance_weight = float(pair_distance_weight)
        self.chain_weight = float(chain_weight)
        self.temporal_weight = float(temporal_weight)
        self.huber_beta = float(huber_beta)
        self.relative_gap_epsilon = float(relative_gap_epsilon)
        self.extension_joint_names = tuple(extension_joint_names or ())
        self.extension_finger_names = tuple(extension_finger_names)
        if len(self.extension_joint_names) != len(self.extension_finger_names):
            raise ValueError("extension joint and finger counts must match")
        self.extension_activation_start = float(extension_activation_start)
        self.extension_activation_full = float(extension_activation_full)
        self.extension_target_indices = np.asarray(
            [self.target_joint_names.index(name) for name in self.extension_joint_names],
            dtype=int,
        )
        robot_q0 = np.asarray(getattr(self.robot, "q0", np.zeros(self.num_joints)), dtype=float)
        self.extension_targets = np.asarray(
            [
                robot_q0[self.robot.dof_joint_names.index(name)]
                for name in self.extension_joint_names
            ],
            dtype=float,
        )
        self.extension_weight = float(extension_weight)
        self.extension_weights = np.full(
            len(self.extension_finger_names),
            self.extension_weight,
            dtype=float,
        )
        self.computed_link_names = list(
            dict.fromkeys(
                [
                    *finger_tip_link_names,
                    *chain_origin_link_names,
                    *chain_task_link_names,
                ]
            )
        )
        self.computed_link_indices = self.get_link_indices(self.computed_link_names)
        self.tip_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in finger_tip_link_names]
        )
        self.chain_origin_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in chain_origin_link_names]
        )
        self.chain_task_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in chain_task_link_names]
        )
        scale = self._open_tip_span() if normalization_length is None else normalization_length
        if not np.isfinite(scale) or scale <= 1e-9:
            raise ValueError("multi-tip IK normalization length must be positive and finite")
        self.normalization_length = float(scale)
        self.opt.set_ftol_abs(1e-6)

    def get_objective_function(
        self,
        target_value: MultiTipIkReference,
        fixed_qpos: np.ndarray,
        last_qpos: np.ndarray,
    ):
        reference = self._require_reference(target_value)
        fixed = np.asarray(fixed_qpos, dtype=float)
        last = np.asarray(last_qpos, dtype=float)
        if fixed.shape != (len(self.idx_pin2fixed),):
            raise ValueError("fixed qpos shape does not match optimizer fixed joints")
        if last.shape != (len(self.idx_pin2target),):
            raise ValueError("last qpos shape does not match optimizer target joints")

        qpos = np.zeros(self.num_joints, dtype=float)
        qpos[self.idx_pin2fixed] = fixed
        target_tips = torch.tensor(reference.fingertip_targets, dtype=torch.float64)
        target_pair_gaps = torch.tensor(reference.pair_distance_targets, dtype=torch.float64)
        target_chains = torch.tensor(reference.chain_vectors, dtype=torch.float64)
        extension_activation = _extension_activation(
            reference.finger_straightness,
            start=self.extension_activation_start,
            full=self.extension_activation_full,
        )
        scale = self.normalization_length

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_pin2target] = x
            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)

            self.robot.compute_forward_kinematics(qpos)
            link_poses = [self.robot.get_link_pose(index) for index in self.computed_link_indices]
            body_positions = np.asarray([pose[:3, 3] for pose in link_poses], dtype=float)
            torch_body_positions = torch.as_tensor(body_positions, dtype=torch.float64)
            torch_body_positions.requires_grad_()

            robot_tips = torch_body_positions[self.tip_indices]
            tip_loss = torch.nn.functional.smooth_l1_loss(
                robot_tips / scale,
                target_tips / scale,
                beta=self.huber_beta,
            )
            relative_loss = _relative_geometry_loss(
                robot_tips / scale,
                target_tips / scale,
                beta=self.huber_beta,
                gap_epsilon=self.relative_gap_epsilon,
            )
            pair_distance_loss = _pair_distance_loss(
                robot_tips / scale,
                target_pair_gaps / scale,
                beta=self.huber_beta,
                gap_epsilon=self.relative_gap_epsilon,
            )
            robot_chains = (
                torch_body_positions[self.chain_task_indices]
                - torch_body_positions[self.chain_origin_indices]
            )
            chain_loss = _chain_direction_loss(
                robot_chains,
                target_chains,
                beta=self.huber_beta,
            )
            body_loss = (
                self.absolute_weight * tip_loss
                + self.relative_weight * relative_loss
                + self.pair_distance_weight * pair_distance_loss
                + self.chain_weight * chain_loss
            )
            temporal_loss = self.temporal_weight * float(np.mean(np.square(x - last)))
            extension_loss = _extension_loss(
                x,
                indices=self.extension_target_indices,
                targets=self.extension_targets,
                weights=self.extension_weights,
                activation=extension_activation,
            )
            result = float(body_loss.detach().cpu().item() + temporal_loss + extension_loss)

            if grad.size > 0:
                jacobians = []
                for link_index, link_pose in zip(
                    self.computed_link_indices,
                    link_poses,
                    strict=True,
                ):
                    local = self.robot.compute_single_link_local_jacobian(
                        qpos,
                        link_index,
                    )[:3]
                    jacobians.append(link_pose[:3, :3] @ local)
                jacobian = np.stack(jacobians)
                body_loss.backward()
                body_gradient = torch_body_positions.grad.detach().cpu().numpy()
                if self.adaptor is not None:
                    jacobian = self.adaptor.backward_jacobian(jacobian)
                else:
                    jacobian = jacobian[..., self.idx_pin2target]
                joint_gradient = np.einsum("li,lij->j", body_gradient, jacobian)
                joint_gradient += 2.0 * self.temporal_weight * (x - last) / max(x.size, 1)
                if self.extension_target_indices.size:
                    joint_gradient[self.extension_target_indices] += _extension_gradient(
                        x,
                        indices=self.extension_target_indices,
                        targets=self.extension_targets,
                        weights=self.extension_weights,
                        activation=extension_activation,
                    )
                grad[:] = joint_gradient

            return result

        return objective

    def measure(
        self,
        reference: MultiTipIkReference,
        qpos: np.ndarray,
    ) -> dict[str, float]:
        target = self._require_reference(reference)
        values = np.asarray(qpos, dtype=float)
        if values.shape != (self.num_joints,) or not np.all(np.isfinite(values)):
            raise ValueError(f"robot qpos must have shape ({self.num_joints},) and be finite")
        self.robot.compute_forward_kinematics(values)
        tips = np.asarray(
            [
                self.robot.get_link_pose(self.robot.get_link_index(name))[:3, 3]
                for name in self.tip_link_names
            ],
            dtype=float,
        )
        errors = np.linalg.norm(tips - target.fingertip_targets, axis=1)
        return {
            "tip_error_mean_m": float(np.mean(errors)),
            "tip_error_max_m": float(np.max(errors)),
            "thumb_index_gap_m": float(np.linalg.norm(tips[0] - tips[1])),
        }

    def _require_reference(self, value: MultiTipIkReference) -> MultiTipIkReference:
        if not isinstance(value, MultiTipIkReference):
            raise TypeError("MultiTipIkOptimizer requires MultiTipIkReference")
        if value.chain_names != self.chain_names:
            raise ValueError("chain reference names do not match optimizer chain names")
        if not self.extension_finger_names:
            if value.finger_straightness_names or value.finger_straightness.size:
                return MultiTipIkReference(
                    fingertip_targets=value.fingertip_targets,
                    chain_vectors=value.chain_vectors,
                    chain_names=value.chain_names,
                    pair_distance_targets=value.pair_distance_targets,
                )
            return value
        if value.finger_straightness_names != self.extension_finger_names:
            if not value.finger_straightness_names and value.finger_straightness.size == 0:
                return MultiTipIkReference(
                    fingertip_targets=value.fingertip_targets,
                    chain_vectors=value.chain_vectors,
                    chain_names=value.chain_names,
                    pair_distance_targets=value.pair_distance_targets,
                    finger_straightness=np.zeros(len(self.extension_finger_names), dtype=float),
                    finger_straightness_names=self.extension_finger_names,
                )
            raise ValueError("straightness reference names do not match optimizer names")
        return value

    def _open_tip_span(self) -> float:
        q0 = np.asarray(self.robot.q0, dtype=float)
        self.robot.compute_forward_kinematics(q0)
        tips = np.asarray(
            [
                self.robot.get_link_pose(self.robot.get_link_index(name))[:3, 3]
                for name in self.tip_link_names
            ],
            dtype=float,
        )
        return float(
            max(np.linalg.norm(tips[second] - tips[first]) for first, second in self.pair_indices)
        )


def _extension_activation(
    straightness: np.ndarray,
    *,
    start: float,
    full: float,
) -> np.ndarray:
    values = np.asarray(straightness, dtype=float)
    linear = np.clip((values - start) / (full - start), 0.0, 1.0)
    return linear * linear * (3.0 - 2.0 * linear)


def _extension_loss(
    x: np.ndarray,
    *,
    indices: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    activation: np.ndarray,
) -> float:
    if indices.size == 0:
        return 0.0
    delta = x[indices] - targets
    return float(np.sum(weights * activation * np.square(delta)) / max(indices.size, 1))


def _extension_gradient(
    x: np.ndarray,
    *,
    indices: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    activation: np.ndarray,
) -> np.ndarray:
    if indices.size == 0:
        return np.asarray([], dtype=float)
    return 2.0 * weights * activation * (x[indices] - targets) / max(indices.size, 1)
