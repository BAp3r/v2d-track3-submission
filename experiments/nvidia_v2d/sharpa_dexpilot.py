"""Independent calibrated DexPilot + phalanx-chain route; CPU reference generation.

The historical custom chain adapter was not recoverable. This reconstruction
calls dex-retargeting 0.5.0's real DexPilotOptimizer, adding an explicit chain
term and a scalar/gradient-consistent temporal term. It is not the multi-tip
optimizer. Shared calibration, FK and surface sampling are read-only helpers.
"""

import argparse
import importlib.metadata
import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.nvidia_v2d.sharpa_calibration import CHAIN_PAIRS, TIPS
from experiments.nvidia_v2d.sharpa_multitip import sha256
from experiments.nvidia_v2d.sharpa_multitip_kinematics import SharpaLocalRobot

BACKEND = "dexpilot-chain-calibrated"
PRIMARY_SHA256 = "6b816d65c81f2e959ea6b76876bb03c2b768be95064a1b4b9e1563d36262cdf8"


@dataclass(frozen=True)
class DexPilotChainConfig:
    project_dist: float = 0.03
    escape_dist: float = 0.05
    eta1: float = 1e-4
    eta2: float = 0.03
    huber_delta: float = 0.03
    chain_weight: float = 0.25
    temporal_weight_30fps: float = 4e-3
    maxeval: int = 200

    def __post_init__(self) -> None:
        numbers = np.asarray(list(asdict(self).values()), dtype=float)
        if not np.isfinite(numbers).all() or np.any(numbers <= 0):
            raise ValueError("DexPilot parameters must be finite and positive")
        if self.project_dist >= self.escape_dist:
            raise ValueError("DexPilot hysteresis requires project_dist < escape_dist")


def primary_provenance() -> dict[str, Any]:
    from dex_retargeting.optimizer import DexPilotOptimizer

    source = Path(inspect.getfile(DexPilotOptimizer))
    version = importlib.metadata.version("dex_retargeting")
    digest = sha256(source)
    if version != "0.5.0" or digest != PRIMARY_SHA256:
        raise ValueError(f"unreviewed DexPilot implementation: {version} {digest}")
    return {
        "recovery_status": "rebuilt_from_installed_primary_code",
        "historical_custom_chain_source_recovered": False,
        "package": "dex-retargeting",
        "version": version,
        "class": "dex_retargeting.optimizer.DexPilotOptimizer",
        "source_repository": "https://github.com/dexsuite/dex-retargeting",
        "paper": "https://arxiv.org/abs/1910.03135",
        "source_path": str(source),
        "source_sha256": digest,
        "adaptations": [
            "Sharpa 22-DOF local-root FK using the existing calibrated hand scale",
            "Mean Huber phalanx-chain-vector loss, weight 0.25 over 16 vectors",
            "Set upstream norm_delta=0; add temporal scalar and matching gradient",
            "Temporal weight 0.004*((1/30)/dt)^2, no postfilter or velocity limiter",
            "Direct NLopt call; solver exceptions never silently reuse previous q",
        ],
        "not_reused": "vendor_dex_multitip.MultiTipIkOptimizer",
    }


class DexPilotChainOptimizer:
    """Compose the actual primary DexPilot objective with calibrated chain loss."""

    def __init__(
        self, robot: SharpaLocalRobot, config: DexPilotChainConfig | None = None
    ):
        from dex_retargeting.optimizer import DexPilotOptimizer

        self.provenance = primary_provenance()
        self.robot = robot
        self.config = config or DexPilotChainConfig()
        c = self.config
        self.primary = DexPilotOptimizer(
            robot,
            robot.dof_joint_names,
            [robot.point_names[i] for i in TIPS],
            robot.point_names[0],
            norm_delta=0.0,
            scaling=1.0,
            huber_delta=c.huber_delta,
            project_dist=c.project_dist,
            escape_dist=c.escape_dist,
            eta1=c.eta1,
            eta2=c.eta2,
        )
        self.primary.set_joint_limit(robot.joint_limits, epsilon=0)
        self.opt = self.primary.opt
        self.opt.set_maxeval(c.maxeval)
        self.chain_a = np.asarray([a for a, _ in CHAIN_PAIRS])
        self.chain_b = np.asarray([b for _, b in CHAIN_PAIRS])
        self.vector_a, self.vector_b = (
            np.asarray(indices, dtype=int)
            for indices in self.primary.generate_link_indices(5)
        )
        self.temporal_weight = c.temporal_weight_30fps

    def reference_vectors(self, targets: np.ndarray) -> np.ndarray:
        targets = np.asarray(targets, dtype=float)
        if targets.shape != (21, 3) or not np.isfinite(targets).all():
            raise ValueError("reference must be finite 21x3 calibrated points")
        points = targets[np.r_[0, TIPS]]
        return points[self.vector_b] - points[self.vector_a]

    def get_objective_function(
        self, targets: np.ndarray, previous: np.ndarray, dt: float
    ):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        vectors = self.reference_vectors(targets)
        base = self.primary.get_objective_function(vectors, np.empty(0), previous)
        chain_target = targets[self.chain_b] - targets[self.chain_a]
        weight = self.config.chain_weight / len(CHAIN_PAIRS)
        self.temporal_weight = self.config.temporal_weight_30fps * ((1 / 30) / dt) ** 2
        beta = self.config.huber_delta

        def objective(q: np.ndarray, gradient: np.ndarray) -> float:
            value = base(q, gradient)
            # base() already updated FK; do not apply a second root transform.
            poses = [self.robot.get_link_pose(i) for i in self.robot.point_indices]
            points = np.asarray([pose[:3, 3] for pose in poses])
            residual = points[self.chain_b] - points[self.chain_a] - chain_target
            distance = np.linalg.norm(residual, axis=1)
            value += (
                weight
                * np.where(
                    distance < beta, distance**2 / (2 * beta), distance - beta / 2
                ).sum()
            )
            delta = q - previous
            value += self.temporal_weight * np.dot(delta, delta)
            if gradient.size:
                jac = np.asarray(
                    [
                        pose[:3, :3]
                        @ self.robot.compute_single_link_local_jacobian(q, i)[:3]
                        for pose, i in zip(poses, self.robot.point_indices, strict=True)
                    ]
                )
                derivative = residual / np.maximum(distance, beta)[:, None]
                gradient += weight * np.einsum(
                    "ni,nij->j", derivative, jac[self.chain_b] - jac[self.chain_a]
                )
                gradient += 2 * self.temporal_weight * delta
            return float(value)

        return objective

    def solve(self, targets: np.ndarray, previous: np.ndarray, dt: float):
        objective = self.get_objective_function(targets, previous, dt)
        self.opt.set_min_objective(objective)
        result = np.asarray(self.opt.optimize(previous), dtype=float)
        status = int(self.opt.last_optimize_result())
        limits = self.robot.joint_limits
        if status < 1 or not np.isfinite(result).all():
            raise ValueError(f"DexPilot solver failed: {status}")
        if np.any(result < limits[:, 0] - 1e-8) or np.any(result > limits[:, 1] + 1e-8):
            raise ValueError("DexPilot solver returned out-of-limit q")
        return result, int(self.opt.get_numevals())


def main() -> None:
    from experiments.nvidia_v2d.sharpa_dexpilot_reference import run_retarget

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_retarget(args.input, args.calibration, args.upstream, args.output)
    print(
        json.dumps(
            {"status": result["status"], "result_sha256": result["result_sha256"]}
        )
    )


if __name__ == "__main__":
    main()
