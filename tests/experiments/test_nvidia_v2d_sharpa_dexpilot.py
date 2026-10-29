from pathlib import Path

import numpy as np
import pytest

from experiments.nvidia_v2d.sharpa_dexpilot import (
    DexPilotChainConfig,
    DexPilotChainOptimizer,
    primary_provenance,
)
from experiments.nvidia_v2d.sharpa_multitip_kinematics import SharpaLocalRobot


@pytest.fixture(params=["left", "right"])
def robot(request):
    pytest.importorskip("pinocchio")
    pytest.importorskip("dex_retargeting")
    root = (
        Path(__file__).resolve().parents[2]
        / "robotic_grounding/source/robotic_grounding/robotic_grounding/assets/xmls/sharpawave"
    )
    return SharpaLocalRobot(root / f"{request.param}_sharpawave.xml", request.param)


def test_pinned_primary_implementation_and_no_multitip_substitution():
    pytest.importorskip("dex_retargeting")
    provenance = primary_provenance()
    assert provenance["class"] == "dex_retargeting.optimizer.DexPilotOptimizer"
    assert provenance["recovery_status"] == "rebuilt_from_installed_primary_code"
    assert provenance["historical_custom_chain_source_recovered"] is False


def test_config_rejects_invalid_hysteresis_and_nonfinite():
    with pytest.raises(ValueError):
        DexPilotChainConfig(project_dist=0.06)
    with pytest.raises(ValueError):
        DexPilotChainConfig(chain_weight=float("nan"))


def test_dexpilot_projection_hysteresis_is_active(robot):
    solver = DexPilotChainOptimizer(robot)
    targets = robot.points(robot.q0)
    # Move index tip towards thumb: primary DexPilot projects, then remembers
    # the projected state in the 3..5 cm hysteresis band, then releases it.
    for distance, projected in [(0.02, True), (0.04, True), (0.06, False)]:
        targets[8] = targets[4] + [distance, 0, 0]
        solver.get_objective_function(targets, robot.q0, 0.05)
        assert bool(solver.primary.projected[0]) == projected
    assert solver.reference_vectors(targets).shape == (15, 3)


def test_complete_objective_gradient_matches_finite_difference(robot):
    solver = DexPilotChainOptimizer(robot)
    target_q = np.clip(np.full(22, 0.22), *robot.joint_limits.T)
    targets = robot.points(target_q)
    q = np.clip(np.full(22, 0.1), *robot.joint_limits.T)
    previous = np.clip(q - 0.02, *robot.joint_limits.T)
    objective = solver.get_objective_function(targets, previous, 0.05)
    gradient = np.empty(22)
    objective(q, gradient)
    numeric = np.empty(22)
    for index in range(22):
        delta = np.zeros(22)
        delta[index] = 1e-6
        numeric[index] = (
            objective(q + delta, np.empty(0)) - objective(q - delta, np.empty(0))
        ) / 2e-6
    np.testing.assert_allclose(gradient, numeric, atol=3e-6, rtol=1e-3)
    with pytest.raises(ValueError):
        solver.get_objective_function(targets, previous, 0)


def test_solver_keeps_fixed_root_and_returns_valid_fk(robot):
    solver = DexPilotChainOptimizer(robot)
    targets = robot.points(robot.q0)
    result, iterations = solver.solve(targets, robot.q0, 0.05)
    assert np.isfinite(result).all() and iterations > 0
    assert np.all(result >= robot.joint_limits[:, 0] - 1e-8)
    assert np.all(result <= robot.joint_limits[:, 1] + 1e-8)
    frames = robot.world_frames(result, np.array([0.2, -0.1, 1.0]), np.eye(3))
    assert frames.shape == (67, 7)
    np.testing.assert_allclose(
        frames[robot.point_indices[0], :3], targets[0] + [0.2, -0.1, 1.0]
    )


def test_existing_output_is_never_overwritten(tmp_path):
    pytest.importorskip("pyarrow")
    from experiments.nvidia_v2d.sharpa_dexpilot_reference import run_retarget

    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        run_retarget(tmp_path / "missing", tmp_path / "missing", tmp_path, output)
