import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.nvidia_v2d.sharpa_multitip_kinematics import (
    SharpaLocalRobot,
    build_optimizer,
    point_frame_names,
    reference_from_targets,
    solve_frame,
)


def test_sharpa_landmarks_preserve_side_and_distinct_finger_tips():
    for side in ("left", "right"):
        names = point_frame_names(side)
        assert len(names) == len(set(names)) == 21
        assert all(name.startswith(side + "_") for name in names)
        assert [names[i] for i in (4, 8, 12, 16, 20)] == [
            f"{side}_{finger}_tip_site" for finger in ("thumb", "index", "middle", "ring", "pinky")
        ]
    with pytest.raises(ValueError):
        point_frame_names("ambiguous")


def test_solver_failure_is_not_replaced_by_previous_pose():
    def fail(_):
        raise RuntimeError("solver failure")

    optimizer = SimpleNamespace(
        get_objective_function=lambda *args: None,
        opt=SimpleNamespace(set_min_objective=lambda _: None, optimize=fail),
    )
    with pytest.raises(RuntimeError, match="solver failure"):
        solve_frame(optimizer, None, np.zeros(22), 0.05)
    assert optimizer.temporal_weight == pytest.approx(4e-3 * (20 / 30) ** 2)
    with pytest.raises(ValueError):
        solve_frame(optimizer, None, np.zeros(22), 0)


def test_existing_results_are_never_overwritten(tmp_path):
    pytest.importorskip("pyarrow")
    from experiments.nvidia_v2d.sharpa_multitip import run_retarget

    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        run_retarget(tmp_path / "missing", tmp_path / "missing", tmp_path, output, "official-ik")
    assert marker.read_text() == "keep"


def test_missing_robot_fields_cannot_silently_drop_solved_motion(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from experiments.nvidia_v2d.sharpa_multitip import run_retarget

    source = tmp_path / "bad.parquet"
    pq.write_table(pa.table({"fps": [20.0]}), source)
    calibration = tmp_path / "calibration.json"
    calibration.write_text("{}")
    output = tmp_path / "new"
    with pytest.raises(ValueError, match="机器人字段"):
        run_retarget(source, calibration, tmp_path, output, "official-ik")
    assert not output.exists()


def robot_fixture(side):
    pytest.importorskip("pinocchio")
    pytest.importorskip("dex_retargeting")
    root = os.environ.get("FMC3_SHARPA_ASSETS")
    if not root:
        pytest.skip("requires separately verified Sharpa MJCF asset directory")
    return SharpaLocalRobot(Path(root) / f"{side}_sharpawave.xml", side)


@pytest.mark.parametrize("side", ["left", "right"])
def test_real_sharpa_zero_pose_and_world_fk_match(side):
    robot = robot_fixture(side)
    optimizer = build_optimizer(robot, side)
    points = robot.points(robot.q0)
    result, _ = solve_frame(optimizer, reference_from_targets(points), robot.q0, 0.05)
    np.testing.assert_allclose(robot.points(result), points, atol=1e-7)
    root = np.array([0.3, -0.2, 1.2])
    frames = robot.world_frames(result, root, np.eye(3))
    np.testing.assert_allclose(frames[robot.point_indices, :3], points + root, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(frames[:, 3:], axis=1), 1, atol=1e-8)
    assert frames.shape == (67, 7)


@pytest.mark.parametrize("side", ["left", "right"])
def test_transferred_optimizer_gradient_matches_finite_difference(side):
    robot = robot_fixture(side)
    optimizer = build_optimizer(robot, side)
    target_q = np.clip(np.full(22, 0.15), robot.joint_limits[:, 0], robot.joint_limits[:, 1])
    reference = reference_from_targets(robot.points(target_q))
    q = np.clip(np.full(22, 0.1), robot.joint_limits[:, 0], robot.joint_limits[:, 1])
    objective = optimizer.get_objective_function(reference, np.empty(0), q)
    analytic = np.empty(22)
    objective(q, analytic)
    numeric = np.empty(22)
    for index in range(22):
        delta = np.zeros(22)
        delta[index] = 1e-6
        numeric[index] = (
            objective(q + delta, np.empty(0)) - objective(q - delta, np.empty(0))
        ) / 2e-6
    np.testing.assert_allclose(analytic, numeric, atol=3e-6, rtol=1e-3)


@pytest.mark.parametrize("side", ["left", "right"])
def test_collision_surface_samples_are_repeatable_and_follow_world_root(side):
    from experiments.nvidia_v2d.sharpa_multitip_surface import CollisionSurfaceSampler

    robot = robot_fixture(side)
    xml = Path(os.environ["FMC3_SHARPA_ASSETS"]) / f"{side}_sharpawave.xml"
    sampler = CollisionSurfaceSampler(xml, robot.link_names, count=512, seed=42)
    again = CollisionSurfaceSampler(xml, robot.link_names, count=512, seed=42)
    origin = robot.world_frames(robot.q0, np.zeros(3), np.eye(3))
    shifted = robot.world_frames(robot.q0, np.array([1.0, -0.2, 0.4]), np.eye(3))
    points = sampler.world_points(origin)
    assert points.shape == (512, 3)
    np.testing.assert_array_equal(points, again.world_points(origin))
    np.testing.assert_allclose(sampler.world_points(shifted), points + [1.0, -0.2, 0.4], atol=1e-12)
    assert sampler.metadata["triangle_count"] > 100
    assert sampler.metadata["collision_geometry_count"] > 10
    assert len(sampler.metadata["asset_sha256"]) > 5
