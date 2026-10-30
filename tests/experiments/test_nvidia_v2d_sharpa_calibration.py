import numpy as np
import pytest

from experiments.nvidia_v2d.sharpa_calibration import (
    MANO_JOINT_NAMES,
    calibrate_open_hand,
    require_timeline,
    validate_source_hand,
)


def open_points():
    points = np.zeros((21, 3))
    for finger in range(5):
        for segment in range(4):
            points[1 + finger * 4 + segment] = (
                (2 - finger) * 0.02,
                0.04 + segment * 0.025,
                0.003 * finger,
            )
    return points


def test_fixed_morphology_scale_does_not_scale_world_wrist():
    source = open_points()
    robot = source * 1.3 + [0.01, 0.02, 0.03]
    calibration = calibrate_open_hand(source, robot, "left")
    assert calibration.scale == pytest.approx(1.3)
    wrist = np.array([5.0, 2.0, -1.0])
    points = source + wrist
    targets, root_position, root_rotation = calibration.transform(points, np.eye(3))
    np.testing.assert_allclose(targets, robot, atol=1e-12)
    np.testing.assert_allclose(root_rotation @ calibration.robot_wrist + root_position, wrist)
    np.testing.assert_allclose(
        targets @ root_rotation.T + root_position, source * 1.3 + wrist, atol=1e-12
    )


def test_world_rotation_is_preserved_through_fixed_palm_mapping():
    source = open_points()
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    robot = source @ rotation.T * 1.2
    calibration = calibrate_open_hand(source, robot, "right")
    wrist = np.array([0.3, 0.4, 0.5])
    targets, root, orientation = calibration.transform(source @ rotation.T + wrist, rotation)
    np.testing.assert_allclose(targets @ orientation.T + root, source @ rotation.T * 1.2 + wrist)
    assert np.linalg.det(orientation) == pytest.approx(1.0)


def test_native_left_open_shape_uses_its_anatomical_wrist_axes():
    source = open_points()
    anatomical_axes = np.diag([-1.0, 1.0, -1.0])
    robot = source * 1.3
    calibration = calibrate_open_hand(
        source, robot, "left", source_open_wrist_rotation=anatomical_axes
    )
    targets, root, orientation = calibration.transform(source, anatomical_axes)
    np.testing.assert_allclose(targets, robot, atol=1e-12)
    np.testing.assert_allclose(orientation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(targets @ orientation.T + root, robot, atol=1e-12)


def test_degenerate_palm_or_invalid_side_is_rejected():
    with pytest.raises(ValueError):
        calibrate_open_hand(np.zeros((21, 3)), open_points(), "left")
    with pytest.raises(ValueError):
        calibrate_open_hand(open_points(), open_points(), "middle")


def test_twenty_fps_is_preserved_and_timestamps_are_not_regenerated():
    times = np.arange(249) / 20.0
    result = require_timeline(times, list(range(249)), 249, 20)
    np.testing.assert_array_equal(result, times)
    assert result[-1] == 12.4
    with pytest.raises(ValueError):
        require_timeline(times, list(range(249)), 249, 30)


@pytest.mark.parametrize("times,indices", [([0, 0], [0, 1]), ([0, 0.05], [1, 0])])
def test_non_monotonic_timeline_is_rejected(times, indices):
    with pytest.raises(ValueError):
        require_timeline(times, indices, 2, 20)


def test_invalid_frames_rejected_but_labeled_extrapolation_allowed():
    kwargs = dict(
        joints=np.repeat(open_points()[None], 2, axis=0),
        wxyz=np.tile([1.0, 0, 0, 0], (2, 21, 1)),
        joint_names=list(MANO_JOINT_NAMES),
    )
    validate_source_hand(**kwargs, valid=[True, True], source_code=[1, 2])
    with pytest.raises(ValueError):
        validate_source_hand(**kwargs, valid=[True, False], source_code=[1, 4])
    validate_source_hand(**kwargs, valid=[True, True], source_code=[1, 3])


def test_quaternion_and_topology_contracts_are_checked():
    kwargs = dict(joints=open_points()[None], valid=[True], source_code=[1])
    with pytest.raises(ValueError):
        validate_source_hand(**kwargs, wxyz=np.zeros((1, 21, 4)), joint_names=list(MANO_JOINT_NAMES))
    with pytest.raises(ValueError):
        validate_source_hand(
            **kwargs,
            wxyz=np.tile([1, 0, 0, 0], (1, 21, 1)),
            joint_names=list(reversed(MANO_JOINT_NAMES)),
        )
