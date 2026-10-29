import numpy as np
import pytest

from experiments.nvidia_v2d.sensor_depth_comparison import depth_comparison


def test_scale_bias_is_measured_on_the_same_visible_pixels():
    sensor = np.array([[0.5, 1.0], [2.0, 4.0]])
    estimated = sensor * 1.5
    valid = np.ones((2, 2), dtype=bool)
    mask = np.array([[True, True], [False, False]])
    result = depth_comparison(estimated, sensor, valid, mask)
    assert result["paired_pixels"] == 2
    assert result["estimated_to_sensor_ratio_median"] == pytest.approx(1.5)
    assert result["signed_error_median_m"] == pytest.approx(0.375)


def test_invalid_measurements_and_mask_exterior_never_enter_error_statistics():
    estimated = np.array([[1.2, np.nan, 30.0], [20.0, 3.0, 2.0]])
    sensor = np.array([[1.0, 1.0, 30.0], [0.0, np.inf, 1.0]])
    valid = np.array([[True, True, False], [True, True, True]])
    mask = np.array([[True, True, True], [True, True, False]])
    result = depth_comparison(estimated, sensor, valid, mask)
    assert result["paired_pixels"] == 1
    assert result["absolute_error_median_m"] == pytest.approx(0.2)


def test_no_paired_pixels_does_not_invent_an_accuracy_measurement():
    result = depth_comparison(np.zeros((2, 2)), np.ones((2, 2)), np.ones((2, 2), dtype=bool))
    assert result == {"paired_pixels": 0}


@pytest.mark.parametrize("shape", [(3, 2), (4,)])
def test_incompatible_rgb_pixel_grids_are_rejected(shape):
    with pytest.raises(ValueError):
        depth_comparison(np.ones(shape), np.ones((2, 2)), np.ones((2, 2), dtype=bool))
