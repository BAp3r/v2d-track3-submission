import numpy as np
import pytest

from experiments.nvidia_v2d.ray_contact_feasibility import (
    lipschitz_lower_bound,
    projection_interval,
)


def test_distance_lower_bound_covers_between_samples_and_endpoints():
    grid = np.linspace(-0.18, 0.18, 121)
    # Distance to a point offset perpendicular to the unit translation ray.
    values = np.hypot(grid - 0.0015, 0.01)
    bound = lipschitz_lower_bound(grid, values)
    assert 0.0085 < bound < 0.01
    assert bound <= np.hypot(np.linspace(-0.18, 0.18, 10000) - 0.0015, 0.01).min()
    with pytest.raises(ValueError, match="1-Lipschitz"):
        lipschitz_lower_bound(np.array([0, 0.01]), np.array([0, 0.1]))


def test_projection_interval_solves_both_asymmetric_perspective_branches():
    def error(shift):
        return 100 * abs(shift) / (1 + shift)

    low, high = projection_interval(error, budget_px=10, shift_limit_m=0.18)
    assert low == pytest.approx(-1 / 11)
    assert high == pytest.approx(1 / 9)
    assert error(low) <= 10 and error(high) <= 10
    assert projection_interval(error, budget_px=100, shift_limit_m=0.18) == (-0.18, 0.18)
