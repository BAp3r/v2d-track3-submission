import itertools

import numpy as np
import pytest

from experiments.nvidia_v2d.ray_contact_refinement import solve_depth_path


def test_dynamic_path_matches_exhaustive_optimum():
    rng = np.random.default_rng(3)
    costs = rng.random((4, 3))
    positions = rng.normal(size=(4, 3, 3)) * 0.02
    path, cost = solve_depth_path(costs, positions)
    options = []
    for route in itertools.product(range(3), repeat=4):
        score = sum(costs[t, route[t]] for t in range(4))
        score += sum(
            np.square(
                (positions[t, route[t]] - positions[t - 1, route[t - 1]]) / 0.03
            ).sum()
            for t in range(1, 4)
        )
        options.append((score, route))
    expected_cost, expected_path = min(options)
    assert cost == pytest.approx(expected_cost)
    np.testing.assert_array_equal(path, expected_path)


def test_dynamic_path_rejects_unusable_inputs():
    with pytest.raises(ValueError):
        solve_depth_path(np.zeros((3, 2)), np.zeros((3, 3, 3)))
    with pytest.raises(ValueError):
        solve_depth_path(np.full((3, 2), np.nan), np.zeros((3, 2, 3)))
    with pytest.raises(ValueError):
        solve_depth_path(np.zeros((3, 2)), np.zeros((3, 2, 3)), step_scale_m=0)
