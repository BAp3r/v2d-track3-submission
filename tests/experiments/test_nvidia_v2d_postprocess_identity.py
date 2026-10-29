import numpy as np

from experiments.nvidia_v2d.postprocess_identity import cross_assignment_candidate, mask_features


def test_explicit_swap_is_flagged_without_relabelling():
    a = np.zeros((20, 20), bool)
    a[3:6, 2:5] = 1
    b = np.zeros_like(a)
    b[12:15, 15:18] = 1
    previous = [mask_features(a), mask_features(b)]
    assert cross_assignment_candidate(previous, previous)["possible_swap"] is False
    assert cross_assignment_candidate(previous, previous[::-1])["possible_swap"] is True


def test_missing_target_is_not_fabricated_as_swap():
    a = np.zeros((20, 20), bool)
    a[3:6, 2:5] = 1
    assert (
        cross_assignment_candidate(
            [mask_features(a), mask_features(a)], [mask_features(a), mask_features(np.zeros_like(a))]
        )["possible_swap"]
        is False
    )


def test_fragmented_mask_reports_largest_component():
    a = np.zeros((20, 20), bool)
    a[:2, :2] = 1
    a[10:12, 10:12] = 1
    assert mask_features(a)["largest_component_fraction"] == 0.5
