import numpy as np
import pytest

from experiments.nvidia_v2d.postprocess_occlusion import visible_depth_support


def test_overlap_removed_from_observation_support_but_original_membership_restored():
    a = np.zeros((2, 10, 10), bool)
    a[0, 2:8, 1:6] = True
    a[1, 3:7, 4:9] = True
    visible, ambiguous, membership = visible_depth_support(a)
    assert ambiguous.sum() == 8
    assert not np.any(visible.sum(0) > 1)
    assert np.array_equal(visible | membership, a)
    assert np.array_equal(visible.any(0) | ambiguous, a.any(0))


def test_three_objects_keep_per_object_ambiguity_provenance():
    a = np.zeros((3, 4, 4), bool)
    a[0, :2, :2] = 1
    a[1, 1:3, 1:3] = 1
    a[2, 2:, 2:] = 1
    visible, ambiguous, membership = visible_depth_support(a)
    assert ambiguous.sum() == 2
    assert membership[0, 1, 1] and not membership[2, 1, 1]
    assert np.array_equal(visible | membership, a)


def test_empty_object_stays_empty_without_inpainting():
    a = np.zeros((2, 4, 4), bool)
    a[0] = True
    visible, ambiguous, membership = visible_depth_support(a)
    assert not ambiguous.any() and not membership.any()
    assert not visible[1].any()


def test_nonboolean_mask_requires_explicit_conversion():
    with pytest.raises(ValueError):
        visible_depth_support(np.ones((2, 4, 4), np.uint8))
