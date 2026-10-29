import numpy as np
import pytest

from experiments.nvidia_v2d.policy_contact_recording import (
    CONTACT_WIDTHS,
    pack_contact_frames,
)


def frame(n):
    return {k: np.zeros((n,) if width is None else (n, width),
                        dtype=np.int64 if k.endswith("_ids") else np.float64)
            for k, width in CONTACT_WIDTHS.items()}


def test_ragged_contact_boundaries_keep_empty_frames_and_force_sign():
    a, b = frame(2), frame(1)
    a["body0_ids"][:] = [-1, 4]
    a["contact_force_w"][0] = [0, 0, -9.81]
    result = pack_contact_frames([frame(0), a, frame(0), b])
    np.testing.assert_array_equal(result["contact_frame_offsets"], [0, 0, 2, 2, 3])
    np.testing.assert_array_equal(result["contacts_body0_ids"], [-1, 4, 0])
    assert result["contacts_contact_force_w"][0, 2] == -9.81


def test_contact_recorder_rejects_incomplete_or_nonfinite_evidence():
    a = frame(2)
    a["signed_gap_m"] = np.array([np.nan, 0])
    with pytest.raises(ValueError, match="signed_gap_m"):
        pack_contact_frames([a])
    a = frame(2)
    a["contact_normal_w"] = np.zeros((1, 3))
    with pytest.raises(ValueError, match="contact_normal_w"):
        pack_contact_frames([a])


def test_no_contacts_remain_typed_and_pickle_free(tmp_path):
    result = pack_contact_frames([frame(0), frame(0)])
    path = tmp_path / "contacts.npz"
    np.savez_compressed(path, **result)
    with np.load(path, allow_pickle=False) as data:
        assert data["contacts_contact_force_w"].shape == (0, 3)
        np.testing.assert_array_equal(data["contact_frame_offsets"], [0, 0, 0])
