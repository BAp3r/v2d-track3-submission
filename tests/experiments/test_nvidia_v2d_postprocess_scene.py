import numpy as np

from experiments.nvidia_v2d.postprocess_scene import prove_shared_coordinates


def fixture():
    c = np.repeat(np.eye(4)[None], 3, axis=0)
    c[:, 0, 3] = [0, 1, 2]
    o = np.repeat(np.eye(4)[None], 3, axis=0)
    o[:, 2, 3] = 1
    data = {
        "camera_to_world_transform": c,
        "object_to_camera_transform": o,
        "object_to_world_transform": c @ o,
        "camera_is_valid": np.ones(3, bool),
        "camera_intrinsics": np.array([3, 3, 2, 2]),
    }
    fingerprints = {
        k: "same_hash"
        for k in ("source_video", "rgb_frames", "slam_trajectory", "slam_poses", "intrinsics")
    }
    return data, fingerprints


def test_same_source_world_can_be_assembled_without_alignment():
    a, f = fixture()
    b = {k: v.copy() for k, v in a.items()}
    b["object_to_camera_transform"][:, 1, 3] = 2
    b["object_to_world_transform"] = b["camera_to_world_transform"] @ b["object_to_camera_transform"]
    p = prove_shared_coordinates([a, b], [f, f])
    assert p["shared_coordinates_proven"]
    assert p["icp_used"] is False


def test_independently_centered_world_cannot_be_invented_by_matching_video():
    a, f = fixture()
    b = {k: v.copy() for k, v in a.items()}
    b["camera_to_world_transform"][:, 0, 3] += 9
    b["object_to_world_transform"] = b["camera_to_world_transform"] @ b["object_to_camera_transform"]
    p = prove_shared_coordinates([a, b], [f, f])
    assert not p["shared_coordinates_proven"]


def test_equal_arrays_without_shared_slam_provenance_are_insufficient():
    a, f = fixture()
    g = dict(f, slam_poses="different")
    assert not prove_shared_coordinates([a, a], [f, g])["shared_coordinates_proven"]


def test_inconsistent_object_world_composition_is_rejected():
    a, f = fixture()
    b = {k: v.copy() for k, v in a.items()}
    b["object_to_world_transform"][:, 2, 3] += 0.1
    assert not prove_shared_coordinates([a, b], [f, f])["shared_coordinates_proven"]


def test_frame_fingerprint_rejects_gaps_even_when_both_objects_share_them(tmp_path):
    import pytest

    from experiments.nvidia_v2d.postprocess_scene import _tree_digest

    (tmp_path / "000000.png").write_bytes(b"frame0")
    (tmp_path / "000002.png").write_bytes(b"frame2")
    with pytest.raises(ValueError, match="完整覆盖"):
        _tree_digest(tmp_path, 3)
