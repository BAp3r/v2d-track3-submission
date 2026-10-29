"""CPU checks for coordinate and source-observation diagnostic claims."""

from pathlib import Path

import numpy as np
import pytest

from experiments.nvidia_v2d import hand_object_diagnostics as diagnostics


def _pose(translation: tuple[float, float, float], rotation: np.ndarray | None = None):
    value = np.eye(4)
    value[:3, 3] = translation
    if rotation is not None:
        value[:3, :3] = rotation
    return value


def _transforms():
    camera = _pose((3.0, -4.0, 2.0), np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]))
    obj = _pose((0.4, -0.2, 1.5), np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))
    hand = _pose((0.7, -0.6, 1.9), np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))
    return camera, obj, hand


def test_common_camera_motion_preserves_relative_pose_and_distance():
    camera, obj, hand = _transforms()
    result = diagnostics.relative_pose_metrics(camera, obj, hand, camera @ obj, camera @ hand)

    assert result["relative_transform_max_abs_error"] == pytest.approx(0.0, abs=1e-12)
    assert result["camera_composition_max_abs_error"] == pytest.approx(0.0, abs=1e-12)
    assert result["origin_distance_camera_m"] == pytest.approx(np.sqrt(0.41))
    assert result["origin_distance_world_m"] == pytest.approx(np.sqrt(0.41))
    assert result["origin_distance_invariance_error_m"] == pytest.approx(0.0, abs=1e-12)


def test_missing_object_world_transform_is_detected():
    camera, obj, hand = _transforms()
    result = diagnostics.relative_pose_metrics(camera, obj, hand, obj, camera @ hand)

    assert result["relative_transform_max_abs_error"] > 1.0
    assert result["camera_composition_max_abs_error"] > 1.0
    assert result["origin_distance_invariance_error_m"] > 1.0


def test_shared_wrong_world_frame_needs_composition_check():
    camera, obj, hand = _transforms()
    wrong_frame = _pose((100.0, 0.0, 0.0))
    result = diagnostics.relative_pose_metrics(
        camera, obj, hand, wrong_frame @ camera @ obj, wrong_frame @ camera @ hand
    )

    assert result["relative_transform_max_abs_error"] == pytest.approx(0.0, abs=1e-12)
    assert result["origin_distance_invariance_error_m"] == pytest.approx(0.0, abs=1e-12)
    assert result["camera_composition_max_abs_error"] == pytest.approx(100.0)


def test_mesh_depth_bounds_apply_rotation_and_translation():
    vertices = np.array([[-2, 1, 3], [4, -1, -1], [0, 2, 2]], dtype=float)
    pose = _pose((0, 0, 10), np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))

    assert diagnostics.transformed_mesh_depth_bounds(vertices, pose) == pytest.approx((6, 12))


@pytest.mark.parametrize("vertices", [np.empty((0, 3)), np.zeros((2, 2)), [[np.nan, 0, 0]]])
def test_mesh_bounds_reject_invalid_vertices(vertices):
    with pytest.raises(ValueError):
        diagnostics.transformed_mesh_depth_bounds(vertices, np.eye(4))


def test_depth_stats_decode_inverse_depth_and_ignore_mask_exterior():
    depth = np.array([[21845, 21845, 0], [21845, 21845, 65535]], dtype=np.uint16)
    mask = np.array([[1, 1, 0], [1, 1, 0]], dtype=bool)

    result = diagnostics.depth_mask_stats(depth, mask, min_valid_pixels=4)

    assert result["valid_pixels"] == 4
    assert result["invalid_depth_pixels"] == 0
    assert result["source_observation_valid"] is True
    assert result["median_m"] == pytest.approx(2.0)


def test_invalid_depth_does_not_become_a_source_observation():
    depth = np.array([[0, 65535], [0, 65535]], dtype=np.uint16)
    result = diagnostics.depth_mask_stats(depth, np.ones((2, 2), dtype=bool), min_valid_pixels=1)

    assert result["valid_pixels"] == 0
    assert result["invalid_depth_pixels"] == 4
    assert result["source_observation_valid"] is False
    assert result["median_m"] is None


def test_observation_requires_minimum_valid_pixel_count():
    depth = np.array([[21845, 0], [65535, 0]], dtype=np.uint16)
    result = diagnostics.depth_mask_stats(depth, np.ones((2, 2), dtype=bool), min_valid_pixels=2)

    assert result["valid_pixels"] == 1
    assert result["invalid_depth_pixels"] == 3
    assert result["source_observation_valid"] is False


def test_empty_mask_does_not_invent_an_observation():
    result = diagnostics.depth_mask_stats(
        np.full((2, 2), 21845, dtype=np.uint16), np.zeros((2, 2), dtype=bool)
    )
    assert result["valid_pixels"] == 0
    assert result["invalid_depth_pixels"] == 0
    assert result["median_m"] is None
    assert result["source_observation_valid"] is False


@pytest.mark.parametrize(
    ("depth", "mask"),
    [
        (np.ones((2, 2), dtype=np.uint8), np.ones((2, 2))),
        (np.ones((2, 2), dtype=np.uint16), np.ones((3, 2))),
        (np.ones(4, dtype=np.uint16), np.ones(4)),
    ],
)
def test_depth_stats_reject_wrong_encoding_or_shape(depth, mask):
    with pytest.raises(ValueError):
        diagnostics.depth_mask_stats(depth, mask)


def _reconstruction(root: Path, frames: int = 3, *, valid_object: bool = True) -> Path:
    result = root / "result"
    result.mkdir(parents=True)
    (result / "mesh.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 1\nf 1 2 3\n")
    poses = np.tile(np.eye(4), (frames, 1, 1))
    obj = poses.copy()
    obj[:, 2, 3] = 2.0
    hand = poses.copy()
    hand[:, 2, 3] = 2.1
    np.savez(
        result / "result.npz",
        camera_to_world_transform=poses,
        camera_is_valid=np.ones(frames, dtype=bool),
        object_to_camera_transform=obj,
        object_to_world_transform=obj,
        object_is_valid=np.full(frames, valid_object, dtype=bool),
        hand_right_wrist_to_camera_transform=hand,
        hand_right_wrist_to_world_transform=hand,
        hand_right_is_valid=np.ones(frames, dtype=bool),
        hand_right_scale=np.ones(frames),
        hand_left_is_valid=np.zeros(frames, dtype=bool),
    )
    return root


def test_short_reconstruction_keeps_four_quarters_and_missing_observations(tmp_path):
    root = _reconstruction(tmp_path / "reconstruction")
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}

    result = diagnostics.diagnose_reconstruction(root)

    assert result["frame_count"] == 3
    assert len(result["quarter_summaries"]) == 4
    assert sum(q["frames"] for q in result["quarter_summaries"]) == 3
    assert all(
        not r["object_depth_observation"]["source_observation_valid"] for r in result["per_frame"]
    )
    assert all(r["mesh_front_minus_depth_median_m"] is None for r in result["per_frame"])
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == before


def test_invalid_object_pose_does_not_produce_hand_object_metrics(tmp_path):
    root = _reconstruction(tmp_path / "reconstruction", valid_object=False)
    result = diagnostics.diagnose_reconstruction(root)

    assert all(not r["object_pose_valid"] for r in result["per_frame"])
    assert all(not r["hands"] for r in result["per_frame"])


@pytest.mark.parametrize("mode", ["inside_source", "existing", "symlink"])
def test_cli_rejects_unsafe_output_without_changing_files(tmp_path, monkeypatch, mode):
    source = tmp_path / "input"
    source.mkdir()
    sentinel = tmp_path / "existing.json"
    sentinel.write_text("preserve me")
    if mode == "inside_source":
        destination = source / "new.json"
    elif mode == "existing":
        destination = sentinel
    else:
        destination = tmp_path / "link.json"
        destination.symlink_to(sentinel)
    monkeypatch.setattr(
        "sys.argv",
        ["diagnostics", "--reconstruction-root", str(source), "--output", str(destination)],
    )

    with pytest.raises(SystemExit) as error:
        diagnostics.main()

    assert error.value.code == 2
    assert sentinel.read_text() == "preserve me"
    assert not (source / "new.json").exists()


def test_cli_writes_new_external_receipt_and_refuses_overwrite(tmp_path, monkeypatch):
    source = _reconstruction(tmp_path / "input")
    destination = tmp_path / "receipts" / "new.json"
    monkeypatch.setattr(
        "sys.argv",
        ["diagnostics", "--reconstruction-root", str(source), "--output", str(destination)],
    )

    diagnostics.main()
    written = destination.read_bytes()
    with pytest.raises(SystemExit):
        diagnostics.main()

    assert destination.read_bytes() == written
