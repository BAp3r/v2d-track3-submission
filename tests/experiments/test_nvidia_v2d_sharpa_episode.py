from pathlib import Path

import numpy as np
import pytest

from experiments.nvidia_v2d.sharpa_episode import (
    EpisodeSource,
    audit_bundle,
    require_new_output,
    scale_about_centroid,
    transform_points,
)


def source(n: int = 10) -> EpisodeSource:
    return EpisodeSource(
        sequence_id="track_3-public-episode-000044",
        object_names=["white_pot"],
        fps=20,
        frame_count=n,
        source_video=Path("/input/video.mp4"),
        source_video_sha256="a" * 64,
    )


def arrays(n: int = 10) -> dict[str, np.ndarray]:
    result = {
        "camera_to_world_transform": np.repeat(np.eye(4)[None], n, axis=0),
        "object_to_world_transform": np.repeat(np.eye(4)[None], n, axis=0),
        "camera_is_valid": np.ones(n, dtype=bool),
        "object_is_valid": np.ones(n, dtype=bool),
        "object_scale": np.array(1.0),
    }
    for side in ("left", "right"):
        prefix = "hand_" + side
        result.update(
            {
                prefix + "_is_valid": np.ones(n, dtype=bool),
                prefix + "_is_observed": np.ones(n, dtype=bool),
                prefix + "_source_code": np.ones(n, dtype=np.uint8),
                prefix + "_betas": np.zeros(10),
                prefix + "_finger_pose": np.zeros((n, 15, 3)),
                prefix + "_wrist_orient_in_camera": np.zeros((n, 3)),
                prefix + "_wrist_trans_in_camera": np.tile([0, 0, 1.0], (n, 1)),
                prefix + "_scale": np.ones(n),
            }
        )
    return result


def test_twenty_fps_is_preserved_and_does_not_drop_last_frame():
    audit = audit_bundle(source(), arrays())
    assert audit.input_contract_passed
    assert audit.timestamps_s == pytest.approx([i / 20 for i in range(10)])
    assert audit.last_frame_time_s == pytest.approx(0.45)
    assert audit.video_duration_s == pytest.approx(0.5)
    assert audit.geometry_quality_accepted is False


def test_rejected_zero_hand_cannot_be_imported_as_a_real_pose():
    data = arrays()
    data["hand_left_is_valid"][3] = False
    data["hand_left_is_observed"][3] = False
    data["hand_left_source_code"][3] = 4
    data["hand_left_wrist_trans_in_camera"][3] = 0
    audit = audit_bundle(source(), data)
    assert not audit.input_contract_passed
    assert "left:unavailable_hand_frames" in audit.blockers
    assert audit.hands["left"].source_counts["rejected"] == 1


def test_only_bounded_two_sided_interpolation_is_eligible():
    data = arrays()
    for k in ("is_observed",):
        data[f"hand_left_{k}"][3:5] = False
    data["hand_left_source_code"][3:5] = 2
    audit = audit_bundle(source(), data)
    assert audit.input_contract_passed
    assert audit.hands["left"].source_counts["interpolated"] == 2
    data["hand_left_source_code"][0] = 2
    data["hand_left_is_observed"][0] = False
    assert "left:unsupported_interpolation" in audit_bundle(source(), data).blockers


def test_long_interpolation_and_extrapolation_block_complete_training():
    data = arrays()
    data["hand_right_source_code"][1:7] = 2
    data["hand_right_is_observed"][1:7] = False
    assert not audit_bundle(source(), data).input_contract_passed
    data = arrays()
    data["hand_right_source_code"][2] = 3
    data["hand_right_is_observed"][2] = False
    assert "right:extrapolated_frames" in audit_bundle(source(), data).blockers


def test_inconsistent_source_flags_and_nonrigid_camera_are_rejected():
    data = arrays()
    data["hand_left_source_code"][1] = 2
    data["camera_to_world_transform"][1, 0, 0] = 2
    result = audit_bundle(source(), data)
    assert "left:source_flags_disagree" in result.blockers
    assert "camera:nonrigid_transform" in result.blockers


def test_scale_geometry_about_mesh_centroid_not_about_wrist():
    vertices = np.array([[[0, 0, 0], [2, 0, 0]]], dtype=float)
    joints = np.array([[[0.5, 0, 0], [1.5, 0, 0]]], dtype=float)
    scaled_vertices, scaled_joints = scale_about_centroid(vertices, joints, np.array([2]))
    assert scaled_vertices[0, :, 0] == pytest.approx([-1, 3])
    assert scaled_joints[0, :, 0] == pytest.approx([0, 2])
    assert scaled_vertices.mean(axis=1) == pytest.approx(vertices.mean(axis=1))


def test_same_rigid_transform_preserves_hand_object_relative_geometry():
    pose = np.eye(4)[None]
    pose[0, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    pose[0, :3, 3] = [4, 5, 6]
    first = np.array([[[1, 0, 0]]], dtype=float)
    second = np.array([[[1, 1, 0]]], dtype=float)
    assert np.linalg.norm(transform_points(pose, first) - transform_points(pose, second)) == 1


def test_multi_object_bundle_is_not_silently_loaded_as_one_body():
    metadata = source().model_copy(update={"object_names": ["pot", "lid"]})
    result = audit_bundle(metadata, arrays())
    assert "multi_object_scene_not_assembled" in result.blockers


def test_output_cannot_overwrite_or_enter_source_tree(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(ValueError, match="输入"):
        require_new_output(raw / "derived", [raw])
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        require_new_output(existing, [raw])
    require_new_output(tmp_path / "new", [raw])


def test_shape_provenance_uses_only_accepted_observations(tmp_path: Path):
    import json

    from experiments.nvidia_v2d.sharpa_bridge import accepted_shape_provenance

    data = arrays(3)
    for side in ("left", "right"):
        directory = tmp_path / "reconstruction/hamer_aligned" / side
        directory.mkdir(parents=True)
        data[f"hand_{side}_source_code"][2] = 4
        data[f"hand_{side}_betas"][:] = 1
        for index, value in enumerate((0, 2, 99)):
            (directory / f"{index}.json").write_text(
                json.dumps(
                    {
                        "is_right": side == "right",
                        "frame_idx": index,
                        "mano": {"betas": [value] * 10},
                    }
                )
            )
    result = accepted_shape_provenance(tmp_path, data)
    assert result["left"]["shape_betas"] == [1] * 10
    assert result["right"]["accepted_frame_indices"] == [0, 1]
    data["hand_left_betas"][:] = 99
    with pytest.raises(ValueError, match="不是可信观测中值"):
        accepted_shape_provenance(tmp_path, data)


def test_mismatched_video_hash_stops_before_output_creation(tmp_path: Path):
    from experiments.nvidia_v2d.sharpa_bridge import export_episode

    job = tmp_path / "input/job"
    (job / "hand-quality").mkdir(parents=True)
    np.savez(job / "hand-quality/quality_result.npz", **arrays())
    video = tmp_path / "input/video.mp4"
    video.write_bytes(b"source-video")
    metadata = source().model_copy(update={"source_video": video})
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="源视频不匹配"):
        export_episode(metadata, job, tmp_path / "upstream", tmp_path / "mano", output)
    assert not output.exists()


def test_blocked_export_writes_audit_and_no_trainable_parquet(tmp_path: Path):
    import json

    from experiments.nvidia_v2d.sharpa_bridge import digest, export_episode

    job = tmp_path / "input/job"
    (job / "hand-quality").mkdir(parents=True)
    npz = job / "hand-quality/quality_result.npz"
    np.savez(npz, **arrays())
    (npz.parent / "manifest.json").write_text(
        json.dumps(
            {
                "artifacts": [{"path": "quality_result.npz", "sha256": digest(npz)}],
            }
        )
    )
    video = tmp_path / "input/video.mp4"
    video.write_bytes(b"source-video")
    metadata = source().model_copy(
        update={
            "source_video": video,
            "source_video_sha256": digest(video),
            "object_names": ["pot", "lid"],
        }
    )
    output = tmp_path / "output"
    result = export_episode(metadata, job, tmp_path / "upstream", tmp_path / "mano", output)
    assert result["status"] == "blocked"
    assert (output / "input-gate.json").is_file()
    assert (output / "manifest.json").is_file()
    assert not (output / "loaded.parquet").exists()
    assert digest(video) == metadata.source_video_sha256


def test_delivery_allows_labeled_inference_but_still_rejects_nonfinite_geometry():
    data = arrays()
    data['hand_left_source_code'][:] = 3
    data['hand_left_is_observed'][:] = False
    assert not audit_bundle(source(), data).input_contract_passed
    allowed = audit_bundle(source(), data, allow_inferred_hands=True)
    assert allowed.input_contract_passed
    assert allowed.hands['left'].source_counts['observed'] == 0
    assert allowed.geometry_quality_accepted is False
    data['object_to_world_transform'][0, 0, 0] = np.nan
    assert not audit_bundle(source(), data, allow_inferred_hands=True).input_contract_passed
