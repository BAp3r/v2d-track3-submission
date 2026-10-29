import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from experiments.nvidia_v2d.depth_source_ab import CameraK
from experiments.nvidia_v2d.postprocess_occlusion import export_visible_support, visible_depth_support
from experiments.nvidia_v2d.postprocess_support_fit import (
    SupportRefitConfig,
    fit_supported_frame,
    shared_evaluation,
    validate_support_bundle,
)
from experiments.nvidia_v2d.postprocess_support_refit import prepare_inputs, run_support_refit
from experiments.nvidia_v2d.sharpa_bridge import digest


def test_support_rejects_reintroduced_ambiguity_and_wrong_membership():
    original = np.ones((2, 10, 10), bool)
    visible, ambiguous, membership = visible_depth_support(original)
    validate_support_bundle(original, visible, ambiguous, membership)
    visible[0, 0, 0] = True
    with pytest.raises(ValueError, match="污染"):
        validate_support_bundle(original, visible, ambiguous, membership)


def test_min_visible_fraction_does_not_accept_tiny_lid_even_with_enough_pixels():
    k = CameraK(width=100, height=100, fx=100, fy=100, cx=50, cy=50)
    original = np.ones((100, 100), bool)
    visible = np.zeros_like(original)
    visible[:5] = True
    result = fit_supported_frame(
        np.ones_like(original, float),
        np.ones_like(original, float),
        visible,
        original,
        k,
        np.array([0, 0, 1]),
        SupportRefitConfig(),
    )
    assert result["fit"] is None
    assert result["shared_pixels"] == 500
    assert "相对原目标可见支持不足" in result["rejection_reasons_zh"]


def test_shared_evaluation_uses_same_pixels_for_all_candidates():
    observed = np.ones((10, 10))
    original = np.ones((10, 10)) + 0.1
    visible = np.ones((10, 10)) + 0.05
    visible[:, :5] = np.nan
    original[:, :5] += 10  # Must not penalize one candidate on another pixel set.
    result = shared_evaluation(
        {"original": original, "visible": visible}, observed, np.ones((10, 10), bool)
    )
    assert result["shared_pixels"] == 50
    assert result["rmse_m"] == pytest.approx({"original": 0.1, "visible": 0.05})
    assert result["independent_ground_truth"] is False


def test_empty_common_evaluation_is_unknown_not_zero_error():
    result = shared_evaluation(
        {"missing": np.full((5, 5), np.nan)}, np.ones((5, 5)), np.ones((5, 5), bool)
    )
    assert result["shared_pixels"] == 0
    assert result["rmse_m"]["missing"] is None


@pytest.fixture
def source_scene(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    video = raw / "source.mp4"
    video.write_bytes(b"test source file; not decoded by this service")
    scene = tmp_path / "scene"
    scene.mkdir()
    k = {"width": 64, "height": 64, "fx": 100.0, "fy": 100.0, "cx": 32.0, "cy": 32.0}
    source = {
        "sequence_id": "test",
        "object_names": ["pot", "lid"],
        "fps": 20.0,
        "frame_count": 2,
        "source_video": str(video),
        "source_video_sha256": digest(video),
    }
    objects, hashes = [], {}
    for i, name in enumerate(source["object_names"]):
        job = tmp_path / name
        recon = job / "reconstruction"
        quality = job / "hand-quality/quality_result.npz"
        quality.parent.mkdir(parents=True)
        for directory in ["depth", "intrinsics", "masks/1", "result"]:
            (recon / directory).mkdir(parents=True)
        pose = np.repeat(np.eye(4)[None], 2, axis=0)
        pose[:, 2, 3] = 1
        camera = np.repeat(np.eye(4)[None], 2, axis=0)
        camera[:, :3, 3] = [2, 3, 4]
        np.savez(
            quality,
            object_to_camera_transform=pose,
            object_is_valid=np.ones(2, bool),
            camera_to_world_transform=camera,
            camera_is_valid=np.ones(2, bool),
            camera_intrinsics=np.array([100.0, 100.0, 32.0, 32.0]),
        )
        mesh = recon / "result/mesh.obj"
        mesh.write_text("v -0.3 -0.3 0\nv 0.3 -0.3 0\nv 0.3 0.3 0\nv -0.3 0.3 0\nf 1 2 3\nf 1 3 4\n")
        stable = recon / "intrinsics_stable.json"
        stable.write_text(json.dumps(k))
        for p in [quality, mesh, stable]:
            hashes[str(p)] = digest(p)
        for frame in range(2):
            (recon / "intrinsics" / f"{frame:06d}.json").write_text(json.dumps(k))
            Image.fromarray(np.full((64, 64), round(65535 / 2.1), np.uint16)).save(
                recon / "depth" / f"{frame:06d}.png"
            )
            mask = np.zeros((64, 64), np.uint8)
            if frame == 0:
                mask[14:51, 14:36] = 255 if i == 0 else 0
                if i == 1:
                    mask[14:51, 29:51] = 255
            else:
                mask[14:51, 14:51] = 255
            Image.fromarray(mask).save(recon / "masks/1" / f"{frame:06d}.png")
        objects.append(
            {
                "name": name,
                "job": str(job),
                "source_video_sha256": digest(video),
                "fps": 20.0,
                "frame_count": 2,
            }
        )
    spec = {"source": source, "objects": objects, "hand_reference_object": "pot"}
    (scene / "input.json").write_text(json.dumps(spec))
    manifest = {
        "status": "assembled_for_inspection",
        "proof": {"shared_coordinates_proven": True},
        "source": source,
        "objects": source["object_names"],
        "input_sha256": hashes,
    }
    (scene / "manifest.json").write_text(json.dumps(manifest))
    support = tmp_path / "support"
    export_visible_support(scene / "manifest.json", support)
    return scene / "manifest.json", support / "manifest.json"


def test_full_refit_preserves_time_rejections_fixed_scale_and_shared_world(source_scene, tmp_path):
    scene, support = source_scene
    baseline = {str(p): digest(p) for p in tmp_path.rglob("*") if p.is_file()}
    result = run_support_refit(
        scene, support, tmp_path / "output", SupportRefitConfig(workers=1, stride=1)
    )
    assert result["status"] == "completed_candidates"
    assert result["paired_candidate_frames"] == {"pot": 1, "lid": 1}
    assert result["geometry_quality_accepted"] is False
    assert result["full_training_accepted"] is False
    with np.load(tmp_path / "output/refit.npz") as data:
        assert data["timestamps_s"].tolist() == [0, 0.05]
        for branch in ["original_mask", "visible_support"]:
            valid = data[branch + "_candidate_valid"]
            assert valid.tolist() == [[True, True], [False, False]]
            assert np.isnan(data[branch + "_object_to_world_transform"][1]).all()
            assert np.allclose(data[branch + "_mesh_scale"], 1.1, atol=1e-3)
            actual = data[branch + "_object_to_world_transform"][0]
            expected = (
                data["camera_to_world_transform"][0] @ data[branch + "_object_to_camera_transform"][0]
            )
            assert np.allclose(actual, expected)
            assert np.allclose(actual[:, 2, 3], 5.1, atol=1e-3)
    assert all(digest(Path(p)) == sha for p, sha in baseline.items())


def test_mutated_source_bundle_rejected_before_creating_output(source_scene, tmp_path):
    scene, support = source_scene
    (tmp_path / "pot/hand-quality/quality_result.npz").write_bytes(b"altered")
    with pytest.raises(ValueError, match="输入改变"):
        run_support_refit(scene, support, tmp_path / "output", SupportRefitConfig())
    assert not (tmp_path / "output").exists()


def test_missing_frame_or_wrong_identity_rejected(source_scene):
    scene, support = source_scene
    value = json.loads(support.read_text())
    value["frames"] = value["frames"][:1]
    support.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="不完整"):
        prepare_inputs(scene, support, SupportRefitConfig())


def test_failure_manifest_keeps_corrupt_support_without_publishing(source_scene, tmp_path):
    scene, support = source_scene
    (support.parent / "frames/000001.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="凭据不匹配"):
        run_support_refit(scene, support, tmp_path / "output", SupportRefitConfig(workers=1))
    result = json.loads((tmp_path / "output/manifest.json").read_text())
    assert result["status"] == "failed"
    assert result["failure_reason"]
    assert result["finished_at"]
    assert not (tmp_path / "output/refit.npz").exists()


def test_reusing_output_or_writing_inside_source_is_rejected(source_scene, tmp_path):
    scene, support = source_scene
    for output in [
        scene.parent / "new",
        support.parent / "new",
        tmp_path / "pot/new",
        tmp_path / "raw/new",
        tmp_path / "scene",
    ]:
        with pytest.raises((ValueError, FileExistsError)):
            run_support_refit(scene, support, output, SupportRefitConfig())


@pytest.mark.parametrize(
    "input_name", ["hand-quality/quality_result.npz", "reconstruction/result/mesh.obj"]
)
def test_source_manifest_must_bind_every_used_asset(source_scene, tmp_path, input_name):
    scene, support = source_scene
    value = json.loads(scene.read_text())
    del value["input_sha256"][str(tmp_path / "pot" / input_name)]
    scene.write_text(json.dumps(value))
    value = json.loads(support.read_text())
    value["source_inputs_sha256"][str(scene)] = digest(scene)
    support.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="凭据缺少"):
        prepare_inputs(scene, support, SupportRefitConfig())


@pytest.mark.parametrize(
    "key",
    ["object_to_camera_transform", "camera_to_world_transform", "object_is_valid", "camera_is_valid"],
)
def test_validity_and_poses_must_share_exact_time_axis(source_scene, tmp_path, key):
    scene, support = source_scene
    asset = tmp_path / "pot/hand-quality/quality_result.npz"
    with np.load(asset) as source:
        arrays = {k: source[k] for k in source.files}
    arrays[key] = arrays[key][:1]
    np.savez(asset, **arrays)
    value = json.loads(scene.read_text())
    value["input_sha256"][str(asset)] = digest(asset)
    scene.write_text(json.dumps(value))
    value = json.loads(support.read_text())
    value["source_inputs_sha256"][str(scene)] = digest(scene)
    support.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="完整时间轴"):
        prepare_inputs(scene, support, SupportRefitConfig())
