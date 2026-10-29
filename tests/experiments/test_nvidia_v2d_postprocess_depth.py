import numpy as np
import pytest

from experiments.nvidia_v2d.postprocess_depth import (
    DepthRefitConfig,
    fit_gate,
    fit_scale_translation,
    fixed_scale_refit,
)


def test_known_fixed_rotation_scale_translation_is_recovered():
    p = np.random.default_rng(42).normal(size=(400, 3)) * 0.05
    q = p * 1.3 + [0.1, 0.2, 0.8]
    fit = fit_scale_translation(p, q)
    assert fit["scale"] == pytest.approx(1.3)
    assert fit["translation_camera_m"] == pytest.approx([0.1, 0.2, 0.8])
    assert fit["rmse_m"] < 1e-10


def test_global_scale_refit_matches_same_correspondences():
    p = np.random.default_rng(42).normal(size=(400, 3)) * 0.05
    q = p * 1.3 + [0.1, 0.2, 0.8]
    fit = fit_scale_translation(p, q)
    result = fixed_scale_refit(fit, 1.3)
    assert result["translation_camera_m"] == pytest.approx([0.1, 0.2, 0.8])
    assert result["rmse_m"] < 1e-8


def test_degenerate_cloud_is_not_published_as_valid_alignment():
    with pytest.raises(ValueError, match="退化"):
        fit_scale_translation(np.ones((20, 3)), np.ones((20, 3)))


def test_large_residual_and_shift_are_rejected():
    fit = {"scale": 1.0, "translation_camera_m": [1, 1, 1], "rmse_m": 0.1}
    reasons = fit_gate(fit, DepthRefitConfig(), np.zeros(3))
    assert len(reasons) == 2


def test_nonfinite_correspondence_is_rejected():
    p = np.ones((20, 3))
    p[0, 0] = np.nan
    with pytest.raises(ValueError):
        fit_scale_translation(p, np.ones((20, 3)))


def test_full_axis_refit_preserves_rejected_frame_and_actual_time(tmp_path):
    import json

    from PIL import Image

    from experiments.nvidia_v2d.postprocess_depth import run_refit
    from experiments.nvidia_v2d.sharpa_bridge import digest

    recon = tmp_path / "reconstruction"
    sensor = tmp_path / "sensor"
    raw = tmp_path / "raw"
    raw.mkdir()
    for directory in [
        recon / "depth",
        recon / "intrinsics",
        recon / "masks/1",
        recon / "poses_smoothed",
        sensor / "depth",
    ]:
        directory.mkdir(parents=True)
    k = {"width": 64, "height": 64, "fx": 100.0, "fy": 100.0, "cx": 32.0, "cy": 32.0}
    (sensor / "camera_intrinsics.json").write_text(json.dumps(k))
    (sensor / "calibration.json").write_text("{}")
    (recon / "mesh_pretransformed.obj").write_text(
        "v -0.2 -0.2 0\nv 0.2 -0.2 0\nv 0.2 0.2 0\nv -0.2 0.2 0\nf 1 2 3\nf 1 3 4\n"
    )
    source_rows = []
    for i in range(2):
        for directory, z in [(recon, 1.1), (sensor, 1.2)]:
            encoded = np.full((64, 64), round(65535 / (1 + z)), np.uint16)
            Image.fromarray(encoded).save(directory / "depth" / f"{i:06d}.png")
        (recon / "intrinsics" / f"{i:06d}.json").write_text(json.dumps(k))
        (recon / "poses_smoothed" / f"{i:06d}.json").write_text(
            json.dumps({"rotation": [1, 0, 0, 0], "translation": [0, 0, 1], "scale": [1, 1, 1]})
        )
        mask = np.zeros((64, 64), np.uint8)
        if i == 0:
            mask[14:51, 14:51] = 255
        Image.fromarray(mask).save(recon / "masks/1" / f"{i:06d}.png")
        source_rows.append(
            {
                "frame_index": i,
                "timestamp_ns": 1000 + i * 35_123_456,
                "sha256": digest(sensor / "depth" / f"{i:06d}.png"),
            }
        )
    (sensor / "manifest.json").write_text(
        json.dumps(
            {
                "recording": str(raw),
                "status": "succeeded",
                "frame_index_basis": "authoritative_raw_container_zero_based",
                "frames": source_rows,
                "source_files_before": [],
            }
        )
    )
    result = run_refit(recon, sensor, tmp_path / "output", DepthRefitConfig(stride=1, workers=1))
    assert result["initial_paired_eligible_frames"] == 1
    assert result["final_paired_candidate_frames"] == 1
    a = np.load(tmp_path / "output/refit.npz")
    assert a["timestamps_ns"].tolist() == [1000, 35_124_456]
    assert a["sensor_candidate_valid"].tolist() == [True, False]
    assert np.isnan(a["sensor_object_to_camera_transform"][1]).all()
    assert float(a["sensor_mesh_scale"]) == pytest.approx(1.2, abs=1e-3)
    assert result["full_training_accepted"] is False
