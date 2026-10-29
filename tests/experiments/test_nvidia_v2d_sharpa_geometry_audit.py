import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("rtree")

from experiments.nvidia_v2d.sharpa_geometry_audit import (  # noqa: E402
    audit_retarget,
    build_training_gate,
    sampled_surface_distances,
    validate_poses,
)


def test_rigid_object_transform_preserves_metric_distance_and_inside_sign():
    mesh = trimesh.creation.box(extents=[2, 4, 6])
    pose = np.eye(4)
    pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    pose[:3, 3] = [1.2, -0.4, 0.7]
    local = np.array([[2, 0, 0], [0.9, 0, 0], [1, 0, 0]], dtype=float)
    points = local @ pose[:3, :3].T + pose[:3, 3]
    before = mesh.vertices.copy()
    distances, penetration = sampled_surface_distances(mesh, points, pose)
    np.testing.assert_allclose(distances, [1, 0.1, 0], atol=1e-12)
    np.testing.assert_allclose(penetration, [0, 0.1, 0], atol=1e-12)
    np.testing.assert_array_equal(mesh.vertices, before)


def test_open_mesh_has_distance_but_no_penetration_inference():
    mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [2, 0, 0], [0, 2, 0]], faces=[[0, 1, 2]], process=False
    )
    distances, penetration = sampled_surface_distances(mesh, np.array([[0.1, 0.1, -0.02]]), np.eye(4))
    np.testing.assert_allclose(distances, [0.02])
    assert penetration is None


@pytest.mark.parametrize("case", ["scale", "reflection", "nan", "bottom"])
def test_invalid_coordinate_contract_rejected(case):
    pose = np.eye(4)
    if case == "scale":
        pose[0, 0] = 1.3
    elif case == "reflection":
        pose[0, 0] = -1
    elif case == "nan":
        pose[0, 3] = np.nan
    else:
        pose[3, 3] = 0
    with pytest.raises(ValueError):
        validate_poses(pose[None, None])


def make_run(tmp_path: Path, *, all_invalid: bool = False) -> Path:
    run = tmp_path / "retarget"
    run.mkdir()
    mesh_path = tmp_path / "object.ply"
    trimesh.creation.box(extents=[2, 2, 2]).export(mesh_path)
    time = np.arange(5) / 20
    poses = np.tile(np.eye(4), (5, 2, 1, 1))
    poses[:, 1, 0, 3] = 10
    codes = np.array([1, 2, 3, 4, 0], dtype=np.int16)
    valid = np.array([True, True, True, False, False]) & (not all_invalid)
    samples = np.tile(np.array([[1.1, 0, 0], [0.9, 0, 0]]), (5, 1, 1))
    samples[4] = np.nan
    preview = {
        "backend": "unit-fixture",
        "timestamps_s": time.tolist(),
        "object_names": ["near", "far"],
        "object_mesh_paths": [str(mesh_path), str(mesh_path)],
        "object_poses_world": poses.tolist(),
    }
    arrays = {"timestamps_s": time}
    for side in ("left", "right"):
        arrays.update(
            {
                f"{side}_world_surface_samples_m": samples,
                f"{side}_valid": valid,
                f"{side}_source_code": codes,
            }
        )
        preview[f"valid_{side}"] = valid.tolist()
        preview[f"source_code_{side}"] = codes.tolist()
    np.savez_compressed(run / "diagnostics.npz", **arrays)
    (run / "preview.json").write_text(json.dumps(preview))
    return run


def test_audit_keeps_quality_and_multiple_objects_and_preserves_sources(tmp_path):
    run = make_run(tmp_path)
    paths = [run / "diagnostics.npz", run / "preview.json", tmp_path / "object.ply"]
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    output = tmp_path / "audit"
    report = audit_retarget(run, output)
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths] == before
    assert report["source_unchanged"] is True
    assert report["physical_rollout"] is report["contact_truth"] is False
    left = report["hands"]["left"]
    assert left["excluded_summary_frames"] == [2, 3, 4]
    assert left["nonfinite_frames"] == [4]
    assert left["objects"][0]["sampled_penetration"]["inside_sample_fraction"] == 0.5
    assert left["objects"][1]["sampled_penetration"]["inside_sample_fraction"] == 0
    assert left["objects"][1]["sample_distance"]["min_m"] == pytest.approx(7.9)
    with np.load(output / "surface-audit.npz", allow_pickle=False) as data:
        assert data["left_object_surface_distance_m"].shape == (5, 2, 2)
        assert np.isnan(data["left_object_surface_distance_m"][4]).all()
        assert np.isfinite(data["left_object_surface_distance_m"][3]).all()
    with pytest.raises(FileExistsError):
        audit_retarget(run, output)
    with pytest.raises(ValueError, match="已有重定向"):
        audit_retarget(run, run / "nested")


def test_all_invalid_frames_do_not_become_zero_distance_success(tmp_path):
    report = audit_retarget(make_run(tmp_path, all_invalid=True), tmp_path / "audit")
    stats = report["hands"]["left"]["frame_minimum_sample_distance"]
    assert stats["count"] == 0
    assert stats["min_m"] is None
    assert report["hands"]["left"]["objects"][0]["sampled_penetration"]["max_depth_m"] is None


def test_timestamp_and_quality_disagreement_is_rejected_before_output(tmp_path):
    run = make_run(tmp_path)
    preview = json.loads((run / "preview.json").read_text())
    preview["timestamps_s"][2] += 0.001
    (run / "preview.json").write_text(json.dumps(preview))
    with pytest.raises(ValueError, match="时间轴"):
        audit_retarget(run, tmp_path / "audit")
    assert not (tmp_path / "audit").exists()
    preview["timestamps_s"][2] -= 0.001
    preview["source_code_left"][0] = 4
    (run / "preview.json").write_text(json.dumps(preview))
    with pytest.raises(ValueError, match="质量标记"):
        audit_retarget(run, tmp_path / "audit")


def gate_fixture():
    return {
        "status": "succeeded",
        "source_unchanged": True,
        "frame_count": 249,
        "objects": [{"sampled_penetration_measurable": False}],
        "hands": {
            side: {"valid_frames": 249, "excluded_summary_frames": [], "nonfinite_frames": []}
            for side in ("left", "right")
        },
    }


def test_open_mesh_gate_allows_only_explicit_mechanism_validation():
    report = gate_fixture()
    default = build_training_gate(report, numerical_contract_passed=True)
    assert default["numerical_input_status"] == "passed"
    assert default["geometry_quality_status"] == "unknown"
    assert default["penetration_status"] == "unknown"
    assert default["training_ready"] is False
    mechanism = build_training_gate(
        report, numerical_contract_passed=True, purpose="mechanism_validation"
    )
    assert mechanism["training_ready"] is True
    assert mechanism["mechanism_validation"] is True
    assert mechanism["geometry_quality_accepted"] is False
    assert mechanism["policy_quality_claim_allowed"] is False


@pytest.mark.parametrize(
    "failure", ["upstream", "source_changed", "nonfinite", "rejected", "missing_hand"]
)
def test_mechanism_training_does_not_bypass_bad_inputs(failure):
    report = gate_fixture()
    if failure == "source_changed":
        report["source_unchanged"] = False
    elif failure == "nonfinite":
        report["hands"]["left"]["nonfinite_frames"] = [10]
    elif failure == "rejected":
        report["hands"]["left"]["excluded_summary_frames"] = [10]
    elif failure == "missing_hand":
        del report["hands"]["left"]
    gate = build_training_gate(
        report, numerical_contract_passed=failure != "upstream", purpose="mechanism_validation"
    )
    assert gate["numerical_input_status"] == "failed"
    assert gate["training_ready"] is False


def test_closed_sampled_mesh_does_not_establish_geometric_quality():
    report = gate_fixture()
    report["objects"][0]["sampled_penetration_measurable"] = True
    gate = build_training_gate(report, numerical_contract_passed=True, purpose="policy_quality")
    assert gate["penetration_status"] == "sampled_only"
    assert gate["geometry_quality_status"] == "unknown"
    assert gate["training_ready"] is False
    with pytest.raises(ValueError):
        build_training_gate(report, numerical_contract_passed=True, purpose="anything")
    with pytest.raises(ValueError):
        build_training_gate(report, numerical_contract_passed="false")
