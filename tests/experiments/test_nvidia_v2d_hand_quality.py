"""用可反证情形验证手部来源、支持下界及拒绝后的保守补齐。"""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from experiments.nvidia_v2d.hand_quality import (
    SOURCE_CODES,
    HandQualityPolicy,
    original_source,
    process_track,
    reject_observation,
    slerp_axisangle,
    support_bounds,
)
from experiments.nvidia_v2d.hand_quality_io import export_quality


def _record(frame=0, z=1.0):
    return {
        "track_id": 2,
        "frame_idx": frame,
        "is_right": True,
        "image_size": [20, 20],
        "bbox": {"x0": 0, "y0": 0, "x1": 20, "y1": 20},
        "mano": {"betas": [0.0] * 10, "global_orient": [0.0] * 3, "hand_pose": [0.0] * 45},
        "camera": {
            "cam_t": [0.0, 0.0, z],
            "intrinsics": {"fx": 10.0, "fy": 10.0, "cx": 10.0, "cy": 10.0},
        },
        "diagnostics": {"scale": 1.0, "n_pixels": 400},
        "hand_scale": 1.0,
    }


def _support(n=400):
    return support_bounds(
        np.full((20, 20), 32767, dtype=np.uint16), np.ones((20, 20)), np.zeros((20, 20)), n
    )


def test_support_bound_does_not_confuse_region_with_rendered_intersection():
    depth = np.full((20, 20), 32767, dtype=np.uint16)
    depth.flat[:200] = 65535
    result = support_bounds(depth, np.ones((20, 20)), np.zeros((20, 20)), 250)
    assert result["region_positive_depth_pixels"] == 200
    assert result["positive_support_lower_bound"] == 50
    assert result["positive_support_upper_bound"] == 200
    # 无效像素最坏全在渲染交集；400像素的大掩膜不能证明250像素有效支持。
    assert "positive_depth_support_not_guaranteed" in reject_observation(
        _record(), result, HandQualityPolicy()
    )


def test_support_lower_bound_is_valid_for_every_possible_hidden_intersection():
    depth = np.array([[0, 65535, 100, 200, 300]], dtype=np.uint16)
    result = support_bounds(depth, np.ones_like(depth), np.zeros_like(depth), 3)
    from itertools import combinations

    actual_counts = [
        sum(depth[0, index] not in (0, 65535) for index in indices)
        for indices in combinations(range(5), 3)
    ]
    assert min(actual_counts) == result["positive_support_lower_bound"] == 1
    assert max(actual_counts) == result["positive_support_upper_bound"] == 3


def test_object_pixels_are_removed_before_support_evidence():
    obj = np.zeros((20, 20), dtype=bool)
    obj[:10] = True
    result = support_bounds(np.full((20, 20), 32767, dtype=np.uint16), np.ones((20, 20)), obj, 201)
    assert result["region_pixels"] == 200
    assert result["counts_consistent"] is False
    assert "geometry_support_count_inconsistent" in reject_observation(
        _record(), result, HandQualityPolicy()
    )


@pytest.mark.parametrize("z", [0, -1, np.nan, np.inf])
def test_invalid_camera_depth_cannot_be_observation(z):
    assert reject_observation(_record(z=z), _support(), HandQualityPolicy())


@pytest.mark.parametrize("scale", [0.2, 5.0, np.nan, -1])
def test_official_fit_domain_is_strict_and_not_a_ground_truth_distance(scale):
    rec = _record(z=1000)
    rec["diagnostics"]["scale"] = scale
    assert "alignment_ratio_outside_fit_domain" in reject_observation(
        rec, _support(), HandQualityPolicy()
    )
    # 没有任意米制上限：仅凭1000米本身不作几何真值判断。
    assert not reject_observation(_record(z=1000), _support(), HandQualityPolicy())


def test_finite_extreme_4422_fallback_is_rejected_with_explicit_evidence():
    rec = _record(4422, z=180.243744)
    rec["diagnostics"].update(n_pixels=0, scale=1.0)
    reasons = reject_observation(rec, _support(0), HandQualityPolicy())
    assert reasons == ["geometry_support_insufficient", "positive_depth_support_not_guaranteed"]


def test_upstream_bool_does_not_hide_constant_extrapolation():
    observed = {3: _record(3), 7: _record(7)}
    filled = {
        0: {"interpolated": True, "interpolation": {"prev": 3, "next": 3}},
        5: {"interpolated": True, "interpolation": {"prev": 3, "next": 7}},
        6: {"interpolated": False},
    }
    assert original_source(0, observed, filled) == "extrapolated"
    assert original_source(5, observed, filled) == "interpolated"
    assert original_source(6, observed, filled) == "rejected"
    assert original_source(3, observed, filled) == "observed"
    assert original_source(2, observed, filled) == "missing"


def test_rejected_observation_can_only_return_as_labeled_short_interpolation():
    observed = {i: _record(i) for i in range(5)}
    observed[2]["camera"]["cam_t"][2] = 180.0
    supports = {i: _support() for i in range(5)}
    supports[2] = _support(0)
    original = copy.deepcopy(observed)
    accepted, rows = process_track(observed, {}, supports, 5, HandQualityPolicy())
    assert observed == original
    assert accepted[2]["camera"]["cam_t"][2] == 1.0
    assert rows[2]["source"] == "interpolated"
    assert rows[2]["original_source"] == "observed"
    assert rows[2]["observation_status"] == "rejected"
    assert rows[2]["is_observed"] is False
    assert "geometry_support_insufficient" in rows[2]["rejection_reasons"]


def test_temporal_isolated_ratio_and_smooth_large_trajectory_are_distinguished():
    supports = {i: _support() for i in range(3)}
    accepted, rows = process_track(
        {0: _record(0), 1: _record(1, 180), 2: _record(2)}, {}, supports, 3, HandQualityPolicy()
    )
    assert "isolated_temporal_depth_ratio_outlier" in rows[1]["rejection_reasons"]
    assert accepted[1]["camera"]["cam_t"][2] == 1
    accepted, rows = process_track(
        {i: _record(i, 100 + i) for i in range(3)}, {}, supports, 3, HandQualityPolicy()
    )
    assert all(row["source"] == "observed" for row in rows)


@pytest.mark.parametrize("end, fps, visible", [(8, 30, True), (6, 10, True), (3, 30, False)])
def test_long_slow_or_invisible_gaps_stay_missing(end, fps, visible):
    supports = {i: _support() for i in range(end + 1)}
    if not visible:
        supports[1] = None
    accepted, rows = process_track(
        {0: _record(0), end: _record(end)}, {}, supports, end + 1, HandQualityPolicy(fps=fps)
    )
    assert 1 not in accepted
    assert not rows[1]["valid"]


def test_no_new_extrapolation_and_old_extrapolation_not_reused():
    observed = {2: _record(2), 4: _record(4)}
    old = {0: {"interpolated": True, "interpolation": {"prev": 2, "next": 2}}}
    accepted, rows = process_track(
        observed, old, {i: _support() for i in range(7)}, 7, HandQualityPolicy()
    )
    assert sorted(accepted) == [2, 3, 4]
    assert rows[0]["original_source"] == "extrapolated" and not rows[0]["valid"]
    assert rows[6]["source"] == "missing"


def test_invalid_camera_blocks_endpoints_and_gap():
    accepted, rows = process_track(
        {0: _record(0), 3: _record(3)},
        {},
        {i: _support() for i in range(4)},
        4,
        HandQualityPolicy(),
        np.array([1, 0, 1, 1], dtype=bool),
    )
    assert sorted(accepted) == [0, 3]
    assert not rows[2]["valid"]


def test_interpolation_rotation_uses_shortest_path_at_pi_boundary():
    result = slerp_axisangle(
        np.array([0, 0, np.deg2rad(179)]), np.array([0, 0, np.deg2rad(-179)]), 0.5
    )
    assert abs(result[2]) == pytest.approx(np.pi)


def _fixture(root: Path):
    (root / "result").mkdir(parents=True)
    n = 5
    arrays = {
        "camera_to_world_transform": np.tile(np.eye(4), (n, 1, 1)),
        "camera_is_valid": np.ones(n, dtype=bool),
        "object_is_valid": np.ones(n, dtype=bool),
        "object_to_camera_transform": np.tile(np.eye(4), (n, 1, 1)),
        "hand_right_betas": np.arange(10, dtype=np.float32),
    }
    arrays["camera_to_world_transform"][:, 0, 3] = np.arange(n)
    np.savez_compressed(root / "result/result.npz", **arrays)
    (root / "result/manifest.json").write_text(
        json.dumps(
            {
                "n_frames": n,
                "sources": {
                    "right_hand_dir": str(root / "hamer_aligned_filled/2"),
                    "hand_pose_source": "hamer",
                },
            }
        )
    )
    for directory in ("depth", "masks/1", "masks/2", "hamer_aligned/2", "hamer_aligned_filled/2"):
        (root / directory).mkdir(parents=True)
    for frame in range(n):
        Image.fromarray(np.full((20, 20), 32767, dtype=np.uint16)).save(
            root / "depth" / f"{frame:06}.png"
        )
        for tid in (1, 2):
            Image.fromarray(np.full((20, 20), 255 if tid == 2 else 0, dtype=np.uint8)).save(
                root / "masks" / str(tid) / f"{frame:06}.png"
            )
        if frame in (1, 3):
            rec = _record(frame)
            (root / "hamer_aligned/2" / f"{frame:06}.json").write_text(json.dumps(rec))
            (root / "hamer_aligned_filled/2" / f"{frame:06}.json").write_text(json.dumps(rec))
    return arrays


def test_export_preserves_inputs_and_synchronizes_every_hand_array(tmp_path):
    root, output = tmp_path / "input", tmp_path / "quality"
    original = _fixture(root)
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
    receipt = export_quality(
        root, output, HandQualityPolicy(max_interpolation_camera_translation_over_depth=2)
    )
    assert receipt["input_files_unchanged"] and not receipt["geometry_quality_accepted"]
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == v for p, v in hashes.items())
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["hands"]["right"][2]["source"] == "interpolated"
    with np.load(output / "quality_result.npz", allow_pickle=False) as data:
        assert data["hand_right_is_valid"].tolist() == [False, True, True, True, False]
        assert data["hand_right_is_observed"].tolist() == [False, True, False, True, False]
        assert data["hand_right_source_code"][2] == SOURCE_CODES["interpolated"]
        assert data["hand_right_wrist_to_world_transform"][2, 0, 3] == 2
        assert data["hand_right_wrist_trans_in_camera"][2, 2] == 1
        assert np.array_equal(
            data["camera_to_world_transform"], original["camera_to_world_transform"]
        )
        assert np.array_equal(data["object_is_valid"], original["object_is_valid"])
        assert np.array_equal(data["hand_right_wrist_to_world_transform"][0], np.eye(4))
        assert not data["hand_left_is_valid"].any()
        rec = json.loads((output / "hands/right/000002.json").read_text())
        assert rec["mano"]["betas"] == data["hand_right_betas"].tolist()
    with pytest.raises(FileExistsError):
        export_quality(root, output)
    with pytest.raises(ValueError):
        export_quality(root, root / "unsafe")


def test_manifest_cannot_escape_reconstruction(tmp_path):
    root = tmp_path / "input"
    _fixture(root)
    manifest = root / "result/manifest.json"
    value = json.loads(manifest.read_text())
    value["sources"]["right_hand_dir"] = str(tmp_path / "outside/2")
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="本次重建"):
        export_quality(root, tmp_path / "quality")


def test_camera_jump_blocks_interpolation_but_does_not_reject_direct_observations():
    cameras = np.tile(np.eye(4), (4, 1, 1))
    cameras[2:, 0, 3] = 2
    accepted, rows = process_track(
        {0: _record(0), 3: _record(3)},
        {},
        {i: _support() for i in range(4)},
        4,
        HandQualityPolicy(),
        camera_to_world=cameras,
    )
    assert sorted(accepted) == [0, 3]
    assert "interpolation_camera_discontinuity" in rows[1]["rejection_reasons"]
    assert rows[0]["is_observed"] and rows[3]["is_observed"]


def test_camera_motion_confounds_temporal_depth_rejection():
    cameras = np.tile(np.eye(4), (3, 1, 1))
    cameras[1, 2, 3] = 180
    accepted, rows = process_track(
        {0: _record(0), 1: _record(1, 180), 2: _record(2)},
        {},
        {i: _support() for i in range(3)},
        3,
        HandQualityPolicy(),
        camera_to_world=cameras,
    )
    assert rows[1]["source"] == "observed"
    assert rows[1]["warnings"] == ["temporal_depth_change_camera_motion_confounded"]
    assert accepted[1]["camera"]["cam_t"][2] == 180


def test_unknown_visibility_anywhere_blocks_entire_gap():
    supports = {i: _support() for i in range(4)}
    supports[1] = None
    accepted, rows = process_track(
        {0: _record(0), 3: _record(3)}, {}, supports, 4, HandQualityPolicy()
    )
    assert sorted(accepted) == [0, 3]
    assert "interpolation_visible_depth_support_missing" in rows[2]["rejection_reasons"]


def test_unknown_hand_array_cannot_silently_keep_unfiltered_values(tmp_path):
    root = tmp_path / "input"
    arrays = _fixture(root)
    arrays["hand_right_future_confidence"] = np.ones(5)
    np.savez_compressed(root / "result/result.npz", **arrays)
    with pytest.raises(ValueError, match="未知手部数组"):
        export_quality(root, tmp_path / "quality")


def test_export_camera_invalid_does_not_become_endpoint(tmp_path):
    root = tmp_path / "input"
    arrays = _fixture(root)
    arrays["camera_is_valid"][1] = False
    np.savez_compressed(root / "result/result.npz", **arrays)
    export_quality(root, tmp_path / "quality")
    value = json.loads((tmp_path / "quality/provenance.json").read_text())
    assert value["hands"]["right"][1]["source"] == "rejected"
    assert not value["hands"]["right"][2]["valid"]
