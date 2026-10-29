import hashlib
import json

import pytest

from experiments.nvidia_v2d.policy_view_data import PolicyStore, PolicyVariant, Scene


def scene():
    return {
        "timestamps_s": [0, 0.05],
        "hands": [{"side": "left", "points": [[[0, 0, 0], [1, 0, 0]]] * 2, "edges": [[0, 1]]}],
        "objects": [],
        "kind": "kinematic_reference",
    }


def test_scene_rejects_broken_time_axis_and_topology():
    assert Scene.model_validate(scene()).timestamps_s[-1] == 0.05
    for times in ([0, 0], [0], [0, float("nan")]):
        with pytest.raises(ValueError):
            Scene.model_validate({**scene(), "timestamps_s": times})
    broken = scene()
    broken["hands"][0]["edges"] = [[0, 5]]
    with pytest.raises(ValueError):
        Scene.model_validate(broken)


def test_read_only_pending_and_hash_bound_scene(tmp_path):
    artifact = tmp_path / "official.json"
    artifact.write_text(json.dumps(scene()))
    index = tmp_path / "index.json"
    index.write_text(
        json.dumps(
            {
                "episodes": {
                    "public-44": {
                        "variants": [
                            {
                                "id": "official",
                                "status": "succeeded",
                                "kind": "kinematic_reference",
                                "scene_path": "official.json",
                                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                            },
                            {"id": "rollout", "status": "pending", "kind": "policy_rollout"},
                        ]
                    }
                }
            }
        )
    )
    before = artifact.stat().st_mtime_ns
    store = PolicyStore(index)
    result = store.episode("public-44")
    assert result.variants[0].scene_url.endswith("/official")
    assert result.variants[1].scene_url is None
    assert store.scene("public-44", "official").kind == "kinematic_reference"
    assert artifact.stat().st_mtime_ns == before
    artifact.write_text("{}")
    with pytest.raises(ValueError, match="SHA"):
        store.scene("public-44", "official")


def test_missing_episode_is_empty_and_arbitrary_paths_rejected(tmp_path):
    store = PolicyStore(tmp_path / "not-created.json")
    assert store.episode("public-44").variants == []
    (tmp_path / "index.json").write_text(
        json.dumps(
            {
                "episodes": {
                    "x": {
                        "variants": [
                            {
                                "id": "official",
                                "status": "succeeded",
                                "scene_path": "../secret",
                                "sha256": "a" * 64,
                            }
                        ]
                    }
                }
            }
        )
    )
    with pytest.raises(ValueError):
        PolicyStore(tmp_path / "index.json").scene("x", "official")


@pytest.mark.parametrize(
    "quality",
    [
        {"valid_frames": [True]},
        {"source_codes": [1]},
        {"source_codes": [1, 5]},
        {"source_codes": [True, 1]},
        {"valid_frames": [1, 0]},
        {"valid_frames": ["true", "false"]},
    ],
)
def test_scene_validates_per_hand_quality_types_and_time_axis(quality):
    raw = scene()
    raw["hands"][0].update(quality)
    with pytest.raises(ValueError):
        Scene.model_validate(raw)


def test_unknown_quality_and_assistance_are_not_assumed_observed_or_unassisted():
    result = Scene.model_validate(scene())
    assert result.hands[0].source_codes is None
    assert result.hands[0].valid_frames is None
    assert result.assistance == "unknown"
    assert (
        PolicyVariant(id="rollout", status="succeeded", kind="policy_rollout").assistance == "unknown"
    )
    for assistance in ("none", "assisted", "unknown"):
        assert Scene.model_validate({**scene(), "assistance": assistance}).assistance == assistance
    with pytest.raises(ValueError):
        Scene.model_validate({**scene(), "assistance": "guessed"})


def test_missing_hand_geometry_requires_no_positive_valid_claim():
    raw = scene()
    raw["hands"][0].update(
        points=[None, [[0, 0, 0], [1, 0, 0]]], valid_frames=[False, True], source_codes=[0, 1]
    )
    assert Scene.model_validate(raw).hands[0].points[0] is None
    raw["hands"][0]["valid_frames"][0] = True
    with pytest.raises(ValueError, match="有效手帧"):
        Scene.model_validate(raw)
