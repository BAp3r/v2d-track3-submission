from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.nvidia_v2d.policy_replay_contract import (
    BASE_OBJECTIVE,
    BASE_WRIST,
    PHYSICAL_WRIST,
    RELEASE_OBJECTIVE,
    SEMANTIC_OBJECTIVE,
    require_replay_compatibility,
)


def saved_config(*, release=None, original_wrist=False):
    """Minimal actual 001/003/005 resolved-header shapes, without GPU imports."""
    objective = {"_target_": BASE_OBJECTIVE, "unintended_contact": {"weight": 0.0}}
    if release is not None:
        objective.update(_target_=RELEASE_OBJECTIVE, release_start_frame=release)
    action = {"wrist_pos_clip": 0.2, "wrist_kp": 25.0}
    wrist = {"_target_": BASE_WRIST} if original_wrist else {
        "_target_": PHYSICAL_WRIST, "kp": 100.0, "kd": 12.0}
    sim = {"fps": 20.0, "substeps": 5}
    return {"objective": objective, "action": action, "sim": sim,
            "env": {"objective": deepcopy(objective), "action": deepcopy(action), "sim": deepcopy(sim)},
            "embodiment": {"config": wrist}, "task": {"motion_speed": 0.5, "parquet": "/input/original.parquet"},
            "reset": {"always_reset_to_first_frame": False},
            "curriculum": {"stages": [{"objective_weights": {"unintended_contact": 0.0}}]}}


def desired(**changes):
    result = {"resume_checkpoint": Path("/archive/state_2621440.safetensors"),
              "resume_replay": Path("/archive/replay_2621440.safetensors"),
              "reset_replay": False, "motion_speed": 0.5, "wrist_kp": 100.0,
              "wrist_kd": 12.0, "wrist_pos_clip": None, "release_start_frame": None,
              "unintended_contact_weight": 0.0, "always_reset_to_first_frame": False,
              "parquet": "/input/original.parquet"}
    return result | changes


def test_005_release_mask_cannot_reuse_003_unmasked_objective_terms():
    # Reweighting old unmasked terms cannot reconstruct the new pre-release zeros.
    cfg = saved_config()
    with pytest.raises(ValueError, match=r"objective\._target_.*release_start_frame.*reset_replay=True"):
        require_replay_compatibility(desired(release_start_frame=407, unintended_contact_weight=-10), cfg)


@pytest.mark.parametrize(("before", "after"), [(407, 408), (407, None), (None, 407)])
def test_release_gate_changes_require_fresh_terms(before, after):
    with pytest.raises(ValueError, match="reset_replay=True"):
        require_replay_compatibility(desired(release_start_frame=after), saved_config(release=before))


@pytest.mark.parametrize(("change", "field"), [
    ({"wrist_pos_clip": 0.075}, "action.wrist_pos_clip"),
    ({"motion_speed": 1.0}, "task.motion_speed"),
    ({"sim_substeps": 10}, "sim.substeps"),
    ({"wrist_kp": 125.0}, "embodiment.wrist_kp"),
    ({"wrist_kd": 15.0}, "embodiment.wrist_kd"),
    ({"wrist_kp": None, "wrist_kd": None}, "embodiment.wrist_kp"),
    ({"parquet": Path("/input/repaired.parquet")}, "task.parquet"),
])
def test_changed_transition_or_action_semantics_reject_old_replay(change, field):
    with pytest.raises(ValueError, match=field.replace(".", r"\.")) as error:
        require_replay_compatibility(desired(**change), saved_config())
    assert "reset_replay=True" in str(error.value)


def test_weight_only_and_reset_distribution_changes_remain_legal():
    cfg = saved_config()
    cfg_before = deepcopy(cfg)
    spec = desired(unintended_contact_weight=-10, always_reset_to_first_frame=True)
    spec_before = deepcopy(spec)
    receipt = require_replay_compatibility(spec, cfg)
    assert receipt["status"] == "compatible"
    assert receipt["replay_reused"] is True
    assert cfg == cfg_before and spec == spec_before


def test_same_release_semantics_and_physical_gains_are_compatible():
    # action.wrist_kp is still 25; physical translational gains are 100/12.
    receipt = require_replay_compatibility(desired(release_start_frame=407), saved_config(release=407))
    assert receipt["saved"]["embodiment.wrist_kp"] == 100.0
    assert receipt["saved"]["embodiment.wrist_kd"] == 12.0


def test_legacy_original_wrist_and_missing_optional_spec_fields_use_frozen_defaults():
    cfg = saved_config(original_wrist=True)
    minimal = {"resume_checkpoint": "state_256.safetensors", "resume_replay": "replay_256.safetensors",
               "parquet": "/input/original.parquet"}
    receipt = require_replay_compatibility(minimal, cfg)
    assert receipt["saved"] == receipt["desired"]
    assert receipt["saved"]["embodiment.wrist_kp"] == 25.0
    assert receipt["saved"]["objective.release_start_frame"] is None


def test_explicit_native_wrist_joint_override_is_not_mistaken_for_default():
    cfg = saved_config(original_wrist=True)
    cfg["embodiment"]["config"]["joint_overrides"] = {r"wrist_[xyz]$": {"kp": 80, "kd": 10}}
    with pytest.raises(ValueError, match="embodiment.wrist_kp"):
        require_replay_compatibility(desired(wrist_kp=None, wrist_kd=None), cfg)


def test_equal_effective_gains_allow_native_to_known_factory_switch():
    receipt = require_replay_compatibility(desired(wrist_kp=25, wrist_kd=6), saved_config(original_wrist=True))
    assert receipt["status"] == "compatible"


@pytest.mark.parametrize("section", ["objective", "action"])
def test_conflicting_env_copy_is_rejected(section):
    cfg = saved_config()
    if section == "objective":
        cfg["env"][section].update(_target_=RELEASE_OBJECTIVE, release_start_frame=407)
    else:
        cfg["env"][section]["wrist_pos_clip"] = 0.075
    with pytest.raises(ValueError, match="copies disagree.*reset_replay=True"):
        require_replay_compatibility(desired(), cfg)


def test_single_resolved_placement_is_supported():
    cfg = saved_config()
    del cfg["action"]
    del cfg["env"]["objective"]
    assert require_replay_compatibility(desired(), cfg)["status"] == "compatible"


@pytest.mark.parametrize("problem", ["missing_config", "missing_speed", "missing_clip", "unknown_objective", "unknown_wrist", "missing_reference"])
def test_unknown_semantics_fail_closed(problem):
    cfg = saved_config()
    if problem == "missing_config":
        cfg = None
    elif problem == "missing_speed":
        del cfg["task"]["motion_speed"]
    elif problem == "missing_clip":
        del cfg["action"]["wrist_pos_clip"]
    elif problem == "unknown_objective":
        cfg["objective"]["_target_"] = "future.objective.Factory"
    elif problem == "missing_reference":
        del cfg["task"]["parquet"]
    else:
        cfg["embodiment"]["config"]["_target_"] = "future.wrist.Factory"
    with pytest.raises(ValueError, match="Cannot verify.*reset_replay=True"):
        require_replay_compatibility(desired(), cfg)


@pytest.mark.parametrize("cfg", [None, {}, {"objective": "unknown"}])
def test_fresh_replay_does_not_need_old_semantics(cfg):
    spec = desired(reset_replay=True, resume_replay=None, release_start_frame=407, wrist_pos_clip=0.075)
    assert require_replay_compatibility(spec, cfg) == {
        "status": "fresh_replay", "replay_reused": False, "checked_fields": []}
    assert require_replay_compatibility({}, cfg)["status"] == "new_training"


def test_attributes_interface_avoids_importing_single_sample():
    assert require_replay_compatibility(SimpleNamespace(**desired()), saved_config())["replay_reused"]


def test_inconsistent_resume_flags_cannot_silently_bypass_check():
    for spec in [desired(resume_replay=None), desired(resume_checkpoint=None), desired(reset_replay=True),
                 desired(reset_replay="false")]:
        with pytest.raises(ValueError):
            require_replay_compatibility(spec, saved_config())


@pytest.mark.parametrize("field", ["wrist_pos_clip", "motion_speed", "wrist_kp", "wrist_kd"])
def test_nonfinite_desired_semantics_are_rejected(field):
    with pytest.raises(ValueError, match="finite positive"):
        require_replay_compatibility(desired(**{field: float("nan")}), saved_config())


def test_semantic_contact_definition_cannot_reuse_old_aggregate_wrench_terms():
    spec = desired(release_start_frame=407, semantic_map_path="/roles.npz",
        semantic_map_sha256="a"*64, semantic_reference_sha256="b"*64, semantic_sigma_m=.02)
    cfg = saved_config(release=407)
    with pytest.raises(ValueError, match="reset_replay=True"):
        require_replay_compatibility(spec, cfg)
    for objective in (cfg["objective"], cfg["env"]["objective"]):
        objective.update(_target_=SEMANTIC_OBJECTIVE, semantic_map_sha256="a"*64,
            semantic_reference_sha256="b"*64, semantic_sigma_m=.02, semantic_motion_speed=.5)
    assert require_replay_compatibility(spec, cfg)["status"] == "compatible"
    with pytest.raises(ValueError, match="semantic_map_sha256"):
        require_replay_compatibility(spec | {"semantic_map_sha256": "c"*64}, cfg)
    with pytest.raises(ValueError, match="semantic_alignment_weight"):
        require_replay_compatibility(spec | {"semantic_alignment_weight": .25}, cfg)
