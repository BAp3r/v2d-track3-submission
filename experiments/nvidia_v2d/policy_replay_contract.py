"""Check the known single-sample semantic changes before reusing SAC replay.

Replay stores already-computed, unweighted objective terms and transitions.
Changing weights can reweight those terms; changing their definition cannot.
This is a bounded check of the fields below, not a proof of compatibility for
future changes to references, observations, dynamics, code, or term schemas.
It does not validate checkpoint/replay identity or read any checkpoint tensors.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

BASE_OBJECTIVE = "flash_chord.objectives.config.ObjectiveConfig"
RELEASE_OBJECTIVE = "experiments.nvidia_v2d.release_contact_objective.make_release_contact_objective"
SEMANTIC_OBJECTIVE = "experiments.nvidia_v2d.semantic_contact_objective.make_semantic_contact_objective"
BASE_WRIST = "flash_chord.embodiments.sharpa_hands.SharpaHandsConfig"
PHYSICAL_WRIST = "experiments.nvidia_v2d.physical_wrist_config.make_sharpa_config"
CHECKED_FIELDS = ("objective._target_", "objective.release_start_frame", "action.wrist_pos_clip",
                  "task.motion_speed", "embodiment.wrist_kp", "embodiment.wrist_kd", "task.parquet",
                  "objective.semantic_map_sha256", "objective.semantic_reference_sha256", "objective.semantic_sigma_m",
                  "objective.semantic_motion_speed", "objective.semantic_alignment_weight", "sim.substeps", "sim.fps")
_MISSING = object()


def _field(value: Mapping[str, Any] | object, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def _release(value: Any) -> int | None:
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError("objective.release_start_frame must be a nonnegative integer or None")
    return value


def _reference_path(value: Any) -> str:
    if not isinstance(value, (str, Path)) or not str(value) or not Path(value).is_absolute():
        raise ValueError("reference parquet must be an explicit absolute path")
    return str(Path(value))


def _path(value: Mapping, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _section(config: Mapping, name: str) -> list[Mapping]:
    """Accept either old placement; reject contradictory top-level/env copies."""
    found = [_path(config, path) for path in (name, f"env.{name}")]
    sections = [item for item in found if item is not _MISSING]
    if not sections or any(not isinstance(item, Mapping) for item in sections):
        raise ValueError(f"saved config lacks a valid {name} section")
    return sections


def _agreed(values: list[Any], name: str) -> Any:
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"saved top-level and env copies disagree for {name}")
    return values[0]


def _desired_semantics(desired: Mapping[str, Any] | object) -> dict[str, Any]:
    release = _release(_field(desired, "release_start_frame"))
    kp, kd = _field(desired, "wrist_kp"), _field(desired, "wrist_kd")
    if (kp is None) != (kd is None):
        raise ValueError("desired wrist gains require both wrist_kp and wrist_kd")
    clip = _field(desired, "wrist_pos_clip")
    semantic = _field(desired, "semantic_map_path") is not None
    return {"objective._target_": SEMANTIC_OBJECTIVE if semantic else BASE_OBJECTIVE if release is None else RELEASE_OBJECTIVE,
            "objective.release_start_frame": release,
            "action.wrist_pos_clip": _number(0.2 if clip is None else clip, "desired wrist_pos_clip"),
            "task.motion_speed": _number(_field(desired, "motion_speed", 0.5), "desired motion_speed"),
            "embodiment.wrist_kp": _number(25.0 if kp is None else kp, "desired wrist_kp"),
            "embodiment.wrist_kd": _number(6.0 if kd is None else kd, "desired wrist_kd"),
            "task.parquet": _reference_path(_field(desired, "parquet")),
            "sim.substeps": _number(_field(desired, "sim_substeps", 5), "desired sim.substeps"),
            "sim.fps": 20.0,
            "objective.semantic_map_sha256": _field(desired, "semantic_map_sha256") if semantic else None,
            "objective.semantic_reference_sha256": _field(desired, "semantic_reference_sha256") if semantic else None,
            "objective.semantic_alignment_weight": _field(desired, "semantic_alignment_weight", 0.) if semantic else None,
            "objective.semantic_motion_speed": _number(_field(desired, "motion_speed", .5), "desired semantic speed") if semantic else None,
            "objective.semantic_sigma_m": _number(_field(desired, "semantic_sigma_m", .02), "desired semantic sigma") if semantic else None}


def _saved_semantics(config: Mapping[str, Any]) -> dict[str, Any]:
    objectives = _section(config, "objective")
    targets = [section.get("_target_") for section in objectives]
    if any(target not in (BASE_OBJECTIVE, RELEASE_OBJECTIVE, SEMANTIC_OBJECTIVE) for target in targets):
        raise ValueError("saved objective target is unknown to this replay contract")
    target = _agreed(targets, "objective._target_")
    release = _agreed([_release(section.get("release_start_frame")) for section in objectives],
                       "objective.release_start_frame")
    if (target != BASE_OBJECTIVE) != (release is not None):
        raise ValueError("saved objective target and release_start_frame are inconsistent")
    actions = _section(config, "action")
    clip = _agreed([_number(section.get("wrist_pos_clip"), "saved action.wrist_pos_clip")
                    for section in actions], "action.wrist_pos_clip")
    wrist = _path(config, "embodiment.config")
    if not isinstance(wrist, Mapping):
        raise TypeError("saved config lacks embodiment.config")
    wrist_target = wrist.get("_target_")
    if wrist_target == BASE_WRIST:
        # The frozen upstream SharpaHandsConfig omits these defaults in Hydra JSON.
        # Its translational wrist drive is kp=25, kd=6 (not action.wrist_kp).
        overrides = wrist.get("joint_overrides", {})
        if not isinstance(overrides, Mapping):
            raise ValueError("saved wrist joint_overrides is not a mapping")
        gains = overrides.get(r"wrist_[xyz]$", {"kp": 25.0, "kd": 6.0})
        if not isinstance(gains, Mapping):
            raise ValueError("saved translational wrist overrides are not a mapping")
        kp, kd = gains.get("kp"), gains.get("kd")
    elif wrist_target == PHYSICAL_WRIST:
        kp, kd = wrist.get("kp"), wrist.get("kd")
    else:
        raise ValueError("saved wrist target is unknown to this replay contract")
    return {"objective._target_": target, "objective.release_start_frame": release,
            "action.wrist_pos_clip": clip,
            "task.motion_speed": _number(_path(config, "task.motion_speed"), "saved task.motion_speed"),
            "embodiment.wrist_kp": _number(kp, "saved wrist_kp"),
            "embodiment.wrist_kd": _number(kd, "saved wrist_kd"),
            "task.parquet": _reference_path(_path(config, "task.parquet")),
            "objective.semantic_alignment_weight": _agreed([o.get("semantic_alignment_weight", 0.)
                for o in objectives], "semantic_alignment_weight") if target == SEMANTIC_OBJECTIVE else None,
            **{f"sim.{key}": _agreed([_number(s.get(key), f"saved sim.{key}")
                for s in _section(config, "sim")], f"sim.{key}") for key in ("substeps", "fps")},
            **{f"objective.{key}": _agreed([o.get(key) for o in objectives], key) if target == SEMANTIC_OBJECTIVE else None
               for key in ("semantic_map_sha256", "semantic_reference_sha256", "semantic_sigma_m", "semantic_motion_speed")}}


def require_replay_compatibility(desired: Mapping[str, Any] | object,
                                 saved_config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Raise ValueError for incompatible replay; return a JSON-serializable receipt.

    ``desired`` may be SingleSampleSpec or its model_dump mapping. ``saved_config``
    is json.loads(read_checkpoint_metadata(state)["resolved_config_json"]).
    No imports from policy_single_sample, FlashCHORD, JAX, or GPU code are needed.
    Weight-only and reset-distribution changes are deliberately outside comparison.
    """
    checkpoint = _field(desired, "resume_checkpoint")
    replay = _field(desired, "resume_replay")
    reset = _field(desired, "reset_replay", False)
    if type(reset) is not bool:
        raise ValueError("reset_replay must be a boolean")
    if reset:
        if replay is not None:
            raise ValueError("reset_replay=True requires removing the old resume_replay path")
        return {"status": "fresh_replay", "replay_reused": False, "checked_fields": []}
    if checkpoint is None and replay is None:
        return {"status": "new_training", "replay_reused": False, "checked_fields": []}
    if checkpoint is None or replay is None:
        raise ValueError("resume requires checkpoint and replay, or explicit reset_replay=True")
    expected = _desired_semantics(desired)
    try:
        if not isinstance(saved_config, Mapping):
            raise TypeError("saved resolved configuration is unavailable")
        saved = _saved_semantics(saved_config)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cannot verify replay semantics: {exc}; require reset_replay=True") from exc
    changed = [name for name in CHECKED_FIELDS if saved[name] != expected[name]]
    if changed:
        details = "; ".join(f"{name}: saved={saved[name]!r}, desired={expected[name]!r}" for name in changed)
        raise ValueError(f"Incompatible replay semantics ({details}); require reset_replay=True")
    return {"status": "compatible", "replay_reused": True, "checked_fields": list(CHECKED_FIELDS),
            "saved": saved, "desired": expected,
            "scope": "known objective target/release gate, wrist residual clip, motion speed, translational wrist gains, reference path; content hashes are separately bound by input gates"}
