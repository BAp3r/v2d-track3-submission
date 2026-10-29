"""Render recorded physical diagnostics; never substitute reference robot poses.

Every world remains in the numerical report. Learned-policy success labels use
that world's recorded completion and errors, never the cohort success rate.
CPU only. Use the standalone geometry environment for ``render`` and a Python
with matplotlib for ``plot``; no simulator import is needed.

For a NEW captured learned-policy rollout (run only once the result exists)::

    python -m experiments.nvidia_v2d.policy_diagnostic_visuals render \
      --root .e2e/zero_success_20261027 \
      --diagnostic /absolute/path/to/capture/result --world 0 \
      --output /absolute/path/to/new-visual-output \
      --assets .e2e/zero_success_20261027/inputs/sharpa-visual-assets \
      --plot-python /absolute/path/to/python-with-matplotlib

The capture must contain diagnostic.npz, diagnostic.manifest.json and
diagnostic.metrics.json. A SUCCESS label requires captured
reference_tracking_criteria plus completion_step, completion_terminated,
completion_truncated and completion_reference_progress arrays, one item per
world. Missing evidence remains UNVERIFIED; aggregate SR never assigns worlds.
Time labels prefer captured ``reference_source_time_s`` with verified
``reference_timing`` metadata. Older captures can recover time from their source
parquet only after its interpolated wrists/object positions match the capture.
Otherwise labels explicitly say nominal progress, not source time. A render
freezes ``timeline.json`` for its plotting subprocess; standalone ``plot`` can
reuse it with ``--timeline /path/to/timeline.json``. No simulator import is used.
Use --selection-post-hoc and --selection-note to document any post-hoc choice. Omit --diagnostic only for
the original four-case comparison, whose world11 choice was post-hoc.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

CASES = (
    ("diagnostic-original", "Infinite plane / zero residual / speed 1.0", 0),
    ("diagnostic-finite-half-speed", "Finite table / zero residual / speed 0.5", 0),
    ("diagnostic-scripted-grasp-001", "Finite table / scripted kp25 / world11", 11),
    ("diagnostic-scripted-grasp-kp100-001", "Finite table / scripted kp100 / world11", 11),
)
COLORS = ("#a76565", "#7f90a6", "#d5a03e", "#2694bc")
ERROR_NAMES = ("wrist_position_m", "wrist_orientation_rad", "object_position_m", "object_orientation_rad")
COMPLETION_FIELDS = ("completion_step", "completion_terminated", "completion_truncated", "completion_reference_progress")
PLAYBACK_METHOD = "flash_chord.data.resampling.playback_times"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def orientation_error(actual: np.ndarray, reference: np.ndarray) -> np.ndarray:
    actual = actual / np.linalg.norm(actual, axis=-1, keepdims=True)
    reference = reference / np.linalg.norm(reference, axis=-1, keepdims=True)
    dot = np.abs(np.sum(actual * reference, axis=-1))
    return 2 * np.arccos(np.clip(dot, 0, 1))


def playback_reference_times(source_count, source_fps, control_fps, motion_speed):
    """Mirror the frozen runtime API; speed1 uses arange+clip, others linspace.

    Source count is the nested object-position array length, not parquet row count.
    This helper alone does not establish provenance: callers must bind metadata
    to a runtime capture or verify interpolated source positions against it.
    """
    if (isinstance(source_count, bool) or int(source_count) != source_count or source_count < 2
            or not np.isfinite([source_fps, control_fps, motion_speed]).all()
            or min(source_fps, control_fps, motion_speed) <= 0):
        raise ValueError("invalid reference timing metadata")
    duration = (source_count - 1) / source_fps
    if motion_speed == 1:
        count = round(duration * control_fps) + 1
        return np.clip(np.arange(count, dtype=float) / control_fps, 0, duration)
    count = int(duration * control_fps / motion_speed)
    if count < 2:
        raise ValueError("reference timing has fewer than two samples")
    return np.linspace(0, duration, count)


def reference_timeline(data, manifest, report, root):
    """Return (timestamps, provenance, additional input hashes), without guessing exact time."""
    steps = len(data["reference_pose_w"])
    fps = float(manifest["fps"])
    speed = float(manifest.get("motion_speed_override") or 1)
    nominal = np.arange(steps) * speed / fps

    def fallback(reason):
        return nominal, {"verified": False, "method": "nominal_control_progress",
                         "reason": reason, "label": "Nominal progress (not verified source time)"}, {}

    if "reference_source_time_s" in data:
        metadata = dict(manifest.get("reference_timing") or {})
        required = ("source_path", "source_fps", "source_frame_count", "control_fps", "motion_speed")
        if (metadata.get("verified") is not True or metadata.get("method") != PLAYBACK_METHOD
                or any(key not in metadata for key in required)):
            raise ValueError("captured reference timestamps lack verified runtime timing provenance")
        values = np.asarray(data["reference_source_time_s"], dtype=float)
        expected = playback_reference_times(metadata["source_frame_count"], metadata["source_fps"],
                                            metadata["control_fps"], metadata["motion_speed"])
        if (values.shape != (steps,) or len(expected) != steps or not np.isfinite(values).all()
                or np.any(np.diff(values) <= 0) or metadata["control_fps"] != fps
                or not np.allclose(values, expected, rtol=0, atol=1e-12)):
            raise ValueError("captured reference timestamps disagree with runtime timing metadata")
        metadata.update(label="Verified source reference time", origin="captured_runtime_array")
        return values, metadata, {}

    path_text = report.get("parquet")
    if not path_text:
        return fallback("capture has no reference timestamp array or source parquet path")
    source_path = Path(path_text)
    # This repository migrated this exact legacy mount into the experiment root.
    legacy_mount = Path("/var/tmp/fmc3-sharpa-validation-20261027-001")
    if source_path.is_relative_to(legacy_mount):
        migrated = root / "inputs/legacy" / source_path.relative_to(legacy_mount)
        if migrated.is_file():
            source_path = migrated
    if not source_path.is_file():
        return fallback("recorded source parquet is unavailable")
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return fallback("pyarrow unavailable; source input cannot be verified in this Python")
    cells = pq.read_table(source_path).to_pylist()
    if len(cells) != 1:
        return fallback("source schema is not the single-row ManoSharpa reference")
    cells = cells[0]
    required = ("fps", "object_body_position", "robot_left_wrist_position", "robot_right_wrist_position")
    if any(key not in cells for key in required) or any(f"reference_{side}_wrist" not in data for side in ("left", "right")):
        return fallback("source/capture lacks positions needed to verify reference timing")
    count, source_fps = len(cells["object_body_position"]), float(cells["fps"])
    values = playback_reference_times(count, source_fps, fps, speed)
    if len(values) != steps:
        return fallback("source metadata does not reproduce captured reference length")
    original = np.arange(count) / source_fps
    for raw, captured in [(cells["object_body_position"], data["reference_pose_w"][..., :3]),
                           *[(cells[f"robot_{side}_wrist_position"], data[f"reference_{side}_wrist"]) for side in ("left", "right")]]:
        raw = np.asarray(raw, dtype=float)
        flat = raw.reshape(count, -1)
        interpolated = np.stack([np.interp(values, original, column) for column in flat.T], axis=-1)
        interpolated = interpolated.reshape((steps, *raw.shape[1:]))
        if interpolated.shape != np.asarray(captured).shape or not np.allclose(interpolated, captured, rtol=0, atol=1e-8):
            return fallback("source interpolation does not match captured reference positions")
    digest = sha256(source_path)
    return values, {"verified": True, "method": PLAYBACK_METHOD, "origin": "source_parquet_verified_against_capture",
                    "source_path": str(source_path.resolve()), "recorded_source_path": path_text, "source_sha256": digest,
                    "source_fps": source_fps, "source_frame_count": count, "control_fps": fps, "motion_speed": speed,
                    "position_match_atol_m": 1e-8, "label": "Verified source reference time"}, {str(source_path.resolve()): digest}


def validate_timeline_record(record, capture_hash, steps):
    """Reuse the rendering process's proven timeline in a different plotting Python."""
    if record.get("capture_npz_sha256") != capture_hash:
        raise ValueError("timeline belongs to a different capture")
    values = np.asarray(record["timestamps_s"], dtype=float)
    if values.shape != (steps,) or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
        raise ValueError("invalid frozen timeline")
    return values, record["provenance"]


def per_world_outcomes(data, manifest, report):
    """Match official score_reference_tracking, requiring per-world completion.

    Older captures lack completion and complete criteria. They can prove a
    threshold crossing, but passing errors or a cohort SR cannot prove which
    particular world completed. Those worlds remain UNVERIFIED.
    """
    errors = np.asarray(data["tracking_error"], dtype=float)
    steps, worlds, width = errors.shape
    if width != 4 or tuple(manifest["error_names"]) != ERROR_NAMES:
        raise ValueError("this public44 renderer requires the four Sharpa tracking errors")
    if (steps < 1 or worlds < 1 or report["world_count"] != worlds or report["step_count"] != steps
            or manifest["world_count"] != worlds or manifest["steps"] != steps):
        raise ValueError("report dimensions disagree with the recorded cohort")
    expected_full = {"reset_mode": "explicit", "start_frame": 0, "motion_start_frame": 0, "motion_end_frame": -1}
    if any(report.get(key) != value for key, value in expected_full.items()):
        raise ValueError("renderer requires a complete sequence with an explicit reset")
    criteria = manifest.get("reference_tracking_criteria")
    criteria_verified = criteria is not None
    if criteria is None:
        criteria = {"error_names": list(ERROR_NAMES),
                    "thresholds": [None, None, report.get("object_position_threshold_m"), report.get("object_orientation_threshold_rad")],
                    "active_after_step": [-1] * 4,
                    "causes": [["wrist", [0, 1]], ["object", [2, 3]]]}
    if (tuple(criteria["error_names"]) != ERROR_NAMES or len(criteria["thresholds"]) != width
            or len(criteria["active_after_step"]) != width):
        raise ValueError("recorded criteria do not match the tracking-error layout")
    if (criteria["thresholds"][2] != report.get("object_position_threshold_m")
            or criteria["thresholds"][3] != report.get("object_orientation_threshold_rad")):
        raise ValueError("criteria object thresholds disagree with the official report")
    failures = np.zeros((steps, worlds), dtype=bool)
    first_failure = np.full(worlds, -1, dtype=int)
    for indices in (indices for _, indices in criteria["causes"]):
        for index in indices:
            limit = criteria["thresholds"][index]
            if limit is None:
                continue
            if not np.isfinite(limit) or limit < 0:
                raise ValueError("invalid recorded tracking threshold")
            active = np.arange(steps) > criteria["active_after_step"][index]
            failures |= active[:, None] & (errors[..., index] > limit)
    for world in range(worlds):
        crossed = np.flatnonzero(failures[:, world])
        if len(crossed):
            first_failure[world] = int(crossed[0])
    invalid = ~np.isfinite(errors).all(axis=(0, 2))
    invalid[np.asarray(report.get("non_finite_worlds", []), dtype=int)] = True
    complete = all(name in data for name in COMPLETION_FIELDS)
    reached_end = None
    if complete:
        for name in COMPLETION_FIELDS:
            if np.asarray(data[name]).shape != (worlds,):
                raise ValueError(f"{name} must contain one item per world")
        terminated = np.asarray(data["completion_terminated"], dtype=bool)
        truncated = np.asarray(data["completion_truncated"], dtype=bool)
        progress = np.asarray(data["completion_reference_progress"], dtype=float)
        if terminated.any():
            raise ValueError("capture terminated early; cannot score full-reference evaluation")
        reached_end = truncated & np.isclose(progress, 1.0, rtol=0, atol=1e-6)
        invalid |= ~np.isfinite(progress)
    verified = complete and criteria_verified
    successes = None
    if verified:
        successes = reached_end & ~invalid & ~failures.any(axis=0)
        rate = float(np.mean(successes))
        tracking_report = report["reference_tracking"]
        for value in (report["metrics"]["chord_sr"], tracking_report["success_rate"]):
            if not np.isclose(rate, value, rtol=0, atol=1e-7):
                raise ValueError("per-world successes disagree with official aggregate CHORD SR")
        if not np.isclose(np.mean(reached_end), tracking_report["reference_end_fraction"], rtol=0, atol=1e-7):
            raise ValueError("per-world completion disagrees with official reference-end fraction")
    outcomes = []
    for world in range(worlds):
        success = bool(successes[world]) if verified else (False if invalid[world] or failures[:, world].any() else None)
        outcomes.append({"world": world, "success": success,
                         "status": "SUCCESS" if success is True else "FAILED" if success is False else "UNVERIFIED",
                         "completion_verified": bool(complete), "criteria_verified": bool(criteria_verified),
                         "reference_end_reached": bool(reached_end[world]) if complete else None,
                         "first_failure_step": int(first_failure[world]) if first_failure[world] >= 0 else None,
                         "invalid": bool(invalid[world])})
    return outcomes, criteria


def controller_label(manifest):
    mode = manifest.get("action_mode", "zero")
    if mode == "policy":
        return "learned-policy", "Learned policy / 学习策略", True
    if mode == "scripted-grasp":
        return "scripted-diagnostic", "Scripted diagnostic / 脚本诊断（非学习策略）", False
    return "reference-diagnostic", "Reference diagnostic / 零残差参考诊断（非学习策略）", False


def load_cases(root: Path, diagnostic: Path | None = None, world: int = 0, timeline: Path | None = None):
    cases = []
    hashes = {}
    frozen_timelines = json.loads(timeline.read_text()) if timeline is not None else None
    descriptions = CASES if diagnostic is None else ((str(diagnostic.resolve()), diagnostic.parent.name, world),)
    for directory, title, selected in descriptions:
        folder = root / directory / "result" if diagnostic is None else Path(directory)
        for name in ("diagnostic.npz", "diagnostic.manifest.json", "diagnostic.metrics.json"):
            path = folder / name
            hashes[str(path.resolve())] = sha256(path)
        manifest = json.loads((folder / "diagnostic.manifest.json").read_text())
        metrics = json.loads((folder / "diagnostic.metrics.json").read_text())
        if metrics["checkpoint_sha256"] != manifest["checkpoint_sha256"]:
            raise ValueError("capture manifest and report refer to different checkpoints")
        for name in ("diagnostic.npz", "diagnostic.metrics.json"):
            if name in manifest.get("files", {}) and manifest["files"][name] != sha256(folder / name):
                raise ValueError(f"capture hash mismatch: {name}")
        with np.load(folder / "diagnostic.npz") as archive:
            data = {key: archive[key] for key in archive.files}
        actual = data["achieved_pose_w"][:, :, 0]
        if not 0 <= selected < actual.shape[1]:
            raise ValueError("selected world is outside the captured cohort")
        reference = data["reference_pose_w"][:, None, 0]
        if frozen_timelines is None:
            times, timing, timing_hashes = reference_timeline(data, manifest, metrics, root)
            hashes.update(timing_hashes)
        else:
            times, timing = validate_timeline_record(frozen_timelines[directory], sha256(folder / "diagnostic.npz"), len(actual))
        errors = actual[..., :3] - reference[..., :3]
        orientations = orientation_error(actual[..., 3:], reference[..., 3:])
        # The simulator's recorded diagnostic is the authority for gate crossing.
        # Independently recomputed pose errors must agree before we visualize them.
        # Float32 subtraction/norm reductions can differ by a few rounding units,
        # especially in failed worlds hundreds of metres from the reference.
        # This tolerance only verifies the capture; per_world_outcomes still uses
        # the original recorded tracking_error and its unchanged strict thresholds.
        np.testing.assert_allclose(np.linalg.norm(errors, axis=-1), data["tracking_error"][..., 2],
                                   rtol=4 * np.finfo(np.float32).eps, atol=2e-6)
        np.testing.assert_allclose(orientations, data["tracking_error"][..., 3], atol=2e-3)
        scales = np.asarray(data["auxiliary_scales"])
        if scales.shape != (len(actual), 2) or not np.isfinite(scales).all() or np.any(scales):
            raise ValueError("a finite zero-assistance record is required for every step")
        object_body = manifest["body_labels"].index("white_pot/object")
        np.testing.assert_array_equal(data["actual_bodies"][:, :, object_body], actual)
        outcomes, criteria = per_world_outcomes(data, manifest, metrics)
        _, controller, learned = controller_label(manifest)
        if diagnostic is not None:
            title = f"{controller.split(' / ')[0]} / world{selected} / {outcomes[selected]['status']}"
        if not timing["verified"]:
            title += " / nominal time only"
        cases.append({"directory": directory, "title": title, "selected_world": selected,
                      "data": data, "manifest": manifest, "metrics": metrics, "times": times, "timing": timing,
                      "capture_npz_sha256": sha256(folder / "diagnostic.npz"),
                      "xyz_error": errors, "orientation_error": orientations, "object_body": object_body,
                      "outcomes": outcomes, "criteria": criteria, "learned_policy": learned})
    return cases, hashes


def statistics(cases):
    rows = []
    for case in cases:
        data = case["data"]
        worlds = case["manifest"]["world_count"]
        counts = data["hand_object_contact_counts"].reshape(len(case["times"]), worlds, -1).sum(axis=-1)
        for world in range(worlds):
            tracking = data["tracking_error"][:, world]
            position = np.linalg.norm(case["xyz_error"][:, world], axis=-1)
            initial_z = data["initial_bodies"][world, case["object_body"], 2]
            def first_above(values, index, times=case["times"], criteria=case["criteria"]):
                limit = criteria["thresholds"][index]
                if limit is None:
                    return None
                indices = np.flatnonzero((np.arange(len(times)) > criteria["active_after_step"][index]) & (values > limit))
                return float(times[indices[0]]) if len(indices) else None
            maximum_recorded_lift = float(data["achieved_pose_w"][:, world, 0, 2].max() - initial_z)
            rows.append({
                "case": case["directory"], "world": world,
                "selected_for_display": world == case["selected_world"],
                "world_count": worlds, "step_count": len(position),
                "object_position_mean_m": float(position.mean()),
                "object_position_max_m": float(position.max()),
                "object_orientation_mean_deg": float(np.rad2deg(tracking[:, 3]).mean()),
                "object_orientation_max_deg": float(np.rad2deg(tracking[:, 3]).max()),
                "wrist_position_mean_m": float(tracking[:, 0].mean()),
                "wrist_position_max_m": float(tracking[:, 0].max()),
                "object_max_lift_from_reset_m": max(0.0, maximum_recorded_lift),
                "object_max_signed_after_step_z_relative_reset_m": maximum_recorded_lift,
                "contact_steps": int(np.count_nonzero(counts[:, world])),
                "first_object_position_failure_source_s": first_above(tracking[:, 2], 2) if case["timing"]["verified"] else None,
                "first_object_orientation_failure_source_s": first_above(tracking[:, 3], 3) if case["timing"]["verified"] else None,
                "first_wrist_failure_source_s": first_above(tracking[:, 0], 0) if case["timing"]["verified"] else None,
                "timeline_verified": case["timing"]["verified"], "timeline_method": case["timing"]["method"],
                "first_failure_nominal_s": (case["outcomes"][world]["first_failure_step"] * (case["manifest"].get("motion_speed_override") or 1) / case["manifest"]["fps"]
                                            if case["outcomes"][world]["first_failure_step"] is not None else None),
                "full_sequence_tracking_success": case["outcomes"][world]["success"],
                "outcome_status": case["outcomes"][world]["status"],
                "completion_verified": case["outcomes"][world]["completion_verified"],
                "criteria_verified": case["outcomes"][world]["criteria_verified"],
                "learned_policy": case["learned_policy"], "auxiliary_scale_max": 0.0,
            })
    return rows


def plot(root: Path, output: Path, diagnostic: Path | None = None, world: int = 0, timeline: Path | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases, _ = load_cases(root, diagnostic, world, timeline)
    figure, axes = plt.subplots(3, 2, figsize=(15, 11), sharex=True, layout="constrained")
    labels = ["Object X error (cm)", "Object Y error (cm)", "Object Z error (cm)",
              "Object orientation error (degrees)", "Maximum wrist position error (cm)",
              "Object height relative to reset (cm)"]
    for index, case in enumerate(cases):
        color = COLORS[index % len(COLORS)]
        world = case["selected_world"]
        data = case["data"]
        initial_z = data["initial_bodies"][world, case["object_body"], 2]
        values = [case["xyz_error"][:, world, k] * 100 for k in range(3)]
        values += [np.rad2deg(data["tracking_error"][:, world, 3]),
                   data["tracking_error"][:, world, 0] * 100,
                   (data["achieved_pose_w"][:, world, 0, 2] - initial_z) * 100]
        for axis, value in zip(axes.flat, values, strict=True):
            axis.plot(case["times"], value, color=color, lw=1.5, label=case["title"])
    last = cases[-1]
    limits = last["criteria"]["thresholds"]
    if limits[3] is not None:
        axes.flat[3].axhline(np.rad2deg(limits[3]), color="black", ls="--", lw=1, label=f"Orientation gate {limits[3]} rad")
    if limits[0] is not None:
        axes.flat[4].axhline(limits[0] * 100, color="black", ls="--", lw=1, label=f"Wrist gate {limits[0] * 100:g} cm")
    ref_z = last["data"]["reference_pose_w"][:, 0, 2]
    axes.flat[5].plot(last["times"], (ref_z - ref_z[0]) * 100, color="black", ls="--", label="Reference object")
    for axis, label in zip(axes.flat, labels, strict=True):
        axis.set_ylabel(label)
        axis.grid(alpha=.22)
        axis.set_xlim(0, max(case["times"][-1] for case in cases))
    for axis in axes[-1]:
        axis.set_xlabel("Verified source reference time (s)" if all(case["timing"]["verified"] for case in cases)
                        else "Time (s): source time only for verified cases; otherwise nominal progress")
    axes.flat[0].legend(fontsize=8, loc="upper left")
    axes.flat[3].legend(fontsize=8, loc="upper left")
    axes.flat[4].legend(fontsize=8, loc="upper left")
    axes.flat[5].legend(fontsize=8, loc="lower left")
    if diagnostic is None:
        heading = ("Public44 physical diagnostics: every displayed rollout FAILED\n"
                   "World11 selected after inspecting 16 worlds; scripted feedback is NOT a trained policy")
    else:
        heading = (f"{last['title']}\nPer-world completion and full error trace determine the status; "
                   "cohort SR is not copied to each world")
    figure.suptitle(heading, fontsize=14)
    for extension in ("png", "svg"):
        figure.savefig(output / f"diagnostic-comparison.{extension}", dpi=155)
    plt.close(figure)

    rows = statistics(cases)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.4), layout="constrained")
    display_cases = cases[2:] if diagnostic is None else cases
    for index, case in enumerate(display_cases):
        color = COLORS[index % len(COLORS)]
        chosen = [row for row in rows if row["case"] == case["directory"]]
        worlds = [row["world"] for row in chosen]
        axes[0].plot(worlds, [row["object_position_mean_m"] * 100 for row in chosen], "o-", color=color, label=case["title"])
        axes[1].plot(worlds, [row["object_max_lift_from_reset_m"] * 100 for row in chosen], "o-", color=color, label=case["title"])
    for axis in axes:
        axis.axvline(last["selected_world"], color="black", ls="--", lw=1, label=f"Displayed world{last['selected_world']}")
        axis.set_xlabel("World index (all captured worlds; no failed/outlier worlds removed)")
        axis.set_xticks(range(max(case["manifest"]["world_count"] for case in display_cases)))
        axis.grid(alpha=.22)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Mean object position error (cm)")
    axes[1].set_ylabel("Maximum lift relative to reset (cm)")
    if diagnostic is None:
        all_heading = "All scripted sweeps: 0 / 16 successes for each gain; world13 fall is retained"
    else:
        outcomes = last["outcomes"]
        all_heading = (f"{controller_label(last['manifest'])[1].split(' / ')[0]}: "
                       f"{sum(row['status'] == 'SUCCESS' for row in outcomes)} success, "
                       f"{sum(row['status'] == 'FAILED' for row in outcomes)} failed, "
                       f"{sum(row['status'] == 'UNVERIFIED' for row in outcomes)} unverified")
    figure.suptitle(all_heading)
    for extension in ("png", "svg"):
        figure.savefig(output / f"all-worlds.{extension}", dpi=155)
    plt.close(figure)


def render(root: Path, output: Path, assets: Path, plot_python: Path,
           diagnostic: Path | None = None, world: int = 0, selection_note: str | None = None,
           selection_post_hoc: bool = False, display_note: str | None = None):
    import trimesh
    from PIL import Image, ImageDraw, ImageFont
    from scipy.spatial.transform import Rotation

    output.mkdir(parents=True, exist_ok=False)
    cases, hashes = load_cases(root, diagnostic, world)
    hashes[str(Path(__file__).resolve())] = sha256(Path(__file__))
    shutil.copy2(Path(__file__), output / "frozen-renderer.py")
    selected = cases[-1]
    timeline_path = output / "timeline.json"
    timeline_path.write_text(json.dumps({case["directory"]: {"capture_npz_sha256": case["capture_npz_sha256"],
                                "timestamps_s": case["times"].tolist(), "provenance": case["timing"]} for case in cases}, indent=2) + "\n")
    selected_world = selected["selected_world"]
    outcome = selected["outcomes"][selected_world]
    controller_type, controller, learned = controller_label(selected["manifest"])
    status = outcome["status"]
    fps = selected["manifest"]["fps"]
    steps = len(selected["times"])
    speed = selected["manifest"].get("motion_speed_override") or 1.0
    selection = selection_note or ("world11 has the lowest mean object position error among16 (post-hoc); all worlds retained"
                                   if diagnostic is None else "explicit world index; no automatic best-world selection; all worlds retained")
    selection_post_hoc = selection_post_hoc or diagnostic is None
    selection_prefix = "选后展示" if selection_post_hoc else "指定展示"
    successful_worlds = [row["world"] for row in selected["outcomes"] if row["success"] is True]
    cohort_verified = all(row["completion_verified"] and row["criteria_verified"] for row in selected["outcomes"])
    cohort_summary = (f"CHORD {len(successful_worlds)}/{selected['manifest']['world_count']}"
                      if cohort_verified else "CHORD逐环境证据不完整")
    maniptrans_sr = selected["metrics"]["metrics"].get("maniptrans_sr")
    if maniptrans_sr is not None:
        cohort_summary += f"; ManipTrans {maniptrans_sr:.1%}"
    status_caption = f"CHORD {status}" if learned else status
    if learned and status == "SUCCESS" and len(successful_worlds) == 1 and selection_post_hoc:
        selection_prefix = "选后唯一成功展示"
    rows = statistics(cases)
    (output / "all-worlds.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    with (output / "all-worlds.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mesh_path = root / "inputs/legacy/episodes/public44-005/assets/white_pot/collision.stl"
    mesh = trimesh.load_mesh(mesh_path, process=False)
    hashes[str(mesh_path.resolve())] = sha256(mesh_path)
    support_path = root / "audit/finite_table_001/manifest.json"
    support = json.loads(support_path.read_text())
    hashes[str(support_path.resolve())] = sha256(support_path)
    support_usda = support_path.with_name("support.usda")
    hashes[str(support_usda.resolve())] = sha256(support_usda)
    if hashes[str(support_usda.resolve())] != selected["manifest"].get("support_sha256"):
        raise ValueError("displayed support differs from the captured physical scene")
    table = trimesh.creation.cylinder(radius=support["fit"]["radius_m"], height=support["thickness_m"], sections=64)
    table.apply_translation([*support["fit"]["center_xy_m"], support["top_m"] - support["thickness_m"] / 2])
    colors = {"left": (40, 179, 225), "right": (249, 143, 76)}
    links = {}
    for side in ("left", "right"):
        urdf = assets / f"urdfs/sharpawave/{side}_sharpawave.urdf"
        hashes[str(urdf.resolve())] = sha256(urdf)
        for link in ET.parse(urdf).getroot().findall("link"):
            parts = []
            for visual in link.findall("visual"):
                element = visual.find("geometry/mesh")
                if element is None:
                    continue
                path = (urdf.parent / element.attrib["filename"]).resolve()
                hashes[str(path)] = sha256(path)
                geometry = trimesh.load_mesh(path, process=False).convex_hull
                origin = visual.find("origin")
                xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
                rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
                vertices = np.asarray(geometry.vertices) * np.fromstring(element.get("scale", "1 1 1"), sep=" ")
                vertices = vertices @ Rotation.from_euler("xyz", rpy).as_matrix().T + xyz
                parts.append((vertices, np.asarray(geometry.faces), colors[side]))
            links[link.attrib["name"]] = parts

    width, height = 800, 580
    u = np.array([.96, -.28, 0]); u /= np.linalg.norm(u)
    forward = np.array([.28, .96, -.50]); forward /= np.linalg.norm(forward)
    v = np.cross(u, forward); v /= np.linalg.norm(v)
    local_items = []
    for body, label in enumerate(selected["manifest"]["body_labels"]):
        name = label.split("/")[-1]
        if name == "object":
            local_items.append((body, np.asarray(mesh.vertices), np.asarray(mesh.faces), (127, 211, 108)))
        for vertices, faces, color in links.get(name, []):
            local_items.append((body, vertices, faces, color))
    # Fit ONE orthographic camera over the complete recorded selected rollout.
    # No per-frame camera following, cropping of a divergent hand, or state edit.
    bounds = [[float(np.min(table.vertices @ axis)), float(np.max(table.vertices @ axis))] for axis in (u, v)]
    poses = selected["data"]["actual_bodies"][:, selected_world]
    for body, vertices, _, _ in local_items:
        rotations = Rotation.from_quat(poses[:, body, 3:]).as_matrix()
        fit_vertices = np.unique(vertices, axis=0)
        for index, axis in enumerate((u, v)):
            direction = np.einsum("tij,i->tj", rotations, axis)
            projected = direction @ fit_vertices.T + (poses[:, body, :3] @ axis)[:, None]
            bounds[index][0] = min(bounds[index][0], float(projected.min()))
            bounds[index][1] = max(bounds[index][1], float(projected.max()))
    center_uv = np.mean(bounds, axis=1)
    camera_scale = min((width - 60) / (bounds[0][1] - bounds[0][0]),
                       (height - 90) / (bounds[1][1] - bounds[1][0]))
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    font = ImageFont.truetype(str(font_path), 22)
    small = ImageFont.truetype(str(font_path), 18)

    def project(points):
        points = np.asarray(points)
        return np.column_stack((width / 2 + (points @ u - center_uv[0]) * camera_scale,
                                (height + 30) / 2 - (points @ v - center_uv[1]) * camera_scale,
                                points @ forward))

    def scene(frame):
        image = Image.new("RGB", (width, height), (20, 29, 42))
        draw = ImageDraw.Draw(image)
        for x in np.arange(-.6, .61, .1):
            xy = project([[x, -.55, 0], [x, .55, 0]])
            draw.line([tuple(p[:2]) for p in xy], fill=(42, 55, 70))
        for y in np.arange(-.5, .51, .1):
            xy = project([[-.6, y, 0], [.6, y, 0]])
            draw.line([tuple(p[:2]) for p in xy], fill=(42, 55, 70))
        items = [(table.vertices, table.faces, (115, 81, 49))]
        poses = selected["data"]["actual_bodies"][frame, selected_world]
        for body, vertices, faces, color in local_items:
            rotation = Rotation.from_quat(poses[body, 3:]).as_matrix()
            position = poses[body, :3]
            items.append((vertices @ rotation.T + position, faces, color))
        triangles, depths, fills = [], [], []
        for vertices, faces, color in items:
            points = project(vertices)[faces]
            normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-10)
            shade = .42 + .58 * np.abs(normals @ np.array([.25, -.5, .829]))
            triangles.extend(points[:, :, :2].tolist())
            depths.extend(points[:, :, 2].mean(axis=1).tolist())
            fills.extend([tuple((np.asarray(color) * value).astype(int)) for value in shade])
        for i in np.argsort(depths)[::-1]:
            draw.polygon([tuple(point) for point in triangles[i]], fill=fills[i])
        draw.text((12, 12), "蓝=左手  橙=右手  绿=实际锅体", font=small, fill="white")
        return image

    video = output / f"{controller_type}-world{selected_world}-{status}.mp4"
    rgb_paths = sorted((root / "inputs/public44-source/job/reconstruction/frames").glob("[0-9][0-9][0-9][0-9][0-9][0-9].png"))
    source_fps = selected["timing"].get("source_fps", 20)
    if not rgb_paths or (selected["timing"]["verified"] and selected["timing"]["source_frame_count"] != len(rgb_paths)):
        raise ValueError("source reference frame count does not match the RGB frame bundle")
    cmd = ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "1600x740",
           "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)]
    writer = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    keyframes = {0, steps - 1, *(int(np.argmin(np.abs(selected["times"] - time))) for time in (3, 3.875, 5, 7, 8, 10))}
    if outcome["first_failure_step"] is not None:
        keyframes.add(outcome["first_failure_step"])
    saved = []
    try:
        for frame, seconds in enumerate(selected["times"]):
            source_frame = int(np.clip(np.floor(seconds * source_fps + .5), 0, len(rgb_paths) - 1))
            raw_path = rgb_paths[source_frame]
            if str(raw_path.resolve()) not in hashes:
                hashes[str(raw_path.resolve())] = sha256(raw_path)
            with Image.open(raw_path) as source:
                rgb = source.convert("RGB")
            rgb.thumbnail((width, height))
            panel = Image.new("RGB", (width, height), (20, 29, 42))
            panel.paste(rgb, ((width - rgb.width) // 2, (height - rgb.height) // 2))
            board = Image.new("RGB", (1600, 740), (11, 17, 27))
            draw = ImageDraw.Draw(board)
            title_color = (133, 230, 166) if status == "SUCCESS" else (255, 151, 145) if status == "FAILED" else (250, 208, 112)
            draw.text((16, 8), f"{status_caption} | {controller} | {selection_prefix} world {selected_world} / 共{selected['manifest']['world_count']}组（0起始编号）", font=font, fill=title_color)
            time_label = "已验证源参考" if selected["timing"]["verified"] else "名义进度（非源时间）"
            draw.text((16, 42), f"{time_label} {seconds:.3f}s / 仿真 {(frame + 1) / fps:.2f}s | {speed:g}倍 | 帧 {frame + 1}/{steps} | {cohort_summary} | 物体辅助=0", font=small, fill="white")
            rgb_label = "最近源帧" if selected["timing"]["verified"] else "估算帧（时间未验证）"
            draw.text((16, 73), f"原始RGB · {rgb_label} {source_frame}", font=small, fill=(220, 228, 239))
            gain = selected["manifest"].get("wrist_gain_ablation")
            scene_label = "实际逐步机器人/物体状态"
            if gain:
                scene_label += f" · kp{gain['kp']:g} kd{gain['kd']:g} · 腕力上限{gain['effort_limit_N']:g}N"
            draw.text((816, 73), scene_label, font=small, fill=(220, 228, 239))
            board.paste(panel, (0, 102))
            board.paste(scene(frame), (800, 102))
            errors = selected["data"]["tracking_error"][frame, selected_world]
            failed = outcome["first_failure_step"] is not None and frame >= outcome["first_failure_step"]
            limits = selected["criteria"]["thresholds"]
            position_limit = f"{limits[2] * 100:g}cm" if limits[2] is not None else "停用"
            orientation_limit = f"{np.rad2deg(limits[3]):.1f}°" if limits[3] is not None else "停用"
            draw.text((16, 684), f"误差：物体位置 {errors[2] * 100:.2f}cm（阈值{position_limit}）；姿态 {np.rad2deg(errors[3]):.1f}°（阈值{orientation_limit}）；腕位 {errors[0] * 100:.2f}cm", font=small, fill="white")
            explanation = ("已越任务阈值；保留完整轨迹。" if failed else "本环境完整动作最终通过所记录任务判据。" if status == "SUCCESS" else "本环境完整动作最终失败。" if status == "FAILED" else "缺逐环境完成/判据凭据，不能声称成功。")
            draw.text((16, 710), (display_note or explanation) + " 连杆凸包显示；桌面高度/厚度为近似。", font=small, fill=(189, 203, 224))
            writer.stdin.write(np.asarray(board).tobytes())
            if frame in keyframes:
                path = output / f"keyframe-{frame:03d}.jpg"
                board.save(path, quality=94)
                saved.append(path.name)
            if frame % 50 == 0:
                print(f"rendered {frame}/{steps}", flush=True)
    finally:
        writer.stdin.close()
        returncode = writer.wait()
    if returncode:
        raise RuntimeError(f"ffmpeg exited {returncode}")
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"], check=True)
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries",
                       "stream=width,height,nb_frames,r_frame_rate,duration", "-of", "json", str(video)]))
    assert probe["streams"][0]["nb_frames"] == str(steps)
    plot_environment = {**os.environ, "MPLCONFIGDIR": str(output / "matplotlib-cache")}
    plot_command = [str(plot_python), str(Path(__file__).resolve()), "plot", "--root", str(root), "--output", str(output)]
    plot_command.extend(["--timeline", str(timeline_path)])
    if diagnostic is not None:
        plot_command.extend(["--diagnostic", str(diagnostic), "--world", str(selected_world)])
    subprocess.run(plot_command, check=True, env=plot_environment)
    if not all(sha256(Path(path)) == digest for path, digest in hashes.items()):
        raise RuntimeError("an input changed during rendering")
    outputs = {path.name: sha256(path) for path in output.iterdir() if path.is_file()}
    manifest = {"schema": "nvidia_v2d.diagnostic_visuals.v2", "claim": controller,
                "selection": selection, "learned_policy": learned, "selected_world_outcome": outcome,
                "selection_post_hoc": selection_post_hoc,
                "display_note": display_note,
                "policy_unassisted_success": bool(learned and outcome["success"] is True),
                "cohort_metrics": {"verified_chord_successful_worlds": successful_worlds,
                                   "world_count": selected["manifest"]["world_count"],
                                   "reported_chord_sr": selected["metrics"]["metrics"]["chord_sr"],
                                   "reported_maniptrans_sr": maniptrans_sr},
                "all_world_outcomes": selected["outcomes"],
                "selected_case": selected["directory"], "selected_world": selected_world, "frames": steps, "fps": fps,
                "playback": f"{fps:g}fps simulator time; each pose is post-step; {selected['timing']['label']}; nearest or explicitly estimated source RGB frame",
                "reference_timing": selected["timing"], "timeline_file": timeline_path.name,
                "rgb_frame_selection": {"fps": source_fps, "frame_count": len(rgb_paths),
                                        "verified_source_timestamps": selected["timing"]["verified"]},
                "lift_metric": "maximum over reset and recorded steps relative to actual reset Z; signed after-step maximum also retained",
                "object_pose_matches_recorded_body_pose_exactly": True, "auxiliary_scale_max": 0,
                "reference_tracking_criteria": selected["criteria"],
                "fixed_camera": {"world_projected_uv_bounds_m": bounds, "center_uv_m": center_uv.tolist(),
                                 "pixels_per_m": camera_scale, "fit_scope": "all displayed geometry across the complete selected recorded rollout",
                                 "per_frame_camera_changes": False},
                "render_simplifications": ["robot visual link convex hulls", "fixed orthographic camera",
                                          "finite tabletop from existing candidate; not measured ground truth"],
                "source_sha256": hashes, "source_unchanged": True, "outputs_sha256": outputs,
                "keyframes": saved, "video_probe": probe,
                "comparison_note": "individual world outcome is not the cohort SR; default legacy comparison also differs in motion speed"}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "video": video.name, "frames": steps, "world_rows": len(rows)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("render", "plot"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--plot-python", type=Path)
    parser.add_argument("--diagnostic", type=Path, help="One result directory containing diagnostic.npz/manifest/metrics; omit for the legacy four-case comparison")
    parser.add_argument("--timeline", type=Path, help="plot only: reuse a render's capture-bound timeline.json across Python environments")
    parser.add_argument("--world", type=int, default=0, help="Explicit captured world index; no automatic success selection")
    parser.add_argument("--selection-note", help="Explain any post-hoc selection; recorded in the output manifest")
    parser.add_argument("--selection-post-hoc", action="store_true", help="Explicitly label a world selected after examining outcomes")
    parser.add_argument("--display-note", help="Short visible caption documenting selection/release qualifications; also saved in manifest")
    args = parser.parse_args()
    if args.mode == "plot":
        plot(args.root, args.output, args.diagnostic, args.world, args.timeline)
    else:
        if args.assets is None or args.plot_python is None:
            parser.error("render requires --assets and --plot-python")
        render(args.root.resolve(), args.output.resolve(), args.assets.resolve(), args.plot_python.resolve(),
               args.diagnostic.resolve() if args.diagnostic else None, args.world, args.selection_note,
               args.selection_post_hoc, args.display_note)


if __name__ == "__main__":
    main()
