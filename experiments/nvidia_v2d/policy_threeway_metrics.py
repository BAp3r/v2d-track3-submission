"""Compare achieved object rollouts without gating training on grasp appearance.

The independent object-only metric uses every recorded frame and world, with
unchanged 0.1 m / 0.5 rad thresholds. Official harness results remain separate.
Geometry identity and embedded object reference/time axes must match before
three retargeting methods can be compared. No reference pose substitutes for
an achieved pose, and no failed or invalid world is dropped.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

POSITION_THRESHOLD_M = 0.1
ORIENTATION_THRESHOLD_RAD = 0.5
METHODS = frozenset({"official_ik", "dexpilot_chain", "multi_tip"})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True)
class ArmSpec:
    name: str
    trajectory: Path
    official_metrics: Path
    geometry_factor: str
    geometry_identity: dict[str, Any]
    evaluation_evidence: Path | None = None
    source_times: Path | None = None
    auxiliary_quality: Path | None = None

    @classmethod
    def from_dict(cls, value: dict, base: Path = Path(".")) -> ArmSpec:
        data = dict(value)
        for key in ("trajectory", "official_metrics", "evaluation_evidence", "source_times", "auxiliary_quality"):
            if data.get(key) is not None:
                path = Path(data[key])
                data[key] = path if path.is_absolute() else base / path
        return cls(**data)


@dataclass
class Trajectory:
    metadata: dict[str, str]
    achieved: np.ndarray  # [T,W,B,7], xyz + xyzw
    reference: np.ndarray  # [T,B,7]
    environment_valid: np.ndarray  # [T,W,B]
    time_s: np.ndarray
    reference_frame: np.ndarray
    object_ids: np.ndarray
    body_names: list[str]
    source_time_s: np.ndarray | None = None


def _canonical_quaternion(q: np.ndarray) -> np.ndarray:
    """Canonicalize sign only; this is used for strict reference equality."""
    q = q.copy()
    first = np.argmax(np.abs(q) > 1e-12, axis=-1)
    sign = np.take_along_axis(q, first[..., None], axis=-1)[..., 0] < 0
    q[sign] *= -1
    q[q == 0] = 0  # canonical positive zero
    return q


def load_trajectory(path: Path, source_times: Path | None = None) -> Trajectory:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    meta = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    fixed = {"schema": "flash_chord_object_trajectories_v1", "position_unit": "meter",
             "quaternion_order": "xyzw", "row_order": "step,environment_id,object_body_id",
             "start_frame": "0", "pose_width": "7"}
    if any(meta.get("flash_chord." + k) != v for k, v in fixed.items()):
        raise ValueError("Trajectory schema, units, order, or full-sequence origin mismatch")
    try:
        t, w, b = (int(meta["flash_chord." + k]) for k in ("step_count", "world_count", "object_body_count"))
        fps = float(meta["flash_chord.control_fps"])
    except (KeyError, ValueError) as exc:
        raise ValueError("Missing trajectory dimensions or control frequency") from exc
    if min(t, w, b) < 1 or not np.isfinite(fps) or fps <= 0 or len(table) != t * w * b:
        raise ValueError("Incomplete trajectory grid")
    axes = ["position_" + a for a in "xyz"] + ["quaternion_" + a for a in "xyzw"]
    needed = ["environment_id", "step", "reference_frame", "time_s", "object_id",
              "object_body_id", "object_body_name", "environment_valid"]
    needed += [prefix + "_" + a for prefix in ("achieved", "reference") for a in axes]
    if any(k not in table.column_names or table[k].null_count for k in needed):
        raise ValueError("Missing or null achieved/reference fields")
    data = {k: np.asarray(table[k].to_pylist()) for k in needed}
    expected = {
        "step": np.repeat(np.arange(t), w * b),
        "environment_id": np.tile(np.repeat(np.arange(w), b), t),
        "object_body_id": np.tile(np.arange(b), t * w),
    }
    for key, indices in expected.items():
        if not np.issubdtype(data[key].dtype, np.integer) or not np.array_equal(data[key], indices):
            raise ValueError(f"Missing, duplicate, reordered or invalid {key} rows")
    if not np.array_equal(data["reference_frame"], expected["step"]):
        raise ValueError("Reference frame axis is not the complete zero-origin sequence")
    times = np.arange(t, dtype=float) / fps
    if not np.allclose(data["time_s"], np.repeat(times, w * b), rtol=0, atol=1e-10):
        raise ValueError("Control time axis mismatch")
    if data["environment_valid"].dtype != np.bool_:
        raise ValueError("environment_valid must be explicit booleans")
    poses = {prefix: np.column_stack([data[prefix + "_" + a] for a in axes]).astype(float).reshape(t, w, b, 7)
             for prefix in ("achieved", "reference")}
    ref = poses["reference"]
    if not np.isfinite(ref).all() or np.max(np.abs(np.linalg.norm(ref[..., 3:], axis=-1) - 1)) > 1e-3:
        raise ValueError("Invalid reference poses or quaternions")
    if not np.array_equal(ref, np.broadcast_to(ref[:, :1], ref.shape)):
        raise ValueError("Different object references across worlds")
    names, objects = [], []
    if not np.issubdtype(data["object_id"].dtype, np.integer):
        raise ValueError("Invalid object IDs")
    for body in range(b):
        selected = data["object_body_id"] == body
        n = np.unique(data["object_body_name"][selected])
        ids = np.unique(data["object_id"][selected])
        if len(n) != 1 or not n[0] or len(ids) != 1 or ids[0] < 0:
            raise ValueError("Changing object identity within a trajectory")
        names.append(str(n[0])); objects.append(int(ids[0]))
    if len(set(names)) != b:
        raise ValueError("Object body names must be unique")
    source = None
    if source_times is not None:
        source = np.asarray(np.load(source_times, allow_pickle=False), dtype=float)
        if source.shape != (t,) or not np.isfinite(source).all() or np.any(np.diff(source) <= 0):
            raise ValueError("Explicit source time array must be finite, increasing, and match all frames")
    return Trajectory(meta, poses["achieved"], ref[:, 0], data["environment_valid"].reshape(t, w, b),
                      times, np.arange(t), np.asarray(objects), names, source)


def pose_errors(achieved: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Float64, uncentered position and sign-invariant SO(3) shortest angle."""
    norms = np.linalg.norm(achieved[..., 3:], axis=-1)
    valid = np.isfinite(achieved).all(axis=-1) & (np.abs(norms - 1) <= 1e-3)
    safe = np.where(valid[..., None], achieved, np.r_[np.zeros(6), 1.])
    q = safe[..., 3:] / np.linalg.norm(safe[..., 3:], axis=-1, keepdims=True)
    r = reference[..., 3:] / np.linalg.norm(reference[..., 3:], axis=-1, keepdims=True)
    relative_vector = q[..., 3:] * r[..., :3] - r[..., 3:] * q[..., :3] - np.cross(q[..., :3], r[..., :3])
    dot = np.sum(q * r, axis=-1)
    angle = 2 * np.arctan2(np.linalg.norm(relative_vector, axis=-1), np.abs(dot))
    distance = np.linalg.norm(safe[..., :3] - reference[..., :3], axis=-1)
    return np.where(valid, distance, np.nan), np.where(valid, angle, np.nan), valid


def _first(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    return int(indices[0]) if len(indices) else None


def _number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _stats(values: np.ndarray) -> dict[str, float | int | None]:
    finite = values[np.isfinite(values)]
    return {"finite_count": len(finite), "total_count": values.size,
            "mean": float(finite.mean()) if len(finite) and len(finite) == values.size else None,
            "finite_observed_mean": float(finite.mean()) if len(finite) else None,
            "max": float(finite.max()) if len(finite) else None}


def measure(data: Trajectory) -> dict:
    ref = data.reference[:, None]
    position, orientation, finite = pose_errors(data.achieved, ref)
    measurable = finite & data.environment_valid
    # The harness may replace non-finite worlds with finite zero-filled poses.
    # Keep the raw arrays, but never interpret those placeholders as tracking.
    position = np.where(measurable, position, np.nan)
    orientation = np.where(measurable, orientation, np.nan)
    threshold_pass = finite & (position <= POSITION_THRESHOLD_M) & (orientation <= ORIENTATION_THRESHOLD_RAD)
    passed = threshold_pass & data.environment_valid
    t, w, b = position.shape
    world_pass = passed.all(axis=(0, 2))
    initial_z = data.achieved[0, :, :, 2]
    lift = data.achieved[..., 2] - initial_z[None]
    lift = np.where(measurable & measurable[:1], lift, np.nan)
    worlds, bodies = [], []
    for world in range(w):
        first_position = _first(np.any(finite[:, world] & (position[:, world] > POSITION_THRESHOLD_M), axis=1))
        first_angle = _first(np.any(finite[:, world] & (orientation[:, world] > ORIENTATION_THRESHOLD_RAD), axis=1))
        first_invalid = _first(~finite[:, world].all(axis=1))
        known = [i for i in (first_position, first_angle, first_invalid) if i is not None]
        first_failure = min(known) if known else None
        record = {"environment_id": world, "object_only_success": bool(world_pass[world]),
                  "first_failure_step": first_failure,
                  "first_failure_control_time_s": None if first_failure is None else float(data.time_s[first_failure]),
                  "first_failure_source_time_s": None if first_failure is None or data.source_time_s is None else float(data.source_time_s[first_failure]),
                  "first_position_failure_step": first_position, "first_orientation_failure_step": first_angle,
                  "first_invalid_pose_step": first_invalid,
                  "environment_flagged_invalid": bool((~data.environment_valid[:, world]).any()),
                  "failure_timing_known": bool(world_pass[world] or first_failure is not None),
                  "max_position_error_m": _stats(position[:, world])["max"],
                  "mean_position_error_m": _stats(position[:, world])["mean"],
                  "max_orientation_error_rad": _stats(orientation[:, world])["max"],
                  "mean_orientation_error_rad": _stats(orientation[:, world])["mean"],
                  "max_actual_lift_m": _stats(lift[:, world])["max"]}
        worlds.append(record)
        for body in range(b):
            values = lift[:, world, body]
            peak = int(np.nanargmax(values)) if np.isfinite(values).any() else None
            before = values[:first_failure] if first_failure is not None else values
            bodies.append({"environment_id": world, "object_body_id": body, "object_body_name": data.body_names[body],
                           "object_only_success": bool(passed[:, world, body].all()),
                           "initial_achieved_z_m": _number(initial_z[world, body]) if measurable[0, world, body] else None,
                           "max_actual_lift_m": None if peak is None else float(values[peak]),
                           "peak_lift_step": peak, "peak_lift_control_time_s": None if peak is None else float(data.time_s[peak]),
                           "peak_lift_after_first_failure": None if peak is None or first_failure is None else peak >= first_failure,
                           "max_actual_lift_before_first_failure_m": _stats(before)["max"]})
    curves = []
    for step in range(t):
        for body in range(b):
            row = {"step": step, "reference_frame": int(data.reference_frame[step]),
                   "control_time_s": float(data.time_s[step]),
                   "source_time_s": None if data.source_time_s is None else float(data.source_time_s[step]),
                   "object_body_id": body, "object_body_name": data.body_names[body],
                   "instant_object_pass_fraction": float(passed[step, :, body].mean()),
                   "valid_pose_world_count": int(finite[step, :, body].sum()),
                   "measurable_world_count": int(measurable[step, :, body].sum())}
            for name, values in (("position_error_m", position), ("orientation_error_rad", orientation), ("actual_lift_m", lift)):
                v = values[step, :, body]; f = v[np.isfinite(v)]
                row.update({name + "_mean": _stats(v)["mean"], name + "_max": _stats(v)["max"],
                            name + "_median": float(np.median(f)) if len(f) else None,
                            name + "_p90": float(np.percentile(f, 90)) if len(f) else None})
            curves.append(row)
    return {"summary": {"world_count": w, "frame_count": t, "body_count": b,
                         "object_only_success_count": int(world_pass.sum()), "object_only_success_rate": float(world_pass.mean()),
                         "successful_worlds": np.flatnonzero(world_pass).tolist(),
                         "position_error_m": _stats(position), "orientation_error_rad": _stats(orientation),
                         "invalid_pose_rows": int((~finite).sum()),
                         "masked_invalid_metric_rows": int((~measurable).sum()),
                         "flagged_invalid_worlds": np.flatnonzero((~data.environment_valid).any(axis=(0, 2))).tolist()},
            "worlds": worlds, "bodies": bodies, "curves": curves, "position": position,
            "orientation": orientation, "lift": lift, "pose_valid": finite, "passed": passed}


def _load_arm(spec: ArmSpec) -> tuple[Trajectory, dict, dict | None, dict]:
    if not spec.geometry_factor or not spec.geometry_identity or not isinstance(spec.geometry_identity, dict):
        raise ValueError("Explicit nonempty geometry factor and identity are required")
    canonical(spec.geometry_identity)
    data = load_trajectory(spec.trajectory, spec.source_times)
    official = json.loads(spec.official_metrics.read_text())
    invalid_worlds = official.get("non_finite_worlds", [])
    if not isinstance(invalid_worlds, list) or any(type(i) is not int or i not in range(data.achieved.shape[1]) for i in invalid_worlds):
        raise ValueError("Invalid official non_finite_worlds list")
    # This independent signal protects against a zero-filled or incorrectly
    # flagged trajectory. It cannot improve any world's result.
    data.environment_valid[:, invalid_worlds, :] = False
    for key, expected in (("world_count", data.achieved.shape[1]), ("step_count", len(data.time_s))):
        observed = official.get(key, official.get("metrics", {}).get(key))
        if observed != expected:
            raise ValueError("Official harness dimensions do not match the trajectory")
    for key, expected in (("object_position_threshold_m", POSITION_THRESHOLD_M), ("object_orientation_threshold_rad", ORIENTATION_THRESHOLD_RAD)):
        if key in official and official[key] != expected:
            raise ValueError("Official thresholds were changed")
    checkpoint = data.metadata.get("flash_chord.checkpoint_sha256")
    if checkpoint and official.get("checkpoint_sha256") != checkpoint:
        raise ValueError("Official metrics and achieved trajectory use different checkpoints")
    evidence = None
    if spec.evaluation_evidence:
        evidence = json.loads(spec.evaluation_evidence.read_text())
        observations = evidence.get("observed_control_scales", [])
        if (evidence.get("trajectory_sha256") != sha256(spec.trajectory)
                or evidence.get("assistance") != "none" or len(observations) != len(data.time_s)
                or any(r.get("voc_scale_max_abs") != 0 or r.get("applied_object_control_scale_max_abs") != 0 for r in observations)):
            raise ValueError("Evidence does not establish all-frame zero assistance for this trajectory")
        support = spec.geometry_identity.get("support_sha256")
        if support and evidence.get("scene_context", {}).get("support_sha256") != support:
            raise ValueError("Geometry support identity disagrees with executed scene")
    auxiliary = {}
    if spec.auxiliary_quality:
        record = json.loads(spec.auxiliary_quality.read_text())
        if record.get("trajectory_sha256") != sha256(spec.trajectory):
            raise ValueError("Auxiliary quality comes from a different cohort")
        for row in record.get("worlds", []):
            world = row.get("environment_id")
            if not isinstance(world, int) or world in auxiliary or world not in range(data.achieved.shape[1]):
                raise ValueError("Invalid or duplicated auxiliary world ID")
            auxiliary[world] = {"aux_" + k: v for k, v in row.items() if k != "environment_id"}
    return data, official, evidence, auxiliary


def validate_comparison(specs: list[ArmSpec], data: list[Trajectory]) -> None:
    if len(specs) != 3 or {s.name for s in specs} != METHODS:
        raise ValueError("Comparison requires official_ik, dexpilot_chain and multi_tip exactly once")
    baseline, initial = specs[0], data[0]
    for spec, item in zip(specs[1:], data[1:]):
        if spec.geometry_factor != baseline.geometry_factor or canonical(spec.geometry_identity) != canonical(baseline.geometry_identity):
            raise ValueError("Different geometry factors or identities cannot be combined")
        if item.achieved.shape != initial.achieved.shape or item.body_names != initial.body_names or not np.array_equal(item.object_ids, initial.object_ids):
            raise ValueError("Different world counts, horizons or object identities")
        if not np.array_equal(item.time_s, initial.time_s) or not np.array_equal(item.reference_frame, initial.reference_frame):
            raise ValueError("Different control/reference time axes")
        if (item.source_time_s is None) != (initial.source_time_s is None) or (item.source_time_s is not None and not np.array_equal(item.source_time_s, initial.source_time_s)):
            raise ValueError("Different or partially supplied source time axes")
        if not np.array_equal(item.reference[..., :3], initial.reference[..., :3]) or not np.array_equal(_canonical_quaternion(item.reference[..., 3:]), _canonical_quaternion(initial.reference[..., 3:])):
            raise ValueError("Different embedded object references")


def _csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def _write_arm(spec: ArmSpec, loaded: tuple, output: Path) -> dict:
    data, official, evidence, auxiliary = loaded
    result = measure(data); output.mkdir(parents=True)
    for row in result["worlds"]:
        row.update(auxiliary.get(row["environment_id"], {}))
    _csv(output / "worlds.csv", result["worlds"]); _csv(output / "bodies.csv", result["bodies"])
    _csv(output / "curves.csv", result["curves"])
    frames = []
    for step, world, body in np.ndindex(result["position"].shape):
        row = {"step": step, "environment_id": world, "object_body_id": body,
               "control_time_s": float(data.time_s[step]),
               "source_time_s": None if data.source_time_s is None else float(data.source_time_s[step]),
               "environment_valid": bool(data.environment_valid[step, world, body]),
               "pose_valid": bool(result["pose_valid"][step, world, body]),
               "position_error_m": _number(result["position"][step, world, body]),
               "orientation_error_rad": _number(result["orientation"][step, world, body]),
               "actual_lift_m": _number(result["lift"][step, world, body]),
               "instant_object_pass": bool(result["passed"][step, world, body])}
        frames.append(row)
    _csv(output / "frames.csv", frames)
    np.savez_compressed(output / "traces.npz", achieved_pose_xyzw=data.achieved, reference_pose_xyzw=data.reference,
                        control_time_s=data.time_s, reference_frame=data.reference_frame,
                        source_time_s=np.array([]) if data.source_time_s is None else data.source_time_s,
                        position_error_m=result["position"], orientation_error_rad=result["orientation"],
                        actual_lift_m=result["lift"], environment_valid=data.environment_valid,
                        pose_valid=result["pose_valid"], instant_object_pass=result["passed"])
    shutil.copyfile(spec.official_metrics, output / "official-harness-metrics.json")
    inputs = [spec.trajectory, spec.official_metrics, Path(__file__)]
    for key in ("evaluation_evidence", "source_times", "auxiliary_quality"):
        value = getattr(spec, key)
        if value:
            inputs.append(value); shutil.copyfile(value, output / (key + value.suffix))
    summary = {"schema": "nvidia_v2d.threeway_object_rollout.v1", "name": spec.name,
               "geometry_factor": spec.geometry_factor, "geometry_identity": spec.geometry_identity,
               "geometry_identity_sha256": hashlib.sha256(canonical(spec.geometry_identity)).hexdigest(),
               "pose_source": "achieved_* parquet fields only", "assistance": "none_verified" if evidence else "unknown",
               "thresholds": {"position_m": POSITION_THRESHOLD_M, "orientation_rad": ORIENTATION_THRESHOLD_RAD},
               "threshold_comparison": "unrounded float64 errors <= fixed thresholds; no centering, grace period, smoothing or failed-frame removal",
               "time_s_semantics": "Parquet time_s is control elapsed time; source time only from explicitly supplied array",
               "full_recorded_horizon": True, "training_gate": False,
               "official_harness_metrics": official,
               "auxiliary_quality_policy": "CHORD, settling and semantic contact are additional quality; no cohort mixing or broadcast world labels",
               "auxiliary_world_count": len(auxiliary),
               "invalid_world_policy": "All worlds remain in denominator. Either parquet environment_valid=false or official non_finite_worlds forces failure and masks numerical metrics, retaining raw poses. Global flags do not identify physical failure time.",
               **result["summary"], "worlds": result["worlds"], "bodies": result["bodies"],
               "input_sha256": {str(p.resolve()): sha256(p) for p in inputs}, "trajectory_metadata": data.metadata}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def summarize_arm(spec: ArmSpec, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Output must be new")
    return _write_arm(spec, _load_arm(spec), output)


def compare_arms(specs: list[ArmSpec], output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Output must be new")
    loaded = [_load_arm(s) for s in specs]
    if any(row[2] is None for row in loaded):
        raise ValueError("Formal three-way comparison requires all-frame zero-assistance evidence for every arm")
    validate_comparison(specs, [row[0] for row in loaded])  # before creating any output
    budgets = [row[1].get("environment_steps") for row in loaded]
    if any(type(v) is not int or v < 1 for v in budgets) or len(set(budgets)) != 1:
        raise ValueError("Three-way comparison requires the same recorded training budget")
    summaries = [_write_arm(s, row, output / s.name) for s, row in zip(specs, loaded)]
    result = {"schema": "nvidia_v2d.threeway_comparison.v1", "geometry_factor": specs[0].geometry_factor,
              "geometry_identity": specs[0].geometry_identity, "same_reference_and_time_axes_verified": True,
              "training_gate": False, "environment_steps": budgets[0], "arms": summaries}
    (output / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def plot_results(output: Path) -> list[Path]:
    """Separate plotting entry permits a matplotlib runtime without PyArrow."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = [output] if (output / "summary.json").exists() else [p.parent for p in sorted(output.glob("*/summary.json"))]
    if not paths:
        raise ValueError("No computed rollout summaries")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for index, path in enumerate(paths):
        summary = json.loads((path / "summary.json").read_text()); traces = np.load(path / "traces.npz")
        time = traces["control_time_s"]; color = f"C{index}"
        for ax, key in zip(axes, ("position_error_m", "orientation_error_rad", "actual_lift_m")):
            # Worst body per world, every world and full recorded horizon, including post-failure.
            values = np.max(traces[key], axis=2)
            ax.plot(time, values, color=color, alpha=.18, linewidth=.65)
            median = np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else np.nan for v in values])
            ax.plot(time, median, color=color, linewidth=1.6,
                    label=f"{summary['name']} ({values.shape[1]} worlds; {len(summary['flagged_invalid_worlds'])} invalid)")
        axes[0].set_title("Achieved object rollouts — every world, full horizon; bold = finite median\nInvalid worlds remain failures in the full denominator")
    for ax, threshold, label in zip(axes, (.1, .5, None), ("Position error (m)", "Orientation error (rad)", "Lift from actual initial z (m)")):
        if threshold is not None: ax.axhline(threshold, color="black", linestyle="--", linewidth=.8)
        ax.set_ylabel(label); ax.grid(alpha=.2); ax.legend(loc="upper left")
    axes[-1].set_xlabel("Control elapsed time (s); source time is a separate stored axis")
    fig.tight_layout()
    files = [output / "object-rollout-curves.png", output / "object-rollout-curves.pdf"]
    for path in files: fig.savefig(path, dpi=160)
    plt.close(fig)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("summarize", "compare"):
        p = sub.add_parser(name); p.add_argument("--spec", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("plot"); p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plot":
        print(json.dumps([str(p) for p in plot_results(args.output)])); return
    value = json.loads(args.spec.read_text())
    if args.command == "summarize":
        result = summarize_arm(ArmSpec.from_dict(value, args.spec.parent), args.output)
        print(json.dumps({k: result[k] for k in ("name", "world_count", "object_only_success_count", "object_only_success_rate")}))
    else:
        result = compare_arms([ArmSpec.from_dict(v, args.spec.parent) for v in value["arms"]], args.output)
        print(json.dumps({"geometry_factor": result["geometry_factor"], "arms": [r["name"] for r in result["arms"]]}))


if __name__ == "__main__":
    main()
