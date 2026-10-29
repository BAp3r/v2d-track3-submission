"""Audit sampled physical release, settling and thumb roles independently of CHORD.

Full collision pairs are required, including shapes outside reward contact slots.
Counts establish contact presence, not calibrated force or continuous contact
between the recorded control-step snapshots. No unsuccessful world is discarded.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.nvidia_v2d.policy_diagnostic_visuals import load_cases, sha256


def physical_quality(data, manifest, support, times, outcomes, *, window_s=1.0,
                     max_linear_speed=.02, max_angular_speed=.25, max_displacement=.005):
    """Return per-world outcomes and full time series; thresholds use SI units."""
    labels = manifest["body_labels"]
    objects = [i for i, name in enumerate(labels) if name.endswith("/object")]
    if len(objects) != 1 or "support" not in labels:
        raise ValueError("quality audit requires one object and an explicit finite support")
    obj, table = objects[0], labels.index("support")
    steps, worlds = data["tracking_error"].shape[:2]
    count = len(labels)
    fps = float(manifest["fps"])
    samples = round(window_s * fps) + 1
    if samples < 2 or samples > steps or len(times) != steps or len(outcomes) != worlds:
        raise ValueError("invalid final quality window or cohort dimensions")
    auxiliary = np.asarray(data["auxiliary_scales"])
    if not np.isfinite(auxiliary).all() or auxiliary.any():
        raise ValueError("physical unassisted audit requires recorded zero auxiliary control")
    offsets = np.asarray(data["contact_frame_offsets"])
    b0, b1 = [np.asarray(data[f"contacts_body{i}_ids"]) for i in (0, 1)]
    if (len(offsets) != steps + 1 or offsets[0] != 0 or offsets[-1] != len(b0)
            or len(b1) != len(b0) or np.any(np.diff(offsets) < 0)):
        raise ValueError("incomplete full contact capture")
    step = np.repeat(np.arange(steps), np.diff(offsets))
    object0 = (b0 >= 0) & (b0 % count == obj)
    object1 = (b1 >= 0) & (b1 % count == obj)
    involved = object0 | object1
    other = np.where(object0, b1, b0)
    world = np.where(object0, b0, b1) // count
    if np.any((other[involved] < 0) | (other[involved] // count != world[involved])):
        raise ValueError("object contact crosses worlds or touches an undeclared infinite ground")
    hand_ids = [i for i, name in enumerate(labels) if name.startswith(("left_", "right_"))]
    hand_pair = involved & np.isin(other % count, hand_ids)
    table_pair = involved & (other % count == table)
    normal = np.asarray(data["contacts_contact_normal_w"]) * np.where(object0, -1, 1)[:, None]
    points = np.asarray(data["contacts_contact_pos_w"])
    center, radius, top = np.asarray(support["center_xy_m"]), float(support["radius_m"]), float(support["top_m"])
    if (center.shape != (2,) or not np.isfinite([*center, radius, top]).all() or radius <= 0):
        raise ValueError("invalid finite support geometry")
    top_pair = (table_pair & (normal[:, 2] > .5) & (np.abs(points[:, 2] - top) <= .01)
                & (np.linalg.norm(points[:, :2] - center, axis=-1) <= radius + .01))
    groups = {"all_hand": hand_pair, "table": table_pair, "table_top": top_pair}
    for side in ("left", "right"):
        for part in ("MC", "PP", "DP"):
            ids = [i for i, name in enumerate(labels) if name.endswith(f"/{side}_thumb_{part}")]
            if len(ids) != 1:
                raise ValueError(f"ambiguous thumb role {side}/{part}")
            groups[f"{side}_thumb_{part}"] = involved & (other % count == ids[0])
    counts = {}
    for name, mask in groups.items():
        values = np.zeros((steps, worlds), dtype=np.int32)
        np.add.at(values, (step[mask], world[mask]), 1)
        counts[name] = values
    pose = np.asarray(data["achieved_pose_w"])[:, :, 0]
    velocity = np.asarray(data["actual_body_velocities"])[:, :, obj]
    finite = np.isfinite(pose).all(axis=(0, 2)) & np.isfinite(velocity).all(axis=(0, 2))
    tail = slice(steps - samples, steps)
    on_table = ((np.linalg.norm(pose[..., :2] - center, axis=-1) < radius)
                & (pose[..., 2] >= top))
    release = ((counts["all_hand"][tail] == 0).all(0) & (counts["table_top"][tail] > 0).all(0)
               & on_table[tail].all(0) & finite)
    linear = np.linalg.norm(velocity[tail, :, :3], axis=-1).max(0)
    angular = np.linalg.norm(velocity[tail, :, 3:], axis=-1).max(0)
    displacement = np.linalg.norm(pose[tail, :, :3] - pose[steps-samples, :, :3], axis=-1).max(0)
    settled = (release & (linear <= max_linear_speed) & (angular <= max_angular_speed)
               & (displacement <= max_displacement))
    carry = (np.asarray(times) >= 4.) & (np.asarray(times) <= 8.)
    if not carry.any():
        raise ValueError("capture does not contain declared carrying window source 4–8 s")
    rows = []
    for w in range(worlds):
        row = {"world": w, "chord_success": outcomes[w]["success"] is True,
               "sampled_release": bool(release[w]), "sampled_settled": bool(settled[w]),
               "chord_and_release": bool(outcomes[w]["success"] is True and release[w]),
               "chord_release_and_settled": bool(outcomes[w]["success"] is True and settled[w]),
               "final_window_max_linear_speed_m_s": float(linear[w]),
               "final_window_max_angular_speed_rad_s": float(angular[w]),
               "final_window_max_displacement_m": float(displacement[w]),
               "full_trajectory_finite": bool(finite[w])}
        for side in ("left", "right"):
            for part in ("MC", "PP", "DP"):
                row[f"carry_{side}_thumb_{part}_contact_fraction"] = float(np.mean(counts[f"{side}_thumb_{part}"][carry, w] > 0))
            distal = (counts[f"{side}_thumb_PP"] + counts[f"{side}_thumb_DP"]) > 0
            row[f"carry_{side}_thumb_PP_or_DP_contact_fraction"] = float(np.mean(distal[carry, w]))
        rows.append(row)
    summary = {"world_count": worlds, "criteria": {"final_window_s": window_s, "samples": samples,
               "max_linear_speed_m_s": max_linear_speed, "max_angular_speed_rad_s": max_angular_speed,
               "max_displacement_m": max_displacement, "carrying_source_window_s": [4., 8.],
               "force_amplitudes_used": False, "scope": "recorded control snapshots; not all physical substeps"}}
    for key in ("chord_success", "sampled_release", "sampled_settled", "chord_and_release", "chord_release_and_settled"):
        summary[key + "_worlds"] = [r["world"] for r in rows if r[key]]
        summary[key + "_count"] = len(summary[key + "_worlds"])
    return rows, summary, {**counts, "source_time_s": np.asarray(times), "actual_pose": pose,
                           "actual_velocity": velocity, "center_over_table": on_table}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--support-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--environment-steps", type=int)
    args = parser.parse_args()
    cases, hashes = load_cases(args.root.resolve(), args.capture.resolve())
    case = cases[0]
    if not case["timing"]["verified"]:
        raise ValueError("task quality needs verified source times")
    support = json.loads(args.support_metadata.read_text())
    if "fit" in support:
        support = {**support["fit"], "top_m": support["top_m"]}
    rows, summary, traces = physical_quality(case["data"], case["manifest"], support,
                                              case["times"], case["outcomes"])
    args.output.mkdir(parents=True, exist_ok=False)
    summary.update(capture=str(args.capture.resolve()), checkpoint_sha256=case["manifest"]["checkpoint_sha256"],
                   timing=case["timing"], original_chord_criteria=case["criteria"])
    hashes[str(args.support_metadata.resolve())] = sha256(args.support_metadata)
    hashes[str(Path(__file__).resolve())] = sha256(Path(__file__))
    for name, content in (("summary.json", summary), ("all-worlds.json", rows), ("input-sha256.json", hashes)):
        (args.output / name).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    with (args.output / "all-worlds.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(args.output / "physical-traces.npz", **traces)
    if args.metrics:
        from experiments.nvidia_v2d.policy_metric_logging import JsonMetricLogger
        if args.environment_steps is None:
            parser.error("--metrics requires --environment-steps")
        logger = JsonMetricLogger(args.metrics, context={"phase": "evaluation", "series": "physical_quality",
            "checkpoint_sha256": summary["checkpoint_sha256"], "capture": summary["capture"]})
        try:
            logger.log({f"evaluation/physical_quality/{k.removesuffix('_count')}": v / len(rows)
                        for k, v in summary.items() if k.endswith("_count") and k != "world_count"}, args.environment_steps)
        finally:
            logger.close()
    print(json.dumps({k: v for k, v in summary.items() if k.endswith("_count")}, indent=2))


if __name__ == "__main__":
    main()
