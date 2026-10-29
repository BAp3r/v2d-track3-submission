"""Bounded ray-only initial-wrist diagnostic; never a training admission gate.

Fix frame zero and frames >= end_frame, and change only robot translations
before any source contact. The full hand does NOT preserve its projection just
because its wrist stays on the same camera ray. Report that tradeoff explicitly.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear


def camera_points(world, camera_to_scene):
    points, camera = np.asarray(world, float), np.asarray(camera_to_scene, float)
    inverse_rotation = np.linalg.inv(camera[:, :3, :3])
    if points.ndim == 2:
        return np.einsum("tij,tj->ti", inverse_rotation, points - camera[:, :3, 3])
    return np.einsum("tij,tkj->tki", inverse_rotation, points - camera[:, None, :3, 3])


def project_world(world, camera_to_scene, intrinsics):
    points = camera_points(world, camera_to_scene)
    if np.any(points[..., 2] <= 0):
        raise ValueError("Projection requires positive camera depth")
    k = np.asarray(intrinsics, float)
    return points[..., :2] / points[..., 2:] * k[:2] + k[2:]


def ray_smooth_candidate(
    wrist_world,
    camera_to_scene,
    source_contact_active,
    *,
    end_frame=10,
    fidelity=10.0,
    max_translation_m=0.1,
):
    """Return a new wrist array and world delta; inputs are never changed.

    Least squares minimizes world second differences (including the unchanged
    outgoing frame), plus fidelity * squared camera-depth corrections. Bounds
    limit Euclidean world translation. Frames 0 and end_frame onward are fixed.
    """
    wrist, camera, active = map(
        np.asarray, (wrist_world, camera_to_scene, source_contact_active)
    )
    if (
        wrist.ndim != 2
        or wrist.shape[1:] != (3,)
        or camera.shape != (len(wrist), 4, 4)
        or active.shape != (len(wrist),)
        or active.dtype.kind != "b"
        or type(end_frame) is not int
        or not 2 <= end_frame < len(wrist) - 1
        or not np.isfinite(wrist).all()
        or not np.isfinite(camera).all()
        or not np.isfinite(fidelity)
        or fidelity < 0
        or not np.isfinite(max_translation_m)
        or max_translation_m <= 0
    ):
        raise ValueError("Invalid bounded ray diagnostic inputs")
    rotation = camera[:, :3, :3]
    if not np.allclose(
        np.swapaxes(rotation, 1, 2) @ rotation, np.eye(3), atol=1e-6
    ) or not np.allclose(np.linalg.det(rotation), 1, atol=1e-6):
        raise ValueError("Camera transforms must be rigid rotations")
    if active[: end_frame + 1].any():
        raise ValueError("Initial diagnostic may not modify a source contact interval")
    local = camera_points(wrist, camera)
    if np.any(local[:, 2] <= 0):
        raise ValueError("Wrist camera depth must be positive")
    direction = np.einsum("tij,tj->ti", rotation, local / local[:, 2:])
    indices = np.arange(1, end_frame)
    design = np.zeros((end_frame + 2, 3, len(indices)))
    for column, frame in enumerate(indices):
        design[frame, :, column] = direction[frame]
    matrix = np.diff(design, n=2, axis=0).reshape(-1, len(indices))
    target = -np.diff(wrist[: end_frame + 2], n=2, axis=0).reshape(-1)
    limit = max_translation_m / np.linalg.norm(direction[indices], axis=1)
    solution = lsq_linear(
        np.vstack((matrix, np.eye(len(indices)) * np.sqrt(fidelity))),
        np.r_[target, np.zeros(len(indices))],
        bounds=(-limit, limit),
    )
    if not solution.success or not np.isfinite(solution.x).all():
        raise ValueError("Ray-only fit did not converge")
    delta = np.zeros_like(wrist, dtype=float)
    delta[indices] = direction[indices] * solution.x[:, None]
    candidate = wrist + delta
    if np.any(camera_points(candidate, camera)[:, 2] <= 0):
        raise ValueError("Candidate moved behind the camera")
    return candidate, delta


def apply_initial_wrist_delta(cells, deltas):
    """Translate robot wrist/frame positions only; this does not admit training."""
    result = copy.deepcopy(cells)
    for side in ("left", "right"):
        delta = np.asarray(deltas[side], float)
        wrist = np.asarray(cells[f"robot_{side}_wrist_position"])
        active = np.asarray(cells[f"mano_{side}_object_contact_part_ids"]).any(axis=1)
        if (
            delta.shape != wrist.shape
            or not np.isfinite(delta).all()
            or np.any(delta[active])
        ):
            raise ValueError("Invalid delta or contact phase change")
        if np.any(delta[0]) or np.any(delta[-1]):
            raise ValueError("Diagnostic endpoints must remain unchanged")
        frames = np.asarray(cells[f"robot_{side}_frames"], dtype=float).copy()
        moving = np.asarray(cells[f"{side}_robot_frame_names"]) != "universe"
        if moving.shape != (frames.shape[1],):
            raise ValueError("Explicit robot frame names must match FK fields")
        frames[:, moving, :3] += delta[:, None, :]
        result[f"robot_{side}_wrist_position"] = (wrist + delta).tolist()
        result[f"robot_{side}_frames"] = frames.tolist()
    return result


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(parquet, transforms, quality, reconstruction, output):
    import pyarrow.parquet as pq
    from PIL import Image, ImageDraw, ImageFont

    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    row = pq.read_table(parquet).to_pylist()[0]
    cameras = np.load(transforms, allow_pickle=False)["original_camera_to_scene"]
    with np.load(quality, allow_pickle=False) as archive:
        k = archive["camera_intrinsics"]
        source_scale = {s: archive[f"hand_{s}_scale"].copy() for s in ("left", "right")}
    fps = float(row["fps"])
    end = 10
    variants, traces, source = {}, {}, {}
    input_paths = [parquet, transforms, quality, Path(__file__)]
    for side, track in (("left", 2), ("right", 3)):
        wrist = np.asarray(row[f"robot_{side}_wrist_position"])
        moving = np.asarray(row[f"{side}_robot_frame_names"]) != "universe"
        frames = np.asarray(row[f"robot_{side}_frames"])[:, moving, :3]
        active = np.asarray(row[f"mano_{side}_object_contact_part_ids"]).any(axis=1)
        old_uv = project_world(frames, cameras, k)
        mano = np.asarray(row[f"mano_{side}_joints"])
        mano_uv = project_world(mano, cameras, k)
        traces[f"{side}_original_wrist"] = wrist
        traces[f"{side}_original_robot_uv"] = old_uv
        traces[f"{side}_source_mano_uv"] = mano_uv
        traces[f"{side}_source_contact_active"] = active
        raw_depth, aligned_depth, added_depth, bbox = [], [], [], []
        for frame in range(end + 1):
            raw_path = reconstruction / f"hamer/{track}/{frame:06d}.json"
            aligned_path = (
                reconstruction.parent / f"hand-quality/hands/{side}/{frame:06d}.json"
            )
            input_paths.extend((raw_path, aligned_path))
            raw = json.loads(raw_path.read_text())
            aligned = json.loads(aligned_path.read_text())
            raw_depth.append(
                raw["camera"]["cam_t"][2] * float(k[0]) / raw["camera"]["focal_length"]
            )
            aligned_depth.append(aligned["camera"]["cam_t"][2])
            added_depth.append(aligned["diagnostics"]["dz"])
            b = raw["bbox"]
            bbox.append([b["x1"] - b["x0"], b["y1"] - b["y0"]])
        source[side] = {
            "first_contact_frame": int(np.flatnonzero(active)[0]),
            "original_wrist_speed_0_to_1_m_s": float(
                np.linalg.norm(wrist[1] - wrist[0]) * fps
            ),
            "raw_hamer_metric_camera_depth_m": raw_depth,
            "aligned_mano_camera_depth_m": aligned_depth,
            "added_alignment_depth_m": added_depth,
            "source_detection_crop_span_px": bbox,
            "estimated_mano_hand_scale": source_scale[side][: end + 1].tolist(),
            "source_mano_projected_span_px": np.ptp(
                mano_uv[: end + 1], axis=1
            ).tolist(),
        }
        for name, fidelity in [("strict", 10.0), ("large", 0.0)]:
            new, delta = ray_smooth_candidate(
                wrist, cameras, active, end_frame=end, fidelity=fidelity
            )
            new_uv = project_world(frames + delta[:, None, :], cameras, k)
            difference = np.linalg.norm(new_uv - old_uv, axis=-1)[1:end]
            wrist_projection = project_world(new, cameras, k) - project_world(
                wrist, cameras, k
            )
            speed = np.linalg.norm(np.diff(new, axis=0), axis=-1) * fps
            old_dist = np.linalg.norm(
                wrist - np.asarray(row["object_body_position"])[:, 0], axis=-1
            )
            new_dist = np.linalg.norm(
                new - np.asarray(row["object_body_position"])[:, 0], axis=-1
            )
            span_ratio = np.ptp(new_uv, axis=1) / np.ptp(old_uv, axis=1)
            report = {
                "fidelity": fidelity,
                "first_step_speed_m_s": float(speed[0]),
                "max_first_0p55s_speed_m_s": float(speed[: end + 1].max()),
                "max_translation_m": float(np.linalg.norm(delta, axis=-1).max()),
                "max_wrist_projection_change_px": float(np.abs(wrist_projection).max()),
                "full_robot_projection_change_p95_px": float(
                    np.quantile(difference, 0.95)
                ),
                "full_robot_projection_change_max_px": float(difference.max()),
                "robot_projected_span_ratio_min": float(span_ratio[: end + 1].min()),
                "robot_projected_span_ratio_max": float(span_ratio[: end + 1].max()),
                "original_wrist_object_center_distance_m": old_dist[: end + 1].tolist(),
                "candidate_wrist_object_center_distance_m": new_dist[
                    : end + 1
                ].tolist(),
                "first_and_frame10_onward_exact": bool(
                    np.array_equal(new[0], wrist[0])
                    and np.array_equal(new[end:], wrist[end:])
                ),
                "source_contact_frames_exact": bool(
                    np.array_equal(new[active], wrist[active])
                ),
                "new_physical_contact_verified": False,
            }
            variants.setdefault(name, {})[side] = report
            traces[f"{name}_{side}_delta_world"] = delta
            traces[f"{name}_{side}_wrist_world"] = new
            traces[f"{name}_{side}_robot_uv"] = new_uv
    np.savez_compressed(
        output / "candidates.npz", camera_to_scene=cameras, intrinsics=k, **traces
    )
    shown = [1, 2, 5, 10]
    width, height, title_height, top = 720, 540, 38, 85
    figure = Image.new("RGB", (width * 4, (height + title_height) * 3 + top), "white")
    draw = ImageDraw.Draw(figure)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 23)
    except OSError:
        font = ImageFont.load_default()
    draw.text(
        (20, 10),
        "Original RGB and full robot-frame projection differences. DIAGNOSTIC ONLY; NOT FOR TRAINING.",
        fill="black",
        font=font,
    )
    draw.text(
        (20, 43),
        "Cyan: original; orange: candidate. A fixed wrist ray does not preserve the whole-hand projection.",
        fill="black",
        font=font,
    )
    for column, frame in enumerate(shown):
        rgb_path = reconstruction / f"frames/{frame:06d}.png"
        input_paths.append(rgb_path)
        rgb = Image.open(rgb_path).convert("RGB")
        scale = np.asarray([width / rgb.width, height / rgb.height])
        for r, name in enumerate(["original", "strict", "large"]):
            panel = rgb.resize((width, height))
            overlay = ImageDraw.Draw(panel)
            if r:
                for side in ("left", "right"):
                    old = traces[f"{side}_original_robot_uv"][frame] * scale
                    new = traces[f"{name}_{side}_robot_uv"][frame] * scale
                    for a, b in zip(old, new, strict=True):
                        overlay.line([tuple(a), tuple(b)], fill="#ffc000", width=2)
                        overlay.ellipse(
                            (a[0] - 2, a[1] - 2, a[0] + 2, a[1] + 2), fill="#00c8ff"
                        )
                        overlay.ellipse(
                            (b[0] - 2, b[1] - 2, b[0] + 2, b[1] + 2), fill="#ff7b00"
                        )
            left, upper = column * width, top + r * (height + title_height)
            draw.text(
                (left + 12, upper + 4),
                f"{name} | frame {frame}, {frame / fps:.2f}s",
                fill="black",
                font=font,
            )
            figure.paste(panel, (left, upper + title_height))
    figure.save(output / "projection-tradeoff.png")
    result = {
        "schema": "nvidia_v2d.initial_wrist_diagnostic.v1",
        "training_allowed": False,
        "status": "diagnostic_tradeoff_not_accepted",
        "fixed_frames": "0 and 10 onward; first0/contact phase constraints are diagnostic choices",
        "changed_fields_candidate_only": [
            "robot wrist positions",
            "robot frame positions",
        ],
        "no_object_or_input_mutation": True,
        "source": source,
        "variants": variants,
        "interpretation": "Large smoothing lowers initial speed but changes full-hand image projection substantially. Strict fidelity preserves image shape better but barely reduces the first jump.",
        "physical_contact_claim": "Source contact labels are unchanged. No physical new-contact/penetration test was performed, and wrist-center distance is not a contact test.",
        "source_size_claim": "MANO bone geometry and estimated scale remain unchanged; crop dimensions are detector crops, not measured anatomy.",
        "projection_frames": "66 moving hand/root/joint/site frames per hand; the fixed universe frame is excluded from both translation and hand-image metrics.",
        "input_sha256": {str(p.resolve()): _sha(p) for p in input_paths},
    }
    (output / "manifest.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    (output / "sha256.json").write_text(
        json.dumps(
            {p.name: _sha(p) for p in sorted(output.iterdir()) if p.is_file()}, indent=2
        )
        + "\n"
    )
    print(json.dumps(variants, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("parquet", "transforms", "quality", "reconstruction", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    run(args.parquet, args.transforms, args.quality, args.reconstruction, args.output)


if __name__ == "__main__":
    main()
