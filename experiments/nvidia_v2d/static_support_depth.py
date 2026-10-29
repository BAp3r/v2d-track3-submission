"""Audit temporal monocular-depth drift on a visible static support.

A shared predicted-depth plane is a consistency constraint, not sensor truth.
The optional hand candidate applies a rigid translation around the wrist ray;
it preserves anatomy, measures changed reprojections, and never passes a
training gate automatically or modifies the original reconstruction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def robust_plane(
    points: np.ndarray, *, threshold: float = 0.012, seed: int = 42
) -> dict:
    """Fit z=a*x+b*y+c with deterministic RANSAC, retaining rejected observations."""
    points = np.asarray(points, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 100
        or not np.isfinite(points).all()
    ):
        raise ValueError("At least 100 finite xyz support observations are required")
    design = np.column_stack((points[:, :2], np.ones(len(points))))
    rng = np.random.default_rng(seed)
    best = np.zeros(len(points), dtype=bool)
    for _ in range(80):
        indices = rng.choice(len(points), 3, replace=False)
        if abs(np.linalg.det(design[indices])) < 1e-7:
            continue
        coefficients = np.linalg.solve(design[indices], points[indices, 2])
        if np.linalg.norm(coefficients[:2]) > 0.3:
            continue
        inside = abs(points[:, 2] - design @ coefficients) <= threshold
        if inside.sum() > best.sum():
            best = inside
    if best.sum() < 100:
        raise ValueError("No near-horizontal support plane has 100 inliers")
    for _ in range(4):
        coefficients = np.linalg.lstsq(design[best], points[best, 2], rcond=None)[0]
        best = abs(points[:, 2] - design @ coefficients) <= threshold
    residual = abs(points[:, 2] - design @ coefficients)
    return {
        "coefficients": coefficients.tolist(),
        "point_count": len(points),
        "inlier_count": int(best.sum()),
        "inlier_fraction": float(best.mean()),
        "inlier_residual_median_m": float(np.median(residual[best])),
        "inlier_residual_p95_m": float(np.quantile(residual[best], 0.95)),
        "all_residual_p95_m": float(np.quantile(residual, 0.95)),
        "tilt_deg": float(np.degrees(np.arctan(np.linalg.norm(coefficients[:2])))),
        "inlier_xy_extent_m": np.ptp(points[best, :2], axis=0).tolist(),
    }


def plane_ray_scale(
    coefficients: np.ndarray,
    camera: np.ndarray,
    anchor_xy: np.ndarray,
    target_height: float,
) -> float:
    """Camera homothety factor mapping the observed plane to the target anchor height."""
    a, b, c = np.asarray(coefficients, dtype=float)
    camera = np.asarray(camera, dtype=float)
    n = np.array([-a, -b, 1.0])
    target_c = target_height - a * anchor_xy[0] - b * anchor_xy[1]
    denominator = n @ camera - c
    if abs(denominator) < 1e-8:
        raise ValueError("Camera lies on the observed support plane")
    scale = float((n @ camera - target_c) / denominator)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Support scale must be finite and positive")
    return scale


def translate_hand_on_wrist_ray(
    points: np.ndarray, wrist: np.ndarray, camera: np.ndarray, scale: float
) -> np.ndarray:
    """Translate the entire rigid hand; preserve local lengths, not every pixel."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Scale must be finite and positive")
    return np.asarray(points) + (scale - 1) * (np.asarray(wrist) - np.asarray(camera))


def project_scene(
    points: np.ndarray, camera_to_scene: np.ndarray, intrinsics: np.ndarray
) -> np.ndarray:
    local = (np.asarray(points) - camera_to_scene[:3, 3]) @ camera_to_scene[:3, :3]
    if np.any(local[:, 2] <= 0):
        raise ValueError("Hand point behind camera")
    return local[:, :2] / local[:, 2:3] * intrinsics[:2] + intrinsics[2:]


def _frame_plane(
    frame: int,
    reconstruction: Path,
    camera_to_scene: np.ndarray,
    intrinsics: np.ndarray,
    roi: tuple[int, int, int, int],
) -> dict:
    import cv2

    rgb_path = reconstruction / "frames" / f"{frame:06d}.png"
    depth_path = reconstruction / "depth" / f"{frame:06d}.png"
    rgb = cv2.imread(str(rgb_path))
    enc = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if (
        rgb is None
        or enc is None
        or enc.dtype != np.uint16
        or enc.shape != rgb.shape[:2]
    ):
        raise ValueError(f"Missing or invalid RGB/depth frame {frame}")
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([4, 65, 40]), np.array([28, 255, 255]))
    x0, y0, x1, y1 = roi
    allowed = np.zeros_like(mask)
    allowed[y0:y1, x0:x1] = 255
    mask &= allowed
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"No visible orange support at frame {frame}")
    component = np.zeros_like(mask)
    cv2.drawContours(
        component, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED
    )
    component &= mask
    component = cv2.erode(component, np.ones((9, 9), np.uint8))
    ys, xs = np.nonzero((component > 0) & (enc > 0) & (enc < 65535))
    # Bounded deterministic subsampling, never interpolates the time axis.
    stride = max(1, int(np.ceil(len(ys) / 6000)))
    ys, xs = ys[::stride], xs[::stride]
    z = 65535.0 / enc[ys, xs] - 1
    camera_points = np.column_stack(
        (
            (xs - intrinsics[2]) / intrinsics[0] * z,
            (ys - intrinsics[3]) / intrinsics[1] * z,
            z,
        )
    )
    points = camera_points @ camera_to_scene[:3, :3].T + camera_to_scene[:3, 3]
    row = robust_plane(points)
    extent = row["inlier_xy_extent_m"]
    row.update(
        frame=frame,
        accepted=(
            row["inlier_fraction"] >= 0.45
            and row["tilt_deg"] < 12
            and min(extent) >= 0.15
        ),
        camera_scene_xyz=camera_to_scene[:3, 3].tolist(),
        source_rgb_sha256=hashlib.sha256(rgb_path.read_bytes()).hexdigest(),
        source_depth_sha256=hashlib.sha256(depth_path.read_bytes()).hexdigest(),
    )
    return row


def _svg(path: Path, series: dict[str, np.ndarray], title: str, ylabel: str) -> None:
    from xml.sax.saxutils import escape

    colors = ["#2563eb", "#dc2626", "#059669", "#9333ea"]
    low = min(float(np.nanmin(x)) for x in series.values())
    high = max(float(np.nanmax(x)) for x in series.values())
    if high <= low:
        high = low + 1
    content = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="410" viewBox="0 0 1000 410">',
        '<rect width="1000" height="410" fill="white"/>',
        f'<text x="60" y="25" font-family="sans-serif" font-size="18">{escape(title)}</text>',
        f'<text x="10" y="60" font-family="sans-serif" font-size="12">{escape(ylabel)}</text>',
    ]
    for i in range(6):
        value = low + (high - low) * i / 5
        y = 340 - 260 * i / 5
        content += [
            f'<line x1="65" x2="970" y1="{y}" y2="{y}" stroke="#ddd"/>',
            f'<text x="5" y="{y}" font-size="12">{value:.3f}</text>',
        ]
    for index, (name, values) in enumerate(series.items()):
        points = " ".join(
            f"{65 + j / (len(values) - 1) * 905:.2f},{340 - (v - low) / (high - low) * 260:.2f}"
            for j, v in enumerate(values)
            if np.isfinite(v)
        )
        color = colors[index % len(colors)]
        content += [
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.5"/>',
            f'<text x="{65 + index * 230}" y="375" fill="{color}" font-size="12">{escape(name)}</text>',
        ]
    content += [
        '<text x="420" y="400" font-size="12">Frame 0 to 248 (20 FPS; all frames retained)</text></svg>'
    ]
    path.write_text("\n".join(content))


def audit_sequence(
    reconstruction: Path,
    quality: Path,
    geometry: Path,
    mesh: Path,
    support_manifest: Path,
    output: Path,
    workers: int = 4,
) -> dict:
    if output.exists():
        raise FileExistsError(output)
    sources = [quality, geometry, mesh, support_manifest, Path(__file__)]
    source_hashes = {
        str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
    }
    output.mkdir(parents=True)
    with np.load(quality) as q, np.load(geometry) as g:
        intrinsics = q["camera_intrinsics"].copy()
        transforms = (
            g["source_to_scene_transform"][None] @ q["camera_to_world_transform"]
        )
        arrays = {key: g[key].copy() for key in g.files}
    support = json.loads(support_manifest.read_text())
    roi = tuple(support["roi_xyxy"])
    anchor_xy = np.array(support["fit"]["center_xy_m"])
    frame_count = len(transforms)

    def one(frame):
        try:
            return _frame_plane(
                frame, reconstruction, transforms[frame], intrinsics, roi
            )
        except ValueError as e:
            return {"frame": frame, "accepted": False, "error": str(e)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, range(frame_count)))
    (output / "per-frame-planes.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False)
    )
    anchors = [120, 140, 160]
    if not all(rows[i]["accepted"] for i in anchors):
        raise ValueError(
            "Previously inspected grasp anchor frames lack reliable visible support"
        )
    heights = np.array(
        [
            np.dot(row["coefficients"][:2], anchor_xy) + row["coefficients"][2]
            if "coefficients" in row
            else np.nan
            for row in rows
        ]
    )
    target_height = float(np.median(heights[anchors]))
    scale = np.array(
        [
            plane_ray_scale(
                row["coefficients"], transforms[i, :3, 3], anchor_xy, target_height
            )
            if row["accepted"]
            else np.nan
            for i, row in enumerate(rows)
        ]
    )
    rigid = {}
    ray = {}
    metrics = {}
    for side in ("left", "right"):
        joints = arrays[side + "_joints"]
        verts = arrays[side + "_vertices"]
        wrist = joints[:, 0]
        delta = (scale - 1)[:, None] * (wrist - transforms[:, :3, 3])
        rigid[side + "_joints"] = joints + delta[:, None]
        rigid[side + "_vertices"] = verts + delta[:, None]
        ray[side + "_vertices"] = transforms[:, None, :3, 3] + scale[:, None, None] * (
            verts - transforms[:, None, :3, 3]
        )
        reproj = []
        anatomy = []
        for i in range(frame_count):
            if not rows[i]["accepted"]:
                reproj.append(np.full(len(verts[i]), np.nan))
                anatomy.append(np.nan)
                continue
            uv = project_scene(verts[i], transforms[i], intrinsics)
            after = project_scene(
                rigid[side + "_vertices"][i], transforms[i], intrinsics
            )
            reproj.append(np.linalg.norm(after - uv, axis=1))
            anatomy.append(
                float(
                    np.abs(
                        np.linalg.norm(
                            rigid[side + "_joints"][i] - rigid[side + "_joints"][i, 0],
                            axis=1,
                        )
                        - np.linalg.norm(joints[i] - joints[i, 0], axis=1)
                    ).max()
                )
            )
        reproj = np.array(reproj)
        before_speed = np.linalg.norm(np.diff(wrist, axis=0), axis=1) * 20
        after_speed = (
            np.linalg.norm(np.diff(rigid[side + "_joints"][:, 0], axis=0), axis=1) * 20
        )
        rigid[side + "_vertex_reprojection_error_px"] = reproj
        metrics[side] = {
            "before_wrist_speed_max_mps": float(np.max(before_speed)),
            "candidate_wrist_speed_max_mps": float(np.nanmax(after_speed)),
            "before_wrist_speed_p95_mps": float(np.quantile(before_speed, 0.95)),
            "candidate_wrist_speed_p95_mps": float(np.nanquantile(after_speed, 0.95)),
            "rigid_anatomy_max_distance_change_m": float(np.nanmax(anatomy)),
            "rigid_vertex_reprojection_median_px": float(np.nanmedian(reproj)),
            "rigid_vertex_reprojection_p95_px": float(np.nanquantile(reproj, 0.95)),
            "translation_max_m": float(np.nanmax(np.linalg.norm(delta, axis=1))),
            "first_three_wrists_before": wrist[:3].tolist(),
            "first_three_wrists_candidate": rigid[side + "_joints"][:3, 0].tolist(),
        }
    np.savez_compressed(
        output / "candidate-diagnostic.npz",
        **rigid,
        ray_left_vertices=ray["left_vertices"],
        ray_right_vertices=ray["right_vertices"],
        scales=scale,
        plane_height=heights,
        camera_to_scene=transforms,
        object_to_world_transform=arrays["object_to_world_transform"],
    )
    # Compare contact only at previously inspected grasp frames; this is unsigned distance, not contact truth.
    import trimesh

    object_mesh = trimesh.load_mesh(mesh, process=False)
    contact = {}
    for i in anchors:
        T = arrays["object_to_world_transform"][i]
        contact[str(i)] = {}
        for side in ("left", "right"):
            for label, vertices in [
                ("original", arrays[side + "_vertices"][i]),
                ("rigid_candidate", rigid[side + "_vertices"][i]),
            ]:
                local = (vertices - T[:3, 3]) @ T[:3, :3]
                distances = []
                for start in range(0, len(local), 64):
                    distances.extend(
                        trimesh.proximity.closest_point(
                            object_mesh, local[start : start + 64]
                        )[1]
                    )
                contact[str(i)][side + "_" + label] = {
                    "minimum_surface_distance_m": float(np.min(distances)),
                    "vertices_within_5mm": int((np.asarray(distances) <= 0.005).sum()),
                }
    alignment = {}
    for side in ("left", "right"):
        hand_paths = [
            quality.parent / "hands" / side / f"{i:06d}.json"
            for i in range(frame_count)
        ]
        if not all(p.is_file() for p in hand_paths):
            continue
        records = [json.loads(p.read_text()) for p in hand_paths]
        before = np.array([r["diagnostics"]["cam_t_pre_dz"] for r in records])
        after = np.array([r["camera"]["cam_t"] for r in records])
        dz = np.array([r["diagnostics"]["dz"] for r in records])
        before_speed = np.linalg.norm(np.diff(before, axis=0), axis=1) * 20
        after_speed = np.linalg.norm(np.diff(after, axis=0), axis=1) * 20
        alignment[side] = {
            "first_three_pre_dz_camera_z_m": before[:3, 2].tolist(),
            "first_three_aligned_camera_z_m": after[:3, 2].tolist(),
            "first_three_dz_m": dz[:3].tolist(),
            "pre_dz_initial_speed_mps": float(before_speed[0]),
            "aligned_initial_speed_mps": float(after_speed[0]),
            "pre_dz_speed_p95_mps": float(np.quantile(before_speed, 0.95)),
            "aligned_speed_p95_mps": float(np.quantile(after_speed, 0.95)),
            "all_frame_table_height_dz_correlation": float(
                np.corrcoef(heights, dz)[0, 1]
            ),
            "source_sha256": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in hand_paths
            },
        }
    source_unchanged = all(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
        for p, digest in source_hashes.items()
    )
    if not source_unchanged:
        raise RuntimeError("An input or audit implementation changed during the run")
    summary = {
        "schema": "nvidia_v2d.static_support_depth_audit.v1",
        "frame_count": frame_count,
        "fps": 20,
        "accepted_frames": sum(row["accepted"] for row in rows),
        "rejected_frames": [r["frame"] for r in rows if not r["accepted"]],
        "source_sha256": source_hashes,
        "source_unchanged": source_unchanged,
        "raw_alignment_audit": alignment,
        "anchor_frames": anchors,
        "anchor_target_height_m": target_height,
        "support_depth_height_range_m": [
            float(np.nanmin(heights)),
            float(np.nanmax(heights)),
        ],
        "support_height_max_adjacent_change_m": float(np.nanmax(abs(np.diff(heights)))),
        "candidate_scale_range": [float(np.nanmin(scale)), float(np.nanmax(scale))],
        "rigid_candidate": metrics,
        "grasp_frame_surface_comparison": contact,
        "training_allowed": False,
        "geometry_quality_accepted": False,
        "interpretation": [
            "Static-table depths drifting in shared coordinates expose monocular temporal inconsistency, not metric GT.",
            "Rigid wrist-ray correction preserves local anatomy but shifts individual projected vertices; it is diagnostic only.",
            "Full ray-scaling preserves projection but changes anatomy each frame and must not be used as robot training geometry.",
            "Object trajectory is unchanged; successful manipulation has not been demonstrated.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    _svg(
        output / "support-depth-drift.svg",
        {
            "predicted static table height": heights,
            "fixed anchor height": np.full(frame_count, target_height),
        },
        "Public44: predicted depth of the same static tabletop",
        "height (m)",
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "reconstruction",
        "quality",
        "geometry",
        "mesh",
        "support-manifest",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    print(json.dumps(audit_sequence(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
