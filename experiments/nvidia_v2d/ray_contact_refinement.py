"""Diagnose fixed-anatomy wrist-ray depth/contact refinement, never a training gate.

Contact roles are explicit inspected frames and thumb surface, not all fingertips.
The source geometry and object trajectory are immutable. Candidate projection
changes are measured against the source reconstruction, not image ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.nvidia_v2d.static_support_depth import project_scene

CONTACT_ANCHORS = (60, 70, 80, 100, 120, 140, 160, 180, 200)


def thumb_vertex_mask(vertices: np.ndarray, joints: np.ndarray) -> np.ndarray:
    """Label vertices once using nearest bone in the inspected open frame 0."""
    chains = [(0, 1), (1, 2), (2, 3), (3, 4)]
    for root in (5, 9, 13, 17):
        chains.extend(
            [(0, root), (root, root + 1), (root + 1, root + 2), (root + 2, root + 3)]
        )
    distances = []
    for a, b in chains:
        axis = joints[b] - joints[a]
        norm2 = float(axis @ axis)
        if norm2 < 1e-12:
            raise ValueError("Degenerate hand bone")
        u = np.clip((vertices - joints[a]) @ axis / norm2, 0, 1)
        distances.append(
            np.linalg.norm(vertices - joints[a] - u[:, None] * axis, axis=1)
        )
    # Exclude wrist-to-thumb-base and the other fingers/palm.
    return np.isin(np.argmin(distances, axis=0), [1, 2, 3])


def solve_depth_path(
    costs: np.ndarray, positions: np.ndarray, *, step_scale_m: float = 0.03
) -> tuple[np.ndarray, float]:
    """Exact finite-grid optimum of frame data costs plus squared wrist steps."""
    costs = np.asarray(costs, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if costs.ndim != 2 or positions.shape != (*costs.shape, 3):
        raise ValueError("Expected costs[T,K] and positions[T,K,3]")
    if (
        not np.isfinite(costs).all()
        or not np.isfinite(positions).all()
        or step_scale_m <= 0
    ):
        raise ValueError("Finite costs/positions and positive step scale required")
    score = costs[0].copy()
    previous = np.zeros(costs.shape, dtype=np.int64)
    for t in range(1, len(costs)):
        transitions = np.square(
            (positions[t, :, None] - positions[t - 1, None, :]) / step_scale_m
        ).sum(axis=-1)
        cumulative = transitions + score[None, :]
        previous[t] = cumulative.argmin(axis=1)
        score = costs[t] + cumulative[np.arange(costs.shape[1]), previous[t]]
    path = np.zeros(len(costs), dtype=np.int64)
    path[-1] = int(score.argmin())
    best = float(score[path[-1]])
    for t in range(len(costs) - 1, 0, -1):
        path[t - 1] = previous[t, path[t]]
    return path, best


def _samples(mesh, count: int = 100000) -> np.ndarray:
    rng = np.random.default_rng(42)
    triangles = mesh.triangles
    area = mesh.area_faces
    chosen = rng.choice(len(triangles), count, p=area / area.sum())
    uv = rng.random((count, 2))
    uv[uv.sum(axis=1) > 1] = 1 - uv[uv.sum(axis=1) > 1]
    tri = triangles[chosen]
    points = (
        tri[:, 0]
        + uv[:, :1] * (tri[:, 1] - tri[:, 0])
        + uv[:, 1:] * (tri[:, 2] - tri[:, 0])
    )
    return np.vstack((mesh.vertices, points))


def refine(
    geometry: Path, quality: Path, depth_audit: Path, mesh: Path, output: Path
) -> dict:
    import trimesh
    from scipy.spatial import cKDTree

    if output.exists():
        raise FileExistsError(output)
    sources = [
        geometry,
        quality,
        depth_audit / "candidate-diagnostic.npz",
        mesh,
        Path(__file__),
    ]
    hashes = {
        str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
    }
    with np.load(geometry) as g:
        source = {k: g[k].copy() for k in g.files}
    with np.load(quality) as q:
        intrinsics = q["camera_intrinsics"].copy()
    with np.load(depth_audit / "candidate-diagnostic.npz") as d:
        camera_to_scene = d["camera_to_scene"].copy()
        table_scales = d["scales"].copy()
    frame_count = len(camera_to_scene)
    if frame_count != 249:
        raise ValueError(
            "This diagnostic uses explicitly inspected public44 contact frames"
        )
    surface_mesh = trimesh.load_mesh(mesh, process=False)
    tree = cKDTree(_samples(surface_mesh))
    grid = np.linspace(-0.18, 0.18, 121)
    candidate = {k: v.copy() for k, v in source.items()}
    summary = {}
    traces = {}
    for side in ("left", "right"):
        joints = source[side + "_joints"]
        vertices = source[side + "_vertices"]
        wrist = joints[:, 0]
        ray = wrist - camera_to_scene[:, :3, 3]
        ray /= np.linalg.norm(ray, axis=1)[:, None]
        thumb = thumb_vertex_mask(vertices[0], joints[0])
        hand_paths = [
            quality.parent / "hands" / side / f"{f:06d}.json"
            for f in range(frame_count)
        ]
        records = [json.loads(p.read_text()) for p in hand_paths]
        for p in hand_paths:
            hashes[str(p.resolve())] = hashlib.sha256(p.read_bytes()).hexdigest()
        shift_camera = np.array([r["camera"]["cam_t"] for r in records]) - np.array(
            [r["diagnostics"]["cam_t_pre_dz"] for r in records]
        )
        shift_scene = np.einsum("nij,nj->ni", camera_to_scene[:, :3, :3], shift_camera)
        depth_shift = np.einsum("ni,ni->n", shift_scene, ray)
        raw_target = np.median(depth_shift[list(CONTACT_ANCHORS)]) - depth_shift
        table_target = (table_scales - 1) * np.linalg.norm(
            wrist - camera_to_scene[:, :3, 3], axis=1
        )
        thumb_q05 = np.zeros((frame_count, len(grid)))
        projection_p95 = np.zeros_like(thumb_q05)
        projection_rms = np.zeros_like(thumb_q05)
        positions = wrist[:, None] + grid[None, :, None] * ray[:, None]
        for t in range(frame_count):
            T = source["object_to_world_transform"][t]
            trial_thumb = vertices[t, thumb][None] + grid[:, None, None] * ray[t]
            local = (trial_thumb - T[:3, 3]) @ T[:3, :3]
            distances = tree.query(local.reshape(-1, 3), workers=1)[0].reshape(
                len(grid), -1
            )
            thumb_q05[t] = np.quantile(distances, 0.05, axis=1)
            original_uv = project_scene(vertices[t], camera_to_scene[t], intrinsics)
            trials = vertices[t][None] + grid[:, None, None] * ray[t]
            uv = project_scene(
                trials.reshape(-1, 3), camera_to_scene[t], intrinsics
            ).reshape(len(grid), -1, 2)
            pixel_error = np.linalg.norm(uv - original_uv, axis=-1)
            projection_p95[t] = np.quantile(pixel_error, 0.95, axis=1)
            projection_rms[t] = np.sqrt(np.square(pixel_error).mean(axis=1))
        # These transparent diagnostic costs are not evaluator or gate changes.
        raw_weight = 0.25 + 1.75 * np.maximum(0, 1 - np.arange(frame_count) / 20)
        costs = raw_weight[:, None] * np.square((grid - raw_target[:, None]) / 0.03)
        costs += 0.2 * np.square((grid - table_target[:, None]) / 0.03)
        costs += np.square(projection_rms / 12)
        costs += 0.05 * np.square(grid / 0.1)
        for t in CONTACT_ANCHORS:
            costs[t] += 4 * np.square(np.maximum(thumb_q05[t] - 0.004, 0) / 0.01)
        path, objective = solve_depth_path(costs, positions)
        translation = grid[path, None] * ray
        candidate[side + "_joints"] = joints + translation[:, None]
        candidate[side + "_vertices"] = vertices + translation[:, None]
        chosen_projection = projection_p95[np.arange(frame_count), path]
        speed_before = np.linalg.norm(np.diff(wrist, axis=0), axis=-1) * 20
        speed_after = np.linalg.norm(np.diff(wrist + translation, axis=0), axis=-1) * 20
        exact = {}
        for t in CONTACT_ANCHORS:
            T = source["object_to_world_transform"][t]
            exact[str(t)] = {}
            for label, verts in (
                ("original", vertices[t]),
                ("candidate", candidate[side + "_vertices"][t]),
            ):
                local = (verts[thumb] - T[:3, 3]) @ T[:3, :3]
                distances = trimesh.proximity.closest_point(surface_mesh, local)[1]
                exact[str(t)][label] = {
                    "thumb_surface_min_m": float(distances.min()),
                    "thumb_surface_q05_m": float(np.quantile(distances, 0.05)),
                    "thumb_vertices_within_5mm": int((distances <= 0.005).sum()),
                }
            allowed = projection_p95[t] <= 15
            best_index = np.flatnonzero(allowed)[np.argmin(thumb_q05[t, allowed])]
            exact[str(t)]["scan_projection_budget_px"] = 15
            exact[str(t)]["best_q05_within_projection_budget_m"] = float(
                thumb_q05[t, best_index]
            )
            exact[str(t)]["best_grid_shift_m"] = float(grid[best_index])
        anatomy_change = np.abs(
            np.linalg.norm(
                candidate[side + "_joints"] - candidate[side + "_joints"][:, :1],
                axis=-1,
            )
            - np.linalg.norm(joints - joints[:, :1], axis=-1)
        )
        summary[side] = {
            "thumb_vertex_count": int(thumb.sum()),
            "objective": objective,
            "shift_range_m": [float(grid[path].min()), float(grid[path].max())],
            "initial_step_speed_before_mps": float(speed_before[0]),
            "initial_step_speed_candidate_mps": float(speed_after[0]),
            "first_20_step_speed_before_max_mps": float(speed_before[:20].max()),
            "first_20_step_speed_candidate_max_mps": float(speed_after[:20].max()),
            "all_speed_before_p95_mps": float(np.quantile(speed_before, 0.95)),
            "all_speed_candidate_p95_mps": float(np.quantile(speed_after, 0.95)),
            "frame_projection_p95_max_px": float(chosen_projection.max()),
            "frame_projection_p95_over_15px": int((chosen_projection > 15).sum()),
            "max_anatomy_change_m": float(anatomy_change.max()),
            "contact_anchors": exact,
        }
        traces.update(
            {
                side + "_" + k: v
                for k, v in {
                    "thumb_vertex_mask": thumb,
                    "sampled_thumb_surface_q05": thumb_q05,
                    "projection_p95_grid": projection_p95,
                    "data_costs": costs,
                    "chosen_path": path,
                    "shift_m": grid[path],
                    "raw_alignment_target_m": raw_target,
                    "table_target_m": table_target,
                }.items()
            }
        )
    unchanged = all(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
        for p, digest in hashes.items()
    )
    if not unchanged:
        raise RuntimeError("Source or implementation changed during the diagnostic")
    output.mkdir(parents=True)
    np.savez_compressed(output / "diagnostic-candidate-geometry.npz", **candidate)
    np.savez_compressed(output / "grid-trace.npz", **traces, grid_m=grid)
    result = {
        "schema": "nvidia_v2d.wrist_ray_contact_diagnostic.v1",
        "source_sha256": hashes,
        "source_unchanged": unchanged,
        "contact_anchor_frames": CONTACT_ANCHORS,
        "contact_role": "thumb surface only; other fingertips visibly extended",
        "source_projection_is_image_ground_truth": False,
        "object_trajectory_unchanged": bool(
            np.array_equal(
                candidate["object_to_world_transform"],
                source["object_to_world_transform"],
            )
        ),
        "training_allowed": False,
        "success_demonstrated": False,
        "limitations": [
            "Surface grid search uses deterministic sampled mesh; selected anchor distances are then exact triangle distances.",
            "Unsigned distance is not physical contact, force closure, collision penetration, or image annotation truth.",
            "Fixed source hand shape is retained per frame; this does not repair any pre-existing articulation or scale errors.",
            "Candidate must pass image/contact/robot dynamics validation before any training use.",
        ],
        "hands": summary,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("geometry", "quality", "depth-audit", "mesh", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    result = refine(**vars(parser.parse_args()))
    print(
        json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, indent=2)
    )


if __name__ == "__main__":
    main()
