"""Exact surface-distance bounds for the fixed-hand wrist-ray diagnostic.

This audits a restricted geometric repair family, not physical contact or image
ground truth. No candidate is exported to a training reference or accepted here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.nvidia_v2d.ray_contact_refinement import CONTACT_ANCHORS
from experiments.nvidia_v2d.static_support_depth import project_scene


def lipschitz_lower_bound(samples: np.ndarray, values: np.ndarray) -> float:
    """Bound a nonnegative 1-Lipschitz function over the sampled closed interval.

    Every interval point is within half the largest sample gap of a sample.
    Unsigned distance to a fixed triangle surface, and any quantile of such
    distances under a unit translation, are 1-Lipschitz in translation length.
    """
    samples = np.asarray(samples, dtype=float)
    values = np.asarray(values, dtype=float)
    if (samples.ndim != 1 or samples.shape != values.shape or len(samples) < 2
            or not np.isfinite(samples).all() or not np.isfinite(values).all()
            or np.any(np.diff(samples) <= 0) or np.any(values < 0)):
        raise ValueError("Require finite increasing samples and nonnegative distances")
    if np.any(np.abs(np.diff(values)) > np.diff(samples) + 1e-7):
        raise ValueError("Sample values violate the claimed 1-Lipschitz bound")
    return float(max(0, values.min() - np.diff(samples).max() / 2 - 1e-9))


def projection_interval(error, *, budget_px: float, shift_limit_m: float) -> tuple[float, float]:
    """Find the two endpoints with monotone perspective error on each ray branch."""
    if budget_px <= 0 or shift_limit_m <= 0 or abs(error(0.0)) > 1e-9:
        raise ValueError("Positive limits and zero projection error at origin required")
    endpoints = []
    for sign in (-1, 1):
        if error(sign * shift_limit_m) <= budget_px:
            endpoints.append(sign * shift_limit_m)
            continue
        low, high = 0.0, shift_limit_m
        # On a fixed positive-depth camera ray, each vertex error has the form
        # |shift|*constant/(z + shift*ray_z), so their quantile is monotone too.
        for _ in range(60):
            middle = (low + high) / 2
            if error(sign * middle) <= budget_px:
                low = middle
            else:
                high = middle
        endpoints.append(sign * low)
    return tuple(endpoints)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(input_root: Path, ray_audit: Path, depth_audit: Path, output: Path) -> dict:
    import trimesh

    if output.exists():
        raise FileExistsError(output)
    index_path = input_root / "local-input-index.json"
    index = json.loads(index_path.read_text())
    paths = {k: (input_root / v).resolve() for k, v in index["paths"].items()}
    prior_path = ray_audit / "summary.json"
    prior = json.loads(prior_path.read_text())
    verified_history = {}
    for historical, digest in prior["source_sha256"].items():
        mapped = index["historical_exact_path_aliases"].get(historical)
        if mapped is None:
            for prefix, local in index["historical_path_prefix_map"].items():
                if historical == prefix or historical.startswith(prefix + "/"):
                    mapped = str(Path(local) / Path(historical).relative_to(prefix))
                    break
        local_path = (input_root / mapped).resolve() if mapped is not None else Path(historical)
        if _sha(local_path) != digest:
            raise ValueError(f"Historical diagnostic input changed: {local_path}")
        verified_history[str(local_path)] = digest
    sources = [index_path, prior_path, ray_audit / "grid-trace.npz",
               ray_audit / "diagnostic-candidate-geometry.npz", paths["episode_geometry"],
               paths["episode_mesh"], paths["hand_quality"],
               depth_audit / "candidate-diagnostic.npz", Path(__file__),
               Path(project_scene.__code__.co_filename)]
    hashes = {str(p.resolve()): _sha(p) for p in sources}
    with np.load(paths["episode_geometry"]) as data:
        geometry = {key: data[key].copy() for key in data.files}
    with np.load(ray_audit / "diagnostic-candidate-geometry.npz") as data:
        candidate = {key: data[key].copy() for key in data.files}
    with np.load(ray_audit / "grid-trace.npz") as data:
        trace = {key: data[key].copy() for key in data.files}
    with np.load(depth_audit / "candidate-diagnostic.npz") as data:
        cameras = data["camera_to_scene"].copy()
    with np.load(paths["hand_quality"]) as data:
        intrinsics = data["camera_intrinsics"].copy()
    mesh = trimesh.load_mesh(paths["episode_mesh"], process=False)
    if len(cameras) != 249:
        raise ValueError("Expected the full 249-frame source sequence")
    rows = []
    arrays = {}
    rigid_checks = {}
    for side in ("left", "right"):
        vertices = geometry[side + "_vertices"]
        wrists = geometry[side + "_joints"][:, 0]
        mask = trace[side + "_thumb_vertex_mask"]
        shifts = candidate[side + "_joints"][:, 0] - wrists
        max_rigid_error = max(float(np.max(np.abs(
            candidate[side + "_" + kind] - geometry[side + "_" + kind] - shifts[:, None]
        ))) for kind in ("vertices", "joints"))
        rigid_checks[side] = {"max_rigid_translation_error_m": max_rigid_error}
        for frame in CONTACT_ANCHORS:
            ray = wrists[frame] - cameras[frame, :3, 3]
            ray /= np.linalg.norm(ray)
            original_uv = project_scene(vertices[frame], cameras[frame], intrinsics)

            def pixel_error(shift, verts=vertices[frame], direction=ray,
                            camera=cameras[frame], initial_uv=original_uv):
                uv = project_scene(verts + shift * direction, camera, intrinsics)
                return float(np.quantile(np.linalg.norm(uv - initial_uv, axis=1), 0.95))

            interval = projection_interval(pixel_error, budget_px=15.0, shift_limit_m=0.18)
            # Include exact constraint endpoints; nearest-sample bounds then cover
            # the entire continuous allowed interval, not just its original grid.
            all_grid = np.unique(np.concatenate((trace["grid_m"], interval)))
            transform = geometry["object_to_world_transform"][frame]
            rotation = transform[:3, :3]
            if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6, rtol=0):
                raise ValueError("Object transform is not a rigid rotation")
            trials = vertices[frame, mask][None] + all_grid[:, None, None] * ray
            local = ((trials - transform[:3, 3]) @ rotation).reshape(-1, 3)
            distances = np.concatenate([
                trimesh.proximity.closest_point(mesh, local[start:start + 512])[1]
                for start in range(0, len(local), 512)
            ]).reshape(len(all_grid), -1)
            values = np.quantile(distances, 0.05, axis=1)
            allowed = (all_grid >= interval[0]) & (all_grid <= interval[1])
            minimum_index = int(values.argmin())
            allowed_index = np.flatnonzero(allowed)[values[allowed].argmin()]
            unrestricted_lower = lipschitz_lower_bound(all_grid, values)
            constrained_lower = lipschitz_lower_bound(all_grid[allowed], values[allowed])
            row = {
                "side": side, "frame": frame, "source_seconds": frame / 20,
                "allowed_shift_interval_m": interval,
                "unrestricted_best_q05_m": float(values[minimum_index]),
                "unrestricted_best_shift_m": float(all_grid[minimum_index]),
                "unrestricted_continuous_lower_bound_m": unrestricted_lower,
                "projection_constrained_best_q05_m": float(values[allowed_index]),
                "projection_constrained_best_shift_m": float(all_grid[allowed_index]),
                "projection_constrained_continuous_lower_bound_m": constrained_lower,
                "q05_5mm_impossible_in_unrestricted_interval": unrestricted_lower > 0.005,
                "q05_5mm_impossible_within_projection_budget": constrained_lower > 0.005,
            }
            rows.append(row)
            arrays[f"{side}_{frame}_shift_m"] = all_grid
            arrays[f"{side}_{frame}_exact_surface_distances_m"] = distances
            print(json.dumps(row), flush=True)
    immutable = [key for key in geometry if key not in {
        "left_vertices", "left_joints", "right_vertices", "right_joints"}]
    if any(not np.array_equal(geometry[key], candidate[key], equal_nan=True) for key in immutable):
        raise ValueError("Candidate changed an undeclared geometry field")
    if max(row["max_rigid_translation_error_m"] for row in rigid_checks.values()) > 1e-12:
        raise ValueError("Candidate is not a per-frame rigid hand translation")
    if any(_sha(Path(path)) != value for path, value in {**hashes, **verified_history}.items()):
        raise ValueError("Inputs or implementation changed during audit")
    output.mkdir(parents=True)
    np.savez_compressed(output / "exact-contact-grid.npz", **arrays)
    result = {
        "schema": "nvidia_v2d.ray_contact_feasibility.v1",
        "source_sha256": hashes, "verified_historical_inputs": len(verified_history),
        "source_unchanged": True, "frame_count": 249,
        "diagnostic_projection_budget_px": 15.0, "diagnostic_thumb_q05_target_m": 0.005,
        "shift_domain_m": [-0.18, 0.18], "maximum_original_grid_gap_m": 0.003,
        "distance_method": "exact closest triangle, unsigned, followed by 5th percentile over fixed thumb vertices",
        "continuous_lower_bound_method": "minimum sampled q05 minus half maximum grid spacing minus 1e-9 m; 1-Lipschitz",
        "training_allowed": False, "geometry_quality_accepted": False,
        "pixel_budget_is_training_gate": False, "source_projection_is_image_ground_truth": False,
        "physical_contact_or_penetration_proven": False,
        "rigid_checks": rigid_checks, "unmodified_geometry_fields": immutable,
        "anchor_count": len(rows),
        "anchors_impossible_even_without_projection_limit": sum(row["q05_5mm_impossible_in_unrestricted_interval"] for row in rows),
        "anchors_impossible_within_projection_budget": sum(row["q05_5mm_impossible_within_projection_budget"] for row in rows),
        "anchors": rows,
        "limitations": [
            "Only the fixed original hand pose/object mesh and one wrist-ray translation per hand are audited.",
            "The thumb role is a fixed nearest-bone vertex mask, not a human-annotated contact patch.",
            "The source hand projection is a reconstruction reference, not 2D ground truth.",
            "The 15px/5mm values define this diagnostic feasibility query only, not repository acceptance thresholds.",
            "The mesh is non-watertight; unsigned triangle distance cannot certify non-penetration or stable grasp.",
        ],
        "files": {"exact-contact-grid.npz": _sha(output / "exact-contact-grid.npz")},
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-root", "ray-audit", "depth-audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    result = audit(**vars(parser.parse_args()))
    print(json.dumps({key: result[key] for key in ("anchor_count", "anchors_impossible_even_without_projection_limit",
                                                   "anchors_impossible_within_projection_budget", "training_allowed")}))


if __name__ == "__main__":
    main()
