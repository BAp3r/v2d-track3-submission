"""Read reconstruction artifacts on CPU; write a new diagnostic receipt explicitly.

Distances between coordinate origins are diagnostic proxies, never physical
contact measurements. Depth uses the upstream MoGe inverse-depth PNG encoding.
No model, CUDA, video decoding, raw recording access, or product API is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

Array = NDArray[Any]


def _pose(value: Array) -> Array:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("A finite 4x4 pose is required")
    return matrix


def relative_pose_metrics(
    camera_to_world: Array,
    object_to_camera: Array,
    hand_to_camera: Array,
    object_to_world: Array,
    hand_to_world: Array,
) -> dict[str, float]:
    """Check shared-frame invariance, including the supplied world compositions."""
    c, oc, hc, ow, hw = map(
        _pose,
        (camera_to_world, object_to_camera, hand_to_camera, object_to_world, hand_to_world),
    )
    camera_distance = float(np.linalg.norm(hc[:3, 3] - oc[:3, 3]))
    world_distance = float(np.linalg.norm(hw[:3, 3] - ow[:3, 3]))
    return {
        "relative_transform_max_abs_error": float(
            np.max(np.abs(np.linalg.solve(oc, hc) - np.linalg.solve(ow, hw)))
        ),
        "camera_composition_max_abs_error": float(
            max(np.max(np.abs(c @ oc - ow)), np.max(np.abs(c @ hc - hw)))
        ),
        "origin_distance_camera_m": camera_distance,
        "origin_distance_world_m": world_distance,
        "origin_distance_invariance_error_m": abs(camera_distance - world_distance),
    }


def transformed_mesh_depth_bounds(vertices: Array, pose: Array) -> tuple[float, float]:
    """Return camera-space z bounds of the actual packaged object mesh."""
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("Nonempty Nx3 mesh vertices are required")
    if not np.isfinite(points).all():
        raise ValueError("Mesh vertices must be finite")
    matrix = _pose(pose)
    z = points @ matrix[2, :3] + matrix[2, 3]
    return float(z.min()), float(z.max())


def depth_mask_stats(
    encoded_depth_uint16: Array, mask: Array, min_valid_pixels: int = 64
) -> dict[str, Any]:
    """Decode upstream depth; exclude zero/invalid depth from observed statistics."""
    encoded = np.asarray(encoded_depth_uint16)
    selected = np.asarray(mask, dtype=bool)
    if encoded.ndim != 2 or encoded.shape != selected.shape:
        raise ValueError("Depth and mask must have the same 2D shape")
    if encoded.dtype != np.uint16:
        raise ValueError("Expected upstream uint16 inverse-depth encoding")
    if min_valid_pixels < 1:
        raise ValueError("min_valid_pixels must be positive")
    pixels = encoded[selected].astype(np.float64)
    valid = (pixels > 0) & (pixels < 65535)
    depths = 65535.0 / pixels[valid] - 1.0
    return {
        "mask_pixels": int(selected.sum()),
        "valid_pixels": int(valid.sum()),
        "invalid_depth_pixels": int((~valid).sum()),
        "source_observation_valid": bool(valid.sum() >= min_valid_pixels),
        "median_m": float(np.median(depths)) if len(depths) else None,
        "p10_m": float(np.percentile(depths, 10)) if len(depths) else None,
        "p90_m": float(np.percentile(depths, 90)) if len(depths) else None,
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path.name}")
    return value


def _mesh_vertices(path: Path) -> Array:
    with path.open() as stream:
        vertices = [[float(x) for x in line.split()[1:4]] for line in stream if line.startswith("v ")]
    result = np.asarray(vertices, dtype=np.float64)
    transformed_mesh_depth_bounds(result, np.eye(4))
    return result


def _observation(root: Path, frame: int, track_id: int) -> dict[str, Any]:
    depth_path = root / "depth" / f"{frame:06d}.png"
    mask_path = root / "masks" / str(track_id) / f"{frame:06d}.png"
    result: dict[str, Any] = {
        "depth_exists": depth_path.is_file(),
        "mask_exists": mask_path.is_file(),
        "source_observation_valid": False,
    }
    if not (result["depth_exists"] and result["mask_exists"]):
        return result
    try:
        with Image.open(depth_path) as image:
            depth = np.asarray(image)
        with Image.open(mask_path) as image:
            mask = np.asarray(image) > 0
        result.update(depth_mask_stats(depth, mask))
    except (OSError, ValueError) as exc:
        result["read_error"] = str(exc)
    return result


def _translation(path: Path) -> list[float] | None:
    if not path.is_file():
        return None
    try:
        value = np.asarray(_load_json(path)["translation"], dtype=np.float64)
        return value.tolist() if value.shape == (3,) and np.isfinite(value).all() else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _distribution(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    x = np.asarray(values)
    return {
        "count": len(values),
        "min": float(x.min()),
        "p10": float(np.percentile(x, 10)),
        "median": float(np.median(x)),
        "p90": float(np.percentile(x, 90)),
        "max": float(x.max()),
    }


def _quarter_summary(rows: list[dict[str, Any]], lo: int, hi: int) -> dict[str, Any]:
    selected = rows[lo:hi]
    metrics: dict[str, list[float]] = {}
    for row in selected:
        for name in ("raw_object_pose_jump_m", "object_z_m", "mesh_front_minus_depth_median_m"):
            if row.get(name) is not None:
                metrics.setdefault(name, []).append(row[name])
        for side, hand in row["hands"].items():
            for name, value in hand.items():
                if isinstance(value, int | float) and not isinstance(value, bool):
                    metrics.setdefault(f"{side}.{name}", []).append(value)
    return {
        "start_frame": lo,
        "end_frame_exclusive": hi,
        "frames": len(selected),
        "source_observation_valid_frames": sum(
            row["object_depth_observation"]["source_observation_valid"] for row in selected
        ),
        "distributions": {key: _distribution(value) for key, value in metrics.items()},
    }


def diagnose_reconstruction(root: Path) -> dict[str, Any]:
    """Read one completed official reconstruction; keep all inputs unchanged."""
    root = root.resolve()
    manifest_path = root / "result" / "manifest.json"
    sources = _load_json(manifest_path).get("sources", {}) if manifest_path.is_file() else {}
    object_id = int(sources.get("object_track_id", 1))
    vertices = _mesh_vertices(root / "result" / "mesh.obj")
    with np.load(root / "result" / "result.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    n_frames = len(arrays["object_to_camera_transform"])
    rows: list[dict[str, Any]] = []
    previous_raw: Array | None = None
    hand_roots: dict[str, Path] = {}
    for side in ("left", "right"):
        if source := sources.get(f"{side}_hand_dir"):
            path = Path(source).resolve()
            if path.is_relative_to(root):
                hand_roots[side] = path
    for frame in range(n_frames):
        obj = arrays["object_to_camera_transform"][frame]
        valid_object = bool(arrays["object_is_valid"][frame])
        raw = _translation(root / "poses" / f"{frame:06d}.json")
        jump = None
        if raw is not None and previous_raw is not None:
            jump = float(np.linalg.norm(np.asarray(raw) - previous_raw))
        previous_raw = np.asarray(raw) if raw is not None else None
        observation = _observation(root, frame, object_id)
        bounds = transformed_mesh_depth_bounds(vertices, obj) if valid_object else None
        median = observation.get("median_m")
        row: dict[str, Any] = {
            "frame": frame,
            "object_pose_valid": valid_object,
            "camera_pose_valid": bool(arrays["camera_is_valid"][frame]),
            "object_z_m": float(obj[2, 3]) if valid_object else None,
            "raw_object_translation": raw,
            "raw_object_translation_valid": raw is not None,
            "raw_object_pose_jump_m": jump,
            "mesh_camera_z_bounds_m": list(bounds) if bounds else None,
            "object_depth_observation": observation,
            "mesh_front_minus_depth_median_m": (
                bounds[0] - median if bounds is not None and median is not None else None
            ),
            "hands": {},
        }
        for side in ("left", "right"):
            prefix = f"hand_{side}"
            if (
                not valid_object
                or not row["camera_pose_valid"]
                or prefix + "_is_valid" not in arrays
                or not arrays[prefix + "_is_valid"][frame]
            ):
                continue
            metrics = relative_pose_metrics(
                arrays["camera_to_world_transform"][frame],
                obj,
                arrays[prefix + "_wrist_to_camera_transform"][frame],
                arrays["object_to_world_transform"][frame],
                arrays[prefix + "_wrist_to_world_transform"][frame],
            )
            metrics["wrist_z_m"] = float(arrays[prefix + "_wrist_to_camera_transform"][frame, 2, 3])
            metrics["hand_scale"] = float(arrays[prefix + "_scale"][frame])
            if side in hand_roots:
                record_path = hand_roots[side] / f"{frame:06d}.json"
                if record_path.is_file():
                    record = _load_json(record_path)
                    for key in ("dz", "n_pixels", "scale"):
                        if key in record.get("diagnostics", {}):
                            metrics[f"depth_alignment_{key}"] = float(record["diagnostics"][key])
            row["hands"][side] = metrics
        rows.append(row)
    boundaries = [round(n_frames * quarter / 4) for quarter in range(5)]
    samples = sorted(set(np.linspace(0, n_frames - 1, min(9, n_frames), dtype=int).tolist()))
    return {
        "schema": "nvidia_v2d.hand_object_diagnostics.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "reconstruction_root": str(root),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (root / "result/result.npz", root / "result/mesh.obj", manifest_path)
            if path.is_file()
        },
        "frame_count": n_frames,
        "mesh_vertex_count": len(vertices),
        "depth_encoding": "upstream MoGe uint16: depth=65535/pixel-1",
        "interpretation": (
            "Estimated metric units, not ground truth. Wrist-to-object-origin distance is not "
            "a contact or surface-distance measurement. Positive mesh_front_minus_depth_median_m "
            "means the entire transformed mesh lies behind the observed median depth; masks, "
            "occlusion and depth quality must also be checked."
        ),
        "per_frame": rows,
        "sample_frames": samples,
        "samples": [rows[index] for index in samples],
        "quarter_summaries": [
            _quarter_summary(rows, boundaries[i], boundaries[i + 1]) for i in range(4)
        ],
        "largest_raw_pose_jumps": sorted(
            [
                {"frame": row["frame"], "translation_jump_m": row["raw_object_pose_jump_m"]}
                for row in rows
                if row["raw_object_pose_jump_m"] is not None
            ],
            key=lambda value: value["translation_jump_m"],
            reverse=True,
        )[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New receipt outside input tree")
    args = parser.parse_args()
    source = args.reconstruction_root.resolve()
    destination = args.output.resolve()
    if destination.is_relative_to(source) or args.output.exists() or args.output.is_symlink():
        parser.error("Output must be a new file outside the reconstruction input tree")
    report = diagnose_reconstruction(source)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, "w") as stream:
        stream.write(payload)
    print(f"Wrote {report['frame_count']} frame diagnostics to {destination}")


if __name__ == "__main__":
    main()
