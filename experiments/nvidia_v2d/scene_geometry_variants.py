"""Derive ground/scale variants of submission inputs without touching the originals.

Two GT-free scene factors, applied to every object job of an episode identically:

* ground: `scene_up_world` = normal of the dominant MoGe depth plane that supports the
  first-frame objects (RANSAC, hand/object masks excluded), signed so the mean camera is
  above it. The bridge then rotates this up to +z instead of the single-object PCA axis.
* scale: c = 1 / median(observed-frame hand_scale) of the anchor job. Camera and object
  translations, hand camera translations and per-frame hand scales are multiplied by c,
  and `object_scale` becomes c, so the hand returns to native MANO size and the scene
  shrinks with it. Rotations, mesh shape and timing are unchanged.

Each variant writes new hand-quality arrays plus a manifest that records the factors.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from experiments.nvidia_v2d.sharpa_bridge import digest, write_json

DEPTH_RANGE_M = (0.1, 2.5)


def _depth_points(reconstruction: Path, frame: int, camera_to_world: np.ndarray,
                  stride: int = 8) -> np.ndarray:
    from PIL import Image

    k = json.loads((reconstruction / "intrinsics_stable.json").read_text())
    inverse = np.asarray(Image.open(reconstruction / f"depth/{frame:06d}.png"), dtype=np.float64)
    depth = 65535.0 / np.maximum(inverse, 1) - 1  # v2d inverse-depth PNG encoding
    height, width = depth.shape
    excluded = np.zeros((height, width), dtype=bool)
    for track in sorted((reconstruction / "masks").iterdir()):
        mask = track / f"{frame:06d}.png"
        if mask.is_file():
            excluded |= np.asarray(Image.open(mask).convert("L").resize((width, height))) > 0
    v, u = np.mgrid[0:height:stride, 0:width:stride]
    z = depth[v, u]
    keep = ~excluded[v, u] & (z > DEPTH_RANGE_M[0]) & (z < DEPTH_RANGE_M[1])
    u, v, z = u[keep], v[keep], z[keep]
    x = (u * k["width"] / width - k["cx"]) / k["fx"] * z
    y = (v * k["height"] / height - k["cy"]) / k["fy"] * z
    points = np.stack([x, y, z], axis=1)
    return points @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]


def support_plane_up(reconstruction: Path, camera_to_world: np.ndarray, object_points: np.ndarray,
                     *, frames=(0, 10, 20), iterations: int = 3000, tolerance: float = 0.01,
                     touch: float = 0.05, seed: int = 0) -> dict:
    """Dominant depth plane touching the objects' lowest points, normal toward the camera."""
    points = np.concatenate([_depth_points(reconstruction, f, camera_to_world[f])
                             for f in frames if f < len(camera_to_world)])
    camera = camera_to_world[:, :3, 3].mean(axis=0)
    rng = np.random.default_rng(seed)
    best_count, best = 0, None
    for _ in range(iterations):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        if (camera - a) @ normal < 0:
            normal = -normal
        if abs(((object_points - a) @ normal).min()) >= touch:
            continue
        inliers = np.abs((points - a) @ normal) < tolerance
        if inliers.sum() > best_count:
            best_count, best = int(inliers.sum()), inliers
    if best is None:
        raise ValueError("no depth plane supports the first-frame objects")
    plane = points[best]
    centre = plane.mean(axis=0)
    normal = np.linalg.svd(plane - centre)[2][2]
    if (camera - centre) @ normal < 0:
        normal = -normal
    return {"up": normal.tolist(), "inlier_fraction": best_count / len(points),
            "camera_height_m": float((camera - centre) @ normal), "frames": list(frames)}


def anchor_hand_scale(arrays: dict[str, np.ndarray]) -> float:
    values = [np.median(arrays[f"hand_{side}_scale"][arrays[f"hand_{side}_source_code"] == 1])
              for side in ("left", "right") if (arrays[f"hand_{side}_source_code"] == 1).any()]
    if not values:
        raise ValueError("no observed hand frame provides a metric scale")
    return float(np.median(values))


def scale_arrays(arrays: dict[str, np.ndarray], factor: float) -> dict[str, np.ndarray]:
    result = dict(arrays)
    for key in ("camera_to_world_transform", "object_to_world_transform", "object_to_camera_transform",
                "hand_left_wrist_to_world_transform", "hand_right_wrist_to_world_transform",
                "hand_left_wrist_to_camera_transform", "hand_right_wrist_to_camera_transform"):
        if key in result:
            value = result[key].astype(np.float64).copy()
            value[..., :3, 3] *= factor
            result[key] = value
    for side in ("left", "right"):
        result[f"hand_{side}_wrist_trans_in_camera"] = arrays[f"hand_{side}_wrist_trans_in_camera"] * factor
        result[f"hand_{side}_scale"] = arrays[f"hand_{side}_scale"] * factor
    result["object_scale"] = np.asarray(float(np.asarray(arrays["object_scale"]).item()) * factor)
    return result


def build_variant(inputs: Path, output: Path, *, ground: bool, scale: bool) -> dict:
    import trimesh

    manifest = json.loads((inputs / "manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    run = Path(manifest["source_run"])
    state = json.loads((run / "run.json").read_text())
    episodes = []
    for row in manifest["episodes"]:
        anchor_job = Path(row["jobs"][row["anchor"]])
        with np.load(anchor_job / "hand-quality/quality_result.npz", allow_pickle=False) as z:
            anchor = {k: z[k] for k in z.files}
        factor = 1 / anchor_hand_scale(anchor) if scale else 1.0
        ground_record = None
        if ground:
            objects = []
            for name, job in row["jobs"].items():
                with np.load(Path(job) / "hand-quality/quality_result.npz", allow_pickle=False) as z:
                    pose = z["object_to_world_transform"][0]
                mesh = trimesh.load(Path(job) / "reconstruction/result/mesh.obj", force="mesh", process=False)
                objects.append(np.asarray(mesh.vertices) @ pose[:3, :3].T + pose[:3, 3])
            source = [j for j in state["jobs"] if j["run_key"] == row["episode_id"]
                      and j["target_name"] == row["anchor"]][0]
            reconstruction = run / ("preserved" if source["status"] == "failed" else "workspace/jobs") \
                / source["job_id"] / "reconstruction"
            ground_record = support_plane_up(reconstruction, anchor["camera_to_world_transform"],
                                             np.concatenate(objects))
            ground_record["source_reconstruction"] = str(reconstruction)
        jobs = {}
        for name, job in row["jobs"].items():
            job = Path(job)
            destination = output / row["episode_id"] / name
            shutil.copytree(job, destination)
            for stale in ("quality_result.npz", "manifest.json"):
                (destination / "hand-quality" / stale).unlink()
            with np.load(job / "hand-quality/quality_result.npz", allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            arrays = scale_arrays(arrays, factor)
            if ground_record is not None:
                # Up is a direction: uniform scaling leaves it unchanged.
                arrays["scene_up_world"] = np.asarray(ground_record["up"], dtype=np.float64)
            npz = destination / "hand-quality/quality_result.npz"
            np.savez_compressed(npz, **arrays)
            write_json(destination / "hand-quality/manifest.json", {
                "schema": "submission_derived_numerical_input.v1", "geometry_quality_accepted": False,
                "artifacts": [{"path": "quality_result.npz", "sha256": digest(npz)}],
                "derived_from": str(job / "hand-quality/quality_result.npz"),
                "derived_from_sha256": digest(job / "hand-quality/quality_result.npz"),
                "scene_scale_factor": factor, "scene_up_world": None if ground_record is None else ground_record["up"]})
            jobs[name] = str(destination)
        episodes.append({**row, "jobs": jobs, "scene_scale_factor": factor, "ground": ground_record})
    receipt = {**manifest, "derived_from": str(inputs / "manifest.json"),
               "derived_from_sha256": digest(inputs / "manifest.json"),
               "factors": {"ground": "depth_support_plane" if ground else "pca", "scale": "inverse_hand_scale" if scale else 1.0},
               "episodes": episodes}
    write_json(output / "manifest.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ground", action="store_true")
    parser.add_argument("--scale", action="store_true")
    args = parser.parse_args()
    receipt = build_variant(args.inputs, args.output, ground=args.ground, scale=args.scale)
    print(json.dumps({"episodes": len(receipt["episodes"]), "factors": receipt["factors"],
                      "scale": [round(e["scene_scale_factor"], 3) for e in receipt["episodes"]]}))


if __name__ == "__main__":
    main()
