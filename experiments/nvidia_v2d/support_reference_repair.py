"""Evidence-bound static-support reference repair; CPU only and no policy input.

The initial tabletop plane fixes a coordinate gauge, not metric ground truth.
Independent wood-texture matches constrain camera drift.  A fixed, physically
stable collision-mesh pose is fitted to released-object silhouettes.  Original
inputs and all frames through the final source contact are preserved.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from experiments.nvidia_v2d.finite_support import intersect_plane, support_usda
from experiments.nvidia_v2d.static_support_depth import project_scene


@dataclass(frozen=True)
class RepairConfig:
    last_unchanged_frame: int = 203
    static_start_frame: int = 210
    camera_anchor_frame: int = 0
    fit_frames: tuple[int, ...] = (210, 215, 220, 225, 230, 235, 240, 248)
    fit_stride: int = 8
    evaluation_stride: int = 4
    camera_min_inliers: int = 20
    camera_max_p95_px: float = 6.0
    max_frame_iou_loss: float = 0.03
    min_mean_iou_improvement: float = 0.03
    max_transition_speed_m_s: float = 1.0
    max_transition_angular_speed_rad_s: float = 4.0
    max_transition_acceleration_m_s2: float = 15.0

    def __post_init__(self):
        if not 0 <= self.last_unchanged_frame < self.static_start_frame:
            raise ValueError("Repair needs a nonempty transition after an unchanged prefix")
        if not self.fit_frames or min(self.fit_frames) < self.static_start_frame:
            raise ValueError("Silhouette fit frames must belong to the static suffix")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def quintic_blend(t):
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    return t**3 * (10.0 + t * (-15.0 + 6.0 * t))


def transform_points(points, transforms):
    points = np.asarray(points, dtype=float)
    shape = points.shape
    flat = points.reshape(shape[0], -1, 3)
    result = np.einsum("tij,tkj->tki", transforms[:, :3, :3], flat)
    result += transforms[:, None, :3, 3]
    return result.reshape(shape)


def transform_rotations(wxyz, transforms):
    values = np.asarray(wxyz, dtype=float)
    shape = values.shape
    r = Rotation.from_quat(values.reshape(-1, 4), scalar_first=True).as_matrix()
    r = r.reshape(shape[0], -1, 3, 3)
    result = np.einsum("tij,tkjl->tkil", transforms[:, :3, :3], r)
    result = Rotation.from_matrix(result.reshape(-1, 3, 3)).as_quat(scalar_first=True)
    return result.reshape(shape)


def recover_mano_rest_root(trans, camera_to_scene, camera_trans):
    """Recover the constant unscaled MANO root from the original FK composition."""
    r = camera_to_scene[:, :3, :3]
    design = (np.eye(3)[None] - r).reshape(-1, 3)
    target = (
        np.einsum("tij,tj->ti", r, camera_trans) + camera_to_scene[:, :3, 3] - trans
    ).reshape(-1)
    root, _, rank, singular = np.linalg.lstsq(design, target, rcond=None)
    error = float(np.max(np.abs(design @ root - target)))
    if rank != 3 or error > 1e-5:
        raise ValueError("Cannot prove the original MANO root/translation convention")
    return root, {"max_equation_error_m": error, "singular_values": singular.tolist()}


def transform_robot_fields(row, gauge, first_changed_frame=204):
    """Rigidly rebase physical robot frames; preserve the inertial universe frame."""
    for side in ("left", "right"):
        for suffix, fn in (("wrist_position", transform_points), ("wrist_wxyz", transform_rotations)):
            key = f"robot_{side}_{suffix}"
            original = np.asarray(row[key])
            changed = fn(original, gauge)
            changed[:first_changed_frame] = original[:first_changed_frame]
            row[key] = changed.tolist()
        key = f"robot_{side}_frames"
        original = np.asarray(row[key])
        changed = np.concatenate(
            (transform_points(original[..., :3], gauge), transform_rotations(original[..., 3:], gauge)),
            axis=-1,
        )
        for index, name in enumerate(row[f"{side}_robot_frame_names"]):
            if name == "universe":
                changed[:, index] = original[:, index]
        changed[:first_changed_frame] = original[:first_changed_frame]
        row[key] = changed.tolist()


def apply_scene_gauge(row, gauge, *, first_changed_frame, mano_rest_roots):
    """Explicitly transform every world-space hand/robot field, preserving sentinels."""
    for side in ("left", "right"):
        prefix = f"mano_{side}_"
        ids = np.asarray(row[prefix + "object_contact_part_ids"])
        for suffix in ("joints", "link_contact_positions", "object_contact_positions"):
            old = np.asarray(row[prefix + suffix])
            new = transform_points(old, gauge)
            if "contact" in suffix:
                new[ids == 0] = 0.0
            new[:first_changed_frame] = old[:first_changed_frame]
            row[prefix + suffix] = new.tolist()
        for suffix in ("link_contact_normals", "object_contact_normals"):
            old = np.asarray(row[prefix + suffix])
            new = np.einsum("tij,tkj->tki", gauge[:, :3, :3], old)
            new[:first_changed_frame] = old[:first_changed_frame]
            row[prefix + suffix] = new.tolist()
        old = np.asarray(row[prefix + "joints_wxyz"])
        new = transform_rotations(old, gauge)
        new[:first_changed_frame] = old[:first_changed_frame]
        row[prefix + "joints_wxyz"] = new.tolist()
        old = np.asarray(row[prefix + "global_orient"])
        new = Rotation.from_matrix(gauge[:, :3, :3] @ Rotation.from_rotvec(old).as_matrix()).as_rotvec()
        new[:first_changed_frame] = old[:first_changed_frame]
        row[prefix + "global_orient"] = new.tolist()
        old = np.asarray(row[prefix + "trans"])
        root = np.asarray(mano_rest_roots[side])
        new = transform_points(old + root, gauge) - root
        new[:first_changed_frame] = old[:first_changed_frame]
        row[prefix + "trans"] = new.tolist()
    transform_robot_fields(row, gauge, first_changed_frame)


def load_support_solution(output: Path):
    """Load the explicit common gauge and separate object correction contract."""
    manifest = json.loads((output / "manifest.json").read_text())
    arrays = dict(np.load(output / "object-transforms.npz", allow_pickle=False))
    return {"manifest": manifest, **arrays}


def apply_support_repair(cells, solution):
    """Compose onto external IK/contact edits without replacing finger or robot FK.

    ``cells`` is one Arrow row as a mapping of column names to values.  ``solution``
    is returned by :func:`load_support_solution`, or is its output directory.
    Input object poses must still be in the original gauge, preventing double use.
    """
    if isinstance(solution, (str, Path)):
        solution = load_support_solution(Path(solution))
    old = solution["original_object_to_world"][:, 0]
    positions = np.asarray(cells["object_body_position"])[:, 0]
    rotations = Rotation.from_quat(np.asarray(cells["object_body_wxyz"])[:, 0], scalar_first=True).as_matrix()
    if not np.allclose(positions, old[:, :3, 3], atol=1e-6) or not np.allclose(rotations, old[:, :3, :3], atol=1e-6):
        raise ValueError("Support solution requires its original object reference; possible double application")
    manifest = solution["manifest"]
    first = manifest["config"]["last_unchanged_frame"] + 1
    result = copy.deepcopy(cells)
    apply_scene_gauge(result, solution["scene_gauge_delta_world"], first_changed_frame=first,
                      mano_rest_roots=manifest["mano_rest_roots"])
    repaired = solution["repaired_object_to_world"][:, 0]
    fields = {"object_body_position": repaired[:, None, :3, 3],
              "object_body_wxyz": Rotation.from_matrix(repaired[:, :3, :3]).as_quat(scalar_first=True)[:, None],
              "object_root_position": repaired[:, :3, 3],
              "object_root_axis_angle": Rotation.from_matrix(repaired[:, :3, :3]).as_rotvec()}
    for key, values in fields.items():
        values = values.copy()
        values[:first] = cells[key][:first]
        result[key] = values.tolist()
    for side in ("left", "right"):
        result[f"mano_{side}_tips_distance"] = solution[f"repaired_{side}_tips_distance"].tolist()
    return result


def static_support_path(original, gauge, static_pose, collision_vertices, config):
    """C2 blend to a constant supported pose, with a nonnegative clearance path."""
    repaired = original.copy()
    rebased = gauge @ original
    for i in range(config.last_unchanged_frame + 1, len(original)):
        if i >= config.static_start_frame:
            repaired[i] = static_pose
            continue
        alpha = float(quintic_blend((i - config.last_unchanged_frame) /
                                    (config.static_start_frame - config.last_unchanged_frame)))
        r0, r1 = rebased[i, :3, :3], static_pose[:3, :3]
        delta = Rotation.from_matrix(r1 @ r0.T).as_rotvec()
        rotation = Rotation.from_rotvec(alpha * delta).as_matrix() @ r0
        translation = (1 - alpha) * rebased[i, :3, 3] + alpha * static_pose[:3, 3]
        old_bottom = float(np.min(collision_vertices @ r0[2] + rebased[i, 2, 3]))
        if old_bottom < -0.002:
            raise ValueError("Transition input penetrates support; cannot silently clamp it")
        clearance = (1 - alpha) * max(old_bottom, 0.0)
        translation[2] = clearance - np.min(collision_vertices @ rotation[2])
        repaired[i, :3, :3], repaired[i, :3, 3] = rotation, translation
    return repaired


def support_metrics(vertices, pose, *, com_local=None, tolerance=0.001,
                    table_center_xy=None, table_radius=None):
    """COM stability from bottom points that actually lie on the finite table.

    Non-contacting geometry can overhang the edge.  Including out-of-table
    bottom points in the support polygon would instead invent support.
    """
    vertices = np.asarray(vertices) @ pose[:3, :3].T + pose[:3, 3]
    com = pose[:3, :3] @ (np.zeros(3) if com_local is None else com_local) + pose[:3, 3]
    low = vertices[vertices[:, 2] <= vertices[:, 2].min() + tolerance]
    count = len(low)
    if (table_center_xy is None) != (table_radius is None):
        raise ValueError("Finite support needs both center and radius")
    if table_radius is not None:
        if table_radius <= 0:
            raise ValueError("Finite support radius must be positive")
        low = low[np.linalg.norm(low[:, :2] - table_center_xy, axis=1) <= table_radius]
    unique = np.unique(low[:, :2], axis=0)
    valid = len(unique) >= 3 and np.linalg.matrix_rank(unique - unique.mean(axis=0)) == 2
    if valid:
        hull = ConvexHull(unique)
        signed = hull.equations[:, :2] @ com[:2] + hull.equations[:, 2]
    return {
        "bottom_z_m": float(vertices[:, 2].min()),
        "contact_band_m": tolerance,
        "contact_band_vertex_count": count,
        "supported_contact_band_vertex_count": len(low),
        "com_xy_margin_m": float(-np.max(signed)) if valid else None,
        "com_inside_contact_hull": bool(np.all(signed <= 1e-8)) if valid else False,
        "contact_hull_xy_m": unique[hull.vertices].tolist() if valid else [],
    }


def raster_mask(vertices, faces, pose, camera, intrinsics, size):
    points = vertices @ pose[:3, :3].T + pose[:3, 3]
    uv = project_scene(points, camera, intrinsics)
    result = np.zeros(size, dtype=np.uint8)
    for triangle in np.round(uv[faces]).astype(np.int32):
        cv2.fillConvexPoly(result, triangle, 1)
    return result.astype(bool)


def silhouette_metrics(rendered, observed, stride):
    union = np.count_nonzero(rendered | observed)
    if union == 0:
        raise ValueError("Empty object masks cannot constrain pose")
    contours = []
    for mask in (rendered, observed):
        eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        contours.append(mask & ~eroded)
    distances = []
    for a, b in ((0, 1), (1, 0)):
        distance = cv2.distanceTransform((~contours[b]).astype(np.uint8), cv2.DIST_L2, 5)
        distances.extend((distance[contours[a]] * stride).tolist())
    return {"iou": float(np.count_nonzero(rendered & observed) / union),
            "boundary_distance_median_px": float(np.median(distances)),
            "boundary_distance_p95_px": float(np.quantile(distances, .95))}


def table_features(reconstruction, frame, sift):
    rgb = cv2.imread(str(reconstruction / "frames" / f"{frame:06d}.png"))[::2, ::2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([4, 55, 30]), np.array([30, 255, 255]))
    roi = np.zeros_like(mask)
    roi[325:750, 290:760] = 255
    mask &= roi
    for role in (1, 2, 3):
        excluded = cv2.imread(str(reconstruction / "masks" / str(role) / f"{frame:06d}.png"), 0)[::2, ::2]
        mask[cv2.dilate(excluded, np.ones((9, 9), np.uint8)) > 0] = 0
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
    keypoints, descriptors = sift.detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY), mask)
    return rgb, keypoints, descriptors, mask


def estimate_cameras(reconstruction, cameras, intrinsics, config, output):
    """Planar wood-texture PnP, with all matches and robust inlier decisions saved."""
    cv2.setNumThreads(2)
    sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=.012)
    anchor, keys, descriptors, anchor_mask = table_features(reconstruction, config.camera_anchor_frame, sift)
    anchor_uv = np.asarray([key.pt for key in keys]) * 2
    anchor_world = intersect_plane(anchor_uv, intrinsics, cameras[config.camera_anchor_frame], 0.0)
    k = np.array([[intrinsics[0] / 2, 0, intrinsics[2] / 2],
                  [0, intrinsics[1] / 2, intrinsics[3] / 2], [0, 0, 1.0]])
    refined = cameras.copy()
    rows, bundles = [], []
    for frame in range(config.static_start_frame, len(cameras)):
        rgb, kp, desc, _ = table_features(reconstruction, frame, sift)
        matches = cv2.BFMatcher().knnMatch(descriptors, desc, k=2)
        matches = [m for m, n in matches if m.distance < .7 * n.distance]
        source = np.asarray([anchor_uv[m.queryIdx] / 2 for m in matches])
        observed = np.asarray([kp[m.trainIdx].pt for m in matches])
        world = np.asarray([anchor_world[m.queryIdx] for m in matches])
        cv2.setRNGSeed(42 + frame)
        homography, inlier = cv2.findHomography(source, observed, cv2.RANSAC, 2.0, maxIters=4000, confidence=.999)
        if homography is None or inlier is None or inlier.sum() < config.camera_min_inliers:
            raise ValueError(f"Frame {frame} lacks independent table matches")
        indices = np.flatnonzero(inlier[:, 0])
        w2c = np.linalg.inv(cameras[frame])
        rvec, tvec = cv2.Rodrigues(w2c[:3, :3])[0], w2c[:3, 3].copy()
        # Planar solvePnPRansac can pick the mirrored branch.  RANSAC is done in
        # image homography space; a nearby physical branch initializes direct PnP.
        ok, rvec, tvec = cv2.solvePnP(world[indices], observed[indices], k, None,
                                    rvec, tvec, True, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise ValueError(f"PnP failed at frame {frame}")
        rvec, tvec = cv2.solvePnPRefineLM(world[indices], observed[indices], k, None, rvec, tvec)
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = cv2.Rodrigues(rvec)[0], tvec[:, 0]
        refined[frame] = np.linalg.inv(pose)
        errors = np.linalg.norm(project_scene(world[indices], refined[frame], intrinsics) - observed[indices] * 2, axis=1)
        original_errors = np.linalg.norm(project_scene(world[indices], cameras[frame], intrinsics) - observed[indices] * 2, axis=1)
        rows.append({"frame": frame, "matches": len(matches), "inliers": len(indices),
                     "original_p95_px": float(np.quantile(original_errors, .95)),
                     "raw_p95_px": float(np.quantile(errors, .95))})
        bundles.append((world, observed * 2, inlier[:, 0].astype(bool), source * 2))
        if frame in (210, 230, 248):
            vis = cv2.drawMatches(anchor, keys, rgb, kp, [matches[i] for i in indices], None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imwrite(str(output / f"table-inliers-{frame:06d}.jpg"), vis)
    first = config.static_start_frame
    raw_delta = refined[first:] @ np.linalg.inv(cameras[first:])
    vectors = np.column_stack((raw_delta[:, :3, 3], Rotation.from_matrix(raw_delta[:, :3, :3]).as_rotvec()))
    selected, selected_window = raw_delta, 1
    for window in (5, 9):
        smoothed = savgol_filter(vectors, window, 2, axis=0)
        candidate = np.tile(np.eye(4), (len(raw_delta), 1, 1))
        candidate[:, :3, 3] = smoothed[:, :3]
        candidate[:, :3, :3] = Rotation.from_rotvec(smoothed[:, 3:]).as_matrix()
        estimates = candidate @ cameras[first:]
        p95 = [float(np.quantile(np.linalg.norm(project_scene(b[0][b[2]], estimates[i], intrinsics) - b[1][b[2]], axis=1), .95))
               for i, b in enumerate(bundles)]
        if max(p95) <= config.camera_max_p95_px:
            selected, selected_window = candidate, window
    gauge = np.tile(np.eye(4), (len(cameras), 1, 1))
    gauge[first:] = selected
    for frame in range(config.last_unchanged_frame + 1, first):
        alpha = float(quintic_blend((frame - config.last_unchanged_frame) / (first - config.last_unchanged_frame)))
        gauge[frame, :3, 3] = alpha * selected[0, :3, 3]
        gauge[frame, :3, :3] = Rotation.from_rotvec(alpha * Rotation.from_matrix(selected[0, :3, :3]).as_rotvec()).as_matrix()
    result = gauge @ cameras
    offsets = np.r_[0, np.cumsum([len(b[0]) for b in bundles])]
    for i, row in enumerate(rows):
        b = bundles[i]
        row["used_p95_px"] = float(np.quantile(np.linalg.norm(project_scene(b[0][b[2]], result[row["frame"]], intrinsics) - b[1][b[2]], axis=1), .95))
    if max(row["used_p95_px"] for row in rows) > config.camera_max_p95_px:
        raise ValueError("Camera correction exceeds the image correspondence residual budget")
    np.savez_compressed(output / "table-correspondences.npz", offsets=offsets,
                        source_world=np.concatenate([b[0] for b in bundles]),
                        target_pixels=np.concatenate([b[1] for b in bundles]),
                        inlier=np.concatenate([b[2] for b in bundles]),
                        source_pixels=np.concatenate([b[3] for b in bundles]),
                        raw_camera_to_scene=refined, anchor_mask=anchor_mask)
    return result, gauge, {"frames": rows, "smoothing_window": selected_window,
                           "anchor_keypoint_count": len(keys), "plane_top_gauge_m": 0.0}


def fit_static_pose(visual, collision, cameras, intrinsics, masks, original, config):
    import trimesh

    poses, probabilities = trimesh.poses.compute_stable_poses(
        collision, center_mass=np.zeros(3), sigma=0.0, n_samples=1, threshold=.001, seed=42)
    mean_rotation = Rotation.from_matrix(original[config.static_start_frame:, :3, :3]).mean().as_matrix()
    candidates = [i for i, pose in enumerate(poses) if
                  np.dot(pose[2, :3], mean_rotation[2]) > np.cos(np.deg2rad(45))]
    if not candidates:
        raise ValueError("No upright stable collision pose agrees with the image reference")
    k = intrinsics / config.fit_stride
    targets = {i: masks[i][::config.fit_stride, ::config.fit_stride] for i in config.fit_frames}
    origin = np.median(original[config.static_start_frame:, :2, 3], axis=0)
    records = []
    for index in candidates:
        stable = poses[index]
        relative = mean_rotation @ stable[:3, :3].T
        yaw = np.arctan2(relative[1, 0] - relative[0, 1], relative[0, 0] + relative[1, 1])

        def make_pose(parameters, stable=stable, yaw=yaw):
            pose = stable.copy()
            pose[:3, :3] = Rotation.from_euler("z", yaw + parameters[2]).as_matrix() @ stable[:3, :3]
            pose[:2, 3] = origin + parameters[:2]
            return pose

        def objective(parameters, make_pose=make_pose):
            pose = make_pose(parameters)
            scores = []
            for frame, target in targets.items():
                rendered = raster_mask(visual.vertices, visual.faces, pose, cameras[frame], k, target.shape)
                scores.append(np.count_nonzero(rendered & target) / np.count_nonzero(rendered | target))
            return 1.0 - np.mean(scores)

        fit = minimize(objective, np.zeros(3), method="Powell",
                       bounds=[(-.15, .15), (-.15, .15), (-.5, .5)],
                       options={"maxiter": 40, "xtol": 1e-5, "ftol": 1e-5})
        records.append({"stable_pose_index": index, "convex_support_probability": float(probabilities[index]),
                        "pose": make_pose(fit.x).tolist(), "mean_fit_iou": float(1 - fit.fun),
                        "success": bool(fit.success), "evaluations": fit.nfev})
    chosen = max(records, key=lambda item: item["mean_fit_iou"])
    return np.asarray(chosen["pose"]), records


def motion_metrics(poses, fps, first):
    velocity = np.diff(poses[:, :3, 3], axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    rotation = Rotation.from_matrix(poses[:, :3, :3])
    angular = (rotation[1:] * rotation[:-1].inv()).magnitude() * fps
    return {"max_speed_m_s": float(np.linalg.norm(velocity[first - 1:], axis=1).max()),
            "max_acceleration_m_s2": float(np.linalg.norm(acceleration[first - 2:], axis=1).max()),
            "max_angular_speed_rad_s": float(angular[first - 1:].max()),
            "boundary_samples": [{"frame": i, "velocity_xyz_m_s": velocity[i - 1].tolist(),
                                   "acceleration_xyz_m_s2": acceleration[i - 2].tolist()}
                                  for i in (203, 204, 209, 210, 211) if 2 <= i < len(poses)]}


def repair(*, parquet, quality, geometry, reconstruction, visual_mesh, collision_mesh,
           support_manifest, output, config=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import trimesh

    config = config or RepairConfig()
    if (output / "manifest.json").exists():
        raise FileExistsError("A completed candidate must not be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    camera_output = output / "camera-evidence"
    camera_output.mkdir(exist_ok=True)
    table = pq.read_table(parquet)
    if table.num_rows != 1:
        raise ValueError("Exactly one source episode is required")
    row = table.to_pylist()[0]
    original_row = copy.deepcopy(row)
    arrays = dict(np.load(geometry, allow_pickle=False))
    q = dict(np.load(quality, allow_pickle=False))
    original = arrays["object_to_world_transform"].copy()
    if original.shape != (249, 4, 4) or config.static_start_frame >= len(original):
        raise ValueError("This evidence configuration requires the verified 249-frame episode")
    if row["fps"] != 20.0:
        raise ValueError("Source timeline differs from the verified 20 fps episode")
    for side in ("left", "right"):
        if np.any(np.asarray(row[f"mano_{side}_object_contact_part_ids"])[config.last_unchanged_frame + 1:]):
            raise ValueError("Repair suffix contains active original source contacts")
    input_paths = [parquet, quality, geometry, visual_mesh, collision_mesh, support_manifest, Path(__file__),
                   Path(__file__).with_name("finite_support.py"), Path(__file__).with_name("static_support_depth.py")]
    used_frames = [config.camera_anchor_frame, *range(config.last_unchanged_frame, len(original))]
    for frame in used_frames:
        input_paths.append(reconstruction / "frames" / f"{frame:06d}.png")
        input_paths.extend(reconstruction / "masks" / str(role) / f"{frame:06d}.png" for role in (1, 2, 3))
    source_hashes = {str(path.resolve()): digest(path) for path in input_paths}
    cameras = arrays["source_to_scene_transform"][None] @ q["camera_to_world_transform"]
    intrinsics = q["camera_intrinsics"].astype(float)
    corrected_cameras, gauge, camera_audit = estimate_cameras(reconstruction, cameras, intrinsics, config, camera_output)
    visual = trimesh.load(visual_mesh, process=False)
    collision = trimesh.load(collision_mesh, process=False)
    masks = {frame: cv2.imread(str(reconstruction / "masks/1" / f"{frame:06d}.png"), 0) > 0
             for frame in range(config.last_unchanged_frame, len(original))}
    static_pose, pose_fit = fit_static_pose(visual, collision, corrected_cameras, intrinsics, masks, gauge @ original, config)
    repaired = static_support_path(original, gauge, static_pose, collision.vertices, config)
    roots, root_proof = {}, {}
    for side in ("left", "right"):
        roots[side], root_proof[side] = recover_mano_rest_root(
            np.asarray(row[f"mano_{side}_trans"]), cameras, q[f"hand_{side}_wrist_trans_in_camera"])
    apply_scene_gauge(row, gauge, first_changed_frame=config.last_unchanged_frame + 1, mano_rest_roots=roots)
    for key, values in (
        ("object_body_position", repaired[:, None, :3, 3]),
        ("object_body_wxyz", Rotation.from_matrix(repaired[:, :3, :3]).as_quat(scalar_first=True)[:, None]),
        ("object_root_position", repaired[:, :3, 3]),
        ("object_root_axis_angle", Rotation.from_matrix(repaired[:, :3, :3]).as_rotvec()),
    ):
        values[:config.last_unchanged_frame + 1] = original_row[key][:config.last_unchanged_frame + 1]
        row[key] = values.tolist()
    for side in ("left", "right"):
        joints = np.asarray(row[f"mano_{side}_joints"])
        distances = np.asarray(row[f"mano_{side}_tips_distance"])
        for frame in range(config.last_unchanged_frame + 1, len(original)):
            local = (joints[frame, [4, 8, 12, 16, 20]] - repaired[frame, :3, 3]) @ repaired[frame, :3, :3]
            distances[frame] = trimesh.proximity.closest_point(visual, local)[1]
        row[f"mano_{side}_tips_distance"] = distances.tolist()
    # Geometry companion uses the original names and an explicit time-varying camera override.
    for side in ("left", "right"):
        arrays[side + "_vertices"] = transform_points(arrays[side + "_vertices"], gauge)
        arrays[side + "_joints"] = np.asarray(row[f"mano_{side}_joints"])
        arrays[side + "_joints_wxyz"] = np.asarray(row[f"mano_{side}_joints_wxyz"])
    arrays["object_to_world_transform"] = repaired
    arrays["camera_to_scene_transform"] = corrected_cameras
    arrays["scene_gauge_delta_world"] = gauge
    pq.write_table(pa.Table.from_pylist([row], schema=table.schema), output / "data.parquet", compression="zstd")
    np.savez_compressed(output / "geometry.npz", **arrays)
    np.savez_compressed(output / "object-transforms.npz", source_frame_index=np.arange(len(original)),
                        timestamps_s=np.arange(len(original)) / row["fps"],
                        original_object_to_world=original[:, None], repaired_object_to_world=repaired[:, None],
                        object_delta_world=(repaired @ np.linalg.inv(original))[:, None],
                        scene_gauge_delta_world=gauge, original_camera_to_scene=cameras,
                        repaired_camera_to_scene=corrected_cameras,
                        repaired_left_tips_distance=np.asarray(row["mano_left_tips_distance"]),
                        repaired_right_tips_distance=np.asarray(row["mano_right_tips_distance"]))
    support = json.loads(support_manifest.read_text())
    (output / "support.usda").write_text(support_usda(support["fit"]["center_xy_m"],
                                                   support["fit"]["radius_m"], 0.0, support["thickness_m"]))
    projection = []
    overlays = []
    overlay_output = output / "projection-evidence"
    overlay_output.mkdir(exist_ok=True)
    for frame in range(config.last_unchanged_frame, len(original)):
        stride = config.evaluation_stride
        target = masks[frame][::stride, ::stride]
        before = raster_mask(visual.vertices, visual.faces, original[frame], cameras[frame], intrinsics / stride, target.shape)
        after = raster_mask(visual.vertices, visual.faces, repaired[frame], corrected_cameras[frame], intrinsics / stride, target.shape)
        projection.append({"frame": frame, "fit_frame": frame in config.fit_frames,
                           "before": silhouette_metrics(before, target, stride),
                           "after": silhouette_metrics(after, target, stride)})
        if frame in {*config.fit_frames, 203, 204, 209}:
            rgb = cv2.imread(str(reconstruction / "frames" / f"{frame:06d}.png"))[::stride, ::stride].copy()
            for mask, color in ((before, (0, 0, 255)), (after, (255, 255, 0))):
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rgb, contours, -1, color, 1)
            cv2.putText(rgb, f"{frame / row['fps']:.2f}s  red original / cyan repaired", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 1)
            cv2.imwrite(str(overlay_output / f"overlay-{frame:06d}.png"), rgb)
            if frame in (203, 204, 209, 210, 215, 220, 230, 240, 248):
                overlays.append(rgb)
    cv2.imwrite(str(output / "projection-overlays.jpg"), np.concatenate([np.concatenate(overlays[i:i + 3], axis=1) for i in range(0, len(overlays), 3)]))
    released = [item for item in projection if item["frame"] >= config.static_start_frame]
    changes = np.array([item["after"]["iou"] - item["before"]["iou"] for item in released])
    static_support = support_metrics(collision.vertices, static_pose,
                                     table_center_xy=support["fit"]["center_xy_m"], table_radius=support["fit"]["radius_m"])
    exact_support = support_metrics(collision.vertices, static_pose, tolerance=1e-7,
                                    table_center_xy=support["fit"]["center_xy_m"], table_radius=support["fit"]["radius_m"])
    radius_sensitivity = [
        {"radius_reduction_m": reduction, "contact_band_m": tolerance,
         **support_metrics(collision.vertices, static_pose, tolerance=tolerance, table_center_xy=support["fit"]["center_xy_m"],
                           table_radius=support["fit"]["radius_m"] - reduction)}
        for tolerance in (1e-7, .001) for reduction in (.002, .005, .010)
    ]
    motion = motion_metrics(repaired, row["fps"], config.last_unchanged_frame + 1)
    radial = np.linalg.norm((collision.vertices @ static_pose[:3, :3].T + static_pose[:3, 3])[:, :2] - support["fit"]["center_xy_m"], axis=1)
    hand_projection = {}
    wrist_changes = {}
    for side in ("left", "right"):
        old = np.asarray(original_row[f"mano_{side}_joints"])
        new = np.asarray(row[f"mano_{side}_joints"])
        error = [np.linalg.norm(project_scene(old[i], cameras[i], intrinsics) - project_scene(new[i], corrected_cameras[i], intrinsics), axis=1).max() for i in range(len(old))]
        hand_projection[side] = float(max(error))
        wrist_changes[side] = {"max_translation_m": float(np.linalg.norm(new[:, 0] - old[:, 0], axis=1).max()),
                               "release_translation_p95_m": float(np.quantile(np.linalg.norm(new[210:, 0] - old[210:, 0], axis=1), .95))}
    serialized_row = pq.read_table(output / "data.parquet").to_pylist()[0]
    changed_fields = {}
    for key in original_row:
        before_hash = hashlib.sha256(json.dumps(original_row[key], separators=(",", ":")).encode()).hexdigest()
        after_hash = hashlib.sha256(json.dumps(serialized_row[key], separators=(",", ":")).encode()).hexdigest()
        if before_hash != after_hash:
            changed_fields[key] = {"before_sha256": before_hash, "after_sha256": after_hash}
    serialization_error = {key: float(np.max(np.abs(np.asarray(serialized_row[key]) - np.asarray(row[key]))))
                           for key in changed_fields}
    prefix_preserved = all(
        serialized_row[key][:config.last_unchanged_frame + 1] == original_row[key][:config.last_unchanged_frame + 1]
        for key in changed_fields if isinstance(original_row[key], list) and len(original_row[key]) == len(original)
    )
    checks = {
        "original_inputs_unchanged": all(digest(Path(path)) == expected for path, expected in source_hashes.items()),
        "original_prefix_preserved": prefix_preserved,
        "serialization_float32_fidelity": max(serialization_error.values(), default=0) <= 2e-7,
        "object_static_after_210": bool(np.array_equal(repaired[210:], np.broadcast_to(static_pose, repaired[210:].shape))),
        "object_support": abs(exact_support["bottom_z_m"]) < .002 and exact_support["com_inside_contact_hull"],
        "object_transition_nonpenetration": bool(min(np.min(collision.vertices @ pose[2, :3] + pose[2, 3]) for pose in repaired[204:]) >= -1e-6),
        "rgb_projection_improved": bool(changes.mean() >= config.min_mean_iou_improvement and changes.min() >= -config.max_frame_iou_loss),
        "hand_projection_preserved": max(hand_projection.values()) < 1e-3,
        "motion_bounded": motion["max_speed_m_s"] <= config.max_transition_speed_m_s and motion["max_angular_speed_rad_s"] <= config.max_transition_angular_speed_rad_s and motion["max_acceleration_m_s2"] <= config.max_transition_acceleration_m_s2,
    }
    manifest = {
        "schema": "nvidia_v2d.support_reference_repair.v1", "config": asdict(config),
        "status": "accepted_support_candidate" if all(checks.values()) else "rejected_support_candidate",
        "training_allowed": all(checks.values()), "metric_ground_truth": False,
        "policy_rollout_used_for_fit": False, "input_sha256": source_hashes,
        "checks": checks, "changed_fields": changed_fields, "camera": camera_audit,
        "serialization_max_abs_error_by_field": serialization_error,
        "static_pose_fit": pose_fit, "static_support": static_support, "strict_coplanar_support": exact_support,
        "support_radius_sensitivity": radius_sensitivity,
        "collision_mesh_max_radial_m": float(radial.max()), "table_radius_m": support["fit"]["radius_m"],
        "object_correction_translation_max_m": float(np.linalg.norm(repaired[:, :3, 3] - original[:, :3, 3], axis=1).max()),
        "object_correction_rotation_max_rad": float(Rotation.from_matrix(repaired[:, :3, :3] @ original[:, :3, :3].transpose(0, 2, 1)).magnitude().max()),
        "motion": motion, "wrist_corrections": wrist_changes, "mano_rest_roots": {k: v.tolist() for k, v in roots.items()},
        "mano_rest_root_proof": root_proof, "hand_projection_max_delta_px": hand_projection,
        "release_iou_improvement_mean": float(changes.mean()), "release_iou_improvement_worst_frame": float(changes.min()),
        "projection": projection,
        "tip_distance_method": "Prefix unchanged original 4096-point sample; frames204+ exact unsigned triangle distance",
        "uncertainty": ["Initial support plane is a coordinate/shape anchor, not sensor ground truth",
                        "Metric scale and shape inherit the original mesh; no scale is optimized",
                        "Wood points assume a rigid planar tabletop; SIFT masks and object masks are estimates",
                        "Static contact is verified for the supplied collision hull and URDF COM at its origin",
                        "Finite support uses actual bottom-band points within the table; other geometry may overhang",
                        "Strict coplanar support survives 2 mm table-radius shrinkage but not 5 mm; the 1 mm contact band survives 5 mm but not 10 mm",
                        "Robot release clearance, contact remapping and closed-loop success require separate gates"],
    }
    files = [p for p in output.iterdir() if p.is_file() and p.name != "manifest.json"]
    files += [p for folder in (camera_output, overlay_output) for p in folder.iterdir() if p.is_file()]
    manifest["output_sha256"] = {str(p.relative_to(output)): digest(p) for p in files}
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parquet", "quality", "geometry", "reconstruction", "visual-mesh", "collision-mesh", "support-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = repair(**vars(args))
    print(json.dumps({"status": result["status"], "checks": result["checks"]}, indent=2))


if __name__ == "__main__":
    main()
