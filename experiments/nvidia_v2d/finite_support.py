"""Derive a finite tabletop candidate from an RGB contour and an explicit scene plane.

The plane height remains an assumption: a silhouette constrains footprint, not
metric depth. This diagnostic never marks geometry accepted or edits source data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def intersect_plane(pixels: np.ndarray, intrinsics: np.ndarray,
                    camera_to_scene: np.ndarray, height: float) -> np.ndarray:
    """Intersect positive camera rays with the horizontal scene plane."""
    pixels = np.asarray(pixels, dtype=float)
    k = np.asarray(intrinsics, dtype=float)
    transform = np.asarray(camera_to_scene, dtype=float)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or k.shape != (4,):
        raise ValueError("Expected Nx2 pixels and fx, fy, cx, cy")
    if transform.shape != (4, 4) or not all(np.isfinite(x).all() for x in (pixels, k, transform)):
        raise ValueError("Geometry must be finite with a 4x4 camera transform")
    if min(k[:2]) <= 0 or not np.isfinite(height):
        raise ValueError("Positive focal lengths and finite plane height required")
    if (not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-5)
            or not np.allclose(transform[3], [0, 0, 0, 1])):
        raise ValueError("Camera rotation is not rigid")
    rays = np.column_stack(((pixels[:, 0] - k[2]) / k[0],
                            (pixels[:, 1] - k[3]) / k[1], np.ones(len(pixels))))
    directions = rays @ transform[:3, :3].T
    if np.any(abs(directions[:, 2]) < 1e-8):
        raise ValueError("Table rays are parallel to the assumed plane")
    distance = (height - transform[2, 3]) / directions[:, 2]
    if np.any(distance <= 0):
        raise ValueError("Assumed support lies behind a camera ray")
    return transform[:3, 3] + distance[:, None] * directions


def fit_circle(points: np.ndarray) -> dict:
    """Fit a circle to a complete, image-observed tabletop outer boundary."""
    xy = np.asarray(points, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 8 or not np.isfinite(xy).all():
        raise ValueError("At least eight finite 2D boundary points are required")
    design = np.column_stack((2 * xy, np.ones(len(xy))))
    if np.linalg.matrix_rank(design) != 3:
        raise ValueError("Degenerate tabletop boundary")
    solution = np.linalg.lstsq(design, np.sum(xy**2, axis=1), rcond=None)[0]
    radius = float(np.sqrt(solution[2] + np.sum(solution[:2]**2)))
    residual = abs(np.linalg.norm(xy - solution[:2], axis=1) - radius)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Invalid tabletop circle")
    return {"center_xy_m": solution[:2].tolist(), "radius_m": radius,
            "radial_residual_median_m": float(np.median(residual)),
            "radial_residual_p95_m": float(np.quantile(residual, .95)),
            "boundary_points": len(xy)}


def support_usda(center: list[float], radius: float, top: float, thickness: float) -> str:
    """Create an explicit finite cylinder accepted by FlashCHORD's support loader."""
    if len(center) != 2 or not np.isfinite([*center, radius, top, thickness]).all():
        raise ValueError("Support dimensions must be finite")
    if radius <= 0 or thickness <= 0:
        raise ValueError("Support radius and thickness must be positive")
    return ("#usda 1.0\n(\n    metersPerUnit = 1\n    upAxis = \"Z\"\n)\n"
            "def Cylinder \"tabletop\" {\n"
            f"    double radius = {radius:.10f}\n    double height = {thickness:.10f}\n"
            "    uniform token axis = \"Z\"\n"
            f"    double3 xformOp:translate = ({center[0]:.10f}, {center[1]:.10f}, {top - thickness / 2:.10f})\n"
            "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n")


def derive_support(*, rgb: Path, quality: Path, geometry: Path, output: Path,
                   roi: tuple[int, int, int, int], top: float = 0.0,
                   thickness: float = .025, depth: Path | None = None) -> dict:
    """Write a candidate, contour proof and assumptions; source files stay read-only."""
    import cv2

    if output.exists():
        raise FileExistsError(output)
    files = [rgb, quality, geometry] + ([depth] if depth else [])
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    image = cv2.imread(str(rgb))
    if image is None:
        raise ValueError("RGB image cannot be read")
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
        raise ValueError("ROI must lie inside the RGB image")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([4, 65, 40]), np.array([28, 255, 255]))
    allowed = np.zeros(mask.shape, np.uint8)
    allowed[y0:y1, x0:x1] = 255
    mask &= allowed
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No tabletop color component in ROI")
    largest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest).reshape(-1, 2)
    if np.any((hull[:, 0] <= x0) | (hull[:, 0] >= x1-1) |
              (hull[:, 1] <= y0) | (hull[:, 1] >= y1-1)):
        raise ValueError("Tabletop boundary is clipped by the ROI")
    with np.load(quality) as q, np.load(geometry) as g:
        transform = g["source_to_scene_transform"] @ q["camera_to_world_transform"][0]
        intrinsics = q["camera_intrinsics"].copy()
        points = intersect_plane(hull, intrinsics, transform, top)
        fit = fit_circle(points[:, :2])
        clearance = {}
        for side in ("left", "right"):
            vertices = g[side + "_vertices"]
            radial = np.linalg.norm(vertices[:, :, :2] - fit["center_xy_m"], axis=2)
            inside = (radial < fit["radius_m"]) & (vertices[:, :, 2] < top)
            inside &= vertices[:, :, 2] > top - thickness
            clearance[side] = {
                "initial_vertex_radial_clearance_m": float(radial[0].min() - fit["radius_m"]),
                "initial_wrist_z_m": float(g[side + "_joints"][0, 0, 2]),
                "mano_vertex_inside_table_frames": np.flatnonzero(inside.any(axis=1)).tolist(),
                "maximum_vertices_inside_table": int(inside.sum(axis=1).max()),
                "collision_truth": False,
            }
    sensitivity = []
    for offset in (-.02, 0.0, .02):
        item = fit_circle(intersect_plane(hull, intrinsics, transform, top + offset)[:, :2])
        sensitivity.append({"assumed_top_m": top + offset, **item})
    report = {"schema": "nvidia_v2d.finite_support_candidate.v1", "status": "candidate",
              "metric_ground_truth": False, "geometry_quality_accepted": False,
              "source_sha256": hashes, "source_frame": 0, "roi_xyxy": list(roi),
              "color_hsv_lower": [4, 65, 40], "color_hsv_upper": [28, 255, 255],
              "fit": fit, "top_m": top, "thickness_m": thickness,
              "assumptions": ["Table is a circular finite surface from RGB outer contour",
                              "Top height is the existing PCA object-bottom plane, not measured depth",
                              "Thickness is an explicit approximation, not a measurement",
                              "No scene trajectories or mesh scale are changed"],
              "height_sensitivity": sensitivity, "mano_clearance": clearance,
              "required_collision_config": {"scene.support": True, "collision.ground": False,
                                            "collision.robot_support_collision": True}}
    output.mkdir(parents=True)
    if depth:
        enc = cv2.imread(str(depth), cv2.IMREAD_UNCHANGED)
        if enc is None or enc.dtype != np.uint16 or enc.shape != mask.shape:
            raise ValueError("Depth must be full RGB-sized uint16 inverse depth")
        component = np.zeros_like(mask)
        cv2.drawContours(component, [largest], -1, 255, cv2.FILLED)
        component &= mask
        component = cv2.erode(component, np.ones((9, 9), np.uint8))
        ys, xs = np.nonzero((component > 0) & (enc > 0) & (enc < 65535))
        ys, xs = ys[::15], xs[::15]
        z = 65535.0 / enc[ys, xs] - 1
        pc = np.column_stack(((xs - intrinsics[2]) / intrinsics[0] * z,
                              (ys - intrinsics[3]) / intrinsics[1] * z, z))
        world = pc @ transform[:3, :3].T + transform[:3, 3]
        design = np.column_stack((world[:, :2], np.ones(len(world))))
        fit_z = np.linalg.lstsq(design, world[:, 2], rcond=None)[0]
        for _ in range(6):
            residual = world[:, 2] - design @ fit_z
            keep = abs(residual - np.median(residual)) < .015
            fit_z = np.linalg.lstsq(design[keep], world[keep, 2], rcond=None)[0]
        residual = abs(world[:, 2] - design @ fit_z)
        report["predicted_depth_crosscheck"] = {
            "source": "MoGe prediction, not sensor truth", "plane_z_equals_ax_by_c": fit_z.tolist(),
            "points": len(world), "inliers_within_15mm": int(keep.sum()),
            "median_residual_m": float(np.median(residual)),
            "p95_residual_m": float(np.quantile(residual, .95)),
            "tilt_from_scene_z_deg": float(np.degrees(np.arctan(np.linalg.norm(fit_z[:2])))),
            "height_disagrees_with_candidate_m": float(fit_z[2] - top),
            "used_to_move_scene": False}
    proof = image.copy()
    cv2.polylines(proof, [hull], True, (20, 220, 30), 5)
    cv2.imwrite(str(output / "table-contour-proof.jpg"), proof)
    np.savez_compressed(output / "table-boundary.npz", pixels=hull, scene_points=points,
                        intrinsics=intrinsics, camera_to_scene=transform)
    (output / "support.usda").write_text(support_usda(fit["center_xy_m"], fit["radius_m"], top, thickness))
    report["source_unchanged"] = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
                                     for p, h in hashes.items())
    (output / "manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("rgb", "quality", "geometry", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--depth", type=Path)
    parser.add_argument("--roi", type=int, nargs=4, required=True, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--top", type=float, default=0.0)
    parser.add_argument("--thickness", type=float, default=.025)
    args = vars(parser.parse_args())
    args["roi"] = tuple(args["roi"])
    print(json.dumps(derive_support(**args), indent=2))


if __name__ == "__main__":
    main()
