"""Offline contact repair against the actual convex collision union.

The union is immutable body-local geometry exported from the simulator. Source
labels are evidence, never an optimization variable. Interior distance bounds
distinguish exposed union faces from hidden interfaces between convex pieces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import linprog, minimize
from scipy.spatial import ConvexHull


@dataclass
class BoundaryPoint:
    point: np.ndarray
    outward: np.ndarray
    shape: int
    plane: int
    distance: float
    lower_bound: float
    exact: bool


class ConvexUnion:
    """Halfspace membership and certified exposed-face distance bounds."""

    def __init__(self, archive: Path | dict):
        data = dict(np.load(archive, allow_pickle=False)) if isinstance(archive, Path) else archive
        self.shape_ids = sorted(int(k.split("_")[1]) for k in data if k.endswith("_vertices"))
        if not self.shape_ids:
            raise ValueError("No collision pieces")
        self.vertices = [np.asarray(data[f"shape_{i}_vertices"], float) for i in self.shape_ids]
        self.planes = [ConvexHull(v).equations for v in self.vertices]
        self.bounds = np.asarray([[v.min(0), v.max(0)] for v in self.vertices])
        if any(not np.isfinite(v).all() for v in self.vertices):
            raise ValueError("Nonfinite collision vertices")

    def signed_planes(self, points):
        points = np.asarray(points, float).reshape(-1, 3)
        result = np.full((len(points), len(self.planes)), np.inf)
        # AABB rejection is exact for membership; plane values outside AABBs
        # are deliberately not used as metric distances.
        for j, eq in enumerate(self.planes):
            selected = ((points >= self.bounds[j, 0] - 1e-8) & (points <= self.bounds[j, 1] + 1e-8)).all(1)
            result[selected, j] = (points[selected] @ eq[:, :3].T + eq[:, 3]).max(1)
        return result

    def inside(self, points, tolerance=1e-8):
        return (self.signed_planes(points) < -tolerance).any(1)

    def transformed_pieces(self, matrices, translations):
        """Transform each convex piece independently without rebuilding hulls."""
        matrices, translations = np.asarray(matrices), np.asarray(translations)
        if matrices.shape != (len(self.planes), 3, 3) or translations.shape != (len(self.planes), 3):
            raise ValueError("One rigid transform per collision piece required")
        result = object.__new__(ConvexUnion)
        result.shape_ids = list(self.shape_ids)
        result.vertices = [v @ r.T + p for v, r, p in zip(self.vertices, matrices, translations, strict=True)]
        result.planes = []
        for eq, r, p in zip(self.planes, matrices, translations, strict=True):
            normal = eq[:, :3] @ r.T
            result.planes.append(np.c_[normal, eq[:, 3]-normal@p])
        result.bounds = np.asarray([[v.min(0), v.max(0)] for v in result.vertices])
        return result

    def ray_exit(self, point, direction):
        """First boundary exit from the union along a normalized direction."""
        point = np.asarray(point, float)
        direction = np.asarray(direction, float)
        direction = direction / np.linalg.norm(direction)
        intervals = []
        for eq in self.planes:
            a, b = eq[:, :3] @ direction, eq[:, :3] @ point + eq[:, 3]
            if np.any((np.abs(a) < 1e-12) & (b > 1e-10)):
                continue
            lo = np.max(-b[a < -1e-12] / a[a < -1e-12], initial=-np.inf)
            hi = np.min(-b[a > 1e-12] / a[a > 1e-12], initial=np.inf)
            if hi >= max(lo, 0.0) - 1e-10:
                intervals.append((max(lo, 0.0), hi))
        end = None
        for lo, hi in sorted(intervals):
            if end is None:
                if lo > 1e-8:
                    raise ValueError("Ray origin is outside the union")
                end = hi
            elif lo <= end + 1e-10:
                end = max(end, hi)
            else:
                break
        if end is None or not np.isfinite(end):
            raise ValueError("No finite union exit")
        return float(end)

    def exposed_nearest(self, point):
        """Return a certified closest union face for an interior point.

        For an interior point of a convex polyhedron, the closest plane foot
        attains its minimum surface distance. A foot exposed to the union is
        a valid upper bound. Equality to the deepest containing ball radius
        proves the exact union distance. Other interior configurations fail
        explicitly instead of returning a hidden face.
        """
        point = np.asarray(point, float)
        values = [eq[:, :3] @ point + eq[:, 3] for eq in self.planes]
        containing = [j for j, v in enumerate(values) if v.max() < -1e-8]
        if not containing:
            raise ValueError("Interior point required")
        lower = max(-values[j].max() for j in containing)
        candidates = []
        for j in containing:
            k = int(values[j].argmax())
            normal = self.planes[j][k, :3]
            distance = -values[j][k]
            foot = point + distance * normal
            if not self.inside(foot[None])[0]:
                candidates.append(BoundaryPoint(foot, normal, self.shape_ids[j], k,
                                                float(distance), float(lower),
                                                bool(abs(distance - lower) < 1e-8)))
        if not candidates:
            raise ValueError("No exposed nearest containing-piece face; exact union projection unresolved")
        return min(candidates, key=lambda x: x.distance)

    def interior_bounds(self, points):
        """Per-point rigorous lower and constructive upper interior distances.

        Upper bounds are exposed plane feet where available, or first union
        exits along containing-face outward rays. Thus an upper bound <=4 mm
        certifies the sampled point is no deeper than 4 mm, even at overlaps.
        """
        points = np.asarray(points, float).reshape(-1, 3)
        plane_values = self.signed_planes(points)
        inside = plane_values < -1e-8
        lower = np.max(np.where(inside, -plane_values, 0.0), axis=1)
        upper = np.where(inside.any(1), np.inf, 0.0)
        outward = np.zeros_like(points)
        boundary = points.copy()
        for j, eq in enumerate(self.planes):
            indices = np.flatnonzero(inside[:, j])
            if not len(indices):
                continue
            value = points[indices] @ eq[:, :3].T + eq[:, 3]
            nearest_plane = value.argmax(1)
            normal = eq[nearest_plane, :3]
            distance = -value[np.arange(len(indices)), nearest_plane]
            foot = points[indices] + distance[:, None] * normal
            exposed = ~self.inside(foot)
            improve = exposed & (distance < upper[indices])
            chosen = indices[improve]
            upper[chosen] = distance[improve]
            outward[chosen] = normal[improve]
            boundary[chosen] = foot[improve]
        for i in np.flatnonzero(np.isinf(upper)):
            candidates = []
            for j in np.flatnonzero(inside[i]):
                eq = self.planes[j]
                value = eq[:, :3] @ points[i] + eq[:, 3]
                normal = eq[int(value.argmax()), :3]
                distance = -value.max()
                foot = points[i] + distance * normal
                if self.inside(foot[None])[0]:
                    distance = self.ray_exit(points[i], normal)
                    foot = points[i] + distance * normal
                candidates.append((float(distance), normal, foot))
            upper[i], outward[i], boundary[i] = min(candidates, key=lambda x: x[0])
        return lower, upper, outward, boundary


def repair_inside_targets(roles, union: ConvexUnion, outside=0.001):
    """Rebind inside trusted targets; retain independent original role fields."""
    if outside != 0.001:
        raise ValueError("The accepted outward offset is fixed at 1 mm")
    result = {key: value.copy() for key, value in roles.items()}
    for key in ["object_point_local", "object_normal_local", "active", "source_status",
                "source_contact_active", "source_slot", "confidence", "feasible"]:
        source_key = "prior_" + key
        if source_key in result:
            raise ValueError("Already repaired role map")
        result[source_key] = roles[key].copy()
    result["collision_surface_rebound"] = np.zeros_like(roles["active"])
    rows = []
    for frame, side in np.argwhere(roles["active"]):
        p, n = roles["object_point_local"][frame, side], roles["object_normal_local"][frame, side]
        target = p - outside * n
        if not union.inside(target[None])[0]:
            continue
        boundary = union.exposed_nearest(target)
        if not boundary.exact:
            raise ValueError("Target projection has not been certified exact")
        result["object_point_local"][frame, side] = boundary.point
        result["object_normal_local"][frame, side] = -boundary.outward
        result["collision_surface_rebound"][frame, side] = True
        new_target = boundary.point + outside * boundary.outward
        if union.inside(new_target[None])[0]:
            raise ValueError("Offset target lies inside another collision piece")
        rows.append({"frame": int(frame), "side": int(side), "shape": boundary.shape,
                     "plane": boundary.plane, "original_inside_depth_m": boundary.distance,
                     "surface_shift_m": float(np.linalg.norm(boundary.point - p)),
                     "target_shift_m": float(np.linalg.norm(new_target - target)),
                     "normal_dot_prior": float((-boundary.outward) @ n)})
    return result, rows


def minimum_smooth_translation(frame_count, constraints, *, release_frame=204):
    """Minimum norm/velocity/acceleration translation under linear clearances.

    Each constraint is (source frame, world outward normal, required offset).
    Only translations at frames 1..release_frame-1 are variables. Frame zero
    and the entire release suffix are exactly fixed. The nonnegative dual is
    solved without adding a solver dependency or relaxing any clearance.
    """
    if not 1 < release_frame < frame_count:
        raise ValueError("Invalid fixed release boundary")
    grouped = {}
    for frame, normal, minimum in constraints:
        key = (frame, *np.round(normal, 12))
        if key not in grouped or minimum > grouped[key][2]:
            grouped[key] = (frame, normal, minimum)
    constraints = list(grouped.values())
    for frame in sorted({c[0] for c in constraints}):
        local = [c for c in constraints if c[0] == frame]
        feasibility = linprog(np.zeros(3), A_ub=-np.asarray([c[1] for c in local]),
                              b_ub=-np.asarray([c[2] for c in local]),
                              bounds=[(-.012, .012)] * 3, method="highs")
        if not feasibility.success:
            raise ValueError(f"Nearest-exit linear branch infeasible within 12 mm axes at frame {frame}; not a proof of articulated-hand infeasibility")
    free = np.arange(1, release_frame)
    embed = np.eye(frame_count)[:, free]
    d1 = np.diff(embed, axis=0)
    d2 = np.diff(embed, n=2, axis=0)
    hessian = np.eye(len(free)) + 20 * d1.T @ d1 + 100 * d2.T @ d2
    factor = cho_factor(hessian)
    a = np.zeros((len(constraints), len(free), 3))
    b = np.empty(len(constraints))
    for i, (frame, normal, minimum) in enumerate(constraints):
        if frame not in free:
            raise ValueError("Correction would alter the fixed initial/release trajectory")
        a[i, frame - 1] = normal
        b[i] = minimum
    if not len(constraints):
        return np.zeros((frame_count, 3)), {"constraints": 0, "violation_m": 0.0}
    inverse_a = np.stack([cho_solve(factor, a[:, :, axis].T) for axis in range(3)], axis=2)
    gram = np.einsum("itk,tjk->ij", a, inverse_a)

    def objective(lam):
        g = gram @ lam - b
        return float(0.5 * lam @ (gram @ lam) - b @ lam), g

    result = minimize(objective, np.zeros(len(b)), method="L-BFGS-B", jac=True,
                      bounds=[(0, None)] * len(b),
                      options={"ftol": 1e-16, "gtol": 1e-12, "maxiter": 30000, "maxls": 100})
    delta = np.zeros((frame_count, 3))
    delta[free] = np.einsum("tik,i->tk", inverse_a, result.x)
    violation = float(np.max(b - np.einsum("itk,tk->i", a, delta[free]), initial=0))
    if violation > 2e-7:
        raise ValueError(f"Translation constraints not solved: {violation} m; {result.message}")
    return delta, {"constraints": len(b), "violation_m": violation,
                   "iterations": int(result.nit), "solver_message": str(result.message),
                   "objective": float(result.fun)}


def refit_surface_role(vertices, faces, normals, matrices, translations, target,
                       target_normal, union, original_point, *, max_error=.006,
                       body_union=None, hand_unions=None):
    """Fit one actual body surface point over its source and NN control states.

    The role's body, target, source evidence and active mask are not variables.
    Every returned point lies on an actual triangle and is verified outside the
    object union at every supplied state. Failure returns None, never a mask.
    """
    import trimesh

    if max_error != .006:
        raise ValueError("Role tolerance must remain 6 mm")
    vertices, faces, normals = np.asarray(vertices), np.asarray(faces), np.asarray(normals)
    matrices, translations = np.asarray(matrices), np.asarray(translations)
    target, target_normal = np.asarray(target), np.asarray(target_normal)
    original_point = np.asarray(original_point)
    centers = np.einsum("nji,nj->ni", matrices, target-translations)
    # No point anywhere in R3 can be within 6 mm of two centres >12 mm apart.
    if np.linalg.norm(centers[:, None]-centers[None], axis=-1).max() > .012 + 1e-10:
        return None
    dots = np.einsum("sij,fj,i->sf", matrices, normals, target_normal)
    eligible = np.flatnonzero(dots.min(0) >= .5)
    if not len(eligible):
        return None
    triangles = vertices[faces[eligible]]
    nearest = trimesh.triangles.closest_point(triangles, np.broadcast_to(centers.mean(0), (len(triangles), 3)))
    lower = np.linalg.norm(nearest-centers.mean(0), axis=1)
    eligible_order = np.argsort(lower)
    accepted = []
    for idx in eligible_order:
        if lower[idx] > max_error + 1e-9:
            continue
        face = int(eligible[idx]); triangle = vertices[faces[face]]
        origin = triangle[0]; basis = (triangle[1:]-origin).T
        initial = np.linalg.lstsq(basis, nearest[idx]-origin, rcond=None)[0]
        exits, body_exits = [], []

        def point(x):
            return origin + basis @ x

        def objective(x):
            delta = point(x)-original_point
            return float(delta@delta / .006**2), 2*basis.T@delta / .006**2

        def constraints(x):
            world = np.einsum("sij,j->si", matrices, point(x))+translations
            result = [1-x.sum(), *((.00598**2-np.sum((world-target)**2, axis=1))/.006**2)]
            result.extend(float((world[state]-boundary)@normal-.00002)/.006 for state, normal, boundary in exits)
            result.extend(float((point(x)-boundary)@normal-.00002)/.006 for normal, boundary in body_exits)
            return np.asarray(result)

        for _ in range(8):
            solved = minimize(objective, initial, jac=True, method="SLSQP", bounds=[(0, 1)]*2,
                              constraints=[{"type": "ineq", "fun": constraints}],
                              options={"maxiter": 120, "ftol": 1e-12})
            if not solved.success or constraints(solved.x).min() < -1e-7:
                break
            p = point(solved.x)
            locations = np.einsum("sij,j->si", matrices, p)+translations
            inside = union.inside(locations)
            robot_unions = ([] if body_union is None else [body_union]) + ([] if hand_unions is None else list(hand_unions))
            body_inside = any(u.inside(p[None])[0] for u in robot_unions)
            outward_hidden = any(u.inside((p+.00001*normals[face])[None])[0] for u in robot_unions)
            if not inside.any() and not body_inside and not outward_hidden:
                errors = np.linalg.norm(locations-target, axis=1)
                if errors.max() <= max_error and dots[:, face].min() >= .5:
                    accepted.append({"point": p, "normal": normals[face].copy(), "face": face,
                                     "max_error_m": float(errors.max()), "min_normal_dot": float(dots[:, face].min()),
                                     "local_point_change_m": float(np.linalg.norm(p-original_point))})
                break
            if body_inside or outward_hidden:
                for robot_union in robot_unions:
                    origin_check = p if robot_union.inside(p[None])[0] else p+.00001*normals[face]
                    if robot_union.inside(origin_check[None])[0]:
                        _, _, ns, bs = robot_union.interior_bounds(origin_check[None])
                        body_exits.append((ns[0], bs[0]))
            for state in np.flatnonzero(inside):
                _, _, ns, bs = union.interior_bounds(locations[state:state+1])
                exits.append((int(state), ns[0], bs[0]))
            initial = solved.x
    return min(accepted, key=lambda x: x["local_point_change_m"]) if accepted else None


def exposed_body_seed(vertices, faces, normals, body_union, point, normal,
                      target_normals_body=None):
    """Seed an exposed same-body point without claiming pose feasibility.

    Hidden original points cannot be made exposed by moving their rigid body.
    This geometric seed therefore precedes articulated IK. Actual object
    distance, source/control exposure and all pose gates must be checked later.
    """
    import trimesh

    point, normal = np.asarray(point), np.asarray(normal)
    vertices, faces, normals = np.asarray(vertices), np.asarray(faces), np.asarray(normals)
    if not body_union.inside(point[None])[0] and not body_union.inside((point+.00001*normal)[None])[0]:
        return None
    origin = point if body_union.inside(point[None])[0] else point+.00001*normal
    candidates = []
    try:
        candidates.append(body_union.exposed_nearest(origin).point)
    except ValueError:
        pass
    directions = [normal]
    if target_normals_body is not None:
        directions.extend(np.asarray(target_normals_body))
    directions.extend(np.unique(np.round(normals[normals @ normal > .25], 6), axis=0))
    for direction in directions:
        direction = direction/np.linalg.norm(direction)
        candidates.append(origin+body_union.ray_exit(origin, direction)*direction)
    accepted = []
    triangles = vertices[faces]
    for proposed in candidates:
        nearest = trimesh.triangles.closest_point(triangles, np.broadcast_to(proposed, (len(faces), 3)))
        distance = np.linalg.norm(nearest-proposed, axis=1)
        for face in np.flatnonzero(distance < 2e-7):
            p, n = nearest[face], normals[face]
            if body_union.inside(p[None])[0] or body_union.inside((p+.00001*n)[None])[0]:
                continue
            dot = float((np.asarray(target_normals_body) @ n).min()) if target_normals_body is not None else float(n@normal)
            accepted.append({"point": p, "normal": n.copy(), "face": int(face),
                             "local_point_change_m": float(np.linalg.norm(p-point)),
                             "current_normal_dot_min": dot, "pose_acceptance_evaluated": False})
    if not accepted:
        raise ValueError("No exposed body surface seed found")
    # Prefer a currently compatible normal; a fallback remains an explicit IK
    # seed requiring orientation repair, never a successful role candidate.
    return min(accepted, key=lambda x: (x["current_normal_dot_min"] < .5,
                                       x["local_point_change_m"]))
