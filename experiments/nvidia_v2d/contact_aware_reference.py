"""CPU contact-role constrained Sharpa reference candidates; never change object poses.

The model is a deliberately small MJCF hinge-tree reader, checked against the
input's official FK frames before optimization. Evidence names semantic contact
roles; no MANO-slot to simulator-slot integer correspondence is assumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _numbers(element, name, default):
    return np.fromstring(element.get(name, default), sep=" ")


def _transform(element):
    if set(element.attrib) & {"euler", "axisangle", "xyaxes", "zaxis"}:
        raise ValueError("Explicit pos/quat transforms are required")
    p = _numbers(element, "pos", "0 0 0")
    q = _numbers(element, "quat", "1 0 0 0")
    return p, Rotation.from_quat(q, scalar_first=True).as_matrix()


def axis_rotation(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


@dataclass
class Body:
    name: str
    parent: int
    position: np.ndarray
    rotation: np.ndarray
    joint_index: int | None
    joint_name: str | None
    axis: np.ndarray
    pivot: np.ndarray
    sites: list


class SharpaCPUModel:
    """Exact rigid FK for explicit one-hinge-per-body MJCF trees; no simulator."""

    def __init__(self, xml_path: Path, frame_names: list[str], joint_names: list[str]):
        self.xml_path = Path(xml_path)
        self.xml = ET.parse(xml_path).getroot()
        self.frame_names = list(frame_names)
        self.joint_names = list(joint_names)
        if len(set(joint_names)) != len(joint_names):
            raise ValueError("Duplicate joint names")
        compiler = self.xml.find("compiler")
        if compiler is not None and compiler.get("angle", "degree") != "radian":
            raise ValueError("Radian MJCF required")
        self.mesh_dir = self.xml_path.parent / (
            compiler.get("meshdir", "") if compiler is not None else ""
        )
        self.mesh_assets = {x.get("name"): x for x in self.xml.findall("asset/mesh")}
        self.nodes = []
        self.elements = []
        self.limits = np.full((len(joint_names), 2), np.nan)

        def visit(element, parent):
            position, rotation = _transform(element)
            joints = element.findall("joint")
            if len(joints) > 1 or element.find("freejoint") is not None:
                raise ValueError("Only one explicit hinge per body is supported")
            joint = joints[0] if joints else None
            index, name, axis, pivot = (
                None,
                None,
                np.array([0.0, 0.0, 1.0]),
                np.zeros(3),
            )
            if joint is not None:
                name = joint.get("name")
                if joint.get("type", "hinge") != "hinge" or name not in joint_names:
                    raise ValueError(f"Unsupported joint {name}")
                index = joint_names.index(name)
                self.limits[index] = _numbers(joint, "range", "nan nan")
                axis = _numbers(joint, "axis", "0 0 1")
                pivot = _numbers(joint, "pos", "0 0 0")
            sites = [(s.get("name"), *_transform(s)) for s in element.findall("site")]
            here = len(self.nodes)
            self.nodes.append(
                Body(
                    element.get("name"),
                    parent,
                    position,
                    rotation,
                    index,
                    name,
                    axis,
                    pivot,
                    sites,
                )
            )
            self.elements.append(element)
            for child in element.findall("body"):
                visit(child, here)

        roots = self.xml.findall("worldbody/body")
        if len(roots) != 1:
            raise ValueError("Exactly one hand root required")
        visit(roots[0], -1)
        if not np.isfinite(self.limits).all():
            raise ValueError("Every input joint requires explicit finite limits")

    def transforms(self, q, wrist_position, wrist_wxyz):
        q = np.asarray(q, float)
        if q.shape != (len(self.joint_names),) or not np.isfinite(q).all():
            raise ValueError("Invalid finger positions")
        root_p = np.asarray(wrist_position, float)
        root_r = Rotation.from_quat(wrist_wxyz, scalar_first=True).as_matrix()
        if not np.isfinite(root_p).all():
            raise ValueError("Invalid wrist position")
        positions, rotations = [], []
        named = {"universe": (np.zeros(3), np.eye(3)), "root_joint": (root_p, root_r)}
        for body in self.nodes:
            pp, rr = (
                (root_p, root_r)
                if body.parent < 0
                else (positions[body.parent], rotations[body.parent])
            )
            position = pp + rr @ body.position
            rotation = rr @ body.rotation
            if body.joint_index is not None:
                turn = axis_rotation(body.axis, q[body.joint_index])
                position = position + rotation @ (body.pivot - turn @ body.pivot)
                rotation = rotation @ turn
                named[body.joint_name] = (position, rotation)
            positions.append(position)
            rotations.append(rotation)
            named[body.name] = (position, rotation)
            for name, local_p, local_r in body.sites:
                named[name] = (position + rotation @ local_p, rotation @ local_r)
        return np.asarray(positions), np.asarray(rotations), named

    def frames(self, q, wrist_position, wrist_wxyz):
        _, _, named = self.transforms(q, wrist_position, wrist_wxyz)
        if any(name not in named for name in self.frame_names):
            raise ValueError("Input FK frame names are not all present in MJCF")
        return np.array(
            [
                np.r_[
                    named[name][0],
                    Rotation.from_matrix(named[name][1]).as_quat(scalar_first=True),
                ]
                for name in self.frame_names
            ]
        )

    def infer_fingers_from_bodyposes(self, bodyposes, labels):
        """Read hinge angles from actual parent/child rotations, independent of targets."""
        lookup = {name.split("/")[-1]: i for i, name in enumerate(labels)}
        q = np.zeros(len(self.joint_names))
        for body in self.nodes:
            if body.joint_index is None:
                continue
            child = Rotation.from_quat(bodyposes[lookup[body.name], 3:]).as_matrix()
            parent = Rotation.from_quat(
                bodyposes[lookup[self.nodes[body.parent].name], 3:]
            ).as_matrix()
            relative = body.rotation.T @ parent.T @ child
            q[body.joint_index] = Rotation.from_matrix(relative).as_rotvec() @ (
                body.axis / np.linalg.norm(body.axis)
            )
        return q


class CollisionSamples:
    """Deterministic samples and outward normals of per-geom convex collision meshes."""

    def __init__(self, model: SharpaCPUModel, samples_per_geometry=64):
        import trimesh

        points, normals, bodies, names = [], [], [], []
        self.asset_hashes = {str(model.xml_path.resolve()): sha256(model.xml_path)}
        rng = np.random.default_rng(20261029)
        self.geometry = []
        for body_index, (body, element) in enumerate(
            zip(model.nodes, model.elements, strict=True)
        ):
            for geom in element.findall("geom"):
                if (
                    int(geom.get("contype", "1"))
                    == int(geom.get("conaffinity", "1"))
                    == 0
                ):
                    continue
                if geom.get("type") != "mesh":
                    raise ValueError("Only explicit collision mesh geoms supported")
                asset = model.mesh_assets[geom.get("mesh")]
                if set(asset.attrib) & {"refpos", "refquat"}:
                    raise ValueError("Mesh reference transforms unsupported")
                path = (model.mesh_dir / asset.get("file")).resolve()
                mesh = trimesh.load_mesh(path, process=False)
                mesh.vertices *= _numbers(asset, "scale", "1 1 1")
                gp, gr = _transform(geom)
                mesh.vertices = mesh.vertices @ gr.T + gp
                hull = mesh.convex_hull
                selected = rng.choice(
                    len(hull.faces), samples_per_geometry, p=hull.area_faces / hull.area
                )
                uv = rng.random((samples_per_geometry, 2))
                uv[uv.sum(1) > 1] = 1 - uv[uv.sum(1) > 1]
                tri = hull.triangles[selected]
                sampled = (
                    tri[:, 0]
                    + uv[:, :1] * (tri[:, 1] - tri[:, 0])
                    + uv[:, 1:] * (tri[:, 2] - tri[:, 0])
                )
                points.append(sampled)
                normals.append(hull.face_normals[selected])
                bodies.extend([body_index] * samples_per_geometry)
                names.extend([body.name] * samples_per_geometry)
                self.asset_hashes[str(path)] = sha256(path)
                self.geometry.append((body_index, body.name, hull))
        self.points = np.concatenate(points)
        self.normals = np.concatenate(normals)
        self.body_indices = np.array(bodies)
        self.body_names = np.array(names)

    def world(self, model, q, wrist_position, wrist_wxyz):
        positions, rotations, _ = model.transforms(q, wrist_position, wrist_wxyz)
        r, p = rotations[self.body_indices], positions[self.body_indices]
        return np.einsum("nij,nj->ni", r, self.points) + p, np.einsum(
            "nij,nj->ni", r, self.normals
        )


class ObjectSurface:
    """Fast oriented surface proxy for optimization; exact triangles for acceptance."""

    def __init__(self, path: Path, count=24000):
        import trimesh
        from scipy.spatial import cKDTree

        self.mesh = trimesh.load_mesh(path)
        if (
            not self.mesh.is_watertight
            or not self.mesh.is_winding_consistent
            or self.mesh.volume <= 0
        ):
            raise ValueError(
                "A watertight consistently outward collision mesh is required"
            )
        rng = np.random.default_rng(20261029)
        face = rng.choice(
            len(self.mesh.faces), count, p=self.mesh.area_faces / self.mesh.area
        )
        uv = rng.random((count, 2))
        uv[uv.sum(1) > 1] = 1 - uv[uv.sum(1) > 1]
        tri = self.mesh.triangles[face]
        self.points = (
            tri[:, 0]
            + uv[:, :1] * (tri[:, 1] - tri[:, 0])
            + uv[:, 1:] * (tri[:, 2] - tri[:, 0])
        )
        self.normals = self.mesh.face_normals[face]
        self.tree = cKDTree(self.points)

    def approximate_signed(self, local_points):
        _, index = self.tree.query(local_points, workers=1)
        return np.sum(
            (local_points - self.points[index]) * self.normals[index], axis=-1
        )

    def exact(self, local_points):
        import trimesh

        closest, distance, face = trimesh.proximity.closest_point(
            self.mesh, local_points
        )
        signed = np.sum(
            (local_points - closest) * self.mesh.face_normals[face], axis=-1
        )
        return distance, signed


def trusted_thumb_evidence(records, frame_count, source_row):
    """One trusted semantic thumb target per hand/frame, never a four-finger rule."""
    result = {}
    for record in records:
        side, frame = record["side"], int(record["frame"])
        if side not in ("left", "right") or not 0 <= frame < frame_count:
            raise ValueError("Evidence side/frame out of range")
        slot = int(record["slot"])
        if not 0 <= slot < len(source_row["mano_link_names"]):
            raise ValueError("Evidence slot out of range")
        if source_row["mano_link_names"][slot] != record["link_name"]:
            raise ValueError("Contact evidence link does not match source slot name")
        if not record.get("trusted", False) or not record["link_name"].startswith(
            "link_thumb"
        ):
            continue
        if int(record["part_id"]) <= 0:
            raise ValueError("Trusted contact requires an active object part")
        if int(source_row[f"mano_{side}_object_contact_part_ids"][frame][slot]) != int(
            record["part_id"]
        ):
            raise ValueError("Contact evidence object part differs from source")
        normal = np.asarray(record["object_local_normal"], float)
        point = np.asarray(record["object_local_position"], float)
        confidence = float(record["confidence"])
        if (
            point.shape != (3,)
            or normal.shape != (3,)
            or not np.isfinite(np.r_[point, normal, confidence]).all()
        ):
            raise ValueError("Nonfinite or malformed contact evidence")
        if (
            not np.isclose(np.linalg.norm(normal), 1, atol=1e-4)
            or not 0 < confidence <= 1
        ):
            raise ValueError(
                "Trusted evidence requires a unit normal and confidence in (0,1]"
            )
        key = (frame, side)
        if key not in result or confidence > result[key]["confidence"]:
            result[key] = record
    return result


def apply_delta(q, wp, wq, delta, joint_indices):
    candidate_q = np.array(q, float, copy=True)
    candidate_q[joint_indices] += delta[6:]
    candidate_p = np.asarray(wp) + delta[:3]
    candidate_r = Rotation.from_rotvec(delta[3:6]) * Rotation.from_quat(
        wq, scalar_first=True
    )
    return candidate_q, candidate_p, candidate_r.as_quat(scalar_first=True)


def solve_contact_frame(
    model,
    surface,
    obj,
    q,
    wp,
    wq,
    object_pose,
    evidence,
    *,
    allowed_suffixes=("thumb_PP", "thumb_DP"),
    previous_delta=None,
    max_evaluations=70,
    support_height=None,
    patch_rank=0,
):
    """Constrain a named robot surface patch to a source object-local target.

    Only wrist and five thumb DOFs vary. The four other fingers retain their
    original joint values. Candidate patches are chosen from reference geometry,
    never from a successful policy rollout.
    """
    import time

    from scipy.optimize import least_squares

    started = time.perf_counter()
    side = model.joint_names[0].split("_")[0]
    joints = np.array(
        [i for i, name in enumerate(model.joint_names) if "_thumb_" in name]
    )
    if len(joints) != 5:
        raise ValueError("Expected exactly five thumb joints")
    q, wp, wq, object_pose = map(np.asarray, (q, wp, wq, object_pose))
    object_r = Rotation.from_quat(object_pose[3:], scalar_first=True).as_matrix()
    cp_local = np.asarray(evidence["object_local_position"])
    normal_local = np.asarray(evidence["object_local_normal"])
    cp = object_r @ cp_local + object_pose[:3]
    normal = object_r @ normal_local
    target = cp - 0.001 * normal
    ref_points, ref_normals = surface.world(model, q, wp, wq)
    prior = np.zeros(11) if previous_delta is None else np.asarray(previous_delta)
    limits = model.limits[joints]
    lower = np.r_[
        np.full(3, -0.03), np.full(3, -0.35), np.maximum(-0.8, limits[:, 0] - q[joints])
    ]
    upper = np.r_[
        np.full(3, 0.03), np.full(3, 0.35), np.minimum(0.8, limits[:, 1] - q[joints])
    ]
    lower = np.minimum(lower, -1e-9)
    upper = np.maximum(upper, 1e-9)
    prior = np.clip(prior, lower + 1e-10, upper - 1e-10)
    # Fixed subset for optimization; all recorded samples checked after solving.
    safety_indices = np.arange(0, len(surface.points), 4)
    scale = np.r_[np.full(3, 0.02), np.full(3, 0.3), np.full(5, 0.5)]
    candidates = []
    for suffix in allowed_suffixes:
        indices = np.flatnonzero(surface.body_names == f"{side}_{suffix}")
        if not len(indices):
            raise ValueError(f"Unknown allowed collision role {suffix}")
        score = np.sum((ref_points[indices] - target) ** 2, axis=1) / 0.02**2
        score += 2 * np.square(1 - np.sum(ref_normals[indices] * normal, axis=1))
        diverse = []
        for candidate_index in indices[np.argsort(score)]:
            if all(
                np.linalg.norm(surface.points[candidate_index] - surface.points[other])
                > 0.008
                or surface.normals[candidate_index] @ surface.normals[other] < 0.85
                for other in diverse
            ):
                diverse.append(int(candidate_index))
            if len(diverse) > patch_rank:
                break
        if patch_rank >= len(diverse):
            raise ValueError("Not enough distinct robot surface patches")
        index = diverse[patch_rank]
        body_index = int(surface.body_indices[index])
        point_local, robot_normal_local = surface.points[index], surface.normals[index]

        def residual(
            delta,
            body_index=body_index,
            point_local=point_local,
            robot_normal_local=robot_normal_local,
        ):
            cq, cwp, cwq = apply_delta(q, wp, wq, delta, joints)
            bp, br, _ = model.transforms(cq, cwp, cwq)
            point = bp[body_index] + br[body_index] @ point_local
            rn = br[body_index] @ robot_normal_local
            sr = br[surface.body_indices[safety_indices]]
            sp = (
                np.einsum("nij,nj->ni", sr, surface.points[safety_indices])
                + bp[surface.body_indices[safety_indices]]
            )
            local = (sp - object_pose[:3]) @ object_r
            penetration = np.maximum(-obj.approximate_signed(local) - 0.0005, 0) / 0.003
            terms = [
                (point - target) / 0.004,
                (rn - normal) / 0.4,
                0.25 * delta / scale,
                0.2 * (delta - prior) / scale,
                penetration,
            ]
            if support_height is not None:
                terms.append(np.maximum(support_height - sp[:, 2] - 0.001, 0) / 0.004)
            return np.concatenate(terms)

        solved = least_squares(
            residual,
            prior,
            bounds=(lower, upper),
            x_scale=scale,
            max_nfev=max_evaluations,
            ftol=2e-4,
            xtol=2e-4,
            gtol=2e-4,
        )
        cq, cwp, cwq = apply_delta(q, wp, wq, solved.x, joints)
        points, normals = surface.world(model, cq, cwp, cwq)
        local = (points - object_pose[:3]) @ object_r
        gap, signed = obj.exact(local)
        target_error = float(np.linalg.norm(points[index] - target))
        normal_dot = float(normals[index] @ normal)
        penetration_max = float(np.maximum(-signed, 0).max())
        target_gap = float(gap[index])
        feasible = bool(
            target_error <= 0.006
            and normal_dot >= 0.5
            and penetration_max <= 0.004
            and np.all(cq >= model.limits[:, 0] - 1e-7)
            and np.all(cq <= model.limits[:, 1] + 1e-7)
        )
        candidates.append(
            {
                "suffix": suffix,
                "delta": solved.x,
                "q": cq,
                "wrist_position": cwp,
                "wrist_wxyz": cwq,
                "robot_body_name": f"{side}_{suffix}",
                "robot_point_local": point_local.copy(),
                "robot_normal_local": robot_normal_local.copy(),
                "point_error_m": target_error,
                "normal_dot": normal_dot,
                "surface_gap_m": target_gap,
                "penetration_proxy_m": penetration_max,
                "feasible": feasible,
                "objective": float(np.sum(solved.fun**2)),
                "optimizer_status": int(solved.status),
                "evaluations": int(solved.nfev),
                "joint_limit_violations": int(
                    (
                        (cq < model.limits[:, 0] - 1e-7)
                        | (cq > model.limits[:, 1] + 1e-7)
                    ).sum()
                ),
            }
        )
    best = min(candidates, key=lambda item: (not item["feasible"], item["objective"]))
    best["alternatives"] = [
        {
            k: v
            for k, v in item.items()
            if k
            not in {
                "q",
                "wrist_position",
                "wrist_wxyz",
                "delta",
                "robot_point_local",
                "robot_normal_local",
            }
        }
        for item in candidates
    ]
    best["solve_time_s"] = time.perf_counter() - started
    return best


def release_decay(correction, last_contact, decay_frames=7):
    """Preserve last contact correction, then return continuously to source posture."""
    result = np.asarray(correction).copy()
    if decay_frames < 1:
        raise ValueError("Release decay requires at least one frame")
    end = min(last_contact + decay_frames, len(result) - 1)
    if end <= last_contact:
        return result
    for frame in range(last_contact + 1, end + 1):
        u = (frame - last_contact) / (end - last_contact)
        weight = 1 - (10 * u**3 - 15 * u**4 + 6 * u**5)
        result[frame] = result[last_contact] * weight
    result[end + 1 :] = 0
    return result


def project_motion(
    values,
    lower,
    upper,
    max_step,
    *,
    vector_step=False,
    weights=None,
    acceleration_weight=0.2,
    iterations=2000,
):
    """Convex ADMM projection of the complete trajectory onto joint/speed bounds.

    The full q trajectory is constrained, not only its correction. Unknown frames
    retain a soft posture prior and acceleration coupling, without contact terms.
    """
    from scipy.linalg import cho_factor, cho_solve

    values = np.asarray(values, float)
    if (
        values.ndim != 2
        or len(values) < 2
        or max_step <= 0
        or not np.isfinite(values).all()
    ):
        raise ValueError("Finite trajectory[T,D] and positive step bound required")
    length = len(values)
    lower = np.broadcast_to(lower, values.shape)
    upper = np.broadcast_to(upper, values.shape)
    if (lower > upper).any():
        raise ValueError("Invalid motion bounds")
    weights = np.ones(length) if weights is None else np.asarray(weights, float)
    if weights.shape != (length,) or (weights <= 0).any():
        raise ValueError("One positive posture weight per frame required")
    first = np.diff(np.eye(length), axis=0)
    second = np.diff(np.eye(length), n=2, axis=0)
    rho = 8.0
    factor = cho_factor(
        np.diag(weights)
        + acceleration_weight * (second.T @ second)
        + rho * (np.eye(length) + first.T @ first)
    )
    x = np.clip(values, lower, upper)
    z0, z1 = x.copy(), np.diff(x, axis=0)
    u0, u1 = np.zeros_like(z0), np.zeros_like(z1)
    primal = np.inf
    for iteration in range(iterations):
        x = cho_solve(
            factor, weights[:, None] * values + rho * (z0 - u0 + first.T @ (z1 - u1))
        )
        old0, old1 = z0.copy(), z1.copy()
        z0 = np.clip(x + u0, lower, upper)
        difference = first @ x
        z1 = difference + u1
        if vector_step:
            z1 *= np.minimum(
                1.0, max_step / np.maximum(np.linalg.norm(z1, axis=1), 1e-15)
            )[:, None]
        else:
            z1 = np.clip(z1, -max_step, max_step)
        u0 += x - z0
        u1 += difference - z1
        primal = max(
            float(np.max(np.abs(x - z0))), float(np.max(np.abs(difference - z1)))
        )
        dual = rho * max(
            float(np.max(np.abs(z0 - old0))), float(np.max(np.abs(z1 - old1)))
        )
        if primal < 2e-8 and dual < 2e-7:
            break
    if primal > 1e-6:
        raise ValueError(f"Motion projection did not converge: primal={primal}")
    return z0, {
        "iterations": iteration + 1,
        "primal_residual": primal,
        "dual_residual": dual,
    }


def smooth_corrections(raw, active, source_active):
    """Soft posture/acceleration interpolation across unknowns, without invented contact."""
    from scipy.linalg import solve

    raw, active, source_active = (
        np.asarray(raw),
        np.asarray(active),
        np.asarray(source_active),
    )
    if (
        raw.ndim != 2
        or active.shape != raw.shape[:1]
        or source_active.shape != active.shape
    ):
        raise ValueError("Correction timeline shapes disagree")
    length = len(raw)
    weights = np.where(active, 100.0, 0.05)
    weights[[0, length - 1]] = 1e6
    contact_ids = np.flatnonzero(source_active)
    if not len(contact_ids):
        return np.zeros_like(raw)
    weights[int(contact_ids[-1]) + 1 :] = 1e6
    second = np.diff(np.eye(length), n=2, axis=0)
    result = solve(
        np.diag(weights) + second.T @ second, weights[:, None] * raw, assume_a="pos"
    )
    result = release_decay(result, int(contact_ids[-1]))
    result[0] = 0
    return result


def smooth_role_wrist_trajectory(
    original_position,
    candidate_position,
    candidate_wxyz,
    active,
    source_active,
    role_point_wrist,
    target_world,
    *,
    max_rotation_step=0.28,
    release_frames=7,
):
    """Smooth complete wrist rotation while retaining trusted surface points.

    Finger q is fixed upstream. ``role_point_wrist`` is therefore each selected
    collider point transformed by finger FK into the wrist frame. After the SO(3)
    trajectory projection, wrist translation places only trusted points at their
    targets. Unknown frames interpolate translation corrections; they do not
    acquire contact evidence. Normal alignment and whole-hand collision still
    require independent final validation and may reject the result.
    """
    original_position, candidate_position, candidate_wxyz = (
        np.asarray(x, float)
        for x in (original_position, candidate_position, candidate_wxyz)
    )
    active, source_active = np.asarray(active, bool), np.asarray(source_active, bool)
    role_point_wrist, target_world = (
        np.asarray(x, float) for x in (role_point_wrist, target_world)
    )
    count = len(original_position)
    if (
        count < 2
        or original_position.shape != (count, 3)
        or candidate_position.shape != (count, 3)
        or candidate_wxyz.shape != (count, 4)
        or role_point_wrist.shape != (count, 3)
        or target_world.shape != (count, 3)
        or active.shape != (count,)
        or source_active.shape != (count,)
        or (active & ~source_active).any()
        or not np.isfinite(
            np.concatenate(
                [
                    x.ravel()
                    for x in (
                        original_position,
                        candidate_position,
                        candidate_wxyz,
                        role_point_wrist,
                        target_world,
                    )
                ]
            )
        ).all()
    ):
        raise ValueError("Invalid wrist trajectory or trusted/source contact masks")
    rotation = Rotation.from_quat(candidate_wxyz, scalar_first=True)
    relative = (rotation[0].inv() * rotation).as_rotvec()
    # A trajectory crossing the pi branch needs quaternion manifold optimization;
    # do not silently turn such a crossing into a long rotation in this projection.
    if np.max(np.linalg.norm(np.diff(relative, axis=0), axis=1)) >= np.pi:
        raise ValueError("Relative wrist rotation crosses the rotation-vector branch")
    lower, upper = np.full_like(relative, -np.inf), np.full_like(relative, np.inf)
    lower[0] = upper[0] = 0
    lower[-1] = upper[-1] = relative[-1]
    projected, diagnostic = project_motion(
        relative,
        lower,
        upper,
        max_rotation_step,
        vector_step=True,
        weights=np.where(active, 3.0, 1.0),
        acceleration_weight=2.0,
    )
    result_rotation = rotation[0] * Rotation.from_rotvec(projected)
    result_position = candidate_position.copy()
    result_position[active] = target_world[active] - result_rotation[active].apply(
        role_point_wrist[active]
    )
    correction = result_position - original_position
    source_frames = np.flatnonzero(source_active)
    if not len(source_frames):
        correction[:] = 0
    else:
        last_contact = int(source_frames[-1])
        end = min(last_contact + release_frames, count - 1)
        if not active[0]:
            correction[0] = 0
        if end > last_contact:
            correction[end:] = 0
        anchors = np.unique(np.r_[0, np.flatnonzero(active), end, count - 1])
        for axis in range(3):
            correction[:, axis] = np.interp(
                np.arange(count), anchors, correction[anchors, axis]
            )
        correction = release_decay(correction, last_contact, release_frames)
    result_position = original_position + correction
    # Exponential-map distance is bounded by its tangent-space path length.
    # Verify actual SO(3) increments as well, rather than reporting only rv deltas.
    actual_step = (result_rotation[:-1].inv() * result_rotation[1:]).magnitude()
    if actual_step.max() > max_rotation_step + 2e-6:
        raise ValueError("Actual wrist rotation exceeds projected step bound")
    diagnostic["actual_rotation_step_max_rad"] = float(actual_step.max())
    diagnostic["max_rotation_step_rad"] = max_rotation_step
    return result_position, result_rotation.as_quat(scalar_first=True), diagnostic


def select_temporal_roles(candidates_by_frame, *, switch_cost=0.25):
    """Choose feasible collider roles with an explicit sequence-level switch cost.

    Entries are already geometrically validated alternatives with ``body_name``
    and ``geometric_cost``. Unknown source frames are absent, rather than given
    invented targets. No infeasible alternative may become valid through the
    switch penalty, and every supplied trusted frame must retain a selection.
    """
    if not np.isfinite(switch_cost) or switch_cost < 0:
        raise ValueError("Finite nonnegative role switch cost required")
    frames = sorted(candidates_by_frame)
    if not frames:
        return {}, {"switch_count": 0, "total_cost": 0.0}
    costs, back = [], []
    for index, frame in enumerate(frames):
        candidates = candidates_by_frame[frame]
        if not candidates or any(
            not item.get("feasible", False)
            or not item.get("body_name")
            or not np.isfinite(item.get("geometric_cost", np.nan))
            or item["geometric_cost"] < 0
            for item in candidates
        ):
            raise ValueError("Every trusted frame needs feasible finite-cost roles")
        local = np.array([item["geometric_cost"] for item in candidates])
        if not index:
            costs.append(local)
            back.append(np.full(len(local), -1, int))
            continue
        previous = candidates_by_frame[frames[index - 1]]
        transition = np.array(
            [
                [switch_cost * (a["body_name"] != b["body_name"]) for a in previous]
                for b in candidates
            ]
        )
        combined = transition + costs[-1][None, :]
        choice = combined.argmin(axis=1)
        costs.append(local + combined[np.arange(len(local)), choice])
        back.append(choice)
    selected = {}
    choice = int(costs[-1].argmin())
    total = float(costs[-1][choice])
    for index in range(len(frames) - 1, -1, -1):
        selected[frames[index]] = choice
        choice = int(back[index][choice])
    names = [candidates_by_frame[f][selected[f]]["body_name"] for f in frames]
    return selected, {
        "switch_count": sum(a != b for a, b in pairwise(names)),
        "total_cost": total,
        "switch_cost": switch_cost,
    }


def _serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _serializable(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(v) for v in value]
    return value


def run_candidate(
    input_parquet,
    contact_evidence,
    assets,
    object_mesh,
    output,
    *,
    workers=2,
    samples_per_geometry=96,
    max_evaluations=50,
):
    import copy
    import csv
    from concurrent.futures import ThreadPoolExecutor

    import pyarrow as pa
    import pyarrow.parquet as pq

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    table = pq.read_table(input_parquet)
    if len(table) != 1:
        raise ValueError("Expected one nested sequence row")
    original = table.to_pylist()[0]
    candidate = copy.deepcopy(original)
    frame_count = len(original["object_body_position"])
    fps = float(original["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid source frame rate")
    evidence_document = json.loads(Path(contact_evidence).read_text())
    records = evidence_document["records"]
    trusted = trusted_thumb_evidence(records, frame_count, original)
    hashes = {
        str(Path(p).resolve()): sha256(Path(p))
        for p in [input_parquet, contact_evidence, object_mesh, Path(__file__)]
    }
    object_poses = np.concatenate(
        (
            np.asarray(original["object_body_position"]),
            np.asarray(original["object_body_wxyz"]),
        ),
        axis=-1,
    )
    if object_poses.shape != (frame_count, 1, 7):
        raise ValueError("This candidate requires one rigid object")

    def process_side(side):
        xml = Path(assets) / f"{side}_sharpawave.xml"
        model = SharpaCPUModel(
            xml,
            original[f"{side}_robot_frame_names"],
            original[f"{side}_robot_finger_joint_names"],
        )
        surface = CollisionSamples(model, samples_per_geometry)
        obj = ObjectSurface(Path(object_mesh))
        q = np.asarray(original[f"robot_{side}_finger_joints"], float)
        wp = np.asarray(original[f"robot_{side}_wrist_position"], float)
        wq = np.asarray(original[f"robot_{side}_wrist_wxyz"], float)
        thumb = np.array([i for i, n in enumerate(model.joint_names) if "_thumb_" in n])
        baseline_frames = np.array(
            [model.frames(q[i], wp[i], wq[i]) for i in range(frame_count)]
        )
        recorded = np.asarray(original[f"robot_{side}_frames"])
        fk_position = float(np.max(abs(baseline_frames[:, :, :3] - recorded[:, :, :3])))
        fk_orientation = float(
            (
                Rotation.from_quat(
                    baseline_frames[:, :, 3:].reshape(-1, 4), scalar_first=True
                ).inv()
                * Rotation.from_quat(
                    recorded[:, :, 3:].reshape(-1, 4), scalar_first=True
                )
            )
            .magnitude()
            .max()
        )
        if fk_position > 2e-6 or fk_orientation > 2e-5:
            raise ValueError(f"{side} MJCF FK disagrees with input reference")
        source_active = np.zeros(frame_count, bool)
        source_status = np.full(frame_count, "inactive", dtype="U16")
        for record in records:
            if record["side"] == side and record["link_name"].startswith("link_thumb"):
                source_active[int(record["frame"])] = True
                source_status[int(record["frame"])] = record["status"]
        active = np.array([(i, side) in trusted for i in range(frame_count)])
        for frame in np.flatnonzero(active):
            source_status[frame] = trusted[(int(frame), side)]["status"]
        raw = np.zeros((frame_count, 11))
        results = {}
        previous = None
        for frame in np.flatnonzero(active):
            result = solve_contact_frame(
                model,
                surface,
                obj,
                q[frame],
                wp[frame],
                wq[frame],
                object_poses[frame, 0],
                trusted[(frame, side)],
                previous_delta=previous,
                max_evaluations=max_evaluations,
            )
            results[int(frame)] = result
            raw[frame] = result["delta"]
            previous = result["delta"]
            if len(results) % 10 == 0:
                print(
                    json.dumps(
                        {
                            "side": side,
                            "solved": len(results),
                            "trusted_total": int(active.sum()),
                            "frame": int(frame),
                            "feasible_before_smoothing": bool(result["feasible"]),
                        }
                    ),
                    flush=True,
                )
        smooth = smooth_corrections(raw, active, source_active)
        cq, cp, cr, cframes = [], [], [], []
        metrics = []
        role = {
            "active": active,
            "source_contact_active": source_active,
            "source_status": source_status,
            "robot_body_name": np.full(frame_count, "", dtype="U64"),
            "robot_point_local": np.zeros((frame_count, 3)),
            "robot_normal_local": np.zeros((frame_count, 3)),
            "object_point_local": np.zeros((frame_count, 3)),
            "object_normal_local": np.zeros((frame_count, 3)),
            "confidence": np.zeros(frame_count),
            "feasible": np.zeros(frame_count, bool),
            "source_slot": np.full(frame_count, -1, int),
            "point_error_m": np.zeros(frame_count),
            "normal_dot": np.zeros(frame_count),
            "penetration_proxy_m": np.zeros(frame_count),
        }
        for frame in range(frame_count):
            smooth[frame, 6:] = np.clip(
                smooth[frame, 6:],
                model.limits[thumb, 0] - q[frame, thumb],
                model.limits[thumb, 1] - q[frame, thumb],
            )
            fq, fp, fr = apply_delta(
                q[frame], wp[frame], wq[frame], smooth[frame], thumb
            )
            fframes = model.frames(fq, fp, fr)
            cq.append(fq)
            cp.append(fp)
            cr.append(fr)
            cframes.append(fframes)
            points, _normals = surface.world(model, fq, fp, fr)
            op = object_poses[frame, 0]
            rotation = Rotation.from_quat(op[3:], scalar_first=True).as_matrix()
            _, signed = obj.exact((points - op[:3]) @ rotation)
            penetration = float(np.maximum(-signed, 0).max())
            metric = {
                "side": side,
                "frame": frame,
                "source_time_s": frame / fps,
                "source_contact_active": bool(source_active[frame]),
                "trusted_active": bool(active[frame]),
                "source_status": str(source_status[frame]),
                "wrist_position_change_m": float(np.linalg.norm(fp - wp[frame])),
                "wrist_rotation_change_rad": float(np.linalg.norm(smooth[frame, 3:6])),
                "max_finger_change_rad": float(np.abs(fq - q[frame]).max()),
                "max_sampled_local_face_penetration_proxy_m": penetration,
                "robot_body_name": "",
                "point_error_m": None,
                "normal_dot": None,
                "feasible": None,
            }
            if active[frame]:
                chosen = results[frame]
                ev = trusted[(frame, side)]
                name = chosen["robot_body_name"]
                body_index = next(
                    i for i, b in enumerate(model.nodes) if b.name == name
                )
                bp, br, _ = model.transforms(fq, fp, fr)
                point = bp[body_index] + br[body_index] @ chosen["robot_point_local"]
                normal = br[body_index] @ chosen["robot_normal_local"]
                object_point = (
                    rotation @ np.asarray(ev["object_local_position"]) + op[:3]
                )
                object_normal = rotation @ np.asarray(ev["object_local_normal"])
                error = float(
                    np.linalg.norm(point - (object_point - 0.001 * object_normal))
                )
                dot = float(normal @ object_normal)
                feasible = bool(error <= 0.006 and dot >= 0.5 and penetration <= 0.004)
                for key in [
                    "robot_body_name",
                    "robot_point_local",
                    "robot_normal_local",
                ]:
                    role[key][frame] = chosen[key]
                role["object_point_local"][frame] = ev["object_local_position"]
                role["object_normal_local"][frame] = ev["object_local_normal"]
                role["confidence"][frame] = ev["confidence"]
                role["source_slot"][frame] = ev["slot"]
                role["point_error_m"][frame] = error
                role["normal_dot"][frame] = dot
                role["penetration_proxy_m"][frame] = penetration
                role["feasible"][frame] = feasible
                metric.update(
                    robot_body_name=name,
                    point_error_m=error,
                    normal_dot=dot,
                    feasible=feasible,
                )
            metrics.append(metric)
        cq, cp, cr, cframes = map(np.asarray, (cq, cp, cr, cframes))
        fields = {
            f"robot_{side}_finger_joints": cq.tolist(),
            f"robot_{side}_wrist_position": cp.tolist(),
            f"robot_{side}_wrist_wxyz": cr.tolist(),
            f"robot_{side}_frames": cframes.tolist(),
        }
        stats = {
            "trusted_frames": int(active.sum()),
            "feasible_frames": int(role["feasible"].sum()),
            "failed_trusted_frames": np.flatnonzero(
                active & ~role["feasible"]
            ).tolist(),
            "source_contact_unknown_frames": np.flatnonzero(
                source_active & ~active
            ).tolist(),
            "fk_max_position_difference_m": fk_position,
            "fk_max_orientation_difference_rad": fk_orientation,
            "wrist_translation_change_max_m": float(
                np.linalg.norm(cp - wp, axis=-1).max()
            ),
            "finger_change_max_rad": float(np.abs(cq - q).max()),
            "finger_speed_max_rad_s": float(np.abs(np.diff(cq, axis=0) * fps).max()),
            "original_finger_speed_max_rad_s": float(
                np.abs(np.diff(q, axis=0) * fps).max()
            ),
            "correction_speed_max_rad_s": float(
                np.abs(np.diff(cq - q, axis=0) * fps).max()
            ),
            "wrist_speed_max_m_s": float(
                np.linalg.norm(np.diff(cp, axis=0) * fps, axis=-1).max()
            ),
            "all_frame_penetration_proxy_max_m": max(
                m["max_sampled_local_face_penetration_proxy_m"] for m in metrics
            ),
            "other_four_finger_joints_unchanged": bool(
                np.array_equal(cq[:, 5:], q[:, 5:])
            ),
        }
        return fields, role, metrics, stats, results, surface.asset_hashes, raw, smooth

    with ThreadPoolExecutor(max_workers=min(2, max(1, workers))) as pool:
        futures = {side: pool.submit(process_side, side) for side in ("left", "right")}
        processed = {side: future.result() for side, future in futures.items()}
    all_metrics = []
    stats = {}
    role_arrays = {
        "source_time_s": np.arange(frame_count) / fps,
        "side_names": np.array(["left", "right"]),
    }
    for side, (
        fields,
        role,
        metrics,
        stat,
        results,
        assets_hash,
        raw,
        smooth,
    ) in processed.items():
        candidate.update(fields)
        all_metrics.extend(metrics)
        stats[side] = stat
        hashes.update(assets_hash)
        (output / f"{side}-solver-results.json").write_text(
            json.dumps(_serializable(results), indent=2) + "\n"
        )
        np.savez_compressed(
            output / f"{side}-corrections.npz", raw=raw, smoothed=smooth
        )
    for key in processed["left"][1]:
        role_arrays[key] = np.stack(
            [processed[side][1][key] for side in ("left", "right")], axis=1
        )
    np.savez_compressed(output / "contact-role-targets.npz", **role_arrays)
    changed = [key for key in original if original[key] != candidate[key]]
    allowed = {
        f"robot_{side}_{suffix}"
        for side in ("left", "right")
        for suffix in ("wrist_position", "wrist_wxyz", "finger_joints", "frames")
    }
    if not set(changed) <= allowed:
        raise AssertionError("Candidate changed fields outside robot FK/reference")
    pq.write_table(
        pa.Table.from_pylist([candidate], schema=table.schema), output / "data.parquet"
    )
    with (output / "all-frame-metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_metrics[0]))
        writer.writeheader()
        writer.writerows(all_metrics)
    (output / "all-frame-metrics.json").write_text(
        json.dumps(all_metrics, indent=2) + "\n"
    )
    report = {
        "schema": "nvidia_v2d.contact_aware_reference.v1",
        "input": str(Path(input_parquet).resolve()),
        "object_and_mano_fields_unchanged": True,
        "changed_fields": changed,
        "frame_count": frame_count,
        "sides": stats,
        "training_allowed": False,
        "training_gate": "Offline geometry candidate only; review failed trusted frames, motion continuity, sampled penetration and physical rollout before training",
        "role_contract": "active means trusted source evidence, even if IK failed; never use feasible to silently erase a target",
        "normal_convention": "object inward; robot outward; aligned directions point from robot into object",
        "inactive_unknown": "No contact attraction; original posture plus neighboring correction acceleration regularization only",
        "legacy_ik_diagnostics": "Original robot_*_frame_task_errors and num_optimization_iterations are historical original-IK diagnostics and preserved verbatim; new candidate diagnostics are in all-frame-metrics and solver-results",
        "acceptance": {
            "point_error_to_1mm_outside_target_max_m": 0.006,
            "normal_dot_min": 0.5,
            "sampled_local_face_penetration_max_m": 0.004,
        },
        "geometry": "Per-geom robot collision convex hulls; fixed surface samples; exact watertight object triangles for final signed face-distance proxy; no force amplitude or policy-success fitting",
        "input_sha256": hashes,
    }
    if not all(sha256(Path(path)) == digest for path, digest in hashes.items()):
        raise RuntimeError("Input changed while candidate was being generated")
    report["outputs_sha256"] = {
        p.name: sha256(p) for p in output.iterdir() if p.is_file()
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"output": str(output), "sides": stats, "training_allowed": False}),
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--contact-evidence", type=Path, required=True)
    parser.add_argument(
        "--assets",
        type=Path,
        required=True,
        help="Directory containing left/right_sharpawave.xml",
    )
    parser.add_argument("--object-mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--samples-per-geometry", type=int, default=96)
    parser.add_argument("--max-evaluations", type=int, default=50)
    args = parser.parse_args()
    run_candidate(
        args.input_parquet,
        args.contact_evidence,
        args.assets,
        args.object_mesh,
        args.output,
        workers=args.workers,
        samples_per_geometry=args.samples_per_geometry,
        max_evaluations=args.max_evaluations,
    )


if __name__ == "__main__":
    main()
