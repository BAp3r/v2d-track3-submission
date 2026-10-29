"""Conservative, mesh-supported repair of isolated contact face jumps.

Stored MANO contact positions/normals are WORLD-space; mesh candidates and the
composable NPZ patch are OBJECT-LOCAL. Normals point inward, as required by the
reference wrench convention. This does not infer measured contact or repair
every ambiguous reconstructed contact. Unknown labels retain their raw values.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

STATUS_CODES = {"inactive": 0, "consistent": 1, "repaired": 2, "unknown": 3}
SIDES = ("left", "right")


@dataclass(frozen=True)
class ContactRepairPolicy:
    max_hand_gap_m: float = 0.01
    max_point_shift_m: float = 0.01
    side_margin_m: float = 0.0001
    min_hand_direction_cos: float = 0.25
    min_anchor_agreement: float = 0.90
    max_jump_dot: float = -0.50
    min_candidate_anchor_dot: float = 0.90
    min_opposite_face_dot: float = 0.80
    surface_atol_m: float = 1e-5
    min_surface_normal_dot: float = 0.99
    ambiguous_gap_margin_m: float = 0.0005
    equivalent_patch_radius_m: float = 0.003
    equivalent_patch_normal_dot: float = 0.95


def _poses(cells: dict) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation

    positions = np.asarray(cells["object_body_position"], dtype=float)
    quaternion = np.asarray(cells["object_body_wxyz"], dtype=float)
    if positions.ndim != 3 or positions.shape[1:] != (1, 3) or quaternion.shape != (*positions.shape[:2], 4):
        raise ValueError("Contact repair currently supports one rigid object body")
    if not np.isfinite(positions).all() or not np.isfinite(quaternion).all():
        raise ValueError("Nonfinite object poses")
    if not np.allclose(np.linalg.norm(quaternion, axis=-1), 1, rtol=0, atol=1e-5):
        raise ValueError("Object quaternions must be normalized")
    return positions[:, 0], Rotation.from_quat(quaternion[:, 0], scalar_first=True).as_matrix()


def _local(vectors: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return np.einsum("tji,tsj->tsi", rotation, vectors)


def _world(vectors: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return np.einsum("tij,tsj->tsi", rotation, vectors)


def _fields(cells: dict, side: str, frames: int) -> tuple[np.ndarray, ...]:
    prefix = f"mano_{side}_"
    ids = np.asarray(cells[prefix + "object_contact_part_ids"])
    values = [np.asarray(cells[prefix + key], dtype=float) for key in
              ("link_contact_positions", "object_contact_positions", "object_contact_normals")]
    if ids.ndim != 2 or ids.shape[0] != frames or any(value.shape != (*ids.shape, 3) for value in values):
        raise ValueError("Contact arrays must share [frames, slots] dimensions")
    if not np.isin(ids, [0, 1]).all() or any(not np.isfinite(value).all() for value in values):
        raise ValueError("Expected finite single-object contact arrays with part IDs 0/1")
    return (ids, *values)


def _side_geometry(hand: np.ndarray, point: np.ndarray, normal: np.ndarray):
    gap = np.linalg.norm(hand - point, axis=-1)
    signed = np.sum(normal * (hand - point), axis=-1)
    direction = -signed / np.maximum(gap, 1e-12)
    return gap, signed, direction


def candidate_faces(mesh, hand, original_point, original_normal, anchors, policy):
    """Evaluate actual closest points on triangles, not a sign-flipped normal."""
    import trimesh

    points = trimesh.triangles.closest_point(mesh.triangles, np.broadcast_to(hand, (len(mesh.faces), 3)))
    normals = -mesh.face_normals
    gap, signed, direction = _side_geometry(hand, points, normals)
    shift = np.linalg.norm(points - original_point, axis=-1)
    anchor_dot = normals @ np.asarray(anchors).T
    opposite = -(normals @ original_normal)
    eligible = ((gap <= policy.max_hand_gap_m) & (shift <= policy.max_point_shift_m)
                & (signed <= -policy.side_margin_m) & (direction >= policy.min_hand_direction_cos)
                & (anchor_dot.min(axis=1) >= policy.min_candidate_anchor_dot)
                & (opposite >= policy.min_opposite_face_dot))
    order = np.lexsort((np.arange(len(mesh.faces)), gap))
    choices = order[eligible[order]]
    best = None if not len(choices) else int(choices[0])
    ambiguous = False
    if best is not None:
        rivals = choices[gap[choices] <= gap[best] + policy.ambiguous_gap_margin_m]
        ambiguous = bool(np.any((normals[rivals] @ normals[best] < policy.equivalent_patch_normal_dot)
                                | (np.linalg.norm(points[rivals] - points[best], axis=-1) > policy.equivalent_patch_radius_m)))
    inspected = sorted(set(order[:12].tolist() + choices[:8].tolist()))
    evidence = [{"face": int(face), "object_local_position": points[face].tolist(),
                 "object_local_normal": normals[face].tolist(), "hand_gap_m": float(gap[face]),
                 "signed_hand_minus_object_m": float(signed[face]), "point_shift_m": float(shift[face]),
                 "direction_cos": float(direction[face]), "anchor_dots": anchor_dot[face].tolist(),
                 "opposite_original_normal_dot": float(opposite[face]), "eligible": bool(eligible[face])}
                for face in inspected]
    return best, ambiguous, points, normals, evidence


def repair_contact_reference(cells: dict, mesh, policy: ContactRepairPolicy | None = None):
    """Return (repaired_cells, audit, object_local_patch), leaving input untouched.

    Only isolated opposite-face outliers bracketed by two independently
    hand-facing, adjacent active contacts may change. Missing/contradictory
    evidence becomes unknown. No inactive frame is bridged and no label is added.
    """
    import trimesh

    policy = policy or ContactRepairPolicy()
    if not mesh.is_winding_consistent:
        raise ValueError("Mesh winding must be consistent before inward normals are meaningful")
    position, rotation = _poses(cells)
    records = []
    patch = {"source_object_body_position": np.asarray(cells["object_body_position"], dtype=float),
             "source_object_body_wxyz": np.asarray(cells["object_body_wxyz"], dtype=float)}
    for side in SIDES:
        ids, hand_world, object_world, normals_world = _fields(cells, side, len(position))
        active = ids > 0
        hand = _local(hand_world - position[:, None], rotation)
        point = _local(object_world - position[:, None], rotation)
        normal = _local(normals_world, rotation)
        lengths = np.linalg.norm(normal, axis=-1)
        if np.any(active & (np.abs(lengths - 1) > 1e-5)):
            raise ValueError("Active reference normals must be unit vectors")
        unit = normal / np.maximum(lengths[..., None], 1e-12)
        frames, slots = np.where(active)
        if active.any():
            _, distances, faces = trimesh.proximity.closest_point(mesh, point[active])
        else:
            distances, faces = np.empty(0), np.empty(0, dtype=int)
        surface = np.zeros(ids.shape, dtype=bool)
        alignment = np.sum(unit[active] * -mesh.face_normals[faces], axis=-1)
        surface[active] = (distances <= policy.surface_atol_m) & (alignment >= policy.min_surface_normal_dot)
        gap, signed, direction = _side_geometry(hand, point, unit)
        good = (active & surface & (gap <= policy.max_hand_gap_m)
                & (signed <= -policy.side_margin_m) & (direction >= policy.min_hand_direction_cos))
        status = np.where(active, STATUS_CODES["unknown"], STATUS_CODES["inactive"]).astype(np.uint8)
        status[good] = STATUS_CODES["consistent"]
        source_status = status.copy()
        confidence = np.where(good, np.clip(direction, 0, 1), 0.)
        updated_point, updated_normal = point.copy(), normal.copy()
        for index, (frame, slot) in enumerate(zip(frames, slots, strict=True)):
            record: dict[str, Any] = {
                "side": side, "frame": int(frame), "source_time_s": float(frame / cells["fps"]),
                "slot": int(slot), "link_name": cells["mano_link_names"][slot], "part_id": int(ids[frame, slot]),
                "original_face": int(faces[index]), "original_object_local_position": point[frame, slot].tolist(),
                "original_object_local_normal": normal[frame, slot].tolist(), "hand_object_local_position": hand[frame, slot].tolist(),
                "original_hand_gap_m": float(gap[frame, slot]), "original_signed_hand_minus_object_m": float(signed[frame, slot]),
                "original_surface_distance_m": float(distances[index]), "original_normal_face_dot": float(alignment[index]),
                "reason": "mesh_normal_and_hand_side_consistent" if good[frame, slot] else "insufficient_or_conflicting_geometry"}
            adjacent = (0 < frame < len(position) - 1 and good[frame - 1, slot] and good[frame + 1, slot]
                        and ids[frame - 1, slot] == ids[frame, slot] == ids[frame + 1, slot])
            if adjacent:
                anchors = unit[[frame - 1, frame + 1], slot]
                old_dots = anchors @ unit[frame, slot]
                anchor_agreement = float(anchors[0] @ anchors[1])
                record.update(adjacent_frame_ids=[int(frame - 1), int(frame + 1)],
                              original_adjacent_normal_dots=old_dots.tolist(), adjacent_anchor_dot=anchor_agreement)
                suspect = (surface[frame, slot] and signed[frame, slot] > policy.side_margin_m
                           and anchor_agreement >= policy.min_anchor_agreement and np.all(old_dots < policy.max_jump_dot))
                if suspect:
                    best, ambiguous, candidates, normals, evidence = candidate_faces(
                        mesh, hand[frame, slot], point[frame, slot], unit[frame, slot], anchors, policy)
                    record["candidate_faces"] = evidence
                    if best is None or ambiguous:
                        record["reason"] = "no_supported_opposite_face" if best is None else "ambiguous_supported_surface_patches"
                    else:
                        updated_point[frame, slot], updated_normal[frame, slot] = candidates[best], normals[best]
                        status[frame, slot] = STATUS_CODES["repaired"]
                        _, new_signed, new_direction = _side_geometry(hand[frame, slot], candidates[best], normals[best])
                        confidence[frame, slot] = min(float(new_direction), float((anchors @ normals[best]).min()))
                        record.update(reason="opposite_face_outlier_with_two_active_side_consistent_anchors",
                                      selected_face=int(best), repaired_signed_hand_minus_object_m=float(new_signed),
                                      repaired_adjacent_normal_dots=(anchors @ normals[best]).tolist())
            record.update(status=next(name for name, code in STATUS_CODES.items() if code == status[frame, slot]),
                          source_status=next(name for name, code in STATUS_CODES.items() if code == source_status[frame, slot]),
                          confidence=float(confidence[frame, slot]), trusted=bool(status[frame, slot] in (1, 2)),
                          object_local_position=updated_point[frame, slot].tolist(),
                          object_local_normal=updated_normal[frame, slot].tolist(),
                          point_shift_m=float(np.linalg.norm(updated_point[frame, slot] - point[frame, slot])))
            records.append(record)
        patch.update({f"{side}_object_local_positions": updated_point, f"{side}_object_local_normals": updated_normal,
                      f"{side}_original_object_local_positions": point, f"{side}_original_object_local_normals": normal,
                      f"{side}_original_world_positions": object_world, f"{side}_original_world_normals": normals_world,
                      f"{side}_part_ids": ids.copy(), f"{side}_status": status, f"{side}_confidence": confidence,
                      f"{side}_trusted": np.isin(status, [1, 2]), f"{side}_source_status": source_status,
                      f"{side}_source_active": active.copy(),
                      f"{side}_ignored_unknown": status == STATUS_CODES["unknown"]})
        for suffix in ("link_contact_positions", "link_contact_normals", "object_contact_positions",
                       "object_contact_normals", "object_contact_part_ids", "tips_distance"):
            if f"mano_{side}_{suffix}" in cells:
                patch[f"{side}_source_{suffix}"] = np.asarray(cells[f"mano_{side}_{suffix}"])
    repaired = apply_object_local_patch(cells, patch, only_changed=True)
    audit = {"schema": "nvidia_v2d.contact_reference_repair.v1", "source_fps": cells["fps"],
             "stored_input_coordinate_frame": "world", "patch_coordinate_frame": "object-local",
             "normal_convention": "inward = negative mesh face normal; expected n dot (hand - object) < 0",
             "confidence_semantics": "geometric consistency score; not a calibrated probability or force measurement",
             "status_codes": STATUS_CODES, "policy": asdict(policy), "records": records,
             "source_status_semantics": "status before any face repair: 0 inactive, 1 geometrically consistent, 3 unknown",
             "counts": dict(Counter(row["status"] for row in records)),
             "mesh_watertight": bool(mesh.is_watertight), "mesh_winding_consistent": bool(mesh.is_winding_consistent),
             "discrete_contact_ids_unchanged": True, "unknown_policy": "preserve raw values; exclude from hard IK targets",
             "ready_for_composition": True, "training_allowed": False}
    return repaired, audit, patch


def apply_object_local_patch(cells: dict, patch: dict, *, only_changed: bool = False) -> dict:
    """Project object-local contacts using cells' CURRENT object poses.

    Default use for composition transforms every active slot after object-pose
    repair. Inactive values and IDs are preserved. ``only_changed`` is for the
    standalone normals candidate with the original poses; it preserves all
    unedited stored world values bit-for-bit.
    """
    position, rotation = _poses(cells)
    if only_changed and (not np.array_equal(cells["object_body_position"], patch["source_object_body_position"])
                         or not np.array_equal(cells["object_body_wxyz"], patch["source_object_body_wxyz"])):
        raise ValueError("only_changed requires the original object poses")
    result = copy.deepcopy(cells)
    for side in SIDES:
        ids, _hand, point, normal = _fields(cells, side, len(position))
        if not np.array_equal(ids, patch[f"{side}_part_ids"]):
            raise ValueError("Contact patch part IDs/timing differ from the target parquet")
        mask = patch[f"{side}_status"] == STATUS_CODES["repaired"] if only_changed else ids > 0
        for suffix, local_values, original, translate in [
            ("positions", patch[f"{side}_object_local_positions"], point, True),
            ("normals", patch[f"{side}_object_local_normals"], normal, False)]:
            if local_values.shape != (*ids.shape, 3) or not np.isfinite(local_values).all():
                raise ValueError("Invalid object-local contact patch dimensions/values")
            world_values = _world(local_values, rotation)
            if translate:
                world_values += position[:, None]
            updated = original.copy()
            updated[mask] = world_values[mask]
            result[f"mano_{side}_object_contact_{suffix}"] = updated.tolist()
    return result


def _spans(mask: np.ndarray, fps: float) -> list[dict]:
    frames = np.flatnonzero(mask)
    groups: list[list[int]] = []
    for frame in frames:
        if not groups or frame > groups[-1][1] + 1:
            groups.append([int(frame), int(frame)])
        else:
            groups[-1][1] = int(frame)
    return [{"first_frame": first, "last_frame": last, "first_time_s": first / fps,
             "last_time_s": last / fps} for first, last in groups]


def training_contact_candidate(cells: dict, patch: dict) -> tuple[dict, dict]:
    """Exclude unknown source contacts from wrench supervision in a NEW candidate.

    Part ID 0 is a transport mask here, NOT evidence of true absence of contact.
    Use only with unintended-contact penalties disabled on the ignored frames.
    Compose object-local geometry FIRST, then call this function; the raw source
    activity/IDs and every original contact array remain separately in the patch.
    """
    result = copy.deepcopy(cells)
    fps = float(cells["fps"])
    per_hand, source_counts, trusted_counts = {}, [], []
    role_counts = []
    for side in SIDES:
        key = f"mano_{side}_object_contact_part_ids"
        ids = np.asarray(cells[key])
        if not np.array_equal(ids, patch[f"{side}_part_ids"]):
            raise ValueError("Training candidate requires the preserved original contact IDs")
        active, trusted = ids > 0, np.asarray(patch[f"{side}_trusted"], dtype=bool)
        unknown = np.asarray(patch[f"{side}_status"]) == STATUS_CODES["unknown"]
        if (trusted.shape != ids.shape or unknown.shape != ids.shape or np.any(trusted & ~active)
                or not np.array_equal(unknown, active & ~trusted)):
            raise ValueError("Unknown/trusted masks do not partition source-active contacts")
        masked = ids.copy()
        masked[unknown] = 0
        result[key] = masked.tolist()
        source_count, trusted_count = active.sum(axis=1), trusted.sum(axis=1)
        source_counts.append(source_count)
        trusted_counts.append(trusted_count)
        for slot in range(ids.shape[1]):
            ignored = unknown[:, slot]
            if ignored.any():
                role_counts.append({"side": side, "slot": slot, "link_name": cells["mano_link_names"][slot],
                                    "ignored_source_slots": int(ignored.sum()),
                                    "ignored_4_to_8s_inclusive": int(ignored[(np.arange(len(ids)) / fps >= 4)
                                                                                & (np.arange(len(ids)) / fps <= 8)].sum())})
        per_hand[side] = {"source_active_slots": int(active.sum()), "trusted_supervision_slots": int(trusted.sum()),
                          "ignored_unknown_slots": int(unknown.sum()),
                          "source_active_count_per_frame": source_count.tolist(),
                          "trusted_target_count_per_frame": trusted_count.tolist(),
                          "source_contact_but_no_reliable_target_spans": _spans((source_count > 0) & (trusted_count == 0), fps)}
    source_any, trusted_any = np.sum(source_counts, axis=0) > 0, np.sum(trusted_counts, axis=0) > 0
    allowed = {f"mano_{side}_object_contact_part_ids" for side in SIDES}
    if any(cells[key] != result[key] for key in cells if key not in allowed):
        raise ValueError("Training mask unexpectedly changed geometric or robot fields")
    receipt = {"schema": "nvidia_v2d.contact_supervision_mask.v1", "mode": "training_candidate",
               "mask_semantics": "exclude unknown from reference wrench/missed-contact supervision; NOT true no-contact labels",
               "raw_source_preserved_in_patch": True, "unknown_by_role": role_counts, "hands": per_hand,
               "any_source_contact_but_no_reliable_target_spans": _spans(source_any & ~trusted_any, fps),
               "source_no_contact_spans": _spans(~source_any, fps),
               "required_training_contract": "unintended_contact penalty must be zero on every ignored-unknown source frame; revalidate after runtime resampling",
               "requires_fresh_replay": True, "all_mask_checks_passed": True,
               "geometry_fields_unchanged_by_mask": True, "training_allowed": False}
    return result, receipt


def validate_repair(original: dict, repaired: dict, patch: dict, mesh) -> dict:
    """Independent whole-mesh geometric checks plus discrete timing invariants."""
    import trimesh

    allowed = {f"mano_{side}_object_contact_{suffix}" for side in SIDES for suffix in ("positions", "normals")}
    if original.keys() != repaired.keys() or any(original[key] != repaired[key] for key in original if key not in allowed):
        raise ValueError("Repair changed unrelated source fields")
    position, rotation = _poses(repaired)
    results = []
    for side in SIDES:
        ids, hand_world, point_world, normal_world = _fields(repaired, side, len(position))
        active = ids > 0
        if not np.array_equal(ids, original[f"mano_{side}_object_contact_part_ids"]):
            raise ValueError("Repair changed discrete contact timing")
        changed = patch[f"{side}_status"] == STATUS_CODES["repaired"]
        point = _local(point_world - position[:, None], rotation)
        normal = _local(normal_world, rotation)
        hand = _local(hand_world - position[:, None], rotation)
        if active.any():
            _, distances, face = trimesh.proximity.closest_point(mesh, point[active])
        else:
            distances, face = np.empty(0), np.empty(0, dtype=int)
        dots = np.sum(normal[active] * -mesh.face_normals[face], axis=-1)
        if np.max(distances, initial=0) > 1e-5 or np.min(dots, initial=1) < .99:
            raise ValueError("Repaired contact point/normal no longer agrees with the mesh surface")
        gap, signed, _ = _side_geometry(hand, point, normal)
        if np.any(signed[changed] >= -1e-4) or np.any(gap[changed] > .01):
            raise ValueError("Repaired points fail independent hand-side/gap constraints")
        for suffix in ("positions", "normals"):
            new = np.asarray(repaired[f"mano_{side}_object_contact_{suffix}"])
            old = np.asarray(original[f"mano_{side}_object_contact_{suffix}"])
            if not np.array_equal(new[~changed], old[~changed]):
                raise ValueError("A contact outside the repair mask changed")
        adjacent_dots = []
        for frame, slot in zip(*np.where(changed), strict=True):
            if not (0 < frame < len(position) - 1 and active[frame - 1, slot] and active[frame + 1, slot]):
                raise ValueError("Repair bridges a missing discrete contact")
            adjacent_dots.extend((normal[[frame - 1, frame + 1], slot] @ normal[frame, slot]).tolist())
        if adjacent_dots and min(adjacent_dots) < .9:
            raise ValueError("Repaired normal fails independent temporal direction check")
        results.append({"side": side, "active_slots": int(active.sum()), "changed_slots": int(changed.sum()),
                        "max_surface_distance_m": float(np.max(distances, initial=0)),
                        "min_inward_mesh_face_dot": float(np.min(dots, initial=1)),
                        "repaired_adjacent_normal_min_dot": min(adjacent_dots) if adjacent_dots else None,
                        "part_ids_exactly_unchanged": True, "unrepaired_world_fields_exactly_unchanged": True})
    return {"all_checks_passed": True, "hands": results,
            "other_source_fields_exactly_unchanged": True,
            "limitation": "local geometric validation; no watertight inside test, measured force, or physical feasibility claim"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import trimesh

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--object-mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-mode", choices=("audit", "training-candidate"), default="audit")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output directory; original candidates are never overwritten")
    source_paths = [args.input_parquet.resolve(), args.object_mesh.resolve(), Path(__file__).resolve()]
    input_sha = {str(path): _sha(path) for path in source_paths}
    table = pq.read_table(args.input_parquet)
    if table.num_rows != 1:
        raise ValueError("Expected a one-row nested-array ManoSharpa parquet")
    cells = table.to_pylist()[0]
    mesh = trimesh.load_mesh(args.object_mesh, process=False)
    repaired, audit, patch = repair_contact_reference(cells, mesh)
    validation = validate_repair(cells, repaired, patch, mesh)
    candidate = repaired
    supervision = None
    if args.export_mode == "training-candidate":
        candidate, supervision = training_contact_candidate(repaired, patch)
    args.output.mkdir(parents=True)
    candidate_table = pa.Table.from_pylist([candidate], schema=table.schema)
    pq.write_table(candidate_table, args.output / "data.parquet")
    # Validate the actual serialized candidate, not only the in-memory arrays.
    serialized_table = pq.read_table(args.output / "data.parquet")
    if not serialized_table.equals(candidate_table, check_metadata=True):
        raise ValueError("Serialized contact candidate differs from the schema-typed fields")
    # The original Arrow schema includes float32 arrays; its normal quantization
    # is expected. Independently recheck the stored geometry, with original IDs
    # restored only in this audit view so masked unknowns remain in the checks.
    serialized = serialized_table.to_pylist()[0]
    geometry_view = copy.deepcopy(serialized)
    for side in SIDES:
        geometry_view[f"mano_{side}_object_contact_part_ids"] = cells[f"mano_{side}_object_contact_part_ids"]
    validate_repair(cells, geometry_view, patch, mesh)
    validation["serialized_geometry_and_typed_fields_verified"] = True
    np.savez_compressed(args.output / "contact-repair.npz", **patch)
    audit.update(evidence_scope="normal repair before the optional supervision mask",
                 export_mode=args.export_mode, exported_part_ids_equal_source=supervision is None)
    (args.output / "contact-evidence.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    (args.output / "validation.json").write_text(json.dumps(validation, indent=2, allow_nan=False) + "\n")
    if supervision is not None:
        (args.output / "supervision-mask.json").write_text(json.dumps(supervision, indent=2, allow_nan=False) + "\n")
    if input_sha != {str(path): _sha(path) for path in source_paths}:
        raise ValueError("An input changed during repair")
    manifest = {"schema": "nvidia_v2d.contact_reference_repair_manifest.v1", "inputs_sha256": input_sha,
                "outputs_sha256": {path.name: _sha(path) for path in sorted(args.output.iterdir())},
                "command": {"input_parquet": str(args.input_parquet.resolve()), "object_mesh": str(args.object_mesh.resolve()),
                            "export_mode": args.export_mode},
                "coordinates": {"parquet_contact_fields": "world", "npz_contact_patch": "object-local; metre positions; inward unit normals"},
                "composition": "apply_object_local_patch(raw_id_cells, patch) FIRST; then training_contact_candidate(combined_cells, patch) to exclude unknown supervision",
                "counts": audit["counts"], "all_geometric_checks_passed": True,
                "unknown_preserved_in_raw_patch": True, "unknown_excluded_from_supervision": supervision is not None,
                "source_status_semantics": audit["source_status_semantics"],
                "ready_for_composition": True, "training_allowed": False}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "counts": audit["counts"], "all_checks_passed": True}))


if __name__ == "__main__":
    main()
