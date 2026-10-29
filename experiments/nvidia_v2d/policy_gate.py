"""训练门控绑定本段、本路线与实际文件，不能借用另一组已通过的凭证。"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal


def verify_gate(
    path: Path,
    *,
    backend: Literal["official", "multitip"],
    parquet: Path,
    episode_id: str | None = None,
) -> dict[str, Any]:
    verified: dict[str, str] = {}

    def check(file: Path, expected: str | None = None) -> bytes:
        if not file.is_absolute() or any(p.is_symlink() for p in (file, *file.parents)):
            raise ValueError("gate文件必须为无符号链接的绝对路径")
        content = file.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError(f"gate绑定的文件已经变化: {file.name}")
        verified[str(file)] = digest
        return content

    def bound(file: Path, expected: str) -> bytes:
        if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
            raise ValueError("gate缺少有效SHA256绑定")
        return check(file, expected)

    try:
        gate = json.loads(check(path))
        if gate.get("schema_version") in ("nvidia_v2d.repaired_reference_gate.v1", "nvidia_v2d.repaired_reference_gate.v2"):
            # A derived reference keeps the original input gate and binds every
            # repair/audit artifact. It cannot borrow the old output's hash or
            # silently declare full geometric/physical quality established.
            if (gate.get("training_ready") is not True or gate.get("blockers")
                    or gate.get("purpose") != "mechanism_validation"
                    or gate.get("geometry_quality_accepted") is not False):
                raise ValueError("repaired reference gate has not passed its explicit input audit")
            parent_path = Path(gate["parent_gate_path"])
            parent = json.loads(bound(parent_path, gate["parent_gate_sha256"]))
            if parent.get("schema_version") != "nvidia_v2d.sharpa_training_gate.v1":
                raise ValueError("repair parent must be the original reference gate")
            original = Path(gate["original_parquet"])
            inherited = verify_gate(parent_path, backend=backend, parquet=original, episode_id=episode_id)
            verified.update(inherited["verified_files"])
            manifest = json.loads(bound(Path(gate["repair_manifest_path"]), gate["repair_manifest_sha256"]))
            if (manifest.get("status") != "completed" or manifest.get("training_allowed") is not True
                    or Path(manifest["input_parquet"]) != original
                    or Path(manifest["output_parquet"]) != parquet
                    or manifest["frame_count"] != inherited["frame_count"]
                    or manifest.get("source_unchanged") is not True
                    or not manifest.get("input_sha256")):
                raise ValueError("repaired reference manifest is incomplete or belongs to another sample")
            for file, digest in manifest["input_sha256"].items():
                bound(Path(file), digest)
            bound(original, manifest["original_sha256"])
            bound(parquet, manifest["output_sha256"])
            audit = json.loads(bound(Path(manifest["audit_path"]), manifest["audit_sha256"]))
            required = ("finite_arrays", "frame_count", "contact_normals", "object_support",
                        "robot_contact_mapping", "release_clearance", "unchanged_auxiliary_and_physics")
            if gate["schema_version"] == "nvidia_v2d.repaired_reference_gate.v2":
                if manifest.get("schema") != "nvidia_v2d.collision_reference_repair.v1":
                    raise ValueError("v2 repair gate requires actual collision union provenance")
                bound(Path(manifest["collision_union_path"]), manifest["collision_union_sha256"])
                if audit.get("collision_union_sha256") != manifest["collision_union_sha256"]:
                    raise ValueError("actual collision union was not accepted with this reference")
                required += ("actual_object_collision_union", "full_hand_collision_union",
                             "temporal_continuity", "source_contact_evidence_preserved")
            if (audit.get("candidate_sha256") != manifest["output_sha256"]
                    or any(audit.get("checks", {}).get(name) is not True for name in required)):
                raise ValueError("repaired reference is missing required independent acceptance checks")
            role_path, role_sha = manifest.get("semantic_map_path"), manifest.get("semantic_map_sha256")
            if role_path is not None or role_sha is not None:
                bound(Path(role_path), role_sha)
                if audit.get("semantic_map_sha256") != role_sha:
                    raise ValueError("semantic map was not accepted with this reference")
            return {**gate, "frame_count": inherited["frame_count"], "verified_files": verified,
                    "semantic_map_path": role_path, "semantic_map_sha256": role_sha}
        if (
            gate.get("schema_version") != "nvidia_v2d.sharpa_training_gate.v1"
            or gate.get("training_ready") is not True
            or gate.get("input_contract_passed") is not True
            or gate.get("purpose") != "mechanism_validation"
            or gate.get("mechanism_validation") is not True
            or gate.get("blockers")
        ):
            raise ValueError("gate没有通过明确的机制验证输入准入")
        documents = {}
        for key in ("episode_gate", "geometry_audit", "retarget_manifest"):
            documents[key] = json.loads(bound(Path(gate[f"{key}_path"]), gate[f"{key}_sha256"]))
        episode, audit, retarget = (
            documents[k] for k in ("episode_gate", "geometry_audit", "retarget_manifest")
        )
        expected_backend = {"official": "official-ik", "multitip": "unidex-inspired-ik-v2"}[backend]
        if retarget["backend"] != expected_backend or audit["backend"] != expected_backend:
            raise ValueError("gate重定向路线不一致")
        if episode_id is not None and episode["source"]["sequence_id"] != episode_id:
            raise ValueError("gate视频编号不一致")
        if episode.get("input_contract_passed") is not True or episode.get("blockers"):
            raise ValueError("gate上游输入未通过准入")
        if retarget["status"] != "completed" or audit["status"] != "succeeded":
            raise ValueError("gate依赖未成功完成")
        if retarget.get("source_unchanged") is not True or audit.get("source_unchanged") is not True:
            raise ValueError("gate依赖未确认输入不变")
        if (
            len(
                {
                    gate["frame_count"],
                    episode["source"]["frame_count"],
                    retarget["frame_count"],
                    audit["frame_count"],
                }
            )
            != 1
        ):
            raise ValueError("gate各阶段时间轴不一致")
        if (
            Path(gate["retarget_dir"]) != parquet.parent
            or Path(audit["retarget_dir"]) != parquet.parent
        ):
            raise ValueError("gate结果目录不一致")
        if Path(gate["retarget_manifest_path"]).parent != parquet.parent:
            raise ValueError("gate重定向凭证目录不一致")
        source_parquet = Path(retarget["input_parquet"])
        if source_parquet.parent != Path(gate["episode_gate_path"]).parent:
            raise ValueError("gate上游输入来源不一致")
        bound(parquet, retarget["result_sha256"])
        bound(source_parquet, retarget["input_sha256"])
        bound(Path(retarget["input_manifest_path"]), retarget["calibration_sha256"])
        for file, digest in audit["input_sha256"].items():
            bound(Path(file), digest)
    except (KeyError, TypeError) as error:
        raise ValueError("gate缺少完整来源绑定") from error
    return {**gate, "verified_files": verified}
