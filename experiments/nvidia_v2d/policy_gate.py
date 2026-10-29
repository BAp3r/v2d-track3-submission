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
