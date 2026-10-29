"""Freeze a matched three-retargeter comparison without imposing contact perfection.

Historical runnable policies remain separate. This contract compares fresh
learners with identical data, physical settings, update budgets and evaluation
budgets. Geometry variants belong to a separately labelled experiment.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from experiments.nvidia_v2d.policy_single_sample import (
    SingleSampleSpec,
    physical_training_overrides,
    require_physical_input_gate,
    sha256,
)

BACKENDS = ("official", "dexpilot", "multitip")
ROUTE_FIELDS = frozenset({"backend", "parquet", "gate", "additional_readonly_inputs"})


def common_configuration(spec: SingleSampleSpec) -> dict[str, Any]:
    """Only input identity may differ; quality objectives are not selection gates."""
    if (spec.resume_checkpoint is not None or spec.resume_replay is not None
            or spec.reset_replay or spec.verify_resume):
        raise ValueError("main comparison requires fresh learners and fresh replay")
    if (spec.semantic_map_path is not None or spec.semantic_alignment_weight
            or spec.semantic_wrist_command_weight):
        raise ValueError("contact geometry objectives belong to a separate ablation")
    return spec.model_dump(mode="json", exclude=ROUTE_FIELDS)


def shared_data_digest(parquet: Path) -> dict[str, Any]:
    """Hash all non-retargeted fields, including MANO/contact/object trajectories.

Arrow serialization metadata is not part of the equality test. NaN source
annotations are retained, never removed to make two inputs look equivalent.
"""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet)
    columns = [name for name in table.column_names if not name.startswith(
        ("robot_left_", "robot_right_", "left_robot_", "right_robot_"))]
    content = json.dumps(table.select(columns).to_pylist(), sort_keys=True,
                         separators=(",", ":"), allow_nan=True).encode()
    return {"columns": columns, "rows": table.num_rows,
            "sha256": hashlib.sha256(content).hexdigest()}


def freeze_comparison(spec_paths: dict[str, Path], output: Path,
                      *, retained_baseline: Path, geometry_factor: str = "raw") -> dict[str, Any]:
    if set(spec_paths) != set(BACKENDS):
        raise ValueError("exactly official, dexpilot and multitip are required")
    if geometry_factor != "raw":
        raise ValueError("mainline uses raw geometry; register geometry ablations separately")
    if output.exists():
        raise FileExistsError(output)
    common = shared = None
    runs = []
    for backend in BACKENDS:
        path = spec_paths[backend].resolve()
        spec = SingleSampleSpec.model_validate_json(path.read_text())
        if spec.backend != backend:
            raise ValueError("backend does not match the registered spec")
        configuration = common_configuration(spec)
        if common is not None and common != configuration:
            raise ValueError("three-way training/physics/evaluation budgets differ")
        common = configuration
        gate = require_physical_input_gate(spec)
        data = shared_data_digest(spec.parquet)
        if shared is not None and shared != data:
            raise ValueError("non-robot source, geometry or time axis differs")
        shared = data
        runs.append({"backend": backend, "spec": str(path), "spec_sha256": sha256(path),
                     "parquet_sha256": sha256(spec.parquet), "gate_sha256": sha256(spec.gate),
                     "verified_input_sha256": gate["verified_files"],
                     "training_overrides": physical_training_overrides(spec, Path("/run"))})
    baseline = json.loads(retained_baseline.read_text())
    for file, digest in {**baseline["files_sha256"], **baseline["code_sha256"]}.items():
        if sha256(Path(file)) != digest:
            raise ValueError("retained runnable baseline changed")
    result = {
        "schema": "nvidia_v2d.threeway_rollout_protocol.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "geometry_factor": geometry_factor, "common_configuration": common,
        "shared_nonrobot_data": shared, "runs": runs,
        "retained_baseline": str(retained_baseline.resolve()),
        "retained_baseline_sha256": sha256(retained_baseline),
        "historical_baseline_in_matched_ranking": False,
        "training_assistance": "identical scaled upstream VOC curriculum; reset VOC boost disabled",
        "evaluation_assistance": "none; verify every step of actual rollout",
        "primary_measure": "whole-sequence actual object position <=0.1m and angle <=0.5rad",
        "additional_measures": ["original CHORD", "lift", "release", "settling", "natural grasp"],
        "checkpoint_selection": "all four fixed checkpoints reported; final endpoint is primary",
        "geometry_ablation": "fixed official IK, separate raw/normal/support/normal+support factors",
        "scope": "single reconstructed sample and paired seed; not a generalization claim",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return result
