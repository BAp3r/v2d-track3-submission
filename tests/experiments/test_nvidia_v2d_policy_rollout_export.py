import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.nvidia_v2d.policy_rollout_export import export_rollout, verified_assistance
from experiments.nvidia_v2d.policy_view_data import Scene


def fixture_files(tmp_path):
    scene = Scene(
        kind="kinematic_reference",
        timestamps_s=[0, 0.05],
        hands=[],
        objects=[
            {
                "name": "box",
                "vertices": [[0, 0, 0]],
                "faces": [],
                "poses": [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]] * 2,
            }
        ],
    )
    reference = tmp_path / "reference.json"
    reference.write_text(scene.model_dump_json())
    rows = []
    for step in range(2):
        for environment in range(2):
            row = {
                "environment_id": environment,
                "step": step,
                "reference_frame": step,
                "time_s": step / 20,
                "object_body_name": "box",
                "object_id": 0,
                "object_body_id": 0,
                "environment_valid": step == 0,
            }
            for source in ["achieved", "reference"]:
                row.update({f"{source}_position_{a}": 0.0 for a in "xyz"})
                row.update({f"{source}_quaternion_{a}": 0.0 for a in "xyz"})
                row[f"{source}_quaternion_w"] = 1.0
            row["achieved_position_x"] = float(step + environment)
            rows.append(row)
    meta = {
        "schema": "flash_chord_object_trajectories_v1",
        "position_unit": "meter",
        "quaternion_order": "xyzw",
        "start_frame": "0",
        "step_count": "2",
        "control_fps": "20",
        "world_count": "2",
        "object_body_count": "1",
        "pose_width": "7",
        "row_order": "step,environment_id,object_body_id",
    }
    table = pa.Table.from_pylist(rows).replace_schema_metadata(
        {f"flash_chord.{k}": v for k, v in meta.items()}
    )
    trajectory = tmp_path / "trajectory.parquet"
    pq.write_table(table, trajectory)
    return trajectory, reference, table


def test_exports_actual_not_reference_and_preserves_invalid_frames(tmp_path):
    trajectory, reference, _ = fixture_files(tmp_path)
    manifest = export_rollout(trajectory, reference, tmp_path / "new")
    result = Scene.model_validate_json((tmp_path / "new/scene.json").read_text())
    assert result.kind == "policy_rollout" and not result.hands
    assert result.assistance == "unknown"
    assert result.objects[0].poses[1][3] == 1.0
    assert result.valid_frames == [True, False]
    assert manifest["environment_id"] == 0
    assert manifest["invalid_frame_count"] == 1
    with pytest.raises(FileExistsError):
        export_rollout(trajectory, reference, tmp_path / "new")


@pytest.mark.parametrize("problem", ["missing", "time", "coordinate", "nonfinite", "unit"])
def test_rejects_incomplete_or_misframed_trajectory(tmp_path, problem):
    trajectory, reference, table = fixture_files(tmp_path)
    rows = table.to_pylist()
    if problem == "missing":
        rows.pop(2)
    elif problem == "time":
        rows[2]["time_s"] = 0.1
    elif problem == "coordinate":
        rows[0]["reference_position_x"] = 1.0
    elif problem == "nonfinite":
        rows[0]["achieved_position_x"] = float("nan")
    metadata = dict(table.schema.metadata)
    if problem == "unit":
        metadata[b"flash_chord.position_unit"] = b"centimeter"
    pq.write_table(pa.Table.from_pylist(rows).replace_schema_metadata(metadata), trajectory)
    with pytest.raises(ValueError):
        export_rollout(trajectory, reference, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_reference_scene_cannot_claim_to_be_execution(tmp_path):
    trajectory, reference, _ = fixture_files(tmp_path)
    raw = json.loads(reference.read_text())
    raw["kind"] = "policy_rollout"
    reference.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="运动学"):
        export_rollout(trajectory, reference, tmp_path / "invalid")


def test_rejects_corrupt_unselected_world(tmp_path):
    trajectory, reference, table = fixture_files(tmp_path)
    rows = table.to_pylist()
    rows[3] = dict(rows[1])
    pq.write_table(
        pa.Table.from_pylist(rows).replace_schema_metadata(table.schema.metadata), trajectory
    )
    with pytest.raises(ValueError):
        export_rollout(trajectory, reference, tmp_path / "invalid", environment=0)
    assert not (tmp_path / "invalid").exists()


def test_unassisted_label_requires_bound_actual_execution_receipt(tmp_path):
    assert verified_assistance(None, {}, "a" * 64, 2) == "unknown"
    evidence = {
        "assistance": "none",
        "voc_scale_max": 0,
        "applied_control_scale_max": 0,
        "action_step_count": 2,
        "trajectory_sha256": "a" * 64,
        "start_frame": 0,
        "motion_end_frame": -1,
        "observed_control_scales": [
            {"voc_scale_max_abs": 0, "applied_object_control_scale_max_abs": 0}
        ]
        * 2,
    }
    metadata = {}
    for key in ["checkpoint_sha256", "eval_code_sha256", "eval_script_sha256"]:
        evidence[key] = "b" * 64
        metadata["flash_chord." + key] = "b" * 64
    receipt = tmp_path / "experiment.json"

    def save():
        receipt.write_text(json.dumps({"status": "succeeded", "evaluation_evidence": evidence}))

    save()
    assert verified_assistance(receipt, metadata, "a" * 64, 2) == "none"
    with pytest.raises(ValueError):
        verified_assistance(receipt, metadata, "c" * 64, 2)
    evidence["applied_control_scale_max"] = 0.1
    save()
    with pytest.raises(ValueError):
        verified_assistance(receipt, metadata, "a" * 64, 2)
