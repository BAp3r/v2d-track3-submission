import pytest

from experiments.nvidia_v2d.policy_trajectory import validate_trajectory


def trajectory_fixture():
    data = {
        "step": [0, 0, 1, 1],
        "reference_frame": [0, 0, 1, 1],
        "environment_id": [0, 1, 0, 1],
        "object_id": [0] * 4,
        "object_body_id": [0] * 4,
        "object_body_name": ["cube"] * 4,
        "time_s": [0.0, 0.0, 0.05, 0.05],
        "environment_valid": [True, True, False, True],
    }
    for prefix in ("achieved", "reference"):
        for axis in "xyz":
            data[f"{prefix}_position_{axis}"] = [0.1] * 4
            data[f"{prefix}_quaternion_{axis}"] = [0.0] * 4
        data[f"{prefix}_quaternion_w"] = [1.0] * 4
    metadata = {
        "flash_chord." + key: value
        for key, value in {
            "schema": "flash_chord_object_trajectories_v1",
            "position_unit": "meter",
            "quaternion_order": "xyzw",
            "row_order": "step,environment_id,object_body_id",
            "start_frame": "0",
            "step_count": "2",
            "world_count": "2",
            "pose_width": "7",
            "control_fps": "20.0",
            "object_body_count": "1",
        }.items()
    }
    return data, metadata


def write(path, data, metadata):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pq.write_table(pa.table(data).replace_schema_metadata(metadata), path)


def test_actual_trajectory_requires_full_finite_world_sequences(tmp_path):
    data, metadata = trajectory_fixture()
    path = tmp_path / "trajectory.parquet"
    write(path, data, metadata)
    summary = validate_trajectory(path, 2, 2)
    assert summary["rows"] == 4
    assert summary["environment_valid_fraction"] == 0.75
    assert summary["invalid_worlds"] == [0]
    assert summary["invalid_row_count"] == 1
    with pytest.raises(ValueError, match="metadata"):
        validate_trajectory(path, 3, 2)
    data["achieved_position_x"][0] = float("nan")
    write(path, data, metadata)
    with pytest.raises(ValueError, match="非有限"):
        validate_trajectory(path, 2, 2)


def test_balanced_missing_and_duplicate_frame_in_world_one_is_rejected(tmp_path):
    data, metadata = trajectory_fixture()
    # Global row count, unique step count and world0 are all still correct.
    data["step"][3] = 0
    data["reference_frame"][3] = 0
    order = sorted(range(4), key=lambda i: (data["step"][i], data["environment_id"][i]))
    data = {key: [values[i] for i in order] for key, values in data.items()}
    path = tmp_path / "bad.parquet"
    write(path, data, metadata)
    with pytest.raises(ValueError, match="缺帧、重复帧"):
        validate_trajectory(path, 2, 2)


@pytest.mark.parametrize(
    "column,value",
    [("time_s", 0.1), ("reference_frame", 0), ("object_id", 3), ("reference_quaternion_w", 0.0)],
)
def test_wrong_time_reference_identity_or_quaternion_is_rejected(tmp_path, column, value):
    data, metadata = trajectory_fixture()
    data[column][3] = value
    path = tmp_path / "bad.parquet"
    write(path, data, metadata)
    with pytest.raises(ValueError):
        validate_trajectory(path, 2, 2)


def test_wrong_fps_or_missing_metadata_is_rejected(tmp_path):
    data, metadata = trajectory_fixture()
    path = tmp_path / "bad.parquet"
    metadata["flash_chord.control_fps"] = "30.0"
    write(path, data, metadata)
    with pytest.raises(ValueError):
        validate_trajectory(path, 2, 2)
    write(path, data, {})
    with pytest.raises(ValueError):
        validate_trajectory(path, 2, 2)
