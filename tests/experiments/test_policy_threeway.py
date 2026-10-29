from pathlib import Path

import pytest

from experiments.nvidia_v2d.policy_budget import TrainingBudget
from experiments.nvidia_v2d.policy_single_sample import SingleSampleSpec
from experiments.nvidia_v2d.policy_threeway import (
    common_configuration,
    shared_data_digest,
)


def spec(**changes):
    return SingleSampleSpec(
        support_usda=Path("/support.usda"), support_sha256="a" * 64,
        budget=TrainingBudget(total_steps=2097152, world_count=256, seeds=(42,), wall_limit_s=7000),
        **changes,
    )


def test_three_routes_share_config_but_different_physics_does_not():
    official = common_configuration(spec())
    assert common_configuration(spec(backend="dexpilot", parquet=Path("/dex.parquet"))) == official
    assert common_configuration(spec(backend="multitip", parquet=Path("/multi.parquet"))) == official
    assert common_configuration(spec(sim_substeps=10)) != official
    with pytest.raises(ValueError, match="fresh"):
        common_configuration(spec(resume_checkpoint=Path("/state_1.safetensors"), reset_replay=True))


def test_shared_data_rejects_object_contact_time_changes_but_allows_robot_motion(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rows = {"fps": [20.], "object_root_position": [[0., 1., 2.]],
            "mano_right_object_contact_normals": [[float("nan"), 0., 1.]],
            "robot_right_finger_joints": [[1., 2.]]}
    def digest(name, changes):
        path = tmp_path / name
        pq.write_table(pa.table({**rows, **changes}), path)
        return shared_data_digest(path)
    original = digest("raw.parquet", {})
    assert digest("robot.parquet", {"robot_right_finger_joints": [[5., 6.]]}) == original
    for key, value in (("fps", [10.]), ("object_root_position", [[0., 0., 2.]]),
                       ("mano_right_object_contact_normals", [[0., 0., -1.]])):
        assert digest(key + ".parquet", {key: value}) != original
