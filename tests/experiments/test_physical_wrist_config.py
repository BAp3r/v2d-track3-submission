from types import SimpleNamespace

import pytest

from experiments.nvidia_v2d.physical_wrist_config import configure_wrist
from experiments.nvidia_v2d.policy_single_sample import (
    SingleSampleSpec,
    physical_training_overrides,
)


def test_gain_ablation_preserves_force_and_other_joints():
    original = {r"wrist_[xyz]$": {"kp": 25.0, "kd": 6.0, "effort_limit": 45.0},
                "wrist_rotvec$": {"kp": 25.0, "kd": 6.0, "effort_limit": 45.0}}
    config = configure_wrist(SimpleNamespace(joint_overrides=original))
    assert original[r"wrist_[xyz]$"]["kp"] == 25.0
    assert config.joint_overrides[r"wrist_[xyz]$"] == {"kp": 100.0, "kd": 12.0, "effort_limit": 45.0}
    assert config.joint_overrides["wrist_rotvec$"] == original["wrist_rotvec$"]
    with pytest.raises(ValueError, match="limited"):
        configure_wrist(config, kp=6000)


def test_penalty_ablation_reaches_every_curriculum_stage(tmp_path):
    spec = SingleSampleSpec(support_usda=tmp_path / "support.usda", support_sha256="a" * 64,
                            wrist_kp=100, wrist_kd=12, unintended_contact_weight=0)
    args = physical_training_overrides(spec, tmp_path)
    stage_overrides = [arg for arg in args if arg.startswith("curriculum.stages.")]
    assert len(stage_overrides) == 11
    assert all(arg.endswith(".objective_weights.unintended_contact=0.0") for arg in stage_overrides)
    assert "+embodiment.config.kp=100.0" in args
    assert "collision.robot_support_collision=true" in args
    assert not any("termination." in arg or "effort_limit" in arg for arg in args)


def test_first_frame_reset_does_not_change_evaluation_thresholds_or_assistance(tmp_path):
    spec = SingleSampleSpec(support_usda=tmp_path / "support.usda", support_sha256="a" * 64,
                            always_reset_to_first_frame=True)
    args = physical_training_overrides(spec, tmp_path)
    assert "reset.always_reset_to_first_frame=true" in args
    assert "reset.voc_decay_steps=0" in args
    assert "task.motion_start_frame=0" in args and "task.motion_end_frame=-1" in args
    assert not any("termination." in arg or "reset_finger_openness" in arg for arg in args)
