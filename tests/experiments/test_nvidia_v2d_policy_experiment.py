import hashlib
import json
from pathlib import Path

import pytest

from experiments.nvidia_v2d.policy_budget import TrainingBudget
from experiments.nvidia_v2d.policy_experiment import (
    ExperimentSpec,
    docker_argv,
    require_update_audit,
    training_argv,
    logged_training_argv,
    evaluation_checkpoints,
)


def sample_spec(backend="official", **updates):
    fields = dict(
        backend=backend,
        parquet=Path(f"/input/{backend}.parquet"),
        gate=Path(f"/input/{backend}.json"),
        budget=TrainingBudget(total_steps=1024, world_count=16, wall_limit_s=1200),
        seed=42,
        replay_capacity=4096,
        replay_minimum=128,
        replay_batch=128,
    )
    fields.update(updates)
    return ExperimentSpec(**fields)


def test_backends_use_same_training_hyperparameters_and_complete_curriculum():
    a, b = training_argv(sample_spec()), training_argv(sample_spec("multitip"))
    assert [x for x in a if not x.startswith("task.parquet=")] == [
        x for x in b if not x.startswith("task.parquet=")
    ]
    assert "task.motion_speed=1.0" in a
    assert "task.motion_end_frame=-1" in a
    assert "training.updates_per_collection=8.0" in a
    assert "training.checkpoint.save_replay=true" in a
    assert "scene.support=false" in a
    assert "reset.voc_decay_steps=0" in a
    assert "logging.log_interval=1" in a
    assert any(x.startswith("curriculum.thresholds=") and x.endswith("1024]") for x in a)


def test_training_uses_logged_driver_and_keeps_overrides(tmp_path):
    overrides = training_argv(sample_spec())
    argv = logged_training_argv(tmp_path / "train-metrics.jsonl", overrides)
    assert argv[1:4] == ["-m", "experiments.nvidia_v2d.policy_metric_logging", "scripts/train_flash_sac.py"]
    assert argv[4] == str(tmp_path / "train-metrics.jsonl")
    assert argv[5:] == overrides


def test_evaluation_curve_uses_real_checkpoints_and_resumed_final(tmp_path):
    train = tmp_path / "train"
    train.mkdir()
    for name in ("policy_64.safetensors", "policy_128.safetensors", "state_128.safetensors"):
        (train / name).write_bytes(b"test fixture")
    resume = tmp_path / "resume"
    resume.mkdir()
    final = resume / "policy_128.safetensors"
    final.write_bytes(b"resumed test fixture")
    assert evaluation_checkpoints(train, final) == [train / "policy_64.safetensors", final]
    with pytest.raises(ValueError):
        evaluation_checkpoints(train, resume / "policy_256.safetensors")


def test_container_has_its_own_hard_timeout_and_read_only_inputs(tmp_path):
    spec = sample_spec()
    args = docker_argv(
        spec,
        tmp_path / "spec.json",
        tmp_path / "job",
        tmp_path / "cache",
        Path("/repo"),
        Path("/input"),
    )
    assert args[args.index("timeout") + 1 : args.index("timeout") + 4] == [
        "--signal=TERM",
        "--kill-after=20s",
        "1200s",
    ]
    assert "/input:/input:ro" in args
    assert "/repo:/adapter:ro" in args
    assert "--network" in args and "none" in args
    assert args.count("--gpus") == 1


def test_bad_replay_size_and_unknown_budget_seed_are_rejected():
    with pytest.raises(ValueError):
        sample_spec(replay_capacity=1)
    with pytest.raises(ValueError):
        sample_spec(seed=99)


def test_contact_overflow_is_a_numerical_failure_even_with_zero_exit_code():
    from experiments.nvidia_v2d.policy_experiment import numerical_log_problem

    assert numerical_log_problem("nefc overflow - please increase njmax to 680") is not None
    assert numerical_log_problem("training complete; loss=1.2") is None


def audit_fixture(tmp_path, actor=49, critic=98):
    from experiments.nvidia_v2d import policy_checkpoint_audit

    checkpoint = tmp_path / "state.safetensors"
    checkpoint.write_bytes(b"fixture checkpoint")
    state = {
        ".actor.optimizer_state.count": actor,
        ".actor.schedule_step": 50,
        ".critic.optimizer_state.count": critic,
        ".critic.schedule_step": 100,
        ".global_update_step": 100,
    }
    data = {
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "auditor_sha256": hashlib.sha256(
            Path(policy_checkpoint_audit.__file__).read_bytes()
        ).hexdigest(),
        "source_unchanged": True,
        "template_all_shapes_dtypes_checked": True,
        "scalar_state": state,
        "updates": policy_checkpoint_audit.summarize_gradient_updates(state),
    }
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(data))
    return path, checkpoint


def test_future_training_rejects_all_skipped_updates_and_wrong_checkpoint(tmp_path):
    audit, checkpoint = audit_fixture(tmp_path, actor=0)
    with pytest.raises(ValueError, match="生效"):
        require_update_audit(audit, checkpoint)
    audit, checkpoint = audit_fixture(tmp_path)
    assert require_update_audit(audit, checkpoint)["updates"]["actor"]["applied"] == 49
    checkpoint.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="绑定"):
        require_update_audit(audit, checkpoint)


def test_resume_must_increase_both_applied_optimizer_counts(tmp_path):
    audit, checkpoint = audit_fixture(tmp_path)
    prior = {"actor": {"applied": 49}, "critic": {"applied": 97}}
    with pytest.raises(ValueError, match="恢复"):
        require_update_audit(audit, checkpoint, previous_updates=prior)
    prior["actor"]["applied"] = 48
    require_update_audit(audit, checkpoint, previous_updates=prior)


@pytest.mark.parametrize("target", [0.0, 0.25])
def test_upstream_zero_decay_removes_reset_boost_and_keeps_curriculum_target(target):
    pytest.importorskip("flash_chord")
    import numpy as np
    import warp as wp
    from flash_chord.lifecycle.reset import advance_reference_reset, sample_reference_reset

    device = "cpu"

    def integer(value):
        return wp.array([value], dtype=wp.int32, device=device)

    def floating(value):
        return wp.array([value], dtype=wp.float32, device=device)

    reset_count, frame, age = integer(0), integer(0), integer(0)
    finger, applied = floating(0), floating(9)
    scale = floating(target)
    wp.launch(
        sample_reference_reset,
        dim=1,
        inputs=[integer(1), 249, 0, 1, floating(1), 0.1, 0.7, 42, scale],
        outputs=[reset_count, frame, finger, age, applied],
        device=device,
    )
    assert np.allclose(applied.numpy(), target)
    episode_step = integer(0)
    wp.launch(
        advance_reference_reset,
        dim=1,
        inputs=[integer(0), integer(0), 0, 0, scale],
        outputs=[age, applied, frame, episode_step],
        device=device,
    )
    assert np.allclose(applied.numpy(), target)
    assert frame.numpy().tolist() == [1]


def test_explicit_episode_identity_reaches_gate(monkeypatch):
    from experiments.nvidia_v2d.policy_experiment import require_experiment_gate
    seen = []

    def verify(path, **kwargs):
        seen.append(kwargs['episode_id'])
        if kwargs['episode_id'] != 'track_3-evaluation-episode-000037':
            raise ValueError('gate视频编号不一致')
        return {'training_ready': True}

    monkeypatch.setattr('experiments.nvidia_v2d.policy_gate.verify_gate', verify)
    spec = sample_spec(episode_id='track_3-evaluation-episode-000037')
    assert require_experiment_gate(spec)['training_ready']
    assert seen == ['track_3-evaluation-episode-000037']
    with pytest.raises(ValueError, match='视频编号'):
        require_experiment_gate(sample_spec())


def test_bounded_checkpoint_cadence_does_not_change_training_budget():
    spec=sample_spec(checkpoint_interval=512)
    args=training_argv(spec)
    assert 'training.checkpoint.save_interval_environment_steps=512' in args
    with pytest.raises(ValueError,match='complete collection'):
        sample_spec(checkpoint_interval=511)
    with pytest.raises(ValueError,match='midpoint'):
        sample_spec(checkpoint_interval=256,verify_resume=True)
