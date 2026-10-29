import argparse
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from experiments.nvidia_v2d.policy_single_sample import (
    FROZEN_IMAGE,
    LEGACY_ROOT,
    SOURCE_FILES,
    SingleSampleSpec,
    docker_command,
    explicit_support,
    physical_training_overrides,
    prepare,
    require_physical_input_gate,
    require_sim_contract,
    sha256,
)


def make_spec(tmp_path, **changes):
    support = tmp_path / "table.usda"
    support.write_text('#usda 1.0\ndef Cube "table" { double size = 1 }\n')
    return SingleSampleSpec(support_usda=support, support_sha256=sha256(support), **changes)


def test_finite_physical_scene_speed_and_logging_are_frozen(tmp_path):
    spec = make_spec(tmp_path)
    args = physical_training_overrides(spec, Path("/run"))
    assert "scene.support=true" in args
    assert "collision.ground=false" in args
    assert "collision.robot_support_collision=true" in args
    assert "reset.voc_decay_steps=0" in args
    assert "logging.log_interval=1" in args
    assert "training.checkpoint.save_replay=true" in args
    assert [arg for arg in args if arg.startswith("task.motion_speed=")] == ["task.motion_speed=0.5"]
    assert "task.motion_start_frame=0" in args and "task.motion_end_frame=-1" in args
    assert spec.budget.world_count == 256


def test_explicit_resume_requires_optimizer_and_replay_state(tmp_path):
    with pytest.raises(ValueError, match="requires both"):
        make_spec(tmp_path, resume_checkpoint=Path("/archive/state_256.safetensors"))
    with pytest.raises(ValueError, match="state_"):
        make_spec(tmp_path, resume_checkpoint=Path("/archive/policy_256.safetensors"),
                  resume_replay=Path("/archive/replay_256.safetensors"))
    spec = make_spec(tmp_path, resume_checkpoint=Path("/archive/state_256.safetensors"),
                     resume_replay=Path("/archive/replay_256.safetensors"))
    args = physical_training_overrides(spec, Path("/run"))
    assert "training.resume=true" in args
    assert "training.checkpoint_path='/archive/state_256.safetensors'" in args
    assert "training.checkpoint.load_replay_path='/archive/replay_256.safetensors'" in args


def test_integration_rate_is_frozen_for_training_and_evaluation(tmp_path):
    spec = make_spec(tmp_path, sim_substeps=20)
    assert "sim.substeps=20" in physical_training_overrides(spec, Path("/run"))
    cfg = {"sim": {"fps": 20.0, "substeps": 20}, "env": {"sim": {"fps": 20.0, "substeps": 20}}}
    require_sim_contract(spec, cfg)
    cfg["env"]["sim"]["substeps"] = 5
    with pytest.raises(ValueError, match="timestep"):
        require_sim_contract(spec, cfg)
    for invalid in (True, 5.5, 0, 41):
        with pytest.raises(ValueError):
            make_spec(tmp_path, sim_substeps=invalid)


def test_release_ablation_keeps_original_task_and_requires_real_penalty(tmp_path):
    with pytest.raises(ValueError, match="negative contact penalty"):
        make_spec(tmp_path, release_start_frame=407, unintended_contact_weight=0)
    with pytest.raises(ValueError):
        make_spec(tmp_path, release_start_frame=407.5)
    spec = make_spec(tmp_path, release_start_frame=407, unintended_contact_weight=-10)
    args = physical_training_overrides(spec, Path("/run"))
    assert "+objective.release_start_frame=407" in args
    assert "objective._target_=experiments.nvidia_v2d.release_contact_objective.make_release_contact_objective" in args
    assert "task.motion_start_frame=0" in args and "task.motion_end_frame=-1" in args
    assert not any(arg.startswith("termination.") for arg in args)


def test_wrist_bound_changes_action_space_without_relaxing_force_or_criteria(tmp_path):
    spec = make_spec(tmp_path, wrist_pos_clip=0.075)
    args = physical_training_overrides(spec, Path("/run"))
    assert "action.wrist_pos_clip=0.075" in args
    assert not any(arg.startswith(("termination.", "action.wrist_tau_max=")) for arg in args)
    for bad in [0, 0.3, float("nan")]:
        with pytest.raises(ValueError):
            make_spec(tmp_path, wrist_pos_clip=bad)


def test_changed_reward_can_explicitly_resume_with_fresh_replay(tmp_path):
    state = Path("/archive/state_2621440.safetensors")
    spec = make_spec(tmp_path, resume_checkpoint=state, reset_replay=True,
                     replay_capacity=262144, replay_minimum=131072)
    args = physical_training_overrides(spec, Path("/run"))
    assert "training.resume=true" in args
    assert "training.checkpoint.load_replay_path=null" in args
    assert "training.replay.minimum_size=131072" in args
    with pytest.raises(ValueError, match="no old replay"):
        make_spec(tmp_path, resume_checkpoint=state, reset_replay=True,
                  resume_replay=Path("/archive/replay_2621440.safetensors"))
    with pytest.raises(ValueError, match="state checkpoint"):
        make_spec(tmp_path, reset_replay=True)


def install_fake_setup(monkeypatch):
    package = ModuleType("flash_chord")
    scene = ModuleType("flash_chord.scene")
    setup = SimpleNamespace(support_usda_for_reference=lambda reference: Path("/original.usda"))
    scene.setup = setup
    package.scene = scene
    monkeypatch.setitem(sys.modules, "flash_chord", package)
    monkeypatch.setitem(sys.modules, "flash_chord.scene", scene)
    return setup


def test_support_is_sample_bound_hashed_and_resolver_restored(tmp_path, monkeypatch):
    spec = make_spec(tmp_path)
    setup = install_fake_setup(monkeypatch)
    original = setup.support_usda_for_reference
    reference = SimpleNamespace(metadata=SimpleNamespace(source_path=str(spec.parquet)))
    with explicit_support(spec) as receipt:
        assert setup.support_usda_for_reference(reference) == spec.support_usda
        assert receipt["resolver_calls"] == 1
        with pytest.raises(ValueError, match="frozen sample"):
            setup.support_usda_for_reference(SimpleNamespace(metadata=SimpleNamespace(source_path="/wrong")))
    assert setup.support_usda_for_reference is original
    assert receipt["ground"] is False and receipt["robot_support_collision"] is True
    spec.support_usda.write_text("changed")
    with pytest.raises(ValueError, match="hash"), explicit_support(spec):
        pass
    assert setup.support_usda_for_reference is original


def test_unused_support_cannot_silently_pass(tmp_path, monkeypatch):
    spec = make_spec(tmp_path)
    setup = install_fake_setup(monkeypatch)
    original = setup.support_usda_for_reference
    with pytest.raises(ValueError, match="never requested"), explicit_support(spec):
        pass
    assert setup.support_usda_for_reference is original


def test_frozen_command_uses_local_migrated_input_readonly_alias(tmp_path):
    spec = make_spec(tmp_path)
    args = docker_command(spec, tmp_path / "spec.json", tmp_path / "job",
                          Path("/repo"), Path("/repo/local-inputs"), Path("/repo/cache"))
    assert f"/repo/local-inputs:{LEGACY_ROOT}:ro" in args
    assert "/repo:/adapter:ro" in args and "/repo:/repo:ro" in args
    assert f"{spec.support_usda}:{spec.support_usda}:ro" in args
    assert FROZEN_IMAGE in args
    assert args[-5:] == ["run", "--spec", "/single-sample-spec.json", "--output", "/run"]


def test_prepare_freezes_reviewable_command_without_execution(tmp_path):
    repository = tmp_path / "repo"
    code = repository / "experiments/nvidia_v2d"
    code.mkdir(parents=True)
    for name in SOURCE_FILES:
        (code / name).write_text("# frozen source fixture\n")
    support = repository / "table.usda"
    support.write_text('#usda 1.0\ndef Cube "table" { double size = 1 }\n')
    output = repository / "run"
    args = argparse.Namespace(repository=repository, output=output, support_usda=support,
        spec=None, total_steps=131072, world_count=256, motion_speed=0.5,
        legacy_inputs=repository / "inputs", cache=repository / "cache",
        swanlog_root=repository / "shared-swanlog", tensorboard_root=repository / "shared-tensorboard",
        swanlab_url="http://127.0.0.1:5092", tensorboard_url="http://127.0.0.1:6006")
    manifest = prepare(args)
    spec = json.loads((output / "spec.json").read_text())
    assert spec["budget"]["total_steps"] == 131072
    assert spec["budget"]["world_count"] == 256
    assert spec["motion_speed"] == 0.5
    assert manifest["status"] == "prepared"
    assert manifest["spec_sha256"] == sha256(output / "spec.json")
    script = (output / "run.sh").read_text()
    assert "policy_logged_launch.py" in script and "--swanlab-mode local" in script
    assert "--observer-python" in script and "--observer-module-root" in script
    assert manifest["argv"][:2] == ["docker", "run"]
    assert manifest["observability"]["swanlab"] and manifest["observability"]["tensorboard"]
    assert manifest["observability"]["swanlog_root"] == str(repository / "shared-swanlog")
    assert manifest["observability"]["tensorboard_url"] == "http://127.0.0.1:6006"
    assert set(manifest["host_source_sha256"]) == {"policy_logged_launch.py", "policy_observability.py"}
    with pytest.raises(FileExistsError):
        prepare(args)


def test_repaired_reference_and_audit_inputs_are_mounted_read_only(tmp_path):
    spec = make_spec(tmp_path, parquet=Path("/candidate/data.parquet"),
        gate=Path("/candidate/gate.json"), additional_readonly_inputs=(Path("/evidence"),))
    args = docker_command(spec, tmp_path / "spec.json", tmp_path / "job",
                          Path("/code"), Path("/legacy"), Path("/cache"))
    assert args.count("/candidate:/candidate:ro") == 1
    assert "/evidence:/evidence:ro" in args
    with pytest.raises(ValueError, match="must not contain"):
        docker_command(spec, tmp_path / "spec.json", Path("/candidate/job"),
                       Path("/code"), Path("/legacy"), Path("/cache"))


def test_semantic_contact_requires_bound_reference_and_preserves_release(tmp_path):
    with pytest.raises(ValueError, match="hashed map/reference"):
        make_spec(tmp_path, semantic_map_path=Path("/roles.npz"))
    spec = make_spec(tmp_path, release_start_frame=407, semantic_map_path=Path("/roles.npz"),
        semantic_map_sha256="a" * 64, semantic_reference_sha256="b" * 64)
    args = physical_training_overrides(spec, Path("/run"))
    assert "objective._target_=experiments.nvidia_v2d.semantic_contact_objective.make_semantic_contact_objective" in args
    assert "+objective.release_start_frame=407" in args
    assert "+objective.semantic_motion_speed=0.5" in args
    assert "+objective.semantic_sigma_m=0.02" in args
    assert not any(value.startswith(("termination.", "action.wrist_tau_max=")) for value in args)


def test_semantic_reference_and_role_map_must_share_an_acceptance_receipt(tmp_path, monkeypatch):
    parquet = tmp_path / "candidate.parquet"; parquet.write_bytes(b"candidate")
    spec = make_spec(tmp_path, release_start_frame=407, parquet=parquet,
        semantic_map_path=Path("/roles.npz"), semantic_map_sha256="a" * 64,
        semantic_reference_sha256=sha256(parquet))
    gate = {"semantic_map_path": "/roles.npz", "semantic_map_sha256": "a" * 64}
    monkeypatch.setattr("experiments.nvidia_v2d.policy_single_sample.require_experiment_gate", lambda _: gate)
    assert require_physical_input_gate(spec) is gate
    gate["semantic_map_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="accepted together"):
        require_physical_input_gate(spec)
