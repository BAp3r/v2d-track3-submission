import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from experiments.nvidia_v2d.policy_demo_capture import capture


@pytest.mark.parametrize("nonfinite", [False, True])
def test_capture_uses_post_action_actual_state_and_restores_harness(tmp_path, monkeypatch, nonfinite):
    from experiments.nvidia_v2d import policy_experiment_evaluation

    checkpoint = tmp_path / "input/model.bin"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    values = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    def array(value):
        return SimpleNamespace(numpy=lambda: value)
    model = SimpleNamespace(
        body_count=2,
        body_label=["wrist", "finger"],
        joint_parent=array(np.array([-1, 0])),
        joint_child=array(np.array([0, 1])),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(model=model),
        world_count=1,
        state_0=SimpleNamespace(body_q=array(values)),
    )

    def step(action):
        values[:, 0] += 1
        if nonfinite:
            values[0, 0] = np.nan
        return action

    def rollout(env, jax_env, policy_action):
        for _ in range(2):
            jax_env.step(policy_action())

    harness = SimpleNamespace(record_object_pose_rollout=rollout)
    package = ModuleType("flash_chord.evaluation")
    package.harness = harness
    monkeypatch.setitem(sys.modules, "flash_chord", ModuleType("flash_chord"))
    monkeypatch.setitem(sys.modules, "flash_chord.evaluation", package)

    def evaluate(checkpoint, worlds, output):
        harness.record_object_pose_rollout(env, SimpleNamespace(step=step), lambda: 1)
        return {"step_count": 2}

    monkeypatch.setattr(policy_experiment_evaluation, "evaluate_with_evidence", evaluate)
    output = tmp_path / "new_output"
    if nonfinite:
        with pytest.raises(ValueError, match="非有限"):
            capture(checkpoint, output, 1)
        assert not (output / "actual_bodies.npz").exists()
    else:
        capture(checkpoint, output, 1)
        with np.load(output / "actual_bodies.npz") as data:
            assert data["body_pose_world0"][:, 0, 0].tolist() == [1, 2]
        assert json.loads((output / "capture-manifest.json").read_text())["edges"] == [[0, 1]]
    assert harness.record_object_pose_rollout is rollout
    assert checkpoint.read_bytes() == b"checkpoint"
