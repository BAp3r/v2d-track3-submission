"""只读 checkpoint 重新评估，记录逐步真实连杆状态；不是参考手回放。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def capture(checkpoint: Path, output: Path, worlds: int = 16) -> dict[str, Any]:
    from flash_chord.evaluation import harness

    from experiments.nvidia_v2d.policy_experiment_evaluation import evaluate_with_evidence
    from experiments.nvidia_v2d.sharpa_episode import require_new_output

    require_new_output(output, [checkpoint.parent])
    output.mkdir(parents=True, exist_ok=False)
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    original = harness.record_object_pose_rollout
    states: list[np.ndarray] = []
    model_data: dict[str, Any] = {}

    def recording(env, jax_env, policy_action, *args, **kwargs):
        model = env.scene.model
        count = model.body_count // env.world_count
        parent, child = model.joint_parent.numpy(), model.joint_child.numpy()
        keep = (parent >= 0) & (parent < count) & (child >= 0) & (child < count)
        model_data.update(
            body_labels=list(model.body_label[:count]),
            edges=np.column_stack((parent[keep], child[keep])).tolist(),
            body_count=count,
            world_count=env.world_count,
        )

        class Recorder:
            def __getattr__(self, name):
                return getattr(jax_env, name)

            def step(self, action):
                result = jax_env.step(action)
                states.append(env.state_0.body_q.numpy().reshape(env.world_count, count, 7)[0].copy())
                return result

        return original(env, Recorder(), policy_action, *args, **kwargs)

    harness.record_object_pose_rollout = recording
    try:
        evidence = evaluate_with_evidence(checkpoint, worlds, output)
    finally:
        harness.record_object_pose_rollout = original
    values = np.asarray(states)
    if len(values) != evidence["step_count"] or not np.isfinite(values).all():
        raise ValueError("真实连杆状态记录不完整或含非有限值")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != sha:
        raise ValueError("评估过程中 checkpoint 改变")
    np.savez_compressed(output / "actual_bodies.npz", body_pose_world0=values)
    result = {
        "schema": "nvidia_v2d.actual_body_demo.v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha,
        "kind": "actual_policy_execution",
        "assistance": "none",
        "state_timing": "after each policy action; same as official object rollout",
        "quaternion_order": "xyzw",
        "unit": "meter",
        "environment_id": 0,
        "frames": len(values),
        "fps": 20,
        **model_data,
        "source_unchanged": True,
        "outputs_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output.iterdir()
            if p.is_file()
        },
    }
    (output / "capture-manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = capture(args.checkpoint, args.output)
    print(json.dumps({k: result[k] for k in ("frames", "body_count", "source_unchanged")}))


if __name__ == "__main__":
    main()
