"""CPU只读审计FlashSAC Adam生效计数，避免把尝试更新次数当成参数更新。"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def summarize_gradient_updates(state: dict[str, float | int]) -> dict[str, Any]:
    """Adam count只在有限梯度实际应用时增加；schedule/global计数含跳过轮次。"""
    result: dict[str, Any] = {}
    for network in ("actor", "critic"):
        applied = state[f".{network}.optimizer_state.count"]
        attempted = state[f".{network}.schedule_step"]
        if (
            any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in (applied, attempted))
            or applied > attempted
        ):
            raise ValueError("Adam生效/尝试计数不可能")
        result[network] = {
            "attempted": attempted,
            "applied": applied,
            "skipped": attempted - applied,
            "skip_fraction": (attempted - applied) / attempted if attempted else None,
        }
    if state[".global_update_step"] != result["critic"]["attempted"]:
        raise ValueError("global update计数与critic尝试轮次不一致")
    passed = result["actor"]["applied"] > 0 and result["critic"]["applied"] > 0
    result.update(
        effective_update_gate_passed=passed,
        failure_reason=None if passed else "actor或critic没有任何实际生效的Adam更新",
    )
    return result


def audit_checkpoint(path: Path) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flash_chord.configuration import instantiate_typed
    from flash_chord.training.checkpoint import read_checkpoint_metadata
    from flash_chord.training.flash_sac.checkpoint import TrainingCheckpoint
    from flash_chord.training.flash_sac.config import TrainingConfig
    from flash_chord.training.flash_sac.exploration import create_exploration
    from flash_chord.training.flash_sac.learner import (
        OptimizedNetworkState,
        TargetCriticState,
        TemperatureState,
        _create_adam,
    )
    from flash_chord.training.flash_sac.network import Actor, Critic, project_unit_parameters
    from flash_chord.training.flash_sac.reward_normalization import create_reward_normalizer
    from flash_chord.training.flash_sac.update import create_loss_scale
    from omegaconf import OmegaConf
    from safetensors import safe_open

    if jax.default_backend() != "cpu":
        raise ValueError("计数审计必须在CPU运行，不占用实验GPU")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = read_checkpoint_metadata(path)
    if metadata.get("checkpoint_kind") != "training_state":
        raise ValueError("需要包含优化器状态的training_state checkpoint")
    cfg = instantiate_typed(
        OmegaConf.create(json.loads(metadata["resolved_config_json"]))["training"], TrainingConfig
    )
    policy = json.loads(metadata["policy_schema_json"])
    critic_schema = json.loads(metadata["critic_schema_json"])
    observation_dim, action_dim = policy["observation"]["dimension"], policy["action"]["dimension"]
    context_dim = critic_schema["context"]["dimension"]
    dtype = jnp.dtype(cfg.compute_dtype) if cfg.mixed_precision else jnp.dtype(jnp.float32)
    actor = Actor(action_dim=action_dim, config=cfg.network, dtype=dtype)
    low, high = cfg.reward_normalization.value_support
    critic = Critic(value_min=low, value_max=high, config=cfg.network, dtype=dtype)
    optimizer = _create_adam(cfg)

    def template() -> Any:
        key = jax.random.PRNGKey(cfg.seed)
        av = actor.init(key, jnp.zeros((1, observation_dim), jnp.float32), training=False)
        cv = critic.init(
            key,
            jnp.zeros((1, observation_dim + context_dim), jnp.float32),
            jnp.zeros((1, action_dim), jnp.float32),
            training=False,
        )
        ap = project_unit_parameters(av["params"], cfg.network.projection_epsilon)
        cp = project_unit_parameters(cv["params"], cfg.network.projection_epsilon)
        zero = jnp.asarray(0, dtype=jnp.int32)
        temperature = jnp.asarray(0.0, dtype=jnp.float32)
        return TrainingCheckpoint(
            actor=OptimizedNetworkState(ap, av["batch_stats"], optimizer.init(ap), zero),
            critic=OptimizedNetworkState(cp, cv["batch_stats"], optimizer.init(cp), zero),
            target_critic=TargetCriticState(cp, cv["batch_stats"]),
            temperature=TemperatureState(temperature, optimizer.init(temperature), zero),
            reward_normalizer=create_reward_normalizer(cfg.world_count),
            exploration=create_exploration(cfg.world_count, action_dim),
            loss_scale=create_loss_scale(
                cfg.optimization, enabled=cfg.mixed_precision and dtype == jnp.dtype(jnp.float16)
            ),
            key=key,
            update_credit=jnp.asarray(0.0, dtype=jnp.float32),
            environment_steps=zero,
            global_update_step=zero,
        )

    shape = jax.eval_shape(template)
    paths, _ = jax.tree_util.tree_flatten_with_path(shape)
    scalars: dict[str, float | int] = {}
    with safe_open(str(path), framework="numpy") as checkpoint:
        manifest = json.loads(checkpoint.metadata()["manifest"])
        if len(paths) != len(manifest):
            raise ValueError("计数模板与检查点叶节点数量不符")
        for (keys, abstract), entry in zip(paths, manifest, strict=True):
            if tuple(entry["shape"]) != abstract.shape or np.dtype(entry["dtype"]) != np.dtype(
                abstract.dtype
            ):
                raise ValueError("计数模板与检查点形状或dtype不符")
            if abstract.shape == ():
                scalars[jax.tree_util.keystr(keys)] = (
                    checkpoint.get_tensor(entry["key"]).reshape(()).item()
                )
    if hashlib.sha256(path.read_bytes()).hexdigest() != before:
        raise ValueError("审计过程中检查点发生变化")
    return {
        "schema": "nvidia_v2d.policy_checkpoint_audit.v1",
        "checkpoint": str(path),
        "checkpoint_sha256": before,
        "source_unchanged": True,
        "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scalar_state": scalars,
        "updates": summarize_gradient_updates(scalars),
        "environment_steps": int(metadata["environment_steps"]),
        "global_update_attempts": int(metadata["global_update_step"]),
        "template_leaf_count": len(paths),
        "template_all_shapes_dtypes_checked": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("审计输出必须为新文件")
    result = audit_checkpoint(args.checkpoint)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, ensure_ascii=False))
    if not result["updates"]["effective_update_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
