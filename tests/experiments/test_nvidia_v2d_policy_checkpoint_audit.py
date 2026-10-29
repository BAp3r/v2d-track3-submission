import json
import os
from pathlib import Path

import numpy as np
import pytest

from experiments.nvidia_v2d.policy_checkpoint_audit import summarize_gradient_updates


def state(actor=49, actor_attempts=50, critic=98, critic_attempts=100):
    return {
        ".actor.optimizer_state.count": actor,
        ".actor.schedule_step": actor_attempts,
        ".critic.optimizer_state.count": critic,
        ".critic.schedule_step": critic_attempts,
        ".global_update_step": critic_attempts,
    }


def test_attempt_counters_do_not_hide_skipped_adam_updates():
    result = summarize_gradient_updates(state())
    assert result["actor"] == {"attempted": 50, "applied": 49, "skipped": 1, "skip_fraction": 0.02}
    assert result["critic"] == {"attempted": 100, "applied": 98, "skipped": 2, "skip_fraction": 0.02}
    assert result["effective_update_gate_passed"]


@pytest.mark.parametrize("values", [state(actor=0), state(critic=0), state(actor=0, critic=0)])
def test_no_effective_actor_or_critic_update_fails_training_gate(values):
    result = summarize_gradient_updates(values)
    assert not result["effective_update_gate_passed"]
    assert result["failure_reason"]


@pytest.mark.parametrize("values", [state(actor=51), state(critic=-1), state(actor=1.5)])
def test_impossible_optimizer_counts_are_rejected(values):
    with pytest.raises(ValueError):
        summarize_gradient_updates(values)


def fixture_checkpoint():
    pytest.importorskip("flash_chord")
    source = os.environ.get("FMC3_FLASHCHORD_AUDIT_CHECKPOINT")
    if not source:
        pytest.skip("requires fixed FlashCHORD CPU image and an existing fixture checkpoint")
    return Path(source)


def test_real_cpu_checkpoint_template_and_effective_counts_match():
    path = fixture_checkpoint()
    from experiments.nvidia_v2d.policy_checkpoint_audit import audit_checkpoint

    result = audit_checkpoint(path)
    assert result["template_all_shapes_dtypes_checked"]
    assert result["template_leaf_count"] == 191
    assert result["updates"]["critic"]["applied"] == 98
    assert result["updates"]["actor"]["applied"] == 49
    assert result["source_unchanged"]


def test_changed_checkpoint_shape_contract_is_rejected(tmp_path):
    path = fixture_checkpoint()
    from safetensors import safe_open
    from safetensors.numpy import save_file

    from experiments.nvidia_v2d.policy_checkpoint_audit import audit_checkpoint

    with safe_open(str(path), framework="numpy") as source:
        metadata = source.metadata()
    entries = json.loads(metadata["manifest"])
    entries[0]["shape"] = [999]
    metadata["manifest"] = json.dumps(entries)
    malformed = tmp_path / "malformed.safetensors"
    save_file({"state.0000": np.zeros(1, dtype=np.float32)}, malformed, metadata=metadata)
    with pytest.raises(ValueError, match="形状或dtype"):
        audit_checkpoint(malformed)
