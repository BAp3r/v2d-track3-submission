"""CPU tests; no simulator, GPU, learned policy or input reference is modified."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

wp = pytest.importorskip("warp")
pytest.importorskip("flash_chord")
from flash_chord.objectives.composition import (
    CurriculumObjective,
    Objective,
    ObjectiveBuffers,
    ObjectiveDiagnostics,
)
from flash_chord.objectives.config import (
    OBJECTIVE_TERM_NAMES,
    ObjectiveConfig,
    ObjectiveTermConfig,
)
from flash_chord.objectives.contact import unintended_contact_penalty
from hydra.utils import instantiate

from experiments.nvidia_v2d.release_contact_objective import (
    ReleaseContactObjective,
    ReleaseContactObjectiveConfig,
    make_release_contact_objective,
)


def arr(values, dtype=wp.float32):
    return wp.array(values, dtype=dtype, device="cpu")


class ContactObjectiveFixture:
    """Real upstream contact penalty/composition with minimal runtime dependencies."""

    def __init__(self):
        self.world_count = 4
        self.config = make_release_contact_objective(release_start_frame=407)
        self.buffers = ObjectiveBuffers.build(4, 0.05, self.config, device="cpu")
        self.action_rate_l2 = arr([1.0, 2.0, 3.0, 4.0])
        self.action_l2 = arr([4.0, 3.0, 2.0, 1.0])
        self.cmd = arr([0.0, 0.0, 0.0, 0.0])
        self.cur = arr([1.0, 1.0, 1.0, 0.0])
        self.reset_seen = None

    score = property(lambda self: self.buffers.score)
    terms = property(lambda self: self.buffers.terms)
    weights = property(lambda self: self.buffers.weights)
    term_names = property(lambda self: OBJECTIVE_TERM_NAMES)

    def evaluate(self, state, timestep, terminated):
        self.buffers.object_keypoints.assign([0.2, 0.4, 0.6, 0.8])
        wp.launch(
            unintended_contact_penalty,
            dim=4,
            inputs=[self.cmd, self.cur, 1, 1, 1],
            outputs=[self.buffers.unintended_contact],
            device="cpu",
        )
        self.buffers.compose(terminated, self.action_rate_l2, self.action_l2)

    def set_weights(self, weights):
        if set(weights) != set(self.term_names):
            raise ValueError("objective weights mismatch")
        self.weights.assign([weights[name] for name in self.term_names])

    def reset(self, reset_mask):
        self.reset_seen = reset_mask


def test_mask_exact_boundary_and_recompose_preserving_other_terms():
    inner = ContactObjectiveFixture()
    wrapped = ReleaseContactObjective(inner, 407)
    timestep = arr([0, 406, 407, 495], wp.int32)
    terminated = arr([0, 0, 1, 0], wp.int32)
    inner.evaluate(None, timestep, terminated)
    raw_terms = inner.terms.numpy().reshape(4, -1).copy()
    raw_score = inner.score.numpy().copy()
    wrapped.evaluate(None, timestep, terminated)
    terms = wrapped.terms.numpy().reshape(4, -1)
    index = OBJECTIVE_TERM_NAMES.index("unintended_contact")
    np.testing.assert_array_equal(terms[:, index], [0.0, 0.0, 2.0, 0.0])
    np.testing.assert_array_equal(np.delete(terms, index, 1), np.delete(raw_terms, index, 1))
    np.testing.assert_allclose(wrapped.score.numpy(), raw_score + [1.0, 1.0, 0.0, 0.0], atol=1e-6)
    assert wrapped.score is inner.score and wrapped.terms is inner.terms
    assert isinstance(wrapped, (Objective,))
    assert isinstance(wrapped, ObjectiveDiagnostics)
    assert isinstance(wrapped, CurriculumObjective)
    assert wrapped.term_names == OBJECTIVE_TERM_NAMES
    reset_mask = arr([1, 0, 0, 0], wp.int32)
    wrapped.reset(reset_mask)
    assert inner.reset_seen is reset_mask
    # A subsequent step/reset must recalculate the raw term before applying its gate.
    wrapped.evaluate(None, arr([407, 407, 406, 0], wp.int32), terminated)
    np.testing.assert_array_equal(inner.buffers.unintended_contact.numpy(), [2, 2, 0, 0])


@pytest.mark.parametrize("frame", [-1, 1.5, True])
def test_invalid_boundary_rejected(frame):
    with pytest.raises(ValueError, match="release_start_frame"):
        make_release_contact_objective(release_start_frame=frame)


@pytest.mark.parametrize("weight", [0.0, 1.0])
def test_nonnegative_config_and_curriculum_weights_rejected(weight):
    with pytest.raises(ValueError, match="negative"):
        make_release_contact_objective(
            release_start_frame=407, unintended_contact=ObjectiveTermConfig(weight=weight)
        )
    inner = ContactObjectiveFixture()
    wrapped = ReleaseContactObjective(inner, 407)
    weights = dict(zip(OBJECTIVE_TERM_NAMES, inner.config.weights, strict=True))
    weights["unintended_contact"] = weight
    with pytest.raises(ValueError, match="negative"):
        wrapped.set_weights(weights)
    weights["unintended_contact"] = -3.0
    wrapped.set_weights(weights)
    assert inner.weights.numpy()[5] == -3.0


def test_disabled_penalty_rejected():
    with pytest.raises(ValueError, match="enabled"):
        make_release_contact_objective(
            release_start_frame=407,
            unintended_contact=ObjectiveTermConfig(weight=-10.0, enabled=False),
        )


def test_reference_range_and_nonzero_suffix_rejected_before_return(monkeypatch):
    calls = []
    support = np.zeros((496, 2), dtype=np.float32)

    def build_inner(*args, **kwargs):
        calls.append(True)
        return SimpleNamespace(ref_wrench_support=arr(support.reshape(-1)))

    monkeypatch.setattr(ObjectiveConfig, "build", build_inner)
    args = [None, None, None, SimpleNamespace(num_frames=496), None, None, None, 4, 0.05]
    with pytest.raises(ValueError, match="inside"):
        make_release_contact_objective(release_start_frame=496).build(*args)
    assert not calls
    cfg = make_release_contact_objective(release_start_frame=407)
    assert isinstance(cfg.build(*args), ReleaseContactObjective)
    support[406, 0] = 1.0  # Earlier contact is allowed.
    assert isinstance(cfg.build(*args), ReleaseContactObjective)
    support[407, 0] = 1e-8  # No tolerance may silently accept a nonzero release target.
    with pytest.raises(ValueError, match="exactly zero"):
        cfg.build(*args)


def test_hydra_json_checkpoint_config_roundtrip():
    config = {
        "_target_": "experiments.nvidia_v2d.release_contact_objective.make_release_contact_objective",
        "release_start_frame": 407,
        "unintended_contact": {
            "_target_": "flash_chord.objectives.config.ObjectiveTermConfig",
            "weight": -10.0,
            "enabled": True,
        },
        "num_wrench_basis": 512,
    }
    restored = instantiate(json.loads(json.dumps(config)), _convert_="all")
    assert isinstance(restored, ObjectiveConfig)
    assert isinstance(restored, ReleaseContactObjectiveConfig)
    assert restored.release_start_frame == 407
    assert restored.unintended_contact.weight == -10.0
