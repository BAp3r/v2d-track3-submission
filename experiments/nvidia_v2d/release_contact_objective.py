"""Gate the existing unintended-contact penalty to a verified release suffix.

The frame is in the runtime reference timeline, after playback resampling.  This
changes only reward composition; reference, actions, contacts and termination
remain owned by the original FlashCHORD implementation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import warp as wp
from flash_chord.objectives.config import ObjectiveConfig


@wp.kernel
def mask_pre_release_contact(
    timestep: wp.array(dtype=wp.int32),
    release_start_frame: int,
    unintended_contact: wp.array(dtype=wp.float32),
) -> None:
    world = wp.tid()
    if timestep[world] < release_start_frame:
        unintended_contact[world] = 0.0


def _require_negative_weight(value: float) -> None:
    if not math.isfinite(value) or value >= 0.0:
        raise ValueError("release unintended_contact weight must be finite and negative")


@dataclass(frozen=True)
class ReleaseContactObjectiveConfig(ObjectiveConfig):
    """Original objective fields plus an explicit runtime release boundary."""

    release_start_frame: int = -1

    def __post_init__(self) -> None:
        super().__post_init__()
        if type(self.release_start_frame) is not int or self.release_start_frame < 0:
            raise ValueError("release_start_frame must be an explicit non-negative integer")
        if not self.unintended_contact.enabled:
            raise ValueError("release unintended_contact must be enabled")
        _require_negative_weight(self.unintended_contact.weight)

    def build(
        self,
        model,
        embodiment,
        robot_reference,
        reference,
        command,
        contact,
        action,
        world_count: int,
        frame_dt: float,
        device=None,
        reference_joint_q_offset=None,
        episode_start_frame=None,
    ):
        if self.release_start_frame >= reference.num_frames:
            raise ValueError("release_start_frame must lie inside the runtime reference")
        inner = super().build(
            model,
            embodiment,
            robot_reference,
            reference,
            command,
            contact,
            action,
            world_count,
            frame_dt,
            device=device,
            reference_joint_q_offset=reference_joint_q_offset,
            episode_start_frame=episode_start_frame,
        )
        support = inner.ref_wrench_support.numpy().reshape(reference.num_frames, -1)
        suffix = support[self.release_start_frame :]
        if not np.isfinite(suffix).all() or np.any(suffix != 0.0):
            raise ValueError("release suffix must have exactly zero reference wrench support")
        return ReleaseContactObjective(inner, self.release_start_frame)


@dataclass
class ReleaseContactObjective:
    """Preserve Objective, ObjectiveDiagnostics and CurriculumObjective protocols."""

    inner: object
    release_start_frame: int

    @property
    def config(self):
        return self.inner.config

    @property
    def buffers(self):
        return self.inner.buffers

    @property
    def score(self):
        return self.inner.score

    @property
    def terms(self):
        return self.inner.terms

    @property
    def weights(self):
        return self.inner.weights

    @property
    def term_names(self):
        return self.inner.term_names

    def set_weights(self, weights: Mapping[str, float]) -> None:
        if "unintended_contact" in weights:
            _require_negative_weight(float(weights["unintended_contact"]))
        self.inner.set_weights(weights)

    def reset(self, reset_mask) -> None:
        self.inner.reset(reset_mask)

    def evaluate(self, state, timestep, terminated) -> None:
        self.inner.evaluate(state, timestep, terminated)
        wp.launch(
            mask_pre_release_contact,
            dim=self.inner.world_count,
            inputs=[timestep, self.release_start_frame],
            outputs=[self.buffers.unintended_contact],
        )
        self.buffers.compose(terminated, self.inner.action_rate_l2, self.inner.action_l2)


def make_release_contact_objective(*, release_start_frame: int, **kwargs):
    """Hydra factory, including checkpoint reload of all original typed terms."""
    return ReleaseContactObjectiveConfig(release_start_frame=release_start_frame, **kwargs)
