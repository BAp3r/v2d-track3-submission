"""Explicit moderate wrist-gain ablation with unchanged physical effort limits."""
from __future__ import annotations


def configure_wrist(config, *, kp: float = 100.0, kd: float = 12.0):
    if not 25 <= kp <= 200 or not 6 <= kd <= 25:
        raise ValueError("moderate wrist ablation is limited to kp25..200, kd6..25")
    config.joint_overrides = dict(config.joint_overrides)
    key = r"wrist_[xyz]$"
    prior = dict(config.joint_overrides[key])
    if prior.get("effort_limit") != 45.0:
        raise ValueError("expected the original 45 N translational effort limit")
    config.joint_overrides[key] = {**prior, "kp": float(kp), "kd": float(kd)}
    return config


def make_sharpa_config(*, kp: float = 100.0, kd: float = 12.0):
    """Hydra factory; affects prismatic robot wrist drives only, never objects."""
    from flash_chord.embodiments.sharpa_hands import SharpaHandsConfig

    return configure_wrist(SharpaHandsConfig(), kp=kp, kd=kd)
