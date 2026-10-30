"""Lossless ragged simulator-contact recording, without changing physics state."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

CONTACT_WIDTHS = {
    "body0_ids": None, "body1_ids": None, "shape0_ids": None, "shape1_ids": None,
    "contact_pos_w": 3, "contact_normal_w": 3, "contact_force_w": 3,
    "contact_torque_w": 3, "signed_gap_m": None,
}


def pack_contact_frames(frames: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Keep every contact and frame boundary; empty frames remain explicit."""
    counts = [len(frame["body0_ids"]) for frame in frames]
    arrays = {"contact_frame_offsets": np.concatenate(([0], np.cumsum(counts))).astype(np.int64)}
    for name, width in CONTACT_WIDTHS.items():
        pieces = []
        for count, frame in zip(counts, frames, strict=True):
            value = np.asarray(frame[name])
            expected = (count,) if width is None else (count, width)
            if value.shape != expected or not np.isfinite(value).all():
                raise ValueError(f"invalid contact array {name}: {value.shape}, expected {expected}")
            if name.endswith("_ids") and value.dtype.kind not in "iu":
                raise ValueError(f"contact IDs must be integers: {name}")
            pieces.append(value)
        dtype = np.int64 if name.endswith("_ids") else np.float64
        shape = (0,) if width is None else (0, width)
        arrays[f"contacts_{name}"] = (np.concatenate(pieces).astype(dtype, copy=False)
                                      if pieces else np.empty(shape, dtype=dtype))
    return arrays
