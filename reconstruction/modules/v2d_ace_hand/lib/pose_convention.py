"""Convert ACE residual MANO poses to the existing flat-mean consumer contract."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

POSE_CONVENTION = "absolute_axis_angle_flat_mean_v2"


def ace_pose_to_flat_right(
    global_orient: np.ndarray,
    hand_pose: np.ndarray,
    hand_mean: np.ndarray,
    *,
    is_right: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return global/finger AA for flat-mean, mirrored-right MANO consumers.

    ACE's rotation matrices encode the *residual* axis-angle passed to smplx
    MANO(flat_hand_mean=False). smplx adds its learned mean component-wise in
    axis-angle space before Rodrigues; composing rotation matrices is different.
    Existing WiLoR consumers use manotorch's flat_hand_mean=True contract, so
    the mean must be included explicitly in the exported finger axis-angle.

    For a physical left MANO model, include the LEFT mean before reflecting
    rotations to the right model's canonical frame. S R S reflection on x maps
    axis-angle (x,y,z) to (x,-y,-z). Translation is not part of this conversion.
    """
    orient = Rotation.from_matrix(np.asarray(global_orient).reshape(3, 3)).as_rotvec()
    fingers = Rotation.from_matrix(np.asarray(hand_pose).reshape(15, 3, 3)).as_rotvec()
    fingers += np.asarray(hand_mean, dtype=np.float64).reshape(15, 3)
    if not is_right:
        orient *= [1.0, -1.0, -1.0]
        fingers *= [1.0, -1.0, -1.0]
    return orient, fingers.reshape(45)
