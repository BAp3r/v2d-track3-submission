"""Regression against the actual smplx producer and manotorch consumer.

Run in v2d_wilor with both packages installed. Point ACE_TEST_SMPLX_ROOT to
converted data/mano (containing mano/MANO_*.pkl), and ACE_TEST_MANOTORCH_ROOT to
assets containing models/MANO_*.pkl. These licensed assets are not bundled.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from pose_convention import ace_pose_to_flat_right  # noqa: E402


@unittest.skipUnless(
    os.environ.get("ACE_TEST_SMPLX_ROOT") and os.environ.get("ACE_TEST_MANOTORCH_ROOT"),
    "Set licensed MANO asset roots to exercise smplx/manotorch consumers",
)
class ActualManoConsumerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import smplx
        import torch
        from manotorch.manolayer import ManoLayer

        cls.torch = torch
        cls.official = {}
        for side in (False, True):
            cls.official[side] = smplx.create(
                os.environ["ACE_TEST_SMPLX_ROOT"], model_type="mano", is_rhand=side,
                use_pca=False, flat_hand_mean=False,
            ).eval()
        cls.consumer = ManoLayer(
            rot_mode="axisang", use_pca=False, side="right", center_idx=None,
            mano_assets_root=os.environ["ACE_TEST_MANOTORCH_ROOT"],
        ).eval()
        # The contract under test is the consumer's real default, not a
        # separately configured non-flat model that would conceal the bug.
        assert cls.consumer.flat_hand_mean is True

    def _vertices(self, is_right, betas):
        torch = self.torch
        go_aa = np.array([0.35, -0.18, 0.11], np.float32)
        hp_aa = np.linspace(-0.32, 0.27, 45, dtype=np.float32).reshape(15, 3)
        official = self.official[is_right]
        go, hp = ace_pose_to_flat_right(
            Rotation.from_rotvec(go_aa).as_matrix(),
            Rotation.from_rotvec(hp_aa).as_matrix(),
            official.pose_mean.detach().numpy()[3:], is_right=is_right,
        )
        with torch.no_grad():
            expected = official(
                global_orient=torch.from_numpy(go_aa[None]),
                hand_pose=torch.from_numpy(hp_aa.reshape(1, 45)),
                betas=torch.tensor(betas, dtype=torch.float32).reshape(1, 10),
            ).vertices[0].numpy()
            actual = self.consumer(
                torch.tensor(np.r_[go, hp], dtype=torch.float32).reshape(1, 48),
                torch.tensor(betas, dtype=torch.float32).reshape(1, 10),
            ).verts[0].numpy()
            # Previous export omitted the mean entirely.
            legacy = self.consumer(
                torch.from_numpy(np.r_[go_aa, hp_aa.ravel()][None]),
                torch.tensor(betas, dtype=torch.float32).reshape(1, 10),
            ).verts[0].numpy()
        if not is_right:
            actual[:, 0] *= -1
        return expected, actual, legacy

    def test_right_residual_includes_nonflat_mean(self):
        expected, actual, legacy = self._vertices(True, np.linspace(-0.5, 0.5, 10))
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
        self.assertGreater(float(np.linalg.norm(legacy - expected, axis=-1).mean()), 0.01)

    @unittest.skipUnless(os.environ.get("ACE_TEST_RAW_ROOT"), "Set ACE_TEST_RAW_ROOT for recorded frame 2400 regression")
    def test_recorded_frame_2400_right_hand(self):
        import pickle

        # This is the observed failure from the original integration, retained
        # as an opt-in external fixture rather than redistributing video data.
        with (Path(os.environ["ACE_TEST_RAW_ROOT"]) / "ace_002349.pkl").open("rb") as stream:
            pred = pickle.load(stream)
        index, slot = 51, 1  # original frame 2349 + 51 = 2400, right hand
        go_matrix = pred["global_orient"][index, slot].reshape(3, 3)
        hp_matrix = pred["hand_pose"][index, slot].reshape(15, 3, 3)
        betas = pred["betas"][index, slot].reshape(1, 10)
        go_residual = Rotation.from_matrix(go_matrix).as_rotvec()
        hp_residual = Rotation.from_matrix(hp_matrix).as_rotvec().reshape(45)
        official, torch = self.official[True], self.torch
        go, hp = ace_pose_to_flat_right(
            go_matrix, hp_matrix, official.pose_mean.detach().numpy()[3:], is_right=True,
        )
        with torch.no_grad():
            expected = official(
                global_orient=torch.tensor(go_residual, dtype=torch.float32).reshape(1, 3),
                hand_pose=torch.tensor(hp_residual, dtype=torch.float32).reshape(1, 45),
                betas=torch.tensor(betas, dtype=torch.float32),
            ).vertices[0].numpy()
            actual = self.consumer(
                torch.tensor(np.r_[go, hp], dtype=torch.float32).reshape(1, 48),
                torch.tensor(betas, dtype=torch.float32),
            ).verts[0].numpy()
            legacy = self.consumer(
                torch.tensor(np.r_[go_residual, hp_residual], dtype=torch.float32).reshape(1, 48),
                torch.tensor(betas, dtype=torch.float32),
            ).verts[0].numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
        legacy_error = np.linalg.norm(legacy - expected, axis=-1)
        self.assertGreater(float(legacy_error.mean()), 0.03)
        self.assertGreater(float(legacy_error.max()), 0.08)

    def test_left_mean_is_added_before_mirrored_right_conversion(self):
        # Zero shape isolates pose conversion from asymmetric left/right shape
        # bases in some MANO releases. Physical left and mirrored right must agree.
        expected, actual, _ = self._vertices(False, np.zeros(10))
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
