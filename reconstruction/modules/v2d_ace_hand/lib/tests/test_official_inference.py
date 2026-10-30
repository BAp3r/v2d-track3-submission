"""CPU regressions against the unchanged official decoder and import contract.

Run inside v2d_ace_hand without --gpus. ACE_OFFICIAL_ROOT may point to another
checkout; /opt/ace is the image's pinned official source by default.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import unittest

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from official_inference import MappedVideoDecoder  # noqa: E402
from video_to_hands import (  # noqa: E402
    _load_native_prediction, _native_cache_valid, _normalize_camera,
    _validate_import_camera, write_native_provenance,
)


class OfficialDecoderEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        official_root = Path(os.environ.get("ACE_OFFICIAL_ROOT", "/opt/ace"))
        spec = importlib.util.spec_from_file_location(
            "reference_video_vae", official_root / "ace_ego_hand/video_vae.py",
        )
        cls.reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.reference)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.video = cls.root / "checker.avi"
        writer = cv2.VideoWriter(str(cls.video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (96, 64))
        if not writer.isOpened():
            raise RuntimeError("OpenCV MJPG encoder is needed for the CPU decoder fixture")
        rng = np.random.default_rng(42)
        for index in range(9):
            frame = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
            frame[:, :8, :] = [20 + index, 70, 230]
            writer.write(frame)
        writer.release()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_pixels_bitwise_equal_for_official_resize_paths(self):
        cases = [{}, {"resize_hw": (64, 32)}, {"resize_hw": (128, 96)},
                 {"resize_w": 64}, {"resize_hw": (96, 64)},
                 {"resize_hw": (96, 64), "resize_w": 48}]
        for options in cases:
            with self.subTest(options=options):
                decoder = MappedVideoDecoder(self.root / "mapped")
                try:
                    expected = self.reference.decode_video(str(self.video), 9, **options)
                    actual = decoder(str(self.video), 9, **options)
                    self.assertEqual(actual.dtype, torch.float32)
                    self.assertTrue(actual.is_contiguous())
                    self.assertTrue(torch.equal(expected, actual))
                    self.assertEqual(float((expected - actual).abs().max()), 0.0)
                    self.assertTrue(all(path.exists() for path in decoder.paths))
                finally:
                    paths = list(decoder.paths)
                    decoder.cleanup()
                    self.assertTrue(all(not path.exists() for path in paths))

    def test_short_video_is_rejected_instead_of_zero_filling(self):
        decoder = MappedVideoDecoder(self.root / "mapped")
        try:
            with self.assertRaisesRegex(RuntimeError, "decoded 9 < expected 10"):
                decoder(str(self.video), 10)
        finally:
            decoder.cleanup()


class OfficialNativeImportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.camera = _normalize_camera(
            {"fx": 689.1781005859375, "fy": 689.0325927734375,
             "cx": 645.9605102539063, "cy": 358.50384521484375,
             "width": 1280, "height": 720}, 1280, 720,
        )
        self.prediction = {
            "go": np.tile(np.eye(3, dtype=np.float32), (9, 2, 1, 1, 1)),
            "hp": np.tile(np.eye(3, dtype=np.float32), (9, 2, 15, 1, 1)),
            "betas": np.zeros((9, 2, 10), np.float32),
            "cam_trans": np.ones((9, 2, 3), np.float32),
            "claim": np.ones((9, 2), bool),
            "exists_2d": np.ones((9, 2), np.float32),
            "is_right": np.tile(np.array([0.0, 1.0], np.float32), (9, 1)),
            "intrinsics": {"image_width": 832, "image_height": 480},
        }
        for key in ("fx", "fy", "cx", "cy"):
            self.prediction["intrinsics"][key] = self.camera["frames"][0]["intrinsics"][key] * (
                832 / 1280 if key in ("fx", "cx") else 480 / 720)
        self.path = self.root / "color.pkl"
        self._save()

    def tearDown(self):
        self.temp.cleanup()

    def _save(self):
        with self.path.open("wb") as stream:
            pickle.dump(self.prediction, stream)

    def test_matching_official_whole_video_import_is_accepted(self):
        data = _load_native_prediction(self.path, 9)
        _validate_import_camera(data, self.camera, 1280, 720)
        official_camera = {"image_width": 1280, "image_height": 720,
                           "frames": [{"intrinsics": self.camera["frames"][0]["intrinsics"]}]}
        self.assertEqual(_normalize_camera(official_camera, 1280, 720), self.camera)

    def test_frame_count_and_presence_semantics_are_checked(self):
        with self.assertRaisesRegex(ValueError, "must have shape"):
            _load_native_prediction(self.path, 8)
        self.prediction["claim"][0, 0] = False
        self._save()
        with self.assertRaisesRegex(ValueError, "gating"):
            _load_native_prediction(self.path, 9)

    def test_wrong_camera_and_kfree_placeholder_are_rejected(self):
        self.prediction["intrinsics"]["fx"] += 10
        with self.assertRaisesRegex(ValueError, "calibration"):
            _validate_import_camera(self.prediction, self.camera, 1280, 720)
        self.prediction["intrinsics"]["source"] = "nominal 60deg placeholder"
        with self.assertRaisesRegex(ValueError, "K-given"):
            _validate_import_camera(self.prediction, self.camera, 1280, 720)

    def test_legacy_cache_or_changed_native_pickle_is_rejected(self):
        provenance = self.root / "provenance.done"
        provenance.write_text(json.dumps({"frames": 9, "chunk_frames": 81}))
        expected = {"inference_implementation": "official_whole_video_tiled_v1"}
        self.assertFalse(_native_cache_valid(self.path, provenance, expected))
        write_native_provenance(provenance, inputs=expected, prediction_path=self.path)
        self.assertTrue(_native_cache_valid(self.path, provenance, expected))
        self.assertFalse(_native_cache_valid(self.path, provenance, {"other": "inputs"}))
        with self.path.open("ab") as stream:
            stream.write(b"changed")
        self.assertFalse(_native_cache_valid(self.path, provenance, expected))


if __name__ == "__main__":
    unittest.main(verbosity=2)
