"""Run the unchanged official ACE entrypoint with equivalent disk-backed decode.

Only decoded pixel storage changes. RGB order, INTER_AREA resizing, float32
normalization, contiguous NCHW layout, the continuous causal VAE and the
official tiled prediction/packing code all retain their official behavior.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import runpy
import sys
import uuid

INFERENCE_IMPLEMENTATION = "official_whole_video_tiled_v1"


class MappedVideoDecoder:
    """Match video_vae.decode_video using a writable file-backed NCHW tensor."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.paths: list[Path] = []

    def __call__(self, path: str, n_frames: int, resize_w: int = 0,
                 resize_hw: tuple[int, int] | None = None):
        import cv2
        import numpy as np
        import torch

        if n_frames <= 0:
            raise ValueError("n_frames must be positive")
        cap = cv2.VideoCapture(str(path))
        data = None
        decoded = 0
        try:
            for index in range(n_frames):
                ok, frame = cap.read()
                if not ok:
                    break
                # Keep the official conditional structure, including resize_w's
                # fallback when resize_hw already matches the source dimensions.
                if resize_hw and (frame.shape[1] != resize_hw[0] or
                                  frame.shape[0] != resize_hw[1]):
                    frame = cv2.resize(frame, resize_hw, interpolation=cv2.INTER_AREA)
                elif resize_w and frame.shape[1] != resize_w:
                    # Official resize_w is width-only (height is unchanged).
                    # The normal entrypoint instead supplies resize_hw.
                    frame = cv2.resize(
                        frame, (resize_w, frame.shape[0]), interpolation=cv2.INTER_AREA,
                    )
                rgb = frame[:, :, ::-1].copy()
                if data is None:
                    height, width = rgb.shape[:2]
                    target = self.cache_dir / (
                        f"ace_pixels_{os.getpid()}_{uuid.uuid4().hex}.f32"
                    )
                    self.paths.append(target)
                    data = np.memmap(target, mode="w+", dtype=np.float32,
                                     shape=(n_frames, 3, height, width))
                # Each operation rounds to float32 exactly as the official
                # np.stack(...).astype(float32) / 255.0 * 2.0 - 1.0 expression.
                values = rgb.astype(np.float32)
                values /= np.float32(255.0)
                values *= np.float32(2.0)
                values -= np.float32(1.0)
                data[index] = values.transpose(2, 0, 1)
                decoded += 1
                if decoded % 500 == 0:
                    print(f"[disk decode] {decoded}/{n_frames}", flush=True)
        finally:
            cap.release()
        if decoded < n_frames:
            raise RuntimeError(f"{path}: decoded {decoded} < expected {n_frames}")
        data.flush()
        return torch.from_numpy(data)

    def cleanup(self) -> None:
        """Unlink only this decoder's temporary backing files after inference."""
        for path in self.paths:
            path.unlink(missing_ok=True)
        self.paths.clear()


def prepare_caption(official_root: Path, backbone_dir: Path, cache_dir: Path) -> None:
    """Call the official caption preparation in its own short-lived process."""
    import torch

    spec = importlib.util.spec_from_file_location(
        "ace_caption", official_root / "scripts/precompute_caption.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = cache_dir
    module.MODEL_ROOT = backbone_dir
    module.cache_caption(torch.device("cuda"))


def run_official_entry(official_root: Path, backbone_dir: Path, cache_dir: Path,
                       official_args: list[str]) -> None:
    from ace_ego_hand import video_vae

    entry = official_root / "infer_video.py"
    original_decode = video_vae.decode_video
    original_model_root = video_vae.MODEL_ROOT
    original_argv = sys.argv
    decoder = MappedVideoDecoder(cache_dir)
    try:
        video_vae.decode_video = decoder
        video_vae.MODEL_ROOT = backbone_dir
        sys.argv = [str(entry), *official_args]
        runpy.run_path(str(entry), run_name="__main__")
    finally:
        sys.argv = original_argv
        video_vae.decode_video = original_decode
        video_vae.MODEL_ROOT = original_model_root
        decoder.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--official_root", default="/opt/ace")
    parser.add_argument("--backbone_dir", required=True)
    parser.add_argument("--decode_cache_dir", required=True)
    parser.add_argument("--prepare_caption", action="store_true")
    args, official_args = parser.parse_known_args()
    root = Path(args.official_root).resolve()
    sys.path.insert(0, str(root))
    if args.prepare_caption:
        prepare_caption(root, Path(args.backbone_dir).resolve(),
                        Path(args.decode_cache_dir).resolve())
    else:
        run_official_entry(root, Path(args.backbone_dir).resolve(),
                           Path(args.decode_cache_dir).resolve(), official_args)


if __name__ == "__main__":
    main()
