"""Compare derived MoGe depth with bounded, registered sensor samples on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image


def depth_comparison(
    estimated: NDArray[Any],
    measured: NDArray[Any],
    measured_valid: NDArray[Any],
    mask: NDArray[Any] | None = None,
) -> dict[str, Any]:
    """Compare the same RGB pixels; empty or invalid evidence stays empty."""
    if estimated.ndim != 2 or estimated.shape != measured.shape:
        raise ValueError("Depth maps must share a 2D RGB pixel grid")
    if measured_valid.shape != measured.shape or (mask is not None and mask.shape != measured.shape):
        raise ValueError("Validity and mask shapes must match depth")
    valid = (
        measured_valid.astype(bool)
        & np.isfinite(estimated)
        & np.isfinite(measured)
        & (estimated > 0)
        & (measured > 0)
    )
    if mask is not None:
        valid &= mask.astype(bool)
    count = int(valid.sum())
    result: dict[str, Any] = {"paired_pixels": count}
    if count == 0:
        return result
    prediction, observation = estimated[valid], measured[valid]
    errors = prediction - observation
    result.update(
        estimated_median_m=float(np.median(prediction)),
        sensor_median_m=float(np.median(observation)),
        signed_error_median_m=float(np.median(errors)),
        absolute_error_median_m=float(np.median(np.abs(errors))),
        absolute_error_p90_m=float(np.percentile(np.abs(errors), 90)),
        estimated_to_sensor_ratio_median=float(np.median(prediction / observation)),
    )
    return result


def compare_samples(reconstruction: Path, samples: Path) -> dict[str, Any]:
    """Read registered samples and available derived outputs without modifying them."""
    metadata = json.loads((samples / "manifest.json").read_text())
    calibration = json.loads((samples / "calibration.json").read_text())
    if metadata["status"] != "succeeded":
        raise ValueError("Sensor sampling must have succeeded")
    result_path = reconstruction / "result" / "manifest.json"
    result = json.loads(result_path.read_text()) if result_path.is_file() else {}
    object_id = int(result.get("sources", {}).get("object_track_id", 1))
    rows = []
    for entry in metadata["frames"]:
        frame = int(entry["frame_index"])
        depth_path = reconstruction / "depth" / f"{frame:06d}.png"
        row: dict[str, Any] = {"frame": frame, "moge_depth_exists": depth_path.is_file()}
        if depth_path.is_file():
            with Image.open(depth_path) as image:
                encoded = np.asarray(image)
            if encoded.dtype != np.uint16:
                raise ValueError("Expected the upstream MoGe inverse-depth PNG encoding")
            estimated = np.zeros(encoded.shape, dtype=np.float64)
            positive = (encoded > 0) & (encoded < 65535)
            estimated[positive] = 65535.0 / encoded[positive] - 1.0
            sample_path = samples / entry["path"]
            digest = hashlib.sha256(sample_path.read_bytes()).hexdigest()
            if digest != entry["sha256"]:
                raise ValueError("Registered sensor sample hash mismatch")
            with np.load(sample_path, allow_pickle=False) as sample:
                measured, valid = sample["depth_color_m"], sample["valid"]
            row["whole_frame"] = depth_comparison(estimated, measured, valid)
            mask_path = reconstruction / "masks" / str(object_id) / f"{frame:06d}.png"
            if mask_path.is_file():
                with Image.open(mask_path) as image:
                    mask = np.asarray(image) > 0
                row["object_mask_pixels"] = int(mask.sum())
                row["object_mask"] = depth_comparison(estimated, measured, valid, mask)
            k_path = reconstruction / "intrinsics" / f"{frame:06d}.json"
            if k_path.is_file():
                intrinsic = json.loads(k_path.read_text())
                row["moge_fx"] = intrinsic["fx"]
                row["moge_to_sensor_fx_ratio"] = intrinsic["fx"] / calibration["color"]["fx"]
        rows.append(row)
    return {
        "schema": "nvidia_v2d.sensor_depth_comparison.v1",
        "reconstruction_root": str(reconstruction),
        "sensor_samples_root": str(samples),
        "sensor_manifest_sha256": hashlib.sha256(
            (samples / "manifest.json").read_bytes()
        ).hexdigest(),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "depth_scale_source": metadata["depth_scale_source"],
        "calibration_sha256": calibration["sha256"],
        "object_track_id": object_id,
        "interpretation": (
            "RGB-pixel paired estimated/sensor z-depth comparison, not a contact metric. "
            "Sensor depth uses recorded extrinsics and the verified capture profile. "
            "Pixel registration holes, occlusion, edge noise and mask errors remain. "
            "MoGe baseline estimated its own focal length; no sensor-depth prior was used."
        ),
        "frames": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction-root", required=True, type=Path)
    parser.add_argument("--sensor-samples", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root, samples, output = (
        args.reconstruction_root.resolve(),
        args.sensor_samples.resolve(),
        args.output.resolve(),
    )
    recording = Path(json.loads((samples / "manifest.json").read_text())["recording"]).resolve()
    if any(output.is_relative_to(path) for path in (root, samples, recording)):
        parser.error("Output must be outside all input trees")
    if args.output.exists() or args.output.is_symlink():
        parser.error("Output must be a new file")
    report = compare_samples(root, samples)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, "w") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {len(report['frames'])} sample checks to {output}")


if __name__ == "__main__":
    main()
