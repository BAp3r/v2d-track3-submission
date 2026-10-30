"""Official whole-video ACE inference, exported to canonical WiLoR records.

The official infer_video.py owns decoding semantics, continuous causal VAE,
windowed prediction, presence gating and native pickle packing. The only decode
change is byte-equivalent disk-backed storage. The separate CPU export includes
MANO's learned finger mean before converting to flat-mean right MANO consumers.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

from official_inference import INFERENCE_IMPLEMENTATION

EXPORT_IMPLEMENTATION = "official_pickle_to_flat_mano_records_v2"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_fingerprint(path: str | Path) -> dict:
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _source_fingerprint(official_root: Path) -> str:
    """Hash actual inference sources, independent of their container mount path."""
    paths = [official_root / "infer_video.py",
             official_root / "scripts/precompute_caption.py"]
    paths.extend((official_root / "ace_ego_hand").rglob("*.py"))
    paths.extend(official_root / "third_party/videox_fun/models" / name for name in (
        "wan_transformer3d.py", "wan_vae3_8.py", "wan_text_encoder.py",
    ))
    paths.append(official_root / "third_party/config/wan2.2/wan_civitai_5b.yaml")
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(official_root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_native_provenance(*, video_path: str | Path, weights_dir: str | Path,
                            backbone_dir: str | Path, mano_dir: str | Path,
                            intrinsics_path: str | Path,
                            caption_embed_path: str | Path,
                            official_root: str | Path = "/opt/ace") -> dict:
    """Build the portable ``inputs`` object for native_predictions/provenance.done.

    An explicitly preseeded official pickle is accepted only with this matching
    object plus its SHA-256. Paths are deliberately excluded from cache identity:
    host and container paths differ, whereas stat/hash values stay the same.
    """
    root = Path(official_root)
    backbone = Path(backbone_dir)
    return {
        "schema_version": 1,
        "inference_implementation": INFERENCE_IMPLEMENTATION,
        "video_fingerprint": file_fingerprint(video_path),
        "intrinsics_sha256": file_sha256(intrinsics_path),
        "checkpoint": file_fingerprint(Path(weights_dir) / "ace_ego_hand_k.pt"),
        "backbone": {
            name: file_fingerprint(backbone / name) for name in (
                "diffusion_pytorch_model.safetensors", "Wan2.2_VAE.pth",
                "models_t5_umt5-xxl-enc-bf16.pth",
            )
        },
        "backbone_config_sha256": file_sha256(backbone / "config.json"),
        "mano": {side: file_fingerprint(Path(mano_dir) / f"MANO_{side}.pkl")
                 for side in ("LEFT", "RIGHT")},
        "caption_embed_sha256": (file_sha256(caption_embed_path)
                                  if Path(caption_embed_path).is_file() else None),
        "official_sources_sha256": _source_fingerprint(root),
        "official_options_sha256": file_sha256(root / "options/ace_ego_hand_k.yml"),
        "decode_helper_sha256": file_sha256(Path(__file__).with_name("official_inference.py")),
        "parameters": {"mode": "tiled", "tile_w": 22, "encode_w": 832,
                       "tap": 15, "max_frames": 0, "split_segments": False},
    }


def write_native_provenance(path: str | Path, *, inputs: dict,
                             prediction_path: str | Path) -> dict:
    """Attest an official pickle after verifying its producing inputs externally."""
    prediction = Path(prediction_path)
    provenance = {"schema_version": 1, "inputs": inputs,
                  "prediction_file": prediction.name,
                  "prediction_sha256": file_sha256(prediction)}
    _write_json(Path(path), provenance)
    return provenance


def _write_json(path: Path, value: dict | list) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def _native_cache_valid(path: Path, provenance_path: Path, inputs: dict) -> bool:
    if not path.is_file() or not provenance_path.is_file():
        return False
    try:
        provenance = json.loads(provenance_path.read_text())
        return (provenance.get("inputs") == inputs and
                provenance.get("prediction_file") == path.name and
                provenance.get("prediction_sha256") == file_sha256(path))
    except (OSError, ValueError, TypeError):
        return False


def _normalize_camera(data: dict, width: int, height: int) -> dict:
    """Accept pipeline flat intrinsics and the official camera JSON convention."""
    import math

    if "frames" in data:
        try:
            intrinsics = dict(data["frames"][0]["intrinsics"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Official camera JSON needs frames[0].intrinsics") from exc
        cal_width = data.get("image_width", width)
        cal_height = data.get("image_height", height)
    else:
        intrinsics = dict(data)
        cal_width = data.get("width", data.get("image_width", width))
        cal_height = data.get("height", data.get("image_height", height))
    for key in ("fx", "fy", "cx", "cy"):
        intrinsics[key] = float(intrinsics[key])
        if not math.isfinite(intrinsics[key]):
            raise ValueError(f"Camera {key} must be finite")
    if intrinsics["fx"] <= 0 or intrinsics["fy"] <= 0:
        raise ValueError("Camera focal lengths must be positive")
    if not math.isfinite(float(cal_width)) or not math.isfinite(float(cal_height)):
        raise ValueError("Camera calibration dimensions must be finite")
    if float(cal_width) <= 0 or float(cal_height) <= 0:
        raise ValueError("Camera calibration dimensions must be positive")
    camera = {"image_width": cal_width, "image_height": cal_height,
              "frames": [{"intrinsics": intrinsics}]}
    for key in ("model", "fisheye"):
        if key in data:
            camera[key] = data[key]
    effective = {**intrinsics,
                 **{key: camera[key] for key in ("model", "fisheye") if key in camera}}
    if effective.get("fisheye") or str(effective.get("model", "")).lower() in ("mei", "kb", "fisheye"):
        raise ValueError("Official ACE checkpoints require pinhole intrinsics; undistort fisheye video first")
    return camera


def _prepare_assets(args, official_root: Path, cache: Path, *, compute_caption: bool) -> Path:
    """Convert licensed MANO assets and prepare the official fixed caption once."""
    mano = cache / "mano/mano"
    mano.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location(
        "ace_mano_convert", official_root / "scripts/convert_mano_pkls.py",
    )
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    mano_sources = {
        "converter_sha256": file_sha256(official_root / "scripts/convert_mano_pkls.py"),
        "files": {side: file_fingerprint(Path(args.mano_dir) / f"MANO_{side}.pkl")
                  for side in ("LEFT", "RIGHT")},
    }
    marker = mano.parent / "source_fingerprint.done"
    try:
        current_sources = json.loads(marker.read_text())
    except (OSError, ValueError, TypeError):
        current_sources = None
    if current_sources != mano_sources or not all(
            (mano / f"MANO_{side}.pkl").is_file() for side in ("LEFT", "RIGHT")):
        for side in ("LEFT", "RIGHT"):
            target = mano / f"MANO_{side}.pkl"
            temporary = target.with_name(target.name + ".convert.tmp")
            converter.convert(Path(args.mano_dir) / target.name, temporary)
            temporary.replace(target)
        _write_json(marker, mano_sources)
    os.environ["ACE_EGO_HAND_MANO_DIR"] = str(mano.parent)
    caption = cache / "cache/caption_embed.pt"
    if not caption.is_file() and compute_caption:
        subprocess.run([
            sys.executable, str(Path(__file__).with_name("official_inference.py")),
            "--official_root", str(official_root),
            "--backbone_dir", str(Path(args.backbone_dir).resolve()),
            "--decode_cache_dir", str(cache), "--prepare_caption",
        ], check=True)
    return caption


def _load_native_prediction(path: Path, total: int) -> dict:
    import numpy as np

    with path.open("rb") as stream:
        pred = pickle.load(stream)
    expected = {"go": (total, 2, 1, 3, 3), "hp": (total, 2, 15, 3, 3),
                "betas": (total, 2, 10), "cam_trans": (total, 2, 3),
                "claim": (total, 2), "exists_2d": (total, 2),
                "is_right": (total, 2)}
    for key, shape in expected.items():
        if key not in pred or np.asarray(pred[key]).shape != shape:
            raise ValueError(f"Not an official whole-video prediction: {key} must have shape {shape}")
    if np.asarray(pred["claim"]).dtype != np.bool_:
        raise ValueError("Official claim array must be boolean")
    if not np.array_equal(pred["claim"], np.asarray(pred["exists_2d"]) > 0.5):
        raise ValueError("Native claim array disagrees with official exists_2d > 0.5 gating")
    if not np.array_equal(pred["is_right"], np.tile([0.0, 1.0], (total, 1))):
        raise ValueError("Official native predictions must use fixed left/right hand slots")
    return pred


def _validate_import_camera(pred: dict, camera: dict, width: int, height: int) -> None:
    """Reject K-free/placeholders and mismatched calibrations on explicit import."""
    import numpy as np

    intrinsics = pred.get("intrinsics")
    if not isinstance(intrinsics, dict) or intrinsics.get("source"):
        raise ValueError("Native import requires official K-given predictions with supplied camera intrinsics")
    scale = min(width, 832) / float(width)
    encode_width = max(32, int(round(width * scale / 32)) * 32)
    encode_height = max(32, int(round(height * scale / 32)) * 32)
    if (intrinsics.get("image_width"), intrinsics.get("image_height")) != (encode_width, encode_height):
        raise ValueError("Native prediction encode dimensions disagree with official encode_w=832")
    supplied = camera["frames"][0]["intrinsics"]
    expected = {
        key: float(supplied[key]) * (
            encode_width / float(camera["image_width"]) if key in ("fx", "cx")
            else encode_height / float(camera["image_height"]))
        for key in ("fx", "fy", "cx", "cy")
    }
    for key, value in expected.items():
        if key not in intrinsics or not np.isclose(float(intrinsics[key]), value, rtol=1e-6, atol=1e-5):
            raise ValueError(f"Native prediction {key} does not match supplied calibration: expected {value}, got {intrinsics.get(key)}")


def _export_records(pred: dict, output_dir: Path, width: int, height: int) -> dict:
    import numpy as np
    import torch
    from pose_convention import POSE_CONVENTION, ace_pose_to_flat_right
    from ace_ego_hand.mano_utils import init_mano_and_renderer, mano_forward_batch_full

    models, _ = init_mano_and_renderer(device="cpu")
    means = {side: model.pose_mean.detach().cpu().numpy()[3:].copy()
             for side, model in models.items()}
    intrinsics = pred["intrinsics"]
    # Match official viz_preds.py: a supplied calibration wins; only K-free's
    # nominal placeholder is replaced by the model-fitted effective camera.
    if intrinsics.get("source") and pred.get("pred_intrinsics"):
        intrinsics = pred["pred_intrinsics"]
    native = {key: float(intrinsics[key]) * (
        width / float(intrinsics["image_width"]) if key in ("fx", "cx")
        else height / float(intrinsics["image_height"]))
        for key in ("fx", "fy", "cx", "cy")}
    if not all(np.isfinite(value) for value in native.values()):
        raise ValueError("Native prediction camera contains non-finite values")
    total = len(pred["claim"])
    counts = {"left_records": 0, "right_records": 0,
              "skipped_behind_camera": 0, "skipped_outside_image": 0}
    for start in range(0, total, 128):
        stop = min(total, start + 128)
        records: list[list[dict]] = [[] for _ in range(stop - start)]
        for slot in (0, 1):
            indices = np.flatnonzero(pred["claim"][start:stop, slot]) + start
            if len(indices) == 0:
                continue
            go = np.asarray(pred["go"][indices, slot], dtype=np.float32).reshape(-1, 3, 3)
            hp = np.asarray(pred["hp"][indices, slot], dtype=np.float32).reshape(-1, 15, 3, 3)
            betas = np.asarray(pred["betas"][indices, slot], dtype=np.float32)
            trans = np.asarray(pred["cam_trans"][indices, slot], dtype=np.float32)
            if not all(np.isfinite(array).all() for array in (go, hp, betas, trans)):
                raise RuntimeError(f"Non-finite claimed ACE hand in frames {start}:{stop}, slot {slot}")
            with torch.no_grad():
                _, verts = mano_forward_batch_full(
                    torch.from_numpy(go), torch.from_numpy(hp), torch.from_numpy(betas),
                    models[bool(slot)],
                )
            xyz = verts.numpy() + trans[:, None, :]
            for batch_index, frame_index in enumerate(indices):
                vertices = xyz[batch_index]
                if (vertices[:, 2] <= 0).any():
                    counts["skipped_behind_camera"] += 1
                    continue
                uv = vertices[:, :2] / vertices[:, 2:] * [native["fx"], native["fy"]]
                uv += [native["cx"], native["cy"]]
                lo = np.maximum(uv.min(0), [0, 0])
                hi = np.minimum(uv.max(0), [width, height])
                if (hi <= lo).any():
                    counts["skipped_outside_image"] += 1
                    continue
                orient, fingers = ace_pose_to_flat_right(
                    go[batch_index], hp[batch_index], means[bool(slot)], is_right=bool(slot),
                )
                records[int(frame_index) - start].append({
                    "is_right": bool(slot),
                    "score": float(pred["exists_2d"][frame_index, slot]),
                    "bbox": {"x0": float(lo[0]), "y0": float(lo[1]),
                             "x1": float(hi[0]), "y1": float(hi[1])},
                    "mano": {"betas": betas[batch_index].tolist(),
                             "global_orient": orient.tolist(), "hand_pose": fingers.tolist()},
                    "camera": {"cam_t": trans[batch_index].tolist(),
                               "focal_length": (native["fx"] + native["fy"]) / 2,
                               "intrinsics": native},
                    "image_size": [width, height], "source": "ace_ego_hand",
                    "pose_convention": POSE_CONVENTION,
                    "mano_pose_convention": POSE_CONVENTION,
                    "inference_implementation": INFERENCE_IMPLEMENTATION,
                })
                counts["right_records" if slot else "left_records"] += 1
        for offset, frame_records in enumerate(records):
            _write_json(output_dir / f"{start + offset:06d}.json", frame_records)
        print(f"[ACE export] {stop}/{total} frames complete", flush=True)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("video_path", "output_dir", "weights_dir", "backbone_dir",
                "mano_dir", "intrinsics_path", "cache_dir"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--chunk_frames", type=int, default=None,
                        help="Deprecated compatibility argument; official inference always uses the whole video")
    parser.add_argument("--official_root", default="/opt/ace")
    parser.add_argument("--native_prediction_path", default=None,
                        help="Explicitly import an official whole-video K-given pickle after camera/frame validation; no GPU inference")
    args = parser.parse_args()
    if args.chunk_frames is not None:
        print("[ACE] chunk_frames is ignored; using the official continuous whole-video VAE", flush=True)
    import cv2
    import yaml
    from pose_convention import POSE_CONVENTION

    output = Path(args.output_dir).resolve()
    cache = Path(args.cache_dir).resolve()
    root = Path(args.official_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video_path)
    width, height = (int(cap.get(key)) for key in
                     (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Cannot read video metadata: {args.video_path}")
    for path in output.glob("*.json"):
        if path.stem.isdigit() and int(path.stem) >= total:
            raise ValueError("output_dir contains frames from a longer video; use a fresh output_dir")

    native_dir = output / "native_predictions"
    native_dir.mkdir(exist_ok=True)
    # Metadata deliberately has no .json suffix here: existing WiLoR readers
    # use */*.json to detect per-track data, whereas this is not a hand track.
    intrinsics_path = Path(args.intrinsics_path)
    camera = _normalize_camera(json.loads(intrinsics_path.read_text()), width, height)
    (native_dir / "input_intrinsics.source").write_bytes(intrinsics_path.read_bytes())
    camera_path = native_dir / "camera.input"
    _write_json(camera_path, camera)
    caption = _prepare_assets(args, root, cache, compute_caption=not bool(args.native_prediction_path))
    inputs = build_native_provenance(
        video_path=args.video_path, weights_dir=args.weights_dir,
        backbone_dir=args.backbone_dir, mano_dir=args.mano_dir,
        intrinsics_path=intrinsics_path, caption_embed_path=caption, official_root=root,
    )
    native_path = native_dir / f"{Path(args.video_path).stem}.pkl"
    provenance_path = native_dir / "provenance.done"
    if args.native_prediction_path:
        imported_path = Path(args.native_prediction_path).resolve()
        imported = _load_native_prediction(imported_path, total)
        _validate_import_camera(imported, camera, width, height)
        imported_hash = file_sha256(imported_path)
        if imported_path != native_path:
            temporary = native_path.with_name(native_path.name + ".import.tmp")
            shutil.copyfile(imported_path, temporary)
            temporary.replace(native_path)
        provenance = write_native_provenance(
            provenance_path, inputs=inputs, prediction_path=native_path,
        )
        provenance["origin"] = "explicit_official_native_prediction_import"
        provenance["imported_from"] = {
            "path": str(imported_path), "fingerprint": file_fingerprint(imported_path),
            "sha256": imported_hash,
            "validation": "official schema, presence gating, fixed slots, framecount, K-given camera and encode dimensions",
        }
        _write_json(provenance_path, provenance)
        del imported
        print("[ACE] imported official native pickle after camera/frame validation", flush=True)
    native_is_current = _native_cache_valid(native_path, provenance_path, inputs)
    complete_path = output / "complete.done"
    expected_complete = {
        "frames": total, "pose_convention": POSE_CONVENTION,
        "inference_implementation": INFERENCE_IMPLEMENTATION,
        "video_fingerprint": inputs["video_fingerprint"],
        "intrinsics_sha256": inputs["intrinsics_sha256"],
        "export_implementation": EXPORT_IMPLEMENTATION,
        "pose_conversion_sha256": file_sha256(Path(__file__).with_name("pose_convention.py")),
    }
    if native_is_current and complete_path.is_file():
        try:
            previous = json.loads(complete_path.read_text())
            if (all(previous.get(key) == value for key, value in expected_complete.items()) and
                    previous.get("native_provenance_sha256") == file_sha256(provenance_path) and
                    previous.get("native_prediction_sha256") == file_sha256(native_path) and
                    all((output / f"{index:06d}.json").is_file() for index in range(total))):
                print("[ACE] current official predictions and converted records already complete", flush=True)
                return
        except (OSError, ValueError, TypeError):
            pass
    complete_path.unlink(missing_ok=True)
    (output / "pose_convention.done").unlink(missing_ok=True)

    options_source = root / "options/ace_ego_hand_k.yml"
    options = yaml.safe_load(options_source.read_text())
    options["paths"].update({
        "model_root": str(Path(args.backbone_dir).resolve()),
        "videox_config": str(root / "third_party/config/wan2.2/wan_civitai_5b.yaml"),
        "caption_embed": str(caption),
    })
    options_path = native_dir / "options.yml"
    options_path.write_text(yaml.safe_dump(options, sort_keys=False))
    (native_dir / "options.original.yml").write_bytes(options_source.read_bytes())
    command = [
        sys.executable, str(Path(__file__).with_name("official_inference.py")),
        "--official_root", str(root),
        "--backbone_dir", str(Path(args.backbone_dir).resolve()),
        "--decode_cache_dir", str(cache / "decoded_pixels"),
        "--video", str(Path(args.video_path).resolve()),
        "--opt", str(options_path),
        "--ckpt", str(Path(args.weights_dir).resolve() / "ace_ego_hand_k.pt"),
        "--camera", str(camera_path), "--out", str(native_dir),
        "--mode", "tiled", "--tile_w", "22", "--tap", "15",
        "--encode_w", "832", "--max_frames", "0", "--device", "cuda",
    ]
    (native_dir / "command.args").write_text(json.dumps(command, indent=2))
    if not native_is_current:
        print("[ACE] running unchanged official infer_video.py over the complete video", flush=True)
        subprocess.run(command, check=True)
        _load_native_prediction(native_path, total)
        write_native_provenance(provenance_path, inputs=inputs, prediction_path=native_path)
    else:
        print("[ACE] reusing official native pickle with matching provenance", flush=True)
    prediction = _load_native_prediction(native_path, total)
    counts = _export_records(prediction, output, width, height)
    complete = {**expected_complete, **counts,
                "mano_pose_convention": POSE_CONVENTION,
                "native_prediction": str(native_path.relative_to(output)),
                "native_prediction_sha256": file_sha256(native_path),
                "native_provenance_sha256": file_sha256(provenance_path)}
    (output / "pose_convention.done").write_text(POSE_CONVENTION + "\n")
    _write_json(complete_path, complete)


if __name__ == "__main__":
    main()
