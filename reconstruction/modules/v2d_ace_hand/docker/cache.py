"""Validate the ACE export contract before reusing downstream artifacts."""
import hashlib
import json
from pathlib import Path

POSE_CONVENTION = 'absolute_axis_angle_flat_mean_v2'
INFERENCE_IMPLEMENTATION = 'official_whole_video_tiled_v1'


def ace_output_is_current(output_dir: str, *, video_path: str | None = None,
                          intrinsics_path: str | None = None) -> bool:
    root = Path(output_dir)
    try:
        marker = json.loads((root / 'complete.done').read_text())
        count = int(marker['frames'])
        if (marker.get('pose_convention') != POSE_CONVENTION or
                marker.get('inference_implementation') != INFERENCE_IMPLEMENTATION or count <= 0):
            return False
        pose_source = Path(__file__).resolve().parents[1] / 'lib' / 'pose_convention.py'
        if marker.get('pose_conversion_sha256') != hashlib.sha256(pose_source.read_bytes()).hexdigest():
            return False
        native_relative = Path(marker['native_prediction'])
        if native_relative.is_absolute() or '..' in native_relative.parts:
            return False
        native = root / native_relative
        provenance = root / 'native_predictions' / 'provenance.done'
        if marker.get('native_prediction_sha256') != hashlib.sha256(native.read_bytes()).hexdigest():
            return False
        if marker.get('native_provenance_sha256') != hashlib.sha256(provenance.read_bytes()).hexdigest():
            return False
        if video_path is not None:
            stat = Path(video_path).stat()
            if marker.get('video_fingerprint') != {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}:
                return False
        if intrinsics_path is not None:
            digest = hashlib.sha256(Path(intrinsics_path).read_bytes()).hexdigest()
            if marker.get('intrinsics_sha256') != digest:
                return False
        return all((root / f'{index:06d}.json').is_file() for index in range(count))
    except (OSError, ValueError, TypeError, KeyError):
        return False


def ace_hand_contract_fingerprint(output_dir: str) -> str:
    """Identify geometry consumed by every dependent hand-stage cache."""
    marker = json.loads((Path(output_dir) / 'complete.done').read_text())
    fields = ('pose_convention', 'inference_implementation', 'intrinsics_sha256',
              'native_prediction_sha256', 'pose_conversion_sha256', 'export_implementation')
    contract = {key: marker[key] for key in fields}
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
