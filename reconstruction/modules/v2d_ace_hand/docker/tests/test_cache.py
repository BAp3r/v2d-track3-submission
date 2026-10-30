import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from v2d.ace_hand.docker.cache import ace_output_is_current, POSE_CONVENTION, INFERENCE_IMPLEMENTATION


class CacheContractTests(unittest.TestCase):
    def test_legacy_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'complete.done').write_text(json.dumps({'frames':1}))
            (p/'000000.json').write_text('[]')
            self.assertFalse(ace_output_is_current(d))

    def test_full_cache_and_source_validation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);video=p/'clip.mp4';video.write_bytes(b'video')
            k=p/'k.json';k.write_text('{"fx":700}')
            stat=video.stat()
            marker={'frames':2,'pose_convention':POSE_CONVENTION,'inference_implementation':INFERENCE_IMPLEMENTATION,
                    'video_fingerprint':{'size':stat.st_size,'mtime_ns':stat.st_mtime_ns},
                    'intrinsics_sha256':hashlib.sha256(k.read_bytes()).hexdigest()}
            native=p/'native_predictions';native.mkdir()
            (native/'color.pkl').write_bytes(b'prediction')
            (native/'provenance.done').write_text('{}')
            pose=Path(__file__).resolve().parents[2]/'lib/pose_convention.py'
            marker.update(pose_conversion_sha256=hashlib.sha256(pose.read_bytes()).hexdigest(),
                          native_prediction='native_predictions/color.pkl',
                          native_prediction_sha256=hashlib.sha256((native/'color.pkl').read_bytes()).hexdigest(),
                          native_provenance_sha256=hashlib.sha256((native/'provenance.done').read_bytes()).hexdigest())
            (p/'complete.done').write_text(json.dumps(marker))
            for i in range(2):(p/f'{i:06d}.json').write_text('[]')
            self.assertTrue(ace_output_is_current(d,video_path=str(video),intrinsics_path=str(k)))
            (p/'000001.json').unlink()
            self.assertFalse(ace_output_is_current(d))
            (p/'000001.json').write_text('[]');k.write_text('{"fx":800}')
            self.assertFalse(ace_output_is_current(d,video_path=str(video),intrinsics_path=str(k)))
            video.write_bytes(b'changed video')
            self.assertFalse(ace_output_is_current(d,video_path=str(video)))

if __name__ == '__main__':unittest.main()
