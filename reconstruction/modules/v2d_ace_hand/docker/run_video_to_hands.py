"""ACE inference in a uv-managed container, with WiLoR-compatible outputs."""
from pathlib import Path
from v2d.docker.container import run_in_container


def run_video_to_hands(video_path: str, output_dir: str, weights_dir: str,
                       backbone_dir: str, mano_dir: str, intrinsics_path: str,
                       cache_dir: str, chunk_frames: int = 81,
                       dev: bool = False, native_prediction_path: str | None = None) -> None:
    video_path, weights_dir, backbone_dir, mano_dir, intrinsics_path = (
        str(Path(p).resolve()) for p in
        (video_path, weights_dir, backbone_dir, mano_dir, intrinsics_path)
    )
    inputs = dict(video_path=video_path, weights_dir=weights_dir,
                  backbone_dir=backbone_dir, mano_dir=mano_dir,
                  intrinsics_path=intrinsics_path)
    if native_prediction_path is not None:
        inputs["native_prediction_path"] = str(Path(native_prediction_path).resolve())
    run_in_container(
        image="v2d_ace_hand", module="video_to_hands",
        inputs=inputs,
        outputs=dict(output_dir=output_dir, cache_dir=cache_dir),
        extra_args=dict(chunk_frames=chunk_frames), gpus=True,
        extra_volumes=[f"{Path(__file__).resolve().parents[1] / 'lib'}:/adapter:ro"],
        env={"PYTHONPATH": "/adapter:/opt/ace:/opt/ace/third_party",
             "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"},
    )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('video_path', 'output_dir', 'weights_dir', 'backbone_dir',
                'mano_dir', 'intrinsics_path', 'cache_dir'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--chunk_frames', type=int, default=81, help='Deprecated compatibility option; official inference uses the whole video')
    parser.add_argument('--native_prediction_path', default=None, help='Import an official native prediction pickle for CPU export')
    parser.add_argument('--dev', action='store_true')
    run_video_to_hands(**vars(parser.parse_args()))
