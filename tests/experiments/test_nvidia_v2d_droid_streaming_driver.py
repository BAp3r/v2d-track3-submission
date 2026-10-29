"""Optional whole-driver equivalence checks; synthetic CPU objects, no GPU imports."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
import subprocess
import sys
import types
import weakref
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

PROFILE = Path(__file__).resolve().parents[2] / "experiments/nvidia_v2d/proposals"


class Tensor:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def detach(self) -> Tensor:
        return self

    def cpu(self) -> Tensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.array


def resize_nearest(array: np.ndarray, size: tuple[int, int], **_kwargs: Any) -> np.ndarray:
    width, height = size
    y = np.arange(height) * array.shape[0] // height
    x = np.arange(width) * array.shape[1] // width
    return array[y[:, None], x]


def _driver(source: str, namespace: dict[str, Any]) -> Callable[..., None]:
    names = {"_get_keyframe_disps", "_solve_global_scale", "run_droid_slam"}
    nodes = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    tree = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), "<isolated-droid-driver>", "exec"), namespace)
    return namespace["run_droid_slam"]


@pytest.fixture
def sources(tmp_path: Path) -> tuple[str, str]:
    source_path = os.environ.get("NVIDIA_V2D_DROID_BASE_SOURCE")
    if not source_path:
        pytest.skip("Set NVIDIA_V2D_DROID_BASE_SOURCE for optional pinned-source CPU comparison")
    source = Path(source_path).read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == (
        "7faf3693a306dc7fedaced69754842ab033d71d95e116c0e7ea43a4fb1d03a22"
    ), "The supplied driver must match the reviewed upstream pin"
    target = tmp_path / "reconstruction/modules/v2d_droid_slam/lib/_slam.py"
    target.parent.mkdir(parents=True)
    target.write_text(source)
    subprocess.run(
        ["git", "apply", "--check", str(PROFILE / "droid-streaming.patch")],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "apply", str(PROFILE / "droid-streaming.patch")],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    return source, target.read_text()


def _run_variant(
    source: str,
    directory: Path,
    upsample: bool,
    has_reference: bool,
    helper: Any,
) -> dict[str, Any]:
    directory.mkdir()
    weights = directory / "droid.pth"
    weights.write_bytes(b"synthetic")
    indices = [3, 7, 11, 13, 19, 29]
    keyframes = np.array([3, 13, 29], dtype=np.float32)
    disps = np.stack([np.full((8, 8), value, dtype=np.float32) for value in (0.5, 1.0, 2.0)])
    images: list[weakref.ReferenceType[np.ndarray]] = []
    outputs: dict[str, Any] = {"poses": {}, "depth": {}, "ply": None}
    pass_count = 0

    def frames() -> Iterator[Any]:
        nonlocal pass_count
        pass_count += 1
        for i, index in enumerate(indices):
            image = np.zeros((16, 24, 3), dtype=np.uint8)
            if pass_count == 1:
                images.append(weakref.ref(image))
            intrinsics = types.SimpleNamespace(fx=20.0, fy=21.0, cx=12.0, cy=8.0, width=24, height=16)
            reference = np.full((16, 24), 2.0 + i, dtype=np.float32) if has_reference else None
            yield types.SimpleNamespace(
                frame_idx=index,
                image_bgr=image,
                intrinsics=intrinsics,
                prior_depth=None,
                align_to_depth=reference,
            )

    class FakeDroid:
        def __init__(self, _args: Any) -> None:
            self.video = types.SimpleNamespace(
                counter=types.SimpleNamespace(value=3),
                disps=Tensor(disps),
                disps_up=Tensor(np.repeat(np.repeat(disps, 2, axis=1), 2, axis=2)),
                tstamp=Tensor(keyframes),
            )

        def track(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def terminate(self, replay: Iterator[Any]) -> np.ndarray:
            outputs["retained_rgb_arrays_at_terminate"] = sum(ref() is not None for ref in images)
            assert [item[0] for item in replay] == indices
            return np.array([[i * 0.1, i * 0.2, i * 0.3, 0, 0, 0, 1] for i in range(len(indices))])

    class Transform:
        def __init__(self, matrix: np.ndarray) -> None:
            self.matrix = matrix

        @classmethod
        def from_matrix(cls, matrix: np.ndarray) -> Transform:
            return cls(matrix.copy())

        def save(self, path: str) -> None:
            outputs["poses"][Path(path).name] = self.matrix.copy()

    class Depth:
        def __init__(self, *, depth: np.ndarray) -> None:
            self.depth = depth

        def to_pil_image(self) -> Depth:
            return self

        def save(self, path: str) -> None:
            outputs["depth"][Path(path).name] = self.depth.copy()

    def ply(_droid: Any, scale: float, path: str, *, min_views: int) -> None:
        outputs["ply"] = scale, Path(path).name, min_views

    namespace = {
        "os": os,
        "np": np,
        "SimpleNamespace": types.SimpleNamespace,
        "cv2": types.SimpleNamespace(resize=resize_nearest, INTER_LINEAR=1),
        "_load_droid_class": lambda: FakeDroid,
        "_resize_for_droid": lambda *_a, **_k: (np.zeros(1), np.zeros(4), None),
        "Transform3d": Transform,
        "DepthImage": Depth,
        "_write_pointcloud_ply": ply,
        "streamed_global_scale": helper.streamed_global_scale,
    }
    run = _driver(source, namespace)
    trajectory = directory / "trajectory.txt"
    run(
        frames,
        str(weights),
        str(directory / "poses"),
        depth_folder=str(directory / "depth"),
        trajectory_path=str(trajectory),
        pointcloud_path=str(directory / "cloud.ply"),
        upsample=upsample,
        pointcloud_min_views=3,
    )
    outputs["trajectory"] = trajectory.read_text()
    outputs["passes"] = pass_count
    return outputs


@pytest.mark.parametrize("upsample,has_reference", [(False, True), (True, True), (False, False)])
def test_streamed_driver_preserves_all_outputs_and_releases_frames(
    sources: tuple[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    upsample: bool,
    has_reference: bool,
) -> None:
    # Original and candidate execute the same source functions with deterministic
    # CPU stand-ins. Geometry/threshold mathematics in the scale helper is real NumPy.
    spec = importlib.util.spec_from_file_location(
        "proposal_streaming_driver", PROFILE / "droid_streaming.py"
    )
    assert spec and spec.loader
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    rotation = types.SimpleNamespace(
        from_quat=lambda _q: types.SimpleNamespace(as_matrix=lambda: np.eye(3))
    )
    monkeypatch.setitem(
        sys.modules, "scipy.spatial.transform", types.SimpleNamespace(Rotation=rotation)
    )
    results = [
        _run_variant(source, tmp_path / f"variant-{variant}", upsample, has_reference, helper)
        for variant, source in enumerate(sources)
    ]
    baseline, streamed = results
    assert baseline["poses"].keys() == streamed["poses"].keys()
    assert baseline["depth"].keys() == streamed["depth"].keys()
    for key in baseline["poses"]:
        np.testing.assert_array_equal(baseline["poses"][key], streamed["poses"][key])
    for key in baseline["depth"]:
        np.testing.assert_array_equal(baseline["depth"][key], streamed["depth"][key])
        assert streamed["depth"][key].shape == (16, 24)
    assert baseline["trajectory"] == streamed["trajectory"]
    assert baseline["ply"] == streamed["ply"]
    assert baseline["retained_rgb_arrays_at_terminate"] == 6
    assert streamed["retained_rgb_arrays_at_terminate"] == 0
    assert baseline["passes"] == 2
    assert streamed["passes"] == (3 if has_reference else 2)
