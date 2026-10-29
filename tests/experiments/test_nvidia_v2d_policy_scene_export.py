import hashlib
import json

import pytest

from experiments.nvidia_v2d.policy_scene_export import export_scenes
from experiments.nvidia_v2d.policy_view_data import Scene


def tiny_preview(tmp_path):
    mesh = tmp_path / "cube.obj"
    mesh.write_text(
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
        "v 0 0 1\nv 1 0 1\nv 1 1 1\nv 0 1 1\n"
        "f 1 3 2\nf 1 4 3\nf 5 6 7\nf 5 7 8\n"
        "f 1 2 6\nf 1 6 5\nf 2 3 7\nf 2 7 6\n"
        "f 3 4 8\nf 3 8 7\nf 4 1 5\nf 4 5 8\n"
    )
    pose = [[1, 0, 0, 2], [0, 1, 0, 3], [0, 0, 1, 4], [0, 0, 0, 1]]
    preview = tmp_path / "preview.json"
    preview.write_text(
        json.dumps(
            {
                "schema": "nvidia_v2d.sharpa_retarget_preview.v1",
                "kind": "kinematic_reference",
                "timestamps_s": [0.0, 0.05],
                "hands_left": [[[0, 0, 0], [0, 0, 1]]] * 2,
                "hands_right": [[[1, 0, 0], [1, 0, 1]]] * 2,
                "edges": [[0, 1]],
                "object_names": ["cube"],
                "object_mesh_paths": [str(mesh)],
                "object_poses_world": [[pose], [pose]],
            }
        )
    )
    return preview, mesh


def test_exports_original_mesh_and_row_major_poses_without_changing_inputs(tmp_path):
    pytest.importorskip("trimesh")
    preview, mesh = tiny_preview(tmp_path)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in [preview, mesh]}
    output = tmp_path / "scenes"
    manifests = export_scenes({"official": preview, "multitip": preview}, output)
    for name in ["official", "multitip"]:
        scene = Scene.model_validate_json((output / f"{name}.json").read_text())
        assert scene.timestamps_s == [0, 0.05]
        assert scene.kind == "kinematic_reference"
        assert scene.assistance == "unknown"
        assert scene.valid_frames == [None, None]
        assert scene.hands[0].source_codes is None
        assert scene.hands[0].valid_frames is None
        assert scene.objects[0].poses[0] == [1, 0, 0, 2, 0, 1, 0, 3, 0, 0, 1, 4, 0, 0, 0, 1]
        assert len(scene.objects[0].vertices) == 8
        assert len(scene.objects[0].faces) == 12
        assert manifests[name]["mesh_simplification"] is None
        assert (
            manifests[name]["output_sha256"]
            == hashlib.sha256((output / f"{name}.json").read_bytes()).hexdigest()
        )
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}
    with pytest.raises(FileExistsError):
        export_scenes({"official": preview}, output)


def test_rejects_frame_mismatch_and_misleading_rollout_label(tmp_path):
    pytest.importorskip("trimesh")
    preview, _ = tiny_preview(tmp_path)
    raw = json.loads(preview.read_text())
    raw["kind"] = "policy_rollout"
    preview.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="运动学"):
        export_scenes({"official": preview}, tmp_path / "invalid")
    raw["kind"] = "kinematic_reference"
    raw["object_poses_world"] = raw["object_poses_world"][:1]
    preview.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="时间轴"):
        export_scenes({"official": preview}, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_rejected_and_interpolated_quality_survives_export(tmp_path):
    pytest.importorskip("trimesh")
    preview, _ = tiny_preview(tmp_path)
    raw = json.loads(preview.read_text())
    raw.update(
        valid_left=[False, True],
        valid_right=[True, True],
        source_code_left=[4, 2],
        source_code_right=[1, 3],
    )
    preview.write_text(json.dumps(raw))
    before = preview.read_bytes()
    export_scenes({"official": preview}, tmp_path / "scenes")
    scene = Scene.model_validate_json((tmp_path / "scenes/official.json").read_text())
    assert scene.hands[0].valid_frames == [False, True]
    assert scene.hands[0].source_codes == [4, 2]
    assert scene.hands[1].source_codes == [1, 3]
    assert scene.valid_frames == [False, True]
    assert scene.hands[0].points[0] == [(0, 0, 0), (0, 0, 1)]
    assert preview.read_bytes() == before


def test_nonfinite_invalid_hand_is_hidden_without_fabricating_geometry(tmp_path):
    pytest.importorskip("trimesh")
    preview, _ = tiny_preview(tmp_path)
    raw = json.loads(preview.read_text())
    raw.update(valid_left=[False, True], source_code_left=[0, 1])
    raw["hands_left"][0][0][0] = float("nan")
    preview.write_text(json.dumps(raw))
    export_scenes({"official": preview}, tmp_path / "scenes")
    scene = Scene.model_validate_json((tmp_path / "scenes/official.json").read_text())
    assert scene.hands[0].points[0] is None
    assert scene.hands[0].source_codes == [0, 1]
    assert scene.valid_frames == [False, None]
    raw["valid_left"][0] = True
    preview.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="有效手帧"):
        export_scenes({"official": preview}, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_quality_length_mismatch_does_not_write_output(tmp_path):
    pytest.importorskip("trimesh")
    preview, _ = tiny_preview(tmp_path)
    raw = json.loads(preview.read_text())
    raw["valid_left"] = [True]
    preview.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="逐手质量"):
        export_scenes({"official": preview}, tmp_path / "invalid")
