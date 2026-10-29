import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

wp = pytest.importorskip("warp")
pytest.importorskip("flash_chord")
from hydra.utils import instantiate

from experiments.nvidia_v2d.semantic_contact_objective import (
    bind_role_map,
    make_semantic_contact_objective,
    match_semantic_contacts,
    penalize_role_alignment,
)


def arr(values, dtype=wp.float32):
    return wp.array(values, dtype=dtype, device="cpu")


def test_reward_requires_correct_surface_position_normal_and_actual_role_contact():
    # Six worlds: aligned+contact, near/no-contact, wrong surface, opposite
    # normal, aligned after global translation, and inactive/unknown reference.
    poses = np.tile([0., 0., 0., 0., 0., 0., 1.], (12, 1))
    poses[5, 0] = .10
    poses[7, 3:] = [0., 0., 1., 0.]
    poses[8:10, :3] = [2., -3., 4.]
    active = np.array([1, 0], np.int32)  # frame0 active, frame1 unknown
    score, match = arr(np.ones(6)), arr(np.zeros(6))
    wp.launch(match_semantic_contacts, dim=6, inputs=[arr(poses, wp.transform),
        arr([0, 0, 0, 0, 0, 1], wp.int32), arr([1, 0, 1, 1, 1, 1], wp.int32),
        arr(active, wp.int32), arr([1, 1], wp.int32), arr([0, 0], wp.int32),
        arr([[0., 0., 0.]] * 2, wp.vec3), arr([[1., 0., 0.]] * 2, wp.vec3),
        arr([[0., 0., 0.]] * 2, wp.vec3), arr([[1., 0., 0.]] * 2, wp.vec3),
        2, 1, 2, 0, 1, 1/(2*.02**2)], outputs=[match, score], device="cpu")
    np.testing.assert_allclose(score.numpy(), [1., .25, np.exp(-12.5), 0., 1., 1.], atol=1e-6)
    np.testing.assert_array_equal(score.numpy(), match.numpy())


def role_fixture(tmp_path):
    ref = tmp_path / "reference.parquet"; ref.write_bytes(b"immutable reference")
    path = tmp_path / "roles.npz"
    active = np.ones((5, 2), bool); active[-1] = False
    normals = np.tile([1., 0., 0.], (5, 2, 1)); normals[-1] = 0
    np.savez(path, source_time_s=np.arange(5)/2, side_names=["left", "right"], active=active,
        reference_sha256=hashlib.sha256(ref.read_bytes()).hexdigest(),
        source_contact_active=active, source_status=np.where(active, "consistent", "inactive"),
        confidence=active.astype(float),
        feasible=np.ones_like(active), robot_body_name=np.tile(["left_thumb_DP", "right_thumb_PP"], (5, 1)),
        robot_point_local=np.zeros((5, 2, 3)), robot_normal_local=normals,
        object_point_local=np.zeros((5, 2, 3)), object_normal_local=normals)
    reference = SimpleNamespace(metadata=SimpleNamespace(source_path=str(ref), source_fps=2.), fps=2., num_frames=8)
    model = SimpleNamespace(body_count=3, body_label=["left/left_thumb_DP", "right/right_thumb_PP", "pot/object"],
        shape_body=SimpleNamespace(numpy=lambda: np.array([0, 1, 2])))
    embodiment = SimpleNamespace(hand=lambda side: SimpleNamespace(contact_links=[SimpleNamespace(shape_ids=[0 if side == "left" else 1])]))
    contact = SimpleNamespace(layout=SimpleNamespace(num_objects=1, sides=("right", "left"), hand_slot_starts=(0, 1)))
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    return path, digest(path), reference, digest(ref), .5, model, embodiment, contact, 1


def test_role_binding_uses_names_and_actual_right_first_slot_order(tmp_path):
    bound = bind_role_map(*role_fixture(tmp_path))
    assert bound["active"].shape == (8, 2)
    np.testing.assert_array_equal(bound["slot_ids"][0], [1, 0])
    np.testing.assert_array_equal(bound["body_ids"][0], [0, 1])
    assert not bound["active"][-1].any()


def test_role_binding_rejects_changed_map_or_reference(tmp_path):
    args = role_fixture(tmp_path)
    args[0].write_bytes(b"changed")
    with pytest.raises(ValueError, match="map hash"):
        bind_role_map(*args)


@pytest.mark.parametrize("bad", ["pairing", "missing_feasible", "unknown_active"])
def test_internally_unverified_role_map_is_rejected_even_with_fresh_file_hash(tmp_path, bad):
    args = list(role_fixture(tmp_path))
    data = dict(np.load(args[0]))
    if bad == "pairing":
        data["reference_sha256"] = np.asarray("c"*64)
    elif bad == "missing_feasible":
        del data["feasible"]
    else:
        data["source_status"][0, 0] = "unknown"
    np.savez(args[0], **data)
    args[1] = hashlib.sha256(args[0].read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        bind_role_map(*args)


def test_hydra_roundtrip_retains_semantic_and_release_contract():
    cfg = instantiate({"_target_": "experiments.nvidia_v2d.semantic_contact_objective.make_semantic_contact_objective",
        "release_start_frame": 407, "semantic_map_path": "/roles.npz", "semantic_map_sha256": "a"*64,
        "semantic_reference_sha256": "b"*64, "semantic_motion_speed": .5, "semantic_sigma_m": .02})
    assert cfg.release_start_frame == 407 and cfg.semantic_sigma_m == .02
    with pytest.raises(ValueError, match="hashed"):
        make_semantic_contact_objective(release_start_frame=407)


def test_alignment_cost_learns_far_role_without_wrench_and_ignores_unknowns():
    poses = np.tile([0., 0., 0., 0., 0., 0., 1.], (18, 1))
    poses[[2, 5, 8, 11], 0] = [.115, .065, .006, 1.]
    poses[17, 3:] = [0., 0., 1., 0.]
    source_poses = poses.copy()
    normal = np.tile([1., 0., 0.], (4, 1)); normal[2:] = np.nan
    point = np.zeros((4, 3)); point[2:] = np.nan
    obj_point = np.tile([.001, 0., 0.], (4, 1)); obj_point[2:] = np.nan
    score, cost = arr([0., 0., 0., 0., .7, 0.]), arr(np.zeros(6))
    body_pose = arr(poses, wp.transform)
    wp.launch(penalize_role_alignment, dim=6, inputs=[body_pose,
        arr([0, 0, 0, 0, 1, 0], wp.int32), arr([1, 1, 0, 0], wp.int32),
        arr([1, 2, 99999, 99999], wp.int32), arr(point, wp.vec3), arr(normal, wp.vec3),
        arr(obj_point, wp.vec3), arr(normal, wp.vec3), 2, 2, 3, 0, .25],
        outputs=[cost, score], device="cpu")
    np.testing.assert_allclose(cost.numpy(), [1.09, .59, 0., 2.94, 0., .5], atol=1e-6)
    np.testing.assert_allclose(score.numpy(), [-.2725, -.1475, 0., -.735, .7, -.125], atol=1e-6)
    assert score.numpy()[0] < score.numpy()[1] < score.numpy()[2]
    np.testing.assert_array_equal(body_pose.numpy(), source_poses.astype(np.float32))


def test_alignment_weight_defaults_to_legacy_zero_and_is_bounded():
    fields = {"release_start_frame": 407, "semantic_map_path": "/roles.npz",
        "semantic_map_sha256": "a"*64, "semantic_reference_sha256": "b"*64}
    assert make_semantic_contact_objective(**fields).semantic_alignment_weight == 0.
    assert make_semantic_contact_objective(**fields, semantic_alignment_weight=.25).semantic_alignment_weight == .25
    for weight in (-.1, 1.1, float('nan')):
        with pytest.raises(ValueError, match="alignment weight"):
            make_semantic_contact_objective(**fields, semantic_alignment_weight=weight)
        with pytest.raises(ValueError, match="wrist command weight"):
            make_semantic_contact_objective(**fields, semantic_wrist_command_weight=weight)
    cfg = make_semantic_contact_objective(**fields, semantic_wrist_command_weight=.25)
    assert cfg.semantic_wrist_command_deadzone_m == .02
    for deadzone in (-.1, .11, float('nan')):
        with pytest.raises(ValueError, match="wrist command deadzone"):
            make_semantic_contact_objective(**fields, semantic_wrist_command_deadzone_m=deadzone)
