import numpy as np
import pytest

from experiments.nvidia_v2d.collision_contact_reference import (
    ConvexUnion,
    minimum_smooth_translation,
    refit_surface_role,
    repair_inside_targets,
)


def cube_data(offset=(0, 0, 0), shape=1):
    vertices = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]], float)
    return {f"shape_{shape}_vertices": vertices + offset}


def test_hidden_interface_is_not_a_union_boundary():
    union = ConvexUnion({**cube_data(), **cube_data((1, 0, 0), 2)})
    point = np.array([0.9, 0, 0])
    assert union.inside(point[None])[0]
    # The closest face of piece 1 (x=1) is hidden in piece 2.
    with pytest.raises(ValueError, match="No exposed"):
        union.exposed_nearest(point)
    assert union.ray_exit(point, [1, 0, 0]) == pytest.approx(1.1)
    lower, upper, _, _ = union.interior_bounds(point[None])
    assert lower[0] <= 1 <= upper[0]


def test_exposed_face_distance_and_outside_offset_are_certified():
    union = ConvexUnion(cube_data())
    point = np.array([0.9, 0.2, 0])
    result = union.exposed_nearest(point)
    assert result.exact and result.distance == pytest.approx(0.1)
    np.testing.assert_allclose(result.point, [1, 0.2, 0])
    assert not union.inside((result.point + .001 * result.outward)[None])[0]


def test_repair_never_changes_trusted_mask_or_unknown_targets():
    roles = {"active": np.array([[True, False]]), "object_point_local": np.array([[[.9, 0, 0], [.9, 0, 0]]]),
             "object_normal_local": np.array([[[-1., 0, 0], [-1., 0, 0]]]),
             "source_status": np.array([["consistent", "unknown"]]),
             "source_contact_active": np.ones((1, 2), bool), "source_slot": np.zeros((1, 2), int),
             "confidence": np.ones((1, 2)), "feasible": np.ones((1, 2), bool)}
    result, rows = repair_inside_targets(roles, ConvexUnion(cube_data()))
    assert len(rows) == 1
    np.testing.assert_array_equal(result["active"], roles["active"])
    np.testing.assert_array_equal(result["prior_object_point_local"], roles["object_point_local"])
    np.testing.assert_array_equal(result["object_point_local"][0, 1], roles["object_point_local"][0, 1])
    np.testing.assert_allclose(result["object_point_local"][0, 0], [1, 0, 0])
    with pytest.raises(ValueError, match="Already repaired"):
        repair_inside_targets(result, ConvexUnion(cube_data()))


def test_smooth_translation_enforces_clearance_and_preserves_release_exactly():
    constraints = [(8, np.array([1., 0, 0]), .005), (10, np.array([0., 1, 0]), .003)]
    delta, diagnostic = minimum_smooth_translation(30, constraints, release_frame=20)
    assert diagnostic["violation_m"] <= 2e-7
    assert delta[8, 0] >= .005 - 2e-7 and delta[10, 1] >= .003 - 2e-7
    np.testing.assert_array_equal(delta[20:], 0)
    np.testing.assert_array_equal(delta[0], 0)
    assert np.linalg.norm(np.diff(delta, axis=0), axis=1).max() < .002
    with pytest.raises(ValueError, match="fixed"):
        minimum_smooth_translation(30, [(20, np.ones(3), .01)], release_frame=20)


def test_surface_role_refit_obeys_every_control_state_without_moving_the_target():
    vertices = np.array([[-.02, -.02, 0], [.02, -.02, 0], [0, .02, 0]])
    faces, normals = np.array([[0, 1, 2]]), np.array([[0., 0, 1]])
    matrices = np.repeat(np.eye(3)[None], 2, axis=0)
    translations = np.array([[-.004, 0, 0], [.004, 0, 0]])
    result = refit_surface_role(vertices, faces, normals, matrices, translations,
                               np.zeros(3), [0, 0, 1], ConvexUnion(cube_data((10, 0, 0))),
                               np.array([.004, 0, 0]))
    assert result is not None and result["max_error_m"] <= .006
    assert result["point"][2] == pytest.approx(0)
    translations *= 3
    assert refit_surface_role(vertices, faces, normals, matrices, translations,
                              np.zeros(3), [0, 0, 1], ConvexUnion(cube_data((10, 0, 0))),
                              np.zeros(3)) is None
