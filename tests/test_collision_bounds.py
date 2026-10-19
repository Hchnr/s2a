import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from s2a.collision_bounds import separation_bound
from s2a.geometry import transform


def test_rotating_and_sliding_geometry_bound_is_conservative():
    rotation = Rotation.from_rotvec([[0, 0, 0], [0, 0, 2 * np.pi / 3]])
    positions = np.array([[0, 0, 0], [0.1, 0, 0]])
    poses = transform(positions, rotation.as_matrix())
    centers = np.array([[0.1, 0, 0], [0.3, 0, 0]])
    geometry = {"finger": dict(corner_centers=centers, geom_rbound_m=0.02)}
    object_center, object_radius = np.array([0.4, 0.4, 0]), 0.04
    bounds = separation_bound(poses, object_center, object_radius, geometry)
    # Check a physical swept-volume case: a moving/sliding sphere rotating
    # through a large arc, including intermediate q and pose interpolation.
    times = np.linspace(0, 1, 201)
    interpolated = Slerp([0, 1], rotation)(times).as_matrix()
    minimum = np.inf
    for opening in np.linspace(0, 1, 51):
        local = centers[0] * (1 - opening) + centers[1] * opening
        world = interpolated @ local + positions[0] * (1 - times[:, None]) + positions[1] * times[:, None]
        minimum = min(minimum, np.min(np.linalg.norm(world - object_center, axis=1)) - 0.06)
    assert bounds["linear_slerp_lower_bound_m"] <= minimum + 1e-12
    # Endpoint-only clearance cannot certify this rotating swept volume.
    assert bounds["linear_slerp_lower_bound_m"] < bounds["endpoint_lower_bound_m"]


def test_positive_bound_certifies_a_distant_object_for_every_opening():
    poses = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    geometries = {"pad": dict(corner_centers=[[0, 0, 0], [0.04, 0, 0]], geom_rbound_m=0.01)}
    bounds = separation_bound(poses, [0.2, 0, 0], 0.05, geometries)
    np.testing.assert_allclose(list(bounds.values()), 0.1, atol=1e-14)
