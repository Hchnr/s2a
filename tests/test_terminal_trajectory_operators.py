import numpy as np
from scipy.spatial.transform import Rotation

from s2a.terminal_trajectory_operators import operators
from s2a.trajectory_objective import opening_loss, position_loss


def test_maps_match_independent_physical_objectives_and_adjoint():
    rng = np.random.default_rng(42)
    n, width = 2, 5
    c = dict(
        phi=rng.normal(size=(n, 40, width)),
        before_rotation=Rotation.random(n * 40, random_state=rng).as_matrix().reshape(n, 40, 3, 3),
        action_scale=np.linspace(0.5, 2, 10),
        action_bias=np.linspace(-0.1, 0.1, 10),
        endpoint_weights=np.arange(40, 0, -1) / 820,
        raw_clean=rng.normal(size=(n, 40, 10)) * 0.01,
        coefficients052=rng.normal(size=(width, 10)) * 0.01,
        initial_position=rng.normal(size=(n, 3)),
        desired_positions=rng.normal(size=(n, 40, 3)),
        desired_opening=rng.uniform(size=(n, 40)),
    )
    ap, bp, ag, bg, w = operators(c)
    delta = rng.normal(size=(width, 10)) * 0.01
    for a in (ap, ag):
        x, y = rng.normal(size=a.shape[1]), rng.normal(size=a.shape[0])
        np.testing.assert_allclose(a.matvec(x) @ y, x @ a.rmatvec(y), atol=1e-10)
    physical = (c["raw_clean"] + c["phi"] @ (c["coefficients052"] + delta) - c["action_bias"]) / c["action_scale"]
    pl, gl = 0.0, 0.0
    for i in range(n):
        pl += (
            position_loss(
                physical[i, :, :3],
                c["before_rotation"][i],
                c["initial_position"][i],
                c["desired_positions"][i],
                c["endpoint_weights"],
            )[0]
            / n
            / 0.05**2
        )
        gl += opening_loss(physical[i, :, 9], c["desired_opening"][i], c["endpoint_weights"])[0] / n
    e = ap.matvec(delta[:, :3].ravel()) + bp
    np.testing.assert_allclose(e @ e, pl, atol=1e-9)
    np.testing.assert_allclose(w @ abs(ag.matvec(delta[:, 9]) + bg), gl, atol=1e-12)
