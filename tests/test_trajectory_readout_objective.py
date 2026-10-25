import numpy as np
from scipy.spatial.transform import Rotation

from s2a.trajectory_readout_objective import readout_objectives


def test_full_parameter_chain_with_nonidentity_normalization():
    rng = np.random.default_rng(42)
    x, coef = rng.normal(size=(6, 4)), rng.normal(size=(5, 10)) * 0.01
    raw = rng.normal(size=(6, 10)) * 0.1
    scale, bias = np.linspace(0.5, 3, 10), np.linspace(-0.2, 0.2, 10)
    args = (
        rng.normal(size=4),
        2.3,
        raw,
        scale,
        bias,
        Rotation.random(6, random_state=rng).as_matrix(),
        np.array([0.3, 0.1, 0.2]),
        rng.normal(size=(6, 3)) * 0.1,
        np.linspace(0.1, 0.9, 6),
        np.array([3, 3, 2, 2, 1, 1]),
    )
    losses, grads = readout_objectives(x, coef, *args)
    for objective, columns in (("position", (0, 1, 2)), ("opening", (9,))):
        for index in np.ndindex(coef.shape):
            plus, minus = coef.copy(), coef.copy()
            plus[index] += 1e-6
            minus[index] -= 1e-6
            numeric = (
                readout_objectives(x, plus, *args)[0][objective] - readout_objectives(x, minus, *args)[0][objective]
            ) / 2e-6
            np.testing.assert_allclose(grads[objective][index], numeric, atol=1e-9, rtol=1e-6)
        inactive = [k for k in range(10) if k not in columns]
        assert not grads[objective][:, inactive].any()
        assert losses[objective] > 0
