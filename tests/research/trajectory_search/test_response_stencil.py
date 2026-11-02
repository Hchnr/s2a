"""Signed nominal derivatives, pose algebra and strict temporal causality."""

from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from s2a.research.trajectory_search.response_stencil import build_response_stencil


def stencil():
    def entry(commands):
        errors = commands.copy()
        squared = np.column_stack(
            (np.sum(errors[:, :3] ** 2, axis=1) / 0.01**2, np.sum(errors[:, 3:] ** 2, axis=1) / 0.05**2)
        )
        safety = np.ones(4369)
        safety[-1] = 1 + commands[-1].sum()
        return dict(commands=commands.copy(), tcp_errors=errors, normalized_squared_errors=squared, safety_slack=safety)

    base = entry(np.zeros((12, 6)))
    deltas = np.where(np.arange(72) % 2, -1e-4, 1e-4)
    probes = []
    for axis, delta in enumerate(deltas):
        command = base["commands"].copy()
        command.ravel()[axis] += delta
        probes.append(entry(command))
    return base, probes, deltas


def test_signed_nominal_derivatives_and_terminal_safety():
    base, probes, deltas = stencil()
    before = deepcopy(probes)
    model = build_response_stencil(base, probes, deltas)
    assert_array_equal(model["Je"], np.diag(np.tile([100, 100, 100, 20, 20, 20], 12)))
    assert_array_equal(model["Js"][:-1], np.zeros((4368, 72)))
    assert_allclose(model["Js"][-1, -6:], np.ones(6), atol=2e-12, rtol=0)
    assert_allclose(model["Jq"][0, :2], [1, -1], atol=2e-15, rtol=0)
    for probe, original in zip(probes, before, strict=True):
        for key in probe:
            assert_array_equal(probe[key], original[key])


def test_compact_command_preflight_without_redundant_squared_errors():
    base, probes, deltas = stencil()
    full = build_response_stencil(base, probes, deltas)
    for probe in probes:
        del probe["normalized_squared_errors"]
    compact = build_response_stencil(base, probes, deltas)
    for key in full:
        assert_array_equal(full[key], compact[key])


@pytest.mark.parametrize("fault", ["future_safety", "future_error", "extra_axis", "nominal", "pose_algebra"])
def test_invalid_or_noncausal_stencil_is_rejected(fault):
    base, probes, deltas = stencil()
    if fault == "future_safety":
        probes[6]["safety_slack"][0] += 1e-8
    elif fault == "future_error":
        probes[6]["tcp_errors"][0, 0] = 1e-4
        probes[6]["normalized_squared_errors"][0, 0] = np.sum(probes[6]["tcp_errors"][0, :3] ** 2) / 0.01**2
    elif fault == "extra_axis":
        probes[0]["commands"][0, 1] = 1e-4
    elif fault == "nominal":
        deltas[0] = 1e-3
    else:
        probes[0]["normalized_squared_errors"][0, 0] += 1e-5
    message = "Future command" if fault.startswith("future_") else None
    with pytest.raises(ValueError, match=message):
        build_response_stencil(base, probes, deltas)
