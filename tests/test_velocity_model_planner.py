import itertools

import numpy as np
import pytest

from s2a.velocity_model_planner import VelocityModelPlanner, fit_velocity_envelope


def parameters():
    p = dict(
        velocity_units="normalized_opening_per_20Hz_period",
        prior="constant_current_velocity",
        mean=[0.0] * 4,
        scale=[1.0] * 4,
        coefficients=np.zeros((15, 2)),
    )
    # A delayed toy plant: q_next=q+V, V_next=m_next.
    p["coefficients"][2, 1] = 1
    p["coefficients"][3, 1] = -1
    return p


def test_delayed_state_response_requires_multi_step_command_choice():
    planner = VelocityModelPlanner(parameters(), velocity_envelope=1)
    assert planner.plan(0.5, [0], 0, [0.5]).command == 0
    result = planner.plan(0.5, [0, 0], 0, [0.5, 0.75])
    assert result.sequence == (-1.0, 0.0)
    np.testing.assert_allclose(result.predicted_openings, [0.5, 0.75], atol=1e-12)
    assert result.cost == pytest.approx(0.0001)


def test_action_ties_follow_registered_lexicographic_order():
    p = parameters()
    p["coefficients"][2, 1] = 0
    p["coefficients"][11, 1] = 4
    result = VelocityModelPlanner(p, velocity_envelope=1).plan(0.5, [0], 0, [0.5, 0.75])
    assert result.sequence == (-1.0, 0.0)


def test_breadth_first_search_matches_independent_full_sequence_enumeration():
    reference = [0.47, 0.52, 0.44, 0.36, 0.5, 0.7, 0.6, 0.5]
    actual = VelocityModelPlanner(parameters(), velocity_envelope=0.6).plan(0.4, [0], 0, reference)
    valid = []
    for sequence in itertools.product((0.0, -1.0, 1.0), repeat=8):
        q = 0.4
        v = 0.0
        m = 0.0
        cost = 0.0
        supported = True
        for desired, u in zip(reference, sequence, strict=True):
            for _ in range(25):
                m = max(-1.0, min(1.0, m - 0.01 * u))
            nextq = max(0.0, min(1.0, q + v))
            nextv = v + (m - v)
            if abs(nextv) > 0.6:
                supported = False
                break
            cost += (nextq - desired) ** 2 + 0.0001 * u * u
            q, v = nextq, nextv
        if supported:
            valid.append((cost, sequence))
    best = min(range(len(valid)), key=lambda i: valid[i][0])
    assert actual.sequence == valid[best][1]
    assert actual.cost == pytest.approx(valid[best][0], abs=1e-12)
    assert actual.valid_sequences == len(valid)
    assert actual.rejected_velocity_sequences == 3**8 - len(valid)


def test_short_reference_and_unused_generated_tail_are_causal():
    planner = VelocityModelPlanner(parameters(), velocity_envelope=1)
    assert planner.plan(0.5, [0], 0, [0.5]).horizon == 1
    first = planner.plan(0.5, [0], 0, [0.5] * 8 + [0.0] * 32)
    second = planner.plan(0.5, [0], 0, [0.5] * 8 + [1.0] * 32)
    assert first == second
    assert first.reference_available_count == 40


def test_velocity_envelope_uses_fit_current_and_next_state_only():
    raw = np.array([[0.2, 0, 0.25, 0.1], [0.3, 0.25, 0.25, -0.3]])
    target = np.array([[0, 0.1], [0, 0.1]])
    assert fit_velocity_envelope(raw, target) == pytest.approx(0.45)


def test_unsupported_hypotheses_are_rejected_without_clipping():
    p = parameters()
    p["coefficients"][-1, 1] = 2
    with pytest.raises(ValueError, match="No finite supported"):
        VelocityModelPlanner(p, velocity_envelope=0.5).plan(0.5, [0], 0, [0.5] * 8)
    p["coefficients"][:] = 1e308
    with pytest.raises(ValueError, match="No finite supported"):
        VelocityModelPlanner(p, velocity_envelope=1).plan(0.8, [1, -1], 0.4, [0.5] * 8)


@pytest.mark.parametrize(
    "cache,velocity,reference",
    [([0, 0.1], 0, [0.5]), ([1.1], 0, [0.5]), ([0], np.nan, [0.5]), ([0], 2, [0.5]), ([0], 0, []), ([0], 0, [1.1])],
)
def test_invalid_measured_state_or_generated_reference_rejected(cache, velocity, reference):
    with pytest.raises(ValueError):
        VelocityModelPlanner(parameters(), velocity_envelope=1).plan(0.5, cache, velocity, reference)
