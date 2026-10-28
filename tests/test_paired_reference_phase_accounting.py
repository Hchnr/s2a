"""Exercise actual response accounting with synthetic data; no native queries."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_paired_reference_phase import admission_inputs, synthetic_response

import s2a.prefix_preserving_trajectory_plan as planning
from s2a.paired_reference_phase import assess_phase_warm


@pytest.fixture
def cache_type(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("probe_tracking_v5_paired_reference_phase").CachedTrajectoryResponses


def test_exact_command_bytes_share_only_identical_responses(cache_type):
    actual = []

    def predict(commands):
        actual.append(commands.copy())
        return {"evaluation": len(actual)}

    cache = cache_type(predict)
    commands = np.zeros((13, 6))
    first = cache(commands)
    assert cache(commands.copy()) is first
    signed_zero = commands.copy()
    signed_zero[0, 0] = -0.0
    assert cache(signed_zero) is not first
    nearby = commands.copy()
    nearby[0, 0] = np.nextafter(0.0, 1.0)
    assert cache(nearby)["evaluation"] == 3
    assert cache.calls == 4 and cache.hits == 1 and len(cache.responses) == len(actual) == 3
    np.testing.assert_array_equal(commands, np.zeros((13, 6)))


def test_budget_counts_unique_actual_calls_and_allows_a_cached_warm_after_full_budget(cache_type):
    actual = []
    cache = cache_type(lambda commands: actual.append(commands.copy()) or {"pose": []}, max_evaluations=2)
    warm = np.zeros((13, 6))
    other = warm.copy()
    other[0, 0] = 1e-4
    first = cache(warm)
    cache(other)
    assert cache(warm) is first
    unseen = other.copy()
    unseen[0, 0] = 2e-4
    with pytest.raises(ValueError, match="fixed budget"):
        cache(unseen)
    assert len(actual) == len(cache.responses) == 2 and cache.hits == 1


@pytest.mark.parametrize("invalid", [np.zeros((12, 6)), np.full((13, 6), np.nan), np.full((13, 6), np.inf)])
def test_invalid_trajectory_never_invokes_response(cache_type, invalid):
    def forbidden(_commands):
        pytest.fail("An invalid command must not run a response")

    cache = cache_type(forbidden)
    with pytest.raises(ValueError, match="Finite thirteen"):
        cache(invalid)
    assert cache.calls == cache.hits == len(cache.responses) == 0


def test_callback_cannot_mutate_the_callers_command(cache_type):
    commands = np.zeros((13, 6))

    def mutating_predict(value):
        value[:] = 99
        return {"immutable_key": True}

    cache = cache_type(mutating_predict)
    first = cache(commands)
    np.testing.assert_array_equal(commands, np.zeros((13, 6)))
    assert cache(commands) is first and len(cache.responses) == 1


def test_preflight_and_frozen_solver_witness_charge_one_response_even_inside_buffer(cache_type, monkeypatch):
    arguments = admission_inputs(stationary=True)

    def response(commands):
        value = synthetic_response(commands)
        value["joint_velocity"][-1, 6] = 5e-11
        value["inner_joint_velocity"][-25:, 6] = 5e-11
        return value

    def forbidden_optimizer(*_args, **_kwargs):
        pytest.fail("Complete warm witness requires no optimization")

    monkeypatch.setattr(planning, "minimize", forbidden_optimizer)
    cache = cache_type(response)
    arguments["rollout"] = cache(arguments["case"]["warm"])
    preflight = assess_phase_warm(**arguments)
    assert preflight["complete_witness"] and preflight["admitted"] and preflight["safety_min"] < 1e-8
    answer = planning.solve_prefix_preserving_trajectory_response(
        cache,
        arguments["previous"],
        arguments["case"]["warm"],
        arguments["interface"],
        arguments["lower"],
        arguments["upper"],
        arguments["case"]["desired_tcp"],
        arguments["site_from_tcp"],
        past_inner_error_rad=arguments["past_inner_error_rad"],
        terminal_normals=arguments["terminal_normals"],
        max_evaluations=4000,
        max_iterations=60,
        finite_difference_step=1e-4,
    )
    assert answer["witness_found"] and answer["response_evaluations"] == len(cache.responses) == 1
    assert cache.calls == 2 and cache.hits == 1
    assert answer["total_iterations"] == 0 and answer["phases"] == []
    assert answer["used_start"].tobytes() == arguments["case"]["warm"].tobytes()


def test_finite_search_reuses_preflight_without_claiming_witness(cache_type, monkeypatch):
    arguments = admission_inputs(stationary=True)
    arguments["case"]["desired_tcp"][:, 0, 3] = 0.02
    cache = cache_type(synthetic_response)
    arguments["rollout"] = cache(arguments["case"]["warm"])
    preflight = assess_phase_warm(**arguments)
    assert preflight["admitted"] and not preflight["complete_witness"]

    def no_progress(_function, x, **_kwargs):
        return SimpleNamespace(x=x.copy(), nit=1, success=True, status=0, message="synthetic no progress")

    monkeypatch.setattr(planning, "minimize", no_progress)
    answer = planning.solve_prefix_preserving_trajectory_response(
        cache,
        arguments["previous"],
        arguments["case"]["warm"],
        arguments["interface"],
        arguments["lower"],
        arguments["upper"],
        arguments["case"]["desired_tcp"],
        arguments["site_from_tcp"],
        past_inner_error_rad=arguments["past_inner_error_rad"],
        terminal_normals=arguments["terminal_normals"],
        max_evaluations=4000,
        max_iterations=60,
        finite_difference_step=1e-4,
    )
    assert not answer["witness_found"] and answer["termination_reason"] == "no_prefix_progress"
    assert answer["response_evaluations"] == len(cache.responses) == 1
    assert cache.calls == 2 and cache.hits == 1
    assert answer["total_iterations"] == 1 and len(answer["phases"]) == 1
