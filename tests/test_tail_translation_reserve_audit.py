"""Independent ledger tests with analytic saved arrays, never robot dynamics."""

import ast
import importlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from test_prefix_optimization_reserve_plan import inputs

from s2a.directional_derivative_probe import probe_directional_derivatives
from s2a.tail_translation_reserve import build_tail_translation_grid, probe_tail_translation_reserve
from s2a.terminal_secant_probe import (
    build_terminal_secant_geometry,
    fit_terminal_secant_row,
    probe_terminal_secant,
)


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("tail_translation_reserve_audit")


def response(commands, *, terminal=None):
    """A deterministic analytic fixture; arrays are not claimed to be physics."""
    poses = np.tile(np.eye(4), (12, 1, 1))
    poses[:, :3, 3] = commands[:, :3] * 0.1
    joint = np.c_[commands * 0.01, np.zeros(12)]
    velocity = np.zeros((12, 7))
    value = 1.32e-8 + 0.1 * commands[-1, 0] if terminal is None else terminal
    velocity[-1, 6] = -value
    inner_q, inner_v = np.repeat(joint, 25, axis=0), np.repeat(velocity, 25, axis=0)
    periods = []
    for i in range(12):
        periods.append(
            dict(
                joint_position=inner_q[i * 25 : (i + 1) * 25].copy(),
                joint_velocity=inner_v[i * 25 : (i + 1) * 25].copy(),
                requested_torque=np.zeros((25, 7)),
                applied_torque=np.zeros((25, 7)),
                clipped=np.zeros((25, 7), dtype=bool),
                mass_change=np.zeros(25),
                jacobian_change=np.zeros(25),
                goal=poses[i].copy(),
                endpoint_pose=poses[i].copy(),
                endpoint_twist=np.zeros(6),
                update_matrices=True,
                dynamics_queries=26,
            )
        )
    return dict(
        inner_joint_position=inner_q,
        inner_joint_velocity=inner_v,
        joint_position=joint,
        joint_velocity=velocity,
        pose=poses,
        twist=np.zeros((12, 6)),
        goals=poses.copy(),
        periods=periods,
    )


@pytest.fixture(scope="module")
def source():
    """Use frozen cores only to generate an internally consistent old archive."""
    args = inputs(response, horizon=12)
    anchor = args.pop("warm")
    args["desired"][0, 0, 3] = 0.01  # Exact original cap; no isclose allowance.
    args["desired"][9:, 0, 3] = 0.02
    args["terminal_normals"] = np.eye(7)[6:]
    directions = np.zeros((3, 12, 6))
    directions[np.arange(3), 11, np.arange(3)] = 1
    old = probe_directional_derivatives(**args, anchor=anchor, directions=directions)
    history = old["evaluated_candidates"][0]
    training = [
        next(
            r
            for r in old["request_ledger"]
            if r["role"] == "grid" and r["source_stage"] == 24 and r["sign"] == 1 and r["step"] == h
        )
        for h in (0.001, 0.0005)
    ]
    training = [old["evaluated_candidates"][r["response_index"]] for r in training]
    original_row = np.zeros(72)
    original_row[66] = 0.0999
    fit = fit_terminal_secant_row(
        anchor,
        directions[0],
        original_row,
        history["safety_slack"][-1],
        np.asarray([r["commands"] for r in training]),
        np.asarray([r["safety_slack"][-1] for r in training]),
    )
    geometry = build_terminal_secant_geometry(anchor, args["previous"], -np.ones(6), np.ones(6), directions)
    prior = probe_terminal_secant(**args, anchor=anchor, geometry=geometry, fit=fit, historical_anchor=history)
    e17 = max(abs(row["original_residual"]) for row in prior["predictions"])
    assert e17 > 0
    return args, anchor, prior, e17


def package(source, *, variant=None):
    args, anchor, prior, e17 = deepcopy(source)
    calls = []

    def predict(command):
        calls.append(command.copy())
        terminal = None
        if variant == "no_improvement":
            terminal = 1.32e-8
        elif variant == "ties" and np.any(command != anchor):
            terminal = 0.0002
        elif variant == "reserve_boundary" and np.any(command != anchor):
            terminal = 1e-8 + 10 * e17 - 1e-13
        result = response(command, terminal=terminal)
        if variant == "small_scale_only" and command[-1, 0] > 0.00075:
            result["joint_position"][-1, 0] = 2.0
            result["inner_joint_position"][-25:, 0] = 2.0
            result["periods"][-1]["joint_position"][:, 0] = 2.0
        if len(calls) == 2:
            if variant == "prefix_period":
                result["periods"][10]["requested_torque"][24, 0] = 1e-15
            elif variant == "prefix_top":
                result["inner_joint_position"][274, 0] = 1e-15
            elif variant == "rotation":
                c, s = np.cos(1e-5), np.sin(1e-5)
                result["periods"][-1]["goal"][:2, :2] = [[c, -s], [s, c]]
            elif variant == "tail_clipping":
                result["periods"][-1]["clipped"][0, 0] = True
        if variant == "repeat" and len(calls) == 14:
            result["periods"][-1]["requested_torque"][0, 0] += 1e-15
        if variant == "history":
            result["goals"][0, 0, 3] += 1e-15
            result["periods"][0]["goal"][0, 3] += 1e-15
        return result

    args["predict"] = predict
    history = prior["evaluated_candidates"][0]
    grid = build_tail_translation_grid(anchor, args["previous"], -np.ones(6), np.ones(6))
    answer = probe_tail_translation_reserve(
        **args, anchor=anchor, grid=grid, historical_anchor=history, prior_max_abs_residual=e17
    )
    return answer, history, prior, calls


def check(audit, package):
    answer, history, prior, _ = package
    return audit.audit_tail_translation_ledger(answer, historical_anchor=history, prior_answer=prior)


def test_full14_source_error_and_json_roundtrip_restore_without_full_witness(audit, source):
    saved = package(source)
    answer, _, prior, calls = saved
    result = check(audit, saved)
    assert result["passed"] and result["improvement_found"] and result["sufficient_reserve_restored"]
    assert len(calls) == result["response_evaluations"] == 14 and result["unique_command_count"] == 13
    assert result["selected_response_index"] == 1 and result["witness_indices"] == []
    assert answer["selected"]["prefix_length"] == 9 and not answer["selected"]["feasible"]
    assert result["causal_checks_all_passed"] and not result["control_qualified"]
    assert not result["full_responses_independently_replayed_here"]
    e17 = audit.audit_tail_translation_prior_residuals(prior)
    assert e17["prior_grid_responses_checked"] == 15 and e17["request_indices"] == list(range(1, 16))
    assert e17["prior_max_abs_residual"] == source[-1]
    loaded = json.loads(json.dumps(saved[:3], default=lambda value: value.tolist()))
    replay = audit.audit_tail_translation_ledger(loaded[0], historical_anchor=loaded[1], prior_answer=loaded[2])
    assert replay["selected_response_index"] == 1 and replay["sufficient_reserve_restored"]


@pytest.mark.parametrize("variant", ["prefix_period", "prefix_top", "rotation"])
def test_exact_causal_failures_are_local_and_cannot_be_relabelled(audit, source, variant):
    saved = package(source, variant=variant)
    result = check(audit, saved)
    assert result["passed"] and not result["causal_checks_all_passed"]
    assert not saved[0]["candidate_checks"][0]["restoration_eligible"]
    assert result["selected_response_index"] == 3
    forged = deepcopy(saved)
    forged[0]["candidate_checks"][0]["restoration_eligible"] = True
    with pytest.raises(ValueError):
        check(audit, forged)


@pytest.mark.parametrize("variant", ["history", "repeat"])
def test_history_and_repeat_gate_every_candidate_without_retry(audit, source, variant):
    saved = package(source, variant=variant)
    result = check(audit, saved)
    assert len(saved[-1]) == 14 and result["passed"]
    assert not result["improvement_found"] and result["selected_response_index"] is None
    assert result["eligible_ranking"] == [] and not result["sufficient_reserve_restored"]


def test_last_period_translation_or_clip_changes_do_not_invalidate_causal_prefix(audit, source):
    saved = package(source, variant="tail_clipping")
    result = check(audit, saved)
    assert result["selected_response_index"] == 1 and result["causal_checks_all_passed"]
    assert saved[0]["clipping_checks"][1]["changed_count"] == 1
    baseline, changed = (saved[0]["evaluated_candidates"][i]["rollout"] for i in (0, 1))
    assert not np.array_equal(baseline["goals"][-1, :3, 3], changed["goals"][-1, :3, 3])
    assert audit.compare_tail_causal_rollouts(baseline, changed, 1)["passed"]


def test_larger_step_unsafe_smaller_step_eligible_and_unchanged_no_improvement(audit, source):
    saved = package(source, variant="small_scale_only")
    result = check(audit, saved)
    assert result["selected_response_index"] == 3
    assert "joint_audit" in saved[0]["candidate_checks"][0]["rejection_reasons"]
    unchanged = package(source, variant="no_improvement")
    result = check(audit, unchanged)
    assert result["eligible_ranking"] and result["selected_response_index"] is None
    assert unchanged[0]["selected"] is None and not result["improvement_found"]


def test_ties_use_actual_linf_then_request_index_and_reject_forged_order(audit, source):
    saved = package(source, variant="ties")
    result = check(audit, saved)
    assert result["selected_response_index"] == 3
    assert result["eligible_ranking"][:6] == [3, 4, 7, 8, 11, 12]
    saved[0]["eligible_ranking"][:2] = [4, 3]
    with pytest.raises(ValueError, match="ranking"):
        check(audit, saved)


@pytest.mark.parametrize("change", ["command_ulp", "rotation_command", "free_repeat", "step", "rate"])
def test_exact_requests_and_all_grid_admission_cannot_be_forged(audit, source, change):
    saved = package(source)
    answer = saved[0]
    if change == "command_ulp":
        command = answer["request_ledger"][1]["commands"]
        command[-1, 0] = np.nextafter(command[-1, 0], np.inf)
    elif change == "rotation_command":
        answer["grid"]["grid_commands"][0]["commands"][-1, 3] = 1e-6
    elif change == "free_repeat":
        answer["request_ledger"][-1].update(outcome="cached", charged=False, response_index=0, calls_after=13)
    elif change == "step":
        answer["request_ledger"][1]["signed_step"] = np.nextafter(0.001, np.inf)
    else:
        anchor = np.zeros((12, 6))
        anchor[-1, 0] = 0.25
        grid = build_tail_translation_grid(anchor, np.zeros(6), -np.ones(6), np.ones(6))
        with pytest.raises(ValueError, match="admission"):
            audit.audit_tail_translation_grid(grid)
        return
    with pytest.raises(ValueError):
        check(audit, saved)


@pytest.mark.parametrize("change", ["old_row", "prior_request", "e17_from_corrected"])
def test_e17_is_all_prior_original_residuals_not_new_or_corrected_fit(audit, source, change):
    saved = package(source)
    if change == "old_row":
        saved[2]["fit"]["original_terminal_row"][66] += 1e-3
    elif change == "prior_request":
        saved[2]["request_ledger"][15]["response_index"] = 14
    else:
        saved[0]["prior_max_abs_residual"] = max(abs(r["corrected_residual"]) for r in saved[2]["predictions"])
    with pytest.raises(ValueError):
        check(audit, saved)


def test_reserve_gate_uses_recomputed_e17_even_within_numeric_compare_tolerance(audit, source):
    saved = package(source, variant="reserve_boundary")
    result = check(audit, saved)
    assert result["improvement_found"] and not result["sufficient_reserve_restored"]
    answer = saved[0]
    answer["prior_max_abs_residual"] -= 2e-14
    answer["reserve_target"] = 10 * answer["prior_max_abs_residual"]
    answer["terminal_threshold"] = 1e-8 + answer["reserve_target"]
    answer["sufficient_reserve_restored"] = True
    answer["selected_summary"]["sufficient_reserve_restored"] = True
    with pytest.raises(ValueError):
        check(audit, saved)


def test_original_pose_caps_remain_strict_below_numeric_comparison_tolerance(audit, source):
    saved = package(source)
    entry = saved[0]["evaluated_candidates"][1]
    entry["tcp_errors"][0, 0] = 0.01 * (1 + 1e-12)
    with pytest.raises(ValueError, match="prefix_length"):
        check(audit, saved)


def test_audit_imports_only_numpy_and_frozen_independent_primitives(audit):
    tree = ast.parse(Path(audit.__file__).read_text())
    names = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    names += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert set(names) <= {"numpy", "directional_derivative_audit", "terminal_secant_audit"}
