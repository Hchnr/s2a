"""Independent ledger tests with analytic arrays, never native robot responses."""

import ast
import importlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from test_prefix_optimization_reserve_plan import inputs

from s2a.directional_derivative_probe import probe_directional_derivatives
from s2a.terminal_secant_probe import (
    build_terminal_secant_geometry,
    fit_terminal_secant_row,
    probe_terminal_secant,
)


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("terminal_secant_audit")


def analytic_response(commands):
    x, y, z = commands[0, :3]
    terminal = 0.05 + 0.7 * x + 0.2 * y - 0.3 * z
    velocity = np.zeros((1, 7))
    velocity[0, 6] = -terminal
    pose = np.eye(4)[None]
    pose[0, 0, 3] = 0.002 + 0.1 * y
    return dict(
        inner_joint_position=np.zeros((25, 7)),
        inner_joint_velocity=np.repeat(velocity, 25, axis=0),
        joint_position=np.zeros((1, 7)),
        joint_velocity=velocity,
        pose=pose,
        twist=np.zeros((1, 6)),
        goals=pose.copy(),
        periods=[dict(clipped=np.zeros((25, 7), dtype=bool))],
    )


def archive(*, change=None):
    """Production core generates saved fixtures; the audit imports no core."""
    args = inputs(analytic_response, horizon=1)
    anchor = args.pop("warm")
    args["terminal_normals"] = np.eye(7)[6:]
    sources = np.zeros((3, 1, 6))
    sources[np.arange(3), 0, np.arange(3)] = 1
    prior = probe_directional_derivatives(**args, anchor=anchor, directions=sources)
    history = prior["evaluated_candidates"][0]
    fit_requests = [
        next(
            row
            for row in prior["request_ledger"]
            if row["role"] == "grid" and row["source_stage"] == 24 and row["sign"] == 1 and row["step"] == step
        )
        for step in (0.001, 0.0005)
    ]
    fit_entries = [prior["evaluated_candidates"][row["response_index"]] for row in fit_requests]
    row = np.array([0.1, 0.2, -0.3, 0.0, 0.0, 0.0])
    geometry = build_terminal_secant_geometry(
        anchor, args["previous"], args["interface"].input_min, args["interface"].input_max, sources
    )
    fit = fit_terminal_secant_row(
        anchor,
        sources[0],
        row,
        history["safety_slack"][-1],
        np.array([entry["commands"] for entry in fit_entries]),
        np.array([entry["safety_slack"][-1] for entry in fit_entries]),
    )
    calls = []

    def predict(commands):
        calls.append(commands.copy())
        response = analytic_response(commands)
        if change == "clipping" and len(calls) == 8:
            response["periods"][0]["clipped"][0, 0] = True
        if change == "repeat" and len(calls) == 17:
            response["pose"][0, 0, 3] += 1e-7
        if change == "history":
            response["goals"][0, 0, 3] += 1e-7
        if change == "small_step_error" and 0 < commands[0, 0] < 2e-4 and np.any(commands[0, 3:] != 0):
            offset = 1e-4
            response["joint_velocity"][0, 6] -= offset
            response["inner_joint_velocity"][:, 6] -= offset
        return response

    args["predict"] = predict
    answer = probe_terminal_secant(**args, anchor=anchor, geometry=geometry, fit=fit, historical_anchor=history)
    return answer, history, prior, calls


@pytest.fixture(scope="module")
def affine():
    return archive()


def checked(audit, package):
    answer, history, prior, _ = package
    return audit.audit_terminal_secant_ledger(answer, historical_anchor=history, source_answer=prior)


def test_complete_affine_recovery_full17_billing_and_json_roundtrip(audit, affine):
    result = checked(audit, affine)
    answer, history, prior, calls = affine
    assert result["passed"] and result["extrapolation_passed"] and result["safety_accuracy_passed"]
    assert result["response_evaluations"] == len(calls) == 17 and result["unique_command_count"] == 16
    assert len(result["witness_indices"]) == 17  # A warm witness must not stop this fixed experiment.
    assert not result["control_qualified"] and not result["full_responses_independently_replayed_here"]
    assert result["fit_check"]["source_archive_binding_checked_here"]
    assert not result["fit_check"]["original_jacobian_source_binding_checked_here"]
    assert answer["fit"]["coefficient"] == pytest.approx(0.6, abs=1e-13)
    assert max(abs(row["corrected_residual"]) for row in answer["predictions"]) < 1e-16
    loaded = json.loads(json.dumps([answer, history, prior], default=lambda value: value.tolist()))
    replay = audit.audit_terminal_secant_ledger(loaded[0], historical_anchor=loaded[1], source_answer=loaded[2])
    assert replay["qualification"] == result["qualification"]


@pytest.mark.parametrize(
    "change,field", [("clipping", "all_clipping_equal"), ("repeat", "repeat_exact"), ("history", "historical_exact")]
)
def test_real_diagnostic_disagreements_retained_but_cannot_be_called_passed(audit, change, field):
    package = archive(change=change)
    result = checked(audit, package)
    assert not result["extrapolation_passed"] and not result["qualification"][field]
    changed = deepcopy(package[0])
    changed["extrapolation_passed"] = True
    with pytest.raises(ValueError):
        audit.audit_terminal_secant_ledger(changed, historical_anchor=package[1], source_answer=package[2])


def test_absolute_improvement_cannot_bypass_normalized_small_step_failure(audit):
    package = archive(change="small_step_error")
    result = checked(audit, package)
    assert all(row["absolute_passed"] for row in result["qualification"]["mixed_direction_checks"])
    assert not any(row["normalized_passed"] for row in result["qualification"]["mixed_direction_checks"])
    assert not result["extrapolation_passed"]
    forged = deepcopy(package[0])
    forged["qualification"]["mixed_direction_checks"][0]["normalized_passed"] = True
    with pytest.raises(ValueError):
        audit.audit_terminal_secant_ledger(forged, historical_anchor=package[1], source_answer=package[2])


@pytest.mark.parametrize("change", ["training_side", "training_label", "training_response", "holdout_refit"])
def test_fixed_training_association_cannot_be_changed_or_refitted_on_holdout(audit, affine, change):
    answer, history, prior, _ = deepcopy(affine)
    if change == "training_side":
        prior["request_ledger"][5]["sign"] = -1
    elif change == "training_label":
        ri = prior["request_ledger"][5]["response_index"]
        prior["evaluated_candidates"][ri]["safety_slack"][-1] += 1e-7
    elif change == "training_response":
        prior["request_ledger"][5]["response_index"] = 0
    else:
        answer["fit"]["fit_commands"][0] = answer["request_ledger"][7]["commands"]
    with pytest.raises(ValueError):
        audit.audit_terminal_secant_ledger(answer, historical_anchor=history, source_answer=prior)


@pytest.mark.parametrize(
    "change", ["seed", "guard_row", "command_ulp", "free_repeat", "swapped_labels", "prediction", "qualification"]
)
def test_geometry_routing_predictions_and_acceptance_tampering_rejected(audit, affine, change):
    answer, history, prior, _ = deepcopy(affine)
    if change == "seed":
        answer["geometry"]["seed"] = 43
    elif change == "guard_row":
        answer["geometry"]["projection_rows"][0, 0] *= 2
    elif change == "command_ulp":
        value = answer["request_ledger"][1]["commands"]
        value[0, 3] = np.nextafter(value[0, 3], np.inf)
    elif change == "free_repeat":
        answer["request_ledger"][-1].update(outcome="cached", charged=False, response_index=0, calls_after=16)
    elif change == "swapped_labels":
        answer["evaluated_candidates"][7]["feasible"] = False
    elif change == "prediction":
        answer["predictions"][9]["corrected_predicted_change"] += 1e-6
    else:
        answer["qualification"]["improvement_factor"] = 0.9
    with pytest.raises(ValueError):
        audit.audit_terminal_secant_ledger(answer, historical_anchor=history, source_answer=prior)


def test_original_pose_caps_remain_strict_at_subtolerance_crossing(audit, affine):
    answer, history, prior, _ = deepcopy(affine)
    entry = answer["evaluated_candidates"][7]
    entry["tcp_errors"][0, :3] = [0.01 * (1 + 1e-12), 0, 0]
    entry["normalized_squared_errors"][0, 0] = (1 + 1e-12) ** 2
    with pytest.raises(ValueError, match="prefix_length"):
        audit.audit_terminal_secant_ledger(answer, historical_anchor=history, source_answer=prior)


def test_signed_interperiod_guard_is_reconstructed_without_row_normalization(audit):
    anchor, previous = np.zeros((2, 6)), np.zeros(6)
    anchor[1, 0] = 0.2495
    source = np.zeros((3, 2, 6))
    source[np.arange(3), 0, np.arange(3)] = 1
    geometry = build_terminal_secant_geometry(anchor, previous, -np.ones(6), np.ones(6), source)
    result = audit.audit_terminal_secant_geometry(geometry)
    assert result["projection_rank"] == 4
    metadata = geometry["projection_row_metadata"][-1]
    assert metadata["group"] == "rate_upper" and metadata["axis"] == 6
    expected = np.zeros(12)
    expected[0], expected[6] = 1, -1
    np.testing.assert_array_equal(geometry["projection_rows"][-1], expected)
    altered = deepcopy(geometry)
    altered["projection_rows"][-1] /= np.sqrt(2)
    with pytest.raises(ValueError):
        audit.audit_terminal_secant_geometry(altered)


def test_audit_has_no_production_or_model_dependency(audit):
    tree = ast.parse(Path(audit.__file__).read_text())
    names = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    names += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert set(names) <= {"numpy", "directional_derivative_audit"}
