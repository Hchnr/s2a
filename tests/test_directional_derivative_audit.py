"""Analytic synthetic direction-grid checks; no native robot responses run."""

import ast
import importlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from test_prefix_optimization_reserve_plan import inputs

from s2a.directional_derivative_probe import probe_directional_derivatives


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("directional_derivative_audit").audit_directional_derivative_ledger


def analytic_response(commands, *, quantized=False):
    x, y = commands[0, :2]
    if quantized:
        x, y = np.rint(x / 1e-4) * 1e-4, 0.0
    q, v = np.zeros(7), np.zeros(7)
    q[0], v[6] = x**2 + 2 * y, 3 * x - 4 * y
    pose = np.eye(4)[None]
    pose[0, 0, 3] = 0.002 + 0.3 * x + 0.2 * y**2
    return dict(
        inner_joint_position=np.tile(q, (25, 1)),
        inner_joint_velocity=np.tile(v, (25, 1)),
        joint_position=q[None].copy(),
        joint_velocity=v[None].copy(),
        pose=pose,
        twist=np.zeros((1, 6)),
    )


def generate(*, quantized=False, duplicate=False, boundary=None, drift=False, collapsed=False):
    calls = []

    def predict(command):
        calls.append(command.copy())
        value = analytic_response(np.zeros_like(command) if collapsed else command, quantized=quantized)
        if drift and len(calls) == 28:
            value["pose"][0, 0, 3] += 1e-7
        return value

    arguments = inputs(predict, horizon=1)
    anchor = arguments.pop("warm")
    anchor[0, :2] = [0.05002 if quantized else 0.05, -0.02]
    arguments["terminal_normals"] = np.eye(7)[6:]
    directions = np.zeros((3, 1, 6))
    if duplicate:
        directions[:, 0, 0] = [1, 2, -1]
    else:
        directions[0, 0, 0], directions[1, 0, 1] = 1, 2
        directions[2, 0, :2] = [3, 3]
    if boundary == "box":
        anchor[0, 0] = arguments["previous"][0] = 1.0
    elif boundary == "rate":
        anchor[0, 0] = 0.25
    elif collapsed:
        anchor[:] = 1e16
        arguments["previous"][:] = 1e16
        interface_type = type(arguments["interface"])
        arguments["interface"] = interface_type(-2e16 * np.ones(6), 2e16 * np.ones(6), -np.ones(6), np.ones(6))
    answer = probe_directional_derivatives(**arguments, anchor=anchor, directions=directions)
    return answer, calls


@pytest.fixture(scope="module")
def smooth():
    return generate()[0]


def test_analytic_derivatives_and_json_roundtrip(audit, smooth):
    checked = audit(smooth)
    assert checked["passed"] and checked["request_count"] == 80
    assert not checked["full_responses_independently_replayed_here"]
    assert not checked["source_archive_binding_checked_here"]
    assert not checked["control_qualified"]
    x, y = smooth["anchor_commands"][0, :2]
    p = 0.002 + 0.3 * x + 0.2 * y**2
    for pair in smooth["differences"][:13]:
        h = pair["step"]
        assert pair["forward"]["safety_slack"][0] == pytest.approx(2 * x + h, abs=3e-8)
        assert pair["backward"]["safety_slack"][0] == pytest.approx(2 * x - h, abs=3e-8)
        assert pair["central"]["safety_slack"][0] == pytest.approx(2 * x, abs=3e-8)
        for kind in ("forward", "backward", "central"):
            assert pair[kind]["safety_slack"][-1] == pytest.approx(-3.0, abs=3e-8)
        assert pair["central"]["normalized_squared_errors"][0, 0] == pytest.approx(6000 * p, abs=2e-6)
    for pair in smooth["differences"][13:26]:
        assert pair["central"]["normalized_squared_errors"][0, 0] == pytest.approx(
            8000 * p * y + 1600 * y * pair["step"] ** 2, abs=2e-6
        )
    loaded = json.loads(json.dumps(smooth, default=lambda value: value.tolist()))
    assert audit(loaded)["repeatability"] == checked["repeatability"]


def test_quantized_outputs_can_have_zero_derivative_with_nonzero_command_delta(audit):
    answer, _ = generate(quantized=True)
    audit(answer)
    for pair in answer["differences"][:13]:
        if pair["step"] <= 1e-5:
            assert pair["central"]["valid"]
            assert not answer["request_ledger"][pair["plus_request_index"]]["zero_displacement"]
            np.testing.assert_array_equal(pair["central"]["safety_slack"], 0)
            np.testing.assert_array_equal(pair["central"]["normalized_squared_errors"], 0)
    assert np.any(answer["differences"][0]["central"]["safety_slack"] != 0)


@pytest.mark.parametrize("boundary,reason", [("box", "box_upper"), ("rate", "rate_upper")])
def test_illegal_side_is_retained_unclipped_and_central_is_unavailable(audit, boundary, reason):
    answer, calls = generate(boundary=boundary)
    audit(answer)
    plus = answer["request_ledger"][1]
    assert plus["outcome"] == "command_rejected" and plus["admission"]["reasons"] == [reason]
    assert plus["commands"][0, 0] == answer["anchor_commands"][0, 0] + 0.01
    assert not plus["charged"] and plus["response_index"] is None
    pair = answer["differences"][0]
    assert not pair["forward"]["valid"] and not pair["central"]["valid"]
    assert pair["backward"]["valid"] and pair["central"]["safety_slack"] is None
    assert len(calls) == answer["response_evaluations"] < 80


def test_cache_saves_grid_calls_but_cannot_consume_mandatory_repeat(audit):
    answer, calls = generate(duplicate=True)
    checked = audit(answer)
    assert checked["unique_command_count"] == 27
    assert checked["response_evaluations"] == len(calls) == 28
    repeat = answer["request_ledger"][-1]
    assert repeat["outcome"] == "repeated" and repeat["charged"]
    assert repeat["response_index"] == 27 and repeat["repeat_of"] == 0
    np.testing.assert_array_equal(calls[0], calls[-1])
    assert checked["repeatability"]["exact"]


def test_nonzero_nominal_steps_that_round_to_anchor_are_not_derivatives(audit):
    answer, calls = generate(collapsed=True)
    audit(answer)
    assert len(calls) == 2 and answer["unique_command_count"] == 1
    for request in answer["request_ledger"][1:-1]:
        assert request["zero_displacement"] and request["outcome"] == "cached"
        assert request["relative_displacement_error_inf"] == 1.0
    for pair in answer["differences"]:
        for kind in ("forward", "backward", "central"):
            assert not pair[kind]["valid"] and pair[kind]["reason"] == "zero_displacement"


def test_real_repeat_disagreement_is_reported_without_inventing_reliability(audit):
    answer, calls = generate(duplicate=True, drift=True)
    checked = audit(answer)
    assert len(calls) == 28
    assert not checked["repeatability"]["exact"] and answer["derivative_reliability_limited"]
    pose = checked["repeatability"]["arrays"]["rollout.pose"]
    assert pose["max_abs_difference"] == pytest.approx(1e-7)
    assert not pose["exact"]
    forged = deepcopy(answer)
    forged["repeatability"]["exact"] = True
    forged["derivative_reliability_limited"] = False
    with pytest.raises(ValueError, match="repeat"):
        audit(forged)


@pytest.mark.parametrize(
    "defect",
    [
        "grid_command",
        "cache_repeat",
        "safety_vector",
        "pose_caps",
        "central_denominator",
        "repeat_array",
        "response_routing",
    ],
)
def test_meaningful_archive_tampering_is_rejected(audit, smooth, defect):
    answer = deepcopy(smooth)
    if defect == "grid_command":
        request = answer["request_ledger"][1]
        request["commands"][0, 0] = np.nextafter(request["commands"][0, 0], np.inf)
        request["command_key"] = request["commands"].tobytes().hex()
    elif defect == "cache_repeat":
        request = answer["request_ledger"][-1]
        request.update(outcome="cached", response_index=0, charged=False, calls_after=request["calls_before"])
    elif defect == "safety_vector":
        answer["evaluated_candidates"][1]["safety_slack"][-1] += 0.01
    elif defect == "pose_caps":
        answer["evaluated_candidates"][1]["normalized_squared_errors"] *= 0.5
    elif defect == "central_denominator":
        answer["differences"][0]["central"]["safety_slack"] *= 2
    elif defect == "repeat_array":
        answer["evaluated_candidates"][-1]["rollout"]["twist"][0, 0] += 0.1
    else:
        answer["request_ledger"][1]["response_index"] = 2
    with pytest.raises(ValueError):
        audit(answer)


def test_auditor_has_no_producer_optimizer_or_dynamics_imports():
    path = Path(__file__).resolve().parents[1] / "scripts/directional_derivative_audit.py"
    tree = ast.parse(path.read_text())
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module)
    assert imports == {"numpy"}


@pytest.fixture
def stencil_audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("directional_derivative_audit").audit_saved_stencil_comparison


@pytest.fixture(scope="module")
def analytic_stencil_archive():
    """Hand-built polynomial fields at the fixed slots, without a predictor."""
    steps = (1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8, 1e-8)
    stages = [24, 25, 26]
    anchor = np.zeros((12, 6))
    anchor[:, 0] = 1

    def fields(command):
        errors = np.zeros((12, 6))
        errors[:, 0] = 0.002 + 0.001 * command[:, 0] + 0.002 * command[:, 1]
        errors[:, 3] = 0.02 + 0.003 * command[:, 3]
        safety = np.array([2 + np.sum(command), 3 + 2 * command[0, 0] - 3 * command[0, 1]])
        squared = np.column_stack((errors[:, 0] ** 2 / 0.01**2, errors[:, 3] ** 2 / 0.05**2))
        return dict(commands=command.copy(), tcp_errors=errors, safety_slack=safety, normalized_squared_errors=squared)

    trace = [None] * 1892
    trace[1819] = fields(anchor)
    deltas = [-1e-4 if axis % 6 == 0 else 1e-4 for axis in range(72)]
    for axis, delta in enumerate(deltas):
        command = anchor.copy()
        command.ravel()[axis] += delta
        trace[1820 + axis] = fields(command)
    ledger, phases, directions = [], [None] * 27, []
    for offset, stage in enumerate(stages):
        radius = (0.01, 0.005, 0.0025)[offset]
        trial = anchor.copy()
        if offset == 0:
            trial[0, 0] -= radius
        elif offset == 1:
            trial[0, 1] += radius
        else:
            trial[0, 0] -= radius
            trial[0, 1] += radius / 2
        direction = trial - anchor
        directions.append(direction)
        start = len(ledger)
        for role in ("stage_anchor", "gradient_base"):
            ledger.append(
                dict(
                    request_index=len(ledger),
                    stage_index=stage,
                    role=role,
                    outcome="cache_hit",
                    response_index=1819,
                    command_key=anchor.tobytes().hex(),
                )
            )
        for axis, delta in enumerate(deltas):
            ledger.append(
                dict(
                    request_index=len(ledger),
                    stage_index=stage,
                    role="gradient_probe",
                    outcome="new_response" if offset == 0 else "cache_hit",
                    response_index=1820 + axis,
                    command_key=trace[1820 + axis]["commands"].tobytes().hex(),
                    axis=axis,
                    nominal_delta=delta,
                )
            )
        phases[stage] = dict(
            stage_index=stage,
            anchor_before_index=1819,
            initial_commands=anchor.copy(),
            qp_direction=dict(commands=trial, displacement=direction),
            requests_before=start,
            generation_requests_after=len(ledger),
            stage_lower_command=np.maximum(-1, anchor - radius),
            stage_upper_command=np.minimum(1, anchor + radius),
            linearization=dict(
                complete=True,
                base_response_index=1819,
                probe_response_indices=list(range(1820, 1892)),
                nominal_deltas=deltas.copy(),
            ),
        )
    old = dict(
        horizon=12, finite_difference_step=1e-4, phases=phases, request_ledger=ledger, evaluated_candidates=trace
    )
    directions = np.asarray(directions)
    scales = np.max(np.abs(directions), axis=(1, 2))
    units = directions / scales[:, None, None]
    new = dict(
        horizon=12,
        source_stages=stages,
        anchor_commands=anchor.copy(),
        directions=directions,
        direction_scales=scales,
        unit_directions=units,
        request_ledger=[],
        evaluated_candidates=[],
    )
    commands = [anchor.copy()]
    commands += [anchor + sign * step * unit for unit in units for step in steps for sign in (1, -1)]
    commands.append(anchor.copy())
    for index, command in enumerate(commands):
        # The outward x side is intentionally unavailable; no clipped substitute.
        admitted = bool(np.all(np.abs(command) <= 1))
        response_index = len(new["evaluated_candidates"]) if admitted else None
        new["request_ledger"].append(
            dict(
                request_index=index,
                commands=command.copy(),
                actual_displacement=command - anchor,
                response_index=response_index,
            )
        )
        if admitted:
            new["evaluated_candidates"].append(fields(command))
    # Analytic coordinate secants of the polynomial fields, including the
    # one-sided O(h) term of squared pose error. These are not producer outputs.
    Js, Jq = np.zeros((2, 72)), np.zeros((24, 72))
    Js[0] = 1
    Js[1, :2] = [2, -3]
    for endpoint in range(12):
        for component, coefficient in ((0, 0.001), (1, 0.002)):
            axis = 6 * endpoint + component
            Jq[2 * endpoint, axis] = (2 * 0.003 * coefficient + deltas[axis] * coefficient**2) / 0.01**2
        axis = 6 * endpoint + 3
        Jq[2 * endpoint + 1, axis] = (2 * 0.02 * 0.003 + deltas[axis] * 0.003**2) / 0.05**2
    saved = dict(
        anchor_response_index=1819,
        source_stages=stages,
        jacobian_safety_slack=Js,
        jacobian_normalized_squared_errors=Jq,
        unit_predictions=[
            dict(
                source_stage=stage,
                safety_slack=Js @ unit.ravel(),
                normalized_squared_errors=(Jq @ unit.ravel()).reshape(12, 2),
            )
            for stage, unit in zip(stages, units, strict=True)
        ],
        request_predictions=[],
    )
    initial = new["evaluated_candidates"][0]
    for request in new["request_ledger"]:
        delta = request["actual_displacement"].ravel()
        predicted = dict(safety_slack=Js @ delta, normalized_squared_errors=(Jq @ delta).reshape(12, 2))
        observed = residual = None
        if request["response_index"] is not None:
            response = new["evaluated_candidates"][request["response_index"]]
            observed = {key: response[key] - initial[key] for key in predicted}
            residual = {key: observed[key] - predicted[key] for key in predicted}
        saved["request_predictions"].append(
            dict(
                request_index=request["request_index"],
                predicted_change=predicted,
                observed_change=observed,
                residual=residual,
            )
        )
    return old, new, saved


def test_fixed_stencil_binding_matches_analytic_secants_and_roundtrips(stencil_audit, analytic_stencil_archive):
    old, answer, saved = analytic_stencil_archive
    checked = stencil_audit(old, answer, saved)
    assert checked["passed"] and checked["original_fd_axes_checked"] == 72
    assert checked["shared_stencils_checked"] == 3 and checked["request_predictions_checked"] == 80
    assert not checked["old_complete_rollout_reconstructed"]
    np.testing.assert_allclose(saved["unit_predictions"][0]["safety_slack"], [-1, -2])
    assert saved["jacobian_normalized_squared_errors"][0, 0] == pytest.approx(0.059999)
    assert saved["request_predictions"][1]["residual"]["normalized_squared_errors"][0, 0] == pytest.approx(
        9.9e-7, abs=1e-12
    )
    assert saved["request_predictions"][2]["observed_change"] is None
    loaded = json.loads(json.dumps([old, answer, saved], default=lambda value: value.tolist()))
    assert stencil_audit(*loaded)["passed"]


@pytest.mark.parametrize(
    "defect",
    [
        "fd_sign",
        "probe_slot",
        "actual_old_direction",
        "new_unit",
        "jacobian",
        "actual_delta",
        "residual",
        "unavailable_observation",
    ],
)
def test_saved_stencil_or_prediction_tampering_is_rejected(stencil_audit, analytic_stencil_archive, defect):
    old, answer, saved = deepcopy(analytic_stencil_archive)
    if defect == "fd_sign":
        old["request_ledger"][2]["nominal_delta"] *= -1
        old["phases"][24]["linearization"]["nominal_deltas"][0] *= -1
    elif defect == "probe_slot":
        probes = old["phases"][25]["linearization"]["probe_response_indices"]
        probes[0], probes[1] = probes[1], probes[0]
    elif defect == "actual_old_direction":
        displacement = old["phases"][24]["qp_direction"]["displacement"]
        displacement[0, 0] = np.nextafter(displacement[0, 0], np.inf)
    elif defect == "new_unit":
        answer["unit_directions"][0] *= 0.5
    elif defect == "jacobian":
        saved["jacobian_safety_slack"][0, 0] += 0.1
    elif defect == "actual_delta":
        answer["request_ledger"][1]["actual_displacement"][0, 0] += 1e-8
    elif defect == "residual":
        saved["request_predictions"][1]["residual"]["safety_slack"][0] += 0.01
    else:
        saved["request_predictions"][2]["observed_change"] = dict(
            safety_slack=np.zeros(2), normalized_squared_errors=np.zeros((12, 2))
        )
    with pytest.raises(ValueError):
        stencil_audit(old, answer, saved)
