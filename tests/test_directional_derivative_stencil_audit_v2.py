"""Archived synthetic stencil checks; no robot or new response evaluation."""

import importlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import test_directional_derivative_audit as original_tests

analytic_stencil_archive = original_tests.analytic_stencil_archive


@pytest.fixture
def stencil_audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("directional_derivative_stencil_audit_v2").audit_saved_stencil_comparison


def roundoff_bounds(old, bound):
    """Keep every recorded command; alter only derived stage-bound roundoff."""
    anchor = old["evaluated_candidates"][1819]["commands"]
    for stage in (24, 25, 26):
        phase = old["phases"][stage]
        if bound == "upper":
            value = anchor[0, 0]
            for _ in range(6):
                value = np.nextafter(value, -np.inf)
            phase["stage_upper_command"][0, 0] = value
            gap = anchor[0, 0] - value
        else:
            # Cancellation in constructing a bound can leave a sub-picounit
            # offset even when the recorded command coordinate is exactly zero.
            value = anchor[0, 1] + 3 * np.spacing(1.0)
            phase["stage_lower_command"][0, 1] = value
            gap = value - anchor[0, 1]
        assert 0 < gap < 1e-12


def binding_snapshot(old, answer):
    trace = old["evaluated_candidates"]
    return (
        tuple(np.asarray(trace[index]["commands"]).tobytes() for index in range(1819, 1892)),
        tuple((row["command_key"], row.get("nominal_delta"), row["response_index"]) for row in old["request_ledger"]),
        tuple(
            (
                np.asarray(old["phases"][stage]["initial_commands"]).tobytes(),
                np.asarray(old["phases"][stage]["qp_direction"]["commands"]).tobytes(),
                np.asarray(old["phases"][stage]["qp_direction"]["displacement"]).tobytes(),
                tuple(old["phases"][stage]["linearization"]["nominal_deltas"]),
            )
            for stage in (24, 25, 26)
        ),
        np.asarray(answer["anchor_commands"]).tobytes(),
        np.asarray(answer["unit_directions"]).tobytes(),
        tuple(
            (np.asarray(row["commands"]).tobytes(), np.asarray(row["actual_displacement"]).tobytes())
            for row in answer["request_ledger"]
        ),
    )


@pytest.mark.parametrize("bound", ["lower", "upper"])
def test_roundoff_bounds_preserve_full_analytic_stencil_and_exact_bytes(stencil_audit, analytic_stencil_archive, bound):
    old, answer, saved = deepcopy(analytic_stencil_archive)
    snapshot = binding_snapshot(old, answer)
    roundoff_bounds(old, bound)
    original = importlib.import_module("directional_derivative_audit").audit_saved_stencil_comparison
    with pytest.raises(ValueError, match="Anchor inside old stage bounds"):
        original(old, answer, saved)
    # Reuse independent closed-form secants, JSON roundtrip and all original
    # prediction assertions on the repaired admission path.
    original_tests.test_fixed_stencil_binding_matches_analytic_secants_and_roundtrips(
        stencil_audit, (old, answer, saved)
    )
    assert binding_snapshot(old, answer) == snapshot


@pytest.mark.parametrize("bound", ["lower", "upper"])
def test_stage_bound_violation_beyond_original_tolerance_is_rejected(stencil_audit, analytic_stencil_archive, bound):
    old, answer, saved = deepcopy(analytic_stencil_archive)
    anchor = old["evaluated_candidates"][1819]["commands"]
    phase = old["phases"][24]
    if bound == "upper":
        phase["stage_upper_command"][0, 0] = anchor[0, 0] - 2e-12
    else:
        phase["stage_lower_command"][0, 1] = anchor[0, 1] + 2e-12
    with pytest.raises(ValueError, match="Anchor inside old stage bounds"):
        stencil_audit(old, answer, saved)


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
def test_existing_stencil_tampering_still_fails_after_roundoff_admission(
    stencil_audit, analytic_stencil_archive, defect
):
    archive = deepcopy(analytic_stencil_archive)
    roundoff_bounds(archive[0], "upper")
    assert stencil_audit(*archive)["passed"]
    original_tests.test_saved_stencil_or_prediction_tampering_is_rejected(stencil_audit, archive, defect)
