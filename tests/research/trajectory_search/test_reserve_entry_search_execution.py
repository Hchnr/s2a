"""Finite paired budgets and independent native decision fields cannot drift."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from scripts.tracking.trajectory_search.check_reserve_entry_search import native_metric_entry, validate_spec
from tests.research.trajectory_search.test_state_vector_gn_search import entry


def specification():
    return json.loads(
        (
            Path(__file__).resolve().parents[3]
            / "configs/tracking/trajectory_search/reserve_entry_search_20261102.json"
        ).read_text()
    )


def test_actual_registered_specification_is_finite_and_matches_original_config():
    spec = specification()
    validate_spec(spec, spec["search_config"])


@pytest.mark.parametrize(
    "field",
    [
        "total_response_budget",
        "total_query_budget",
        "qp_budget_per_arm",
        "model_budget",
        "owner_timeout_seconds",
        "automatic_retry",
    ],
)
def test_budget_or_restart_drift_rejected(field):
    spec = specification()
    original = deepcopy(spec["search_config"])
    spec[field] = True if field == "automatic_retry" else spec[field] + 1
    with pytest.raises(ValueError):
        validate_spec(spec, original)


def native_case():
    e = entry(np.zeros((12, 6)), np.zeros(72))
    metric = {
        k: deepcopy(e[k])
        for k in [
            "safety_slack",
            "tcp_errors",
            "normalized_squared_errors",
            "feasible",
            "prefix_length",
            "maximum_normalized_squared_error",
            "maximum_pose_cap_violation",
            "pose_loss",
        ]
    }
    metric["joint_feasible"] = e["audit"]["feasible"]
    return e, metric


def test_native_fields_are_copied_without_aliasing_production_or_metric_arrays():
    e, m = native_case()
    rollout = dict(trace=123)
    rebuilt = native_metric_entry(e, rollout, m)
    assert rebuilt["rollout"] is rollout
    m["tcp_errors"][0, 0] = 9
    assert rebuilt["tcp_errors"][0, 0] == e["tcp_errors"][0, 0] == 0


@pytest.mark.parametrize("field", ["safety_slack", "tcp_errors", "joint_feasible"])
def test_any_native_decision_drift_fails_exact_gate(field):
    e, m = native_case()
    if field == "joint_feasible":
        m[field] = False
    else:
        m[field].flat[0] += 1e-14
    with pytest.raises(ValueError):
        native_metric_entry(e, dict(trace=0), m)
