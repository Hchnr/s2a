"""Inspect cached constraints without changing the underlying solver."""

import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint

from scripts.tracking.trajectory_search import audit_cached_live_stop125_constraints as audit


def test_cached_postfilter_distinguishes_rate_and_absolute_pose_rejections():
    bounds = Bounds(-np.ones(6), np.ones(6))
    rate = LinearConstraint(np.eye(6), -0.25, 0.25)
    joint = LinearConstraint(np.eye(6), -0.8, 0.8)
    constraints = [rate, joint, {"type": "ineq"}]

    def score(position, rotation):
        return {"score": {"residual": np.array([100 * position, 0, 0, 20 * rotation, 0, 0]), "objective": 1.0}}

    cache = {
        tuple(np.zeros(6)): score(0.009, 0.04),
        (0.250000000002, 0, 0, 0, 0, 0): score(0.009, 0.04),
        (0.1, 0, 0, 0, 0, 0): score(0.010000001, 0.04),
    }
    rows = audit.inspect_cache(cache, bounds, constraints, np.zeros(6))
    assert rows[0]["accepted"] is True
    assert rows[1]["failed_checks"] == ["rate"]
    assert rows[2]["failed_checks"] == ["position"]
    assert rows[1]["maximum_rate"] == pytest.approx(0.250000000002)
    assert len(cache) == len(rows) == 3


def test_terminal_and_braking_constraints_are_checked_from_cached_plans():
    bounds = Bounds(-np.ones(6), np.ones(6))
    rate = LinearConstraint(np.eye(6), -0.25, 0.25)
    joint = LinearConstraint(np.eye(6), -0.05, 0.05)
    terminal = LinearConstraint(np.ones((1, 6)), -np.inf, 0.05)
    commands = (0.1, 0, 0, 0, 0, 0)
    cache = {commands: {"score": {"residual": np.zeros(6), "objective": 0.0}}}
    row = audit.inspect_cache(cache, bounds, [rate, joint, {"type": "ineq"}, terminal], np.zeros(6))[0]
    assert row["failed_checks"] == ["joint_braking", "terminal"]
    assert row["joint_braking_minimum_slack_rad"] == pytest.approx(-0.05)
    assert row["terminal_maximum_excess"] == pytest.approx(0.05)


def test_unknown_constraint_layout_is_rejected():
    with pytest.raises(ValueError, match="Original rate"):
        audit.inspect_cache({}, Bounds(-np.ones(6), np.ones(6)), [], np.zeros(6))


MARKER = "original"


def read_marker():
    return MARKER


def test_instrumentation_uses_an_isolated_function_namespace():
    instrumented = audit.isolated(read_marker, {"MARKER": "observed"})
    assert instrumented() == "observed"
    assert read_marker() == "original"
