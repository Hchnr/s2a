"""Both search strategies receive identical input and independent witness checks."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.tracking.trajectory_search import compare_multitask_feasibility_first as compare


def sample():
    interface = SimpleNamespace(
        input_min=np.full(6, -1.0), input_max=np.ones(6), output_min=np.full(6, -0.1), output_max=np.full(6, 0.1)
    )
    args = (
        {"q": np.zeros(7)},
        np.zeros(7),
        np.zeros((8, 6)),
        np.zeros(6),
        interface,
        np.eye(4),
        np.eye(4),
        np.repeat(np.eye(4)[None], 8, axis=0),
        np.zeros((8, 6)),
        np.full(7, -1.0),
        np.ones(7),
    )
    context = dict(
        mass=np.eye(7),
        jacobian=np.zeros((6, 7)),
        contract=dict(kp=np.ones(6), kd=np.ones(6)),
        q=np.zeros(7),
        velocity=np.zeros(7),
        interface=interface,
        previous=np.zeros(7),
        site=np.eye(4),
        limits=np.tile([-1.0, 1.0], (7, 1)),
    )
    stats = dict(
        optimizer_calls_attempted=0,
        optimizer_calls_completed=0,
        optimizer_responses=0,
        independent_endpoint_queries_charged=0,
        independent_transition_steps_charged=0,
    )
    return args, context, np.zeros((25, 7)), stats


def test_both_solvers_share_inputs_and_only_audited_witness_is_accepted():
    args, context, profile, stats = sample()
    seen = []

    def solve(*values, **kwargs):
        seen.append(compare.input_fingerprint(values, kwargs["past_inner_error_rad"]))
        return dict(witness_found=True, response_evaluations=2, rollout={})

    def audit(*_args, **_kwargs):
        return dict(
            passed=True,
            first_pose_caps_satisfied=True,
            first_pose_caps_enforced=True,
            braking_constraint_enforced=True,
            independent_endpoint_queries=200,
            independent_transition_steps=2600,
        )

    result = compare.compare_solvers(
        args,
        context,
        np.zeros(7),
        profile,
        stats,
        solvers=(("original", solve), ("feasibility_first", solve)),
        auditor=audit,
    )
    assert seen == [result["input_sha256"]] * 2
    assert all(item["accepted_model_witness"] for item in result["solvers"].values())
    assert stats["optimizer_calls_completed"] == 2 and stats["independent_endpoint_queries_charged"] == 400


def test_mutated_shared_seed_is_rejected_before_another_solver():
    args, context, profile, stats = sample()

    def mutate(*values, **_kwargs):
        values[2][0, 0] = 1.0
        return dict(witness_found=False, response_evaluations=1)

    with pytest.raises(ValueError, match="modified"):
        compare.compare_solvers(
            args, context, np.zeros(7), profile, stats, solvers=(("original", mutate), ("feasibility_first", mutate))
        )


def test_raw_witness_is_rejected_when_independent_audit_fails():
    args, context, profile, stats = sample()

    def solve(*_args, **_kwargs):
        return dict(witness_found=True, response_evaluations=1, rollout={})

    def reject(*_args, **_kwargs):
        raise ValueError("independent limit mismatch")

    result = compare.compare_solvers(
        args,
        context,
        np.zeros(7),
        profile,
        stats,
        solvers=(("original", solve), ("feasibility_first", solve)),
        auditor=reject,
    )
    assert all(
        not item["accepted_model_witness"] and item["independent_audit_error"] for item in result["solvers"].values()
    )
    assert stats["independent_endpoint_queries_charged"] == 400


def test_mismatched_terminal_directions_are_rejected():
    args, context, profile, stats = sample()

    def original(*_args, **_kwargs):
        return dict(witness_found=False, response_evaluations=1, terminal_normals=np.zeros((1, 7)))

    def changed(*_args, **_kwargs):
        return dict(witness_found=False, response_evaluations=1, terminal_normals=np.ones((1, 7)))

    with pytest.raises(ValueError, match="Unequal solver constraint: terminal_normals"):
        compare.compare_solvers(
            args, context, np.zeros(7), profile, stats, solvers=(("original", original), ("feasibility_first", changed))
        )
