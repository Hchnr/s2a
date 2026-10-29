"""Independent fixed-grid/score/recurrence checks using synthetic arrays only."""

import ast
import importlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from s2a.saved_direction_backtracking import build_batch


@pytest.fixture
def checker(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("check_tracking_v5_saved_direction_backtracking")


def fixed_sources():
    """Prevalidated synthetic source records; no saved experiments or responses."""
    rng = np.random.default_rng(42)
    diagnostic, producers = {"cases": {}}, {}
    for name, horizon, count, target in (("prefix", 12, 10, 9), ("early", 13, 5, 10), ("late", 13, 1, 10)):
        trace, pairs = [], [{"main_subset": False}]
        for _ in range(count):
            base = rng.uniform(-0.05, 0.05, (horizon, 6))
            trial = base + rng.uniform(-0.03, 0.03, (horizon, 6))
            base[0, 0], trial[0, 0] = -0.0, 0.0
            errors = np.zeros((horizon, 6))
            errors[:, 0], errors[:, 3] = 0.006, 0.025
            errors[target, 0] = 0.012
            scores = np.column_stack(
                (np.sum(errors[:, :3] ** 2, axis=1) / 0.01**2, np.sum(errors[:, 3:] ** 2, axis=1) / 0.05**2)
            )
            base_index, trial_index = len(trace), len(trace) + 1
            for command in (base, trial):
                trace.append(
                    dict(
                        commands=command,
                        tcp_errors=errors.copy(),
                        safety_min=0.1,
                        prefix_length=target,
                        joint_feasible=True,
                        feasible=False,
                    )
                )
            pairs.append(
                dict(
                    main_subset=True,
                    base_index=base_index,
                    trial_index=trial_index,
                    phase_target_cursor=target,
                    prefix_before=target,
                    qbase=scores,
                    target={"base_max": float(np.max(scores[target]))},
                )
            )
        diagnostic["cases"][name] = dict(horizon=horizon, target_index=target, pairs=pairs)
        producers[name] = {"answer": dict(horizon=horizon, evaluated_candidates=trace)}
    return diagnostic, producers


def test_fixed_batch_independently_matches_producer_and_preserves_endpoint_bytes(checker):
    diagnostic, producers = fixed_sources()
    original = deepcopy(producers)
    requests = checker.rebuild_requests(diagnostic, producers)
    other = build_batch(diagnostic, producers)
    assert len(requests) == len(other) == 96
    assert len({item["commands"].tobytes() for item in requests}) == 96
    assert sum(item["source_trace_index"] is not None for item in requests) == 32
    assert sum(item["horizon"] * 25 for item in requests) == 29700
    assert 3 + sum(item["horizon"] * 26 for item in requests) == 30891
    for request, production in zip(requests, other, strict=True):
        checker.compare_contract(request, production)
        assert request["commands"].tobytes() == production["commands"].tobytes()
        assert request["pair_index"] >= 1  # Preserve the original all-pairs index.
        trace = producers[request["case_name"]]["answer"]["evaluated_candidates"]
        if request["source_trace_index"] is not None:
            source = trace[request["source_trace_index"]]["commands"]
            assert request["commands"].tobytes() == source.tobytes()
            assert not np.shares_memory(request["commands"], source)
    for name, producer in producers.items():
        for row, old in zip(
            producer["answer"]["evaluated_candidates"], original[name]["answer"]["evaluated_candidates"], strict=True
        ):
            assert row["commands"].tobytes() == old["commands"].tobytes()


@pytest.mark.parametrize("corruption", ["drop_direction", "qbase", "reuse_interior"])
def test_fixed_batch_rejects_changed_selection_source_scores_and_free_cached_interiors(checker, corruption):
    diagnostic, producers = fixed_sources()
    if corruption == "drop_direction":
        diagnostic["cases"]["early"]["pairs"][1]["main_subset"] = False
    elif corruption == "qbase":
        diagnostic["cases"]["early"]["pairs"][1]["qbase"][0, 0] += 1e-8
    else:
        trace = producers["early"]["answer"]["evaluated_candidates"]
        reused = deepcopy(trace[0])
        reused["commands"] = trace[0]["commands"] + 0.125 * (trace[1]["commands"] - trace[0]["commands"])
        # Matching a different case's archived trace must also disqualify a new interior.
        producers["late"]["answer"]["evaluated_candidates"].append(reused)
    with pytest.raises(ValueError):
        checker.rebuild_requests(diagnostic, producers)


def extract_functions(checker, names, namespace):
    tree = ast.parse(Path(checker.__file__).read_text())
    found = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in found} == set(names)
    module = ast.fix_missing_locations(ast.Module(body=found, type_ignores=[]))
    exec(compile(module, checker.__file__, "exec"), namespace)


def test_native_mathematics_are_the_frozen_independent_functions(checker):
    proof = checker.native_math_proof()
    assert len(proof["functions"]) == 8 and all(proof["functions"].values())


@pytest.mark.parametrize("horizon", [12, 13])
def test_each_alpha_rollout_resets_q_velocity_and_history_latch(checker, horizon):
    q0, v0, latch = np.arange(7) / 10, np.arange(7) / 20, np.diag([1.0, -1.0, -1.0])
    starts = []

    def period(q, v, fixed, command, inherited):
        starts.append((q.copy(), v.copy(), inherited.copy(), fixed))
        # A causal synthetic period exposes changes in both propagated state and goal.
        positions = q + np.arange(1, 26)[:, None] * command[0]
        velocities = v + np.arange(1, 26)[:, None] * command[1]
        goal = np.eye(4)
        goal[:3, :3] = inherited @ np.diag([-1.0, -1.0, 1.0])
        return (
            dict(
                joint_position=positions,
                joint_velocity=velocities,
                goal=goal,
                endpoint_pose=np.eye(4),
                endpoint_twist=np.zeros(6),
            ),
            {},
            0.0,
        )

    namespace = dict(
        np=np,
        require=checker.require,
        HORIZON=horizon,
        q0=q0.copy(),
        v0=v0.copy(),
        latch=latch.copy(),
        fixed={"finger": [0.1, 0.0]},
        reconstruct=period,
    )
    extract_functions(checker, {"reconstruct_trajectory"}, namespace)
    commands = np.tile([0.01, -0.002, 0.0, 0.0, 0.0, 0.0], (horizon, 1))
    first, _ = namespace["reconstruct_trajectory"](commands)
    second, _ = namespace["reconstruct_trajectory"](commands * 0.5)
    for offset in (0, horizon):
        np.testing.assert_array_equal(starts[offset][0], q0)
        np.testing.assert_array_equal(starts[offset][1], v0)
        np.testing.assert_array_equal(starts[offset][2], latch)
    for index in range(1, horizon):
        np.testing.assert_array_equal(starts[index][0], first["joint_position"][index - 1])
        np.testing.assert_array_equal(starts[index][1], first["joint_velocity"][index - 1])
        np.testing.assert_array_equal(starts[index][2], first["goals"][index - 1, :3, :3])
    assert first["inner_joint_position"].shape == second["inner_joint_velocity"].shape == (25 * horizon, 7)
    np.testing.assert_array_equal(namespace["q0"], q0)
    np.testing.assert_array_equal(namespace["v0"], v0)
    np.testing.assert_array_equal(namespace["latch"], latch)


@pytest.mark.parametrize(
    "target_score,safety,joint,prefix,descent,witness",
    [
        (1.1, 1e-8, True, 9, True, False),
        (1.1, np.nextafter(1e-8, 0.0), True, 9, False, False),
        (1.1, 0.1, False, 9, False, False),
        (1.1, 0.1, True, 8, False, False),
        (1.0, -5e-11, True, 12, False, True),
    ],
)
def test_diagnostic_descent_keeps_prefix_buffer_and_full_witness_distinct(
    checker, target_score, safety, joint, prefix, descent, witness
):
    normalized = np.zeros((12, 2))
    normalized[9, 0] = target_score
    metrics = dict(
        tcp_errors=np.zeros((12, 6)),
        normalized_squared_errors=normalized,
        safety_min=safety,
        prefix_length=prefix,
        joint_feasible=joint,
        feasible=witness,
    )
    request = dict(target_index=9, prefix_before=9, base_target_max=1.2)
    labels = checker.backtracking_classification(metrics, request)
    assert labels["feasible_descent"] is descent
    assert labels["feasible"] is witness
    assert labels["target_passed"] is (target_score <= 1)
    assert labels["target_improvement"] == pytest.approx(1.2 - target_score)


def test_old_endpoint_scope_and_strict_labels_do_not_inherit_comparison_tolerance(checker):
    diagnostic, producers = fixed_sources()
    request = checker.rebuild_requests(diagnostic, producers)[0]
    archived = producers["prefix"]["answer"]["evaluated_candidates"][0]
    metrics = {
        key: deepcopy(archived[key])
        for key in ("tcp_errors", "safety_min", "prefix_length", "joint_feasible", "feasible")
    }

    def compare(name, actual, expected):
        difference = checker.finite_max_abs_difference(actual, expected, name=name)
        checker.require_absolute_tolerance(difference, 1e-10, name=name)
        return difference

    evidence = checker.endpoint_evidence(request, metrics, producers["prefix"], compare)
    assert not evidence["old_full_rollout_available"]
    assert set(evidence["fields_checked"]) == set(metrics) | {"commands"}
    metrics["tcp_errors"][9, 0] += 1e-12
    metrics["prefix_length"] += 1
    with pytest.raises(ValueError, match="Strict"):
        checker.endpoint_evidence(request, metrics, producers["prefix"], compare)
