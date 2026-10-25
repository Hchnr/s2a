"""New aggregate identity must preserve the old metric rules and full QR gate."""

import ast
import copy

import numpy as np
import pytest

from s2a.artifacts import REPO
from s2a.synchronous_state_metrics import diagnosis as previous_diagnosis
from s2a.terminal_candidate import ROOT, candidate_head
from s2a.terminal_identity_state_metrics import diagnosis as aggregate_diagnosis
from s2a.wam_clean_readout import CleanResidualReadout


def test_metric_arithmetic_and_gate_thresholds_are_unchanged():
    class Rename(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                return ast.copy_location(ast.Constant(node.value.replace("probe052", "terminal058")), node)
            return node

    def functions(name):
        return {
            n.name: n for n in ast.parse((REPO / "src/s2a" / name).read_text()).body if isinstance(n, ast.FunctionDef)
        }

    old = functions("synchronous_state_metrics.py")
    new = functions("terminal_identity_state_metrics.py")
    for name in ("pool", "state_gate", "diagnosis"):
        assert ast.dump(Rename().visit(old[name])) == ast.dump(new[name])


def test_all_groups_and_failure_checks_match_previous_scorer():
    rows = []
    for task in range(3):
        for demo in range(2):
            for anchor in range(4):
                for seed in (42, 43):
                    for variant in ("raw", "probe052"):
                        factor = 1.0 if variant == "raw" else (1.5 if (task, demo, seed) == (2, 1, 43) else 0.8)
                        errors = {
                            k: [factor * v] * 40 for k, v in (("position", 0.03), ("rotation", 0.1), ("opening", -0.08))
                        }
                        rows.append(
                            dict(
                                name=f"t{task}d{demo}a{anchor}_{variant}_s{seed}",
                                episode=f"t{task}d{demo}",
                                task_index=task,
                                sampling_seed=seed,
                                variant=variant,
                                errors=dict(
                                    wam=errors,
                                    hold={k: [v * 2 for v in a] for k, a in errors.items()},
                                    constant_velocity={k: [v * 3 for v in a] for k, a in errors.items()},
                                ),
                            )
                        )
    old_groups, old_gate = previous_diagnosis(rows)
    new_rows = copy.deepcopy(rows)
    for row in new_rows:
        if row["variant"] == "probe052":
            row["variant"] = "terminal058"
            row["name"] = row["name"].replace("probe052", "terminal058")
    new_groups, new_gate = aggregate_diagnosis(new_rows)

    def renamed(value):
        if isinstance(value, dict):
            return {
                (k.replace("probe052", "terminal058") if isinstance(k, str) else k): renamed(v)
                for k, v in value.items()
            }
        return value

    assert new_groups == renamed(old_groups)
    assert new_gate == old_gate and new_gate["passed"] is False


def evidence():
    def ref(name):
        return dict(path=str(ROOT / name), sha256="a" * 64)

    proof, fit, params, prepared = map(
        ref, ("fit_verified.json", "fit_result.json", "parameters.json", "prepared.json")
    )
    baseline_ref = dict(
        path=str(ROOT / "baseline.json"), sha256="ca01b9d8a49f2f1585c917f492e52a0a89cdb0b447f6bd2beee050ed0d2fccc0"
    )
    baseline = CleanResidualReadout(np.ones((1291, 10)), np.zeros(1290), 2.0, 0.1 / 1290).to_parameters()
    updated = copy.deepcopy(baseline)
    updated["coefficients"][0][9] += 0.1
    docs = {
        proof["path"]: dict(
            passed=True,
            complete=True,
            requests=300,
            rows=12000,
            performance_qualified=False,
            position_relative_gradient_residual=1e-9,
            opening_gap=1e-6,
            producer_ref=fit,
            params_ref=params,
            new_model_calls=0,
            physics_steps=0,
            fresh_cases=0,
        ),
        fit["path"]: dict(
            passed=True,
            complete=True,
            producer_fits=1,
            cumulative_producer_attempts=2,
            params_ref=params,
            independent_numerical_qualification=False,
            position=dict(cg_info=0, iterations=351, relative_normal_residual=1e-9),
            opening_history=[
                dict(outer_step=i + 1, cg_info=0, iterations=10, feasible=True, gap=1e-6) for i in range(100)
            ],
            prepared_ref=prepared,
            new_model_calls=0,
            physics_steps=0,
            fresh_cases=0,
        ),
        prepared["path"]: dict(params_ref=baseline_ref),
        params["path"]: updated,
        baseline_ref["path"]: baseline,
    }
    return proof, docs


def test_terminal_candidate_has_its_own_numeric_identity():
    proof, docs = evidence()
    policy, params = candidate_head(proof, read_json=lambda r: docs[r["path"]])
    assert params["path"] == str(ROOT / "parameters.json")
    assert policy.at_step(0) is policy.baseline
    assert policy.at_step(34) is policy.terminal


@pytest.mark.parametrize("fault", ["failed", "rows", "budget", "gap", "rotation", "future-work"])
def test_incomplete_or_different_qualification_cannot_be_used(fault):
    proof, docs = evidence()
    verified = docs[proof["path"]]
    fit = docs[verified["producer_ref"]["path"]]
    if fault == "failed":
        verified["passed"] = False
    elif fault == "rows":
        verified["rows"] = 11999
    elif fault == "budget":
        fit["opening_history"] = fit["opening_history"][:30]
    elif fault == "gap":
        verified["opening_gap"] = 0.003
    elif fault == "rotation":
        docs[fit["params_ref"]["path"]]["coefficients"][0][3] += 1
    else:
        fit["fresh_cases"] = 1
    with pytest.raises(ValueError):
        candidate_head(proof, read_json=lambda r: docs[r["path"]])
