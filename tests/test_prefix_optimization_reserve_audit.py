"""Prove the solver-only edit and preserve original independent ray qualification."""

import ast
import hashlib
import importlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from test_prefix_callback_backtracking_audit import capture_trial, classifications
from test_prefix_preserving_trajectory_plan import inputs

import s2a.prefix_callback_backtracking_plan as frozen_planning


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("prefix_optimization_reserve_audit")


def test_real_source_proof_binds_only_source_bytes_and_explicit_scope(audit):
    proof = audit.optimization_reserve_source_proof()
    assert proof["inverse_ast_equal"] and proof["prefix_optimization_margin"] == 1e-4
    assert len(proof["source_refs"]) == 4
    for ref in proof["source_refs"]:
        assert hashlib.sha256(Path(ref["path"]).read_bytes()).hexdigest() == ref["sha256"]
    assert proof["source_refs"][0]["sha256"] == audit.BASELINE_SHA256
    assert "does not prove any main or numerical response equivalent" in proof["scope"]


def test_only_public_documentation_and_ast_neutral_formatting_are_ignored(audit):
    tree = ast.parse(audit.RESERVE_SOURCE.read_bytes())
    tree.body[0].value = ast.Constant("Alternative module description")
    solver = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    solver.body[0].value = ast.Constant("Alternative public solver description")
    audit._prove_inverse_ast(audit.BASELINE_SOURCE.read_bytes(), ast.unparse(tree).encode())


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("PREFIX_OPTIMIZATION_MARGIN = 1e-4", "PREFIX_OPTIMIZATION_MARGIN = 2e-4"),
        ("PREFIX_OPTIMIZATION_MARGIN = 1e-4", "PREFIX_OPTIMIZATION_MARGIN = 1e-4\nPREFIX_OPTIMIZATION_MARGIN = 1e-4"),
        ("PREFIX_OPTIMIZATION_MARGIN = 1e-4", ""),
        ("1.0 - PREFIX_OPTIMIZATION_MARGIN - normalized[:cursor].ravel()", "1.0 - normalized[:cursor].ravel()"),
        ("1.0 + z[width] - normalized[cursor]", "1.0 - PREFIX_OPTIMIZATION_MARGIN + z[width] - normalized[cursor]"),
        ("np.c_[-dp[: 2 * cursor]", "np.c_[dp[: 2 * cursor]"),
        ("safety_buffer = 1e-8", "safety_buffer = 1e-10"),
        ("/ 0.01**2", "/ 0.010001**2"),
        ('candidate["prefix_length"] < cursor', 'candidate["prefix_length"] < cursor - 1'),
        ("range(9)", "range(10)"),
        ("prefix_optimization_margin=PREFIX_OPTIMIZATION_MARGIN", "prefix_optimization_margin=1e-4"),
        ("max_evaluations=4000", "max_evaluations=4001"),
        ("import numpy as np", "import os\nimport numpy as np"),
        (
            "    def witness_rank(record):\n",
            '    def witness_rank(record):\n        "New local documentation is not whitelisted."\n',
        ),
    ],
)
def test_any_extra_or_missing_registered_ast_change_is_rejected(audit, before, after):
    baseline = audit.BASELINE_SOURCE.read_bytes()
    reserve = audit.RESERVE_SOURCE.read_text()
    assert reserve.count(before) == 1
    altered = reserve.replace(before, after, 1).encode()
    with pytest.raises(ValueError):
        audit._prove_inverse_ast(baseline, altered)


def test_frozen_baseline_cannot_be_replaced_with_a_matching_edited_pair(audit):
    baseline = audit.BASELINE_SOURCE.read_bytes().replace(b"safety_buffer = 1e-8", b"safety_buffer = 1e-10")
    reserve = audit.RESERVE_SOURCE.read_bytes().replace(b"safety_buffer = 1e-8", b"safety_buffer = 1e-10")
    with pytest.raises(ValueError, match="baseline source SHA"):
        audit._prove_inverse_ast(baseline, reserve)


def test_source_proof_rejects_unpinned_ledger_even_if_solver_ast_matches(audit, monkeypatch, tmp_path):
    changed = tmp_path / "ledger.py"
    changed.write_bytes(audit.FROZEN_LEDGER_SOURCE.read_bytes() + b"\n# changed\n")
    monkeypatch.setattr(audit, "FROZEN_LEDGER_SOURCE", changed)
    with pytest.raises(ValueError, match="callback ledger source SHA"):
        audit.optimization_reserve_source_proof()


def near_boundary_witness(monkeypatch):
    monkeypatch.setattr(
        frozen_planning,
        "minimize",
        lambda *args, **kwargs: pytest.fail("A full original-cap witness needs no optimizer"),
    )
    values = inputs()
    values["desired"][:, 0, 3] = 0.0099999
    answer = frozen_planning.solve_prefix_callback_backtracking_response(**values)
    answer["prefix_optimization_margin"] = 1e-4
    return answer, classifications(answer, values)


def test_original_cap_witness_is_not_rejected_for_missing_optimization_reserve(audit, monkeypatch):
    answer, metrics = near_boundary_witness(monkeypatch)
    assert np.all(np.asarray(answer["selected"]["normalized_squared_errors"])[:, 0] > 1 - 1e-4)
    assert np.all(np.asarray(answer["selected"]["normalized_squared_errors"]) <= 1)
    checked = audit.audit_optimization_reserve_ledger(answer, metrics)
    assert checked["witness_indices"] == [0] and checked["termination_reason"] == "witness"
    assert checked["total_iterations"] == 0 and checked["accepted_anchor_indices"] == [0]


@pytest.mark.parametrize("margin", [None, 0, 0.00010000000000000002, float("nan")])
def test_margin_metadata_must_be_the_exact_fixed_float(audit, monkeypatch, margin):
    answer, metrics = near_boundary_witness(monkeypatch)
    if margin is None:
        answer.pop("prefix_optimization_margin")
    else:
        answer["prefix_optimization_margin"] = margin
    with pytest.raises(ValueError, match="optimizer-only prefix margin"):
        audit.audit_optimization_reserve_ledger(answer, metrics)


def test_wrapper_keeps_original_first_eligible_ray_and_rejects_forged_ledger(audit, monkeypatch):
    monkeypatch.setattr(frozen_planning, "minimize", capture_trial)
    values = inputs()
    values["desired"][1, 0, 3] = 0.1
    answer = frozen_planning.solve_prefix_callback_backtracking_response(**values, max_stages=1)
    answer["prefix_optimization_margin"] = 1e-4
    metrics = classifications(answer, values)
    checked = audit.audit_optimization_reserve_ledger(answer, metrics)
    assert checked["accepted_anchor_indices"] == [0, 2]
    assert checked["witness_indices"] == [] and checked["total_backtracking_attempts"] == 2
    forged = deepcopy(answer)
    forged["phases"][0]["ray_attempts"][-1]["alpha"] = 0.25
    with pytest.raises(ValueError):
        audit.audit_optimization_reserve_ledger(forged, metrics)
