"""Keep complete numerical replay and original development outcome gates."""

import ast
from pathlib import Path

import pytest
import verify_serialized_preview_wam_development as verifier
from serialized_preview_wam_audit_admission import admission

ROOT = Path(__file__).resolve().parents[1]


def cases(failed):
    return [
        dict(case=case, outcome=verifier.outcome([True] * 560 + [case["case_index"] not in failed] * 40))
        for case in verifier.order()
    ]


def test_terminal_candidate_requires_five_of_six_stable_cases():
    assert verifier.score(cases({11}))["eligible_for_later_fresh_candidate_freeze"]
    assert not verifier.score(cases({9, 11}))["eligible_for_later_fresh_candidate_freeze"]
    assert not verifier.score(cases({9, 11}))["fresh_performance_authorized"]


def test_partial_or_fabricated_outcome_is_not_complete_physics():
    for rows in ([True] * 599, [1] * 600):
        with pytest.raises(ValueError):
            verifier.outcome(rows)
    with pytest.raises(ValueError):
        verifier.score(cases(set())[:-1])


def test_independent_admission_rejects_old_schema_before_reading_prerequisites():
    class Evidence:
        def json(self, ref):
            return dict(schema_version="s2a.terminal-development-preparation.v1")

    with pytest.raises(ValueError, match="execution identity"):
        admission(
            Evidence(),
            {},
            {},
            {},
            require=verifier.require,
            normal_owned=verifier.normal_owned,
            gpu=verifier.GPU,
            order=verifier.order,
        )


def test_all_per_request_numerical_and_pairing_checks_preserved():
    bodies = []
    for name in ("verify_wam_terminal_development.py", "verify_serialized_preview_wam_development.py"):
        tree = ast.parse((ROOT / "scripts" / name).read_text())
        loops = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "index"
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
            and len(node.iter.args) == 1
            and isinstance(node.iter.args[0], ast.Constant)
            and node.iter.args[0].value == 75
        ]
        assert len(loops) == 1
        bodies.append(ast.dump(loops[0], include_attributes=False))
    assert bodies[0] == bodies[1]
