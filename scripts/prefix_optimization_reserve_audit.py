"""Prove the registered solver-only AST delta; retain the frozen ray ledger.

No producer solver is imported or executed. Numerical replay remains the
caller's responsibility, including every completed ray and accepted anchor.
"""

import ast
import hashlib
from pathlib import Path

from check_tracking_v5_prefix_trust_control import numerical_compare
from prefix_callback_backtracking_audit import audit_callback_backtracking_ledger

REPO = Path(__file__).resolve().parents[1]
BASELINE_SOURCE = REPO / "src/s2a/prefix_callback_backtracking_plan.py"
RESERVE_SOURCE = REPO / "src/s2a/prefix_optimization_reserve_plan.py"
BASELINE_SHA256 = "46e78f95e8725bba5d98236f31a33140ed138e9f8d2755b3593b74ea75401460"
FROZEN_LEDGER_SOURCE = REPO / "scripts/prefix_callback_backtracking_audit.py"
FROZEN_LEDGER_SHA256 = "122a2a31a0b6a0d2f4d18b599ff73747ec4c2f26b07e2e2a4560e94b3964e211"
PREFIX_OPTIMIZATION_MARGIN = 1e-4
OLD_NAME = "solve_prefix_callback_backtracking_response"
NEW_NAME = "solve_prefix_optimization_reserve_response"
OLD_EXPRESSION = "1.0 - normalized[:cursor].ravel()"
NEW_EXPRESSION = "1.0 - PREFIX_OPTIMIZATION_MARGIN - normalized[:cursor].ravel()"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dump(node):
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _public_function(tree, name):
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    require(len(matches) == 1, "Exactly one declared public solver")
    return matches[0]


def _strip_docstring(node):
    require(
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str),
        "Only the existing module and public solver docstrings may change",
    )
    node.body.pop(0)


def _prove_inverse_ast(baseline_bytes, reserve_bytes):
    """Inspect bytes only; invert four unique changes and compare the full AST."""
    require(hashlib.sha256(baseline_bytes).hexdigest() == BASELINE_SHA256, "Frozen baseline source SHA mismatch")
    old, new = ast.parse(baseline_bytes), ast.parse(reserve_bytes)
    old_solver, new_solver = _public_function(old, OLD_NAME), _public_function(new, NEW_NAME)
    new_solver.name = OLD_NAME

    assignment = ast.parse("PREFIX_OPTIMIZATION_MARGIN = 1e-4").body[0]
    assignments = [node for node in ast.walk(new) if dump(node) == dump(assignment)]
    require(len(assignments) == 1 and assignments[0] in new.body, "One exact top-level fixed margin assignment")
    position = new.body.index(assignments[0])
    require(
        position + 1 < len(new.body)
        and isinstance(new.body[position + 1], ast.Assign)
        and dump(new.body[position + 1].targets[0]) == dump(ast.Name(id="RADII", ctx=ast.Store())),
        "The sole new constant precedes the unchanged radius declaration",
    )
    new.body.pop(position)

    expression = ast.parse(NEW_EXPRESSION, mode="eval").body
    matches = [node for node in ast.walk(new) if dump(node) == dump(expression)]
    constraints = [
        node for node in ast.walk(new_solver) if isinstance(node, ast.FunctionDef) and node.name == "constraint"
    ]
    require(
        len(matches) == len(constraints) == 1 and any(node is matches[0] for node in ast.walk(constraints[0])),
        "Exactly one registered prefix expression inside the optimizer constraint",
    )

    class RestorePrefix(ast.NodeTransformer):
        def visit_BinOp(self, node):
            if node is matches[0]:
                return ast.parse(OLD_EXPRESSION, mode="eval").body
            return self.generic_visit(node)

    RestorePrefix().visit(new)
    final = new_solver.body[-1]
    require(
        isinstance(final, ast.Return)
        and isinstance(final.value, ast.Call)
        and isinstance(final.value.func, ast.Name)
        and final.value.func.id == "dict"
        and final.value.keywords,
        "The public solver retains its explicit final dict return",
    )
    expected_keyword = ast.keyword(
        arg="prefix_optimization_margin", value=ast.Name(id="PREFIX_OPTIMIZATION_MARGIN", ctx=ast.Load())
    )
    keywords = [
        node for node in ast.walk(new) if isinstance(node, ast.keyword) and node.arg == "prefix_optimization_margin"
    ]
    require(
        len(keywords) == 1 and keywords[0] is final.value.keywords[0] and dump(keywords[0]) == dump(expected_keyword),
        "One exact returned margin field, in the public solver's final dict",
    )
    final.value.keywords.pop(0)

    for tree, solver in ((old, old_solver), (new, new_solver)):
        _strip_docstring(tree)
        _strip_docstring(solver)
    require(
        dump(old) == dump(new),
        "Inverse AST mismatch: no other imports, expressions, branches, defaults, local docs or declarations may change",
    )


def optimization_reserve_source_proof():
    """Bind the complete source files to a strict inverse AST proof, without execution."""
    paths = (BASELINE_SOURCE, RESERVE_SOURCE, Path(__file__).resolve(), FROZEN_LEDGER_SOURCE)
    payloads = [path.read_bytes() for path in paths]
    refs = [
        dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest())
        for path, payload in zip(paths, payloads, strict=True)
    ]
    require(refs[-1]["sha256"] == FROZEN_LEDGER_SHA256, "Frozen callback ledger source SHA mismatch")
    _prove_inverse_ast(payloads[0], payloads[1])
    require(
        all(path.read_bytes() == payload for path, payload in zip(paths, payloads, strict=True)),
        "Proof sources unchanged while read",
    )
    return dict(
        source_refs=refs,
        inverse_ast_equal=True,
        prefix_optimization_margin=PREFIX_OPTIMIZATION_MARGIN,
        allowed_changes=[
            "Public solver function name and module/public-function docstrings",
            "One top-level PREFIX_OPTIMIZATION_MARGIN = 1e-4 declaration",
            "Only optimizer prefix residual 1-q becomes 1-PREFIX_OPTIMIZATION_MARGIN-q",
            "One final-return prefix_optimization_margin field referring to that constant",
        ],
        scope="Full solver-module inverse AST equality against the pinned frozen callback source; does not prove any main or numerical response equivalent, nor grant control qualification",
    )


def audit_optimization_reserve_ledger(answer, classifications, *, compare=numerical_compare):
    """Keep original caps, ray acceptance and global ranking after checking metadata."""
    require(
        type(answer.get("prefix_optimization_margin")) is float
        and answer["prefix_optimization_margin"] == PREFIX_OPTIMIZATION_MARGIN,
        "Exact registered optimizer-only prefix margin",
    )
    return audit_callback_backtracking_ledger(answer, classifications, compare=compare)
