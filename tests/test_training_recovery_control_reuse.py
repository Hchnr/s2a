"""Prove the recovery collector preserves every original control expression."""

import ast
from pathlib import Path


def test_control_body_exact_except_explicit_reference_and_observer_seams():
    original = ast.parse(Path("src/s2a/stability_position_simulation.py").read_text())
    fork = ast.parse(Path("src/s2a/training_recovery_simulation.py").read_text())
    old = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "run_position_case")
    new = next(n for n in fork.body if isinstance(n, ast.FunctionDef) and n.name == "run_training_recovery_case")
    assert [a.arg for a in new.args.kwonlyargs[-2:]] == ["oracle_reference", "endpoint_observer"]
    new.args.kwonlyargs = new.args.kwonlyargs[:-2]
    new.args.kw_defaults = new.args.kw_defaults[:-2]
    new.name = old.name
    new.body[0] = old.body[0]
    guard = new.body.pop(1)
    assert isinstance(guard, ast.If) and "endpoint_observer" in ast.unparse(guard.test)
    calls = []
    references = []

    class Restore(ast.NodeTransformer):
        def visit_Expr(self, node):
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "endpoint_observer"
            ):
                calls.append(ast.unparse(node))
                return None
            return self.generic_visit(node)

        def visit_Assign(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == "oracle_reference":
                references.append(ast.unparse(node))
                node.value = ast.parse("_read_oracle(config, record, sampling_seed)", mode="eval").body
            return self.generic_visit(node)

    new = Restore().visit(new)
    assert calls == ["endpoint_observer(env, 0)", "endpoint_observer(env, step + 1)"]
    assert references == ["reference, reference_opening, oracle_signatures = oracle_reference"]
    assert ast.dump(new, include_attributes=False) == ast.dump(old, include_attributes=False)
