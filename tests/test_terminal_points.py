import ast
import importlib

import pytest

from s2a.artifacts import REPO


def test_complete_point_checker_changes_only_explicit_per_node_head(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    helper = importlib.import_module("verify_wam_terminal_points")
    baseline = ast.parse((REPO / "scripts/verify_wam_clean_training_state.py").read_text())
    expected = next(n for n in baseline.body if isinstance(n, ast.FunctionDef) and n.name == "compact_points")

    class SelectHead(ast.NodeTransformer):
        def visit_keyword(self, node):
            if node.arg == "head" and isinstance(node.value, ast.Name) and node.value.id == "head":
                node.value = ast.parse("step_head(head, i)", mode="eval").body
            return self.generic_visit(node)

    actual = ast.parse((REPO / "scripts/verify_wam_terminal_points.py").read_text())
    actual = next(n for n in actual.body if isinstance(n, ast.FunctionDef) and n.name == "compact_points")
    assert ast.dump(SelectHead().visit(expected)) == ast.dump(actual)
    first, last = object(), object()
    assert all(helper.step_head((first, last), i) is first for i in range(34))
    assert helper.step_head((first, last), 34) is last
    assert helper.step_head(None, 34) is None
    with pytest.raises(ValueError):
        helper.step_head((first, last), 35)
    with pytest.raises(ValueError):
        helper.step_head(first, 34)
