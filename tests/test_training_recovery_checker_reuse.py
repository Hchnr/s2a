"""The063 checker changes admission/targets/budget, not control tolerances."""

import ast
from pathlib import Path


def function(path):
    tree = ast.parse(Path(path).read_text())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "verify_case")


def test_independent_checker_keeps_original_mechanical_body():
    old = function("scripts/wam_position_acceptance_checker.py")
    new = function("scripts/training_recovery_acceptance_checker.py")
    assert new.args.kwonlyargs.pop().arg == "oracle_reference"
    new.args.kw_defaults.pop()
    assert ast.unparse(new.body.pop(0)) == "assert source == 'oracle'"
    old_branch = next(n for n in old.body if isinstance(n, ast.If) and ast.unparse(n.test) == "source == 'oracle'")
    new_branch = next(n for n in new.body if isinstance(n, ast.If) and ast.unparse(n.test) == "source == 'oracle'")
    assert ast.dump(ast.Module(body=new_branch.body[:3], type_ignores=[])) == ast.dump(
        ast.Module(body=old_branch.body[:3], type_ignores=[])
    )
    new_branch.body = old_branch.body

    class Restore(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value == "src/s2a/training_recovery_simulation.py":
                node.value = "src/s2a/stability_position_simulation.py"
            return node

        def visit_Compare(self, node):
            if ast.unparse(node).startswith("len(rows) == result['steps']"):
                node.comparators[-1] = ast.Constant(600)
            return self.generic_visit(node)

        def visit_keyword(self, node):
            if node.arg in ("expected_steps", "steps") and ast.unparse(node.value) == "config['online_max_steps']":
                node.value = ast.Constant(600)
            return self.generic_visit(node)

    new = Restore().visit(new)
    assert ast.dump(new, include_attributes=False) == ast.dump(old, include_attributes=False)
