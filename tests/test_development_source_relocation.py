"""Prove finite source edits are reversible and preserve the numerical program."""

import ast

import pytest

from s2a.clean_training_state_handoff import BUNDLE
from s2a.development_source_relocation import FILES, PATH_PARTS, transform
from s2a.durable_stage_supervisor import REPO

NEW = "outputs/clean_training_development_dependency_repair_20261024/source"
OLD = str(BUNDLE.relative_to(REPO))
TEMPLATE = str(
    REPO / "outputs/clean_training_development_dependency_repair_20261024/inputs/libero_wam_effect_tool_relocated.json"
)


class Undo(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            for name in (*FILES, "continuation_runtime.py"):
                if node.value == NEW + "/" + name:
                    return ast.copy_location(ast.Constant(OLD + "/" + name), node)
        return node

    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Div) and isinstance(node.right, ast.Constant):
            for before, after in PATH_PARTS.items():
                if node.right.value == after:
                    node.right = ast.copy_location(ast.Constant(before), node.right)
        return node

    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        if node.name == "make_case_config":
            expected = ast.parse(f'value["policy_template"] = {TEMPLATE!r}').body[0]
            found = [s for s in node.body if ast.dump(s) == ast.dump(expected)]
            assert len(found) == 1
            node.body = [s for s in node.body if s not in found]
        return node


@pytest.mark.parametrize("name", FILES)
def test_actual_frozen_source_diff_contains_only_declared_metadata_edits(name):
    original = (BUNDLE / name).read_text()
    generated, changes = transform(original, filename=name, old_bundle=OLD, new_bundle=NEW, template=TEMPLATE)
    assert ast.dump(Undo().visit(ast.parse(generated))) == ast.dump(ast.parse(original))
    assert changes and {item["kind"] for item in changes} <= {"entrypoint", "output_path", "policy_template"}
    assert all(
        item["after"].startswith(
            (
                NEW,
                "development_dependency_repair_20261024",
                "orchestration/DEVELOPMENT_dependency_repair_20261024",
                "review/DEVELOPMENT_dependency_repair_20261024",
                str(REPO),
            )
        )
        for item in changes
    )
    compile(generated, name, "exec")


def test_registration_key_and_scalar_division_are_not_relocated():
    code = """def example(reg, root, value):
    assert reg["development"]["port"] == 31339
    return root / "development", value / 20
"""
    generated, _ = transform(code, filename=FILES[1], old_bundle=OLD, new_bundle=NEW, template=TEMPLATE)
    tree = ast.parse(generated)
    original = ast.parse(code)
    assert ast.dump(tree.body[0].body[0]) == ast.dump(original.body[0].body[0])
    assert ast.dump(tree.body[0].body[1].value.elts[1]) == ast.dump(original.body[0].body[1].value.elts[1])


def test_unknown_entrypoint_is_refused():
    with pytest.raises(ValueError, match="only four"):
        transform("pass", filename="other.py", old_bundle=OLD, new_bundle=NEW, template=TEMPLATE)


def test_generator_refuses_main_runtime_before_touching_outputs():
    import os
    import subprocess
    import sys

    if sys.version_info[:2] == (3, 11):
        pytest.skip("this regression checks the incompatible main interpreter")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="")
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts/prepare_wam_clean_development_sources.py")],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert out.returncode != 0
    assert "source generation requires actual Python3.11 runtime" in out.stderr
