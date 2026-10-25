"""Finite auditor overlay and independent whole-AST regeneration proof."""

import ast
from pathlib import Path

import pytest

from s2a.clean_training_state_handoff import BUNDLE
from s2a.development_runtime_sources import overlay_auditor, runtime_bridge_source, verify_generated
from s2a.development_source_relocation import FILES, transform
from s2a.durable_stage_supervisor import reference


class UndoOverlay(ast.NodeTransformer):
    def visit_Module(self, node):
        node.body = [
            s for s in node.body if not isinstance(s, ast.ImportFrom) or s.module != "s2a.development_repair_admission"
        ]
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if node.name == "admission":
            expected = [
                ast.dump(ast.parse(s).body[0])
                for s in (
                    'repair_sources = current_sources(reg["source_sha256"])',
                    "sources = current_sources(sources)",
                )
            ]
            node.body = [s for s in node.body if ast.dump(s) not in expected]
        return self.generic_visit(node)

    def visit_Name(self, node):
        if node.id == "repair_sources":
            return ast.copy_location(ast.parse('reg["source_sha256"]', mode="eval").body, node)
        return node


def test_overlay_reverses_exactly_to_the_frozen_whole_program():
    original = (BUNDLE / FILES[-1]).read_text()
    changed = overlay_auditor(original)
    assert ast.dump(UndoOverlay().visit(ast.parse(changed))) == ast.dump(ast.parse(original))


@pytest.fixture
def generated(tmp_path):
    old = tmp_path / "old"
    new = tmp_path / "outputs/new"
    old.mkdir()
    new.mkdir(parents=True)
    rows = []
    template = str(tmp_path / "template.json")
    for name in FILES:
        path = old / name
        path.write_text((BUNDLE / name).read_text())
        code, _ = transform(
            path.read_text(), filename=name, old_bundle="old", new_bundle="outputs/new", template=template
        )
        if name == FILES[-1]:
            code = overlay_auditor(code)
        (new / name).write_text(code)
        rows.append(dict(original_ref=reference(path), generated_ref=reference(new / name)))
    path = old / "continuation_runtime.py"
    path.write_text((BUNDLE / path.name).read_text())
    (new / path.name).write_text(runtime_bridge_source(reference(path)))
    rows.append(dict(original_ref=reference(path), generated_ref=reference(new / path.name)))
    return dict(generated=rows), dict(old_bundle=old, new_bundle=new, template=template, repo=tmp_path)


def test_independent_finite_ast_verifier_accepts_only_the_declared_program(generated):
    manifest, kwargs = generated
    verify_generated(manifest, **kwargs)


def test_forged_hash_cannot_hide_a_score_change(generated):
    manifest, kwargs = generated
    row = manifest["generated"][3]
    path = Path(row["generated_ref"]["path"])
    text = path.read_text()
    assert ">= 5" in text
    path.write_text(text.replace(">= 5", ">= 1"))
    row["generated_ref"] = reference(path)
    with pytest.raises(ValueError, match="actual program differs"):
        verify_generated(manifest, **kwargs)
