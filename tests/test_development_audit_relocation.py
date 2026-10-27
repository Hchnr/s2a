"""Reject controller drift while permitting a proven mount relocation."""

import ast
import json

import pytest

from s2a.development_audit_relocation import audit_tree, equivalent_template
from s2a.durable_stage_supervisor import reference


def migration(tmp_path):
    def write(name, value):
        p = tmp_path / name
        p.write_text(json.dumps(value))
        return reference(p)

    tool = write("tool.json", {"unit": "m"})
    old = write("old.json", {"tool_contract": "/unavailable/historical.json", "gain": 1})
    new = write("new.json", {"tool_contract": tool["path"], "gain": 1})
    historical = write("manifest.json", {"tool_sha256": tool["sha256"]})
    return dict(
        old_template_ref=old,
        new_template_ref=new,
        safe_copy_ref=tool,
        historical_sha256=tool["sha256"],
        historical_proof_ref=historical,
        only_template_field_changed="tool_contract",
    )


def test_proven_migration_and_controller_drift(tmp_path):
    m = migration(tmp_path)
    assert equivalent_template(m) == m["new_template_ref"]["path"]
    p = tmp_path / "new.json"
    value = json.loads(p.read_text())
    value["gain"] = 2
    p.write_text(json.dumps(value))
    m["new_template_ref"] = reference(p)
    with pytest.raises(ValueError, match="controller parameters"):
        equivalent_template(m)


def test_historical_manifest_must_attest_contract(tmp_path):
    m = migration(tmp_path)
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"tool_sha256": "0" * 64}))
    m["historical_proof_ref"] = reference(p)
    with pytest.raises(ValueError, match="historical manifest"):
        equivalent_template(m)


def test_ast_preserves_checks_and_score():
    source = """def score(cases):
    return len(cases) == 12

def audit(args):
    base = physics.json(REPO / 'configs/wam_short_replan_8_alpha1_20261021.json')
    assert config == base
    return score(cases)
"""
    tree = audit_tree(source, "/safe/template.json")
    body = tree.body[1].body
    assert ast.unparse(body.pop(1)) == "base['policy_template'] = '/safe/template.json'"
    assert ast.dump(tree) == ast.dump(ast.parse(source))
    with pytest.raises(ValueError, match="anchor"):
        audit_tree(source.replace("base =", "other ="), "/safe/template.json")


def test_seed_overlay_preserves_every_original_statement():
    source = """def audit(args):
    prepared, reg, run, state, control, sources = admission(ev, refs['prepared'], refs['execution_registration'], refs['development'])
    base = physics.json(REPO / 'configs/wam_short_replan_8_alpha1_20261021.json')
    assert trace['noise_roster'] == prepared['noise_roster']
    return score(cases)
"""
    tree = audit_tree(source, "/safe/template.json", seed_overlay=True)
    body = tree.body[0].body
    assert ast.unparse(body.pop(1)) == "prepared = apply_registered_seed_plan(prepared, reg, ev, refs)"
    assert ast.unparse(body.pop(2)) == "base['policy_template'] = '/safe/template.json'"
    assert ast.dump(tree) == ast.dump(ast.parse(source))
    with pytest.raises(ValueError, match="admission anchor"):
        audit_tree(source.replace("= admission(", "= different("), "/safe/template.json", seed_overlay=True)
