"""Explicit post-run template equivalence check; no performance authorization."""

import ast
import copy
from pathlib import Path

from s2a.durable_stage_supervisor import read, reference, require


def equivalent_template(migration):
    old = read(migration["old_template_ref"])
    new = read(migration["new_template_ref"])
    require(migration["only_template_field_changed"] == "tool_contract", "unexpected migration field")
    require(
        {k: v for k, v in old.items() if k != "tool_contract"}
        == {k: v for k, v in new.items() if k != "tool_contract"},
        "migration changed controller parameters",
    )
    safe = migration["safe_copy_ref"]
    require(reference(Path(new["tool_contract"])) == safe, "template does not use proven safe contract")
    require(safe["sha256"] == migration["historical_sha256"], "historical tool digest differs")
    historical = read(migration["historical_proof_ref"])
    require(historical["tool_sha256"] == safe["sha256"], "historical manifest tool digest differs")
    return migration["new_template_ref"]["path"]


def audit_tree(source, template, *, seed_overlay=False):
    """Insert one baseline path assignment; preserve every original audit statement."""
    tree = ast.parse(source)
    result = copy.deepcopy(tree)
    functions = [n for n in result.body if isinstance(n, ast.FunctionDef) and n.name == "audit"]
    require(len(functions) == 1, "audit function missing or duplicated")
    body = functions[0].body
    anchors = [
        i
        for i, n in enumerate(body)
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "base"
        and ast.unparse(n.value) == "physics.json(REPO / 'configs/wam_short_replan_8_alpha1_20261021.json')"
    ]
    require(len(anchors) == 1, "original baseline anchor differs")
    statement = ast.Assign(
        targets=[
            ast.Subscript(
                value=ast.Name(id="base", ctx=ast.Load()), slice=ast.Constant("policy_template"), ctx=ast.Store()
            )
        ],
        value=ast.Constant(template),
    )
    body.insert(anchors[0] + 1, statement)
    seed_index = None
    if seed_overlay:
        admissions = [
            i
            for i, n in enumerate(body)
            if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Name)
            and n.value.func.id == "admission"
        ]
        require(len(admissions) == 1, "original admission anchor differs")
        seed_index = admissions[0] + 1
        body.insert(seed_index, ast.parse("prepared = apply_registered_seed_plan(prepared, reg, ev, refs)").body[0])
    ast.fix_missing_locations(result)
    proof = copy.deepcopy(result)
    proof_audit = next(n for n in proof.body if isinstance(n, ast.FunctionDef) and n.name == "audit")
    if seed_index is not None:
        del proof_audit.body[seed_index]
    del proof_audit.body[anchors[0] + 1]
    require(ast.dump(proof) == ast.dump(tree), "supplement changed original audit math")
    return result
