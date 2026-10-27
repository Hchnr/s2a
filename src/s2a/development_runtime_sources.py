"""Finite051 addition of source-ledger admission to the independent audit."""

import ast

from s2a.durable_stage_supervisor import require


def overlay_auditor(source):
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "admission"]
    require(len(functions) == 1, "one frozen auditor admission required")
    function = functions[0]
    loops = [
        n
        for n in function.body
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "required"
    ]
    require(len(loops) == 1, "auditor source dependency loop differs")
    loop = loops[0]

    class Ledger(ast.NodeTransformer):
        def visit_Subscript(self, node):
            if ast.dump(node) == ast.dump(ast.parse('reg["source_sha256"]', mode="eval").body):
                return ast.copy_location(ast.Name(id="repair_sources", ctx=ast.Load()), node)
            return self.generic_visit(node)

    index = function.body.index(loop)
    function.body.insert(index, ast.parse('repair_sources = current_sources(reg["source_sha256"])').body[0])
    Ledger().visit(loop)
    returns = [n for n in function.body if isinstance(n, ast.Return)]
    require(len(returns) == 1, "auditor admission return differs")
    function.body.insert(function.body.index(returns[0]), ast.parse("sources = current_sources(sources)").body[0])
    index = 1 if isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) else 0
    tree.body.insert(index, ast.parse("from s2a.development_repair_admission import current_sources").body[0])
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def runtime_bridge_source(original):
    return f'''"""051 explicit admission/source overlay; all other runtime functions remain original."""
from s2a.clean_training_import_bridge import load_frozen_delegate
from s2a.development_runtime_bridge import admit_runtime
from s2a.development_repair_admission import current_qualification
_original = load_frozen_delegate({original!r}, name="_s2a051_original_runtime")
for _name in dir(_original):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_original, _name)
def admit_execution(prepared_ref, registration_ref, *, required_sources):
    return admit_runtime(_original, prepared_ref, registration_ref,
                         required_sources=required_sources, qualification=current_qualification())
'''


def verify_generated(generation, *, old_bundle, new_bundle, template, repo):
    from pathlib import Path

    from s2a.development_source_relocation import FILES, transform
    from s2a.durable_stage_supervisor import reference

    rows = generation["generated"]
    require(
        [Path(row["generated_ref"]["path"]).name for row in rows] == [*FILES, "continuation_runtime.py"],
        "051 five finite sources required",
    )
    for row in rows:
        new = Path(row["generated_ref"]["path"])
        old = Path(row["original_ref"]["path"])
        require(
            new == new_bundle / new.name and old == old_bundle / new.name, "051 generated/original source route differs"
        )
        require(
            reference(new) == row["generated_ref"] and reference(old) == row["original_ref"],
            "051 generated/original source hash differs",
        )
        if new.name == "continuation_runtime.py":
            expected = runtime_bridge_source(row["original_ref"])
        else:
            expected, _ = transform(
                old.read_text(),
                filename=new.name,
                old_bundle=str(old_bundle.relative_to(repo)),
                new_bundle=str(new_bundle.relative_to(repo)),
                template=template,
            )
            if new.name == FILES[-1]:
                expected = overlay_auditor(expected)
        require(
            ast.dump(ast.parse(new.read_text())) == ast.dump(ast.parse(expected)),
            "051 actual program differs from finite metadata transform",
        )
