"""Replay the frozen H12 audit with a verified full-model asset snapshot.

Only prerequisite admission and the new output directory change in the copied
main. The independent dynamics closure receives the complete migrated XML;
its numerical functions, comparisons, trace coverage and accounting are frozen.
The original checker module and its failed result are never modified.
"""

import argparse
import ast
import copy
import hashlib
from pathlib import Path

import check_tracking_v5_prefix_trust_control as frozen

OLD_SUFFIX = "_check_20261028"
NEW_SUFFIX = "_check_v2_20261029"


def load_asset_repair_prerequisites():
    from s2a.prefix_trust_asset_repair import load_asset_repair_prerequisites as load

    return load()


def load_snapshot():
    from s2a.prefix_trust_asset_repair import load_snapshot as load

    return load()


def case_engine(name, original, source_reg, prefix_check_reg, result, refs, compare):
    """Replace only the verified XML locator, retaining every original input."""
    snapshot = load_snapshot()
    original_xml, migrated_xml = snapshot["original_xml_ref"], snapshot["migrated_xml_ref"]
    frozen.require(source_reg["xml_ref"] == original_xml, "Snapshot binds this exact original full XML")
    for ref in (
        original_xml,
        migrated_xml,
        snapshot["historical_manifest_ref"],
        snapshot["snapshot_ref"],
        snapshot["compile_ref"],
    ):
        frozen.require(refs.get(ref["path"]) == ref, "Asset loading evidence already bound by repair admission")
    migrated_reg = dict(source_reg, xml_ref=copy.deepcopy(migrated_xml))
    result["asset_loading_proof"] = dict(
        original_xml_ref=copy.deepcopy(original_xml),
        migrated_xml_ref=copy.deepcopy(migrated_xml),
        historical_manifest_ref=copy.deepcopy(snapshot["historical_manifest_ref"]),
        snapshot_ref=copy.deepcopy(snapshot["snapshot_ref"]),
        compile_ref=copy.deepcopy(snapshot["compile_ref"]),
        asset_count=len(snapshot["assets"]),
        complete_model_preserved=True,
        model_body_or_geom_pruning=False,
        main_ast_proof=main_ast_proof(),
    )
    return frozen.case_engine(name, original, migrated_reg, prefix_check_reg, result, refs, compare)


class _MainRewrite(ast.NodeTransformer):
    def __init__(self, *, inverse=False):
        self.old_call, self.new_call = "load_prerequisites", "load_asset_repair_prerequisites"
        self.old_suffix, self.new_suffix = OLD_SUFFIX, NEW_SUFFIX
        if inverse:
            self.old_call, self.new_call = self.new_call, self.old_call
            self.old_suffix, self.new_suffix = self.new_suffix, self.old_suffix
        self.calls = self.outputs = 0

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == self.old_call:
            frozen.require(not node.args and not node.keywords, "The original admission call has no arguments")
            node.func.id = self.new_call
            self.calls += 1
        return node

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "output":
            frozen.require(
                isinstance(node.value, ast.BinOp)
                and isinstance(node.value.op, ast.Div)
                and isinstance(node.value.left, ast.Name)
                and node.value.left.id == "REPO"
                and isinstance(node.value.right, ast.JoinedStr),
                "Frozen output remains one repository-relative formatted path",
            )
            for part in node.value.right.values:
                if isinstance(part, ast.Constant) and part.value == self.old_suffix:
                    part.value = self.new_suffix
                    self.outputs += 1
        return node


def _rewritten_main():
    source = Path(frozen.__file__).resolve()
    payload = source.read_bytes()
    functions = [node for node in ast.parse(payload).body if isinstance(node, ast.FunctionDef) and node.name == "main"]
    frozen.require(len(functions) == 1, "Exactly one frozen checker main")
    original = functions[0]
    rewrite = _MainRewrite()
    changed = rewrite.visit(copy.deepcopy(original))
    frozen.require(rewrite.calls == rewrite.outputs == 1, "Exactly one admission and one output suffix replacement")
    inverse = _MainRewrite(inverse=True)
    restored = inverse.visit(copy.deepcopy(changed))
    frozen.require(
        inverse.calls == inverse.outputs == 1
        and ast.dump(restored, include_attributes=False) == ast.dump(original, include_attributes=False),
        "Inverse transformation preserves the entire frozen main AST",
    )
    proof = dict(
        source_ref=dict(path=str(source), sha256=hashlib.sha256(payload).hexdigest()),
        admission_call_replacements=rewrite.calls,
        output_suffix_replacements=rewrite.outputs,
        original_output_suffix=OLD_SUFFIX,
        migrated_output_suffix=NEW_SUFFIX,
        inverse_ast_equal=True,
    )
    return changed, proof


def main_ast_proof():
    """Describe and validate the two changes without loading a model or assets."""
    return _rewritten_main()[1]


def build_main():
    """Compile an isolated main; retain the frozen module's own global namespace."""
    changed, _ = _rewritten_main()
    namespace = dict(vars(frozen))
    namespace.update(case_engine=case_engine, load_asset_repair_prerequisites=load_asset_repair_prerequisites)
    tree = ast.fix_missing_locations(ast.Module(body=[changed], type_ignores=[]))
    exec(compile(tree, str(Path(__file__).resolve()), "exec"), namespace)
    return namespace["main"]


def main(case, dry=False):
    return build_main()(case, dry)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=frozen.CASES, required=True)
    parser.add_argument("--dry", action="store_true", help="Verify metadata only; no model loading or responses")
    arguments = parser.parse_args()
    main(arguments.case, arguments.dry)
