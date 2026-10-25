"""Finite051 metadata/path edits to frozen development entrypoints, never math."""

import ast
import copy

from s2a.durable_stage_supervisor import require

FILES = (
    "run_wam_clean_training_development.py",
    "serve_wam_clean_training.py",
    "execute_wam_clean_training_case.py",
    "verify_wam_clean_training_development.py",
)
NAMESPACE = "development_dependency_repair_20261024"
PATH_PARTS = {
    "development": NAMESPACE,
    "development/cases": NAMESPACE + "/cases",
    "development/service": NAMESPACE + "/service",
    "orchestration/DEVELOPMENT": "orchestration/DEVELOPMENT_dependency_repair_20261024",
    "review/DEVELOPMENT_verified.json": "review/DEVELOPMENT_dependency_repair_20261024_verified.json",
}


class Relocate(ast.NodeTransformer):
    def __init__(self, old_bundle, new_bundle, template):
        self.paths = {old_bundle + "/" + name: new_bundle + "/" + name for name in (*FILES, "continuation_runtime.py")}
        self.template = template
        self.changes = []

    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value in self.paths:
            self.changes.append(dict(kind="entrypoint", before=node.value, after=self.paths[node.value]))
            return ast.copy_location(ast.Constant(self.paths[node.value]), node)
        return node

    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Div) and isinstance(node.right, ast.Constant) and node.right.value in PATH_PARTS:
            value = node.right.value
            self.changes.append(dict(kind="output_path", before=value, after=PATH_PARTS[value]))
            node.right = ast.copy_location(ast.Constant(PATH_PARTS[value]), node.right)
        return node

    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        if node.name == "make_case_config":
            anchors = [
                i
                for i, statement in enumerate(node.body)
                if isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id == "value"
                and ast.dump(statement.value) == ast.dump(ast.parse("copy.deepcopy(base)", mode="eval").body)
            ]
            require(len(anchors) == 1, "frozen make_case_config anchor differs")
            statement = ast.Assign(
                targets=[
                    ast.Subscript(
                        value=ast.Name(id="value", ctx=ast.Load()),
                        slice=ast.Constant("policy_template"),
                        ctx=ast.Store(),
                    )
                ],
                value=ast.Constant(self.template),
            )
            node.body.insert(anchors[0] + 1, statement)
            self.changes.append(dict(kind="policy_template", after=self.template))
        return node


def transform(source, *, filename, old_bundle, new_bundle, template):
    require(filename in FILES, "only four frozen development entrypoints may be relocated")
    require(old_bundle != new_bundle and new_bundle.startswith("outputs/"), "new bundle must be explicit S2A output")
    original = ast.parse(source)
    worker = Relocate(old_bundle, new_bundle, template)
    generated = ast.fix_missing_locations(worker.visit(copy.deepcopy(original)))
    require(worker.changes, "no finite relocation edits found")
    template_edits = sum(item["kind"] == "policy_template" for item in worker.changes)
    require(template_edits == int(filename == FILES[0]), "unexpected template edit count")
    return ast.unparse(generated) + "\n", worker.changes
