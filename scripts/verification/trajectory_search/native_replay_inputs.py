"""Bind the historical AST helper to an explicitly approved current module."""

import ast
import copy
import sys
from pathlib import Path
from types import FunctionType, ModuleType

from s2a.infrastructure.current_layout import CurrentLayout

HELPER_IDENTITY = "scripts/check_tracking_v5_state_updated_response.py"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def module_name(relative):
    path = Path(relative)
    if path.suffix != ".py":
        return None
    parts = path.with_suffix("").parts
    if parts[0] == "src":
        parts = parts[1:]
    elif parts[0] != "scripts":
        return None
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def namespace_map(context, layout):
    require(len(context.migration_refs) == 1, "One approved cumulative current manifest required")
    manifest = context.read_current_json(context.migration_refs[0])
    require(manifest.get("script_namespace") == "canonical_scripts", "Canonical current script namespace required")
    mapping = {}
    for move in manifest["moves"]:
        entry = layout.entry(move["old_path"])
        old = module_name(move["old_path"])
        current = module_name(entry["current_ref"]["path"])
        if old is None or current is None:
            continue
        old_path = Path(move["old_path"])
        aliases = (
            (old, old_path.stem)
            if old_path.parts[0] == "scripts" and len(old_path.parts) == 2 and old_path.stem != "__init__"
            else (old,)
        )
        for name in aliases:
            require(name not in mapping or mapping[name] == current, "Ambiguous approved import namespace")
            mapping[name] = current
    return mapping


def module_ast_identity(historical, current, mapping):
    """Rewrite import names only; every other statement remains mandatory."""
    original = ast.parse(historical)
    actual = ast.parse(current)
    changes = []

    class Namespace(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if not node.level and node.module in mapping:
                previous = node.module
                node.module = mapping[previous]
                if previous != node.module:
                    changes.append(dict(original=previous, current=node.module))
            return self.generic_visit(node)

        def visit_Import(self, node):
            for alias in node.names:
                if alias.name in mapping:
                    previous = alias.name
                    alias.name = mapping[previous]
                    if previous != alias.name:
                        changes.append(dict(original=previous, current=alias.name))
            return node

    expected = Namespace().visit(copy.deepcopy(original))

    def ordered(tree):
        body, index = [], 0
        while index < len(tree.body):
            if isinstance(tree.body[index], (ast.Import, ast.ImportFrom)):
                imports = []
                while index < len(tree.body) and isinstance(tree.body[index], (ast.Import, ast.ImportFrom)):
                    imports.append(tree.body[index])
                    index += 1
                body.extend(sorted(imports, key=lambda node: ast.dump(node, include_attributes=False)))
            else:
                body.append(tree.body[index])
                index += 1
        tree.body = body
        return ast.dump(tree, include_attributes=False)

    require(ordered(expected) == ordered(actual), "AST helper whole module changed beyond approved import namespaces")
    return dict(
        full_module_namespace_ast_equal=True,
        namespace_changes=changes,
        import_order_scope="Contiguous module import blocks only; all other AST remains exact",
    )


def bind_native_ast_helper(context, declared_ref, native_functions):
    """Verify original Git bytes and current source separately, executing neither."""
    expected = context.historical_reference(HELPER_IDENTITY)
    require(declared_ref == expected, "Original AST helper declared path/SHA differs")
    historical = context.historical_bytes(declared_ref)
    require(len(context.migration_refs) == 1, "One approved cumulative current manifest required")
    layout = CurrentLayout(
        root=context.root,
        migration_ref=context.migration_refs[0],
        catalog_ref=context.catalog_ref,
        baseline_commit=context.locator.baseline_commit,
    )
    path = layout.source_path(HELPER_IDENTITY)
    current_ref = context.bind_current_source(path)
    current = path.read_bytes()
    proof = module_ast_identity(historical, current, namespace_map(context, layout))
    expected_module = module_name(layout.entry(HELPER_IDENTITY)["current_ref"]["path"])
    module = sys.modules.get(expected_module)
    require(isinstance(module, ModuleType), "Current AST helper canonical module is not loaded")
    namespace = vars(module)
    spec = namespace.get("__spec__")
    require(
        namespace.get("__name__") == expected_module
        and namespace.get("__file__") == str(path)
        and spec is not None
        and spec.name == expected_module
        and spec.origin == str(path),
        "Current AST helper module origin differs",
    )
    require(
        isinstance(native_functions, FunctionType)
        and namespace.get("native_functions") is native_functions
        and native_functions.__module__ == expected_module
        and native_functions.__globals__ is namespace
        and native_functions.__code__.co_filename == str(path),
        "Actual native_functions function origin differs",
    )
    for alias, loaded in tuple(sys.modules.items()):
        if alias == expected_module:
            continue
        require(
            alias not in {Path(HELPER_IDENTITY).stem, module_name(HELPER_IDENTITY)},
            "Duplicate/obsolete AST helper namespace",
        )
        if not isinstance(loaded, ModuleType):
            continue
        loaded_file = vars(loaded).get("__file__")
        require(loaded is not module and loaded_file != str(path), "Duplicate AST helper module identity")
    layout.finish()
    context.finish()
    return dict(
        declared_ref=copy.deepcopy(declared_ref),
        current_ref=current_ref,
        module_name=expected_module,
        function_name="native_functions",
        function_origin=dict(module=native_functions.__module__, filename=native_functions.__code__.co_filename),
        historical_identity_scope="byte_identity_only",
        historical_source_executed_here=False,
        historical_source_execution_qualified=False,
        **proof,
    )


def bind_closed_native_owner(context, ref):
    """Retain the original owner guards while resolving source byte identities."""
    owner = context.read_historical_json(ref)
    require(owner["complete"] is owner["passed"] is True, "Prerequisite owner must have completed successfully")
    for item in [owner["registration_ref"], owner["result_ref"], *owner["owned_command"].values()]:
        context.remember_historical(item)
    launch = context.read_historical_json(owner["owned_command"]["launch"])
    exited = context.read_historical_json(owner["owned_command"]["exit"])
    release = context.read_historical_json(owner["owned_command"]["release"])
    require(
        launch["automatic_retry"] is exited["automatic_retry"] is False
        and launch["selected_environment"] == dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="")
        and exited["actual_wait"] is True
        and exited["returncode"] == 0
        and release["reaped"] is True
        and release["returncode"] == 0
        and release["signal_sent"] is False
        and release["original_error"] is None
        and release["pid"] == launch["child"]["pid"]
        and release["start_ticks"] == launch["child"]["start_ticks"],
        "Matching CPU launch and actual successful wait/reap without signals or retry",
    )
    result = context.read_historical_json(owner["result_ref"])
    require(result["complete"] is result["passed"] is True, "Complete prerequisite result")
    for path, digest in context.read_historical_json(owner["registration_ref"])["source_sha256"].items():
        context.remember_historical(dict(path=path, sha256=digest))
    context.finish()
    return owner, result
