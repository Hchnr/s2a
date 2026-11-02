"""Isolated asset-path and partial-cleanup repair for the frozen bootstrap.

Model XML bytes, runtime calls, counters, output/source path guards, initial
state, controller and rendering qualification remain the frozen v1 code.
Only its asset validator receives a private canonical-path resolver; only the
nested cleanup call is replaced in an independently compiled namespace.
No old module globals or executed sources are modified.
"""

import ast
import hashlib
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import FunctionType

from s2a.infrastructure.current_layout import BASELINE_COMMIT, CATALOG_REF
from s2a.infrastructure.repository_paths import repository_root
from s2a.infrastructure.source_identity import SourceIdentityLocator
from s2a.research.native_runtime import bootstrap as frozen

LIMITS = dict(frozen.LIMITS)
BootstrapFailure = frozen.BootstrapFailure
BootstrapRuntime = frozen.BootstrapRuntime
FROZEN_CORE_PATH = Path(frozen.__file__).resolve()
FROZEN_CORE_DECLARED_PATH = "src/s2a/native_runtime_bootstrap.py"
FROZEN_CORE_SHA256 = "f9eefa9b43dc9ef841809fe9744974a7e8336e2ba67ecfd62d12e0ff46b198f0"


def normalize_xml_asset_path(declared, roots, *, exists=True):
    """Resolve only absolute asset paths that never leave one approved root.

    Check each component before consuming a following ``..``. This rejects a
    symlink or nonexistent/non-directory component even if lexical normpath
    would discard it. Leaving a root and subsequently returning is forbidden.
    The returned canonical filename is for identity checks only: the original
    XML file attribute passed to MuJoCo is not rewritten.
    """
    frozen._require(exists is True, "asset resolution requires existing paths")
    path = Path(declared)
    frozen._require(path.is_absolute(), "asset path must be absolute")
    frozen._require(not str(path).startswith("/share/project/"), "retired asset path forbidden")
    approved = [Path(root) for root in roots if path.is_relative_to(root)]
    frozen._require(bool(approved), f"asset does not begin inside an approved root: {path}")
    root = max(approved, key=lambda item: len(item.parts))
    frozen._safe_path(root, (frozen.REPO,))
    frozen._require(root.is_dir(), "asset root must be a directory")
    current = root
    components = path.parts[len(root.parts) :]
    for index, component in enumerate(components):
        if component == "..":
            frozen._require(current != root, f"asset traversal leaves approved root: {path}")
            current = current.parent
        else:
            current /= component
            frozen._require(not current.is_symlink(), f"symlink asset component forbidden: {current}")
            frozen._require(current.exists(), f"asset component missing: {current}")
        if index < len(components) - 1:
            frozen._require(current.is_dir(), f"asset intermediate component is not a directory: {current}")
    return frozen._safe_path(current, (root,))


def _private_asset_records(xml, roots, expected):
    # This code object is exactly frozen v1. Its resolver override is local to
    # this one function, never the bootstrap/output/source namespace.
    namespace = {**vars(frozen), "_safe_path": normalize_xml_asset_path}
    validate = FunctionType(frozen._asset_records.__code__, namespace, "_asset_records")
    records = validate(xml, roots, expected)
    for record in records:
        record["canonical_path"] = record["reference"]["path"]
        record["normalization_changed"] = record["declared_path"] != record["canonical_path"]
    return records


def _close_runtime_target(target, receipt):
    """Do not call MujocoEnv.close before its sim/viewer exist."""
    state = vars(target)
    if receipt["cleanup_target"] == "inner_partial" and state.get("sim") is None and state.get("viewer") is None:
        receipt.update(cleanup_outcome="no_resources_created", cleanup_close_called=False)
        # Explicitly not a successful close. The triggering failure is retained.
        return False
    receipt.update(cleanup_outcome="close_attempted", cleanup_close_called=True)
    target.close()
    receipt["cleanup_outcome"] = "closed"
    return True


def _dump(node):
    return ast.dump(node, include_attributes=False)


def _historical_core_source(declared_path, sha256):
    """Read catalogued Git bytes for AST proof, without executing that module."""
    root = repository_root(anchor=__file__)
    locator = SourceIdentityLocator(
        root=root,
        baseline_commit=BASELINE_COMMIT,
        catalog_ref=CATALOG_REF,
        cache_directory=root / "outputs/native_runtime_bootstrap_historical_source_cache_20261029",
    )
    resolved = locator.resolve(dict(path=declared_path, sha256=sha256))
    data = Path(resolved.storage_ref["path"]).read_bytes()
    frozen._require(hashlib.sha256(data).hexdigest() == sha256, "frozen bootstrap source changed")
    locator.finish()
    return data, asdict(resolved)


def _verify_frozen_namespace(historical, current):
    """Allow exactly the repository locator and one simulator import relocation."""
    expected = deepcopy(historical)
    assignments = [
        node
        for node in expected.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "REPO" for target in node.targets)
    ]
    frozen._require(
        len(assignments) == 1
        and _dump(assignments[0].value) == _dump(ast.parse("Path(__file__).resolve().parents[2]", mode="eval").body),
        "exactly one historical repository locator required",
    )
    assignment = assignments[0]
    assignment.value = ast.parse("repository_root(anchor=__file__)", mode="eval").body
    expected.body.insert(
        expected.body.index(assignment),
        ast.parse("from s2a.infrastructure.repository_paths import repository_root").body[0],
    )
    imports = [
        node
        for node in ast.walk(expected)
        if isinstance(node, ast.ImportFrom) and node.module == "s2a.libero_simulation"
    ]
    frozen._require(len(imports) == 1, "exactly one historical simulator import required")
    imports[0].module = "s2a.backends.libero.simulation"
    frozen._require(_dump(expected) == _dump(current), "bootstrap changed beyond explicit namespace relocation")


def _frozen_core_source():
    data, identity = _historical_core_source(FROZEN_CORE_DECLARED_PATH, FROZEN_CORE_SHA256)
    current = FROZEN_CORE_PATH.read_bytes()
    _verify_frozen_namespace(ast.parse(data), ast.parse(current))
    return data, dict(
        historical_identity=identity,
        current_ref=dict(path=str(FROZEN_CORE_PATH), sha256=hashlib.sha256(current).hexdigest()),
        current_namespace_ast_equal=True,
    )


def _repaired_function():
    data, _ = _frozen_core_source()
    tree = ast.parse(data, filename=FROZEN_CORE_DECLARED_PATH)
    original = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_native_bootstrap"
    )
    modified = deepcopy(original)
    cleanup = next(node for node in ast.walk(modified) if isinstance(node, ast.FunctionDef) and node.name == "cleanup")
    target = ast.parse("target.close()\ncleanup_complete = True").body
    replacement = ast.parse("cleanup_complete = _close_runtime_target(target, receipt)").body[0]
    matches = []
    for node in ast.walk(cleanup):
        if isinstance(node, ast.Try):
            for index in range(len(node.body) - 1):
                if [_dump(part) for part in node.body[index : index + 2]] == [_dump(part) for part in target]:
                    matches.append((node, index))
    frozen._require(len(matches) == 1, "exactly one frozen cleanup replacement required")
    container, index = matches[0]
    container.body[index : index + 2] = [replacement]
    restored = deepcopy(modified)
    reverse = []
    for node in ast.walk(restored):
        if isinstance(node, ast.Try):
            for position, child in enumerate(node.body):
                if _dump(child) == _dump(replacement):
                    reverse.append((node, position))
    frozen._require(len(reverse) == 1, "exactly one inverse cleanup replacement required")
    reverse[0][0].body[reverse[0][1] : reverse[0][1] + 1] = deepcopy(target)
    frozen._require(_dump(restored) == _dump(original), "non-cleanup bootstrap AST changed")
    return modified


def compile_repair_proof():
    """Check the limited source transformation without importing a runtime."""
    _repaired_function()
    _, source = _frozen_core_source()
    return dict(
        passed=True,
        frozen_core_ref=dict(path=FROZEN_CORE_DECLARED_PATH, sha256=FROZEN_CORE_SHA256),
        frozen_core_identity=source["historical_identity"],
        current_core_ref=source["current_ref"],
        current_namespace_ast_equal=source["current_namespace_ast_equal"],
        historical_source_execution_qualified=False,
        run_inverse_ast_equal=True,
        cleanup_replacement_count=1,
        private_asset_code_equal=True,
        global_path_guards_unchanged=True,
        record_xml_precondition_unchanged=True,
        model_xml_rewritten=False,
        new_runtime_loads=0,
        new_model_constructs=0,
    )


def run_native_bootstrap(
    runtime_config,
    record,
    output,
    *,
    allowed_asset_roots,
    asset_refs=None,
    runtime_factory=None,
    progress=None,
):
    """Invoke v1 in a private namespace with only the two documented repairs."""
    function = _repaired_function()
    namespace = {
        **vars(frozen),
        "_asset_records": _private_asset_records,
        "_close_runtime_target": _close_runtime_target,
    }
    code = compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(__file__), "exec")
    exec(code, namespace)
    return namespace["run_native_bootstrap"](
        runtime_config,
        record,
        output,
        allowed_asset_roots=allowed_asset_roots,
        asset_refs=asset_refs,
        runtime_factory=runtime_factory,
        progress=progress,
    )


run_native_bootstrap_v2 = run_native_bootstrap
