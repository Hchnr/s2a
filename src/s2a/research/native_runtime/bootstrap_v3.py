"""Defer historical record parent-component checks until asset remapping.

The v2 constructor asset resolver already permits only existing, symlink-free
paths that remain within an approved root. This repair removes the redundant
ban on ``..`` in the historical *declared* record path. Absolute declarations,
existing string-only /assets/ relocation, complete non-file XML identity,
compile-time asset validation, counters and all other guards are unchanged.
"""

import ast
import hashlib
from copy import deepcopy
from pathlib import Path

from s2a.infrastructure.current_layout import CurrentLayout
from s2a.research.native_runtime import bootstrap_v2 as prior

LIMITS = dict(prior.LIMITS)
BootstrapFailure = prior.BootstrapFailure
BootstrapRuntime = prior.BootstrapRuntime
PRIOR_CORE_PATH = Path(prior.__file__).resolve()
PRIOR_CORE_DECLARED_PATH = "src/s2a/native_runtime_bootstrap_v2.py"
PRIOR_CORE_SHA256 = "5f1d7da14c592383eac171dd3079dabacef777b5f3cef42c36336636b7dfa125"
OLD_PREDICATE = 'Path(file).is_absolute() and ".." not in Path(file).parts'
NEW_PREDICATE = "Path(file).is_absolute()"
GUARD_MESSAGE = "invalid declared XML path"


def _record_guards(function):
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_require"
        and len(node.args) == 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == GUARD_MESSAGE
    ]


def _approved_prior_source():
    layout = CurrentLayout(root=prior.repository_root(anchor=__file__))
    path = layout.source_path(PRIOR_CORE_DECLARED_PATH)
    prior.frozen._require(path == PRIOR_CORE_PATH, "registered current v2 source origin differs")
    data = path.read_bytes()
    approved = layout.entry(PRIOR_CORE_DECLARED_PATH)["current_ref"]
    prior.frozen._require(hashlib.sha256(data).hexdigest() == approved["sha256"], "approved current v2 bytes changed")
    layout.finish()
    return data, dict(path=str(path), sha256=approved["sha256"])


def _verify_prior_runtime_functions(historical, current):
    """Preserve every original v2 AST node except explicit identity plumbing."""
    expected = deepcopy(historical)
    for node in expected.body:
        if isinstance(node, ast.ImportFrom) and node.module == "s2a":
            prior.frozen._require(
                [(item.name, item.asname) for item in node.names] == [("native_runtime_bootstrap", "frozen")],
                "exact historical bootstrap import required",
            )
            node.module = "s2a.research.native_runtime"
            node.names[0].name = "bootstrap"
        elif isinstance(node, ast.FunctionDef) and node.name == "_repaired_function":
            node.body[:3] = ast.parse(
                "data, _ = _frozen_core_source()\ntree = ast.parse(data, filename=FROZEN_CORE_DECLARED_PATH)"
            ).body
        elif isinstance(node, ast.FunctionDef) and node.name == "compile_repair_proof":
            node.body.insert(-1, ast.parse("_, source = _frozen_core_source()").body[0])
            result = node.body[-1].value
            index = next(i for i, item in enumerate(result.keywords) if item.arg == "frozen_core_ref")
            result.keywords[index].value = ast.parse(
                "dict(path=FROZEN_CORE_DECLARED_PATH, sha256=FROZEN_CORE_SHA256)", mode="eval"
            ).body
            result.keywords[index + 1 : index + 1] = ast.parse(
                "dict(frozen_core_identity=source['historical_identity'], current_core_ref=source['current_ref'], "
                "current_namespace_ast_equal=source['current_namespace_ast_equal'], "
                "historical_source_execution_qualified=False)",
                mode="eval",
            ).body.keywords
    # New identity helper bodies come from the independently pinned current
    # registration, rather than from the AST offered by the caller as proof.
    approved, _ = _approved_prior_source()
    helpers = {}
    for name in ("_historical_core_source", "_verify_frozen_namespace", "_frozen_core_source"):
        matches = [node for node in ast.parse(approved).body if isinstance(node, ast.FunctionDef) and node.name == name]
        prior.frozen._require(len(matches) == 1, "one approved v2 identity helper definition required")
        helpers[name] = deepcopy(matches[0])
    body = []
    for node in expected.body:
        if isinstance(node, ast.ImportFrom) and node.module == "pathlib":
            body.extend(ast.parse("from dataclasses import asdict").body)
        if isinstance(node, ast.ImportFrom) and node.module == "s2a.research.native_runtime":
            body.extend(
                ast.parse(
                    "from s2a.infrastructure.current_layout import BASELINE_COMMIT, CATALOG_REF\n"
                    "from s2a.infrastructure.repository_paths import repository_root\n"
                    "from s2a.infrastructure.source_identity import SourceIdentityLocator"
                ).body
            )
        if isinstance(node, ast.FunctionDef) and node.name == "_repaired_function":
            body.extend(helpers.values())
        body.append(node)
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "FROZEN_CORE_PATH"
        ):
            body.extend(ast.parse('FROZEN_CORE_DECLARED_PATH = "src/s2a/native_runtime_bootstrap.py"').body)
    expected.body = body
    prior.frozen._require(
        prior._dump(expected) == prior._dump(current), "v2 AST changed beyond namespace/identity plumbing"
    )


def _prior_source():
    data, identity = prior._historical_core_source(PRIOR_CORE_DECLARED_PATH, PRIOR_CORE_SHA256)
    current, current_ref = _approved_prior_source()
    _verify_prior_runtime_functions(ast.parse(data), ast.parse(current))
    return data, dict(
        historical_identity=identity,
        current_ref=current_ref,
    )


def _prior_function():
    data, _ = _prior_source()
    historical = ast.parse(data)
    _, core = prior._frozen_core_source()
    # Execute only the historical pure AST transformer, never its module or
    # runtime. Its file input is the explicitly verified original v1 Git blob.
    definitions = [
        deepcopy(node)
        for node in historical.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_dump", "_repaired_function"}
    ]
    prior.frozen._require(len(definitions) == 2, "complete historical v2 AST transformer required")
    namespace = dict(
        ast=ast,
        hashlib=hashlib,
        deepcopy=deepcopy,
        frozen=prior.frozen,
        FROZEN_CORE_PATH=Path(core["historical_identity"]["storage_ref"]["path"]),
        FROZEN_CORE_SHA256=prior.FROZEN_CORE_SHA256,
    )
    code = compile(
        ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[])), PRIOR_CORE_DECLARED_PATH, "exec"
    )
    exec(code, namespace)
    expected = namespace["_repaired_function"]()
    actual = prior._repaired_function()
    prior.frozen._require(prior._dump(expected) == prior._dump(actual), "complete historical v2 repair AST differs")
    return actual


def _repaired_function():
    function = _prior_function()
    guards = _record_guards(function)
    old = ast.parse(OLD_PREDICATE, mode="eval").body
    prior.frozen._require(
        len(guards) == 1 and prior._dump(guards[0].args[0]) == prior._dump(old),
        "exactly one original absolute-and-parent record guard required",
    )
    # Keep the original absolute-path node and its source location.
    guards[0].args[0] = guards[0].args[0].values[0]
    return function


def compile_repair_proof(repaired_function=None):
    """Inverse-check the single changed predicate, with v2's proof nested."""
    original = _prior_function()
    modified = _repaired_function() if repaired_function is None else deepcopy(repaired_function)
    guards = _record_guards(modified)
    expected = ast.parse(NEW_PREDICATE, mode="eval").body
    prior.frozen._require(
        len(guards) == 1 and prior._dump(guards[0].args[0]) == prior._dump(expected),
        "exactly one repaired absolute-only record guard required",
    )
    guards[0].args[0] = ast.parse(OLD_PREDICATE, mode="eval").body
    prior.frozen._require(prior._dump(modified) == prior._dump(original), "non-record-guard bootstrap AST changed")
    _, source = _prior_source()
    return dict(
        passed=True,
        prior_core_ref=dict(path=PRIOR_CORE_DECLARED_PATH, sha256=PRIOR_CORE_SHA256),
        prior_core_identity=source["historical_identity"],
        current_prior_core_ref=source["current_ref"],
        current_prior_runtime_ast_equal=True,
        historical_source_execution_qualified=False,
        prior_repair_proof=prior.compile_repair_proof(),
        run_inverse_to_v2_ast_equal=True,
        record_guard_replacement_count=1,
        record_xml_precondition_unchanged=False,
        record_xml_absolute_check_preserved=True,
        asset_resolution_deferred=True,
        asset_validator_and_cleanup_unchanged=True,
        output_and_source_path_guards_unchanged=True,
        non_file_xml_content_unchanged=True,
        additional_model_xml_rewrites=False,
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
    """Run the isolated v2 path with one delayed record-path validation."""
    function = _repaired_function()
    compile_repair_proof(function)
    namespace = {
        **vars(prior.frozen),
        "_asset_records": prior._private_asset_records,
        "_close_runtime_target": prior._close_runtime_target,
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


run_native_bootstrap_v3 = run_native_bootstrap
