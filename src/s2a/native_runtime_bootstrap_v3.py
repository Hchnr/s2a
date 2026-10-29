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

from s2a import native_runtime_bootstrap_v2 as prior

LIMITS = dict(prior.LIMITS)
BootstrapFailure = prior.BootstrapFailure
BootstrapRuntime = prior.BootstrapRuntime
PRIOR_CORE_PATH = Path(prior.__file__).resolve()
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


def _prior_function():
    prior.frozen._require(
        hashlib.sha256(PRIOR_CORE_PATH.read_bytes()).hexdigest() == PRIOR_CORE_SHA256,
        "frozen v2 bootstrap source changed",
    )
    return prior._repaired_function()


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
    return dict(
        passed=True,
        prior_core_ref=dict(path=str(PRIOR_CORE_PATH), sha256=PRIOR_CORE_SHA256),
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
