"""Isolate the asset-path repair while preserving the failed bootstrap CLI."""

import argparse
import ast
import copy
import hashlib
from pathlib import Path

from prepare_native_runtime_path_snapshot import checked, read
from prepare_native_runtime_path_snapshot import reference as reference

from s2a.native_runtime_path_snapshot import REPO

BASELINE_REF = dict(
    path=str(REPO / "scripts/verify_native_runtime_bootstrap.py"),
    sha256="6f20f96e8232f6db0ea8b616c953565dab9faf5024cf92deeb0ec228c06d6eab",
)
PARENT_REFS = dict(
    original_config_ref=dict(
        path=str(REPO / "configs/native_runtime_bootstrap_20261029.json"),
        sha256="c0792de33a4a1fb14b1c1eb762f7091ebc58f260fc3efd10dae97286fb71cb99",
    ),
    qa_ref=dict(
        path=str(REPO / "outputs/native_runtime_bootstrap_qa_20261029_642s15ix/result.json"),
        sha256="e0630479475f117174ed3823d65017f6195e79806ec6079ad6ed1d45bc7b0f62",
    ),
    admission_ref=dict(
        path=str(REPO / "outputs/native_runtime_bootstrap_admission_20261029/result.json"),
        sha256="aea519230e66a2abf4fa4196885d1d81cb51e7de2c8a1608340e55b3db725fdc",
    ),
    result_ref=dict(
        path=str(REPO / "outputs/native_runtime_bootstrap_20261029/result.json"),
        sha256="c4e1742a0480a9a02dca19f6c65fab8f686c7c9c48e300fc634b288159da90b4",
    ),
    owner_ref=dict(
        path=str(REPO / "outputs/native_runtime_bootstrap_execution_20261029/result.json"),
        sha256="98e3a7d514d0abc76c79aa5dda50bf380831aa219657d72d1fc4798577a7892a",
    ),
)
NEW_SOURCES = (
    "src/s2a/native_runtime_bootstrap_v2.py",
    "scripts/verify_native_runtime_bootstrap_v2.py",
    "plans/v0.0.1/native_runtime_bootstrap_v2_20261029.md",
    "configs/native_runtime_bootstrap_v2_20261029.json",
    "tests/test_native_runtime_bootstrap_v2.py",
    "tests/test_native_runtime_bootstrap_v2_execution.py",
)
PATH_CHANGES = {
    "configs/native_runtime_bootstrap_20261029.json": "configs/native_runtime_bootstrap_v2_20261029.json",
    "plans/v0.0.1/native_runtime_bootstrap_20261029.md": "plans/v0.0.1/native_runtime_bootstrap_v2_20261029.md",
    "outputs/native_runtime_bootstrap_20261029/result.json": "outputs/native_runtime_bootstrap_v2_20261029/result.json",
}
CORE_MODULE = "s2a.native_runtime_bootstrap"
REPAIR_FIELDS = ("repair_parent", "cli_repair_proof", "core_repair_proof")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _dump(node):
    return ast.dump(node, include_attributes=False)


def _baseline_tree():
    source = Path(BASELINE_REF["path"]).read_bytes()
    _require(hashlib.sha256(source).hexdigest() == BASELINE_REF["sha256"], "Frozen original CLI bytes changed")
    return ast.parse(source, filename=BASELINE_REF["path"])


def _receipt_calls(tree):
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    found = {}
    for node in ast.walk(main):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("summary", "result"):
                _require(
                    name not in found
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "dict",
                    "Original result and dry summary constructors",
                )
                found[name] = node.value
    _require(set(found) == {"summary", "result"}, "Exactly two metadata receipts")
    return found.values()


def _path_literals(tree):
    """Locate the four entry bindings; never rewrite the frozen source roster."""
    nodes = []
    for name in ("CONFIG", "PLAN", "RESULT"):
        assignments = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ]
        _require(len(assignments) == 1 and isinstance(assignments[0].value, ast.BinOp), "Exact entry path assignment")
        nodes.append(assignments[0].value.right)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    assignments = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "config_ref"
    ]
    _require(len(assignments) == 1 and isinstance(assignments[0].value, ast.Call), "Exact admitted config reference")
    digests = [keyword.value for keyword in assignments[0].value.keywords if keyword.arg == "sha256"]
    _require(len(digests) == 1 and isinstance(digests[0], ast.Subscript), "Exact config QA digest lookup")
    nodes.append(digests[0].slice)
    _require(all(isinstance(node, ast.Constant) and isinstance(node.value, str) for node in nodes), "Path literals")
    return nodes


def repaired_cli_ast():
    """Only rewire paths/imports and add provenance metadata to two receipts."""
    tree = _baseline_tree()
    _require(isinstance(tree.body[-1], ast.If), "Original CLI entry guard")
    tree.body.pop()  # This module supplies its own command-line parser below.
    for node in _path_literals(tree):
        _require(node.value in PATH_CHANGES, "Original entry path")
        node.value = PATH_CHANGES[node.value]
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == CORE_MODULE:
            node.module = CORE_MODULE + "_v2"
    for call in _receipt_calls(tree):
        for field in REPAIR_FIELDS:
            call.keywords.append(ast.keyword(arg=field, value=ast.parse(f"inputs[{field!r}]", mode="eval").body))
    return ast.fix_missing_locations(tree)


def compile_cli_repair_proof(repaired_tree=None):
    """Inverse-check the inherited CLI only; new admission wrappers are separate."""
    original = _baseline_tree()
    restored = copy.deepcopy(repaired_cli_ast() if repaired_tree is None else repaired_tree)
    counts = dict.fromkeys(PATH_CHANGES, 0)
    inverse = {value: key for key, value in PATH_CHANGES.items()}
    imports = 0
    for node in _path_literals(restored):
        _require(node.value in inverse, "Repaired entry path")
        node.value = inverse[node.value]
        counts[node.value] += 1
    for node in ast.walk(restored):
        if isinstance(node, ast.ImportFrom) and node.module == CORE_MODULE + "_v2":
            node.module = CORE_MODULE
            imports += 1
    _require(list(counts.values()) == [2, 1, 1] and imports == 4, "Exactly four path literals and four core imports")
    for call in _receipt_calls(restored):
        for field in REPAIR_FIELDS:
            matches = [keyword for keyword in call.keywords if keyword.arg == field]
            expected = ast.parse(f"inputs[{field!r}]", mode="eval").body
            _require(len(matches) == 1 and _dump(matches[0].value) == _dump(expected), "Exact repair receipt metadata")
            call.keywords.remove(matches[0])
    restored.body.append(copy.deepcopy(original.body[-1]))
    _require(_dump(restored) == _dump(original), "All remaining inherited CLI AST must be identical")
    return dict(
        passed=True,
        frozen_cli_ref=dict(BASELINE_REF),
        inherited_cli_inverse_ast_equal=True,
        path_literal_changes=4,
        core_import_changes=4,
        metadata_fields=list(REPAIR_FIELDS),
        metadata_receipts=2,
        entry_parser_replaced=True,
        new_admission_wrappers_claimed_identical=False,
    )


# A separate module dictionary preserves the old module and its function globals.
# No simulator import occurs in this inherited module's top-level code.
_tree = repaired_cli_ast()
compile_cli_repair_proof(_tree)
exec(compile(_tree, str(Path(__file__).resolve()), "exec"), globals())
_inherited_verify_native_qa = globals()["verify_native_qa"]
_inherited_inspect_inputs = globals()["inspect_inputs"]
_inherited_main = globals()["main"]
require = globals()["require"]
collect_refs = globals()["collect_refs"]
add_ref = globals()["add_ref"]
CONFIG, PLAN, RESULT = (globals()[name] for name in ("CONFIG", "PLAN", "RESULT"))
NATIVE_PYTHON = globals()["NATIVE_PYTHON"]
CPU_ENVIRONMENT = globals()["CPU_ENVIRONMENT"]
PRIOR_SOURCE_PATHS = globals()["SOURCE_PATHS"]
PRIOR_QA_TESTS = globals()["QA_TESTS"]
SOURCE_PATHS = (*PRIOR_SOURCE_PATHS, *NEW_SOURCES)
QA_SOURCES = SOURCE_PATHS
QA_TESTS = (*PRIOR_QA_TESTS, *NEW_SOURCES[-2:])
PRIOR_QA_REF = PARENT_REFS["qa_ref"]


def _prior_qa():
    qa = read(PARENT_REFS["qa_ref"])
    require(
        qa["complete"]
        is qa["passed"]
        is qa["actual_wait"]
        is qa["reaped"]
        is qa["source_unchanged"]
        is qa["driver_unchanged"]
        is True
        and qa["returncode"] == 0
        and qa["signal_sent"] is qa["automatic_retry"] is False,
        "The failed attempt used successfully closed original QA",
    )
    require(
        qa["source_count"] == len(PRIOR_SOURCE_PATHS) == 17
        and qa["test_module_count"] == len(PRIOR_QA_TESTS) == 5
        and set(qa["source_sha256"]) == set(PRIOR_SOURCE_PATHS),
        "Exactly the original17-source/5-module QA",
    )
    for path, digest in qa["source_sha256"].items():
        checked(dict(path=str(REPO / path), sha256=digest))
    checked(qa["log_ref"])
    checked(qa["driver_ref"])
    return qa


def verify_native_qa(qa_ref):
    qa = _inherited_verify_native_qa(qa_ref)
    prior = _prior_qa()
    require(
        len(SOURCE_PATHS) == len(set(SOURCE_PATHS)) == 23
        and len(QA_TESTS) == len(set(QA_TESTS)) == 7
        and all(qa["source_sha256"][path] == digest for path, digest in prior["source_sha256"].items()),
        "New23-source/7-module QA preserves every original17-source byte",
    )
    return qa


def _repair_parent(config, refs):
    require(config.get("repair_parent") == PARENT_REFS, "Exact preserved failed-parent references")
    collect_refs(PARENT_REFS, refs)
    old_config = read(PARENT_REFS["original_config_ref"])
    restored = copy.deepcopy(config)
    restored.pop("repair_parent")
    restored["output_dir"] = old_config["output_dir"]
    require(
        restored == old_config and old_config["output_dir"] == str(REPO / "outputs/native_runtime_bootstrap_20261029"),
        "Only output directory and repair provenance differ from original configuration",
    )
    prior_qa = _prior_qa()
    failed, owner, admission = (read(PARENT_REFS[key]) for key in ("result_ref", "owner_ref", "admission_ref"))
    require(
        failed["complete"] is failed["passed"] is failed["bootstrap_verified"] is False
        and failed["runtime_loaded"] is failed["source_unchanged"] is True
        and failed["qa_ref"] == owner["qa_ref"] == admission["qa_ref"] == PARENT_REFS["qa_ref"]
        and failed["config_ref"] == PARENT_REFS["original_config_ref"]
        and failed["failure_stage"] == "constructor",
        "The preserved first attempt failed at constructor admission",
    )
    require(
        owner["complete"] is owner["passed"] is owner["automatic_retry"] is owner["control_qualified"] is False
        and owner["environment"] == CPU_ENVIRONMENT
        and owner["admission_ref"] == PARENT_REFS["admission_ref"]
        and len(owner["commands"]) == 1,
        "One preserved failed owner, no retry",
    )
    command = owner["commands"][0]
    require(
        command["actual_wait"] is command["reaped"] is True
        and command["returncode"] == 1
        and command["signal_sent"] is command["automatic_retry"] is False
        and command["timeout_seconds"] == 600
        and type(command["child_pid"]) is int
        and command["child_pid"] > 0
        and command["command"]
        == [
            str(NATIVE_PYTHON),
            "scripts/verify_native_runtime_bootstrap.py",
            "--qa-ref",
            PARENT_REFS["qa_ref"]["path"],
            "--qa-sha256",
            PARENT_REFS["qa_ref"]["sha256"],
        ],
        "Original child actually failed and was reaped without signals",
    )
    require(admission["complete"] is admission["passed"] is True, "Original CLI admission completed")
    core = failed["bootstrap"]
    require(
        core["counters"]["configure"] == dict(attempted=1, completed=1)
        and core["counters"]["model_construct"] == dict(attempted=1, completed=0)
        and all(
            core["counters"][name] == dict(attempted=0, completed=0)
            for name in core["counters"]
            if name not in ("configure", "model_construct")
        )
        and "assets/scenes/../textures/" in core["error"]["message"],
        "The preserved failure reached no compiled model, context, forward or integration",
    )
    collect_refs([failed, owner, admission, prior_qa], refs)
    return dict(
        passed=True,
        **PARENT_REFS,
        original_source_count=17,
        original_test_module_count=5,
        actual_wait=True,
        reaped=True,
        returncode=1,
        automatic_retry=False,
        preserved_counters=core["counters"],
    )


def inspect_inputs(config):
    from s2a.native_runtime_bootstrap_v2 import compile_repair_proof

    parent_refs = {}
    parent = _repair_parent(config, parent_refs)
    cli_proof, core_proof = compile_cli_repair_proof(), compile_repair_proof()
    require(core_proof["passed"] is True, "Finite core repair proof")
    inputs = _inherited_inspect_inputs(config)
    for ref in parent_refs.values():
        add_ref(ref, inputs["refs"])
    inputs.update(repair_parent=parent, cli_repair_proof=cli_proof, core_repair_proof=core_proof)
    return inputs


def main(qa_ref, dry=False):
    return _inherited_main(qa_ref, dry=dry)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Isolated native runtime bootstrap v2")
    parser.add_argument("--qa-ref", required=True)
    parser.add_argument("--qa-sha256", required=True)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    main(dict(path=args.qa_ref, sha256=args.qa_sha256), dry=args.dry)
