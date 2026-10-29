"""Bind the record-XML repair to both preserved failed bootstrap attempts."""

import argparse
import ast
import copy
import math
from pathlib import Path
from types import FunctionType

import verify_native_runtime_bootstrap_v2 as frozen_cli
from prepare_native_runtime_path_snapshot import checked, read

from s2a.native_runtime_path_snapshot import REPO

FROZEN_CLI_REF = dict(
    path=str(REPO / "scripts/verify_native_runtime_bootstrap_v2.py"),
    sha256="c1e834876879a96442fd45de89948c9cfba0917ce12e27a01502ababc369c956",
)
PREDECESSOR_REFS = dict(
    original_config_ref=dict(
        path=str(REPO / "configs/native_runtime_bootstrap_v2_20261029.json"),
        sha256="b37fda536c0739aaa659a3392a3260c2d1a4949e62a6d7dc332d789c9b7b15ef",
    ),
    qa_ref=dict(
        path=str(REPO / "outputs/native_runtime_bootstrap_v2_qa_20261029_jbrxlsyy/result.json"),
        sha256="49808224572128be90e07317cba22620574b10efb1552a093ab0c2de55552916",
    ),
    admission_ref=dict(
        path=str(REPO / "outputs/native_runtime_bootstrap_v2_admission_20261029/result.json"),
        sha256="7b044b5ba9f0b155d7b81406745eee1c8240919fb21df674e19d7122cc67faf4",
    ),
    result_ref=dict(
        path=str(REPO / "outputs/native_runtime_bootstrap_v2_20261029/result.json"),
        sha256="91265cadcbe8c1e9f9a1b4a44a16ace135ba91c938e36f8998e8b4e2d2211873",
    ),
    owner_ref=dict(
        path=str(REPO / "outputs/native_runtime_bootstrap_v2_execution_20261029/result.json"),
        sha256="8e1e94c83b4427bf1feefbf9e5ccef1c93dc3ae89de68c380870df9f3d66207b",
    ),
)
NEW_SOURCES = (
    "src/s2a/native_runtime_bootstrap_v3.py",
    "scripts/verify_native_runtime_bootstrap_v3.py",
    "plans/v0.0.1/native_runtime_bootstrap_v3_20261029.md",
    "configs/native_runtime_bootstrap_v3_20261029.json",
    "tests/test_native_runtime_bootstrap_v3.py",
    "tests/test_native_runtime_bootstrap_v3_execution.py",
)
PATH_CHANGES = {value: value.replace("bootstrap_v2", "bootstrap_v3") for value in frozen_cli.PATH_CHANGES.values()}
CORE_MODULE = "s2a.native_runtime_bootstrap_v2"


def repaired_cli_ast():
    checked(FROZEN_CLI_REF)
    tree = frozen_cli.repaired_cli_ast()
    for node in frozen_cli._path_literals(tree):
        frozen_cli.require(node.value in PATH_CHANGES, "Frozen v2 entry path")
        node.value = PATH_CHANGES[node.value]
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == CORE_MODULE:
            node.module = "s2a.native_runtime_bootstrap_v3"
    for call in frozen_cli._receipt_calls(tree):
        call.keywords.append(
            ast.keyword(arg="repair_predecessor", value=ast.parse("inputs['repair_predecessor']", mode="eval").body)
        )
    return ast.fix_missing_locations(tree)


def compile_cli_repair_proof(repaired_tree=None):
    checked(FROZEN_CLI_REF)
    original = frozen_cli.repaired_cli_ast()
    restored = copy.deepcopy(repaired_cli_ast() if repaired_tree is None else repaired_tree)
    inverse = {value: key for key, value in PATH_CHANGES.items()}
    counts = dict.fromkeys(PATH_CHANGES, 0)
    for node in frozen_cli._path_literals(restored):
        frozen_cli.require(node.value in inverse, "Repaired v3 entry path")
        node.value = inverse[node.value]
        counts[node.value] += 1
    imports = 0
    for node in ast.walk(restored):
        if isinstance(node, ast.ImportFrom) and node.module == "s2a.native_runtime_bootstrap_v3":
            node.module = CORE_MODULE
            imports += 1
    frozen_cli.require(list(counts.values()) == [2, 1, 1] and imports == 4, "Four path literals and four core imports")
    for call in frozen_cli._receipt_calls(restored):
        matches = [keyword for keyword in call.keywords if keyword.arg == "repair_predecessor"]
        expected = ast.parse("inputs['repair_predecessor']", mode="eval").body
        frozen_cli.require(
            len(matches) == 1 and frozen_cli._dump(matches[0].value) == frozen_cli._dump(expected),
            "Exact predecessor receipt metadata",
        )
        call.keywords.remove(matches[0])
    frozen_cli.require(
        frozen_cli._dump(restored) == frozen_cli._dump(original), "Remaining inherited CLI AST identical"
    )
    return dict(
        passed=True,
        frozen_cli_ref=dict(FROZEN_CLI_REF),
        inherited_cli_inverse_ast_equal=True,
        predecessor_cli_proof=frozen_cli.compile_cli_repair_proof(restored),
        path_literal_changes=4,
        core_import_changes=4,
        added_metadata_fields=["repair_predecessor"],
        metadata_receipts=2,
        new_admission_wrappers_claimed_identical=False,
    )


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
LIMIT_NAMES = globals()["LIMIT_NAMES"]
PRIOR_SOURCE_PATHS = (*globals()["SOURCE_PATHS"], *frozen_cli.NEW_SOURCES)
PRIOR_QA_TESTS = (*globals()["QA_TESTS"], *frozen_cli.NEW_SOURCES[-2:])
SOURCE_PATHS = (*PRIOR_SOURCE_PATHS, *NEW_SOURCES)
QA_SOURCES = SOURCE_PATHS
QA_TESTS = (*PRIOR_QA_TESTS, *NEW_SOURCES[-2:])
PARENT_REFS = copy.deepcopy(frozen_cli.PARENT_REFS)
PRIOR_QA_REF = PREDECESSOR_REFS["qa_ref"]


def _repair_parent(config, refs):
    """Run the exact original v1 failure admission with isolated bindings."""
    stripped = copy.deepcopy(config)
    stripped.pop("repair_predecessor")
    namespace = {**vars(frozen_cli), "REPO": REPO, "NATIVE_PYTHON": NATIVE_PYTHON, "PARENT_REFS": PARENT_REFS}
    namespace["_prior_qa"] = FunctionType(frozen_cli._prior_qa.__code__, namespace)
    return FunctionType(frozen_cli._repair_parent.__code__, namespace)(stripped, refs)


def _predecessor_qa():
    namespace = {**globals(), "SOURCE_PATHS": PRIOR_SOURCE_PATHS, "QA_TESTS": PRIOR_QA_TESTS}
    return FunctionType(_inherited_verify_native_qa.__code__, namespace)(PREDECESSOR_REFS["qa_ref"])


def verify_native_qa(qa_ref):
    qa = _inherited_verify_native_qa(qa_ref)
    prior = _predecessor_qa()
    require(
        len(SOURCE_PATHS) == len(set(SOURCE_PATHS)) == 29
        and len(QA_TESTS) == len(set(QA_TESTS)) == 9
        and len(PRIOR_SOURCE_PATHS) == 23
        and len(PRIOR_QA_TESTS) == 7
        and all(qa["source_sha256"][path] == digest for path, digest in prior["source_sha256"].items()),
        "New29-source/9-module QA preserves every previous23-source byte",
    )
    return qa


def _repair_predecessor(config, refs):
    require(config.get("repair_predecessor") == PREDECESSOR_REFS, "Exact preserved v2 failure references")
    collect_refs(PREDECESSOR_REFS, refs)
    old_config = read(PREDECESSOR_REFS["original_config_ref"])
    restored = copy.deepcopy(config)
    restored.pop("repair_predecessor")
    restored["output_dir"] = old_config["output_dir"]
    require(
        restored == old_config
        and old_config["output_dir"] == str(REPO / "outputs/native_runtime_bootstrap_v2_20261029")
        and old_config["repair_parent"] == PARENT_REFS,
        "Only output directory and predecessor provenance differ from v2 configuration",
    )
    prior_qa = _predecessor_qa()
    failed, owner, admission = (read(PREDECESSOR_REFS[key]) for key in ("result_ref", "owner_ref", "admission_ref"))
    require(
        failed["complete"] is failed["passed"] is failed["bootstrap_verified"] is False
        and failed["runtime_loaded"] is failed["source_unchanged"] is True
        and failed["failure_stage"] == "record_xml"
        and failed["config_ref"] == PREDECESSOR_REFS["original_config_ref"]
        and failed["qa_ref"] == owner["qa_ref"] == admission["qa_ref"] == PREDECESSOR_REFS["qa_ref"],
        "Preserved v2 failure is the closed record XML admission failure",
    )
    require(
        owner["complete"] is owner["passed"] is owner["automatic_retry"] is owner["control_qualified"] is False
        and owner["environment"] == CPU_ENVIRONMENT
        and owner["admission_ref"] == PREDECESSOR_REFS["admission_ref"]
        and len(owner["commands"]) == 1
        and admission["complete"] is admission["passed"] is True,
        "One v2 failed owner following completed admission",
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
            "scripts/verify_native_runtime_bootstrap_v2.py",
            "--qa-ref",
            PREDECESSOR_REFS["qa_ref"]["path"],
            "--qa-sha256",
            PREDECESSOR_REFS["qa_ref"]["sha256"],
        ],
        "V2 child actually failed and was reaped without signals or retry",
    )
    counts = dict.fromkeys(("configure", *LIMIT_NAMES.values()), 0)
    counts.update(configure=1, model_construct=2, context=2, forward=9, step1=2)
    counters = {name: dict(attempted=count, completed=count) for name, count in counts.items()}
    core = failed["bootstrap"]
    require(
        core["counters"] == counters
        and core["failure_stage"] == "record_xml"
        and core["error"]["message"] == "invalid declared XML path"
        and core["cleanup_complete"] is core["cleanup_close_called"] is True
        and core["cleanup_outcome"] == "closed"
        and core["cleanup_target"] == "outer",
        "Exact v2 record admission counters and completed cleanup",
    )
    time_checks = core["time_checks"]
    require(
        len(time_checks) == 11
        and sum(item["operation"] == "forward" for item in time_checks) == 9
        and sum(item["operation"] == "step1" for item in time_checks) == 2
        and all(
            type(item["before"]) in (int, float)
            and type(item["after"]) in (int, float)
            and math.isfinite(item["before"])
            and item["before"] == item["after"]
            for item in time_checks
        ),
        "All eleven preserved nonintegrating calls left simulation time unchanged",
    )
    collect_refs([failed, owner, admission, prior_qa], refs)
    return dict(
        passed=True,
        **PREDECESSOR_REFS,
        original_source_count=23,
        original_test_module_count=7,
        actual_wait=True,
        reaped=True,
        returncode=1,
        automatic_retry=False,
        preserved_counters=core["counters"],
        cleanup_complete=True,
        failure_stage="record_xml",
        time_checks_unchanged=True,
        time_checks=time_checks,
    )


def inspect_inputs(config):
    from s2a.native_runtime_bootstrap_v3 import compile_repair_proof

    prior_refs = {}
    parent = _repair_parent(config, prior_refs)
    predecessor = _repair_predecessor(config, prior_refs)
    cli_proof, core_proof = compile_cli_repair_proof(), compile_repair_proof()
    require(core_proof["passed"] is True, "Finite v3 core repair proof")
    inputs = _inherited_inspect_inputs(config)
    for ref in prior_refs.values():
        add_ref(ref, inputs["refs"])
    inputs.update(
        repair_parent=parent,
        repair_predecessor=predecessor,
        cli_repair_proof=cli_proof,
        core_repair_proof=core_proof,
    )
    return inputs


def main(qa_ref, dry=False):
    return _inherited_main(qa_ref, dry=dry)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Isolated native runtime bootstrap v3")
    parser.add_argument("--qa-ref", required=True)
    parser.add_argument("--qa-sha256", required=True)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    main(dict(path=args.qa_ref, sha256=args.qa_sha256), dry=args.dry)
