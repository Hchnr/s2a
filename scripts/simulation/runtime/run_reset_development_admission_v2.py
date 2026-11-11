"""Recheck development resets with an exact, scoped canvas thread repair."""

import argparse
import os
import tempfile
from pathlib import Path
from types import FunctionType

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.publication import publish_result as publish
from scripts.simulation.runtime import prepare_reset_development as preparation
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_development_admission as original

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/joint_control/reset_native_admission_v2_20261111.json"
PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/reset_canvas_thread_repair_20261111.md"
OUTPUT = ROOT / "outputs/reset_native_admission_v2_20261111"
DRY = ROOT / "outputs/reset_native_admission_v2_dry_20261111/result.json"
QA = ROOT / "outputs/qa/reset_native_admission_v2_20261111/result.json"
MODULE = "scripts.simulation.runtime.run_reset_development_admission_v2"
AUDIT = {
    "path": str(ROOT / "outputs/reset_canvas_threads_audit_20261111/result.json"),
    "sha256": "e433f00f1b22970f6720116fdf736e691de98073efb69907c0e0c72759581268",
}
FAILURE_SHA = "96e3732026a52470348b51a4aeabe001e01adc9261a8ab728f18564c945310a0"


def validate_config(config):
    changed = dict(config)
    if (
        changed.pop("schema", None) != "s2a.reset-native-admission.v2"
        or changed.pop("canvas_torch_threads", None) != 2
        or changed.pop("canvas_thread_audit_sha256", None) != AUDIT["sha256"]
        or changed.pop("original_failure_sha256", None) != FAILURE_SHA
    ):
        raise ValueError("The exact separately registered canvas repair is required")
    changed["schema"] = "s2a.reset-native-admission.v1"
    original.validate_config(changed)


def validate_audit(audit):
    if not (
        audit["complete"] is audit["passed"] is audit["sources_unchanged"] is True
        and audit["control_qualified"] is False
        and audit["original_failure_not_reclassified"] is True
        and audit["failure_ref"]["sha256"] == FAILURE_SHA
        and len(audit["cases"]) == 12
        and audit["native_models"] == audit["forward_queries"] == audit["physics_steps"] == 0
        and audit["world_model_calls"] == audit["fresh_cases"] == 0
        and audit["torch_threads_restored"] == 1
        and all(
            row["case"]["case_index"] == i and row["one_matches_saved"] is row["two_matches_recorded"] is True
            for i, row in enumerate(audit["cases"])
        )
    ):
        raise ValueError("The full exact-pixel audit and original failure are required")


def load_config():
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    validate_config(config)
    audit = prefix.read(AUDIT)
    validate_audit(audit)
    for source in audit["source_refs"]:
        prefix.checked(source)
    return ref, config


def source_refs():
    return original.source_refs() + [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__).resolve(),
            ROOT / "src/s2a/research/native_runtime/reset_canvas_threads.py",
            ROOT / "tests/research/native_runtime/test_reset_canvas_threads.py",
            ROOT / "tests/research/native_runtime/test_reset_development_admission_v2.py",
        )
    ]


def worker(index):
    directory = OUTPUT / f"case_{index:02d}"
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        case_index=index,
        automatic_retry=False,
        world_model_calls=0,
        fresh_cases=0,
    )
    try:
        temp, cache = directory / "tmp", directory / "cache"
        temp.mkdir()
        cache.mkdir()
        os.environ.update(TMPDIR=str(temp), XDG_CACHE_HOME=str(cache), NUMBA_CACHE_DIR=str(cache / "numba"))
        tempfile.tempdir = None
        prefix.environment()
        original.require_no_simulator_import()
        registration = prefix.read(prefix.reference(directory / "registration.json"))
        inputs = preparation.load_inputs(index, verify_assets=False)
        prefix.require(
            registration["inputs"] == original.case_receipt(inputs, index)
            and registration["source_refs"] == source_refs(),
            "Registered inputs and sources required",
        )
        original.require_no_simulator_import()
        cv2 = inputs["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        original.require_no_simulator_import()
        from s2a.research.native_runtime.development_prefix_runtime import PrefixReplayFailure
        from s2a.research.native_runtime.reset_canvas_threads import TwoThreadCanvasResetRuntime

        case = inputs["cases"][0]
        adapter = TwoThreadCanvasResetRuntime(
            expected=case["expected"],
            expected_receipt=case["expected_receipt"],
            chunk=case["chunk"],
            controller=case["controller"],
            tool=case["tool"],
        )
        try:
            runtime = adapter.run(
                case["config"],
                case["record"],
                case["record_xml"],
                directory / "runtime",
                inputs["asset_refs"],
                registered_seed=case["case"]["sampling_seed"],
            )
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        result["runtime"] = runtime
        prefix.require(
            runtime["canvas_thread_scope"] == {"before": 1, "requested": 2, "inside": 2, "after": 1},
            "The scoped Torch thread count must be restored",
        )
        result.update(complete=True, passed=True, case=case["case"])
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def isolated_owner(function):
    """Reuse the tested owner with a private v2 namespace and no global mutations."""
    namespace = function.__globals__.copy()
    namespace.update(OUTPUT=OUTPUT, DRY=DRY, QA=QA, MODULE=MODULE, load_config=load_config, source_refs=source_refs)
    result = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--worker", type=int, choices=range(12))
    args = parser.parse_args()
    if args.dry_run:
        isolated_owner(original.dry_run)()
    elif args.execute:
        isolated_owner(original.execute)()
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
