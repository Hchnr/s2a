"""Own one registered metadata-corrected independent check; keep failed evidence."""

import argparse
import json
import os
from pathlib import Path

from execute_tracking_v5_saved_direction_backtracking import verify_native_qa as original_verify_native_qa
from run_wam_readout import owned_command

from s2a.native_environment import strict_native_environment
from s2a.preview_result_publication import publish_result
from s2a.saved_direction_backtracking_check_repair import (
    EXECUTION_ROOT,
    FAILED_OWNER_REF,
    FAILED_RESULT_REF,
    NEW_TEST,
    PLAN,
    QA_SOURCES,
    REPAIR_ROOT,
    repair_prerequisites,
)
from s2a.saved_direction_backtracking_evidence import (
    CPU_ENVIRONMENT,
    NATIVE_PYTHON,
    REPO,
    ROOT,
    SOURCES,
    closed_owner,
    finish,
    load_prerequisites,
    read,
    reference,
    remember,
    require,
)


def verify_native_qa(ref):
    qa = original_verify_native_qa(ref)
    require(
        set(QA_SOURCES) <= set(qa["source_sha256"])
        and (NEW_TEST in qa["command"] or str(REPO / NEW_TEST) in qa["command"]),
        "Same complete original QA plus the repair and actual endpoint metadata regressions",
    )
    return qa


def main(qa_ref, dry=False):
    strict_native_environment()
    require(
        not EXECUTION_ROOT.exists() and not REPAIR_ROOT.exists(), "New repair output; no overwrite or automatic retry"
    )
    qa = verify_native_qa(qa_ref)
    _, _, _, refs = load_prerequisites()
    for ref in repair_prerequisites().values():
        remember(ref, refs)
    remember(qa_ref, refs)
    qa_log_ref = remember(dict(path=str(Path(qa_ref["path"]).parent / "pytest.log"), sha256=qa["log_sha256"]), refs)
    producer_owner_ref = remember(
        reference(REPO / "outputs/tracking_v5_saved_direction_backtracking_execution_20261028/result.json"), refs
    )
    owner, producer = closed_owner(producer_owner_ref, refs)
    producer_ref = remember(reference(ROOT / "verified.json"), refs)
    require(
        owner["result_ref"] == producer_ref
        and producer_ref["sha256"] == "0957f81930cd65d08761220c76000374e1b7c9826872145319e311a42e26028e",
        "Original complete producer remains unchanged and is never rerun",
    )
    original_qa_ref = read(owner["registration_ref"])["native_component_qa_ref"]
    original_qa = read(remember(original_qa_ref, refs))
    require(
        all(original_qa["source_sha256"][relative] == qa["source_sha256"][relative] for relative in SOURCES),
        "Every original executed source, test and plan remains frozen",
    )
    for ref in producer["source_refs_unchanged"]:
        remember(ref, refs)
    finish(refs)
    if dry:
        print(
            json.dumps(
                dict(
                    dry_run=True,
                    passed=True,
                    source_count=len(refs),
                    preserved_failed_inner_steps=300,
                    preserved_failed_queries=313,
                )
            ),
            flush=True,
        )
        return
    environment = dict(os.environ, **CPU_ENVIRONMENT, PYTHONPATH=f"{REPO / 'src'}:{REPO / 'scripts'}")
    for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM"):
        environment.pop(key, None)
    registration = publish_result(
        EXECUTION_ROOT / "registration.json",
        dict(
            stage="check_v2",
            source_sha256={path: ref["sha256"] for path, ref in refs.items()},
            plan_ref=reference(PLAN),
            native_component_qa_ref=qa_ref,
            native_component_qa_log_ref=qa_log_ref,
            original_qa_ref=original_qa_ref,
            producer_owner_ref=producer_owner_ref,
            producer_ref=producer_ref,
            failed_owner_ref=FAILED_OWNER_REF,
            failed_result_ref=FAILED_RESULT_REF,
            preserved_failed_work=dict(trajectory_responses=1, numerical_inner_steps=300, query_count=313),
            response_budget=96,
            numerical_inner_steps=29700,
            query_count=30891,
            timeout_seconds=1800,
            cpu_runtime_environment=CPU_ENVIRONMENT,
            automatic_retry=False,
        ),
    )
    result = dict(complete=False, passed=False, stage="check_v2", registration_ref=registration)
    try:
        result["owned_command"] = owned_command(
            [str(NATIVE_PYTHON), str(REPO / "scripts/check_tracking_v5_saved_direction_backtracking_v2.py")],
            environment,
            EXECUTION_ROOT / "capture",
            timeout=1800,
        )
        result_ref = reference(REPAIR_ROOT / "verified.json")
        receipt = read(result_ref)
        require(receipt["complete"] is receipt["passed"] is True, "Complete independent96-response audit")
        finish(refs)
        strict_native_environment()
        result.update(complete=True, passed=True, result_ref=result_ref)
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish_result(EXECUTION_ROOT / "result.json", result)
    print(json.dumps(dict(complete=True, passed=True, stage="check_v2")), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-ref", required=True, type=Path)
    parser.add_argument("--qa-sha256", required=True)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    main(dict(path=str(args.qa_ref.resolve()), sha256=args.qa_sha256), args.dry)
