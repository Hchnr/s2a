"""Own only the complete v3 replay of the already closed v2 producer."""

import argparse
import json
import os
from pathlib import Path

from execute_tracking_v5_directional_derivative_v2 import (
    EXPERIMENT,
    STEP_SIZES,
    ZERO_FIELDS,
)
from execute_tracking_v5_directional_derivative_v2 import validate_receipt as validate_v2_receipt
from execute_tracking_v5_directional_derivative_v2 import verify_native_qa as verify_v2_qa
from run_wam_readout import owned_command

from s2a.directional_derivative_check_v3_evidence import (
    CPU_ENVIRONMENT,
    NATIVE_PYTHON,
    PLAN,
    PRODUCER_OWNER_REF,
    PRODUCER_QA_REF,
    PRODUCER_REF,
    QA_SOURCES,
    QA_TESTS,
    REPO,
    case_root,
    checked,
    closed_owner,
    finish,
    load_prerequisites,
    read,
    reference,
    remember,
    require,
)
from s2a.directional_derivative_v2_evidence import PLAN as PRODUCER_PLAN
from s2a.directional_derivative_v2_evidence import QA_SOURCES as PRODUCER_QA_SOURCES
from s2a.native_environment import strict_native_environment
from s2a.preview_result_publication import publish_result


def verify_native_qa(ref):
    qa = verify_v2_qa(ref)
    require(
        len(QA_SOURCES) == len(set(QA_SOURCES)) == 75
        and len(QA_TESTS) == len(set(QA_TESTS)) == 23
        and set(QA_SOURCES) <= set(qa["source_sha256"])
        and all(path in qa["command"] or str(REPO / path) in qa["command"] for path in QA_TESTS),
        "Complete frozen native coverage:75sources and23test modules including the isolated v3 replay",
    )
    for path in QA_SOURCES:
        checked(dict(path=str(REPO / path), sha256=qa["source_sha256"][path]))
    return qa


def entry_command(stage):
    require(stage in ("diagnose", "check"), "Only the original producer identity and new checker are registered")
    name = (
        "probe_tracking_v5_directional_derivative_v2.py"
        if stage == "diagnose"
        else "check_tracking_v5_directional_derivative_v3.py"
    )
    return [str(NATIVE_PYTHON), str(REPO / "scripts" / name)]


def validate_receipt(receipt, stage="check"):
    validate_v2_receipt(receipt, stage)
    require(
        receipt["response_evaluations"] == 77
        and receipt["unique_command_count"] == 76
        and receipt["numerical_inner_steps"] == 23100
        and receipt["query_count"] == 24025,
        "The complete original77 calls and76unique commands; no partial replay or new producer",
    )
    if stage == "check":
        require(
            receipt["saved_stencil_check"]["passed"] is True and receipt["historical_anchor_check"]["passed"] is True,
            "Complete grid replay plus saved stencil and historical anchor independently checked",
        )


def verify_producer_qa_subset(qa, refs):
    old_qa = verify_v2_qa(remember(PRODUCER_QA_REF, refs))
    require(
        set(old_qa["source_sha256"]) == set(PRODUCER_QA_SOURCES)
        and len(PRODUCER_QA_SOURCES) == 68
        and all(qa["source_sha256"].get(path) == old_qa["source_sha256"][path] for path in PRODUCER_QA_SOURCES),
        "The original68producer QA sources must be the identical digest subset of the new QA",
    )
    remember(dict(path=str(Path(PRODUCER_QA_REF["path"]).parent / "pytest.log"), sha256=old_qa["log_sha256"]), refs)
    return old_qa


def main(qa_ref, dry=False):
    strict_native_environment()
    require(all(os.environ.get(key) == value for key, value in CPU_ENVIRONMENT.items()), "Fixed CPU runtime")
    root, output = case_root(check=True, owner=True), case_root(check=True) / "verified.json"
    require(not root.exists() and not output.parent.exists(), "Exclusive new v3 replay; no overwrite or retry")
    qa = verify_native_qa(qa_ref)
    _, _, _, sqp_closure, refs = load_prerequisites()
    remember(qa_ref, refs)
    log_ref = remember(dict(path=str(Path(qa_ref["path"]).parent / "pytest.log"), sha256=qa["log_sha256"]), refs)
    verify_producer_qa_subset(qa, refs)
    producer_owner_ref = remember(PRODUCER_OWNER_REF, refs)
    require(
        producer_owner_ref == reference(case_root(owner=True) / "result.json")
        and PRODUCER_REF == reference(case_root() / "verified.json"),
        "Exact fixed v2 producer locations and hashes",
    )
    owner, producer = closed_owner(producer_owner_ref, refs)
    producer_ref = remember(PRODUCER_REF, refs)
    registration = read(remember(owner["registration_ref"], refs))
    require(
        owner["result_ref"] == producer_ref
        and owner["experiment"] == registration["experiment"] == producer["experiment"] == EXPERIMENT
        and owner["stage"] == registration["stage"] == "diagnose"
        and registration["native_component_qa_ref"] == PRODUCER_QA_REF
        and registration["plan_ref"] == reference(PRODUCER_PLAN)
        and read(owner["owned_command"]["launch"])["command"] == entry_command("diagnose"),
        "Exact normally closed v2 producer with its original native QA and plan",
    )
    validate_receipt(producer, "diagnose")
    for ref in producer["source_refs_unchanged"]:
        remember(ref, refs)
    finish(refs)
    if dry:
        print(
            json.dumps(dict(dry_run=True, passed=True, stage="check", source_count=len(refs), new_responses=0)),
            flush=True,
        )
        return
    registration_ref = publish_result(
        root / "registration.json",
        dict(
            experiment=EXPERIMENT,
            stage="check",
            plan_ref=reference(PLAN),
            source_sha256={path: ref["sha256"] for path, ref in refs.items()},
            native_component_qa_ref=qa_ref,
            native_component_qa_log_ref=log_ref,
            producer_native_component_qa_ref=PRODUCER_QA_REF,
            producer_qa_subset_count=68,
            prior_sqp_closure=sqp_closure,
            producer_owner_ref=producer_owner_ref,
            producer_ref=producer_ref,
            timeout_seconds=1800,
            cpu_runtime_environment=CPU_ENVIRONMENT,
            seed=42,
            horizon=12,
            anchor_response_index=1819,
            source_stages=[24, 25, 26],
            step_sizes=STEP_SIZES,
            planned_request_count=80,
            expected_responses=77,
            expected_unique_commands=76,
            response_budget=80,
            unique_command_budget=79,
            mandatory_anchor_repeats=1,
            iteration_budget=0,
            **dict.fromkeys(ZERO_FIELDS, 0),
            automatic_retry=False,
        ),
    )
    result = dict(complete=False, passed=False, experiment=EXPERIMENT, stage="check", registration_ref=registration_ref)
    environment = dict(os.environ, **CPU_ENVIRONMENT, PYTHONPATH=f"{REPO / 'src'}:{REPO / 'scripts'}")
    for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM"):
        environment.pop(key, None)
    try:
        result["owned_command"] = owned_command(entry_command("check"), environment, root / "capture", timeout=1800)
        result_ref = reference(output)
        receipt = read(result_ref)
        validate_receipt(receipt)
        require(
            receipt["response_evaluations"] == producer["response_evaluations"]
            and receipt["unique_command_count"] == producer["unique_command_count"],
            "No partial replay or omitted anchor repeat",
        )
        finish(refs)
        strict_native_environment()
        result.update(complete=True, passed=True, result_ref=result_ref)
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish_result(root / "result.json", result)
    print(json.dumps(dict(complete=True, passed=True, stage="check")), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-ref", type=Path, required=True)
    parser.add_argument("--qa-sha256", required=True)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    main(dict(path=str(args.qa_ref.resolve()), sha256=args.qa_sha256), args.dry)
