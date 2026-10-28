"""Own one isolated full replay repair of the unchanged tail translation producer."""

import argparse
import json
import os
from pathlib import Path

from execute_tracking_v5_tail_translation_reserve import entry_command as prior_entry_command
from execute_tracking_v5_tail_translation_reserve import validate_receipt as validate_receipt
from execute_tracking_v5_tail_translation_reserve import verify_native_qa as verify_prior_qa
from run_wam_readout import owned_command

from s2a.native_environment import strict_native_environment
from s2a.preview_result_publication import publish_result
from s2a.tail_translation_check_v2_evidence import (
    CPU_ENVIRONMENT,
    FAILED_CHECK_REF,
    FAILED_OWNER_REF,
    NATIVE_PYTHON,
    PLAN,
    PRIOR_QA_REF,
    PRODUCER_OWNER_REF,
    PRODUCER_REF,
    QA_SOURCES,
    QA_TESTS,
    REPAIR_PLAN,
    REPO,
    case_root,
    closed_owner,
    finish,
    load_prerequisites,
    read,
    reference,
    remember,
    require,
    verify_failed_replay,
)
from s2a.tail_translation_reserve_evidence import QA_SOURCES as PRIOR_QA_SOURCES


def verify_native_qa(ref):
    qa = verify_prior_qa(ref)
    require(
        len(QA_SOURCES) == len(set(QA_SOURCES)) == 100
        and len(QA_TESTS) == len(set(QA_TESTS)) == 30
        and set(qa["source_sha256"]) == set(QA_SOURCES)
        and qa["command"][4:-1] == list(QA_TESTS),
        "Complete100-source/30-module repair QA",
    )
    prior = verify_prior_qa(PRIOR_QA_REF)
    require(
        set(prior["source_sha256"]) == set(PRIOR_QA_SOURCES)
        and all(qa["source_sha256"][p] == sha for p, sha in prior["source_sha256"].items()),
        "All executed95 sources remain an unchanged QA subset",
    )
    for path, sha in qa["source_sha256"].items():
        require(reference(REPO / path)["sha256"] == sha, "All current repair sources match actual QA")
    return qa


def entry_command(stage):
    if stage == "diagnose":
        return prior_entry_command(stage)
    require(stage == "check", "Only the original producer or isolated checker")
    return [str(NATIVE_PYTHON), str(REPO / "scripts/check_tracking_v5_tail_translation_reserve_v2.py")]


def validate_checker_registration(registration):
    require(
        registration["producer_ref"] == PRODUCER_REF
        and registration["producer_owner_ref"] == PRODUCER_OWNER_REF
        and registration["plan_ref"] == reference(PLAN)
        and registration["repair_plan_ref"] == reference(REPAIR_PLAN)
        and registration["failed_checker_ref"] == FAILED_CHECK_REF
        and registration["failed_owner_ref"] == FAILED_OWNER_REF,
        "Repaired checker binds the original producer, both plans and preserved failure",
    )


def main(qa_ref, dry=False):
    strict_native_environment()
    require(all(os.environ.get(k) == v for k, v in CPU_ENVIRONMENT.items()), "Fixed single-thread CPU")
    root = case_root(check=True, owner=True)
    output = case_root(check=True) / "verified.json"
    require(not root.exists() and not output.parent.exists(), "Exclusive check-only repair; no overwrite or retry")
    qa = verify_native_qa(qa_ref)
    _, _, _, _, refs = load_prerequisites()
    remember(qa_ref, refs)
    log_ref = remember(dict(path=str(Path(qa_ref["path"]).parent / "pytest.log"), sha256=qa["log_sha256"]), refs)
    prior = read(remember(PRIOR_QA_REF, refs))
    remember(dict(path=str(Path(PRIOR_QA_REF["path"]).parent / "pytest.log"), sha256=prior["log_sha256"]), refs)
    failure = verify_failed_replay(refs)
    owner, producer = closed_owner(remember(PRODUCER_OWNER_REF, refs), refs)
    registration = read(remember(owner["registration_ref"], refs))
    require(
        owner["result_ref"] == PRODUCER_REF
        and owner["stage"] == "diagnose"
        and registration["native_component_qa_ref"] == PRIOR_QA_REF
        and registration["plan_ref"] == reference(PLAN)
        and read(owner["owned_command"]["launch"])["command"] == entry_command("diagnose"),
        "The original producer used its original95-source QA and remains closed",
    )
    validate_receipt(producer, "diagnose")
    for ref in producer["source_refs_unchanged"]:
        remember(ref, refs)
    finish(refs)
    if dry:
        print(
            json.dumps(dict(dry_run=True, passed=True, source_count=len(refs), new_responses=0, new_queries=0)),
            flush=True,
        )
        return
    registration_ref = publish_result(
        root / "registration.json",
        dict(
            experiment="tail_translation_reserve_check_v2",
            stage="check",
            seed=42,
            plan_ref=reference(REPAIR_PLAN),
            original_plan_ref=reference(PLAN),
            native_component_qa_ref=qa_ref,
            native_component_qa_log_ref=log_ref,
            prior_native_component_qa_ref=PRIOR_QA_REF,
            prior_qa_subset_count=95,
            producer_owner_ref=PRODUCER_OWNER_REF,
            producer_ref=PRODUCER_REF,
            failed_checker_ref=FAILED_CHECK_REF,
            failed_owner_ref=FAILED_OWNER_REF,
            failure=failure,
            source_sha256={p: ref["sha256"] for p, ref in refs.items()},
            cpu_runtime_environment=CPU_ENVIRONMENT,
            timeout_seconds=1800,
            expected_responses=14,
            expected_unique_commands=13,
            expected_numerical_inner_steps=4200,
            expected_queries=4369,
            iteration_budget=0,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            actual_consumed_reference_count=0,
            automatic_retry=False,
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        experiment="tail_translation_reserve_check_v2",
        stage="check",
        registration_ref=registration_ref,
    )
    environment = dict(os.environ, **CPU_ENVIRONMENT, PYTHONPATH=f"{REPO / 'src'}:{REPO / 'scripts'}")
    for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM"):
        environment.pop(key, None)
    try:
        result["owned_command"] = owned_command(entry_command("check"), environment, root / "capture", timeout=1800)
        result_ref = remember(reference(output), refs)
        receipt = read(result_ref)
        validate_receipt(receipt, "check")
        require(
            receipt["witness_indices"] == producer["witness_indices"]
            and receipt["witness_found"] is producer["witness_found"],
            "All original witness labels unchanged",
        )
        for field in ("improvement_found", "sufficient_reserve_restored", "selected_response_index"):
            require(
                receipt["ledger_check"][field]
                == receipt["independent_restoration_check"][field]
                == producer["answer"][field],
                "Independent original scientific labels, allowing False",
            )
        for ref in receipt["source_refs_unchanged"]:
            remember(ref, refs)
        validate_checker_registration(read(remember(receipt["registration_ref"], refs)))
        for ref in result["owned_command"].values():
            remember(ref, refs)
        finish(refs)
        strict_native_environment()
        result.update(complete=True, passed=True, result_ref=result_ref, source_refs_unchanged=list(refs.values()))
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish_result(root / "result.json", result)
    print(json.dumps(dict(complete=True, passed=True, stage="check_v2")), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-ref", type=Path, required=True)
    parser.add_argument("--qa-sha256", required=True)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    main(dict(path=str(args.qa_ref.resolve()), sha256=args.qa_sha256), args.dry)
