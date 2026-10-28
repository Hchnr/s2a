"""Own one restored-anchor H12 SQP search or its complete independent replay."""

import argparse
import json
import os
from pathlib import Path

from execute_tracking_v5_accepted_point_sqp import verify_native_qa as verify_accepted_qa
from run_wam_readout import owned_command

from s2a.native_environment import strict_native_environment
from s2a.preview_result_publication import publish_result
from s2a.restored_anchor_sqp_evidence import (
    CPU_ENVIRONMENT,
    NATIVE_PYTHON,
    PLAN,
    PREFIX_OPTIMIZATION_MARGIN,
    PRIOR_QA_REF,
    PRIOR_QA_SOURCES,
    QA_SOURCES,
    QA_TESTS,
    REPO,
    SOURCES,
    TAIL_CLOSURE_REF,
    TAIL_INITIAL_REF,
    TAIL_PRODUCER_REF,
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


def verify_native_qa(ref):
    qa = verify_accepted_qa(ref)
    prior = read(PRIOR_QA_REF)
    require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["prior_sources_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is qa["automatic_retry"] is False
        and qa["returncode"] == 0
        and qa["source_count"] == len(QA_SOURCES) == 110
        and qa["test_module_count"] == len(QA_TESTS) == 33
        and set(qa["source_sha256"]) == set(QA_SOURCES)
        and qa["prior_native_component_qa_ref"] == PRIOR_QA_REF
        and qa["prior_source_count"] == len(PRIOR_QA_SOURCES) == 100
        and set(prior["source_sha256"]) == set(PRIOR_QA_SOURCES)
        and all(qa["source_sha256"][path] == sha for path, sha in prior["source_sha256"].items()),
        "Complete110-source/33-module native QA preserves the exact prior100-source subset",
    )
    command = qa["command"]
    require(
        len(command) == 38
        and command[:4] == [str(NATIVE_PYTHON), "-m", "pytest", "-q"]
        and command[4:-1] == list(QA_TESTS)
        and command[-1].startswith("--basetemp="),
        "All exact native test modules executed without selection flags",
    )
    for path in QA_SOURCES:
        checked(dict(path=str(REPO / path), sha256=qa["source_sha256"][path]))
    return qa


def validate_initial_guard(guard):
    require(
        guard["checked"] is guard["passed"] is guard["exact"] is guard["comparison"]["exact"] is True
        and guard["expected_ref"] == TAIL_INITIAL_REF
        and all(
            type(guard[key]) is int and guard[key] == value
            for key, value in {
                "hook_calls": 1,
                "request_index": 0,
                "response_index": 0,
                "request_count": 1,
                "response_count": 1,
                "stages_before": 0,
                "iterations_before": 0,
                "qp_iterations_before": 0,
            }.items()
        ),
        "One complete exact initial record gate ran before every FD/QP/stage",
    )


def entry_command(stage):
    require(stage in ("diagnose", "check"), "One explicit stage")
    prefix = "probe" if stage == "diagnose" else "check"
    return [str(NATIVE_PYTHON), str(REPO / f"scripts/{prefix}_tracking_v5_restored_anchor_sqp.py")]


def validate_receipt(receipt, stage):
    require(
        receipt["complete"] is receipt["passed"] is True
        and receipt["case_name"] == "candidate"
        and receipt["prefix_optimization_margin"] == PREFIX_OPTIMIZATION_MARGIN,
        "Complete registered candidate or independent replay",
    )
    count = receipt["response_evaluations"]
    require(
        type(count) is int
        and 1 <= count <= 4000
        and receipt["numerical_inner_steps"] == count * 300
        and receipt["query_count"] == 1 + count * 312
        and receipt["control_qualified"] is False
        and all(receipt[key] == 0 for key in ("physics_steps", "world_model_calls", "fresh_cases")),
        "Every complete unique H12 trajectory is charged without granting control qualification",
    )
    if stage == "diagnose":
        answer = receipt["answer"]
        validate_initial_guard(receipt["initial_record_guard"])
        require(receipt["initial_record_guard"] == answer["initial_record_guard"], "Same producer initial guard")
        require(
            receipt["prediction_calls"]
            == answer["response_evaluations"]
            == len(answer["evaluated_candidates"])
            == count
            and receipt["total_iterations"] == answer["total_iterations"] <= 60
            and answer["total_stages"] <= 60
            and answer["prefix_optimization_margin"] == PREFIX_OPTIMIZATION_MARGIN
            and answer["method"] == "restored_anchor_lagrangian_sqp"
            and answer["qp_iteration_limit"] == 100
            and answer["qp_certificate_tolerance"] == 1e-6
            and 0 <= answer["total_qp_iterations"] <= 6000,
            "All producer responses and shared finite search budgets agree",
        )
    else:
        require(receipt["initial_record_check"]["passed"] is True, "Independent complete initial record gate audit")
        require(
            receipt["verified_responses"] == count
            and receipt["mj_forward_queries"] == 1 + count * 312
            and receipt["boundary_classification_disagreements"] == []
            and receipt["ledger_check"]["passed"] is True
            and receipt["ledger_check"]["initial_record_check"] == receipt["initial_record_check"]
            and receipt["native_quaternion_jit_signatures"]
            and type(receipt["witness_found"]) is bool
            and receipt["witness_found"] is bool(receipt["ledger_check"]["witness_indices"]),
            "Entire unique trace independently verified with identical strict classification",
        )


def main(stage, qa_ref, dry=False):
    strict_native_environment()
    require(all(os.environ.get(key) == value for key, value in CPU_ENVIRONMENT.items()), "Fixed CPU runtime")
    require(stage in ("diagnose", "check"), "One explicit registered stage")
    root = case_root(check=stage == "check", owner=True)
    output = case_root(check=stage == "check") / "verified.json"
    require(not root.exists() and not output.parent.exists(), "Exclusive new result; no overwrite or retry")
    qa = verify_native_qa(qa_ref)
    _, _, _, refs = load_prerequisites()
    remember(qa_ref, refs)
    log_ref = remember(dict(path=str(Path(qa_ref["path"]).parent / "pytest.log"), sha256=qa["log_sha256"]), refs)
    producer_owner_ref, producer_ref = None, None
    if stage == "check":
        producer_owner_ref = remember(reference(case_root(owner=True) / "result.json"), refs)
        owner, producer = closed_owner(producer_owner_ref, refs)
        producer_ref = remember(reference(case_root() / "verified.json"), refs)
        registration = read(owner["registration_ref"])
        require(
            owner["result_ref"] == producer_ref
            and owner["case_name"] == registration["case_name"] == producer["case_name"] == "candidate"
            and owner["stage"] == registration["stage"] == "diagnose"
            and registration["native_component_qa_ref"] == qa_ref
            and registration["plan_ref"] == reference(PLAN)
            and registration["tail_closure_ref"] == TAIL_CLOSURE_REF
            and registration["expected_initial_ref"] == TAIL_INITIAL_REF
            and read(owner["owned_command"]["launch"])["command"] == entry_command("diagnose"),
            "Exact completed candidate search with the identical frozen source-bound QA",
        )
        validate_receipt(producer, "diagnose")
        for ref in producer["source_refs_unchanged"]:
            remember(ref, refs)
    finish(refs)
    if dry:
        print(
            json.dumps(
                dict(dry_run=True, passed=True, stage=stage, source_count=len(refs), new_responses=0, new_queries=0)
            ),
            flush=True,
        )
        return
    timeout = 1200 if stage == "diagnose" else 5400
    registration_ref = publish_result(
        root / "registration.json",
        dict(
            case_name="candidate",
            experiment="restored_anchor_sqp",
            stage=stage,
            plan_ref=reference(PLAN),
            source_sha256={path: ref["sha256"] for path, ref in refs.items()},
            native_component_qa_ref=qa_ref,
            native_component_qa_log_ref=log_ref,
            tail_producer_ref=TAIL_PRODUCER_REF,
            tail_closure_ref=TAIL_CLOSURE_REF,
            expected_initial_ref=TAIL_INITIAL_REF,
            prefix_optimization_margin=PREFIX_OPTIMIZATION_MARGIN,
            producer_owner_ref=producer_owner_ref,
            producer_ref=producer_ref,
            timeout_seconds=timeout,
            cpu_runtime_environment=CPU_ENVIRONMENT,
            response_budget=4000,
            iteration_budget=60,
            stage_budget=60,
            qp_iteration_limit=100,
            qp_total_iteration_limit=6000,
            qp_certificate_tolerance=1e-6,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
    )
    result = dict(complete=False, passed=False, case_name="candidate", stage=stage, registration_ref=registration_ref)
    environment = dict(os.environ, **CPU_ENVIRONMENT, PYTHONPATH=f"{REPO / 'src'}:{REPO / 'scripts'}")
    for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM"):
        environment.pop(key, None)
    try:
        result["owned_command"] = owned_command(entry_command(stage), environment, root / "capture", timeout=timeout)
        result_ref = remember(reference(output), refs)
        receipt = read(result_ref)
        validate_receipt(receipt, stage)
        if stage == "check":
            require(receipt["response_evaluations"] == producer["response_evaluations"], "No partial replay")
            require(
                receipt["ledger_check"]["witness_indices"]
                == [i for i, row in enumerate(producer["answer"]["evaluated_candidates"]) if row["feasible"]]
                and receipt["witness_found"] is producer["answer"]["witness_found"],
                "All independent full-witness labels agree with the completed producer",
            )
        child_registration = read(remember(receipt["registration_ref"], refs))
        require(
            child_registration["plan_ref"] == reference(PLAN)
            and child_registration["expected_initial_ref"] == TAIL_INITIAL_REF
            and child_registration["seed"] == 42
            and {ref["path"]: ref["sha256"] for ref in child_registration["source_refs"]}
            == {str(REPO / path): qa["source_sha256"][path] for path in SOURCES}
            and len(child_registration["source_refs"]) == len(SOURCES),
            "The completed child binds the same plan, initial record and all exact QA sources",
        )
        if stage == "diagnose":
            require(child_registration["algorithm_math_proof"]["passed"] is True, "Registered algorithm AST proof")
        else:
            require(
                child_registration["producer_ref"] == producer_ref
                and child_registration["producer_owner_ref"] == producer_owner_ref,
                "The independent child binds the exact completed producer and owner",
            )
        for ref in [
            *child_registration["source_refs"],
            *child_registration["input_refs"],
            *receipt["source_refs_unchanged"],
            *result["owned_command"].values(),
        ]:
            remember(ref, refs)
        finish(refs)
        strict_native_environment()
        result.update(complete=True, passed=True, result_ref=result_ref, source_refs_unchanged=list(refs.values()))
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish_result(root / "result.json", result)
    print(json.dumps(dict(complete=True, passed=True, stage=stage)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("diagnose", "check"), required=True)
    parser.add_argument("--qa-ref", type=Path, required=True)
    parser.add_argument("--qa-sha256", required=True)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    main(args.stage, dict(path=str(args.qa_ref.resolve()), sha256=args.qa_sha256), args.dry)
