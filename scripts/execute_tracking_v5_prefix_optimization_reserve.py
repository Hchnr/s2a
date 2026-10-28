"""Own one source-bound callback-backtracking search or its complete replay."""

import argparse
import json
import os
from pathlib import Path

from execute_tracking_v5_prefix_callback_backtracking import verify_native_qa as verify_callback_qa
from run_wam_readout import owned_command

from s2a.native_environment import strict_native_environment
from s2a.prefix_optimization_reserve_evidence import (
    CALLBACK_CHECK_REG_REF,
    CPU_ENVIRONMENT,
    NATIVE_PYTHON,
    PLAN,
    PREFIX_OPTIMIZATION_MARGIN,
    QA_SOURCES,
    QA_TESTS,
    REPO,
    TRUST_REGISTRATION_REFS,
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
from s2a.preview_result_publication import publish_result


def verify_native_qa(ref):
    qa = verify_callback_qa(ref)
    require(
        set(QA_SOURCES) <= set(qa["source_sha256"])
        and all(path in qa["command"] or str(REPO / path) in qa["command"] for path in QA_TESTS),
        "Same complete frozen native coverage plus every new callback-backtracking source/test/plan",
    )
    for path in QA_SOURCES:
        checked(dict(path=str(REPO / path), sha256=qa["source_sha256"][path]))
    return qa


def entry_command(stage):
    require(stage in ("diagnose", "check"), "One explicit stage")
    prefix = "probe" if stage == "diagnose" else "check"
    return [str(NATIVE_PYTHON), str(REPO / f"scripts/{prefix}_tracking_v5_prefix_optimization_reserve.py")]


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
        require(
            receipt["prediction_calls"]
            == answer["response_evaluations"]
            == len(answer["evaluated_candidates"])
            == count
            and receipt["total_iterations"] == answer["total_iterations"] <= 60
            and answer["total_stages"] <= 60
            and answer["prefix_optimization_margin"] == PREFIX_OPTIMIZATION_MARGIN,
            "All producer responses and shared finite search budgets agree",
        )
    else:
        require(
            receipt["verified_responses"] == count
            and receipt["mj_forward_queries"] == 1 + count * 312
            and receipt["boundary_classification_disagreements"] == [],
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
            and read(owner["owned_command"]["launch"])["command"] == entry_command("diagnose"),
            "Exact completed candidate search with the identical frozen source-bound QA",
        )
        validate_receipt(producer, "diagnose")
        for ref in producer["source_refs_unchanged"]:
            remember(ref, refs)
    finish(refs)
    if dry:
        print(
            json.dumps(dict(dry_run=True, passed=True, stage=stage, source_count=len(refs), new_responses=0)),
            flush=True,
        )
        return
    timeout = 1200 if stage == "diagnose" else 5400
    registration_ref = publish_result(
        root / "registration.json",
        dict(
            case_name="candidate",
            experiment="prefix_optimization_reserve",
            stage=stage,
            plan_ref=reference(PLAN),
            source_sha256={path: ref["sha256"] for path, ref in refs.items()},
            native_component_qa_ref=qa_ref,
            native_component_qa_log_ref=log_ref,
            prior_replay_registration_refs=TRUST_REGISTRATION_REFS,
            prior_callback_check_registration_ref=CALLBACK_CHECK_REG_REF,
            prefix_optimization_margin=PREFIX_OPTIMIZATION_MARGIN,
            producer_owner_ref=producer_owner_ref,
            producer_ref=producer_ref,
            timeout_seconds=timeout,
            cpu_runtime_environment=CPU_ENVIRONMENT,
            response_budget=4000,
            iteration_budget=60,
            stage_budget=60,
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
        result_ref = reference(output)
        receipt = read(result_ref)
        validate_receipt(receipt, stage)
        if stage == "check":
            require(receipt["response_evaluations"] == producer["response_evaluations"], "No partial replay")
        finish(refs)
        strict_native_environment()
        result.update(complete=True, passed=True, result_ref=result_ref)
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
