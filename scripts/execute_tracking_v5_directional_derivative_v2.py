"""Own the isolated v2 directional diagnostic or its complete replay."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from execute_tracking_v5_directional_derivative import verify_native_qa as verify_v1_qa
from run_wam_readout import owned_command

from s2a.directional_derivative_v2_evidence import (
    CPU_ENVIRONMENT,
    NATIVE_PYTHON,
    PLAN,
    QA_SOURCES,
    QA_TESTS,
    REPO,
    SQP_OWNER_REF,
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
from s2a.native_environment import strict_native_environment
from s2a.preview_result_publication import publish_result

EXPERIMENT = "directional_derivative_consistency"
STEP_SIZES = (1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8, 1e-8)
ZERO_FIELDS = ("physics_steps", "world_model_calls", "fresh_cases", "actual_consumed_reference_count")


def verify_native_qa(ref):
    qa = verify_v1_qa(ref)
    require(
        len(QA_SOURCES) == len(set(QA_SOURCES)) == 68
        and len(QA_TESTS) == len(set(QA_TESTS)) == 21
        and set(QA_SOURCES) <= set(qa["source_sha256"])
        and all(path in qa["command"] or str(REPO / path) in qa["command"] for path in QA_TESTS),
        "Complete frozen native coverage:68sources and21test modules including this diagnostic",
    )
    for path in QA_SOURCES:
        checked(dict(path=str(REPO / path), sha256=qa["source_sha256"][path]))
    return qa


def entry_command(stage):
    require(stage in ("diagnose", "check"), "One explicit registered stage")
    prefix = "probe" if stage == "diagnose" else "check"
    return [str(NATIVE_PYTHON), str(REPO / f"scripts/{prefix}_tracking_v5_directional_derivative_v2.py")]


def command_key(commands):
    value = np.asarray(commands, dtype=np.float64)
    require(value.shape == (12, 6) and np.all(np.isfinite(value)), "Complete finite H12 commands")
    return value.tobytes()


def validate_receipt(receipt, stage):
    require(stage in ("diagnose", "check"), "One explicit registered stage")
    require(
        receipt["complete"] is receipt["passed"] is True and receipt["experiment"] == EXPERIMENT,
        "Complete fixed diagnostic or independent replay",
    )
    count, unique = receipt["response_evaluations"], receipt["unique_command_count"]
    require(
        type(count) is int
        and type(unique) is int
        and 2 <= count <= 80
        and 1 <= unique == count - 1 <= 79
        and receipt["numerical_inner_steps"] == count * 300
        and receipt["query_count"] == 1 + count * 312
        and receipt["total_iterations"] == 0
        and receipt["control_qualified"] is False
        and all(receipt[key] == 0 for key in ZERO_FIELDS),
        "Every full H12 call including the mandatory repeat is billed without control qualification",
    )
    if stage == "diagnose":
        answer = receipt["answer"]
        entries, ledger = answer["evaluated_candidates"], answer["request_ledger"]
        require(
            receipt["prediction_calls"] == answer["response_evaluations"] == len(entries) == count
            and answer["unique_command_count"] == unique
            and answer["total_iterations"] == 0
            and answer["method"] == "fixed_directional_derivative_probe"
            and answer["horizon"] == 12
            and answer["max_responses"] == 80
            and answer["planned_request_count"] == len(ledger) == 80
            and answer["source_stages"] == [24, 25, 26]
            and tuple(answer["step_sizes"]) == STEP_SIZES
            and answer["control_qualified"] is False,
            "The entire registered fixed grid is retained without optimization or added requests",
        )
        keys = [command_key(entry["commands"]) for entry in entries]
        require(
            len(set(keys[:-1])) == unique
            and keys[-1] == keys[0] == command_key(answer["anchor_commands"])
            and ledger[0]["role"] == "anchor"
            and ledger[-1]["role"] == "anchor_repeat"
            and ledger[-1]["outcome"] == "repeated"
            and ledger[-1]["charged"] is True
            and ledger[-1]["response_index"] == count - 1
            and [row["response_index"] for row in ledger if row["charged"]] == list(range(count))
            and answer["repeatability"]["initial_response_index"] == 0
            and answer["repeatability"]["repeated_response_index"] == count - 1,
            "Only the final anchor call repeats exact bytes and every actual call is charged once",
        )
    else:
        require(
            receipt["verified_responses"] == count
            and receipt["mj_forward_queries"] == 1 + count * 312
            and receipt["repeated_responses_checked"] == 1
            and receipt["ledger_check"]["passed"] is True
            and receipt["boundary_classification_disagreements"] == [],
            "Every actual call including the repeat independently verified with identical strict labels",
        )


def main(stage, qa_ref, dry=False):
    strict_native_environment()
    require(all(os.environ.get(key) == value for key, value in CPU_ENVIRONMENT.items()), "Fixed CPU runtime")
    require(stage in ("diagnose", "check"), "One explicit registered stage")
    root = case_root(check=stage == "check", owner=True)
    output = case_root(check=stage == "check") / "verified.json"
    require(not root.exists() and not output.parent.exists(), "Exclusive new result; no overwrite or retry")
    qa = verify_native_qa(qa_ref)
    _, _, _, sqp_closure, refs = load_prerequisites()
    remember(qa_ref, refs)
    log_ref = remember(dict(path=str(Path(qa_ref["path"]).parent / "pytest.log"), sha256=qa["log_sha256"]), refs)
    producer_owner_ref, producer_ref = None, None
    if stage == "check":
        producer_owner_ref = remember(reference(case_root(owner=True) / "result.json"), refs)
        owner, producer = closed_owner(producer_owner_ref, refs)
        producer_ref = remember(reference(case_root() / "verified.json"), refs)
        registration = read(remember(owner["registration_ref"], refs))
        require(
            owner["result_ref"] == producer_ref
            and owner["experiment"] == registration["experiment"] == producer["experiment"] == EXPERIMENT
            and owner["stage"] == registration["stage"] == "diagnose"
            and registration["native_component_qa_ref"] == qa_ref
            and registration["plan_ref"] == reference(PLAN)
            and read(owner["owned_command"]["launch"])["command"] == entry_command("diagnose"),
            "Exact completed diagnostic with the identical frozen source-bound QA and plan",
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
    timeout = 600 if stage == "diagnose" else 1800
    registration_ref = publish_result(
        root / "registration.json",
        dict(
            experiment=EXPERIMENT,
            stage=stage,
            plan_ref=reference(PLAN),
            source_sha256={path: ref["sha256"] for path, ref in refs.items()},
            native_component_qa_ref=qa_ref,
            native_component_qa_log_ref=log_ref,
            prior_sqp_owner_ref=SQP_OWNER_REF,
            prior_sqp_closure=sqp_closure,
            producer_owner_ref=producer_owner_ref,
            producer_ref=producer_ref,
            timeout_seconds=timeout,
            cpu_runtime_environment=CPU_ENVIRONMENT,
            seed=42,
            horizon=12,
            anchor_response_index=1819,
            source_stages=[24, 25, 26],
            step_sizes=STEP_SIZES,
            planned_request_count=80,
            response_budget=80,
            unique_command_budget=79,
            mandatory_anchor_repeats=1,
            iteration_budget=0,
            **dict.fromkeys(ZERO_FIELDS, 0),
            automatic_retry=False,
        ),
    )
    result = dict(complete=False, passed=False, experiment=EXPERIMENT, stage=stage, registration_ref=registration_ref)
    environment = dict(os.environ, **CPU_ENVIRONMENT, PYTHONPATH=f"{REPO / 'src'}:{REPO / 'scripts'}")
    for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM"):
        environment.pop(key, None)
    try:
        result["owned_command"] = owned_command(entry_command(stage), environment, root / "capture", timeout=timeout)
        result_ref = reference(output)
        receipt = read(result_ref)
        validate_receipt(receipt, stage)
        if stage == "check":
            require(
                receipt["response_evaluations"] == producer["response_evaluations"]
                and receipt["unique_command_count"] == producer["unique_command_count"],
                "No partial replay or omitted repeat",
            )
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
