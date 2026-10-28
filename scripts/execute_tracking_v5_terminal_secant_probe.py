"""Own the complete fixed17-call terminal secant diagnostic or independent replay."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from execute_tracking_v5_directional_derivative_check_v3 import verify_native_qa as verify_v3_qa
from run_wam_readout import owned_command

from s2a.directional_derivative_check_v3_evidence import QA_SOURCES as V3_QA_SOURCES
from s2a.native_environment import strict_native_environment
from s2a.preview_result_publication import publish_result
from s2a.terminal_secant_evidence import (
    CPU_ENVIRONMENT,
    NATIVE_PYTHON,
    PLAN,
    QA_SOURCES,
    QA_TESTS,
    REPO,
    V3_QA_REF,
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

EXPERIMENT = "terminal_secant_probe"
STEP_SIZES = (1e-3, 7.5e-4, 5e-4, 2.5e-4, 1e-4)
DIRECTION_NAMES = ("orthogonal", "source_plus_orthogonal", "source_minus_orthogonal")
ZERO_FIELDS = ("physics_steps", "world_model_calls", "fresh_cases", "actual_consumed_reference_count")


def verify_native_qa(ref):
    qa = verify_v3_qa(ref)
    require(
        len(QA_SOURCES) == len(set(QA_SOURCES)) == 85
        and len(QA_TESTS) == len(set(QA_TESTS)) == 26
        and set(QA_SOURCES) <= set(qa["source_sha256"])
        and all(path in qa["command"] or str(REPO / path) in qa["command"] for path in QA_TESTS),
        "Complete native coverage:85sources and26test modules including the fixed secant diagnostic",
    )
    for path in QA_SOURCES:
        checked(dict(path=str(REPO / path), sha256=qa["source_sha256"][path]))
    return qa


def verify_prior_qa_subset(qa, refs):
    prior = verify_v3_qa(remember(V3_QA_REF, refs))
    require(
        len(V3_QA_SOURCES) == 75
        and set(prior["source_sha256"]) == set(V3_QA_SOURCES)
        and all(qa["source_sha256"].get(path) == prior["source_sha256"][path] for path in V3_QA_SOURCES),
        "Original75-source v3 QA must remain an identical digest subset of the new QA",
    )
    remember(dict(path=str(Path(V3_QA_REF["path"]).parent / "pytest.log"), sha256=prior["log_sha256"]), refs)
    return prior


def entry_command(stage):
    require(stage in ("diagnose", "check"), "One explicit registered stage")
    prefix = "probe" if stage == "diagnose" else "check"
    return [str(NATIVE_PYTHON), str(REPO / f"scripts/{prefix}_tracking_v5_terminal_secant_probe.py")]


def validate_call_counts(receipt):
    """Bill the repeated anchor as a complete response without scientific qualification."""
    require(
        type(receipt["response_evaluations"]) is int
        and type(receipt["unique_command_count"]) is int
        and receipt["response_evaluations"] == 17
        and receipt["unique_command_count"] == 16
        and receipt["numerical_inner_steps"] == 5100
        and receipt["query_count"] == 5305
        and receipt["total_iterations"] == 0
        and receipt["control_qualified"] is False
        and all(receipt[key] == 0 for key in ZERO_FIELDS),
        "All17full H12 calls,16unique commands and one current query; no control or reference consumption",
    )


def validate_receipt(receipt, stage):
    require(stage in ("diagnose", "check"), "One explicit registered stage")
    require(
        receipt["complete"] is receipt["passed"] is True and receipt["experiment"] == EXPERIMENT,
        "Complete fixed secant diagnostic or independent replay",
    )
    validate_call_counts(receipt)
    if stage == "diagnose":
        answer = receipt["answer"]
        entries, ledger = answer["evaluated_candidates"], answer["request_ledger"]
        require(
            receipt["prediction_calls"] == answer["response_evaluations"] == len(entries) == len(ledger) == 17
            and answer["unique_command_count"] == 16
            and answer["total_iterations"] == 0
            and answer["method"] == "fixed_terminal_secant_probe"
            and answer["horizon"] == 12
            and answer["max_responses"] == answer["planned_request_count"] == 17
            and tuple(answer["step_sizes"]) == STEP_SIZES
            and tuple(answer["direction_names"]) == DIRECTION_NAMES
            and answer["control_qualified"] is False,
            "Every fixed validation direction and scale is retained, with no optimization or added requests",
        )
        keys = [command_key(entry["commands"]) for entry in entries]
        require(
            len(set(keys[:-1])) == 16
            and keys[-1] == keys[0] == command_key(answer["anchor_commands"])
            and answer["repeatability"]["initial_response_index"] == 0
            and answer["repeatability"]["repeated_response_index"] == 16,
            "Exactly one final byte-identical anchor repeat is retained",
        )
        for index, row in enumerate(ledger):
            grid = 1 <= index <= 15
            require(
                row["request_index"] == row["response_index"] == row["calls_before"] == index
                and row["calls_after"] == index + 1
                and row["charged"] is True
                and row["role"] == ("grid" if grid else "anchor" if index == 0 else "anchor_repeat")
                and row["outcome"] == ("repeated" if index == 16 else "computed")
                and row["repeat_of"] == (0 if index == 16 else None)
                and command_key(row["commands"]) == keys[index]
                and row["command_key"] == keys[index].hex()
                and row["direction_index"] == ((index - 1) // 5 if grid else None)
                and row["direction_name"] == (DIRECTION_NAMES[(index - 1) // 5] if grid else None)
                and row["step_index"] == ((index - 1) % 5 if grid else None)
                and row["step"] == (STEP_SIZES[(index - 1) % 5] if grid else None),
                "Exact anchor, fifteen ordered grid calls and final paid repeat; no cache or skipped request",
            )
        for field in ("extrapolation_passed", "safety_accuracy_passed"):
            require(
                type(answer[field]) is bool and answer[field] is answer["qualification"][field],
                "Scientific success or failure is reported consistently, never required for execution completeness",
            )
        require(
            len(answer["predictions"]) == 15
            and len(answer["clipping_checks"]) == 17
            and receipt["witness_indices"] == [i for i, entry in enumerate(entries) if entry["feasible"]]
            and receipt["witness_found"] is bool(receipt["witness_indices"]),
            "All saved predictions, clipping records and witness labels are retained",
        )
    else:
        require(
            receipt["verified_responses"] == 17
            and receipt["mj_forward_queries"] == 5305
            and receipt["repeated_responses_checked"] == 1
            and all(receipt[field]["passed"] is True for field in ("geometry_check", "fit_check", "ledger_check"))
            and receipt["boundary_classification_disagreements"] == []
            and isinstance(receipt["native_quaternion_jit_signatures"], list)
            and bool(receipt["native_quaternion_jit_signatures"])
            and all(isinstance(value, str) and value for value in receipt["native_quaternion_jit_signatures"]),
            "Every full call independently replayed with native JIT, exact strict labels and independent secant algebra",
        )


def command_key(commands):
    value = np.asarray(commands, dtype=np.float64)
    require(value.shape == (12, 6) and np.isfinite(value).all(), "Complete finite H12 commands")
    return value.tobytes()


def main(stage, qa_ref, dry=False):
    strict_native_environment()
    require(all(os.environ.get(key) == value for key, value in CPU_ENVIRONMENT.items()), "Fixed CPU runtime")
    require(stage in ("diagnose", "check"), "One explicit registered stage")
    root = case_root(check=stage == "check", owner=True)
    output = case_root(check=stage == "check") / "verified.json"
    require(not root.exists() and not output.parent.exists(), "Exclusive new secant result; no overwrite or retry")
    qa = verify_native_qa(qa_ref)
    _, _, _, prior_closure, refs = load_prerequisites()
    remember(qa_ref, refs)
    log_ref = remember(dict(path=str(Path(qa_ref["path"]).parent / "pytest.log"), sha256=qa["log_sha256"]), refs)
    verify_prior_qa_subset(qa, refs)
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
            "Normally closed complete secant producer with the same frozen native QA and plan",
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
            prior_v3_native_component_qa_ref=V3_QA_REF,
            prior_qa_subset_count=75,
            prior_closure=prior_closure,
            producer_owner_ref=producer_owner_ref,
            producer_ref=producer_ref,
            timeout_seconds=timeout,
            cpu_runtime_environment=CPU_ENVIRONMENT,
            seed=42,
            horizon=12,
            response_budget=17,
            expected_responses=17,
            planned_request_count=17,
            unique_command_budget=16,
            expected_unique_commands=16,
            expected_numerical_inner_steps=5100,
            expected_queries=5305,
            mandatory_anchor_repeats=1,
            step_sizes=STEP_SIZES,
            direction_names=DIRECTION_NAMES,
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
        for ref in receipt["source_refs_unchanged"]:
            remember(ref, refs)
        if stage == "check":
            require(
                receipt["response_evaluations"] == producer["response_evaluations"]
                and receipt["unique_command_count"] == producer["unique_command_count"]
                and receipt["witness_indices"] == producer["witness_indices"]
                and receipt["witness_found"] is producer["witness_found"],
                "No partial secant replay, omitted repeat or changed strict witness classification",
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
