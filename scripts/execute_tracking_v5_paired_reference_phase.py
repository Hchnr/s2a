"""Own one fixed H13 case or its complete independent CPU reconstruction."""

import argparse
import json
from pathlib import Path

from probe_tracking_v5_paired_reference_phase import (
    PLAN,
    PREFIX_OWNER,
    PREFIX_RESULT,
    closed_owner,
    phase_prerequisite,
)
from run_tracking_preview_wam_development_v5 import cpu_environment
from run_wam_readout import owned_command

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require

QA_TESTS = (
    "tests/test_paired_reference_phase.py",
    "tests/test_paired_reference_phase_independent_check.py",
    "tests/test_paired_reference_phase_accounting.py",
)
QA_SOURCES = (
    "scripts/probe_tracking_v5_paired_reference_phase.py",
    "scripts/check_tracking_v5_paired_reference_phase.py",
    "scripts/execute_tracking_v5_paired_reference_phase.py",
    "src/s2a/paired_reference_phase.py",
    "plans/v0.0.1/paired_reference_phase_20261028.md",
    *QA_TESTS,
)
NATIVE_PYTHON = REPO / "outputs/libero_env311/bin/python"
CPU_ENVIRONMENT = dict(
    CUDA_VISIBLE_DEVICES="",
    LD_LIBRARY_PATH="",
    PYTHONDONTWRITEBYTECODE="1",
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1",
    LP_NUM_THREADS="1",
)


def verify_native_qa(ref):
    """Admit the final persisted synthetic QA, bound to every new source."""
    qa = read(ref)
    command = qa["command"]
    require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is True
        and qa["returncode"] == 0
        and command[:3] == [str(NATIVE_PYTHON), "-m", "pytest"]
        and all(relative in command or str(REPO / relative) in command for relative in QA_TESTS)
        and qa["runtime"]["python"].startswith("3.11.")
        and qa["runtime"]["numpy"] == "1.26.4"
        and qa["runtime"]["scipy"] == "1.15.3"
        and all(qa["environment"].get(key) == value for key, value in CPU_ENVIRONMENT.items()),
        "All new synthetic tests passed in the fixed CPU scientific runtime",
    )
    require(set(QA_SOURCES) <= set(qa["source_sha256"]), "QA must bind every new source, test and plan")
    for relative, digest in qa["source_sha256"].items():
        path = (REPO / relative).resolve()
        require(path.is_relative_to(REPO), "QA source remains inside this repository")
        checked(dict(path=str(path), sha256=digest))
    log_ref = dict(path=str(Path(ref["path"]).parent / "pytest.log"), sha256=qa["log_sha256"])
    checked(log_ref)
    return qa, log_ref


def main(case_name, stage, qa_ref):
    require(case_name in ("early", "late") and stage in ("diagnose", "check"), "One explicit registered case/stage")
    suffix = "" if stage == "diagnose" else "_check"
    root = REPO / f"outputs/tracking_v5_paired_reference_phase_{case_name}{suffix}_execution_20261028"
    output = REPO / f"outputs/tracking_v5_paired_reference_phase_{case_name}{suffix}_20261028/verified.json"
    entry = REPO / f"scripts/{'probe' if stage == 'diagnose' else 'check'}_tracking_v5_paired_reference_phase.py"
    require(not root.exists() and not output.parent.exists(), "No overwrite or automatic retry")
    qa, qa_log_ref = verify_native_qa(qa_ref)
    sources = {}

    def remember(ref):
        checked(ref)
        require(ref["path"] not in sources or sources[ref["path"]] == ref["sha256"], "One immutable source digest")
        sources[ref["path"]] = ref["sha256"]
        return ref

    prefix_owner_ref = remember(reference(PREFIX_OWNER))
    prefix_owner, prefix_check = closed_owner(prefix_owner_ref, remember)
    prefix_check_reg = read(remember(prefix_check["registration_ref"]))
    prefix_ref = remember(reference(PREFIX_RESULT))
    require(
        prefix_check_reg["producer_ref"] == prefix_ref
        and prefix_check["trace_count"] == 4000
        and prefix_check["numerical_inner_steps"] == 1200000
        and prefix_check["mj_forward_queries"] == 1248001
        and prefix_check["witness_found"] is prefix_check["control_qualified"] is False,
        "Prefix prerequisite includes every response before this new case",
    )
    for ref in prefix_check["source_refs_unchanged"]:
        remember(ref)
    prefix_producer_owner, prefix_producer = closed_owner(prefix_check_reg["producer_owner_ref"], remember)
    require(prefix_producer_owner["result_ref"] == prefix_ref, "Closed exact prefix producer")
    prefix_reg = read(remember(prefix_producer["registration_ref"]))
    phase_ref = phase_prerequisite(prefix_ref, prefix_reg["prior_trajectory_ref"], remember)
    prior_ref = read(prefix_owner["registration_ref"])["prior_registration_ref"]
    prior = read(remember(prior_ref))
    prepared = read(remember(prior["prepared_ref"]))
    for relative in QA_SOURCES:
        remember(reference(REPO / relative))
    for relative in (
        "src/s2a/native_goal_trajectory.py",
        "src/s2a/native_goal_joint_response.py",
        "src/s2a/nonlinear_joint_response.py",
        "src/s2a/mujoco_robot_dynamics.py",
        "src/s2a/prefix_preserving_trajectory_plan.py",
        "src/s2a/joint_preview_constraint_audit.py",
        "scripts/check_tracking_v5_prefix_preserving_trajectory_control.py",
        "scripts/check_tracking_v5_state_updated_response.py",
        "scripts/run_wam_readout.py",
        "scripts/run_tracking_preview_wam_development_v5.py",
    ):
        remember(reference(REPO / relative))
    for relative, digest in qa["source_sha256"].items():
        remember(dict(path=str((REPO / relative).resolve()), sha256=digest))
    remember(qa_ref)
    remember(qa_log_ref)
    producer_owner_ref = None
    producer_ref = None
    if stage == "check":
        producer_owner_ref = remember(
            reference(REPO / f"outputs/tracking_v5_paired_reference_phase_{case_name}_execution_20261028/result.json")
        )
        producer_owner, producer = closed_owner(producer_owner_ref, remember)
        producer_owner_reg = read(producer_owner["registration_ref"])
        producer_ref = remember(
            reference(REPO / f"outputs/tracking_v5_paired_reference_phase_{case_name}_20261028/verified.json")
        )
        require(
            producer_owner["result_ref"] == producer_ref
            and producer["case_name"] == case_name
            and producer_owner_reg["case_name"] == case_name
            and producer_owner_reg["stage"] == "diagnose"
            and producer_owner_reg["native_component_qa_ref"] == qa_ref,
            "Check only the exact closed case with the same source-bound QA",
        )
        require(
            read(producer_owner["owned_command"]["launch"])["command"]
            == [
                str(NATIVE_PYTHON),
                str(REPO / "scripts/probe_tracking_v5_paired_reference_phase.py"),
                "--case",
                case_name,
            ],
            "Producer child command binds this independent case",
        )
        for ref in producer["source_refs_unchanged"]:
            remember(ref)
    else:
        peer = "late" if case_name == "early" else "early"
        peer_registration = (
            REPO / f"outputs/tracking_v5_paired_reference_phase_{peer}_execution_20261028/registration.json"
        )
        if peer_registration.exists():
            peer_reg = read(reference(peer_registration))
            require(
                peer_reg["case_name"] == peer
                and peer_reg["stage"] == "diagnose"
                and peer_reg["native_component_qa_ref"] == qa_ref
                and peer_reg["source_sha256"] == sources,
                "Both cases must use exactly the same frozen sources and QA; no result-conditioned revision",
            )
    timeout = 1200 if stage == "diagnose" else 5400
    environment = cpu_environment(prepared)
    environment.update(CPU_ENVIRONMENT)
    reg = publish(
        root / "registration.json",
        dict(
            case_name=case_name,
            stage=stage,
            source_sha256=sources,
            prior_registration_ref=prior_ref,
            prior_prefix_check_owner_ref=prefix_owner_ref,
            prior_prefix_check_ref=prefix_owner["result_ref"],
            prior_prefix_ref=prefix_ref,
            saved_phase_diagnostic_ref=phase_ref,
            plan_ref=reference(PLAN),
            native_component_qa_ref=qa_ref,
            native_component_qa_log_ref=qa_log_ref,
            producer_owner_ref=producer_owner_ref,
            producer_ref=producer_ref,
            timeout_seconds=timeout,
            cpu_runtime_environment=CPU_ENVIRONMENT,
            response_budget=4000,
            iteration_budget=60,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            actual_consumed_reference_count=0,
            automatic_retry=False,
        ),
    )
    result = dict(complete=False, passed=False, case_name=case_name, stage=stage, registration_ref=reg)
    try:
        command = owned_command(
            [str(NATIVE_PYTHON), str(entry), "--case", case_name],
            environment,
            root / "capture",
            timeout=timeout,
        )
        result["owned_command"] = command
        ref = reference(output)
        receipt = read(ref)
        require(
            receipt["complete"] is receipt["passed"] is True and receipt["case_name"] == case_name,
            "This bounded case/audit completed, without implying a witness or physical control",
        )
        for path, digest in sources.items():
            checked(dict(path=path, sha256=digest))
        result.update(complete=True, passed=True, result_ref=ref)
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish(root / "result.json", result)
    print(json.dumps(dict(complete=True, passed=True, case_name=case_name, stage=stage)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=("early", "late"))
    parser.add_argument("--stage", required=True, choices=("diagnose", "check"))
    parser.add_argument("--qa-ref", required=True, type=Path)
    parser.add_argument("--qa-sha256", required=True)
    args = parser.parse_args()
    main(args.case, args.stage, dict(path=str(args.qa_ref.resolve()), sha256=args.qa_sha256))
