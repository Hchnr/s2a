"""Own one finite matched-warm H12 producer or its complete independent audit."""

import argparse
import json
import os
from pathlib import Path

from run_wam_readout import owned_command

from s2a.native_environment import strict_native_environment
from s2a.prefix_trust_control_evidence import (
    BACKTRACKING_CHECK_OWNER_REF,
    BACKTRACKING_REF,
    CPU_ENVIRONMENT,
    NATIVE_PYTHON,
    PLAN,
    QA_TESTS,
    REPO,
    SOURCES,
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
    qa = read(ref)
    command = qa["command"]
    require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is True
        and qa["returncode"] == 0
        and command[:3] == [str(NATIVE_PYTHON), "-m", "pytest"]
        and all(relative in command or str(REPO / relative) in command for relative in QA_TESTS)
        and qa["runtime"]["python"].startswith("3.11.")
        and qa["runtime"]["numpy"] == "1.26.4"
        and qa["runtime"]["scipy"] == "1.15.3"
        and all(qa["environment"].get(key) == value for key, value in CPU_ENVIRONMENT.items()),
        "All registered synthetic tests passed in the fixed CPU scientific runtime with actual wait",
    )
    require(set(SOURCES) <= set(qa["source_sha256"]), "QA covers every new source, test and plan")
    for relative, digest in qa["source_sha256"].items():
        path = (REPO / relative).resolve()
        require(path.is_relative_to(REPO), "QA sources remain in this repository")
        checked(dict(path=str(path), sha256=digest))
    checked(dict(path=str(Path(ref["path"]).parent / "pytest.log"), sha256=qa["log_sha256"]))
    return qa


def main(case_name, stage, qa_ref, dry=False):
    strict_native_environment()
    require(stage in ("diagnose", "check"), "One explicit registered stage")
    root = case_root(case_name, check=stage == "check", owner=True)
    output = case_root(case_name, check=stage == "check") / "verified.json"
    require(not root.exists() and not output.parent.exists(), "No overwrite or automatic retry")
    qa = verify_native_qa(qa_ref)
    _, _, _, refs = load_prerequisites()
    remember(qa_ref, refs)
    qa_log_ref = remember(dict(path=str(Path(qa_ref["path"]).parent / "pytest.log"), sha256=qa["log_sha256"]), refs)
    producer_owner_ref, producer_ref = None, None
    if stage == "check":
        producer_owner_ref = remember(reference(case_root(case_name, owner=True) / "result.json"), refs)
        owner, producer = closed_owner(producer_owner_ref, refs)
        producer_ref = remember(reference(case_root(case_name) / "verified.json"), refs)
        owner_reg = read(owner["registration_ref"])
        require(
            owner["result_ref"] == producer_ref
            and producer["case_name"] == case_name
            and owner_reg["case_name"] == case_name
            and owner_reg["stage"] == "diagnose"
            and owner_reg["native_component_qa_ref"] == qa_ref
            and read(owner["owned_command"]["launch"])["command"]
            == [
                str(NATIVE_PYTHON),
                str(REPO / "scripts/probe_tracking_v5_prefix_trust_control.py"),
                "--case",
                case_name,
            ],
            "Complete exact arm produced using the identical frozen source-bound QA",
        )
        for ref in producer["source_refs_unchanged"]:
            remember(ref, refs)
    else:
        peer = "candidate" if case_name == "baseline" else "baseline"
        peer_registration = case_root(peer, owner=True) / "registration.json"
        if peer_registration.exists():
            prior = read(reference(peer_registration))
            require(
                prior["case_name"] == peer
                and prior["stage"] == "diagnose"
                and prior["native_component_qa_ref"] == qa_ref
                and prior["source_sha256"] == {path: ref["sha256"] for path, ref in refs.items()},
                "Both comparison arms use identical frozen source and QA; no outcome-conditioned revisions",
            )
    finish(refs)
    if dry:
        print(
            json.dumps(dict(dry_run=True, passed=True, case_name=case_name, stage=stage, source_count=len(refs))),
            flush=True,
        )
        return
    environment = dict(os.environ, **CPU_ENVIRONMENT, PYTHONPATH=f"{REPO / 'src'}:{REPO / 'scripts'}")
    for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM"):
        environment.pop(key, None)
    timeout = 1200 if stage == "diagnose" else 5400
    registration = publish_result(
        root / "registration.json",
        dict(
            case_name=case_name,
            stage=stage,
            source_sha256={path: ref["sha256"] for path, ref in refs.items()},
            plan_ref=reference(PLAN),
            backtracking_ref=BACKTRACKING_REF,
            backtracking_check_owner_ref=BACKTRACKING_CHECK_OWNER_REF,
            native_component_qa_ref=qa_ref,
            native_component_qa_log_ref=qa_log_ref,
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
            actual_consumed_reference_count=0,
            automatic_retry=False,
        ),
    )
    result = dict(complete=False, passed=False, case_name=case_name, stage=stage, registration_ref=registration)
    entry = REPO / f"scripts/{'probe' if stage == 'diagnose' else 'check'}_tracking_v5_prefix_trust_control.py"
    try:
        result["owned_command"] = owned_command(
            [str(NATIVE_PYTHON), str(entry), "--case", case_name],
            environment,
            root / "capture",
            timeout=timeout,
        )
        result_ref = reference(output)
        receipt = read(result_ref)
        require(
            receipt["complete"] is receipt["passed"] is True and receipt["case_name"] == case_name,
            "This complete arm or full audit does not imply task success",
        )
        finish(refs)
        strict_native_environment()
        result.update(complete=True, passed=True, result_ref=result_ref)
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish_result(root / "result.json", result)
    print(json.dumps(dict(complete=True, passed=True, case_name=case_name, stage=stage)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=("baseline", "candidate"))
    parser.add_argument("--stage", required=True, choices=("diagnose", "check"))
    parser.add_argument("--qa-ref", required=True, type=Path)
    parser.add_argument("--qa-sha256", required=True)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    main(args.case, args.stage, dict(path=str(args.qa_ref.resolve()), sha256=args.qa_sha256), args.dry)
