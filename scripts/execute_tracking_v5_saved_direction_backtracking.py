"""Own the one fixed backtracking batch or its complete independent CPU audit."""

import argparse
import json
import os
from pathlib import Path

from run_wam_readout import owned_command

from s2a.native_environment import strict_native_environment
from s2a.preview_result_publication import publish_result
from s2a.saved_direction_backtracking_evidence import (
    CHECK_ROOT,
    CPU_ENVIRONMENT,
    DIAGNOSTIC_REF,
    NATIVE_PYTHON,
    PLAN,
    QA_TESTS,
    REPO,
    ROOT,
    SOURCES,
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
    """Admit real successful synthetic QA for the unchanged final source roster."""
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
        "All new synthetic tests passed with actual wait in the fixed CPU scientific runtime",
    )
    require(set(SOURCES) <= set(qa["source_sha256"]), "QA binds every new source, test and plan")
    for relative, digest in qa["source_sha256"].items():
        path = (REPO / relative).resolve()
        require(path.is_relative_to(REPO), "QA sources remain inside this repository")
        checked(dict(path=str(path), sha256=digest))
    checked(dict(path=str(Path(ref["path"]).parent / "pytest.log"), sha256=qa["log_sha256"]))
    return qa


def main(stage, qa_ref, dry=False):
    strict_native_environment()
    require(stage in ("diagnose", "check"), "One explicit registered stage")
    suffix = "" if stage == "diagnose" else "_check"
    root = REPO / f"outputs/tracking_v5_saved_direction_backtracking{suffix}_execution_20261028"
    output = (ROOT if stage == "diagnose" else CHECK_ROOT) / "verified.json"
    entry = REPO / f"scripts/{'probe' if stage == 'diagnose' else 'check'}_tracking_v5_saved_direction_backtracking.py"
    require(not root.exists() and not output.parent.exists(), "No overwrite or automatic retry")
    qa = verify_native_qa(qa_ref)
    _, _, _, refs = load_prerequisites()
    remember(qa_ref, refs)
    qa_log_ref = remember(dict(path=str(Path(qa_ref["path"]).parent / "pytest.log"), sha256=qa["log_sha256"]), refs)
    for relative, digest in qa["source_sha256"].items():
        remember(dict(path=str((REPO / relative).resolve()), sha256=digest), refs)
    producer_ref, producer_owner_ref = None, None
    if stage == "check":
        producer_owner_ref = remember(
            reference(REPO / "outputs/tracking_v5_saved_direction_backtracking_execution_20261028/result.json"), refs
        )
        owner, producer = closed_owner(producer_owner_ref, refs)
        producer_ref = remember(reference(ROOT / "verified.json"), refs)
        owner_reg = read(owner["registration_ref"])
        require(
            owner["result_ref"] == producer_ref
            and owner_reg["stage"] == "diagnose"
            and owner_reg["native_component_qa_ref"] == qa_ref
            and read(owner["owned_command"]["launch"])["command"]
            == [str(NATIVE_PYTHON), str(REPO / "scripts/probe_tracking_v5_saved_direction_backtracking.py")],
            "Audit exactly the closed producer with the same source-bound QA",
        )
        for ref in producer["source_refs_unchanged"]:
            remember(ref, refs)
    finish(refs)
    if dry:
        print(json.dumps(dict(dry_run=True, passed=True, stage=stage, source_count=len(refs))), flush=True)
        return
    environment = dict(os.environ, **CPU_ENVIRONMENT, PYTHONPATH=f"{REPO / 'src'}:{REPO / 'scripts'}")
    for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM"):
        environment.pop(key, None)
    timeout = 1200 if stage == "diagnose" else 1800
    registration = publish_result(
        root / "registration.json",
        dict(
            stage=stage,
            source_sha256={path: ref["sha256"] for path, ref in refs.items()},
            diagnostic_ref=DIAGNOSTIC_REF,
            plan_ref=reference(PLAN),
            native_component_qa_ref=qa_ref,
            native_component_qa_log_ref=qa_log_ref,
            producer_owner_ref=producer_owner_ref,
            producer_ref=producer_ref,
            timeout_seconds=timeout,
            cpu_runtime_environment=CPU_ENVIRONMENT,
            response_budget=96,
            numerical_inner_steps=29700,
            query_count=30891,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            actual_consumed_reference_count=0,
            automatic_retry=False,
        ),
    )
    result = dict(complete=False, passed=False, stage=stage, registration_ref=registration)
    try:
        result["owned_command"] = owned_command(
            [str(NATIVE_PYTHON), str(entry)], environment, root / "capture", timeout=timeout
        )
        result_ref = reference(output)
        receipt = read(result_ref)
        require(
            receipt["complete"] is receipt["passed"] is True,
            "Complete batch/check without implying control qualification",
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
    parser.add_argument("--stage", required=True, choices=("diagnose", "check"))
    parser.add_argument("--qa-ref", required=True, type=Path)
    parser.add_argument("--qa-sha256", required=True)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    main(args.stage, dict(path=str(args.qa_ref.resolve()), sha256=args.qa_sha256), args.dry)
