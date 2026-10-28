"""Own one complete H12 replay using the same verified full asset snapshot."""

import argparse
import json
import os
from pathlib import Path

from execute_tracking_v5_prefix_trust_control import verify_native_qa as original_verify_native_qa
from run_wam_readout import owned_command

from s2a.native_environment import strict_native_environment
from s2a.prefix_trust_asset_repair import (
    ALL_QA_TESTS as QA_TESTS,
)
from s2a.prefix_trust_asset_repair import (
    FAILED_OWNER_REF,
    FAILED_RESULT_REF,
    PLAN,
    PRODUCER_OWNER_REFS,
    PRODUCER_REFS,
    QA_SOURCES,
    load_asset_repair_prerequisites,
    load_snapshot,
    repair_root,
)
from s2a.prefix_trust_control_evidence import (
    CPU_ENVIRONMENT,
    NATIVE_PYTHON,
    REPO,
    checked,
    finish,
    read,
    reference,
    remember,
    require,
)
from s2a.preview_result_publication import publish_result


def verify_native_qa(ref):
    qa = original_verify_native_qa(ref)
    require(
        set(QA_SOURCES) <= set(qa["source_sha256"])
        and all(path in qa["command"] or str(REPO / path) in qa["command"] for path in QA_TESTS),
        "Same original tests plus every asset repair source and synthetic regression",
    )
    for path in QA_SOURCES:
        checked(dict(path=str(REPO / path), sha256=qa["source_sha256"][path]))
    return qa


def main(case_name, qa_ref, dry=False):
    strict_native_environment()
    require(all(os.environ.get(k) == v for k, v in CPU_ENVIRONMENT.items()), "Fixed single-thread CPU environment")
    root, output_root = repair_root(case_name, owner=True), repair_root(case_name)
    require(not root.exists() and not output_root.exists(), "Fresh v2 output; no overwrite or automatic retry")
    qa = verify_native_qa(qa_ref)
    _, _, _, refs = load_asset_repair_prerequisites()
    remember(qa_ref, refs)
    log_ref = remember(dict(path=str(Path(qa_ref["path"]).parent / "pytest.log"), sha256=qa["log_sha256"]), refs)
    snapshot = load_snapshot()
    peer = "candidate" if case_name == "baseline" else "baseline"
    peer_registration = repair_root(peer, owner=True) / "registration.json"
    if peer_registration.exists():
        prior = read(reference(peer_registration))
        require(
            prior["native_component_qa_ref"] == qa_ref
            and prior["source_sha256"] == {path: ref["sha256"] for path, ref in refs.items()},
            "Both complete replays share exactly one repair QA and asset/source closure",
        )
    finish(refs)
    count = 4000 if case_name == "baseline" else 3148
    if dry:
        print(
            json.dumps(
                dict(
                    dry_run=True,
                    passed=True,
                    case_name=case_name,
                    source_count=len(refs),
                    expected_responses=count,
                    new_responses=0,
                )
            ),
            flush=True,
        )
        return
    registration = publish_result(
        root / "registration.json",
        dict(
            case_name=case_name,
            stage="check_v2",
            plan_ref=reference(PLAN),
            source_sha256={path: ref["sha256"] for path, ref in refs.items()},
            native_component_qa_ref=qa_ref,
            native_component_qa_log_ref=log_ref,
            producer_owner_ref=PRODUCER_OWNER_REFS[case_name],
            producer_ref=PRODUCER_REFS[case_name],
            failed_owner_ref=FAILED_OWNER_REF,
            failed_result_ref=FAILED_RESULT_REF,
            snapshot_ref=snapshot["snapshot_ref"],
            compile_ref=snapshot["compile_ref"],
            preserved_failed_work=dict(response_evaluations=0, query_count=0, numerical_inner_steps=0),
            expected_responses=count,
            expected_inner_steps=count * 300,
            expected_queries=count * 312 + 1,
            timeout_seconds=5400,
            cpu_runtime_environment=CPU_ENVIRONMENT,
            automatic_retry=False,
        ),
    )
    result = dict(complete=False, passed=False, case_name=case_name, stage="check_v2", registration_ref=registration)
    environment = dict(os.environ, **CPU_ENVIRONMENT, PYTHONPATH=f"{REPO / 'src'}:{REPO / 'scripts'}")
    for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM"):
        environment.pop(key, None)
    try:
        result["owned_command"] = owned_command(
            [
                str(NATIVE_PYTHON),
                str(REPO / "scripts/check_tracking_v5_prefix_trust_control_v2.py"),
                "--case",
                case_name,
            ],
            environment,
            root / "capture",
            timeout=5400,
        )
        result_ref = reference(output_root / "verified.json")
        report = read(result_ref)
        require(
            report["complete"] is report["passed"] is True
            and report["case_name"] == case_name
            and report["response_evaluations"] == report["verified_responses"] == count
            and report["numerical_inner_steps"] == count * 300
            and report["query_count"] == report["mj_forward_queries"] == count * 312 + 1
            and report["boundary_classification_disagreements"] == []
            and report["control_qualified"] is False,
            "Complete original trace independently replayed without altered labels or physical qualification",
        )
        finish(refs)
        strict_native_environment()
        result.update(complete=True, passed=True, result_ref=result_ref)
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish_result(root / "result.json", result)
    print(json.dumps(dict(complete=True, passed=True, case_name=case_name, stage="check_v2")), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--qa-ref", type=Path, required=True)
    parser.add_argument("--qa-sha256", required=True)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    main(args.case, dict(path=str(args.qa_ref.resolve()), sha256=args.qa_sha256), args.dry)
