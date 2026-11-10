"""Compare one causal quality warning with the reviewed development logs."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from s2a.infrastructure.publication import publish_result as publish
from s2a.research.joint_control.limits.quality_warning import summarize_observations
from scripts.simulation.runtime import replay_multicase_development_prefix as safe

REPO = safe.REPO
OUTPUT = REPO / "outputs/combined_quality_warning_20261110"
QA = REPO / "outputs/qa/combined_quality_warning_20261110/result.json"
PRIOR = {
    "path": str(REPO / "outputs/causal_joint_warning_audit_20261110/result.json"),
    "sha256": "3814fa596db7f5c76fcdacad9869468d29b6cf40776b23afe18270553ec7cad1",
}
MODULE = "scripts.tracking.joint_control.limits.audit_combined_quality_warning"


def source_refs():
    return [
        safe.reference(path)
        for path in (
            Path(__file__),
            REPO / "plans/v0.0.1/tracking/joint_control/combined_quality_warning_20261110.md",
            REPO / "src/s2a/research/joint_control/limits/quality_warning.py",
            REPO / "tests/research/joint_control/limits/test_quality_warning.py",
            REPO / "tests/research/joint_control/limits/test_combined_quality_warning_audit.py",
            Path(safe.__file__),
            REPO / "src/s2a/geometry/__init__.py",
            REPO / "src/s2a/infrastructure/publication.py",
        )
    ]


def checked_case(case, index, *, expected_endpoints=600):
    safe.require(case["case"]["case_index"] == index, "Original case order and identity")
    observations = case["observations"]
    safe.require(len(observations) == expected_endpoints, "Complete original observation prefix")
    analyzed = summarize_observations(observations)
    safe.require(
        analyzed["rules"]["velocity"]["first_warning_endpoint"]
        == case["rules"]["velocity_0p2s"]["first_alarm_endpoint"],
        "Original velocity warning identity",
    )
    return {
        "case": case["case"],
        "original_endpoint_outside_count_including_600": case["original_endpoint_outside_count"],
        **analyzed,
    }


def load_prior(seen):
    prior = safe.read(PRIOR, seen)
    safe.require(
        prior["complete"] is prior["passed"] is prior["sources_unchanged"] is True
        and prior["control_qualified"] is False,
        "The original audit must be complete but not a control qualification",
    )
    safe.require(
        prior["physics_steps"]
        == prior["model_initializations"]
        == prior["world_model_calls"]
        == prior["fresh_cases"]
        == 0,
        "Original audit has no physical, model, or final-holdout execution",
    )
    refs = prior["case_refs"]
    safe.require(len(refs) == 12 and len({ref["path"] for ref in refs}) == 12, "Twelve unique original case files")
    for ref in refs:
        safe.checked(ref, seen)
    return refs


def worker():
    safe.environment()
    registration = safe.read(safe.reference(OUTPUT / "registration.json"))
    before_sources = source_refs()
    safe.require(before_sources == registration["source_refs"], "Worker source identity")
    seen = {}
    refs = load_prior(seen)
    safe.require(refs == registration["case_refs"], "Registered original case identities")
    cases = [checked_case(safe.read(ref, seen), index) for index, ref in enumerate(refs)]
    case_refs = [publish(OUTPUT / f"case_{index:02d}.json", case) for index, case in enumerate(cases)]
    safe.require(source_refs() == before_sources, "Worker sources unchanged")
    safe.require(
        all(safe.reference(path)["sha256"] == digest for path, digest in seen.items()), "Worker inputs unchanged"
    )
    publish(
        OUTPUT / "worker_result.json",
        {
            "complete": True,
            "passed": True,
            "source_unchanged": True,
            "input_sha256": seen,
            "case_refs": case_refs,
            "endpoints": sum(len(case["observations"]) for case in cases),
            "physics_steps": 0,
            "model_initializations": 0,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "control_qualified": False,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        return worker()
    safe.environment()
    safe.require(not OUTPUT.exists(), "Exclusive new output; automatic retry prohibited")
    qa_ref = safe.reference(QA)
    qa = safe.read(qa_ref)
    safe.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "32-worker CPU QA must be complete and reaped",
    )
    before_sources = source_refs()
    qa_sources = safe.read(qa["source_before_ref"])["sources"]
    safe.require(
        all(qa_sources.get(str(Path(ref["path"]).relative_to(REPO))) == ref["sha256"] for ref in before_sources),
        "Audit source identities match CPU QA",
    )
    seen = {}
    refs = load_prior(seen)
    registration = publish(
        OUTPUT / "registration.json",
        {
            "prior_ref": PRIOR,
            "case_refs": refs,
            "qa_ref": qa_ref,
            "source_refs": before_sources,
            "automatic_retry": False,
            "control_qualified": False,
        },
    )
    started = time.monotonic()
    with (OUTPUT / "worker.log").open("xb") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", MODULE, "--worker"], cwd=REPO, stdout=log, stderr=subprocess.STDOUT
        )
        signalled = False
        try:
            returncode = process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            signalled = True
            process.terminate()
            try:
                returncode = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait()
    worker_path = OUTPUT / "worker_result.json"
    worker_ref = safe.reference(worker_path) if returncode == 0 and worker_path.is_file() else None
    worker_result = safe.read(worker_ref) if worker_ref else None
    unchanged = source_refs() == before_sources and all(
        safe.reference(path)["sha256"] == digest for path, digest in seen.items()
    )
    complete = bool(
        returncode == 0
        and not signalled
        and unchanged
        and worker_result
        and worker_result["complete"]
        and worker_result["passed"]
        and worker_result["source_unchanged"]
        and worker_result["input_sha256"] == seen
        and len(worker_result["case_refs"]) == 12
        and worker_result["endpoints"] == 7200
    )
    result_ref = publish(
        OUTPUT / "result.json",
        {
            "complete": complete,
            "passed": complete,
            "registration_ref": registration,
            "worker_ref": worker_ref,
            "worker_pid": process.pid,
            "worker_returncode": returncode,
            "actual_wait": True,
            "reaped": process.poll() is not None,
            "signal_sent": signalled,
            "source_and_input_unchanged": unchanged,
            "elapsed_seconds": time.monotonic() - started,
            "automatic_retry": False,
            "control_qualified": False,
        },
    )
    print(result_ref, flush=True)
    if not complete:
        raise RuntimeError("Combined warning audit did not complete")


if __name__ == "__main__":
    main()
