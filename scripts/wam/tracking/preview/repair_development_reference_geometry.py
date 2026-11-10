"""Separately complete the six cases stopped by the exact rotation assertion."""

import argparse
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from s2a.infrastructure.artifacts import REPO
from s2a.research.wam.tracking.preview.development_reference_geometry_roundoff import RoundoffBoundedProbe
from s2a.research.wam.training.clean.clean_training_runtime import publish, read, require
from scripts.wam.tracking.preview.diagnose_development_reference_geometry import (
    REVIEW,
    SNAPSHOT,
    asset_paths,
    process_case,
    safe_environment,
    safe_ref,
)

OUTPUT = REPO / "outputs/multicase_geometry_roundoff_repair_20261110"
ASSESSMENT = {
    "path": str(REPO / "outputs/multicase_development_reference_geometry_v1_close_20261110/assessment.json"),
    "sha256": "20b8722bf806834bfb92493300cb82b57f8088889d6aa3ddf5fd81e63567bb55",
}
QA = REPO / "outputs/qa/multicase_geometry_roundoff_repair_20261110/result.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/multicase_geometry_roundoff_repair_20261110.md"
ORIGINAL_PLAN = REPO / "plans/v0.0.1/tracking/joint_control/multicase_development_reference_geometry_20261110.md"
ORIGINAL_SCRIPT = REPO / "scripts/wam/tracking/preview/diagnose_development_reference_geometry.py"
LIBRARY = REPO / "src/s2a/research/wam/tracking/preview/development_reference_geometry.py"
REPAIR_LIBRARY = REPO / "src/s2a/research/wam/tracking/preview/development_reference_geometry_roundoff.py"
TEST = REPO / "tests/research/wam/tracking/preview/test_development_reference_geometry.py"
REPAIR_TEST = REPO / "tests/research/wam/tracking/preview/test_development_reference_geometry_roundoff.py"
MISSING = (1, 3, 4, 5, 7, 8)
COMPLETED = (0, 2, 6, 9, 10, 11)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-index", type=int, choices=MISSING)
    args = parser.parse_args()
    safe_environment()
    review, snapshot, assessment = read(REVIEW), read(SNAPSHOT), read(ASSESSMENT)
    require(review["complete"] is review["passed"] is True and len(review["cases"]) == 12, "Full reviewed matrix")
    require(tuple(x["case_index"] for x in assessment["completed"]) == COMPLETED, "Exactly six closed cases")
    require(tuple(x["case_index"] for x in assessment["failed"]) == MISSING, "Exactly six incomplete cases")

    if args.case_index is not None:
        reg = read(safe_ref(OUTPUT / "registration.json"))
        require(
            reg["assessment_ref"] == ASSESSMENT and reg["case_indices"] == list(MISSING), "Explicit repair registration"
        )
        paths = {a["declared_path"]: a["snapshot_ref"]["path"] for a in snapshot["assets"]}
        result = process_case(args.case_index, review, paths, probe_factory=RoundoffBoundedProbe)
        print(publish(OUTPUT / f"case_{args.case_index:02d}/result.json", result), flush=True)
        return

    require(not OUTPUT.exists(), "One new output directory; no overwrite or retry")
    qa_ref = safe_ref(QA)
    qa = read(qa_ref)
    require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "QA closed",
    )
    sources = [
        safe_ref(p)
        for p in (Path(__file__), ORIGINAL_SCRIPT, LIBRARY, REPAIR_LIBRARY, TEST, REPAIR_TEST, PLAN, ORIGINAL_PLAN)
    ]
    qa_sources = read(qa["source_before_ref"])["sources"]
    require(
        all(qa_sources.get(str(Path(r["path"]).relative_to(REPO))) == r["sha256"] for r in sources),
        "QA source identity",
    )
    assets = asset_paths(snapshot)
    old_refs = {entry["case_index"]: entry["result_ref"] for entry in assessment["completed"]}
    require(all(safe_ref(ref["path"]) == ref for ref in old_refs.values()), "Closed original results unchanged")
    registration = publish(
        OUTPUT / "registration.json",
        {
            "assessment_ref": ASSESSMENT,
            "review_ref": REVIEW,
            "snapshot_ref": SNAPSHOT,
            "qa_ref": qa_ref,
            "source_refs": sources,
            "case_indices": list(MISSING),
            "preserved_case_refs": old_refs,
            "object_rotation_atol": 1e-12,
            "object_rotation_rtol": 0,
            "max_queries_per_case": 1201,
            "max_total_queries": 7206,
            "max_models": 6,
            "physics_steps": 0,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
    )

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Owner received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    workers = []
    launch_error = None
    try:
        for index in MISSING:
            directory = OUTPUT / f"case_{index:02d}"
            directory.mkdir()
            log = (directory / "worker.log").open("x")
            command = [
                sys.executable,
                "-m",
                "scripts.wam.tracking.preview.repair_development_reference_geometry",
                "--case-index",
                str(index),
            ]
            try:
                process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            except BaseException:
                log.close()
                raise
            workers.append({"index": index, "process": process, "log": log})
    except BaseException:
        launch_error = traceback.format_exc()
        for worker in workers:
            if worker["process"].poll() is None:
                try:
                    worker["process"].terminate()
                    worker["signaled"] = True
                except ProcessLookupError:
                    pass

    deadline = time.monotonic() + 1200
    lifecycle = []
    for worker in workers:
        process = worker["process"]
        signaled = worker.get("signaled", False)
        try:
            code = process.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            signaled = True
            process.terminate()
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait()
        except BaseException:
            launch_error = traceback.format_exc()
            for other in workers:
                if other["process"].poll() is None:
                    try:
                        other["process"].terminate()
                        other["signaled"] = True
                    except ProcessLookupError:
                        pass
            signaled = worker.get("signaled", False)
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait()
        worker["log"].close()
        lifecycle.append(
            {
                "case_index": worker["index"],
                "pid": process.pid,
                "returncode": code,
                "actual_wait": True,
                "reaped": True,
                "signal_sent": signaled,
                "log_ref": safe_ref(OUTPUT / f"case_{worker['index']:02d}/worker.log"),
            }
        )
    workers_ref = publish(OUTPUT / "workers.json", {"workers": lifecycle, "launch_error": launch_error})
    if launch_error or len(lifecycle) != 6 or not all(w["returncode"] == 0 and not w["signal_sent"] for w in lifecycle):
        publish(
            OUTPUT / "result.json",
            {
                "complete": False,
                "passed": False,
                "registration_ref": registration,
                "workers_ref": workers_ref,
                "automatic_retry": False,
            },
        )
        raise ValueError("All six repair workers must succeed; partial results retained")
    require(asset_paths(read(SNAPSHOT)) == assets, "Snapshot assets unchanged")
    require(all(safe_ref(r["path"]) == r for r in sources), "Frozen repair source unchanged")
    require(all(safe_ref(r["path"]) == r for r in old_refs.values()), "Original closed cases unchanged")
    new_refs = {i: safe_ref(OUTPUT / f"case_{i:02d}/result.json") for i in MISSING}
    results = [read(new_refs[i]) for i in MISSING]
    count = sum(case["mj_forward_queries"] for case in results)
    require(count <= 7206, "Repair query budget")
    require(all(case["max_object_rotation_change"] <= 1e-12 for case in results), "Object rotation guard")
    combined_refs = [old_refs[i] if i in old_refs else new_refs[i] for i in range(12)]
    print(
        publish(
            OUTPUT / "result.json",
            {
                "complete": True,
                "passed": True,
                "registration_ref": registration,
                "workers_ref": workers_ref,
                "repair_case_refs": [new_refs[i] for i in MISSING],
                "combined_case_refs": combined_refs,
                "mj_forward_queries": count,
                "model_initializations": 6,
                "physics_steps": 0,
                "world_model_calls": 0,
                "fresh_cases": 0,
                "sources_unchanged": True,
                "original_failed_result_preserved": True,
                "scope": "Saved-development endpoints and frozen-object static geometry; no motion or task qualification",
            },
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
