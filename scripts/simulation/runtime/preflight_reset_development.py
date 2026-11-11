"""Check all registered historical first-query reset identities without simulation."""

import json
from pathlib import Path

import numpy as np

from s2a.infrastructure.publication import publish_result as publish
from s2a.research.native_runtime.reset_endpoint import saved_reset_endpoint, verify_reset_endpoint
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.response_models import multicase_slide_table_collision as study

ROOT = prefix.REPO
OUTPUT = ROOT / "outputs/reset_start_source_preflight_20261111/result.json"
QA = ROOT / "outputs/qa/reset_start_preflight_20261111/result.json"
PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/reset_start_development_20261111.md"


def execute():
    prefix.environment()
    before = snapshot(ROOT)["sources"]
    qa = prefix.read(prefix.reference(QA))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == 0
        and not qa["collection"]["collection_errors"]
        and prefix.read(qa["source_before_ref"])["sources"] == before,
        "Source-matched 32-worker reset qualification required",
    )
    inputs = study.load_inputs(verify_assets=True)
    entries = []
    for index in range(12):
        case = study.case_inputs(index, inputs)
        worker_paths = [path for path in case["source_sha256"] if path.endswith("/worker_result.json")]
        prefix.require(len(worker_paths) == 1, "One source-pinned worker per case")
        worker = json.loads(Path(worker_paths[0]).read_bytes())
        receipt_ref = worker["endpoint_receipts_ref"]
        prefix.require(
            inputs["review"]["source_and_artifact_sha256"].get(receipt_ref["path"]) == receipt_ref["sha256"],
            "The original independent review must pin the endpoint receipts",
        )
        receipt_path = prefix.checked(receipt_ref)
        with receipt_path.open() as stream:
            receipt = json.loads(stream.readline())
        with np.load(case["files"]["arrays"], allow_pickle=False) as archive:
            saved = saved_reset_endpoint(archive)
        manifest = json.loads(case["files"]["manifest"].read_bytes())
        prefix.require(
            receipt["sim_state_sha256"] == manifest["record"]["initial_state_sha256"],
            "Manifest and actual original reset receipt must agree",
        )
        # This is a source self-consistency check, not a new physical reset.
        verified = verify_reset_endpoint(
            saved, receipt, saved, receipt, case["chunks"][0], case["controller"], case["tool"]
        )
        entries.append(
            dict(
                case=case["case"],
                original_receipt_ref=receipt_ref,
                original_source_sha256=case["source_sha256"],
                reset=verified,
            )
        )
        for path, digest in case["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
    prefix.require(
        len(entries) == 12
        and study.historical.asset_paths(inputs["assets"]) == inputs["paths"]
        and snapshot(ROOT)["sources"] == before,
        "All cases, snapshots and sources must remain unchanged",
    )
    refs = [
        prefix.reference(Path(__file__).resolve()),
        prefix.reference(PLAN),
        prefix.reference(ROOT / "src/s2a/research/native_runtime/reset_endpoint.py"),
        prefix.reference(ROOT / "src/s2a/research/joint_control/preview/reset_history.py"),
    ]
    print(
        publish(
            OUTPUT,
            dict(
                complete=True,
                passed=True,
                control_qualified=False,
                scope="recorded source self-consistency only; no actual reset, physical step, or control qualification",
                qa_ref=prefix.reference(QA),
                source_refs=refs,
                source_unchanged=True,
                inputs=study.input_receipt(inputs),
                cases=entries,
                native_models=0,
                forward_queries=0,
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
            ),
        ),
        flush=True,
    )


if __name__ == "__main__":
    execute()
