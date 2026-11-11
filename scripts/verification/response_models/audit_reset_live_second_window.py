"""Audit a real current-image WAM response and its independent reset replay."""

from pathlib import Path

import numpy as np

from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_live_two_window_replay as replay
from scripts.simulation.runtime import run_reset_live_checker_repair as repair
from scripts.verification.response_models.audit_reset_start_feedback import compare_replay, metrics
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro
from scripts.wam.tracking.preview import ordered_feedback_wam_inference_20261111 as engine
from scripts.wam.tracking.preview.live_stream_requests import _check_response_identity

ROOT = prefix.REPO
CHECKER = {
    "path": str(ROOT / "outputs/reset_live_checker_repair_20261111/result.json"),
    "sha256": "2efb8543c58de08a038dae31f5f075d84d712cc43f502832480bb51b801208b5",
}
QA = ROOT / "outputs/qa/reset_live_second_audit_v3_20261111/result.json"
OUTPUT = ROOT / "outputs/reset_live_second_audit_20261111/result.json"


def compare_window(producer, checker, *, start_step):
    """Validate absolute time indices before using the eight-period comparator."""
    prefix.require(
        type(start_step) is int
        and start_step >= 0
        and [row.get("step") for row in producer] == list(range(start_step, start_step + 8))
        and [row.get("step") for row in checker] == list(range(start_step, start_step + 8)),
        "Both windows must have the registered absolute step indices",
    )
    return compare_replay(
        [dict(row, step=index) for index, row in enumerate(producer)],
        [dict(row, step=index) for index, row in enumerate(checker)],
    )


def validate_model_ledger(model):
    prefix.require(
        model["complete"] is model["passed"] is True
        and model["gpu_uuid"] == repair.original_inputs.GPU
        and model["model_initialization_attempts"] == model["model_initializations"] == 1
        and model["model_requests_attempted"] == model["model_requests_completed"] == model["responses_sent"] == 1
        and model["denoise_calls_attempted"] == model["denoise_calls_completed"] == 35
        and len(model["denoise_evidence"]) == 35
        and all(item.get("completed") is True for item in model["denoise_evidence"])
        and model["physics_steps"] == model["fresh_cases"] == 0
        and model["automatic_retry"] is False,
        "One completed bound WAM request with 35 actual denoise calls is required",
    )
    engine.validate_mask_evidence(model["mask_evidence"], 1)


def validate_imported_sources(imported, sources_map):
    """Verify file modules and retain file-less Python namespace packages."""
    files = namespaces = 0
    for item in imported.values():
        if item == {"file": None, "namespace_package": True}:
            namespaces += 1
            continue
        prefix.require(
            type(item) is dict
            and type(item.get("file")) is str
            and type(item.get("sha256")) is str
            and len(item["sha256"]) == 64
            and sources_map.get(item["file"]) == item["sha256"]
            and repro.sha256(item["file"]) == item["sha256"],
            "Every imported file module must match the source manifest",
        )
        files += 1
    return files, namespaces


def phase_workers(owner, *, pair):
    lifecycle = prefix.read(owner["workers_ref"])
    workers = lifecycle["workers"]
    if pair:
        prefix.require(
            lifecycle["passed"] is True and [item["name"] for item in workers] == ["simulation", "model"],
            "The owned model and simulation pair is required",
        )
    else:
        prefix.require(
            len(workers) == 1 and workers[0]["key"] == "checker", "The independent checker worker is required"
        )
    prefix.require(
        all(
            item["returncode"] == 0 and item["actual_wait"] is item["reaped"] is True and item["signal_sent"] is False
            for item in workers
        ),
        "All new workers must actually exit and be reaped",
    )
    for item in workers:
        path = Path(item["log"])
        prefix.reference(path if path.is_absolute() else ROOT / path)


def execute():
    prefix.environment()
    before = snapshot(ROOT)["sources"]
    sources = [
        prefix.reference(p)
        for p in (
            Path(__file__).resolve(),
            ROOT / "tests/research/native_runtime/test_reset_live_second_audit.py",
            repair.PLAN,
        )
    ]
    qa = prefix.read(prefix.reference(QA))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and not qa["collection"]["collection_errors"]
        and prefix.read(qa["source_before_ref"])["sources"] == before,
        "Source-matched audit regression required",
    )
    inputs = repair.load_inputs()
    producer, model, branch = inputs["producer"], inputs["model"], inputs["branch"]
    checker = prefix.read(CHECKER)
    prefix.require(
        checker["complete"]
        is checker["passed"]
        is checker["sources_unchanged"]
        is checker["original_failure_preserved"]
        is True
        and checker["control_qualified"] is False
        and checker["world_model_calls"] == checker["fresh_cases"] == 0
        and checker["counts"]["control_steps"] == 16
        and checker["counts"]["native_steps"] == 400,
        "The repaired independent checker must be closed",
    )
    checked_branch = prefix.read(checker["branch_ref"])
    prefix.require(
        checked_branch["complete"] is checked_branch["passed"] is checked_branch["candidate_modules_unloaded"] is True
        and checked_branch["case"] == branch["case"] == inputs["original"]["case"]["case"],
        "The independently isolated same-case replay is required",
    )
    phase_workers(producer, pair=True)
    phase_workers(checker, pair=False)
    validate_model_ledger(model)
    prefix.require(
        producer["gpu_before"]["memory_mib"] == producer["gpu_after"]["memory_mib"] == 0
        and producer["gpu_before"]["compute_apps"] == producer["gpu_after"]["compute_apps"] == []
        and producer["gpu_released"] is True,
        "The selected GPU must be idle before and released after",
    )
    preflight = prefix.read(prefix.reference(repair.original_inputs.OUTPUT / "preflight.json"))
    prefix.require(
        preflight["complete"] is preflight["passed"] is True
        and preflight["source_files_verified"] >= 100
        and preflight["weight_files_verified"] == 163
        and preflight["gpu_uuid"] == model["gpu_uuid"],
        "The frozen source and checkpoint preflight is required",
    )
    imported = model["imported_cosmos_sources"]
    sources_map = inputs["original"]["model_source_hashes"]
    file_modules, namespace_modules = validate_imported_sources(imported, sources_map)
    prefix.require(file_modules >= 100, "The imported model source roster is incomplete")
    producer_runtime, checker_runtime = branch["runtime"], checked_branch["runtime"]
    first = producer_runtime["continuation"]["first_window"]["periods"]
    live = producer_runtime["continuation"]["stream"]["segments"][0]["continuation"]["periods"]
    check_first = checker_runtime["continuation"]["first_window"]["periods"]
    check_live = checker_runtime["continuation"]["replay"]["periods"]
    prefix.require(
        len(first) == len(live) == len(check_first) == len(check_live) == 8
        and producer_runtime["continuation"]["stream"]["complete"] is True
        and producer_runtime["continuation"]["stream"]["stop"] is None,
        "Both actual eight-period windows must be complete",
    )
    first_diff = compare_window(first, check_first, start_step=0)
    live_diff = compare_window(live, check_live, start_step=8)
    old_diff = compare_window(inputs["original"]["baseline_periods"], first, start_step=0)
    capture = producer_runtime["streaming_captures"][0]
    checker_capture = checker_runtime["checker_stream_capture"]
    replay.compare_capture(capture, checker_capture)
    prefix.require(
        producer_runtime["stream_canvas_scopes"] == [dict(before=1, requested=2, inside=2, after=1)]
        and checker_runtime["checker_stream_canvas_scope"] == dict(before=1, requested=2, inside=2, after=1),
        "Both current-image render scopes must restore threads",
    )
    exchange = producer_runtime["streaming_model_exchanges"][0]
    model_row = model["responses"][0]
    request = prefix.read(model_row["model_request_ref"])
    response = prefix.read(model_row["actual_response_ref"])
    copy_image = Path(request["vision_path"])
    copy_ref = prefix.reference(copy_image)
    image = dict(step=8, path=str(copy_image.relative_to(ROOT)), sha256=copy_ref["sha256"])
    engine.validate_request(request, image)
    _check_response_identity(
        response,
        model_request=request,
        task=inputs["original"]["config"]["task"],
        checkpoint=inputs["original"]["prepared"]["model_config"]["checkpoint"],
    )
    prefix.require(
        copy_ref["sha256"] == capture["canvas_png"]["sha256"]
        and model_row["reference_sent"] is True
        and model_row["step"] == 8,
        "The model must consume this exact actual current image",
    )
    registration = repair.original_inputs.model_registration(
        inputs["original"], repair.original_inputs.OUTPUT / "producer/simulation/runtime"
    )
    poses, openings = replay.decode_response(
        registration, capture, exchange, response, inputs["original"]["case"]["tool"]
    )
    prefix.require(
        poses.shape == (40, 4, 4) and openings.shape == (40,), "The actual response must decode to one H40 reference"
    )
    for offset, period in enumerate(live):
        np.testing.assert_allclose(
            period["completed_feedback"]["score"]["opening_reference"], openings[offset], atol=1e-10, rtol=0
        )
    rows = [
        dict(
            error=p["tcp_error"],
            opening_error=p["completed_feedback"]["score"]["opening_error"],
            task_success=p["task_success"],
            target_grasp=p["target_grasp"],
        )
        for p in live
    ]
    prefix.require(
        snapshot(ROOT)["sources"] == before and sources == [prefix.reference(ref["path"]) for ref in sources],
        "Audit sources must remain unchanged",
    )
    result = dict(
        complete=True,
        passed=True,
        control_qualified=False,
        original_failure_preserved=True,
        producer_ref=repair.PRODUCER,
        failed_checker_ref=repair.FAILURE,
        checker_ref=CHECKER,
        qa_ref=prefix.reference(QA),
        source_refs=sources,
        sources_unchanged=True,
        original_first_vs_producer=old_diff,
        first_vs_checker=first_diff,
        live_vs_checker=live_diff,
        live_metrics=metrics(rows),
        live_minimum_inner_margin_rad=min(p["minimum_inner_margin_rad"] for p in live),
        current_image_ref=capture["canvas_png"],
        model_request_ref=model_row["model_request_ref"],
        model_response_ref=model_row["actual_response_ref"],
        model_elapsed_seconds=model_row["model_elapsed_seconds"],
        imported_cosmos_modules=len(imported),
        imported_file_modules=file_modules,
        imported_namespace_modules=namespace_modules,
        producer_control_steps=16,
        failed_checker_control_steps=8,
        repaired_checker_control_steps=16,
        cumulative_native_steps=1000,
        cumulative_models=9,
        cumulative_contexts=9,
        model_initializations=1,
        model_requests=1,
        denoise_calls=35,
        audit_models=0,
        audit_physics_steps=0,
        fresh_cases=0,
    )
    print(publish(OUTPUT, result), flush=True)


if __name__ == "__main__":
    execute()
