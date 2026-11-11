"""Recheck actual reset receipts and every saved current-image pixel."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from s2a.backends.libero.simulation import canvas
from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import prepare_reset_development as preparation
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

ROOT = prefix.REPO
RESULT = {
    "path": str(ROOT / "outputs/reset_native_admission_v2_20261111/result.json"),
    "sha256": "b6fc7fd1bd62485297e2b33ad9299a9c682cf4df0cd95da3ec58ac264539d411",
}
CANVAS_AUDIT = {
    "path": str(ROOT / "outputs/reset_canvas_threads_audit_20261111/result.json"),
    "sha256": "e433f00f1b22970f6720116fdf736e691de98073efb69907c0e0c72759581268",
}
QA = ROOT / "outputs/qa/reset_native_admission_v2_audit_20261111/result.json"
OUTPUT = ROOT / "outputs/reset_native_admission_v2_audit_20261111/result.json"


def validate_worker(worker, index):
    """Enforce actual zero-step reset, exact state and scoped-thread receipt."""
    runtime = worker["runtime"]
    counters, comparison, capture = runtime["counters"], runtime["reset_endpoint_comparison"], runtime["reset_render"]
    endpoint, before, after = capture["current_endpoint"], capture["render_before"], capture["render_after"]
    if not (
        worker["complete"] is worker["passed"] is True
        and worker["control_qualified"] is False
        and worker["case_index"] == index
        and worker["case"]["case_index"] == index
        and runtime["reset_observation_verified"] is runtime["reset_image_match"] is runtime["cleanup_complete"] is True
        and runtime["behavioural_replay_verified"] is False
        and runtime["previous_saved_command"] is None
        and runtime["prefix_control_steps"]
        == runtime["control_steps_attempted"]
        == runtime["control_steps_completed"]
        == 0
        and runtime["checked_endpoints"] == runtime["exact_state_endpoints"] == 1
        and comparison["state_bytes_equal"] is True
        and comparison["max_field_difference"] == 0
        and comparison["recorded_state_sha256"] == comparison["actual_state_sha256"] == endpoint["sim_state_sha256"]
        and endpoint["endpoint_step"] == 0
        and before == after
        and before["sim_state_sha256"] == endpoint["sim_state_sha256"]
        and before["sim_time"] == endpoint["sim_time"]
        and runtime["canvas_thread_scope"] == {"before": 1, "requested": 2, "inside": 2, "after": 1}
        and counters["model_construct"]["attempted"] == counters["model_construct"]["completed"] == 3
        and counters["context"]["attempted"] == counters["context"]["completed"] == 3
        and counters["mj_step"]["attempted"] == counters["mj_step"]["completed"] == 0
        and counters["mj_step2"]["attempted"] == counters["mj_step2"]["completed"] == 0
    ):
        raise ValueError("The complete reset receipt, state binding and zero-step budget are required")
    return capture


def pixel_sha(pixels):
    return hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest()


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
        "Source-matched CPU audit regression required",
    )
    result, prior = prefix.read(RESULT), prefix.read(CANVAS_AUDIT)
    prefix.require(
        result["complete"] is result["passed"] is result["sources_unchanged"] is True
        and result["control_qualified"] is False
        and result["totals"] == dict(models=36, contexts=36, control_steps=0, physics_steps=0)
        and result["image_matches"] == result["exact_state_matches"] == len(result["case_refs"]) == 12
        and prior["complete"] is prior["passed"] is True
        and len(prior["cases"]) == 12,
        "A complete v2 reset and the separately closed original canvas audit are required",
    )
    for ref in result["source_refs"]:
        prefix.checked(ref)
    workers = prefix.read(result["workers_ref"])
    prefix.require(
        workers["launch_error"] is None
        and len(workers["workers"]) == 12
        and all(
            item["key"] == f"case_{i:02d}"
            and item["actual_wait"] is item["reaped"] is True
            and item["returncode"] == 0
            and item["signal_sent"] is False
            for i, item in enumerate(workers["workers"])
        ),
        "All twelve worker processes must actually exit and be reaped",
    )
    inputs = preparation.load_inputs(verify_assets=True)
    prefix.require(
        result["inputs"] == preparation.input_receipt(inputs), "The reviewed original source cases must match"
    )
    prefix.require(torch.get_num_threads() == 1, "The verifier starts with one Torch thread")
    checked = []
    try:
        torch.set_num_threads(2)
        for i, (case_ref, historical, case) in enumerate(
            zip(result["case_refs"], prior["cases"], inputs["cases"], strict=True)
        ):
            worker = prefix.read(case_ref)
            capture = validate_worker(worker, i)
            prefix.require(worker["case"] == case["case"] == historical["case"], "Case identity must remain fixed")
            with Image.open(prefix.checked(historical["original_png_ref"])) as image:
                recorded = np.asarray(image.convert("RGB")).copy()
            with Image.open(prefix.checked(capture["canvas_png"])) as image:
                actual = np.asarray(image.convert("RGB")).copy()
            with np.load(prefix.checked(capture["raw_cameras"]), allow_pickle=False) as archive:
                images = [archive["live_auxiliary_0"].copy(), archive["live_left_wrist"].copy()]
                for role, pixels in zip(("agentview_image", "robot0_eye_in_hand_image"), images, strict=True):
                    prefix.require(
                        np.array_equal(archive[role][::-1, ::-1], pixels), "Original camera roles and flips required"
                    )
            recomputed = canvas(images)
            expected = case["chunk"]["image_sha256"]
            prefix.require(
                recorded.shape == actual.shape == recomputed.shape == (960, 640, 3)
                and np.array_equal(actual, recorded)
                and np.array_equal(recomputed, actual)
                and pixel_sha(actual) == expected == capture["canvas_rgb_sha256"] == historical["recorded_rgb_sha256"],
                "All current and original RGB pixels must match",
            )
            checked.append(
                dict(
                    case=case["case"],
                    worker_ref=case_ref,
                    current_png_ref=capture["canvas_png"],
                    raw_cameras_ref=capture["raw_cameras"],
                    original_png_ref=historical["original_png_ref"],
                    image_rgb_sha256=expected,
                    state_sha256=capture["current_endpoint"]["sim_state_sha256"],
                )
            )
    finally:
        torch.set_num_threads(1)
    prefix.require(
        torch.get_num_threads() == 1 and snapshot(ROOT)["sources"] == before,
        "Sources and Torch threads must be restored",
    )
    sources = [
        prefix.reference(Path(__file__).resolve()),
        prefix.reference(ROOT / "tests/research/native_runtime/test_reset_native_admission_v2_audit.py"),
    ]
    print(
        publish(
            OUTPUT,
            dict(
                complete=True,
                passed=True,
                control_qualified=False,
                result_ref=RESULT,
                original_canvas_audit_ref=CANVAS_AUDIT,
                qa_ref=prefix.reference(QA),
                source_refs=sources,
                cases=checked,
                actual_workers_waited_and_reaped=12,
                native_models=0,
                forward_queries=0,
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
                torch_threads_restored=1,
                scope="saved reset receipts and current-image pixels; no new physical replay or control qualification",
            ),
        ),
        flush=True,
    )


if __name__ == "__main__":
    execute()
