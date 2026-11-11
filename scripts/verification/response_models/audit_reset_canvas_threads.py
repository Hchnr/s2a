"""Audit original and live reset canvases under two explicit CPU thread counts."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from s2a.backends.libero.simulation import canvas
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.native_runtime.canvas_rounding_audit import bilinear_half_integer_mask, compare_canvases
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import prepare_reset_development as preparation
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

ROOT = prefix.REPO
FAILURE = {
    "path": str(ROOT / "outputs/reset_native_admission_20261111/result.json"),
    "sha256": "96e3732026a52470348b51a4aeabe001e01adc9261a8ab728f18564c945310a0",
}
PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/reset_canvas_thread_repair_20261111.md"
QA = ROOT / "outputs/qa/reset_canvas_threads_audit_20261111/result.json"
OUTPUT = ROOT / "outputs/reset_canvas_threads_audit_20261111/result.json"


def pixel_sha(pixels):
    return hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest()


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            PLAN,
            Path(__file__).resolve(),
            ROOT / "src/s2a/research/native_runtime/canvas_rounding_audit.py",
            ROOT / "src/s2a/backends/libero/simulation.py",
            ROOT / "tests/research/native_runtime/test_canvas_rounding_audit.py",
        )
    ]


def execute():
    prefix.environment()
    before, sources = snapshot(ROOT)["sources"], source_refs()
    qa = prefix.read(prefix.reference(QA))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == 0
        and not qa["collection"]["collection_errors"]
        and prefix.read(qa["source_before_ref"])["sources"] == before,
        "A source-matched CPU regression is required",
    )
    failure = prefix.read(FAILURE)
    lifecycle = prefix.read(failure["workers_ref"])
    prefix.require(
        failure["complete"] is failure["passed"] is False
        and len(failure["partial_cases"]) == len(lifecycle["workers"]) == 12
        and lifecycle["launch_error"] is None
        and all(
            row["actual_wait"] is row["reaped"] is True and row["returncode"] == 1 and row["signal_sent"] is False
            for row in lifecycle["workers"]
        ),
        "The original failed owner and all twelve actual worker exits are required",
    )
    inputs = preparation.load_inputs(verify_assets=True)
    prefix.require(failure["inputs"] == preparation.input_receipt(inputs), "Same reviewed source cases required")
    review = prefix.read(inputs["study_inputs"]["refs"]["review"])
    prefix.require(torch.get_num_threads() == 1, "The auditing process must start with one Torch thread")
    rows = []
    for index, (case, partial) in enumerate(zip(inputs["cases"], failure["partial_cases"], strict=True)):
        prefix.require(partial["case_index"] == index and case["case"]["case_index"] == index, "Case order required")
        worker = prefix.read(partial["ref"])
        runtime = worker["runtime"]
        prefix.require(
            worker["complete"] is worker["passed"] is False
            and worker["error"]["message"] == "Live reset image differs from the original WAM request"
            and runtime["reset_endpoint_comparison"]["state_bytes_equal"] is True
            and runtime["reset_endpoint_comparison"]["max_field_difference"] == 0
            and runtime["reset_image_match"] is False
            and runtime["cleanup_complete"] is True
            and runtime["counters"]["model_construct"]["completed"] == 3
            and runtime["counters"]["context"]["completed"] == 3
            and runtime["counters"]["mj_step"]["attempted"] == 0,
            "A zero-step image-mismatch failure after the exact recorded reset is required",
        )
        capture = runtime["reset_render"]
        with np.load(prefix.checked(capture["raw_cameras"]), allow_pickle=False) as archive:
            images = [archive["live_auxiliary_0"].copy(), archive["live_left_wrist"].copy()]
            for role, raw in zip(("agentview_image", "robot0_eye_in_hand_image"), images, strict=True):
                prefix.require(np.array_equal(archive[role][::-1, ::-1], raw), "Original camera flip must match")
        manifests = [path for path in case["original_source_sha256"] if path.endswith("/manifest.json")]
        prefix.require(len(manifests) == 1, "One original manifest required")
        recorded_path = Path(manifests[0]).parent / "input_0000.png"
        original_ref = prefix.reference(recorded_path)
        prefix.require(
            review["source_and_artifact_sha256"].get(str(recorded_path)) == original_ref["sha256"],
            "The original input PNG must be independently source-pinned",
        )
        with Image.open(prefix.checked(original_ref)) as image:
            recorded = np.asarray(image.convert("RGB")).copy()
        with Image.open(prefix.checked(capture["canvas_png"])) as image:
            saved = np.asarray(image.convert("RGB")).copy()
        prefix.require(
            pixel_sha(recorded) == case["chunk"]["image_sha256"] and pixel_sha(saved) == capture["canvas_rgb_sha256"],
            "Recorded RGB identities required",
        )
        try:
            torch.set_num_threads(1)
            one = canvas(images)
            torch.set_num_threads(2)
            two = canvas(images)
        finally:
            torch.set_num_threads(1)
        tie_mask = bilinear_half_integer_mask(images)
        comparison = compare_canvases(recorded, saved, one, two, tie_mask)
        prefix.require(
            comparison["one_matches_saved"] is comparison["two_matches_recorded"] is True,
            "Both complete saved and original pixel arrays must reproduce exactly",
        )
        rows.append(
            dict(
                case=case["case"],
                worker_ref=partial["ref"],
                original_png_ref=original_ref,
                raw_cameras_ref=capture["raw_cameras"],
                live_png_ref=capture["canvas_png"],
                recorded_rgb_sha256=pixel_sha(recorded),
                one_thread_rgb_sha256=pixel_sha(one),
                two_thread_rgb_sha256=pixel_sha(two),
                **comparison,
            )
        )
    prefix.require(
        torch.get_num_threads() == 1 and snapshot(ROOT)["sources"] == before and source_refs() == sources,
        "Torch threads and source files must be restored",
    )
    print(
        publish(
            OUTPUT,
            dict(
                complete=True,
                passed=True,
                control_qualified=False,
                failure_ref=FAILURE,
                qa_ref=prefix.reference(QA),
                source_refs=sources,
                sources_unchanged=True,
                cases=rows,
                native_models=0,
                forward_queries=0,
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
                torch_threads_restored=1,
                original_failure_not_reclassified=True,
            ),
        ),
        flush=True,
    )


if __name__ == "__main__":
    execute()
