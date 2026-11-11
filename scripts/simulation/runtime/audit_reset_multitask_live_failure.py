"""Audit the closed three-case live pilot without re-running models or physics."""

import hashlib
import subprocess

import numpy as np

from s2a.infrastructure.publication import publish_result as publish
from s2a.research.wam.training.native.native_training import state_sha256
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro

ROOT = prefix.REPO
OUTPUT = ROOT / "outputs/reset_multitask_live_failure_audit_20261111/result.json"
PRODUCER = {
    "path": str(ROOT / "outputs/reset_multitask_live_48_20261111/producer/result.json"),
    "sha256": "6485e226e9ff231dce9fcffdce5fa26cce5c7332f2c18fc380be539e24aab5f3",
}
WORKER_PATH = "scripts/simulation/runtime/reset_multitask_live_workers.py"
WORKER_SHA256 = "e88e17a1c03ecdbe61026fd6a45779bddb449ed7a54ac023d751eed1bd15ccfb"
SOURCE_COMMIT = "77299f2b764d8e2b4eb0bc9b90e01292655e157b"


def summarize_periods(periods, start):
    """Summarize saved producer observations without claiming independent replay."""
    prefix.require(
        len(periods) == 8 and [row["step"] for row in periods] == list(range(start, start + 8)),
        "Eight consecutive actual periods are required",
    )
    for row in periods:
        endpoint = row["endpoint_receipt"]
        prefix.require(
            len(row["actual_inner"]) == 25
            and endpoint["endpoint_step"] == row["step"] + 1
            and state_sha256(np.asarray(row["actual_state"], dtype=np.float64)) == endpoint["sim_state_sha256"],
            "Actual state and all 25 inner samples must bind each recorded endpoint",
        )
    errors = np.asarray([row["tcp_error"] for row in periods], dtype=float)
    prefix.require(errors.shape == (8, 6) and np.isfinite(errors).all(), "Finite actual TCP errors are required")
    return dict(
        periods=8,
        inner_steps=200,
        pose_caps_satisfied=sum(row["pose_caps_satisfied"] for row in periods),
        position_rmse_m=float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
        rotation_rmse_rad=float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
        minimum_inner_margin_rad=float(min(row["minimum_inner_margin_rad"] for row in periods)),
        target_grasp_periods=sum(row["target_grasp"] for row in periods),
        task_success_periods=sum(row["task_success"] for row in periods),
    )


def audit_case(index, owner):
    """Check both completed windows, the failed exchange and actual worker costs."""
    prefix.require(owner["complete"] is owner["passed"] is False, "The failed case must remain failed")
    sim_ref, model_ref = owner["available_branch_ref"], owner["available_model_ref"]
    sim, model = prefix.read(sim_ref), prefix.read(model_ref)
    lifecycle = prefix.read(owner["workers_ref"])
    runtime, stream = sim["runtime"], sim["runtime"]["streaming_partial"]
    counters = runtime["counters"]
    prefix.require(
        sim["complete"] is sim["passed"] is model["complete"] is model["passed"] is False
        and sim["case_index"] == model["case_index"] == index
        and sim["error"]["message"] == "Peer closed during a frame"
        and model["error"]["message"] == "025 unregistered forward or exhausted budget"
        and runtime["control_steps_attempted"] == runtime["control_steps_completed"] == 16
        and counters["mj_step"]["attempted"] == counters["mj_step"]["completed"] == 400
        and counters["model_construct"]["completed"] == counters["context"]["completed"] == 3
        and stream["requests_issued"] == 2
        and stream["responses_accepted"] == 1
        and stream["control_steps_attempted"]
        == stream["control_steps_completed"]
        == stream["feedback_periods_committed"]
        == 8
        and len(stream["segments"]) == 1
        and model["model_initialization_attempts"] == model["model_initializations"] == 1
        and model["local_requests_received"] == model["model_requests_attempted"] == 2
        and model["model_requests_completed"] == model["responses_sent"] == 1
        and model["denoise_calls_attempted"] == model["denoise_calls_completed"] == 35
        and len(model["denoise_evidence"]) == 35
        and all(call["completed"] is True for call in model["denoise_evidence"])
        and len(lifecycle["workers"]) == 2
        and all(item["actual_wait"] is item["reaped"] is True for item in lifecycle["workers"]),
        "The failed exchange and actual model/physical budget must be fully accounted for",
    )
    captures, exchanges = runtime["streaming_captures"], runtime["streaming_model_exchanges"]
    prefix.require(
        len(captures) == len(exchanges) == 2
        and [item["step"] for item in exchanges] == [8, 16]
        and all(item["request_sent"] is True for item in exchanges)
        and exchanges[0]["response_received"] is exchanges[0]["response_accepted"] is True
        and exchanges[1]["response_received"] is exchanges[1]["response_accepted"] is False,
        "The step8 response completed and the step16 response did not arrive",
    )
    for capture, exchange in zip(captures, exchanges, strict=True):
        request = prefix.read(exchange["request_ref"])
        image = capture["canvas_png"]
        prefix.checked(image)
        prefix.require(
            capture["frame_index"] == request["step"] == exchange["step"]
            and request["image"]["sha256"] == image["sha256"]
            and request["state_sha256"] == capture["render_before"]["sim_state_sha256"]
            and capture["render_before"] == capture["render_after"],
            "Each request must bind its recorded current image and unchanged state",
        )
    response = prefix.read(exchanges[0]["response_ref"])
    model_row = model["responses"][0]
    actual = prefix.read(model_row["actual_response_ref"])
    arrays = repro.response_arrays(actual)
    np.testing.assert_array_equal(response["action"], arrays["action"])
    np.testing.assert_array_equal(response["normalized_action"], arrays["normalized_action"])
    prefix.require(model_row["step"] == 8 and model_row["reference_sent"] is True, "Actual model response required")
    first = summarize_periods(runtime["reset_first_window"]["periods"], 0)
    live = summarize_periods(stream["segments"][0]["continuation"]["periods"], 8)
    return dict(
        case_index=index,
        simulation_ref=sim_ref,
        model_ref=model_ref,
        workers_ref=owner["workers_ref"],
        first_window=first,
        current_image_window=live,
        model_initializations=1,
        model_requests_attempted=2,
        model_requests_completed=1,
        denoise_calls=35,
        producer_failure=model["error"],
        worker_exits=[
            dict(name=row["name"], returncode=row["returncode"], signal_sent=row["signal_sent"], reaped=row["reaped"])
            for row in lifecycle["workers"]
        ],
        independently_replayed=False,
    )


def main():
    prefix.environment()
    producer = prefix.read(PRODUCER)
    prefix.require(
        producer["complete"] is producer["passed"] is False
        and producer["owners_closed"]
        is producer["descendants_reaped"]
        is producer["gpu_released"]
        is producer["sources_unchanged"]
        is True
        and producer["registered_cases"] == [0, 4, 8]
        and producer["completed_cases"] == 0
        and producer["automatic_retry"] is False,
        "All failed owners must be closed and all GPUs released",
    )
    saved = subprocess.check_output(["git", "show", f"{SOURCE_COMMIT}:{WORKER_PATH}"], cwd=ROOT)
    prefix.require(
        hashlib.sha256(saved).hexdigest() == WORKER_SHA256
        and prefix.reference(ROOT / WORKER_PATH)["sha256"] == WORKER_SHA256
        and b"BoundaryCounter(service.model, 35)" in saved,
        "The actual failing worker source must match its immutable merged commit",
    )
    cases = [audit_case(index, producer["cases"][str(index)]) for index in (0, 4, 8)]
    result = dict(
        complete=True,
        passed=True,
        control_qualified=False,
        producer_ref=PRODUCER,
        source_commit=SOURCE_COMMIT,
        worker_source=dict(path=WORKER_PATH, sha256=WORKER_SHA256),
        cases=cases,
        actual_physical_periods=48,
        actual_inner_steps=1200,
        actual_sim_models=9,
        actual_osmesa_contexts=9,
        model_initializations=3,
        model_requests_attempted=6,
        model_requests_completed=3,
        denoise_calls=105,
        independently_replayed=False,
        new_native_models=0,
        new_physics_steps=0,
        new_model_requests=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    print(publish(OUTPUT, result), flush=True)


if __name__ == "__main__":
    main()
