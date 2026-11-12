"""Read and independently compare the completed multitask producer and checker."""

import numpy as np

from s2a.infrastructure.publication import publish_result as publish
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_live_two_window_replay as captures
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro

ROOT = prefix.REPO
OUTPUT = ROOT / "outputs/reset_multitask_live_recovery_audit_20261111/result.json"
PRODUCER = {
    "path": str(ROOT / "outputs/reset_multitask_live_48_recovery_20261111/producer/result.json"),
    "sha256": "cb76a13aef9e4c424fab469346d9f551a6b90ee4bad9b8e1ca011ced949c7eb1",
}
CHECKER = {
    "path": str(ROOT / "outputs/reset_multitask_live_48_recovery_20261111/checker/result.json"),
    "sha256": "390622e41da9ff37a17ec3c25785c628f963b17129f7e9dfa0d33d0332c64782",
}
INCIDENT = {
    "path": str(ROOT / "outputs/reset_multitask_live_failure_audit_20261111/result.json"),
    "sha256": "e6aa8c57a9ad299b71f7b49d0330a65b525de5247dbc41e321d04417c07e7937",
}


def compare_periods(produced, replayed):
    """Compare actual commands, complete state, inner samples, scores and labels."""
    prefix.require(len(produced) == len(replayed), "Producer and checker must have the same physical period count")
    maxima = dict(command=0.0, state=0.0, tcp_error=0.0, inner_q=0.0, inner_qvel=0.0, inner_time=0.0, gripper=0.0)

    def compare(name, left, right):
        a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
        prefix.require(
            a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), "Comparable finite values required"
        )
        difference = float(np.max(np.abs(a - b))) if a.size else 0.0
        maxima[name] = max(maxima[name], difference)
        prefix.require(difference <= 1e-10, f"Producer and checker differ in {name}")

    for left, right in zip(produced, replayed, strict=True):
        prefix.require(
            left["step"] == right["step"]
            and left["endpoint_receipt"] == right["endpoint_receipt"]
            and len(left["actual_inner"]) == len(right["actual_inner"]) == 25
            and all(
                left[key] == right[key]
                for key in (
                    "pose_caps_satisfied",
                    "target_grasp",
                    "task_success",
                    "inner_outside_samples",
                    "inner_below_buffer_samples",
                )
            ),
            "The same endpoint, inner count and measured labels are required",
        )
        for name, key in (("command", "command"), ("state", "actual_state"), ("tcp_error", "tcp_error")):
            compare(name, left[key], right[key])
        for name, key in (("inner_q", "q"), ("inner_qvel", "qvel"), ("inner_time", "time")):
            compare(name, [item[key] for item in left["actual_inner"]], [item[key] for item in right["actual_inner"]])
        a, b = left["completed_feedback"]["score"], right["completed_feedback"]["score"]
        for key in ("opening", "opening_reference", "opening_error", "qpos", "qvel", "motor_target"):
            compare("gripper", a[key], b[key])
    return maxima


def audit_case(index, producer, checker):
    produced = prefix.read(producer["branch_ref"])
    replayed = prefix.read(checker["branch_ref"])
    model = prefix.read(producer["model_ref"])
    pr, cr = produced["runtime"], replayed["runtime"]
    p, c = pr["continuation"], cr["continuation"]
    first, first_check = p["first_window"]["periods"], c["first_window"]["periods"]
    live = [row for segment in p["stream"]["segments"] for row in segment["continuation"]["periods"]]
    live_check = [row for replay in c["replays"] for row in replay["periods"]]
    prefix.require(
        produced["case"]["case_index"] == replayed["case"]["case_index"] == model["case_index"] == index
        and produced["complete"] is produced["passed"] is replayed["complete"] is replayed["passed"] is True
        and replayed["candidate_modules_unloaded"] is True
        and model["complete"] is model["passed"] is True
        and c["reported_producer_stop"] == p["stream"]["stop"]
        and c["stop_reason_independently_verified"] is False,
        "The closed matching producer, isolated checker and reported stop are required",
    )
    initial_diff, live_diff = compare_periods(first, first_check), compare_periods(live, live_check)
    seen, reproduced = pr["streaming_captures"], cr["checker_stream_captures"]
    exchanges, responses = pr["streaming_model_exchanges"], model["responses"]
    prefix.require(
        len(seen) == len(reproduced) == len(exchanges) == len(responses) == model["responses_sent"]
        and model["model_requests_attempted"] == model["model_requests_completed"] == len(responses)
        and model["denoise_calls_attempted"] == model["denoise_calls_completed"] == 35 * len(responses)
        and len(model["denoise_evidence"]) == 35 * len(responses)
        and all(row["completed"] is True for row in model["denoise_evidence"]),
        "Every accepted live reference must bind a completed actual model request",
    )
    for capture, checked, exchange, row in zip(seen, reproduced, exchanges, responses, strict=True):
        captures.compare_capture(capture, checked)
        request, envelope = prefix.read(exchange["request_ref"]), prefix.read(exchange["response_ref"])
        actual = repro.response_arrays(prefix.read(row["actual_response_ref"]))
        prefix.require(
            capture["frame_index"] == exchange["step"] == request["step"] == row["step"]
            and row["reference_sent"] is True
            and request["image"]["sha256"] == capture["canvas_png"]["sha256"] == envelope["image_sha256"]
            and all(
                exchange[key] is True
                for key in ("request_sent", "response_received", "response_accepted", "policy_ready")
            ),
            "The actual response must match the independently reproduced current image",
        )
        np.testing.assert_array_equal(actual["action"], envelope["action"])
        np.testing.assert_array_equal(actual["normalized_action"], envelope["normalized_action"])
    pc, cc = producer["counts"], checker["counts"]
    prefix.require(
        pc["control_steps"] == cc["control_steps"] == len(first) + len(live)
        and pc["native_steps"] == cc["native_steps"] == 25 * pc["control_steps"]
        and pc["model_requests"] == len(responses)
        and cc["model_requests"] == 0,
        "Physical and model counts must agree across independently owned phases",
    )
    return dict(
        case_index=index,
        producer_ref=producer["branch_ref"],
        checker_ref=checker["branch_ref"],
        model_ref=producer["model_ref"],
        physical_periods=pc["control_steps"],
        inner_steps=pc["native_steps"],
        model_requests=pc["model_requests"],
        model_denoises=model["denoise_calls_completed"],
        model_elapsed_seconds=[row["model_elapsed_seconds"] for row in responses],
        initial_window_max_absolute_difference=initial_diff,
        live_windows_max_absolute_difference=live_diff,
        independently_reproduced_images=len(seen),
        pose_caps_satisfied=pc["pose_caps_satisfied"],
        position_rmse_m=pc["position_rmse_m"],
        rotation_rmse_rad=pc["rotation_rmse_rad"],
        minimum_inner_margin_rad=pc["minimum_inner_margin_rad"],
        target_grasp_periods=pc["target_grasp_periods"],
        task_success_periods=pc["task_success_periods"],
        stop_reason=None if pc["stop"] is None else pc["stop"]["reason"],
        stop_step=None if pc["stop"] is None else pc["stop"].get("at_step"),
        stop_reason_independently_verified=False,
    )


def main():
    prefix.environment()
    producer, checker, incident = prefix.read(PRODUCER), prefix.read(CHECKER), prefix.read(INCIDENT)
    for phase in (producer, checker):
        prefix.require(
            phase["complete"]
            is phase["passed"]
            is phase["owners_closed"]
            is phase["descendants_reaped"]
            is phase["gpu_released"]
            is phase["sources_unchanged"]
            is True
            and phase["automatic_retry"] is False
            and phase["registered_cases"] == [0, 4, 8],
            "All case owners and descendants must be closed, with unchanged sources",
        )
    cases = [audit_case(index, producer["cases"][str(index)], checker["cases"][str(index)]) for index in (0, 4, 8)]
    periods = sum(case["physical_periods"] for case in cases)
    requests = sum(case["model_requests"] for case in cases)
    denoises = sum(case["model_denoises"] for case in cases)
    prefix.require(
        periods == 114
        and requests == 12
        and denoises == 420
        and producer["totals"]["control_steps"] == checker["totals"]["control_steps"] == periods
        and producer["totals"]["native_steps"] == checker["totals"]["native_steps"] == periods * 25,
        "All actual period, model and denoise budgets must be accounted for",
    )
    result = dict(
        complete=True,
        passed=True,
        control_qualified=False,
        producer_ref=PRODUCER,
        checker_ref=CHECKER,
        prior_failure_audit_ref=INCIDENT,
        cases=cases,
        producer_periods=periods,
        checker_periods=periods,
        producer_inner_steps=periods * 25,
        checker_inner_steps=periods * 25,
        producer_models=9,
        checker_models=9,
        producer_contexts=9,
        checker_contexts=9,
        model_initializations=3,
        model_requests=requests,
        denoise_calls=denoises,
        cumulative_periods=incident["actual_physical_periods"] + 2 * periods,
        cumulative_inner_steps=incident["actual_inner_steps"] + 2 * periods * 25,
        cumulative_sim_models=incident["actual_sim_models"] + 18,
        cumulative_contexts=incident["actual_osmesa_contexts"] + 18,
        cumulative_model_initializations=incident["model_initializations"] + 3,
        cumulative_requests_attempted=incident["model_requests_attempted"] + requests,
        cumulative_requests_completed=incident["model_requests_completed"] + requests,
        cumulative_denoises=incident["denoise_calls"] + denoises,
        audit_model_requests=0,
        audit_physics_steps=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    print(publish(OUTPUT, result), flush=True)


if __name__ == "__main__":
    main()
