"""Audit generated versus actually consumed gripper intent without model or physics work."""

import numpy as np

from s2a.infrastructure.publication import publish_result as publish
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro

ROOT = prefix.REPO
OUTPUT = ROOT / "outputs/reset_multitask_gripper_intent_audit_20261112/result.json"
SOURCE = {
    "path": str(ROOT / "outputs/reset_multitask_live_recovery_audit_20261111/result.json"),
    "sha256": "218582f9851373f02a3dbd22d392cb48783507dc241d4835fc6783fc8bf4e2a3",
}


def window_summary(openings, consumed, anchor):
    """Separate generated H40, executed prefix and never-executed suffix."""
    values = np.asarray(openings, dtype=float)
    prefix.require(
        values.shape == (40,)
        and np.isfinite(values).all()
        and np.all((values >= 0) & (values <= 1))
        and type(consumed) is int
        and 0 <= consumed <= 8
        and type(anchor) in (float, int)
        and np.isfinite(anchor)
        and 0 <= anchor <= 1,
        "Finite H40 openings, current anchor and at most eight consumed endpoints are required",
    )

    def part(start, end):
        subset = values[start:end]
        close = np.flatnonzero(subset <= 0.2)
        decrease = np.flatnonzero(subset <= anchor - 0.2)
        return dict(
            endpoints=len(subset),
            minimum=None if len(subset) == 0 else float(subset.min()),
            absolute_close_count=len(close),
            first_absolute_close_index=None if not len(close) else int(start + close[0]),
            anchor_drop_count=len(decrease),
            first_anchor_drop_index=None if not len(decrease) else int(start + decrease[0]),
        )

    return dict(all=part(0, 40), consumed=part(0, consumed), unconsumed=part(consumed, 40))


def audit_case(case):
    index = case["case_index"]
    producer, checker, model = (prefix.read(case[key]) for key in ("producer_ref", "checker_ref", "model_ref"))
    prefix.require(
        producer["complete"] is producer["passed"] is checker["complete"] is checker["passed"] is True
        and producer["case"]["case_index"] == checker["case"]["case_index"] == model["case_index"] == index
        and checker["candidate_modules_unloaded"] is True,
        "The audited physical producer, isolated replay and actual model must match",
    )
    runtime = producer["runtime"]
    stream = runtime["continuation"]["stream"]
    captures, exchanges, segments, responses = (
        runtime["streaming_captures"],
        runtime["streaming_model_exchanges"],
        stream["segments"],
        model["responses"],
    )
    prefix.require(
        len(captures) == len(exchanges) == len(segments) == len(responses) == case["model_requests"]
        and model["denoise_calls_completed"] == 35 * len(responses),
        "Every recorded model response and actual segment must be accounted for",
    )
    windows, actual_openings = [], []
    for capture, exchange, segment, response in zip(captures, exchanges, segments, responses, strict=True):
        request = prefix.read(exchange["request_ref"])
        envelope = prefix.read(exchange["response_ref"])
        arrays = repro.response_arrays(prefix.read(response["actual_response_ref"]))
        np.testing.assert_array_equal(arrays["action"], envelope["action"])
        np.testing.assert_array_equal(arrays["normalized_action"], envelope["normalized_action"])
        prefix.require(
            capture["frame_index"] == exchange["step"] == segment["start_step"] == request["step"] == response["step"]
            and request["image"]["sha256"] == envelope["image_sha256"] == capture["canvas_png"]["sha256"]
            and response["reference_sent"] is True,
            "Each response must belong to its own observed current image",
        )
        periods = segment["continuation"]["periods"]
        openings = np.clip(arrays["action"][:, 9], 0, 1)
        refs = [row["completed_feedback"]["score"]["opening_reference"] for row in periods]
        np.testing.assert_allclose(openings[: len(periods)], refs, atol=1e-10, rtol=0)
        actual_openings.extend(row["completed_feedback"]["score"]["opening"] for row in periods)
        windows.append(
            dict(
                start_step=exchange["step"],
                anchor_opening=capture["current_opening"],
                response_ref=response["actual_response_ref"],
                observation_sha256=request["image"]["sha256"],
                opening=window_summary(openings, len(periods), capture["current_opening"]),
            )
        )
    initial = runtime["continuation"]["first_window"]["periods"]
    all_periods = initial + [row for segment in segments for row in segment["continuation"]["periods"]]
    prefix.require(
        len(all_periods) == case["physical_periods"]
        and sum(row["target_grasp"] for row in all_periods) == case["target_grasp_periods"]
        and sum(row["task_success"] for row in all_periods) == case["task_success_periods"],
        "Actual physical period and task labels must match the independent audit",
    )
    return dict(
        case_index=index,
        physical_periods=len(all_periods),
        live_consumed_periods=len(actual_openings),
        minimum_actual_live_opening=None if not actual_openings else float(min(actual_openings)),
        target_grasp_periods=case["target_grasp_periods"],
        task_success_periods=case["task_success_periods"],
        windows=windows,
    )


def main():
    prefix.environment()
    source = prefix.read(SOURCE)
    prefix.require(
        source["complete"] is source["passed"] is True
        and source["control_qualified"] is False
        and source["producer_periods"] == source["checker_periods"] == 114
        and source["model_requests"] == 12
        and [case["case_index"] for case in source["cases"]] == [0, 4, 8],
        "The complete independently replayed pilot is required",
    )
    cases = [audit_case(case) for case in source["cases"]]
    windows = [window for case in cases for window in case["windows"]]
    totals = {
        part: {
            key: sum(window["opening"][part][key] for window in windows)
            for key in ("endpoints", "absolute_close_count", "anchor_drop_count")
        }
        for part in ("all", "consumed", "unconsumed")
    }
    prefix.require(
        len(windows) == 12
        and totals["all"]["endpoints"] == 480
        and totals["consumed"]["endpoints"] == 90
        and totals["unconsumed"]["endpoints"] == 390,
        "Generated, actually consumed and unused response endpoints must retain their separate denominators",
    )
    result = dict(
        complete=True,
        passed=True,
        control_qualified=False,
        source_ref=SOURCE,
        thresholds=dict(absolute_opening_at_most=0.2, drop_from_measured_anchor_at_least=0.2),
        retrospective_not_blind=True,
        cases=cases,
        totals=totals,
        new_models=0,
        new_model_requests=0,
        new_physics_steps=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    print(publish(OUTPUT, result), flush=True)


if __name__ == "__main__":
    main()
