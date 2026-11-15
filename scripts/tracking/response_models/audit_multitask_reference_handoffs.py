"""Audit consecutive H40 predictions at their common future endpoint."""

import collections
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import validate_transform
from s2a.infrastructure.publication import publish_result
from s2a.references.action_reference import decode_action_targets
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_multitask_adaptive_admission as admission
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro
from scripts.wam.tracking.references.audit_consumed_gripper_intent import window_summary

ROOT = prefix.REPO
SOURCE = {
    "path": str(ROOT / "outputs/reset_multitask_adaptive_complete_audit_20261113/result.json"),
    "sha256": "7fe0f7d17ddea1913ffab2f5a4243c5b8e4b6a5def9633269ebec95efc82f8fe",
}
PLAN = ROOT / "plans/v0.0.1/tracking/references/multitask_reference_handoffs_20261113.md"
OUTPUT = ROOT / "outputs/multitask_reference_handoffs_20261113/result.json"
INDICES = (0, 2, 4, 6, 8, 10)


def compare_handoff(old, new):
    """Compare old[n] with new[0]: both target old_start+n+1."""
    count, old_step, new_step = old["consumed"], old["start_step"], new["start_step"]
    if (
        type(count) is not int
        or not 1 <= count <= 8
        or type(old_step) is not int
        or type(new_step) is not int
        or new_step != old_step + count
    ):
        raise ValueError("Consecutive completed windows must share a future endpoint")
    old_poses, new_poses = validate_transform(old["poses"]), validate_transform(new["poses"])
    old_open, new_open = np.asarray(old["openings"], dtype=float), np.asarray(new["openings"], dtype=float)
    if old_poses.shape != (40, 4, 4) or new_poses.shape != (40, 4, 4):
        raise ValueError("Both decoded H40 pose trajectories are required")
    if (
        old_open.shape != (40,)
        or new_open.shape != (40,)
        or not np.isfinite(old_open).all()
        or not np.isfinite(new_open).all()
        or np.any((old_open < 0) | (old_open > 1))
        or np.any((new_open < 0) | (new_open > 1))
    ):
        raise ValueError("Both decoded opening trajectories must be finite and in range")
    left, right = old_poses[count], new_poses[0]
    rotation = Rotation.from_matrix(left[:3, :3].T @ right[:3, :3]).magnitude()
    return dict(
        previous_start_step=old_step,
        new_start_step=new_step,
        common_endpoint_step=new_step + 1,
        old_cursor=count,
        boundary=old["boundary"],
        position_difference_m=float(np.linalg.norm(left[:3, 3] - right[:3, 3])),
        rotation_difference_rad=float(rotation),
        old_opening=float(old_open[count]),
        new_opening=float(new_open[0]),
        absolute_opening_difference=float(abs(old_open[count] - new_open[0])),
        old_closed_new_open=bool(old_open[count] <= 0.2 and new_open[0] >= 0.8),
    )


def summarize(rows):
    if not rows:
        raise ValueError("At least one handoff is required")
    result = dict(handoffs=len(rows), old_closed_new_open=sum(row["old_closed_new_open"] for row in rows))
    for key in ("position_difference_m", "rotation_difference_rad", "absolute_opening_difference"):
        values = np.asarray([row[key] for row in rows])
        result[key] = dict(
            median=float(np.median(values)), p90=float(np.percentile(values, 90)), max=float(values.max())
        )
    return result


def audit_case(row, consumed):
    index = row["case_index"]
    producer, model = (prefix.read(row[key], consumed) for key in ("producer_ref", "model_ref"))
    inputs = admission.load_inputs(index, verify_assets=False)
    for path, digest in inputs["source_sha256"].items():
        prefix.checked({"path": path, "sha256": digest}, consumed)
    tool = validate_transform(inputs["case"]["tool"])
    runtime = producer["runtime"]
    segments, captures, exchanges, responses = (
        runtime["continuation"]["adaptive"]["segments"],
        runtime["adaptive_captures"],
        runtime["adaptive_model_exchanges"],
        model["responses"],
    )
    prefix.require(
        producer["complete"] is producer["passed"] is model["complete"] is model["passed"] is True
        and len(segments) == len(captures) == len(exchanges) == len(responses) == row["counts"]["model_requests"],
        "Every retained WAM reference must bind the original physical and model run",
    )
    windows, reasons, opening_summaries = [], collections.Counter(), []
    for segment, capture, exchange, response in zip(segments, captures, exchanges, responses, strict=True):
        request, envelope = (prefix.read(exchange[key], consumed) for key in ("request_ref", "response_ref"))
        actual_response = prefix.read(response["actual_response_ref"], consumed)
        arrays = repro.response_arrays(actual_response)
        np.testing.assert_array_equal(arrays["action"], envelope["action"])
        start = segment["start_step"]
        prefix.require(
            start == capture["frame_index"] == exchange["step"] == request["step"] == response["step"]
            and request["image"]["sha256"] == envelope["image_sha256"] == capture["canvas_png"]["sha256"],
            "The request, original model output and current image must share identity",
        )
        anchor = validate_transform(np.asarray(capture["current_source_pose"]) @ tool)
        poses, openings = decode_action_targets(anchor, arrays["action"], horizon=40)
        periods = segment["continuation"]["periods"]
        count = len(periods)
        refs = [period["completed_feedback"]["score"]["opening_reference"] for period in periods]
        np.testing.assert_allclose(openings[:count], refs, atol=1e-10, rtol=0)
        prefix.require(
            [period["step"] for period in periods] == list(range(start, start + count)),
            "Only actually executed consecutive reference endpoints may be counted",
        )
        opening_summaries.append(window_summary(openings, count, capture["current_opening"]))
        for signal in segment["signals"]:
            if signal["replan"]:
                reasons[signal["reason"]] += 1
        windows.append(
            dict(start_step=start, consumed=count, poses=poses, openings=openings, boundary=segment.get("boundary"))
        )
    handoffs = [compare_handoff(left, right) for left, right in zip(windows, windows[1:])]
    groups = {
        name: [item for item in handoffs if item["boundary"] == name] for name in ("quality_trigger", "window_end")
    }
    prefix.require(
        sum(map(len, groups.values())) == len(handoffs), "Every nonterminal handoff must have a known boundary"
    )
    parts = {}
    for name in ("all", "consumed", "unconsumed"):
        subsets = [item[name] for item in opening_summaries]
        parts[name] = {
            "endpoints": sum(item["endpoints"] for item in subsets),
            "absolute_close_count": sum(item["absolute_close_count"] for item in subsets),
            "anchor_drop_count": sum(item["anchor_drop_count"] for item in subsets),
        }
    return dict(
        case_index=index,
        case_id=row["case_id"],
        requests=len(windows),
        consumed_lengths=[item["consumed"] for item in windows],
        replan_reasons=dict(reasons),
        opening=parts,
        handoffs=handoffs,
        handoff_summary=summarize(handoffs),
        boundary_summaries={name: summarize(items) for name, items in groups.items() if items},
    )


def main():
    prefix.environment()
    consumed = {}
    source = prefix.read(SOURCE, consumed)
    prefix.require(
        source["complete"] is True and [case["case_index"] for case in source["cases"]] == list(INDICES),
        "The complete six-case audit is required",
    )
    cases = [audit_case(case, consumed) for case in source["cases"]]
    all_handoffs = [item for case in cases for item in case["handoffs"]]
    prefix.require(
        sum(case["requests"] for case in cases) == 213 and len(all_handoffs) == 207, "Full denominator required"
    )
    for path, digest in consumed.items():
        prefix.checked({"path": path, "sha256": digest})
    print(
        publish_result(
            OUTPUT,
            dict(
                complete=True,
                passed=True,
                retrospective=True,
                control_qualified=False,
                source_refs=[prefix.reference(PLAN), prefix.reference(Path(__file__).resolve())],
                consumed_sha256=consumed,
                cases=cases,
                pooled_handoff_summary=summarize(all_handoffs),
                new_model_requests=0,
                new_physics_steps=0,
                fits=0,
                fresh_cases=0,
            ),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
