"""Audit completed adaptive actions retained by the failed long-run checker."""

import argparse
from pathlib import Path

from s2a.infrastructure.publication import publish_result as publish
from s2a.research.native_runtime.physical_cost_bounds import physical_cost_bounds
from scripts.simulation.runtime import adaptive_live_bowl_audit as historical_audit
from scripts.simulation.runtime import adaptive_live_bowl_long_admission as admission
from scripts.simulation.runtime import adaptive_live_bowl_long_owner as owner
from scripts.simulation.runtime import adaptive_live_bowl_recovery_audit as cost_audit
from scripts.simulation.runtime import adaptive_live_bowl_recovery_owner as accounting
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import registered_adaptive_execution as execution
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_live_two_window_replay as capture_comparison

PRODUCER = {
    "path": str(admission.OUTPUT / "producer/result.json"),
    "sha256": "afc31cfb643b013c584a95940dba7c4b7fa427273ed53d84362da020d2e204c5",
}
CHECKER = {
    "path": str(admission.OUTPUT / "checker/result.json"),
    "sha256": "39e80cb7b0861f2fcf8d6c8b350152e2c554d0cff3002e7ada006631c5addd8b",
}
FAILURE = "Replay endpoint observation changed the physical state"


def compare_partial(producer, checker):
    """Compare completed actions and image bytes without claiming checker closure."""
    runtime, replay = checker["runtime"], checker["runtime"]["adaptive_replay_partial"]
    produced = producer["runtime"]["continuation"]["adaptive"]
    prefix.require(
        producer["complete"] is producer["passed"] is True
        and checker["complete"] is checker["passed"] is False
        and runtime["cleanup_complete"] is True
        and runtime["failure_stage"] == "continuation"
        and runtime["error"]["message"] == replay["error"]["message"] == FAILURE
        and replay["complete"] is False
        and replay["reported_producer_stop"] == produced["stop"]
        and replay["stop_reason_independently_verified"] is False
        and replay["automatic_retry"] is False
        and replay["control_steps_attempted"]
        == replay["control_steps_completed"]
        == replay["replay_periods_committed"]
        == 40
        and runtime["control_steps_attempted"] == runtime["control_steps_completed"] == 147
        and runtime["newly_selected_commands"] == runtime["world_model_calls"] == 0,
        "The exact failed checker and its completed physical work are required",
    )
    expected_segments, actual_segments = produced["segments"], replay["segments"]
    prefix.require(len(expected_segments) == len(actual_segments) == 7, "Every recorded adaptive segment is required")
    expected = [row for segment in expected_segments for row in segment["continuation"]["periods"]]
    actual = [row for segment in actual_segments for row in segment["continuation"]["periods"]]
    prefix.require(
        len(expected) == len(actual) == 40
        and [row["step"] for row in expected] == [row["step"] for row in actual] == list(range(107, 147)),
        "Exactly 40 consecutive completed adaptive actions are required",
    )
    maxima = comparison.compare_periods(expected, actual)
    prefix.require(all(value == 0 for value in maxima.values()), "All saved adaptive samples must match exactly")
    expected_captures, actual_captures = producer["runtime"]["adaptive_captures"], runtime["adaptive_checker_captures"]
    prefix.require(len(expected_captures) == len(actual_captures) == 7, "Every request image must be retained")
    for left, right in zip(expected_captures, actual_captures, strict=True):
        capture_comparison.compare_capture(left, right)
        for name in ("raw_cameras", "canvas_png"):
            prefix.checked(left[name])
            prefix.checked(right[name])
            prefix.require(left[name]["sha256"] == right[name]["sha256"], "Actual saved image bytes differ")
    prior, adaptive = historical_audit.periods(producer, "producer")
    rows = prior + adaptive
    trailing = 0
    for row in reversed(rows):
        if not row["task_success"]:
            break
        trailing += 1
    return dict(
        adaptive_periods=len(actual),
        adaptive_inner_samples=sum(len(row["actual_inner"]) for row in actual),
        adaptive_max_absolute_difference=maxima,
        capture_steps=[row["frame_index"] for row in actual_captures],
        capture_bytes_match=True,
        failed_adaptive_endpoints=[row["step"] + 1 for row in actual if not row["task_success"]],
        trailing_success_periods=trailing,
        final_endpoint_receipt_matches=expected[-1]["endpoint_receipt"] == actual[-1]["endpoint_receipt"],
        checker_complete=False,
        checker_error=FAILURE,
    )


def audit():
    """Revalidate original sources, worker lifecycle and conservative costs."""
    prefix.environment()
    inputs = admission.load_inputs()
    trial = owner.make_trial()
    sources = trial.source_refs()
    producer, checker = prefix.read(PRODUCER), prefix.read(CHECKER)
    prior_physical, prior_model = execution.verified_producer_bounds(trial, inputs, producer, sources)
    prefix.require(
        checker["complete"] is checker["passed"] is False
        and checker["phase"] == "checker"
        and checker["automatic_retry"] is False
        and checker["partial_source_match"] is True
        and checker["error"]["message"] == "Every registered feedback worker must exit and be reaped without a signal",
        "The original failed checker result must be retained",
    )
    workers = prefix.read(checker["workers_ref"])
    prefix.require(
        workers["launch_error"] is None
        and len(workers["workers"]) == 1
        and workers["workers"][0]["key"] == "checker"
        and workers["workers"][0]["returncode"] == 1
        and workers["workers"][0]["actual_wait"] is workers["workers"][0]["reaped"] is True
        and workers["workers"][0]["signal_sent"] is False,
        "The failed checker must have a real closed worker receipt",
    )
    prefix.checked(workers["workers"][0]["log_ref"])
    produced, replayed = prefix.read(producer["branch_ref"]), prefix.read(checker["available_branch_ref"])
    details = compare_partial(produced, replayed)
    runtime, config = replayed["runtime"], inputs["config"]
    counters = runtime["counters"]
    prefix.require(
        counters["mj_step"]["attempted"] == counters["mj_step"]["completed"] == 3675
        and counters["model_construct"]["attempted"] == counters["model_construct"]["completed"] == 3
        and counters["context"]["attempted"] == counters["context"]["completed"] == 3,
        "The failed replay's actual physical counters must be complete",
    )
    limits = dict(control_steps=147, native_steps=3675, models=3, contexts=3)
    phase_physical = physical_cost_bounds(runtime, limits=limits, receipt_final=True)
    zero = dict(
        model_initialization_attempts=0,
        model_initializations=0,
        model_requests_attempted=0,
        model_requests_completed=0,
        denoise_calls_attempted=0,
        denoise_calls_completed=0,
    )
    phase_model = accounting.model_cost_bounds(zero, config=config, receipt_final=True)
    cumulative_physical = cost_audit.add_cost_bounds(prior_physical, phase_physical)
    cumulative_model = accounting.add_model_cost_bounds(prior_model, phase_model)
    prefix.require(
        checker["phase_physical_bounds"] == phase_physical
        and checker["phase_model_bounds"] == phase_model
        and checker["cumulative_physical_bounds"] == cumulative_physical
        and checker["cumulative_model_bounds"] == cumulative_model,
        "The failed checker's actual costs and previous uncertainty must remain accounted for",
    )
    return dict(
        complete=True,
        passed=True,
        evidence_level="saved_partial_checker_audit",
        control_qualified=False,
        producer_ref=PRODUCER,
        failed_checker_ref=CHECKER,
        producer_branch_ref=producer["branch_ref"],
        checker_branch_ref=checker["available_branch_ref"],
        checker_workers_ref=checker["workers_ref"],
        audit_source_ref=prefix.reference(Path(__file__).resolve()),
        details=details,
        cumulative_physical_bounds=cumulative_physical,
        cumulative_model_bounds=cumulative_model,
        new_model_requests=0,
        new_physics_steps=0,
        automatic_retry=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = (prefix.REPO / args.output).resolve()
    prefix.require(output.is_relative_to(prefix.REPO / "outputs"), "Audit output must stay under repository outputs")
    print(publish(output, audit()), flush=True)


if __name__ == "__main__":
    main()
