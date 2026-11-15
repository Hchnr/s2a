"""Execute and independently replay two causal-lookahead physical periods."""

import argparse
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from s2a.geometry import finite
from s2a.infrastructure.publication import publish_result as publish
from scripts.simulation.runtime import prepare_quality_rate_repair_recorded_feedback as historical_admission
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_stream_live_cached as cached
from scripts.simulation.runtime import run_early_causal_innovation_physical as early
from scripts.simulation.runtime import run_ordered_correct_stream as prior
from scripts.simulation.runtime import run_quality_rate_repair_recorded_feedback as previous
from scripts.wam.tracking.preview import run_causal_stream_live_48 as live_owner
from scripts.wam.tracking.preview import run_causal_stream_live_cached as cached_owner

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/joint_control/anticipatory_causal_physical_20261113.json"
PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/anticipatory_causal_physical_20261113.md"
OUTPUT = ROOT / "outputs/anticipatory_causal_physical_20261113"
QA = ROOT / "outputs/qa/anticipatory_causal_physical_20261113/result.json"
DRY = ROOT / "outputs/anticipatory_causal_physical_dry_20261113/result.json"
MODULE = "scripts.simulation.runtime.run_anticipatory_causal_physical"
FIXED = {
    "schema": "s2a.anticipatory-causal-physical.v1",
    "output": "outputs/anticipatory_causal_physical_20261113",
    "static_result": "outputs/anticipatory_causal_pose_static_20261113/result.json",
    "static_result_sha256": "def8ecbfedb3d0e6666b528b7a9eab8cdfbc482d200803a8aea3e63173567971",
    "case_index": 8,
    "seed": 42,
    "prefix_steps": 80,
    "prior_periods": 41,
    "first_decision_step": 121,
    "new_periods": 2,
    "corrected_position_cap_m": 0.007,
    "corrected_rotation_cap_rad": 0.04,
    "max_control_steps_per_phase": 123,
    "max_native_steps_per_phase": 3075,
    "max_decisions": 44,
    "max_baseline_model_responses": 21120,
    "max_quality_model_responses": 7200,
    "max_independent_endpoint_queries": 8600,
    "max_independent_transition_steps": 111800,
    "max_gripper_candidates": 288684,
    "new_model_initializations": 0,
    "new_model_requests": 0,
    "new_denoise_calls": 0,
    "fresh_cases": 0,
    "worker_timeout_seconds": 2400,
    "automatic_retry": False,
}
FORBIDDEN = early.FORBIDDEN + (
    "s2a.research.joint_control.preview.causal_lookahead_feedback",
    "s2a.research.trajectory_search.causal_first_pose_constraint",
    "s2a.research.trajectory_search.staged_first_feasibility",
)


def load_inputs():
    prefix.environment()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    prefix.require(
        type(config) is dict
        and set(config) == set(FIXED)
        and all(type(config[key]) is type(value) and config[key] == value for key, value in FIXED.items()),
        "The exact anticipatory physical protocol and budgets are required",
    )
    inputs = historical_admission.load_inputs()
    prior_receipt = copy.deepcopy(historical_admission.input_receipt(inputs))
    consumed = dict(inputs["source_sha256"])
    static_ref = {"path": str(ROOT / config["static_result"]), "sha256": config["static_result_sha256"]}
    static = prefix.read(static_ref, consumed)
    row = static["analysis"]["candidates"]["anticipatory"]
    prefix.require(
        static["complete"] is static["passed"] is static["tight_margin_witness"] is True
        and static["completed_step"] == 120
        and static["decision_step"] == 121
        and static["constrained_future_step"] == 122
        and static["reference_generated_step"] == 120
        and static["task_labels_used"] is False
        and static["physics_steps"] == static["model_requests"] == static["fresh_cases"] == 0
        and row["certified_model_witness"] is True,
        "A completed independent static preview is required before the physical probe",
    )
    for ref in static["source_refs"]:
        prefix.checked(ref, consumed)
    prefix.checked(config_ref, consumed)
    prefix.reference(PLAN)
    prefix.require(
        len(inputs["records"]) == 6 and [row["step"] for row in inputs["prior_periods"][:41]] == list(range(80, 121)),
        "Six source-bound references and the first 41 completed periods are required",
    )
    inputs.update(
        prior_receipt=prior_receipt,
        feedback_config=config,
        anticipatory_config_ref=config_ref,
        anticipatory_static_ref=static_ref,
        source_sha256=consumed,
    )
    return inputs


def input_receipt(inputs):
    return {
        "prior": inputs["prior_receipt"],
        "config_ref": inputs["anticipatory_config_ref"],
        "static_ref": inputs["anticipatory_static_ref"],
        "source_sha256": inputs["source_sha256"],
    }


ADMISSION = SimpleNamespace(load_inputs=load_inputs, input_receipt=input_receipt)


def source_refs():
    paths = (
        CONFIG,
        PLAN,
        Path(__file__),
        ROOT / "src/s2a/research/joint_control/preview/causal_lookahead_feedback.py",
        ROOT / "src/s2a/research/trajectory_search/causal_first_pose_constraint.py",
        ROOT / "src/s2a/research/trajectory_search/staged_first_feasibility.py",
        ROOT / "tests/research/joint_control/preview/test_causal_lookahead_feedback.py",
        ROOT / "tests/research/native_runtime/test_anticipatory_causal_physical.py",
    )
    return early.source_refs() + [prefix.reference(path) for path in paths]


def checker_isolation():
    early.checker_isolation()
    prefix.require(all(name not in sys.modules for name in FORBIDDEN), "The checker cannot import the candidate policy")


def bind_after_completed(refiner, period, receipt, *, adapter=None, expected_solver=None):
    """Attach a two-action budget after this run has actually completed step120."""
    if adapter is None:
        from s2a.research.joint_control.preview.causal_lookahead_feedback import CausalLookaheadSolverAudit

        adapter = CausalLookaheadSolverAudit(position_cap_m=0.007, rotation_cap_rad=0.04)
    if expected_solver is None:
        from s2a.research.trajectory_search.quality_capped_inner_joint_plan import (
            solve_interior_plan as expected_solver,
        )

    prefix.require(
        period["step"] == 120
        and period["decision"]["quality"]["verification"]["passed"] is True
        and refiner.solver is expected_solver,
        "The actual completed step120 and original quality solver are required",
    )
    predicted = finite(period["decision"]["quality"]["first_tcp_error"], (6,))
    actual = finite(period["tcp_error"], (6,))
    adapter.observe_completed(120, predicted, actual, next_pose_index=1)
    record = {
        "last_initial_completed_step": 120,
        "solver_calls": 0,
        "limit_stops": 0,
        "task_labels_used": False,
        "events": adapter.events,
    }

    def bounded_solver(*args, **kwargs):
        if record["solver_calls"] >= 2:
            record["limit_stops"] += 1
            return {"witness_found": False, "status": "registered_two_period_limit", "response_evaluations": 0}
        record["solver_calls"] += 1
        return adapter.solve(*args, **kwargs)

    refiner.solver, refiner.auditor = bounded_solver, adapter.audit
    refiner.max_quality_response_evaluations = 320
    receipt["causal_lookahead_binding"] = record
    return adapter


def worker_inputs(phase):
    prefix.environment()
    registration = prefix.read(prefix.reference(OUTPUT / phase / "registration.json"))
    inputs = load_inputs()
    prefix.require(
        registration["phase"] == phase
        and registration["inputs"] == input_receipt(inputs)
        and registration["source_refs"] == source_refs(),
        "Worker inputs and sources must match the phase registration",
    )
    return inputs, registration


def simulation_worker():
    inputs, registration = worker_inputs("producer")
    holder = {}

    def make_initial(env, values, gripper, receipt):
        policy, owner = prior.make_initial_policy(env, values, gripper, receipt)
        holder.update(tracking=policy.arm_policy.tracking, receipt=receipt)
        return policy, owner

    def guard(_old_periods, period):
        step = period.get("step")
        prefix.require(type(step) is int and 80 <= step <= 122, "One bounded completed period is required")
        if step <= 120:
            historical_admission.matched_prior_period(period, inputs["prior_periods"][step - 80])
        if step == 120:
            prefix.require(set(holder) == {"tracking", "receipt"}, "The causal feedback owner must exist")
            holder["adapter"] = bind_after_completed(holder["tracking"]._refiner, period, holder["receipt"])
        elif step in (121, 122):
            prefix.require("adapter" in holder, "The completed causal history must already be bound")
            holder["adapter"].observe_completed(
                step,
                period["decision"]["quality"]["first_tcp_error"],
                period["tcp_error"],
                next_pose_index=0,
            )
        return True

    proxy = SimpleNamespace(reconstruct_gripper=prior.reconstruct_gripper, make_initial_policy=make_initial)
    run = previous.clone(
        cached.simulation_worker, {"prior": proxy, "guard_period": guard, "_run_cached_prefix": early.bounded_prefix}
    )
    run(output=OUTPUT, loader=lambda _phase: (inputs, registration))


def relay_worker():
    run = previous.clone(previous.relay_worker, {"OUTPUT": OUTPUT, "worker_inputs": worker_inputs})
    run()


def checker_worker():
    inputs, registration = worker_inputs("checker")
    checker_isolation()
    result = {"before": True, "after": False, "forbidden_modules": list(FORBIDDEN)}
    try:
        run = previous.clone(cached.checker_worker, {"_run_cached_prefix": early.bounded_prefix})
        run(output=OUTPUT, loader=lambda _phase: (inputs, registration))
        checker_isolation()
        result["after"] = True
    finally:
        print(publish(OUTPUT / "checker/policy_isolation.json", result), flush=True)


def summarize(branch, *, phase, producer=None):
    counts = live_owner.summarize_branch(branch, phase=phase, producer=producer)
    runtime, rows = branch["runtime"], historical_admission.periods(branch)
    prefix.require(
        counts["control_steps"] <= 123 and counts["native_steps"] <= 3075 and runtime["limits"]["mj_step"] == 3075,
        "The two-action native integration budget must be enforced",
    )
    if phase == "producer":
        receipts = runtime["segment_receipts"]
        quality = [item["quality_refiner"] for item in receipts]
        gripper = [item["gripper_proposals"] for item in receipts]
        totals = {
            "quality_calls": sum(item["quality_calls_attempted"] for item in quality),
            "baseline_responses": sum(item["baseline_response_evaluations"] for item in quality),
            "quality_responses": sum(item["quality_response_evaluations"] for item in quality),
            "endpoint_queries_charged": sum(item["endpoint_queries_charged"] for item in quality),
            "transition_steps_charged": sum(item["transition_steps_charged"] for item in quality),
            "gripper_candidates": sum(item["candidate_sequences"] for item in gripper),
        }
        prefix.require(
            totals["quality_calls"] <= FIXED["max_decisions"]
            and totals["baseline_responses"] <= FIXED["max_baseline_model_responses"]
            and totals["quality_responses"] <= FIXED["max_quality_model_responses"]
            and totals["endpoint_queries_charged"] <= FIXED["max_independent_endpoint_queries"]
            and totals["transition_steps_charged"] <= FIXED["max_independent_transition_steps"]
            and totals["gripper_candidates"] <= FIXED["max_gripper_candidates"],
            "All registered decision and response budgets must hold",
        )
        if len(rows) >= 41:
            binding = runtime["causal_lookahead_binding"]
            prefix.require(
                binding["last_initial_completed_step"] == 120
                and binding["task_labels_used"] is False
                and binding["solver_calls"] <= 2
                and binding["limit_stops"] <= 1,
                "Only completed causal observations and two new decisions may be used",
            )
        if len(rows) == 43:
            stop = runtime["continuation"]["stop"]
            prefix.require(
                stop["at_step"] == 123
                and stop["solver_status"] == "registered_two_period_limit"
                and stop["next_action_dispatched"] is False,
                "The explicit experimental limit must stop before step123 dispatch",
            )
        counts.update(**totals, prior_periods_matched=min(len(rows), 41))
    else:
        expected = historical_admission.periods(producer)
        prefix.require(len(rows) == len(expected), "The checker must replay all physical actions")
        maximum = {key: 0.0 for key in ("state", "q", "qvel", "time")}
        for actual, checked in zip(expected, rows, strict=True):
            previous.historical.compare_first_period(actual, checked)
            maximum["state"] = max(
                maximum["state"], float(np.max(np.abs(np.asarray(actual["actual_state"]) - checked["actual_state"])))
            )
            for key in ("q", "qvel", "time"):
                left = np.asarray([item[key] for item in actual["actual_inner"]])
                right = np.asarray([item[key] for item in checked["actual_inner"]])
                maximum[key] = max(maximum[key], float(np.max(np.abs(left - right))))
        counts["replay_max_difference"] = maximum
    counts.update(cache_events=cached_owner.check_cache_events(branch), metrics=previous.metrics(branch))
    return counts


def dry_run():
    inputs = load_inputs()
    _service, relay = previous.relay_service(inputs, OUTPUT / "producer/simulation/runtime")
    prefix.require(len(relay.records) == 6, "The complete saved reference roster is required")
    print(
        publish(
            DRY,
            {
                "complete": True,
                "passed": True,
                "inputs": input_receipt(inputs),
                "source_refs": source_refs(),
                "new_model_initializations": 0,
                "new_model_requests": 0,
                "new_denoise_calls": 0,
                "new_physics_steps": 0,
                "fresh_cases": 0,
                "automatic_retry": False,
            },
        ),
        flush=True,
    )


def execute(phase):
    check = previous.clone(
        previous.check_admission, {"QA": QA, "DRY": DRY, "admission": ADMISSION, "source_refs": source_refs}
    )
    owner = previous.clone(
        previous.execute,
        {
            "OUTPUT": OUTPUT,
            "QA": QA,
            "DRY": DRY,
            "MODULE": MODULE,
            "admission": ADMISSION,
            "source_refs": source_refs,
            "check_admission": check,
            "summarize": summarize,
        },
    )
    owner(phase)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--producer", action="store_true")
    group.add_argument("--checker", action="store_true")
    group.add_argument("--worker", choices=("simulation", "relay", "checker"))
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    elif args.producer or args.checker:
        execute("producer" if args.producer else "checker")
    elif args.worker == "simulation":
        simulation_worker()
    elif args.worker == "relay":
        relay_worker()
    else:
        checker_worker()


if __name__ == "__main__":
    main()
