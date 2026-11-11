"""Run one early innovation-controlled physical action and isolated replay."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import finite
from s2a.infrastructure.publication import publish_result as publish
from scripts.simulation.runtime import prepare_early_causal_innovation_physical as admission
from scripts.simulation.runtime import prepare_quality_rate_repair_recorded_feedback as previous_admission
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_stream_live_cached as cached
from scripts.simulation.runtime import run_ordered_correct_stream as prior
from scripts.simulation.runtime import run_quality_rate_repair_recorded_feedback as previous
from scripts.wam.tracking.preview import run_causal_stream_live_48 as live_owner
from scripts.wam.tracking.preview import run_causal_stream_live_cached as cached_owner

ROOT, OUTPUT, QA, DRY = admission.ROOT, admission.OUTPUT, admission.QA, admission.DRY
MODULE = "scripts.simulation.runtime.run_early_causal_innovation_physical"
FORBIDDEN = previous.FORBIDDEN + ("s2a.research.trajectory_search.innovation_capped_quality_solver",)


def source_refs():
    return previous.source_refs() + [
        prefix.reference(path)
        for path in (
            admission.CONFIG,
            admission.PLAN,
            Path(__file__).resolve(),
            ROOT / "scripts/simulation/runtime/prepare_early_causal_innovation_physical.py",
            ROOT / "src/s2a/research/trajectory_search/innovation_capped_quality_solver.py",
            ROOT / "tests/research/native_runtime/test_early_causal_innovation_physical_admission.py",
            ROOT / "tests/research/native_runtime/test_early_causal_innovation_physical.py",
            ROOT / "tests/research/trajectory_search/test_innovation_capped_quality_solver.py",
        )
    ]


def checker_isolation():
    prior.checker_isolation()
    prefix.require(all(name not in sys.modules for name in FORBIDDEN), "Checker must not import candidate solvers")


def make_prefix_runtime():
    from s2a.research.native_runtime.guarded_stream_prefix import GuardedStreamPrefixRuntime

    return GuardedStreamPrefixRuntime()


def bounded_prefix(old, directory, continuation):
    """Reuse the guarded endpoint cache with a stricter native 43-period budget."""

    def fixed_budget(values, output, callback):
        runtime = make_prefix_runtime()
        return runtime._run(
            values["runtime_config"],
            values["record"],
            values["record_xml"],
            values["commands"],
            values["saved"],
            values["receipts"],
            output / "runtime",
            values["asset_refs"],
            values["controller"],
            continuation=callback,
            continuation_budget=43,
            continuation_limit=48,
            registered_seed=42,
            capture_final_prefix_inner=True,
        )

    adapted = previous.clone(cached._run_cached_prefix, {"frozen": SimpleNamespace(_run_prefix=fixed_budget)})
    return adapted(old, directory, continuation)


def bind_after_completed(refiner, period, receipt, *, solver=None, expected_solver=None):
    """Install one solver call using only the newly completed step121 error."""
    if solver is None:
        from s2a.research.trajectory_search.innovation_capped_quality_solver import solve_interior_plan as solver
    if expected_solver is None:
        from s2a.research.trajectory_search.quality_capped_inner_joint_plan import (
            solve_interior_plan as expected_solver,
        )
    from s2a.research.trajectory_search.innovation_capped_quality_solver import corrected_error

    prefix.require(
        period["step"] == 121
        and period["decision"]["quality"]["verification"]["passed"] is True
        and refiner.solver is expected_solver,
        "One independently checked completed step121 and original solver are required",
    )
    predicted = finite(period["decision"]["quality"]["first_tcp_error"], (6,))
    actual = finite(period["tcp_error"], (6,))
    translation = actual[:3] - predicted[:3]
    rotation = (Rotation.from_rotvec(actual[3:]).inv() * Rotation.from_rotvec(predicted[3:])).as_matrix()
    record = {
        "last_completed_step": 121,
        "translation_innovation_m": translation.copy(),
        "rotation_discrepancy": rotation.copy(),
        "new_action_budget": 1,
        "solver_calls": 0,
        "limit_stops": 0,
        "independent_audit_attempts": 0,
        "independent_audit_passes": 0,
        "task_labels_used": False,
    }
    original_auditor = refiner.auditor

    def selected_solver(*args, **kwargs):
        if record["solver_calls"]:
            record["limit_stops"] += 1
            return {"witness_found": False, "status": "registered_one_period_limit", "response_evaluations": 0}
        record["solver_calls"] += 1
        return solver(*args, translation_innovation=translation, rotation_discrepancy=rotation, **kwargs)

    def checked_auditor(witness, *args, **kwargs):
        record["independent_audit_attempts"] += 1
        checked = original_auditor(witness, *args, **kwargs)
        corrected = corrected_error(checked["first_tcp_error"], translation, rotation)
        np.testing.assert_allclose(corrected, witness["innovation_corrected_first_tcp_error"], atol=1e-10, rtol=0)
        prefix.require(
            checked["passed"] is checked["first_pose_caps_satisfied"] is checked["first_pose_caps_enforced"] is True
            and np.linalg.norm(corrected[:3]) <= 0.01 + 1e-10
            and np.linalg.norm(corrected[3:]) <= 0.05 + 1e-10,
            "Independent original and causal-innovation quality gates are required",
        )
        record["independent_audit_passes"] += 1
        return {**checked, "innovation_corrected_first_tcp_error": corrected, "innovation_gate_passed": True}

    refiner.solver, refiner.auditor = selected_solver, checked_auditor
    receipt["early_innovation_binding"] = record
    return record


def worker_inputs(phase):
    prefix.environment()
    registration = prefix.read(prefix.reference(OUTPUT / phase / "registration.json"))
    inputs = admission.load_inputs()
    prefix.require(
        registration["phase"] == phase
        and registration["inputs"] == admission.input_receipt(inputs)
        and registration["source_refs"] == source_refs(),
        "Worker inputs and sources must match the owner registration",
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
        if step < 122:
            previous_admission.matched_prior_period(period, inputs["prior_periods"][step - 80])
        if step == 121:
            prefix.require(set(holder) == {"tracking", "receipt"}, "Causal feedback owner must exist")
            bind_after_completed(holder["tracking"]._refiner, period, holder["receipt"])
        return True

    proxy = SimpleNamespace(reconstruct_gripper=prior.reconstruct_gripper, make_initial_policy=make_initial)
    run = previous.clone(
        cached.simulation_worker, {"prior": proxy, "guard_period": guard, "_run_cached_prefix": bounded_prefix}
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
        run = previous.clone(cached.checker_worker, {"_run_cached_prefix": bounded_prefix})
        run(output=OUTPUT, loader=lambda _phase: (inputs, registration))
        checker_isolation()
        result["after"] = True
    finally:
        print(publish(OUTPUT / "checker/policy_isolation.json", result), flush=True)


def summarize(branch, *, phase, producer=None):
    counts = live_owner.summarize_branch(branch, phase=phase, producer=producer)
    runtime = branch["runtime"]
    rows = previous_admission.periods(branch)
    prefix.require(
        counts["control_steps"] <= 123 and counts["native_steps"] <= 3075 and runtime["limits"]["mj_step"] == 3075,
        "The one-period native integration budget must be enforced",
    )
    if phase == "producer":
        receipts, config = runtime["segment_receipts"], admission.FIXED
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
            totals["quality_calls"] <= config["max_decisions"]
            and totals["baseline_responses"] <= config["max_baseline_model_responses"]
            and totals["quality_responses"] <= config["max_quality_model_responses"]
            and totals["endpoint_queries_charged"] <= config["max_independent_endpoint_queries"]
            and totals["transition_steps_charged"] <= config["max_independent_transition_steps"]
            and totals["gripper_candidates"] <= config["max_gripper_candidates"],
            "The registered decision and model-response budgets must hold",
        )
        if len(rows) >= 42:
            binding = runtime["early_innovation_binding"]
            prefix.require(
                binding["last_completed_step"] == 121
                and binding["task_labels_used"] is False
                and binding["solver_calls"] <= 1
                and binding["limit_stops"] <= 1,
                "Only the completed causal innovation and one new decision may be used",
            )
        if len(rows) == 43:
            stop = runtime["continuation"]["stop"]
            prefix.require(
                stop["at_step"] == 123
                and stop["solver_status"] == "registered_one_period_limit"
                and stop["next_action_dispatched"] is False,
                "The explicit experimental limit must stop before step123 dispatch",
            )
        counts.update(**totals, prior_periods_matched=min(len(rows), 42))
    else:
        expected = previous_admission.periods(producer)
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
    inputs = admission.load_inputs()
    _service, relay = previous.relay_service(inputs, OUTPUT / "producer/simulation/runtime")
    prefix.require(len(relay.records) == 6, "The complete saved reference roster is required")
    print(
        publish(
            DRY,
            {
                "complete": True,
                "passed": True,
                "inputs": admission.input_receipt(inputs),
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
        previous.check_admission, {"QA": QA, "DRY": DRY, "admission": admission, "source_refs": source_refs}
    )
    owner = previous.clone(
        previous.execute,
        {
            "OUTPUT": OUTPUT,
            "QA": QA,
            "DRY": DRY,
            "MODULE": MODULE,
            "admission": admission,
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
