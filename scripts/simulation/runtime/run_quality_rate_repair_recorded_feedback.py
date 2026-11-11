"""Run bounded physical feedback against six source-verified saved responses."""

import argparse
import os
import socket
import sys
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np

from s2a.infrastructure.framed_channel import FramedJSONChannel
from s2a.infrastructure.lifecycle.owned_socket_pair import OwnedSocketPairInterrupted, run_owned_socket_pair
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import prepare_quality_rate_repair_recorded_feedback as admission
from scripts.simulation.runtime import registered_feedback_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_stream_live_cached as cached
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.simulation.runtime import run_causal_wam_two_window_feedback as historical
from scripts.simulation.runtime import run_ordered_correct_stream as prior
from scripts.wam.tracking.preview import run_causal_stream_live_48 as live_owner
from scripts.wam.tracking.preview import run_causal_stream_live_cached as cached_owner

ROOT, OUTPUT, QA, DRY = admission.ROOT, admission.OUTPUT, admission.QA, admission.DRY
MODULE = "scripts.simulation.runtime.run_quality_rate_repair_recorded_feedback"
FORBIDDEN = (
    "s2a.research.trajectory_search.quality_rate_repair_solver",
    "s2a.research.trajectory_search.rate_boundary_repair",
)


def source_refs():
    return cached_owner.source_refs() + [
        prefix.reference(path)
        for path in (
            admission.CONFIG,
            admission.PLAN,
            Path(__file__).resolve(),
            ROOT / "scripts/simulation/runtime/prepare_quality_rate_repair_recorded_feedback.py",
            ROOT / "src/s2a/research/trajectory_search/quality_rate_repair_solver.py",
            ROOT / "src/s2a/research/trajectory_search/rate_boundary_repair.py",
            ROOT / "src/s2a/research/wam/tracking/runtime/recorded_reference_relay.py",
            ROOT / "tests/research/native_runtime/test_quality_rate_repair_recorded_admission.py",
            ROOT / "tests/research/native_runtime/test_quality_rate_repair_recorded_feedback.py",
            ROOT / "tests/research/trajectory_search/test_quality_rate_repair_solver.py",
        )
    ]


def clone(function, replacements):
    namespace = function.__globals__.copy()
    namespace.update(replacements)
    cloned = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    cloned.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    return cloned


def checker_isolation():
    prior.checker_isolation()
    prefix.require(all(name not in sys.modules for name in FORBIDDEN), "Checker must not import the repaired solver")


def relay_config(inputs, image_root):
    config = physical.model_config(inputs, image_root)
    config["max_requests"] = 6
    return config


def relay_service(inputs, image_root):
    from s2a.research.wam.tracking.runtime.causal_stream_protocol import CausalModelAdmission
    from s2a.research.wam.tracking.runtime.recorded_reference_relay import RecordedReferenceRelay

    service = CausalModelAdmission(**relay_config(inputs, image_root))
    return service, RecordedReferenceRelay(service, inputs["records"])


def dry_run():
    inputs = admission.load_inputs()
    _service, relay = relay_service(inputs, OUTPUT / "producer/simulation/runtime")
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


def check_admission(inputs):
    check = clone(physical.check_admission, {"QA": QA, "DRY": DRY, "admission": admission, "source_refs": source_refs})
    return check(inputs)


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


def serve_saved(channel, inputs, result, *, image_root):
    from s2a.research.wam.tracking.runtime.causal_stream_control import check_stop

    service, relay = relay_service(inputs, image_root)
    for _index in range(6):
        frame = channel.receive()
        if frame.get("kind") == "stop":
            result["stop_frame"] = check_stop(service, frame)
            break
        result["local_reference_requests"] += 1
        response, visible = relay.respond(frame)
        prefix.require(
            set(visible) == {"image_bytes", "task"} and visible["task"] == inputs["config"]["task"],
            "Only the current verified image and public task enter the saved-response relay",
        )
        channel.send(response)
        result["responses_sent"] += 1
        result["records"] = relay.used
    prefix.require(
        service.phase == "done" or result.get("stop_frame") is not None,
        "The relay must exhaust its roster or receive a valid stop",
    )


def relay_worker():
    inputs, _registration = worker_inputs("producer")
    result = {
        "complete": False,
        "passed": False,
        "local_reference_requests": 0,
        "responses_sent": 0,
        "new_model_initializations": 0,
        "new_model_requests": 0,
        "new_denoise_calls": 0,
        "records": [],
        "automatic_retry": False,
    }
    channel = None
    try:
        channel = FramedJSONChannel(
            socket.socket(fileno=int(os.environ["S2A_CHANNEL_FD"])), max_bytes=2 * 1024 * 1024, timeout_seconds=2200
        )
        serve_saved(channel, inputs, result, image_root=OUTPUT / "producer/simulation/runtime")
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if channel is not None:
            channel.close()
        print(publish(OUTPUT / "producer/model/result.json", result), flush=True)


def inject_solver(policy, receipt):
    """Change only this producer's explicitly selected refiner instance."""
    from s2a.research.trajectory_search import quality_capped_inner_joint_plan, quality_rate_repair_solver

    refiner = policy.arm_policy.tracking._refiner
    prefix.require(
        refiner.solver is quality_capped_inner_joint_plan.solve_interior_plan,
        "The original quality solver must be bound before injection",
    )
    refiner.solver = quality_rate_repair_solver.solve_interior_plan
    receipt["quality_rate_repair_solver"] = {"injected": True, "default_source_unchanged": True}


def simulation_worker():
    inputs, registration = worker_inputs("producer")

    def make_initial(env, values, gripper, receipt):
        policy, owner = prior.make_initial_policy(env, values, gripper, receipt)
        inject_solver(policy, receipt)
        return policy, owner

    def guard(_previous_periods, period):
        step = period.get("step")
        prefix.require(type(step) is int and 80 <= step < 128, "A registered completed period is required")
        if step < 125:
            return admission.matched_prior_period(period, inputs["prior_periods"][step - 80])
        return True

    proxy = SimpleNamespace(reconstruct_gripper=prior.reconstruct_gripper, make_initial_policy=make_initial)
    run = clone(cached.simulation_worker, {"prior": proxy, "guard_period": guard})
    run(output=OUTPUT, loader=lambda _phase: (inputs, registration))


def checker_worker():
    inputs, registration = worker_inputs("checker")
    checker_isolation()
    result = {"before": True, "after": False, "forbidden_modules": list(FORBIDDEN)}
    try:
        cached.checker_worker(output=OUTPUT, loader=lambda _phase: (inputs, registration))
        checker_isolation()
        result["after"] = True
    finally:
        print(publish(OUTPUT / "checker/policy_isolation.json", result), flush=True)


def metrics(branch):
    rows = admission.periods(branch)
    tail = 0
    for row in reversed(rows):
        if not row["task_success"]:
            break
        tail += 1
    return {
        "continuation_periods": len(rows),
        "task_success_periods": sum(row["task_success"] for row in rows),
        "terminal_consecutive_success_periods": tail,
        "pose_caps_satisfied_periods": sum(row["pose_caps_satisfied"] for row in rows),
        "minimum_inner_joint_margin_rad": min((row["minimum_inner_margin_rad"] for row in rows), default=None),
        "below_buffer_samples": sum(row["inner_below_buffer_samples"] for row in rows),
        "outside_samples": sum(row["inner_outside_samples"] for row in rows),
        "control_qualified": False,
    }


def summarize(branch, *, phase, producer=None):
    counts = live_owner.summarize_branch(branch, phase=phase, producer=producer)
    cache = cached_owner.check_cache_events(branch)
    rows = admission.periods(branch)
    if phase == "producer":
        runtime, config = branch["runtime"], admission.FIXED
        receipts = runtime["segment_receipts"]
        quality = [item["quality_refiner"] for item in receipts]
        gripper = [item["gripper_proposals"] for item in receipts]
        totals = {
            "baseline_responses": sum(item["baseline_response_evaluations"] for item in quality),
            "quality_responses": sum(item["quality_response_evaluations"] for item in quality),
            "endpoint_queries_charged": sum(item["endpoint_queries_charged"] for item in quality),
            "transition_steps_charged": sum(item["transition_steps_charged"] for item in quality),
            "gripper_candidates": sum(item["candidate_sequences"] for item in gripper),
        }
        prefix.require(
            runtime["quality_rate_repair_solver"]["injected"] is True
            and totals["baseline_responses"] <= config["max_baseline_model_responses"]
            and totals["quality_responses"] <= config["max_quality_model_responses"]
            and totals["endpoint_queries_charged"] <= config["max_independent_endpoint_queries"]
            and totals["transition_steps_charged"] <= config["max_independent_transition_steps"]
            and totals["gripper_candidates"] <= config["max_gripper_candidates"],
            "The injected solver and all registered response budgets are required",
        )
        counts.update(**totals, prior_periods_matched=min(len(rows), 45))
    else:
        expected = admission.periods(producer)
        prefix.require(len(rows) == len(expected), "Checker must replay every completed action")
        maximum = {key: 0.0 for key in ("state", "q", "qvel", "time")}
        for actual, checked in zip(expected, rows, strict=True):
            historical.compare_first_period(actual, checked)
            maximum["state"] = max(
                maximum["state"], float(np.max(np.abs(np.asarray(actual["actual_state"]) - checked["actual_state"])))
            )
            for key in ("q", "qvel", "time"):
                left = np.asarray([item[key] for item in actual["actual_inner"]])
                right = np.asarray([item[key] for item in checked["actual_inner"]])
                maximum[key] = max(maximum[key], float(np.max(np.abs(left - right))))
        counts["replay_max_difference"] = maximum
    counts.update(cache_events=cache, metrics=metrics(branch))
    return counts


def execute(phase):
    prefix.environment()
    prefix.require(phase in ("producer", "checker"), "One registered phase is required")
    inputs = admission.load_inputs()
    before, sources = check_admission(inputs), source_refs()
    producer, producer_ref = None, None
    if phase == "checker":
        previous = prefix.read(prefix.reference(OUTPUT / "producer/result.json"))
        prefix.require(
            previous["complete"] is previous["passed"] is previous["sources_unchanged"] is True,
            "A closed producer phase is required",
        )
        producer_ref = previous["branch_ref"]
        producer = prefix.read(producer_ref)
    directory = OUTPUT / phase
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "simulation").mkdir()
    registration = publish(
        directory / "registration.json",
        {
            "phase": phase,
            "inputs": admission.input_receipt(inputs),
            "source_refs": sources,
            "producer_branch_ref": producer_ref,
        },
    )
    result = {
        "complete": False,
        "passed": False,
        "phase": phase,
        "control_qualified": False,
        "registration_ref": registration,
        "new_model_initializations": 0,
        "new_model_requests": 0,
        "new_denoise_calls": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
    }
    try:
        simulation_env = physical._cpu_environment(directory / "simulation")
        command = [sys.executable, "-m", MODULE]
        if phase == "producer":
            (directory / "model").mkdir()
            model_env = physical._cpu_environment(directory / "model")
            specs = {
                "simulation": {
                    "argv": command + ["--worker", "simulation"],
                    "env": simulation_env,
                    "log": str((directory / "simulation.log").relative_to(ROOT)),
                },
                "model": {
                    "argv": command + ["--worker", "relay"],
                    "env": model_env,
                    "log": str((directory / "model.log").relative_to(ROOT)),
                },
            }
            try:
                lifecycle = run_owned_socket_pair(
                    specs, timeout_seconds=inputs["feedback_config"]["worker_timeout_seconds"]
                )
            except OwnedSocketPairInterrupted as error:
                result["workers_ref"] = publish(directory / "workers.json", error.result)
                result["interrupted"] = True
                raise
            result["workers_ref"] = publish(directory / "workers.json", lifecycle)
            prefix.require(lifecycle["passed"] is True, "Both owned producer children must exit and be reaped")
            relay_ref = prefix.reference(directory / "model/result.json")
            relay = prefix.read(relay_ref)
            prefix.require(
                relay["complete"] is relay["passed"] is True
                and relay["new_model_initializations"] == relay["new_model_requests"] == relay["new_denoise_calls"] == 0
                and relay["local_reference_requests"] == relay["responses_sent"] == len(relay["records"])
                and [row["historical_response_sha256"] for row in relay["records"]]
                == [row["response_sha256"] for row in inputs["records"][: len(relay["records"])]],
                "Only the registered saved response prefix may be relayed",
            )
            result["relay_ref"] = relay_ref
        else:
            spec = {"key": "checker", "argv": command + ["--worker", "checker"], "log": directory / "checker.log"}
            with physical._temporary_environment(simulation_env):
                lifecycle = run_owned_workers(
                    [spec], cwd=ROOT, timeout_seconds=inputs["feedback_config"]["worker_timeout_seconds"]
                )
            result["workers_ref"] = publish(directory / "workers.json", lifecycle)
            registered_feedback_phase.require_closed_workers(lifecycle, ["checker"])
            isolation_ref = prefix.reference(directory / "policy_isolation.json")
            isolation = prefix.read(isolation_ref)
            prefix.require(isolation["before"] is isolation["after"] is True, "Checker policy isolation is required")
            result["policy_isolation_ref"] = isolation_ref
        branch_ref = prefix.reference(directory / "simulation/result.json")
        branch = prefix.read(branch_ref)
        counts = summarize(branch, phase=phase, producer=producer)
        if phase == "producer":
            prefix.require(
                relay["responses_sent"] == branch["runtime"]["continuation"]["responses_accepted"],
                "Every accepted reference must have a saved-response receipt",
            )
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(
            source_refs() == sources and snapshot(ROOT)["sources"] == before, "Sources changed during execution"
        )
        result.update(complete=True, passed=True, sources_unchanged=True, branch_ref=branch_ref, counts=counts)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


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
