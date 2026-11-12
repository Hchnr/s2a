"""Replay a frozen H40 and capture its actual step-48 observation."""

import argparse
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.native_runtime.reset_canvas_threads import render_with_two_torch_threads
from s2a.research.native_runtime.streaming_continuation import measured_snapshot
from s2a.research.wam.training.native.native_training import state_sha256
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import registered_feedback_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_bowl_fixed_h40_admission as previous_admission
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.simulation.runtime import run_reset_bowl_fixed_h40 as previous_owner
from scripts.simulation.runtime.reset_live_two_window_replay_v2 import minimal_renderer_endpoint

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/reset_bowl_h40_reobserve_capture_20261112.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/reset_bowl_h40_reobserve_20261112.md"
OUTPUT = ROOT / "outputs/reset_bowl_h40_reobserve_capture_20261112"
QA = ROOT / "outputs/qa/reset_bowl_h40_reobserve_capture_20261112/result.json"
DRY = ROOT / "outputs/reset_bowl_h40_reobserve_capture_dry_20261112/result.json"
AUDIT = {
    "path": str(ROOT / "outputs/reset_bowl_fixed_h40_audit_20261112/result.json"),
    "sha256": "df5e3d2a7ff15fb432f22aad566e4b35c5ea092e1568b3552c05f52484442e8b",
}
PRIOR = {
    "path": str(ROOT / "outputs/bowl_h40_closing_intent_audit_20261112/result.json"),
    "sha256": "12fc31f320a48e6a8bb9b4b925b587f97c53a647a47691a66714717d596413d2",
}
FIXED = dict(
    schema="s2a.reset-bowl-h40-reobserve-capture.v1",
    output=str(OUTPUT.relative_to(ROOT)),
    case_id="task0_init0_s42_raw",
    seed=42,
    capture_step=48,
    max_control_steps=48,
    max_native_steps=1200,
    max_models=3,
    max_contexts=3,
    previous_control_steps=558,
    previous_native_steps=13950,
    previous_models=45,
    previous_contexts=45,
    previous_model_initializations=6,
    previous_requests_attempted=18,
    previous_requests_completed=15,
    previous_denoises=525,
    model_initializations=0,
    model_requests=0,
    fresh_cases=0,
    automatic_retry=False,
    worker_timeout_seconds=2400,
)


def source_refs():
    own = (
        CONFIG,
        PLAN,
        Path(__file__).resolve(),
        ROOT / "tests/research/native_runtime/test_reset_bowl_h40_reobserve_capture.py",
    )
    return previous_admission.source_refs() + [prefix.reference(path) for path in own]


def load_inputs(*, verify_assets=True):
    config = prefix.read(prefix.reference(CONFIG))
    prefix.require(
        type(config) is dict
        and set(config) == set(FIXED)
        and all(type(config[key]) is type(value) and config[key] == value for key, value in FIXED.items()),
        "The registered case and exact capture budget are required",
    )
    base = previous_admission.load_inputs(verify_assets=verify_assets)
    audit, prior = prefix.read(AUDIT), prefix.read(PRIOR)
    prefix.require(
        audit["complete"] is audit["passed"] is prior["complete"] is prior["passed"] is True
        and audit["control_qualified"] is prior["control_qualified"] is False
        and base["case"]["case"]["case_id"] == config["case_id"]
        and prior["cumulative_periods"] == config["previous_control_steps"]
        and prior["cumulative_inner_steps"] == config["previous_native_steps"]
        and prior["cumulative_sim_models"] == config["previous_models"]
        and prior["cumulative_contexts"] == config["previous_contexts"]
        and prior["cumulative_model_initializations"] == config["previous_model_initializations"]
        and prior["cumulative_requests_attempted"] == config["previous_requests_attempted"]
        and prior["cumulative_requests_completed"] == config["previous_requests_completed"]
        and prior["cumulative_denoises"] == config["previous_denoises"],
        "The original audited run and complete prior costs are required",
    )
    producer_ref = audit["producer_branch_ref"]
    producer = prefix.read(producer_ref)
    fixed = producer["runtime"]["continuation"]["fixed"]
    prefix.require(
        producer["complete"] is producer["passed"] is fixed["complete"] is True
        and fixed["stop"] is None
        and sum(len(segment["continuation"]["periods"]) for segment in fixed["segments"]) == 40,
        "All 40 generated endpoints must have been executed",
    )
    return dict(config=config, base=base, producer_ref=producer_ref, producer=producer)


def input_receipt(inputs):
    return dict(
        config=prefix.reference(CONFIG),
        plan=prefix.reference(PLAN),
        original_audit=AUDIT,
        previous_cost_audit=PRIOR,
        producer_branch_ref=inputs["producer_ref"],
        original_inputs=previous_admission.input_receipt(inputs["base"]),
    )


def capture_after_replay(
    env, terminal, observers, receipt, *, checker, capture_render=render_with_two_torch_threads, **kwargs
):
    """Capture only the replayed terminal endpoint, retaining the actual current state."""
    saved = []

    def endpoint(actual_env, step):
        sample, evidence = observers.endpoint(actual_env, step)
        if step == 48:
            saved.append((copy.deepcopy(sample), copy.deepcopy(evidence)))
        return sample, evidence

    wrapped = SimpleNamespace(
        endpoint=endpoint,
        current=observers.current,
        render_anchor=observers.render_anchor,
        render_cameras=observers.render_cameras,
    )
    result = checker(env, terminal, wrapped, receipt, **kwargs)
    periods = [row for segment in result["replays"] for row in segment["periods"]]
    expected = kwargs["producer"]["runtime"]["continuation"]["fixed"]["segments"][-1]["continuation"]["periods"][-1]
    prefix.require(
        len(result["first_window"]["periods"]) == 8
        and len(periods) == 40
        and len(saved) == 1
        and result["reported_producer_stop"] is None
        and periods[-1]["endpoint_receipt"] == expected["endpoint_receipt"] == saved[0][1],
        "The complete replay must end at the producer's actual step-48 endpoint",
    )
    sample, endpoint_receipt = saved[0]
    current = measured_snapshot(env, 48)
    identity = minimal_renderer_endpoint(endpoint_receipt, 48)
    binding = {"sim_time": current["sim_time_s"], "sim_state_sha256": current["state_sha256"]}
    prefix.require(
        identity == {"endpoint_step": 48, **binding} and state_sha256(sample["sim_states"]) == current["state_sha256"],
        "The live endpoint must match the measured state",
    )
    expected_pose = np.asarray(sample["source_pose"]).copy()
    expected_opening = sample["opening"]
    expected_identity = {"endpoint_step": 48, **binding}
    scope = {}
    capture = capture_render(
        observers.render_anchor, scope, env, 48, sample, copy.deepcopy(identity), kwargs["runtime_output"]
    )
    prefix.require(
        measured_snapshot(env, 48) == current
        and capture["frame_index"] == 48
        and capture["current_endpoint"] == expected_identity
        and capture["render_before"] == capture["render_after"] == binding
        and capture["future_rgb_frames"] == 0,
        "Rendering must use the current endpoint without changing the simulation",
    )
    np.testing.assert_array_equal(capture["current_source_pose"], expected_pose)
    prefix.require(capture["current_opening"] == expected_opening, "The measured gripper opening must match")
    receipt["reobserve_step48"] = dict(
        snapshot=current, capture=capture, canvas_thread_scope=scope.get("canvas_thread_scope")
    )
    return result


def dry_run():
    inputs = load_inputs()
    result = dict(
        complete=True,
        passed=True,
        inputs=input_receipt(inputs),
        source_refs=source_refs(),
        physics_steps=0,
        model_initializations=0,
        model_requests=0,
        automatic_retry=False,
    )
    print(publish(DRY, result), flush=True)


def worker():
    from scripts.simulation.runtime import reset_bowl_fixed_h40_replay as replay

    original_checker = replay.run_checker

    def capture_checker(*args, **kwargs):
        return capture_after_replay(*args, checker=original_checker, **kwargs)

    replay.run_checker = capture_checker
    previous_owner.OUTPUT = OUTPUT
    previous_owner.worker("checker")


def execute():
    prefix.environment()
    inputs = load_inputs()
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    before = snapshot(ROOT)["sources"]
    sources = source_refs()
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and not qa["collection"]["collection_errors"]
        and prefix.read(qa["source_before_ref"])["sources"] == before
        and dry["complete"] is dry["passed"] is True
        and dry["inputs"] == input_receipt(inputs)
        and dry["source_refs"] == sources
        and dry["physics_steps"] == dry["model_initializations"] == dry["model_requests"] == 0,
        "A source-matched 32-worker regression and zero-physics dry are required",
    )
    directory = OUTPUT / "checker"
    directory.mkdir(parents=True, exist_ok=False)
    simulation = directory / "simulation"
    simulation.mkdir()
    registration = publish(
        directory / "registration.json",
        dict(
            phase="checker",
            inputs=previous_admission.input_receipt(inputs["base"]),
            source_refs=previous_admission.source_refs(),
            producer_branch_ref=inputs["producer_ref"],
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        registration_ref=registration,
        inputs=input_receipt(inputs),
        model_initializations=0,
        model_requests=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        environment = physical._cpu_environment(simulation)
        spec = dict(
            key="checker",
            argv=[sys.executable, "-m", "scripts.simulation.runtime.run_reset_bowl_h40_reobserve_capture", "--worker"],
            log=directory / "worker.log",
        )
        with physical._temporary_environment(environment):
            lifecycle = run_owned_workers([spec], cwd=ROOT, timeout_seconds=inputs["config"]["worker_timeout_seconds"])
        result["workers_ref"] = publish(directory / "workers.json", lifecycle)
        registered_feedback_phase.require_closed_workers(lifecycle, ["checker"])
        branch_ref = prefix.reference(simulation / "result.json")
        branch = prefix.read(branch_ref)
        counts = previous_owner.summarize_branch(branch, "checker", producer=inputs["producer"])
        recorded = inputs["producer"]["runtime"]["continuation"]
        replayed = branch["runtime"]["continuation"]
        produced_periods = recorded["first_window"]["periods"] + [
            row for segment in recorded["fixed"]["segments"] for row in segment["continuation"]["periods"]
        ]
        replayed_periods = replayed["first_window"]["periods"] + [
            row for segment in replayed["replays"] for row in segment["periods"]
        ]
        differences = comparison.compare_periods(produced_periods, replayed_periods)
        capture = branch["runtime"]["reobserve_step48"]["capture"]
        prefix.checked(capture["canvas_png"])
        prefix.checked(capture["raw_cameras"])
        prefix.require(
            counts["control_steps"] == 48
            and counts["native_steps"] == 1200
            and counts["models"] == counts["contexts"] == 3
            and snapshot(ROOT)["sources"] == before
            and source_refs() == sources,
            "Capture counts and source identity must match the registered limits",
        )
        result.update(
            complete=True,
            passed=True,
            sources_unchanged=True,
            branch_ref=branch_ref,
            counts=counts,
            independent_max_absolute_difference=differences,
            capture=capture,
            cumulative_control_steps=inputs["config"]["previous_control_steps"] + counts["control_steps"],
            cumulative_native_steps=inputs["config"]["previous_native_steps"] + counts["native_steps"],
            cumulative_models=inputs["config"]["previous_models"] + counts["models"],
            cumulative_contexts=inputs["config"]["previous_contexts"] + counts["contexts"],
        )
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        branch_path = simulation / "result.json"
        if branch_path.is_file():
            result["available_branch_ref"] = prefix.reference(branch_path)
        print(publish(OUTPUT / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    elif args.execute:
        execute()
    else:
        worker()


if __name__ == "__main__":
    main()
