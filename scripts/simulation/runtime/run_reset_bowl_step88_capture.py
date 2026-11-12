"""Independently replay the 88-period bowl run and capture its terminal image."""

import argparse
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.native_runtime.verified_terminal_capture import capture_current_endpoint
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import registered_feedback_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_bowl_reference_tail as previous

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/reset_bowl_step88_capture_20261112.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/reset_bowl_step88_reobserve_20261112.md"
OUTPUT = ROOT / "outputs/reset_bowl_step88_capture_20261112"
QA = ROOT / "outputs/qa/reset_bowl_step88_capture_20261112/result.json"
DRY = ROOT / "outputs/reset_bowl_step88_capture_dry_20261112/result.json"
PRODUCER = {
    "path": str(ROOT / "outputs/reset_bowl_h40_tail_20261112/producer/result.json"),
    "sha256": "006c62083fa238862f238a5e017aed2be41dbdc0d7e0d0a546c7bb1955c6b4ef",
}
CHECKER = {
    "path": str(ROOT / "outputs/reset_bowl_h40_tail_20261112/checker/result.json"),
    "sha256": "fae7d8d6eea76ab77f4b8a1d9322049dcbf2560147ddbf19be5560cd8f53c996",
}
STATE_SHA = "64f5fd441e30498b43753f91725ddefe625c016c1253cd5f99fceebf844ea789"
FIXED = dict(
    schema="s2a.reset-bowl-step88-capture.v1",
    output=str(OUTPUT.relative_to(ROOT)),
    case_id="task0_init0_s42_raw",
    seed=42,
    capture_step=88,
    state_sha256=STATE_SHA,
    max_control_steps=88,
    max_native_steps=2200,
    max_models=3,
    max_contexts=3,
    previous_control_steps=894,
    previous_native_steps=22350,
    previous_models=60,
    previous_contexts=60,
    previous_model_initializations=7,
    previous_requests_attempted=19,
    previous_requests_completed=16,
    previous_denoises=560,
    model_initializations=0,
    model_requests=0,
    fresh_cases=0,
    automatic_retry=False,
    worker_timeout_seconds=2400,
)


def source_refs():
    paths = (
        CONFIG,
        PLAN,
        Path(__file__).resolve(),
        ROOT / "src/s2a/research/native_runtime/verified_terminal_capture.py",
        ROOT / "tests/research/native_runtime/test_verified_terminal_capture.py",
        ROOT / "tests/research/native_runtime/test_reset_bowl_step88_capture.py",
    )
    return previous.source_refs() + [prefix.reference(path) for path in paths]


def load_inputs():
    prefix.environment()
    config = prefix.read(prefix.reference(CONFIG))
    prefix.require(
        type(config) is dict
        and set(config) == set(FIXED)
        and all(type(config[key]) is type(value) and config[key] == value for key, value in FIXED.items()),
        "The exact step-88 capture and resource limits are required",
    )
    base = previous.load_inputs()
    producer, checker = prefix.read(PRODUCER), prefix.read(CHECKER)
    prefix.require(
        all(item["complete"] is item["passed"] is item["sources_unchanged"] is True for item in (producer, checker))
        and producer["control_qualified"] is checker["control_qualified"] is False
        and producer["counts"]["control_steps"] == checker["counts"]["control_steps"] == 88
        and producer["counts"]["tail_steps"] == checker["counts"]["tail_steps"] == 32
        and producer["counts"]["stop"] is checker["counts"]["stop"] is None
        and all(value == 0 for value in checker["counts"]["independent_max_absolute_difference"].values())
        and checker["counts"]["cumulative_control_steps"] == config["previous_control_steps"]
        and checker["counts"]["cumulative_native_steps"] == config["previous_native_steps"]
        and checker["counts"]["cumulative_models"] == config["previous_models"]
        and checker["counts"]["cumulative_contexts"] == config["previous_contexts"]
        and base["config"]["previous_model_initializations"] == config["previous_model_initializations"]
        and base["config"]["previous_requests_attempted"] == config["previous_requests_attempted"]
        and base["config"]["previous_requests_completed"] == config["previous_requests_completed"]
        and base["config"]["previous_denoises"] == config["previous_denoises"],
        "The independently replayed 88-period trajectory and prior costs are required",
    )
    producer_ref = producer["branch_refs"][0]
    branch = prefix.read(producer_ref)
    endpoint = branch["runtime"]["continuation"]["tail"]["segments"][-1]["continuation"]["periods"][-1][
        "endpoint_receipt"
    ]
    prefix.require(
        branch["complete"] is branch["passed"] is True
        and branch["case"]["case_id"] == config["case_id"]
        and endpoint["endpoint_step"] == 88
        and endpoint["sim_state_sha256"] == STATE_SHA,
        "The exact recorded terminal state is required",
    )
    return dict(config=config, base=base, producer_ref=producer_ref, producer=branch)


def input_receipt(inputs):
    return dict(
        config=prefix.reference(CONFIG),
        plan=prefix.reference(PLAN),
        producer=PRODUCER,
        checker=CHECKER,
        producer_branch=inputs["producer_ref"],
        previous=previous.input_receipt(inputs["base"]),
    )


def capture_after_checker(env, terminal, observers, receipt, *, checker, producer, capture_render=None, **kwargs):
    """Render only after the actual full replay reaches the recorded endpoint."""
    from scripts.simulation.runtime import reset_bowl_reference_tail as tail

    saved = []

    def endpoint(actual_env, step):
        sample, evidence = observers.endpoint(actual_env, step)
        if step == 88:
            saved.append((copy.deepcopy(sample), copy.deepcopy(evidence)))
        return sample, evidence

    wrapped = SimpleNamespace(
        endpoint=endpoint,
        current=observers.current,
        render_anchor=observers.render_anchor,
        render_cameras=observers.render_cameras,
    )
    result = checker(env, terminal, wrapped, receipt, producer=producer, **kwargs)
    replayed = tail.prefix_periods(result["prefix"], replay=True) + tail.tail_periods(result["tail"])
    recorded = producer["runtime"]["continuation"]
    expected = tail.prefix_periods(recorded["prefix"]) + tail.tail_periods(recorded["tail"])
    prefix.require(
        len(expected) == len(replayed) == 88 and len(saved) == 1 and result["tail"]["stop"] is None,
        "The entire original 88-period trajectory must be replayed",
    )
    comparison.compare_periods(expected, replayed)
    options = dict(step=88, expected_endpoint=expected[-1]["endpoint_receipt"], runtime_output=kwargs["runtime_output"])
    if capture_render is not None:
        options["render_scope"] = capture_render
    receipt["terminal_capture"] = capture_current_endpoint(
        env, observers.render_anchor, saved[0][0], saved[0][1], **options
    )
    return result


def dry_run():
    inputs = load_inputs()
    print(
        publish(
            DRY,
            dict(
                complete=True,
                passed=True,
                inputs=input_receipt(inputs),
                source_refs=source_refs(),
                native_models=0,
                physics_steps=0,
                model_initializations=0,
                model_requests=0,
                automatic_retry=False,
            ),
        ),
        flush=True,
    )


def check_qa_and_dry(inputs):
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    before = snapshot(ROOT)["sources"]
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
        and dry["source_refs"] == source_refs()
        and dry["native_models"] == dry["physics_steps"] == dry["model_initializations"] == dry["model_requests"] == 0,
        "A matching 32-worker regression and zero-physics dry are required",
    )
    return before


def worker():
    from scripts.simulation.runtime import reset_bowl_reference_tail as tail

    inputs = load_inputs()
    registration = prefix.read(prefix.reference(OUTPUT / "checker/capture_registration.json"))
    prefix.require(
        registration["inputs"] == input_receipt(inputs) and registration["source_refs"] == source_refs(),
        "The capture worker must use the registered inputs and source",
    )
    original_checker = tail.run_checker

    def capture_checker(*args, **kwargs):
        return capture_after_checker(*args, checker=original_checker, **kwargs)

    tail.run_checker = capture_checker
    previous.OUTPUT = OUTPUT
    previous.worker("checker", 0, "raw")


def execute():
    prefix.environment()
    inputs = load_inputs()
    before = check_qa_and_dry(inputs)
    sources = source_refs()
    directory = OUTPUT / "checker"
    directory.mkdir(parents=True, exist_ok=False)
    capture_registration = publish(
        directory / "capture_registration.json", dict(inputs=input_receipt(inputs), source_refs=sources)
    )
    registration = publish(
        directory / "registration.json",
        dict(
            phase="checker",
            inputs=previous.input_receipt(inputs["base"]),
            source_refs=previous.source_refs(),
            producer_branch_refs={"case_00": inputs["producer_ref"]},
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        capture_registration_ref=capture_registration,
        registration_ref=registration,
        inputs=input_receipt(inputs),
        model_initializations=0,
        model_requests=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        spec = dict(
            key="checker",
            argv=[sys.executable, "-m", "scripts.simulation.runtime.run_reset_bowl_step88_capture", "--worker"],
            log=directory / "case_00/worker.log",
        )
        lifecycle = run_owned_workers([spec], cwd=ROOT, timeout_seconds=inputs["config"]["worker_timeout_seconds"])
        result["workers_ref"] = publish(directory / "workers.json", lifecycle)
        registered_feedback_phase.require_closed_workers(lifecycle, ["checker"])
        branch_ref = prefix.reference(directory / "case_00/result.json")
        branch = prefix.read(branch_ref)
        counts = previous.summarize_branch(branch, "checker", previous.FIXED, producer=inputs["producer"])
        from scripts.simulation.runtime import reset_bowl_reference_tail as tail

        recorded = inputs["producer"]["runtime"]["continuation"]
        replayed = branch["runtime"]["continuation"]
        actual = tail.prefix_periods(recorded["prefix"]) + tail.tail_periods(recorded["tail"])
        checked = tail.prefix_periods(replayed["prefix"], replay=True) + tail.tail_periods(replayed["tail"])
        differences = comparison.compare_periods(actual, checked)
        captured = branch["runtime"]["terminal_capture"]
        prefix.checked(captured["capture"]["canvas_png"])
        prefix.checked(captured["capture"]["raw_cameras"])
        prefix.require(
            counts["control_steps"] == 88
            and counts["native_steps"] == 2200
            and counts["models"] == counts["contexts"] == 3
            and captured["snapshot"]["state_sha256"] == STATE_SHA
            and snapshot(ROOT)["sources"] == before
            and source_refs() == sources,
            "Capture counts, state and source identity must match registration",
        )
        result.update(
            complete=True,
            passed=True,
            sources_unchanged=True,
            branch_ref=branch_ref,
            counts=counts,
            independent_max_absolute_difference=differences,
            terminal_capture=captured,
            cumulative_control_steps=inputs["config"]["previous_control_steps"] + counts["control_steps"],
            cumulative_native_steps=inputs["config"]["previous_native_steps"] + counts["native_steps"],
            cumulative_models=inputs["config"]["previous_models"] + counts["models"],
            cumulative_contexts=inputs["config"]["previous_contexts"] + counts["contexts"],
        )
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        branch_path = directory / "case_00/result.json"
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
