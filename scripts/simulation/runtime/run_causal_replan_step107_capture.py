"""Replay a verified prefix to its planned current-image observation boundary."""

import argparse
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.native_runtime.recorded_reference_prefix import capture_tail_prefix
from s2a.research.native_runtime.verified_terminal_capture import capture_current_endpoint
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import registered_feedback_phase as lifecycle_check
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_bowl_next_reference as reference
from scripts.simulation.runtime import reset_bowl_reference_tail as tail
from scripts.simulation.runtime import run_reset_bowl_next_reference as original
from scripts.tracking.trajectory_search import audit_replayed_reference_pose_bounds as bound
from scripts.tracking.trajectory_search import audit_replayed_reference_pose_bounds_v2 as source

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/causal_replan_step107_capture_20261112.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/causal_replan_step107_capture_20261112.md"
QA = ROOT / "outputs/qa/causal_replan_step107_capture_20261112/result.json"
DRY = ROOT / "outputs/causal_replan_step107_capture_dry_20261112/result.json"
OUTPUT = ROOT / "outputs/causal_replan_step107_capture_20261112"
MODULE = "scripts.simulation.runtime.run_causal_replan_step107_capture"
SHADOW = {
    "path": str(ROOT / "outputs/causal_partial_replan_shadow_20261112/result.json"),
    "sha256": "1d5c64a878801858e02d0f8e456e1b2851203642d1b6bfd71361e3e82c32b71c",
}
FIXED = dict(
    schema="s2a.causal-replan-step107-capture.v1",
    case_id="task0_init0_s42_raw",
    seed=42,
    reference_start_step=88,
    capture_step=107,
    state_sha256="04b8eea4cd6e611302b68ff036594129808fec0bb88cab97f4e55106ee4c148d",
    max_control_steps=107,
    max_native_steps=2675,
    max_models=3,
    max_contexts=3,
    previous_control_steps=1198,
    previous_native_steps=29950,
    previous_models=69,
    previous_contexts=69,
    previous_static_models=2,
    previous_model_initializations=8,
    previous_requests_attempted=20,
    previous_requests_completed=17,
    previous_denoises=595,
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
        ROOT / "src/s2a/research/native_runtime/recorded_reference_prefix.py",
        ROOT / "src/s2a/research/native_runtime/verified_terminal_capture.py",
        ROOT / "tests/research/native_runtime/test_recorded_reference_prefix.py",
        ROOT / "tests/research/native_runtime/test_causal_replan_step107_capture.py",
    )
    return original.source_refs() + [prefix.reference(path) for path in paths]


def load_inputs():
    prefix.environment()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    prefix.require(
        type(config) is dict
        and set(config) == set(FIXED)
        and all(type(config[key]) is type(value) and config[key] == value for key, value in FIXED.items()),
        "The exact observed-prefix identity and budgets are required",
    )
    inputs = source.load_inputs()
    shadow = prefix.read(SHADOW)
    shadow_registration = prefix.read(shadow["registration"])
    prefix.require(
        shadow["complete"] is shadow["passed"] is shadow["sources_unchanged"] is True
        and shadow["control_qualified"] is False
        and shadow["first_trigger"]["current_step"] == config["capture_step"]
        and shadow["first_trigger"]["reason"] == "quality_margin_consumed"
        and shadow_registration["inputs"] == bound.input_receipt(inputs),
        "The first completed-period shadow trigger and its sources are required",
    )
    last = inputs["periods"][config["capture_step"] - 1]
    prefix.require(
        inputs["producer"]["case"]["case_id"] == config["case_id"]
        and last["step"] == config["capture_step"] - 1
        and last["endpoint_receipt"]["endpoint_step"] == config["capture_step"]
        and last["endpoint_receipt"]["sim_state_sha256"] == config["state_sha256"],
        "The replayed endpoint must be the registered current state",
    )
    recorded = inputs["producer"]["runtime"]["continuation"]["new_reference"]
    selected = capture_tail_prefix(recorded, reference_start=config["reference_start_step"], capture_step=107)
    prefix.require(
        selected["generated_periods"] == 19 and selected["stop"]["reason"] == "requested_observation",
        "The capture must withhold the recorded action at step 107",
    )
    return dict(config=config, config_ref=config_ref, inputs=inputs, selected_tail=selected, shadow=shadow)


def input_receipt(inputs):
    return dict(
        config=inputs["config_ref"],
        plan=prefix.reference(PLAN),
        shadow=SHADOW,
        bound_inputs=bound.input_receipt(inputs["inputs"]),
    )


def derived_producer(inputs):
    """Keep the original branch immutable and label the replay-only truncation."""
    branch = copy.deepcopy(inputs["inputs"]["producer"])
    branch["runtime"]["continuation"]["new_reference"] = copy.deepcopy(inputs["selected_tail"])
    branch["capture_derivation"] = dict(
        scope="recorded_actions_only",
        source=inputs["inputs"]["refs"]["producer_branch"],
        capture_step=inputs["config"]["capture_step"],
        synthetic_stop="requested_observation",
    )
    return branch


def capture_after_checker(env, terminal, observers, receipt, *, checker, expected_periods, step, **kwargs):
    """Render only after the recorded prefix has been physically reproduced."""
    observed = []

    def endpoint(actual_env, current_step):
        sample, evidence = observers.endpoint(actual_env, current_step)
        if current_step == step:
            observed.append((copy.deepcopy(sample), copy.deepcopy(evidence)))
        return sample, evidence

    wrapped = SimpleNamespace(
        endpoint=endpoint,
        current=observers.current,
        render_anchor=observers.render_anchor,
        render_cameras=observers.render_cameras,
    )
    result = checker(env, terminal, wrapped, receipt, **kwargs)
    replayed = reference.prefix_periods(result["prefix88"], replay=True) + tail.tail_periods(result["new_reference"])
    prefix.require(
        len(replayed) == len(expected_periods) == step
        and len(observed) == 1
        and result["new_reference"]["stop"]
        == dict(reason="requested_observation", at_step=step, next_action_dispatched=False),
        "Exactly the previously dispatched prefix must reach the observation boundary",
    )
    comparison.compare_periods(expected_periods, replayed)
    options = dict(
        step=step, expected_endpoint=expected_periods[-1]["endpoint_receipt"], runtime_output=kwargs["runtime_output"]
    )
    if kwargs.get("render_scope") is not None:
        options["render_scope"] = kwargs["render_scope"]
    receipt["terminal_capture"] = capture_current_endpoint(
        env, observers.render_anchor, observed[0][0], observed[0][1], **options
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
        "Matching CPU regression and zero-physics dry are required",
    )
    return before


def worker():
    inputs = load_inputs()
    registration = prefix.read(prefix.reference(OUTPUT / "checker/capture_registration.json"))
    prefix.require(
        registration["inputs"] == input_receipt(inputs) and registration["source_refs"] == source_refs(),
        "The capture worker must use the registered sources",
    )
    from s2a.research.native_runtime import reset_bounded_runtime as bounded

    original_adapter = bounded.ResetBoundedRuntime
    original.OUTPUT = OUTPUT
    original.bind_worker(inputs["inputs"]["source"])
    original_checker = tail.run_checker

    class Adapter107(original_adapter):
        def run_feedback(self, *args, max_control_steps, **kwargs):
            prefix.require(max_control_steps == 88, "The inherited worker must request its original bound")
            return super().run_feedback(*args, max_control_steps=107, **kwargs)

    def capture_checker(*args, **kwargs):
        return capture_after_checker(
            *args, checker=original_checker, expected_periods=inputs["inputs"]["periods"][:107], step=107, **kwargs
        )

    bounded.ResetBoundedRuntime = Adapter107
    tail.run_checker = capture_checker
    original.previous.worker("checker", 0, "raw")


def execute():
    prefix.environment()
    inputs = load_inputs()
    before, sources = check_qa_and_dry(inputs), source_refs()
    directory = OUTPUT / "checker"
    directory.mkdir(parents=True, exist_ok=False)
    derivation = publish(OUTPUT / "derived_producer.json", derived_producer(inputs))
    capture_registration = publish(
        directory / "capture_registration.json", dict(inputs=input_receipt(inputs), source_refs=sources)
    )
    registration = publish(
        directory / "registration.json",
        dict(
            phase="checker",
            inputs=original.input_receipt(inputs["inputs"]["source"]),
            source_refs=original.source_refs(),
            producer_branch_refs={"case_00": derivation},
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        capture_registration_ref=capture_registration,
        registration_ref=registration,
        derived_producer_ref=derivation,
        model_initializations=0,
        model_requests=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        spec = dict(
            key="checker", argv=[sys.executable, "-m", MODULE, "--worker"], log=directory / "case_00/worker.log"
        )
        workers = run_owned_workers([spec], cwd=ROOT, timeout_seconds=inputs["config"]["worker_timeout_seconds"])
        for item in workers["workers"]:
            item["log_ref"] = prefix.reference(item.pop("log"))
        result["workers_ref"] = publish(directory / "workers.json", workers)
        lifecycle_check.require_closed_workers(workers, ["checker"])
        branch_ref = prefix.reference(directory / "case_00/result.json")
        branch = prefix.read(branch_ref)
        counts = original.summarize_branch(branch, "checker", original.FIXED, producer=derived_producer(inputs))
        replayed = branch["runtime"]["continuation"]
        checked = reference.prefix_periods(replayed["prefix88"], replay=True) + tail.tail_periods(
            replayed["new_reference"]
        )
        differences = comparison.compare_periods(inputs["inputs"]["periods"][:107], checked)
        captured = branch["runtime"]["terminal_capture"]
        prefix.checked(captured["capture"]["canvas_png"])
        prefix.checked(captured["capture"]["raw_cameras"])
        config = inputs["config"]
        prefix.require(
            counts["control_steps"] == config["max_control_steps"]
            and counts["native_steps"] == config["max_native_steps"]
            and counts["models"] == config["max_models"]
            and counts["contexts"] == config["max_contexts"]
            and captured["snapshot"]["state_sha256"] == config["state_sha256"]
            and all(value == 0 for value in differences.values())
            and snapshot(ROOT)["sources"] == before
            and source_refs() == sources,
            "Capture counts, endpoint and source identity must match registration",
        )
        result.update(
            complete=True,
            passed=True,
            sources_unchanged=True,
            branch_ref=branch_ref,
            counts=counts,
            independent_max_absolute_difference=differences,
            terminal_capture=captured,
            cumulative_control_steps=config["previous_control_steps"] + counts["control_steps"],
            cumulative_native_steps=config["previous_native_steps"] + counts["native_steps"],
            cumulative_models=config["previous_models"] + counts["models"],
            cumulative_contexts=config["previous_contexts"] + counts["contexts"],
        )
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        path = directory / "case_00/result.json"
        if path.is_file():
            result["available_branch_ref"] = prefix.reference(path)
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
