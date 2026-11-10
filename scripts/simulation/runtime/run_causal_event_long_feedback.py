"""Extend the verified task1 matched references to sixteen fixed periods."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.native_runtime.development_continuation import current_joint_site_state, run_continuation
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import prepare_causal_event_feedback as binding
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_event_feedback as previous

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/causal_event_long_feedback_20261110.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/causal_event_long_feedback_20261110.md"
QA = REPO / "outputs/qa/causal_event_long_feedback_20261110/result.json"
DRY = REPO / "outputs/causal_event_long_feedback_dry_20261110/result.json"
ROOT = REPO / "outputs/causal_event_long_feedback_20261110"
PRIOR_PRODUCER = {
    "path": str(REPO / "outputs/causal_event_feedback_20261110/producer/result.json"),
    "sha256": "2e8d31155a7005cbc9b5c8ceca70723af537130d162bd7c3dd8ea1c8bcec8311",
}
PRIOR_CHECKER = {
    "path": str(REPO / "outputs/causal_event_feedback_20261110/checker/result.json"),
    "sha256": "c47902296c6b8ccd810d04c858a0b448f85096f3bd555833983343b8d43029b9",
}
BRANCHES = ("latest", "remembered")


def source_refs():
    return previous.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            CONFIG,
            PLAN,
            REPO / "tests/research/native_runtime/test_causal_event_long_feedback.py",
        )
    ]


def read_config():
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    prefix.require(
        config
        == {
            "schema": "s2a.causal-event-long-feedback.v1",
            "case_index": 5,
            "case_id": "task1_init0_s42_terminal058",
            "prefix_steps": 360,
            "old_step": 344,
            "task_id": 9,
            "target": "wine_bottle_1",
            "branches": list(BRANCHES),
            "seed": 42,
            "continuation_steps": 16,
            "previous_verified_steps": 8,
            "control_dt_s": 0.05,
            "joint_buffer_rad": 0.05,
            "position_cap_m": 0.01,
            "rotation_cap_rad": 0.05,
            "arm_rate_limit": 0.25,
            "comparison_atol": 1e-10,
            "comparison_rtol": 0,
            "max_control_steps_per_phase": 752,
            "max_native_steps_per_phase": 18800,
            "max_models_per_phase": 6,
            "max_contexts_per_phase": 6,
            "max_refiner_calls_per_phase": 32,
            "max_refiner_response_evaluations_per_phase": 15360,
            "max_quality_calls_per_phase": 32,
            "max_quality_response_evaluations_per_phase": 5120,
            "max_gripper_planner_calls_per_phase": 752,
            "max_gripper_candidate_sequences_per_phase": 4933872,
            "max_workers": 2,
            "worker_timeout_seconds": 1800,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
        "Exact registered 16-period paired comparison",
    )
    return ref, config


def load_inputs():
    config_ref, config = read_config()
    producer, checker = prefix.read(PRIOR_PRODUCER), prefix.read(PRIOR_CHECKER)
    prefix.require(
        producer["complete"]
        is producer["passed"]
        is producer["sources_unchanged"]
        is checker["complete"]
        is checker["passed"]
        is checker["sources_unchanged"]
        is True
        and len(producer["branch_refs"]) == len(checker["branch_refs"]) == 4,
        "Earlier matched producer and independent replay closed",
    )
    inputs = binding.load_inputs()
    case = next(item for item in inputs["cases"] if item["spec"]["case_index"] == 5)
    prefix.require(
        all(
            case["spec"][key] == config[key]
            for key in ("case_index", "case_id", "prefix_steps", "old_step", "task_id", "target")
        ),
        "Exact prior task1 seed42 case and source reference",
    )
    refs = dict(inputs["source_sha256"])
    refs.update(
        {
            config_ref["path"]: config_ref["sha256"],
            PRIOR_PRODUCER["path"]: PRIOR_PRODUCER["sha256"],
            PRIOR_CHECKER["path"]: PRIOR_CHECKER["sha256"],
        }
    )
    prior_branches = {}
    for mode, position in (("latest", 2), ("remembered", 3)):
        producer_ref, checker_ref = producer["branch_refs"][position], checker["branch_refs"][position]
        original, independent = prefix.read(producer_ref, refs), prefix.read(checker_ref, refs)
        prefix.require(
            original["complete"] is original["passed"] is independent["complete"] is independent["passed"] is True
            and original["case"]["case_id"] == independent["case"]["case_id"] == config["case_id"]
            and original["branch"] == independent["branch"] == mode
            and original["runtime"]["continuation"]["steps_completed"]
            == independent["runtime"]["continuation"]["steps_completed"]
            == 8
            and original["runtime"]["continuation"]["stop"] is independent["runtime"]["continuation"]["stop"] is None,
            "Both original 8-period branches were independently replayed",
        )
        prior_branches[mode] = {"producer_ref": producer_ref, "checker_ref": checker_ref}
        binding.selected_case(case, mode, horizon=16)
    return {
        "config_ref": config_ref,
        "config": config,
        "case": case,
        "source_sha256": refs,
        "asset_refs": inputs["asset_refs"],
        "cv2_admission": inputs["cv2_admission"],
        "prior_branches": prior_branches,
    }


def input_receipt(inputs):
    return {
        "config_ref": inputs["config_ref"],
        "case": inputs["case"]["spec"],
        "source_sha256": inputs["source_sha256"],
        "asset_refs": inputs["asset_refs"],
        "prior_branches": inputs["prior_branches"],
    }


def target_position(env, target):
    """Read object position only for completed-state evaluation."""
    core = env.env
    prefix.require(target in core.obj_body_id, "Registered evaluation object")
    position = np.asarray(env.sim.data.body_xpos[core.obj_body_id[target]], dtype=float).copy()
    prefix.require(position.shape == (3,) and np.isfinite(position).all(), "Finite actual target position")
    return position


def check_previous_period(period, expected):
    """Require unchanged first-eight actions and physical outcomes."""
    np.testing.assert_array_equal(period["command"], expected["command"])
    np.testing.assert_allclose(period["actual_state"], expected["actual_state"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(period["tcp_error"], expected["tcp_error"], atol=1e-10, rtol=0)
    prefix.require(len(period["actual_inner"]) == len(expected["actual_inner"]) == 25, "Complete original inner period")
    for key in ("q", "qvel", "time"):
        np.testing.assert_allclose(
            [item[key] for item in period["actual_inner"]],
            [item[key] for item in expected["actual_inner"]],
            atol=1e-10,
            rtol=0,
        )
    for key in (
        "step",
        "inner_outside_samples",
        "inner_below_buffer_samples",
        "pose_caps_satisfied",
        "task_success",
        "target_grasp",
    ):
        prefix.require(period[key] == expected[key], f"Earlier physical label differs: {key}")


def worker(phase, mode):
    prefix.environment()
    prefix.require(mode in BRANCHES, "Registered reference branch")
    directory = ROOT / phase / mode
    registration = prefix.read(prefix.reference(ROOT / phase / "registration.json"))
    inputs = load_inputs()
    prefix.require(
        input_receipt(inputs) == registration["inputs"] and source_refs() == registration["source_refs"],
        "Frozen long-feedback sources and inputs",
    )
    case = inputs["case"]
    selected = binding.selected_case(case, mode, horizon=16)
    earlier = prefix.read(inputs["prior_branches"][mode]["producer_ref"])
    earlier_periods = earlier["runtime"]["continuation"]["periods"]
    expected, expected_stop = None, None
    if phase == "checker":
        producer = prefix.read(registration["producer_branch_refs"][mode])
        prefix.require(producer["complete"] is producer["passed"] is True, "Closed long producer branch")
        expected, expected_stop = (
            producer["runtime"]["continuation"]["periods"],
            producer["runtime"]["continuation"]["stop"],
        )
    result = {
        "complete": False,
        "passed": False,
        "control_qualified": False,
        "phase": phase,
        "case": case["spec"],
        "branch": mode,
        "automatic_retry": False,
    }
    try:
        gripper = None if phase == "checker" else previous.reconstruct_gripper(case, result)
        cv2 = inputs["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        from s2a.research.native_runtime.development_prefix_runtime import PrefixReplayFailure, run_prefix_runtime

        def continuation(env, terminal, observers, receipt, *, progress=None):
            previous.check_prefix_inner(receipt["final_prefix_inner"], earlier["runtime"]["final_prefix_inner"])
            start_position = target_position(env, case["spec"]["target"])
            receipt["target_start_position"] = start_position
            receipt["previous_verified_steps"] = 0
            policy = None
            if phase == "checker":
                np.testing.assert_allclose(
                    start_position, producer["runtime"]["target_start_position"], atol=1e-10, rtol=0
                )
                action = previous.checker_action(expected)
            else:
                policy = previous.make_feedback(env, case, selected, gripper, receipt, max_steps=16)
                action = policy.select

            def observed(actual_env, command, offset):
                period = receipt["continuation_partial"]["periods"][-1]
                prefix.require(period["step"] == 360 + offset, "Just-completed matched endpoint")
                history = None
                if policy is not None:
                    history = policy.complete(
                        actual_env, command, offset, [item["q"] for item in period["actual_inner"]]
                    )
                expected_score = None
                if expected is not None:
                    expected_score = expected[offset]["completed_feedback"]["scores"]
                elif offset < 8:
                    expected_score = earlier_periods[offset]["completed_feedback"]["scores"]
                scores = previous.score_completed(
                    actual_env, case, selected, mode, offset, period["tcp_error"], expected=expected_score
                )
                if offset < 8:
                    check_previous_period(period, earlier_periods[offset])
                    receipt["previous_verified_steps"] += 1
                object_position = target_position(actual_env, case["spec"]["target"])
                height_change = float(object_position[2] - start_position[2])
                if expected is not None:
                    earlier_score = expected[offset]["completed_feedback"]
                    np.testing.assert_allclose(object_position, earlier_score["object_position"], atol=1e-10, rtol=0)
                    np.testing.assert_allclose(
                        height_change, earlier_score["object_height_change_m"], atol=1e-10, rtol=0
                    )
                return {
                    "scores": scores,
                    "object_position": object_position,
                    "object_height_change_m": height_change,
                    "policy_history": history,
                    "policy_history_recomputed": policy is not None,
                }

            outcome = run_continuation(
                env,
                terminal,
                observers,
                receipt,
                start_step=360,
                previous_command=case["commands"][-1],
                targets=selected["fixed_targets"],
                tool=case["tool"],
                target_object=case["spec"]["target"],
                choose_action=action,
                stop_on_outside=True,
                expected_periods=expected,
                on_completed=observed,
                progress=progress,
                continuation_limit=16,
            )
            prefix.require(
                receipt["previous_verified_steps"] == 8, "The complete earlier 8-period outcome must reproduce"
            )
            if phase == "checker":
                q, limits, _jac = current_joint_site_state(env)
                margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
                return previous.reconcile_stop(
                    outcome,
                    expected_stop,
                    start_step=360,
                    current_margin=margin,
                    prefix_inner_loss=receipt["final_prefix_inner"]["below_buffer_samples"] > 0,
                    horizon=16,
                )
            return outcome

        def progress(receipt):
            completed = receipt.get("control_steps_completed", 0)
            if completed % 64 == 0 or completed >= 360:
                print(
                    json.dumps(
                        {
                            "phase": phase,
                            "branch": mode,
                            "control_completed": completed,
                            "new_commands": receipt.get("newly_selected_commands", 0),
                            "native_steps": receipt["counters"]["mj_step"],
                        },
                        allow_nan=False,
                    ),
                    flush=True,
                )

        try:
            runtime = run_prefix_runtime(
                case["config"],
                case["record"],
                case["record_xml"],
                case["commands"],
                case["saved"],
                case["receipts"],
                directory / "runtime",
                inputs["asset_refs"],
                case["controller"],
                continuation=continuation,
                continuation_budget=16,
                continuation_limit=16,
                registered_seed=42,
                capture_final_prefix_inner=True,
                progress=progress,
            )
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        result.update(complete=True, passed=True, runtime=runtime)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def execute(phase):
    prefix.environment()
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "Frozen 16-period CPU QA closed",
    )
    before = snapshot(REPO)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "CPU QA source identity")
    inputs, sources = load_inputs(), source_refs()
    prefix.require(
        dry["complete"] is dry["passed"] is True
        and dry["inputs"] == input_receipt(inputs)
        and dry["source_refs"] == sources,
        "Zero-model dry preserves long inputs",
    )
    producer_refs = None
    if phase == "checker":
        producer = prefix.read(prefix.reference(ROOT / "producer/result.json"))
        prefix.require(
            producer["complete"] is producer["passed"] is producer["sources_unchanged"] is True
            and len(producer["branch_refs"]) == 2,
            "Both long producer branches closed",
        )
        producer_refs = dict(zip(BRANCHES, producer["branch_refs"], strict=True))
    phase_dir = ROOT / phase
    phase_dir.mkdir(parents=True, exist_ok=False)
    registration_ref = publish(
        phase_dir / "registration.json",
        {
            "phase": phase,
            "inputs": input_receipt(inputs),
            "source_refs": sources,
            "producer_branch_refs": producer_refs,
        },
    )
    specs = [
        {
            "key": mode,
            "argv": [
                sys.executable,
                "-m",
                "scripts.simulation.runtime.run_causal_event_long_feedback",
                "--worker",
                phase,
                mode,
            ],
            "log": phase_dir / mode / "worker.log",
        }
        for mode in BRANCHES
    ]
    lifecycle = run_owned_workers(specs, cwd=REPO, timeout_seconds=1800)
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    workers_ref = publish(phase_dir / "workers.json", lifecycle)
    available = [
        prefix.reference(phase_dir / mode / "result.json")
        for mode in BRANCHES
        if (phase_dir / mode / "result.json").is_file()
    ]
    result = {
        "complete": False,
        "passed": False,
        "control_qualified": False,
        "phase": phase,
        "registration_ref": registration_ref,
        "workers_ref": workers_ref,
        "available_branch_refs": available,
        "world_model_calls": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
    }
    try:
        prefix.require(
            not lifecycle["launch_error"]
            and len(lifecycle["workers"]) == 2
            and all(
                item["returncode"] == 0 and item["actual_wait"] and item["reaped"] and not item["signal_sent"]
                for item in lifecycle["workers"]
            ),
            "Both long-feedback workers closed and were reaped",
        )
        branch_refs = [prefix.reference(phase_dir / mode / "result.json") for mode in BRANCHES]
        branches = [prefix.read(ref) for ref in branch_refs]
        prefix.require(
            all(
                item["complete"] is item["passed"] is True
                and item["runtime"]["cleanup_complete"] is True
                and item["runtime"]["previous_verified_steps"] == 8
                for item in branches
            ),
            "Earlier 8 periods, both branches and cleanup closed",
        )
        control = sum(item["runtime"]["control_steps_completed"] for item in branches)
        native = sum(item["runtime"]["counters"]["mj_step"]["completed"] for item in branches)
        models = sum(item["runtime"]["counters"]["model_construct"]["completed"] for item in branches)
        contexts = sum(item["runtime"]["counters"]["context"]["completed"] for item in branches)
        selected = sum(item["runtime"]["newly_selected_commands"] for item in branches)
        prefix.require(
            control <= 752 and native == 25 * control and native <= 18800 and models == contexts == 6,
            "Registered long physical budget",
        )
        counts = {
            "control_steps": control,
            "native_steps": native,
            "model_initializations": models,
            "contexts": contexts,
            "newly_selected_commands": selected,
        }
        if phase == "producer":
            histories = [item["gripper_history"] for item in branches]
            proposals = [item["runtime"].get("gripper_proposals", {}) for item in branches]
            calls = sum(item["planner_calls"] for item in histories) + sum(
                item.get("planner_calls", 0) for item in proposals
            )
            sequences = sum(item["candidate_sequences"] for item in histories) + sum(
                item.get("candidate_sequences", 0) for item in proposals
            )
            refiner_calls = sum(item["runtime"].get("refiner_calls", {}).get("attempted", 0) for item in branches)
            refiner_evals = sum(
                item["runtime"].get("quality_refiner", {}).get("baseline_response_evaluations", 0) for item in branches
            )
            quality_calls = sum(
                item["runtime"].get("quality_refiner", {}).get("quality_calls_attempted", 0) for item in branches
            )
            quality_evals = sum(
                item["runtime"].get("quality_refiner", {}).get("quality_response_evaluations", 0) for item in branches
            )
            prefix.require(
                calls <= 752
                and sequences <= 4933872
                and refiner_calls <= 32
                and refiner_evals <= 15360
                and quality_calls <= 32
                and quality_evals <= 5120,
                "Registered solver budgets",
            )
            counts.update(
                gripper_planner_calls=calls,
                gripper_candidate_sequences=sequences,
                refiner_calls=refiner_calls,
                refiner_response_evaluations=refiner_evals,
                quality_calls=quality_calls,
                quality_response_evaluations=quality_evals,
            )
        else:
            prefix.require(selected == 0, "Long checker only replays saved actions")
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(
            source_refs() == sources and snapshot(REPO)["sources"] == before, "Unchanged long-run sources and inputs"
        )
        result.update(complete=True, passed=True, sources_unchanged=True, branch_refs=branch_refs, counts=counts)
    finally:
        print(publish(phase_dir / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--producer", action="store_true")
    group.add_argument("--checker", action="store_true")
    group.add_argument("--worker", nargs=2, metavar=("PHASE", "BRANCH"))
    args = parser.parse_args()
    prefix.environment()
    if args.dry_run:
        print(
            publish(
                DRY,
                {
                    "complete": True,
                    "passed": True,
                    "source_refs": source_refs(),
                    "inputs": input_receipt(load_inputs()),
                    "model_initializations": 0,
                    "physics_steps": 0,
                    "world_model_calls": 0,
                    "fresh_cases": 0,
                },
            ),
            flush=True,
        )
    elif args.producer or args.checker:
        execute("producer" if args.producer else "checker")
    else:
        phase, mode = args.worker
        prefix.require(phase in ("producer", "checker") and mode in BRANCHES, "Registered long phase and branch")
        worker(phase, mode)


if __name__ == "__main__":
    main()
