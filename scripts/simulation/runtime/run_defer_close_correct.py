"""Run and independently replay the fixed correct-reference deferral diagnostic."""

import argparse
import sys
from pathlib import Path

import numpy as np

from s2a.geometry import inverse
from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.native_runtime.development_continuation import (
    current_joint_site_state,
    current_robot_dynamics,
    run_continuation,
)
from scripts.simulation.runtime import prepare_defer_close_correct as binding
from scripts.simulation.runtime import registered_feedback_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_event_feedback as common
from scripts.verification.reference_ports.check_defer_close_release import independent_proposal

REPO = prefix.REPO
ROOT = REPO / "outputs/defer_close_correct_feedback_20261111"
DRY = REPO / "outputs/defer_close_correct_feedback_dry_20261111/result.json"
QA = REPO / "outputs/qa/defer_close_correct_feedback_20261111/result.json"
BRANCHES = ((8, "baseline"), (8, "deferred"))
CANDIDATE_MODULE = "s2a.research.gripper.defer_close_release"


def source_refs():
    return binding.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            REPO / "src/s2a/research/gripper/defer_close_release.py",
            REPO / "src/s2a/research/gripper/coupled_gripper_preview.py",
            REPO / "src/s2a/research/gripper/gripper_reference_preview.py",
            REPO / "src/s2a/research/gripper/gripper_tracking.py",
            REPO / "src/s2a/research/native_runtime/development_continuation.py",
            REPO / "scripts/verification/reference_ports/check_defer_close_release.py",
            REPO / "scripts/wam/tracking/preview/shadow_defer_close_release.py",
            REPO / "tests/research/native_runtime/test_defer_close_correct_feedback.py",
        )
    ]


def branch_key(index, branch):
    prefix.require((index, branch) in BRANCHES, "one registered correct-reference branch required")
    return f"case_{index:02d}_{branch}"


def checker_isolation():
    prefix.require(CANDIDATE_MODULE not in sys.modules, "checker cannot import the candidate proposal")


def diagnostic_command(commands, offset, branch, proposal=None):
    commands = np.asarray(commands, dtype=float)
    if commands.shape != (8, 7) or not np.isfinite(commands).all() or type(offset) is not int or not 0 <= offset < 8:
        raise ValueError("eight fixed finite diagnostic commands required")
    if branch not in ("baseline", "deferred"):
        raise ValueError("known diagnostic branch required")
    command = commands[offset].copy()
    changed = branch == "deferred" and offset == 5
    if changed:
        if (
            command[6] != 1.0
            or not isinstance(proposal, dict)
            or proposal.get("deferred") is not True
            or proposal.get("command") != 0.0
        ):
            raise ValueError("the fixed +1 to 0 proposal is required at offset five")
        command[6] = 0.0
    return command, changed


def current_tcp_and_gripper(env, inputs):
    _q, _limits, jacobian = current_joint_site_state(env)
    _mass, _velocity, site = current_robot_dynamics(env, jacobian)
    tcp = site @ inverse(np.asarray(inputs["controller"]["source_from_site"])) @ inputs["tool"]
    return tcp, common.measured_gripper(env)


def proposal_inputs(env, inputs):
    tcp, gripper = current_tcp_and_gripper(env, inputs)
    row = inputs["baseline_rows"][5]
    np.testing.assert_allclose(
        np.asarray(env.sim.get_state().flatten()), inputs["continuation_states"][4], atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(tcp, np.asarray(row["measured_source"]) @ inputs["tool"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(gripper["opening"], row["opening_before"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(gripper["motor_target"], row["motor_target_before"], atol=1e-10, rtol=0)
    decision = inputs["baseline_intervention_decision"]
    prefix.require(
        decision["command"] == 1.0 and decision["latch"]["closing_episode"] is True,
        "verified original closing decision",
    )
    return (
        inputs["targets"],
        inputs["openings"],
        5,
        tcp,
        gripper["opening"],
        gripper["motor_target"],
        1.0,
        decision["latch"]["closing_episode"],
    )


def score_completed(env, inputs, offset):
    tcp, gripper = current_tcp_and_gripper(env, inputs)
    return dict(
        actual_tcp=tcp,
        measured_gripper=gripper,
        opening_target=float(inputs["openings"][offset]),
        opening_error=float(gripper["opening"] - inputs["openings"][offset]),
    )


def compare_scores(actual, expected):
    for key in ("actual_tcp", "opening_target", "opening_error"):
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-10, rtol=0)
    for key in ("opening", "qpos", "qvel", "motor_target"):
        np.testing.assert_allclose(
            actual["measured_gripper"][key], expected["measured_gripper"][key], atol=1e-10, rtol=0
        )


def check_original_endpoint(period, inputs, offset):
    row = inputs["baseline_rows"][offset]
    np.testing.assert_allclose(period["actual_state"], inputs["continuation_states"][offset], atol=1e-10, rtol=0)
    np.testing.assert_allclose(period["tcp_error"], row["tcp_error"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        period["completed_feedback"]["measured_gripper"]["opening"], row["opening"], atol=1e-10, rtol=0
    )
    prefix.require(
        period["task_success"] == row["success"]
        and period["target_grasp"] == row["objects"][inputs["config"]["target_object"]]["grasp_contact"],
        "original endpoint task and grasp labels must agree",
    )


def worker(phase, index, branch):
    prefix.environment()
    key = branch_key(index, branch)
    directory = ROOT / phase / key
    registration = prefix.read(prefix.reference(ROOT / phase / "registration.json"))
    inputs = binding.load_inputs()
    prefix.require(
        registration["inputs"] == binding.input_receipt(inputs) and registration["source_refs"] == source_refs(),
        "frozen correct-reference inputs and source identity",
    )
    original = None
    expected = None
    if phase == "checker":
        checker_isolation()
        original = prefix.read(registration["producer_branch_refs"][key])
        prefix.require(original["complete"] is original["passed"] is True, "closed original diagnostic branch required")
        expected = original["runtime"]["continuation"]["periods"]
    else:
        from s2a.research.gripper.defer_close_release import propose_deferred_close

    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        phase=phase,
        branch=branch,
        case=inputs["spec"],
        automatic_retry=False,
    )
    try:
        cv2 = inputs["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        from s2a.research.native_runtime.development_prefix_runtime import PrefixReplayFailure
        from s2a.research.native_runtime.task2_seed42_correct_prefix import CorrectTask2Seed42PrefixRuntime

        def continuation(env, terminal, observers, receipt, *, progress=None):
            if phase == "checker":
                checker_isolation()
                common.check_prefix_inner(receipt["final_prefix_inner"], original["runtime"]["final_prefix_inner"])
            prefix_inner_loss = receipt["final_prefix_inner"]["below_buffer_samples"] > 0

            def choose(q, limits, _jacobian, _previous, offset):
                if phase == "checker":
                    checker_isolation()
                    if branch == "deferred" and offset == 5:
                        checked = independent_proposal(*proposal_inputs(env, inputs))
                        prefix.require(
                            checked["deferred"] is True and checked["command"] == 0.0,
                            "independent original-state deferral",
                        )
                    return dict(command=np.asarray(expected[offset]["command"]), newly_selected=False)
                periods = receipt["continuation_partial"]["periods"]
                previous_loss = prefix_inner_loss if not periods else periods[-1]["inner_below_buffer_samples"] > 0
                if previous_loss:
                    margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
                    return dict(
                        stop=dict(reason="observed_joint_buffer_loss", minimum_current_margin_rad=margin),
                        newly_selected=False,
                    )
                proposal = None
                if branch == "deferred" and offset == 5:
                    args = proposal_inputs(env, inputs)
                    proposal = propose_deferred_close(*args)
                    checked = independent_proposal(*args)
                    prefix.require(
                        proposal["deferred"] is checked["deferred"] is True
                        and proposal["command"] == checked["command"] == 0.0,
                        "original-state proposal must match independent fixed rule",
                    )
                command, changed = diagnostic_command(inputs["fixed_continuation_commands"], offset, branch, proposal)
                return dict(command=command, newly_selected=changed, single_diagnostic_override=changed)

            def completed(actual_env, _command, offset):
                period = receipt["continuation_partial"]["periods"][-1]
                scores = score_completed(actual_env, inputs, offset)
                if expected is not None:
                    compare_scores(scores, expected[offset]["completed_feedback"])
                period["completed_feedback"] = scores
                if phase == "producer" and (branch == "baseline" or offset < 5):
                    check_original_endpoint(period, inputs, offset)
                return scores

            outcome = run_continuation(
                env,
                terminal,
                observers,
                receipt,
                start_step=80,
                previous_command=inputs["commands"][-1],
                targets=inputs["targets"][:8],
                tool=inputs["tool"],
                target_object=inputs["config"]["target_object"],
                choose_action=choose,
                stop_on_outside=True,
                expected_periods=expected,
                on_completed=completed,
                progress=progress,
            )
            if phase == "checker":
                q, limits, _jac = current_joint_site_state(env)
                margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
                outcome = common.reconcile_stop(
                    outcome,
                    original["runtime"]["continuation"]["stop"],
                    start_step=80,
                    current_margin=margin,
                    prefix_inner_loss=prefix_inner_loss,
                )
            return outcome

        try:
            runtime = CorrectTask2Seed42PrefixRuntime().run(
                inputs["runtime_config"],
                inputs["record"],
                inputs["record_xml"],
                inputs["commands"],
                inputs["saved"],
                inputs["receipts"],
                directory / "runtime",
                inputs["asset_refs"],
                inputs["controller"],
                continuation=continuation,
                continuation_budget=8,
                registered_seed=42,
                capture_final_prefix_inner=True,
            )
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        if phase == "checker":
            checker_isolation()
            prefix.require(runtime["newly_selected_commands"] == 0, "checker only replays actual commands")
        result.update(
            complete=True, passed=True, runtime=runtime, candidate_modules_unloaded=CANDIDATE_MODULE not in sys.modules
        )
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def summarize_phase(branches, phase, config):
    prefix.require(
        len(branches) == 2 and all(b["complete"] is b["passed"] is True for b in branches),
        "two completed diagnostic branches",
    )
    runtimes = [branch["runtime"] for branch in branches]
    counts = dict(
        control_steps=sum(r["control_steps_completed"] for r in runtimes),
        native_steps=sum(r["counters"]["mj_step"]["completed"] for r in runtimes),
        newly_selected_commands=sum(r["newly_selected_commands"] for r in runtimes),
        model_initializations=sum(r["counters"]["model_construct"]["completed"] for r in runtimes),
        contexts=sum(r["counters"]["context"]["completed"] for r in runtimes),
        continuation_steps=[len(r["continuation"]["periods"]) for r in runtimes],
    )
    prefix.require(
        counts["control_steps"] <= config["max_control_steps_per_phase"]
        and counts["native_steps"] <= config["max_native_steps_per_phase"]
        and counts["newly_selected_commands"] <= (1 if phase == "producer" else 0)
        and all(r["cleanup_complete"] is True and r["checked_endpoints"] == 81 for r in runtimes)
        and (phase != "checker" or all(b["candidate_modules_unloaded"] for b in branches)),
        "bounded completed replay and independent checker required",
    )
    return counts


def dry_run():
    from scripts.wam.tracking.preview.shadow_defer_close_release import analyze_one

    inputs = binding.load_inputs()
    row = inputs["baseline_rows"][5]
    result, difference = analyze_one(
        inputs["targets"],
        inputs["openings"],
        5,
        np.asarray(row["measured_source"]) @ inputs["tool"],
        row["opening_before"],
        row["motor_target_before"],
        row["command"][6],
        row["gripper"],
    )
    prefix.require(result["deferred"] is True and result["command"] == 0.0, "registered original trigger must exist")
    print(
        publish(
            DRY,
            dict(
                complete=True,
                passed=True,
                inputs=binding.input_receipt(inputs),
                source_refs=source_refs(),
                independent_max_abs=difference,
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
            ),
        )
    )


def execute(phase):
    return registered_feedback_phase.execute_phase(
        phase,
        root=ROOT,
        qa_path=QA,
        dry_path=DRY,
        branches=BRANCHES,
        worker_module="scripts.simulation.runtime.run_defer_close_correct",
        branch_key=branch_key,
        load_inputs=binding.load_inputs,
        input_receipt=binding.input_receipt,
        source_refs=source_refs,
        summarize_phase=summarize_phase,
        config_key="config",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--producer", action="store_true")
    group.add_argument("--checker", action="store_true")
    group.add_argument("--worker", nargs=3, metavar=("PHASE", "CASE", "BRANCH"))
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    elif args.producer:
        execute("producer")
    elif args.checker:
        execute("checker")
    else:
        phase, index, branch = args.worker
        prefix.require(phase in ("producer", "checker"), "known diagnostic phase")
        worker(phase, int(index), branch)


if __name__ == "__main__":
    main()
