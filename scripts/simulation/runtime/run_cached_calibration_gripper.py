"""Compare frozen raw and calibrated references with real arm/gripper feedback."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from s2a.geometry import inverse, matrix_to_rot6d, pose_error
from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.native_runtime.development_continuation import (
    current_joint_site_state,
    current_robot_dynamics,
    run_continuation,
)
from scripts.simulation.runtime import build_causal_quality_feedback as builder
from scripts.simulation.runtime import prepare_cached_calibration_gripper as binding
from scripts.simulation.runtime import registered_feedback_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_event_feedback as common
from scripts.simulation.runtime.compare_multicase_emergency_retreat import read_array_item
from scripts.simulation.runtime.reconstruct_causal_gripper_history import read_chunk_prefix

REPO = prefix.REPO
ROOT = REPO / "outputs/cached_calibration_gripper_feedback_20261111"
DRY = REPO / "outputs/cached_calibration_gripper_feedback_dry_20261111/result.json"
QA = REPO / "outputs/qa/cached_calibration_gripper_feedback_20261111/result.json"
PROBE = REPO / "outputs/cached_calibration_gripper_binding_probe_20261111/result.json"
TEST = REPO / "tests/research/native_runtime/test_cached_calibration_gripper_runtime.py"
BRANCHES = tuple((index, mode) for index in (2, 6, 10) for mode in ("raw", "calibrated"))
CHECKER_FORBIDDEN = common.FORBIDDEN_CHECKER_MODULES + (
    "s2a.research.joint_control.preview.shared_shift_fallback",
    "s2a.research.wam.training.cached_endpoint_calibration",
)


def source_refs():
    return binding.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            TEST,
            Path(builder.__file__),
            REPO / "src/s2a/research/joint_control/preview/shared_shift_fallback.py",
            REPO / "src/s2a/research/joint_control/preview/loaded_gripper_feedback.py",
            REPO / "src/s2a/research/gripper/causal_loaded_history.py",
            REPO / "src/s2a/research/gripper/loaded_gripper_transaction.py",
        )
    ]


def branch_key(index, mode):
    prefix.require((index, mode) in BRANCHES, "registered matched branch required")
    return f"case_{index:02d}_{mode}"


def paired_references(chunk, candidate, config, *, independent):
    """Only the query's measured anchor and already generated H40 enter calibration."""
    anchor = np.asarray(chunk["anchor_tcp"], dtype=float)
    raw = np.asarray(chunk["effective_targets_tcp"], dtype=float)
    opening = np.asarray(chunk["openings"], dtype=float)
    current_opening = float(chunk["anchor_opening"])
    if independent:
        from scripts.verification.wam.training.verify_cached_endpoint_calibration import corrected

        state = np.r_[anchor[:3, 3], matrix_to_rot6d(anchor[:3, :3]), current_opening]
        position, rotation, adjusted = corrected(state, raw[:, :3, 3], raw[:, :3, :3], opening, candidate, config)
        calibrated = raw.copy()
        calibrated[:, :3, 3], calibrated[:, :3, :3] = position, rotation
    else:
        from s2a.research.wam.training.cached_endpoint_calibration import apply_calibration

        calibrated, adjusted = apply_calibration(
            anchor,
            raw,
            opening,
            current_opening,
            candidate["pose_coefficients"],
            candidate["opening_coefficients"],
            config,
        )
    return dict(
        anchor=anchor,
        anchor_opening=current_opening,
        raw=dict(poses=raw.copy(), openings=opening.copy()),
        calibrated=dict(poses=calibrated, openings=adjusted),
    )


def score_both(references, actual_tcp, actual_opening, offset):
    """Score both references only after the physical period has completed."""
    return {
        mode: dict(
            tcp_error=pose_error(references[mode]["poses"][offset], actual_tcp),
            opening_error=float(actual_opening - references[mode]["openings"][offset]),
            opening_target=float(references[mode]["openings"][offset]),
        )
        for mode in ("raw", "calibrated")
    }


def reconstruct_gripper(entry, receipt):
    """Restore all and only completed gripper decisions before the split."""
    from s2a.research.gripper.causal_loaded_history import reconstruct_loaded_history
    from s2a.research.gripper.loaded_gripper_transaction import LoadedGripperTransaction

    case, refs = entry["case"], entry["gripper_refs"]
    start = case["spec"]["prefix_steps"]
    rows = prefix.read_recorded_prefix(prefix.checked(refs["steps"]), start, 600)
    chunks = read_chunk_prefix(prefix.checked(refs["chunks"]), start // 8)
    controller = prefix.read(refs["controller"])
    parameters = prefix.read(refs["parameters"])
    tool = np.asarray(prefix.read(refs["tool"])["source_from_tcp"])
    np.testing.assert_allclose(tool, case["tool"], atol=1e-12, rtol=0)
    original, details = reconstruct_loaded_history(
        rows,
        chunks,
        tool,
        parameters,
        velocity_envelope=controller["gripper_velocity_envelope"],
        gripper_config=controller["gripper_config"],
    )
    prefix.require(details["last_completed_command"] == float(case["commands"][-1, 6]), "completed gripper command")
    receipt["gripper_history_reconstruction"] = details
    return LoadedGripperTransaction(original)


def make_feedback(env, entry, references, mode, gripper, receipt):
    from s2a.research.joint_control.preview.loaded_gripper_feedback import LoadedGripperFeedback
    from s2a.research.joint_control.preview.shared_shift_fallback import SharedShiftFallbackAudit

    case, group = entry["case"], entry["group"]
    policy_case = binding.previous.policy_case(group, case)
    policy_case["targets"] = references[mode]["poses"].copy()
    adapter = SharedShiftFallbackAudit(receipt)
    arm, history = builder.build_refiner(env, policy_case, receipt, lambda _receipt: adapter)
    adapter.attach_history(arm.tracking.history)
    site_from_tcp = inverse(np.asarray(case["controller"]["source_from_site"])) @ case["tool"]
    return LoadedGripperFeedback(
        env,
        arm,
        gripper,
        history,
        references[mode]["poses"],
        references[mode]["openings"],
        policy_case["intervention_spec"]["cursor"],
        site_from_tcp,
        case["spec"]["prefix_steps"],
        receipt,
        initial_inner_loss=receipt["final_prefix_inner"]["below_buffer_samples"] > 0,
        max_steps=8,
    )


def compare_scores(actual, expected):
    for mode in ("raw", "calibrated"):
        for field in ("tcp_error", "opening_error", "opening_target"):
            np.testing.assert_allclose(actual[mode][field], expected[mode][field], atol=1e-10, rtol=0)


def worker(phase, index, mode):
    prefix.environment()
    key = branch_key(index, mode)
    directory = ROOT / phase / key
    registration = prefix.read(prefix.reference(ROOT / phase / "registration.json"))
    inputs = binding.load_inputs()
    prefix.require(
        registration["inputs"] == binding.input_receipt(inputs) and registration["source_refs"] == source_refs(),
        "fixed branch inputs and source identities",
    )
    entry, case = inputs["cases"][index], inputs["cases"][index]["case"]
    chunk = read_array_item(prefix.checked(entry["gripper_refs"]["chunks"]), case["spec"]["prefix_steps"] // 8)
    binding.validate_chunk(case, entry["group"], chunk)
    candidate = prefix.read(inputs["candidate_ref"])
    calibration_config = prefix.read(candidate["config_ref"])
    refs = paired_references(chunk, candidate, calibration_config, independent=phase == "checker")
    original = None
    if phase == "checker":
        prefix.require(
            all(name not in sys.modules for name in CHECKER_FORBIDDEN), "checker cannot import candidate policies"
        )
        original = prefix.read(registration["producer_branch_refs"][key])
        prefix.require(original["complete"] is original["passed"] is True, "closed producer branch")
        for label in ("raw", "calibrated"):
            for field in ("poses", "openings"):
                np.testing.assert_allclose(refs[label][field], original["references"][label][field], atol=1e-10, rtol=0)
    expected = None if original is None else original["runtime"]["continuation"]["periods"]
    expected_stop = None if original is None else original["runtime"]["continuation"]["stop"]
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        phase=phase,
        case=case["spec"],
        branch=mode,
        references=refs,
        automatic_retry=False,
    )
    try:
        gripper_receipt = {}
        gripper = None if phase == "checker" else reconstruct_gripper(entry, gripper_receipt)
        result["gripper_reconstruction"] = gripper_receipt
        cv2 = inputs["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        from s2a.research.native_runtime.development_prefix_runtime import PrefixReplayFailure, run_prefix_runtime
        from s2a.research.native_runtime.task2_prefix_binding import IsolatedTask2PrefixRuntime

        def continuation(env, terminal, observers, receipt, *, progress=None):
            if original is not None:
                common.check_prefix_inner(receipt["final_prefix_inner"], original["runtime"]["final_prefix_inner"])
            measured = common.measured_gripper(env)
            np.testing.assert_allclose(measured["opening"], refs["anchor_opening"], atol=1e-10, rtol=0)
            policy = None if phase == "checker" else make_feedback(env, entry, refs, mode, gripper, receipt)
            select = common.checker_action(expected) if policy is None else policy.select

            def completed(actual_env, command, offset):
                period = receipt["continuation_partial"]["periods"][-1]
                history = (
                    None
                    if policy is None
                    else policy.complete(actual_env, command, offset, [i["q"] for i in period["actual_inner"]])
                )
                _q, _limits, jac = current_joint_site_state(actual_env)
                _mass, _velocity, site = current_robot_dynamics(actual_env, jac)
                actual_tcp = site @ inverse(np.asarray(case["controller"]["source_from_site"])) @ case["tool"]
                current_gripper = common.measured_gripper(actual_env)
                scores = score_both(refs, actual_tcp, current_gripper["opening"], offset)
                np.testing.assert_allclose(scores[mode]["tcp_error"], period["tcp_error"], atol=1e-10, rtol=0)
                if expected is not None:
                    compare_scores(scores, expected[offset]["completed_feedback"]["scores"])
                    previous_gripper = expected[offset]["completed_feedback"]["measured_gripper"]
                    for field in ("opening", "qpos", "qvel", "motor_target"):
                        np.testing.assert_allclose(current_gripper[field], previous_gripper[field], atol=1e-10, rtol=0)
                return dict(
                    scores=scores,
                    measured_gripper=current_gripper,
                    policy_history=history,
                    policy_history_recomputed=policy is not None,
                )

            selected = binding.raw_reference(entry["group"], case)
            outcome = run_continuation(
                env,
                terminal,
                observers,
                receipt,
                start_step=case["spec"]["prefix_steps"],
                previous_command=case["commands"][-1],
                targets=refs[mode]["poses"][:8],
                tool=case["tool"],
                target_object=selected["intervention_spec"]["target"],
                choose_action=select,
                stop_on_outside=True,
                expected_periods=expected,
                on_completed=completed,
                progress=progress,
            )
            if phase == "checker":
                q, limits, _ = current_joint_site_state(env)
                margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
                outcome = common.reconcile_stop(
                    outcome,
                    expected_stop,
                    start_step=case["spec"]["prefix_steps"],
                    current_margin=margin,
                    prefix_inner_loss=receipt["final_prefix_inner"]["below_buffer_samples"] > 0,
                )
            return outcome

        def progress(receipt):
            count = receipt.get("control_steps_completed", 0)
            if count >= case["spec"]["prefix_steps"]:
                print(
                    json.dumps(
                        dict(
                            phase=phase,
                            case_index=index,
                            mode=mode,
                            control_completed=count,
                            new_commands=receipt.get("newly_selected_commands", 0),
                        )
                    ),
                    flush=True,
                )

        run = run_prefix_runtime if entry["group"] == "replan" else IsolatedTask2PrefixRuntime().run
        try:
            runtime = run(
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
                continuation_budget=8,
                registered_seed=43,
                capture_final_prefix_inner=True,
                progress=progress,
            )
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        unloaded = all(name not in sys.modules for name in CHECKER_FORBIDDEN)
        if phase == "checker":
            prefix.require(unloaded and runtime["newly_selected_commands"] == 0, "checker only replays commands")
        result.update(complete=True, passed=True, runtime=runtime, candidate_modules_unloaded=unloaded)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def summarize_phase(branches, phase, config):
    prefix.require(
        [(r["case"]["case_index"], r["branch"]) for r in branches] == list(BRANCHES)
        and all(r["complete"] is r["passed"] is r["runtime"]["cleanup_complete"] is True for r in branches),
        "all matched physical branches must close",
    )
    runtimes = [r["runtime"] for r in branches]
    counts = dict(
        control_steps=sum(r["control_steps_completed"] for r in runtimes),
        native_steps=sum(r["counters"]["mj_step"]["completed"] for r in runtimes),
        newly_selected_commands=sum(r["newly_selected_commands"] for r in runtimes),
        model_initializations=sum(r["counters"]["model_construct"]["completed"] for r in runtimes),
        contexts=sum(r["counters"]["context"]["completed"] for r in runtimes),
    )
    prefix.require(
        counts["control_steps"] <= config["max_control_steps_per_phase"]
        and counts["native_steps"] <= config["max_native_steps_per_phase"]
        and counts["newly_selected_commands"] <= config["max_new_commands_per_phase"]
        and (phase != "checker" or counts["newly_selected_commands"] == 0),
        "registered physical budget exceeded",
    )
    return counts


def execute(phase):
    registered_feedback_phase.execute_phase(
        phase,
        root=ROOT,
        qa_path=QA,
        dry_path=DRY,
        branches=BRANCHES,
        worker_module="scripts.simulation.runtime.run_cached_calibration_gripper",
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
    prefix.environment()
    if args.dry_run:
        inputs, probe = binding.load_inputs(), prefix.read(prefix.reference(PROBE))
        prefix.require(
            probe["complete"] is probe["passed"] is True and probe["inputs"] == binding.input_receipt(inputs),
            "all three gripper histories reconstructed",
        )
        print(
            publish(
                DRY,
                dict(
                    complete=True,
                    passed=True,
                    inputs=binding.input_receipt(inputs),
                    source_refs=source_refs(),
                    binding_probe_ref=prefix.reference(PROBE),
                    model_initializations=0,
                    physics_steps=0,
                ),
            )
        )
    elif args.producer or args.checker:
        execute("producer" if args.producer else "checker")
    else:
        phase, index, mode = args.worker
        prefix.require(phase in ("producer", "checker"), "known phase required")
        worker(phase, int(index), mode)


if __name__ == "__main__":
    main()
