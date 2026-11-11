"""Replay recorded arm/gripper actions without importing candidate policies."""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import inverse, matrix_to_rot6d
from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.native_runtime.development_continuation import (
    current_joint_site_state,
    current_robot_dynamics,
    run_continuation,
)
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import prepare_cached_calibration_gripper as binding
from scripts.simulation.runtime import registered_feedback_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_event_feedback as common
from scripts.simulation.runtime.compare_multicase_emergency_retreat import read_array_item
from scripts.verification.wam.training.verify_cached_endpoint_calibration import corrected

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/cached_calibration_replay_repair_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/cached_calibration_replay_repair_20261111.md"
TEST = REPO / "tests/research/native_runtime/test_cached_calibration_gripper_replay.py"
ROOT = REPO / "outputs/cached_calibration_gripper_replay_20261111"
DRY = REPO / "outputs/cached_calibration_gripper_replay_dry_20261111/result.json"
QA = REPO / "outputs/qa/cached_calibration_gripper_replay_20261111/result.json"
BRANCHES = tuple((index, mode) for index in (2, 6, 10) for mode in ("raw", "calibrated"))
FORBIDDEN = common.FORBIDDEN_CHECKER_MODULES + (
    "s2a.research.joint_control.preview.shared_shift_fallback",
    "s2a.research.wam.training.cached_endpoint_calibration",
)


def require_independence():
    imported = [name for name in FORBIDDEN if name in sys.modules]
    prefix.require(not imported, f"checker imported candidate policy modules: {imported}")


def configured_ref(config, key):
    return {"path": str(REPO / config[key]), "sha256": config[key + "_sha256"]}


def source_refs():
    return binding.source_refs() + [prefix.reference(path) for path in (CONFIG, PLAN, TEST, Path(__file__))]


def branch_key(index, mode):
    prefix.require((index, mode) in BRANCHES, "registered original branch required")
    return f"case_{index:02d}_{mode}"


def load_inputs():
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    prefix.require(
        config["schema"] == "s2a.cached-calibration-replay-repair.v1"
        and config["output"] == "outputs/cached_calibration_gripper_replay_20261111"
        and config["seed"] == 43
        and config["control_steps"] == 148
        and config["native_steps"] == 3700
        and config["new_commands"] == 0
        and config["workers"] == 6
        and config["world_model_calls"] == config["fresh_cases"] == 0
        and config["automatic_retry"] is False,
        "fixed independent replay budget required",
    )
    producer_ref, failed_ref = configured_ref(config, "producer"), configured_ref(config, "failed_checker")
    producer, failed = prefix.read(producer_ref), prefix.read(failed_ref)
    prefix.require(
        producer["complete"] is producer["passed"] is producer["sources_unchanged"] is True
        and producer["phase"] == "producer"
        and len(producer["branch_refs"]) == 6
        and producer["counts"]["control_steps"] == 148
        and producer["counts"]["native_steps"] == 3700
        and producer["counts"]["newly_selected_commands"] == 20
        and failed["complete"] is failed["passed"] is False
        and failed["phase"] == "checker"
        and failed["available_branch_refs"] == [],
        "completed original producer and failed pre-physics checker required",
    )
    failed_workers = prefix.read(failed["workers_ref"])["workers"]
    prefix.require(
        len(failed_workers) == 6
        and all(
            w["returncode"] == 1 and w["actual_wait"] and w["reaped"] and not w["signal_sent"] for w in failed_workers
        ),
        "original checker failure lifecycle required",
    )
    old_registration = prefix.read(producer["registration_ref"])
    original_inputs = binding.load_inputs()
    prefix.require(old_registration["inputs"] == binding.input_receipt(original_inputs), "original matched inputs")
    for ref in old_registration["source_refs"]:
        prefix.checked(ref)
    originals = {}
    for identity, ref in zip(BRANCHES, producer["branch_refs"], strict=True):
        result = prefix.read(ref)
        prefix.require(
            result["complete"] is result["passed"] is True
            and (result["case"]["case_index"], result["branch"]) == identity,
            "exact original completed branch order",
        )
        originals[branch_key(*identity)] = ref
    refs = dict(original_inputs["source_sha256"])
    for ref in (config_ref, producer_ref, failed_ref, producer["registration_ref"], failed["workers_ref"]):
        prefix.checked(ref, refs)
    for ref in old_registration["source_refs"] + list(originals.values()):
        prefix.checked(ref, refs)
    require_independence()
    return dict(
        config=config,
        config_ref=config_ref,
        producer_ref=producer_ref,
        failed_ref=failed_ref,
        originals=originals,
        original_inputs=original_inputs,
        source_sha256=refs,
    )


def receipt(inputs):
    return dict(
        config_ref=inputs["config_ref"],
        producer_ref=inputs["producer_ref"],
        failed_ref=inputs["failed_ref"],
        originals=inputs["originals"],
        original_inputs=binding.input_receipt(inputs["original_inputs"]),
        source_sha256=inputs["source_sha256"],
    )


def independent_references(chunk, candidate, config):
    anchor = np.asarray(chunk["anchor_tcp"], dtype=float)
    raw, opening = np.asarray(chunk["effective_targets_tcp"], dtype=float), np.asarray(chunk["openings"], dtype=float)
    state = np.r_[anchor[:3, 3], matrix_to_rot6d(anchor[:3, :3]), chunk["anchor_opening"]]
    position, rotation, adjusted = corrected(state, raw[:, :3, 3], raw[:, :3, :3], opening, candidate, config)
    calibrated = raw.copy()
    calibrated[:, :3, 3], calibrated[:, :3, :3] = position, rotation
    return dict(
        anchor=anchor,
        anchor_opening=float(chunk["anchor_opening"]),
        raw=dict(poses=raw, openings=opening),
        calibrated=dict(poses=calibrated, openings=adjusted),
    )


def independent_scores(references, actual_tcp, actual_opening, offset):
    scores = {}
    for mode in ("raw", "calibrated"):
        target = np.asarray(references[mode]["poses"][offset])
        rotation_error = Rotation.from_matrix(target[:3, :3] @ actual_tcp[:3, :3].T).as_rotvec()
        opening_target = float(references[mode]["openings"][offset])
        scores[mode] = dict(
            tcp_error=np.r_[target[:3, 3] - actual_tcp[:3, 3], rotation_error],
            opening_error=float(actual_opening - opening_target),
            opening_target=opening_target,
        )
    return scores


def compare_scores(actual, expected):
    for mode in ("raw", "calibrated"):
        for field in ("tcp_error", "opening_error", "opening_target"):
            np.testing.assert_allclose(actual[mode][field], expected[mode][field], atol=1e-10, rtol=0)


def worker(index, mode):
    prefix.environment()
    require_independence()
    registration = prefix.read(prefix.reference(ROOT / "registration.json"))
    inputs = load_inputs()
    prefix.require(
        registration["inputs"] == receipt(inputs) and registration["source_refs"] == source_refs(),
        "frozen replay inputs",
    )
    original_inputs = inputs["original_inputs"]
    entry, case = original_inputs["cases"][index], original_inputs["cases"][index]["case"]
    original = prefix.read(inputs["originals"][branch_key(index, mode)])
    expected = original["runtime"]["continuation"]["periods"]
    expected_stop = original["runtime"]["continuation"]["stop"]
    chunk = read_array_item(prefix.checked(entry["gripper_refs"]["chunks"]), case["spec"]["prefix_steps"] // 8)
    binding.validate_chunk(case, entry["group"], chunk)
    candidate = prefix.read(original_inputs["candidate_ref"])
    references = independent_references(chunk, candidate, prefix.read(candidate["config_ref"]))
    for label in ("raw", "calibrated"):
        for field in ("poses", "openings"):
            np.testing.assert_allclose(
                references[label][field], original["references"][label][field], atol=1e-10, rtol=0
            )
    result = dict(
        complete=False,
        passed=False,
        case=case["spec"],
        branch=mode,
        original_ref=inputs["originals"][branch_key(index, mode)],
        control_qualified=False,
    )
    try:
        cv2 = original_inputs["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        from s2a.research.native_runtime.development_prefix_runtime import PrefixReplayFailure, run_prefix_runtime
        from s2a.research.native_runtime.task2_prefix_binding import IsolatedTask2PrefixRuntime

        def continuation(env, terminal, observers, run_receipt, *, progress=None):
            require_independence()
            common.check_prefix_inner(run_receipt["final_prefix_inner"], original["runtime"]["final_prefix_inner"])
            np.testing.assert_allclose(
                common.measured_gripper(env)["opening"], references["anchor_opening"], atol=1e-10, rtol=0
            )
            action = common.checker_action(expected)

            def completed(actual_env, _command, offset):
                period = run_receipt["continuation_partial"]["periods"][-1]
                _q, _limits, jac = current_joint_site_state(actual_env)
                _mass, _velocity, site = current_robot_dynamics(actual_env, jac)
                actual_tcp = site @ inverse(np.asarray(case["controller"]["source_from_site"])) @ case["tool"]
                gripper = common.measured_gripper(actual_env)
                scores = independent_scores(references, actual_tcp, gripper["opening"], offset)
                np.testing.assert_allclose(scores[mode]["tcp_error"], period["tcp_error"], atol=1e-10, rtol=0)
                recorded = expected[offset]["completed_feedback"]
                compare_scores(scores, recorded["scores"])
                for field in ("opening", "qpos", "qvel", "motor_target"):
                    np.testing.assert_allclose(gripper[field], recorded["measured_gripper"][field], atol=1e-10, rtol=0)
                return dict(scores=scores, measured_gripper=gripper, policy_history_recomputed=False)

            selected = binding.raw_reference(entry["group"], case)
            outcome = run_continuation(
                env,
                terminal,
                observers,
                run_receipt,
                start_step=case["spec"]["prefix_steps"],
                previous_command=case["commands"][-1],
                targets=references[mode]["poses"][:8],
                tool=case["tool"],
                target_object=selected["intervention_spec"]["target"],
                choose_action=action,
                stop_on_outside=True,
                expected_periods=expected,
                on_completed=completed,
                progress=progress,
            )
            q, limits, _ = current_joint_site_state(env)
            margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
            return common.reconcile_stop(
                outcome,
                expected_stop,
                start_step=case["spec"]["prefix_steps"],
                current_margin=margin,
                prefix_inner_loss=run_receipt["final_prefix_inner"]["below_buffer_samples"] > 0,
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
                ROOT / branch_key(index, mode) / "runtime",
                original_inputs["asset_refs"],
                case["controller"],
                continuation=continuation,
                continuation_budget=8,
                registered_seed=43,
                capture_final_prefix_inner=True,
            )
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        require_independence()
        prefix.require(
            runtime["newly_selected_commands"] == 0 and runtime["cleanup_complete"] is True,
            "replay-only actions and cleanup",
        )
        result.update(complete=True, passed=True, runtime=runtime, candidate_modules_unloaded=True)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(ROOT / branch_key(index, mode) / "result.json", result), flush=True)


def dry_run():
    require_independence()
    inputs = load_inputs()
    print(
        publish(
            DRY,
            dict(
                complete=True,
                passed=True,
                inputs=receipt(inputs),
                source_refs=source_refs(),
                model_initializations=0,
                physics_steps=0,
            ),
        )
    )


def execute():
    require_independence()
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "passing independent CPU QA required",
    )
    before = snapshot(REPO)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "QA source identity")
    inputs, sources = load_inputs(), source_refs()
    prefix.require(
        dry["complete"] is dry["passed"] is True and dry["inputs"] == receipt(inputs) and dry["source_refs"] == sources,
        "exact dry identity",
    )
    prefix.require(not ROOT.exists(), "new independent replay output required")
    registration = publish(ROOT / "registration.json", dict(inputs=receipt(inputs), source_refs=sources))
    specs = [
        dict(
            key=branch_key(*branch),
            argv=[
                sys.executable,
                "-m",
                "scripts.simulation.runtime.replay_cached_calibration_gripper",
                "--worker",
                str(branch[0]),
                branch[1],
            ],
            log=ROOT / branch_key(*branch) / "worker.log",
        )
        for branch in BRANCHES
    ]
    lifecycle = run_owned_workers(specs, cwd=REPO, timeout_seconds=inputs["config"]["worker_timeout_seconds"])
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    workers = publish(ROOT / "workers.json", lifecycle)
    available = [
        prefix.reference(ROOT / key["key"] / "result.json")
        for key in specs
        if (ROOT / key["key"] / "result.json").exists()
    ]
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        registration_ref=registration,
        workers_ref=workers,
        available_branch_refs=available,
        new_commands=0,
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        registered_feedback_phase.require_closed_workers(lifecycle, [spec["key"] for spec in specs])
        refs = [prefix.reference(ROOT / branch_key(*branch) / "result.json") for branch in BRANCHES]
        branches = [prefix.read(ref) for ref in refs]
        prefix.require(
            all(b["complete"] is b["passed"] is b["candidate_modules_unloaded"] is True for b in branches),
            "all independent branches closed",
        )
        runtimes = [b["runtime"] for b in branches]
        counts = dict(
            control_steps=sum(r["control_steps_completed"] for r in runtimes),
            native_steps=sum(r["counters"]["mj_step"]["completed"] for r in runtimes),
            new_commands=sum(r["newly_selected_commands"] for r in runtimes),
            model_initializations=sum(r["counters"]["model_construct"]["completed"] for r in runtimes),
            contexts=sum(r["counters"]["context"]["completed"] for r in runtimes),
        )
        prefix.require(
            counts["control_steps"] == 148 and counts["native_steps"] == 3700 and counts["new_commands"] == 0,
            "exact physical replay denominator",
        )
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(source_refs() == sources and snapshot(REPO)["sources"] == before, "source bytes changed")
        result.update(complete=True, passed=True, source_unchanged=True, branch_refs=refs, counts=counts)
    finally:
        print(publish(ROOT / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--worker", nargs=2, metavar=("CASE", "BRANCH"))
    args = parser.parse_args()
    prefix.environment()
    if args.dry_run:
        dry_run()
    elif args.execute:
        execute()
    else:
        index, mode = args.worker
        worker(int(index), mode)


if __name__ == "__main__":
    main()
