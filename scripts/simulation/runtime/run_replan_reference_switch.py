"""Compare the two references available at one exact replan endpoint."""

import argparse
import json
import signal
import sys
from pathlib import Path

import numpy as np

from s2a.geometry import inverse, pose_error
from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.joint_control.tracking.replan_reference_scoring import score_reference_pair
from s2a.research.native_runtime.development_continuation import (
    current_joint_site_state,
    current_robot_dynamics,
    run_continuation,
)
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import compare_multicase_emergency_retreat as previous
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_combined_warning_feedback as base

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/replan_reference_switch_20261110.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/replan_reference_switch_20261110.md"
QA = REPO / "outputs/qa/replan_reference_switch_20261110/result.json"
DRY = REPO / "outputs/replan_reference_switch_dry_20261110/result.json"
ROOT = REPO / "outputs/replan_reference_switch_20261110"
PRIOR_PRODUCER = base.PRIOR_PRODUCER
PRIOR_LONG = {
    "path": str(REPO / "outputs/combined_warning_long_feedback_20261110/checker/result.json"),
    "sha256": "2166f517817c637db1a2e40bf7095678add714222b441e4604ea889e54846216",
}
BRANCHES = ((2, "new"), (2, "old"), (6, "new"), (6, "old"))
TARGETS = {2: "akita_black_bowl_1", 6: "wine_bottle_1"}
CURSORS = {"old": 8, "new": 0}


def source_refs():
    paths = (
        Path(__file__),
        CONFIG,
        PLAN,
        REPO / "src/s2a/research/joint_control/tracking/replan_reference_scoring.py",
        REPO / "tests/research/joint_control/tracking/test_replan_reference_scoring.py",
        REPO / "tests/research/native_runtime/test_replan_reference_switch_inputs.py",
    )
    return base.source_refs(longer=True) + [prefix.reference(path) for path in paths]


def load_inputs():
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    prefix.require(
        config
        == {
            "schema": "s2a.replan-reference-switch.v1",
            "cases": [2, 6],
            "sampling_seed": 43,
            "prefix_steps": 16,
            "old_chunk_index": 1,
            "old_cursor": 8,
            "new_chunk_index": 2,
            "new_cursor": 0,
            "branches": ["new", "old"],
            "continuation_steps": 8,
            "control_dt_s": 0.05,
            "joint_buffer_rad": 0.05,
            "position_cap_m": 0.01,
            "rotation_cap_rad": 0.05,
            "arm_rate_limit": 0.25,
            "max_control_steps_per_phase": 96,
            "max_native_steps_per_phase": 2400,
            "max_models_per_phase": 12,
            "max_contexts_per_phase": 12,
            "max_refiner_calls_per_phase": 32,
            "max_refiner_response_evaluations_per_phase": 15360,
            "worker_timeout_seconds": 1800,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
        "Exact registered reference-switch comparison",
    )
    for ref in (PRIOR_PRODUCER, PRIOR_LONG):
        earlier = prefix.read(ref)
        prefix.require(
            earlier["complete"] is earlier["passed"] is earlier["sources_unchanged"] is True,
            "Earlier feedback evidence closed",
        )
    inputs = prefix.load_inputs({2: 16, 6: 16})
    refs = inputs["source_sha256"]
    refs.update(
        {
            str(CONFIG): config_ref["sha256"],
            PRIOR_PRODUCER["path"]: PRIOR_PRODUCER["sha256"],
            PRIOR_LONG["path"]: PRIOR_LONG["sha256"],
        }
    )
    review = prefix.read(prefix.REVIEW)
    source_map = review["source_and_artifact_sha256"]
    for case in inputs["cases"]:
        index, n = case["spec"]["case_index"], case["spec"]["prefix_steps"]
        prefix.require(index in TARGETS and n == 16, "Exact two original cases and completed prefixes")
        folder = Path(review["cases"][index]["mechanical"]["case"])
        prefix.require(folder.is_relative_to(REPO / "outputs"), "Reviewed local development directory")
        chunk_path, tool_path, rows_path = (
            folder / "chunks.json",
            Path(case["config"]["tool_contract"]),
            folder / "steps.jsonl",
        )
        for path in (chunk_path, tool_path, rows_path):
            ref = prefix.reference(path)
            prefix.require(source_map.get(ref["path"]) == ref["sha256"], "Reviewed chunk, tool, and prefix identities")
            refs[ref["path"]] = ref["sha256"]
        rows = prefix.read_recorded_prefix(rows_path, n, 600)
        old, new = previous.read_array_item(chunk_path, 1), previous.read_array_item(chunk_path, 2)
        prefix.require(old["step"] == 8 and new["step"] == 16, "Both chunks existed by the current endpoint")
        chunks = {"old": np.asarray(old["effective_targets_tcp"]), "new": np.asarray(new["effective_targets_tcp"])}
        anchors = {"old": np.asarray(old["anchor_tcp"]), "new": np.asarray(new["anchor_tcp"])}
        tool = np.asarray(json.loads(tool_path.read_text())["source_from_tcp"])
        prefix.require(
            all(value.shape == (40, 4, 4) and np.isfinite(value).all() for value in chunks.values())
            and all(value.shape == (4, 4) and np.isfinite(value).all() for value in anchors.values())
            and tool.shape == (4, 4)
            and np.isfinite(tool).all(),
            "Finite full reference chunks and transforms",
        )
        measured_tcp = np.asarray(case["saved"]["source_pose"][-1]) @ tool
        np.testing.assert_array_equal(anchors["new"], measured_tcp)
        prefix.require(rows[-1]["chunk_index"] == 1 and rows[-1]["chunk_physical_step"] == 7, "Completed old chunk")
        np.testing.assert_allclose(rows[-1]["desired_tcp"], chunks["old"][7], atol=1e-10, rtol=0)
        old_row = rows[-1]
        case.update(
            physical_folder=folder,
            reference_chunks=chunks,
            reference_anchors=anchors,
            scoring_targets={"old": chunks["old"][8:16].copy(), "new": chunks["new"][:8].copy()},
            tool=tool,
            completed_history=base.completed_from_prefix(rows, case["commands"]),
            previous_evidence={
                "site": old_row["qualified_forward_input"]["x_site_pose"][0],
                "twist": old_row["qualified_forward_input"]["x_site_twist"][0],
                "coupled_prediction": old_row["coupled_response_history"]["commit"]["nominal_response"],
                "tracking_prediction": old_row["tracking_refinement"]["nominal_response"],
            },
        )
    return inputs


def selected_case(case, mode):
    """Pass exactly one already-generated reference to the feedback policy."""
    prefix.require(mode in CURSORS, "Registered reference choice")
    selected = {
        key: value
        for key, value in case.items()
        if key not in ("reference_chunks", "reference_anchors", "scoring_targets")
    }
    cursor = CURSORS[mode]
    selected.update(
        targets=case["reference_chunks"][mode],
        anchor=case["reference_anchors"][mode],
        fixed_targets=case["scoring_targets"][mode],
        intervention_spec={"cursor": cursor, "target": TARGETS[case["spec"]["case_index"]]},
    )
    return selected


def completed_reference_scores(actual_tcp, scoring_targets, mode, offset, own_error, *, expected=None):
    """Score the completed actual endpoint; references never choose an action here."""
    prefix.require(mode in CURSORS and set(scoring_targets) == set(CURSORS), "Both aligned scoring references required")
    scores = score_reference_pair(actual_tcp, scoring_targets["old"], scoring_targets["new"], offset)
    np.testing.assert_allclose(scores[mode], own_error, atol=1e-10, rtol=0)
    result = {"actual_tcp_pose": np.asarray(actual_tcp).copy(), "reference_errors": scores}
    if expected is not None:
        np.testing.assert_allclose(result["actual_tcp_pose"], expected["actual_tcp_pose"], atol=1e-10, rtol=0)
        for name in CURSORS:
            np.testing.assert_allclose(scores[name], expected["reference_errors"][name], atol=1e-10, rtol=0)
    return result


def branch_key(index, mode):
    prefix.require((index, mode) in BRANCHES, "Fixed registered reference branch")
    return f"case_{index:02d}_{mode}"


def worker(phase, index, mode):
    def interrupted(signum, _frame):
        raise InterruptedError(f"Worker received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    directory = ROOT / phase / branch_key(index, mode)
    registration = prefix.read(prefix.reference(ROOT / phase / "registration.json"))
    inputs = load_inputs()
    prefix.require(
        inputs["source_sha256"] == registration["source_sha256"]
        and inputs["asset_refs"] == registration["asset_refs"]
        and source_refs() == registration["source_refs"],
        "Frozen branch inputs and sources",
    )
    case = next(item for item in inputs["cases"] if item["spec"]["case_index"] == index)
    selected = selected_case(case, mode)
    expected, expected_stop = None, None
    if phase == "checker":
        original = prefix.read(prefix.reference(ROOT / "producer" / branch_key(index, mode) / "result.json"))
        prefix.require(original["complete"] is original["passed"] is True, "Closed producer branch required")
        expected = original["runtime"]["continuation"]["periods"]
        expected_stop = original["runtime"]["continuation"]["stop"]
    result = {
        "complete": False,
        "passed": False,
        "phase": phase,
        "case_index": index,
        "branch": mode,
        "automatic_retry": False,
    }
    try:
        cv2 = inputs["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        from s2a.research.native_runtime.development_prefix_runtime import PrefixReplayFailure, run_prefix_runtime

        def continuation(env, terminal, observers, receipt, *, progress=None):
            receipt["preview_calls"] = {"attempted": 0, "completed": 0}
            receipt["refiner_calls"] = {"attempted": 0, "completed": 0}
            receipt["refiner_response_evaluations"] = 0
            policy = None
            if phase == "checker":
                action = base.quality.checker_action(expected)
            else:
                policy = base.quality.baseline.make_refiner(env, selected, receipt)

                def compare_original(decision):
                    row = prefix.read_recorded_prefix(case["physical_folder"] / "steps.jsonl", 17, 600)[-1]
                    prefix.require(row["step"] == 16 and row["chunk_index"] == 2, "Original current decision identity")
                    np.testing.assert_allclose(
                        row["desired_tcp"], case["scoring_targets"]["new"][0], atol=1e-10, rtol=0
                    )
                    return base.first_decision_comparison(decision, row)

                action = base.guarded_action(
                    policy.select, receipt, first_decision_check=compare_original if mode == "new" else None
                )

            def observed(actual_env, command, offset):
                policy_history = None if policy is None else policy.observe(actual_env, command, offset)
                _q, _limits, jacobian = current_joint_site_state(actual_env)
                _mass, _velocity, site = current_robot_dynamics(actual_env, jacobian)
                site_from_tcp = inverse(np.asarray(case["controller"]["source_from_site"])) @ case["tool"]
                actual_tcp = site @ site_from_tcp
                period = receipt["continuation_partial"]["periods"][-1]
                prefix.require(period["step"] == 16 + offset, "Only the just-completed endpoint is scored")
                expected_score = None if expected is None else expected[offset]["completed_feedback"]
                score = completed_reference_scores(
                    actual_tcp, case["scoring_targets"], mode, offset, period["tcp_error"], expected=expected_score
                )
                score["policy_history"] = policy_history
                return score

            outcome = run_continuation(
                env,
                terminal,
                observers,
                receipt,
                start_step=16,
                previous_command=case["commands"][-1],
                targets=selected["fixed_targets"],
                tool=case["tool"],
                target_object=selected["intervention_spec"]["target"],
                choose_action=action,
                stop_on_outside=True,
                expected_periods=expected,
                on_completed=observed,
                progress=progress,
            )
            if phase == "checker":
                return base.reconcile_checker_stop(outcome, expected_stop, mode=mode, start_step=16)
            return outcome

        def progress(receipt):
            print(
                json.dumps(
                    {
                        "phase": phase,
                        "case": index,
                        "branch": mode,
                        "checked_prefix_endpoints": receipt.get("checked_endpoints", 0),
                        "control_completed": receipt.get("control_steps_completed", 0),
                        "new_commands": receipt.get("newly_selected_commands", 0),
                        "refiner": receipt.get("refiner_calls"),
                        "native_steps": receipt["counters"]["mj_step"],
                    },
                    allow_nan=False,
                ),
                flush=True,
            )

        try:
            receipt = run_prefix_runtime(
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
                progress=progress,
            )
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        if phase == "checker":
            prefix.require(
                base.quality.QUALITY_MODULE not in sys.modules
                and base.quality.baseline.POLICY_MODULE not in sys.modules
                and base.quality.baseline.PREVIEW_MODULE not in sys.modules
                and receipt["newly_selected_commands"] == 0
                and receipt["preview_calls"]["attempted"] == receipt["refiner_calls"]["attempted"] == 0,
                "Checker only replays saved actions and scores actual states",
            )
        result.update(
            complete=True,
            passed=True,
            runtime=receipt,
            candidate_policy_imported=base.quality.baseline.POLICY_MODULE in sys.modules,
            preview_module_imported=base.quality.baseline.PREVIEW_MODULE in sys.modules,
            quality_module_imported=base.quality.QUALITY_MODULE in sys.modules,
        )
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def run_phase(phase):
    directory = ROOT / phase
    prefix.require(not directory.exists(), "New phase output only; no automatic retry")
    dry_ref, qa_ref = prefix.reference(DRY), prefix.reference(QA)
    dry, qa = prefix.read(dry_ref), prefix.read(qa_ref)
    prefix.require(dry["complete"] is dry["passed"] is True and dry["physics_steps"] == 0, "Zero-model dry closed")
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "CPU QA closed",
    )
    before = snapshot(REPO)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "Full QA source identity")
    sources, inputs = source_refs(), load_inputs()
    prefix.require(
        dry["source_refs"] == sources
        and dry["source_sha256"] == inputs["source_sha256"]
        and dry["asset_refs"] == inputs["asset_refs"],
        "Dry source and input identity",
    )
    producer_ref = None
    if phase == "checker":
        producer_ref = prefix.reference(ROOT / "producer/result.json")
        producer = prefix.read(producer_ref)
        prefix.require(
            producer["complete"] is producer["passed"] is producer["sources_unchanged"] is True
            and len(producer["branch_refs"]) == 4,
            "Producer must complete before replay",
        )
        for ref in producer["branch_refs"]:
            prefix.checked(ref)
    registration = publish(
        directory / "registration.json",
        {
            "dry_ref": dry_ref,
            "qa_ref": qa_ref,
            "producer_ref": producer_ref,
            "source_refs": sources,
            "source_sha256": inputs["source_sha256"],
            "asset_refs": inputs["asset_refs"],
            "branches": [list(branch) for branch in BRANCHES],
            "max_control_steps": 96,
            "max_native_steps": 2400,
            "max_refiner_calls": 32,
            "max_refiner_response_evaluations": 15360,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
    )
    specs = [
        {
            "key": branch_key(index, mode),
            "argv": [
                sys.executable,
                "-m",
                "scripts.simulation.runtime.run_replan_reference_switch",
                "--worker",
                phase,
                str(index),
                mode,
            ],
            "log": directory / branch_key(index, mode) / "worker.log",
        }
        for index, mode in BRANCHES
    ]
    lifecycle = run_owned_workers(specs, cwd=REPO, timeout_seconds=1800)
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    workers_ref = publish(directory / "workers.json", lifecycle)
    result = {
        "complete": False,
        "passed": False,
        "phase": phase,
        "registration_ref": registration,
        "workers_ref": workers_ref,
        "world_model_calls": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
    }
    try:
        prefix.require(
            not lifecycle["launch_error"]
            and len(lifecycle["workers"]) == 4
            and all(
                worker["returncode"] == 0 and worker["actual_wait"] and worker["reaped"] and not worker["signal_sent"]
                for worker in lifecycle["workers"]
            ),
            "All four workers closed and reaped",
        )
        refs = [prefix.reference(directory / branch_key(index, mode) / "result.json") for index, mode in BRANCHES]
        branches = [prefix.read(ref) for ref in refs]
        prefix.require(all(branch["complete"] is branch["passed"] is True for branch in branches), "Complete branches")
        prefix.require(
            all(prefix.reference(path)["sha256"] == digest for path, digest in inputs["source_sha256"].items())
            and source_refs() == sources
            and snapshot(REPO)["sources"] == before,
            "Sources and inputs unchanged",
        )
        controls = sum(branch["runtime"]["control_steps_completed"] for branch in branches)
        native = sum(branch["runtime"]["counters"]["mj_step"]["completed"] for branch in branches)
        models = sum(branch["runtime"]["counters"]["model_construct"]["completed"] for branch in branches)
        contexts = sum(branch["runtime"]["counters"]["context"]["completed"] for branch in branches)
        refiners = sum(branch["runtime"]["refiner_calls"]["attempted"] for branch in branches)
        evaluations = sum(branch["runtime"]["refiner_response_evaluations"] for branch in branches)
        prefix.require(
            controls <= 96
            and native == controls * 25
            and native <= 2400
            and models <= 12
            and contexts <= 12
            and refiners <= 32
            and evaluations <= 15360,
            "Registered phase budgets",
        )
        if phase == "checker":
            prefix.require(
                refiners == evaluations == 0
                and all(
                    not branch["candidate_policy_imported"]
                    and not branch["preview_module_imported"]
                    and not branch["quality_module_imported"]
                    and branch["runtime"]["newly_selected_commands"] == 0
                    for branch in branches
                ),
                "No independent checker decisions",
            )
        result.update(
            complete=True,
            passed=True,
            sources_unchanged=True,
            branch_refs=refs,
            control_steps=controls,
            native_steps=native,
            model_initializations=models,
            software_contexts=contexts,
            refiner_calls=refiners,
            refiner_response_evaluations=evaluations,
            newly_selected_commands=sum(branch["runtime"]["newly_selected_commands"] for branch in branches),
        )
    finally:
        print(publish(directory / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--produce", action="store_true")
    group.add_argument("--check", action="store_true")
    group.add_argument("--worker", nargs=3, metavar=("PHASE", "CASE", "BRANCH"))
    args = parser.parse_args()
    prefix.environment()
    if args.dry_run:
        prefix.require(not DRY.exists(), "New dry output only")
        inputs = load_inputs()
        print(
            publish(
                DRY,
                {
                    "complete": True,
                    "passed": True,
                    "source_refs": source_refs(),
                    "source_sha256": inputs["source_sha256"],
                    "asset_refs": inputs["asset_refs"],
                    "prefix_steps": [case["spec"]["prefix_steps"] for case in inputs["cases"]],
                    "completed_history_periods": [
                        case["completed_history"].completed_periods for case in inputs["cases"]
                    ],
                    "first_reference_differences": [
                        pose_error(case["scoring_targets"]["new"][0], case["scoring_targets"]["old"][0])
                        for case in inputs["cases"]
                    ],
                    "models": 0,
                    "physics_steps": 0,
                    "world_model_calls": 0,
                    "fresh_cases": 0,
                },
            ),
            flush=True,
        )
    elif args.produce:
        run_phase("producer")
    elif args.check:
        run_phase("checker")
    else:
        phase, index, mode = args.worker
        prefix.require(phase in ("producer", "checker"), "Registered phase")
        worker(phase, int(index), mode)


if __name__ == "__main__":
    main()
