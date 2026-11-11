"""Shadow a fixed close-deferral proposal on reviewed historical trajectories."""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.gripper.defer_close_release import propose_deferred_close
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import prepare_causal_opening_feedback as studied_binding
from scripts.simulation.runtime import registered_feedback_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.verification.reference_ports.check_defer_close_release import independent_proposal, verify_proposal
from scripts.wam.tracking.preview import bind_bounded_release_shadow as binding

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/gripper/defer_close_release_shadow_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/gripper/defer_close_release_shadow_20261111.md"
ROOT = REPO / "outputs/defer_close_release_shadow_20261111"
DRY = REPO / "outputs/defer_close_release_shadow_dry_20261111/result.json"
QA = REPO / "outputs/qa/defer_close_release_shadow_20261111/result.json"


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__),
            REPO / "src/s2a/research/gripper/defer_close_release.py",
            REPO / "scripts/verification/reference_ports/check_defer_close_release.py",
            REPO / "tests/research/gripper/test_defer_close_release.py",
            REPO / "tests/research/gripper/test_defer_close_release_shadow.py",
            Path(binding.__file__),
            Path(studied_binding.__file__),
        )
    ]


def read_config():
    prefix.environment()
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    expected = dict(
        schema="s2a.defer-close-release-shadow.v1",
        output="outputs/defer_close_release_shadow_20261111",
        binding_config="configs/tracking/joint_control/bounded_release_revision_20261111.json",
        binding_config_sha256="1501fcca9c1ccd92c38137b2348f9643fa40c2d8eb2eeef9934a920182e1b7d6",
        studied_producer="outputs/causal_opening_event_feedback_20261111/producer/result.json",
        studied_producer_sha256="411b2531393ade8964c63b137d88f3d185d66857ca5483f9e2f4c61a91964f97",
        studied_checker="outputs/causal_opening_event_feedback_20261111/checker/result.json",
        studied_checker_sha256="e62ebbf826cef61463209a7f5766a3456589aa602144bbd6082aefa03df9af8c",
        correct_cases=18,
        correct_steps=4907,
        development_cases=12,
        development_steps_per_case=600,
        studied_case_index=7,
        studied_branch="latest",
        studied_steps=8,
        preview_steps=6,
        motor_distance_m_per_period=0.01,
        opening_span_m=0.079,
        original_closing_command=1.0,
        deferred_command=0.0,
        max_workers=32,
        worker_timeout_seconds=600,
        world_model_calls=0,
        physics_steps=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    prefix.require(config == expected, "fixed close-deferral shadow protocol required")
    return ref, config


def configured_ref(config, name):
    ref = {"path": str(REPO / config[name]), "sha256": config[name + "_sha256"]}
    prefix.checked(ref)
    return ref


def verify_qa_sources():
    qa = prefix.read(prefix.reference(QA))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True
        and qa["all_tests_passed"] is True
        and qa["skipped_count"] == 0,
        "complete zero-skip close-deferral CPU regression required",
    )
    before = snapshot(REPO)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "CPU regression source identity")
    return before


def find_studied_branch(ref, phase):
    result = prefix.read(ref)
    prefix.require(
        result["phase"] == phase
        and result["complete"] is result["passed"] is result["sources_unchanged"] is True
        and len(result["branch_refs"]) == 4,
        "completed original opening-event phase required",
    )
    matching = []
    for branch_ref in result["branch_refs"]:
        branch = prefix.read(branch_ref)
        if branch["case"]["case_index"] == 7 and branch["branch"] == "latest":
            prefix.require(branch["complete"] is branch["passed"] is True, "complete studied branch")
            matching.append(branch_ref)
    prefix.require(len(matching) == 1, "unique original task1 latest-reference branch required")
    return matching[0]


def load_inputs():
    config_ref, config = read_config()
    binding_ref = configured_ref(config, "binding_config")
    producer_ref, checker_ref = (configured_ref(config, name) for name in ("studied_producer", "studied_checker"))
    cohort = binding.load_inputs()
    prefix.require(
        cohort["config_ref"] == binding_ref
        and len(cohort["cases"]) == 30
        and all(case["family"] == "correct_reference" for case in cohort["cases"][:18])
        and sum(case["steps"] for case in cohort["cases"][:18]) == 4907
        and all(case["family"] == "development" and case["steps"] == 600 for case in cohort["cases"][18:]),
        "full original correct-reference and development denominators required",
    )
    studied = studied_binding.load_inputs()
    case = next(case for case in studied["cases"] if case["spec"]["case_index"] == 7)
    producer_branch = find_studied_branch(producer_ref, "producer")
    checker_branch = find_studied_branch(checker_ref, "checker")
    prefix.require(
        case["spec"]["prefix_steps"] == 232
        and case["spec"]["case_id"] == "task1_init0_s43_terminal058"
        and studied_binding.selected_case(case, "latest")["cursor"] == 0,
        "fixed already-replayed studied release window required",
    )
    return dict(
        config_ref=config_ref,
        config=config,
        cases=cohort["cases"],
        binding_ref=binding_ref,
        studied_producer_ref=producer_ref,
        studied_checker_ref=checker_ref,
        studied_producer_branch_ref=producer_branch,
        studied_checker_branch_ref=checker_branch,
        studied_source_sha256=studied["source_sha256"],
        studied_case=case,
    )


def input_receipt(inputs):
    return {key: value for key, value in inputs.items() if key not in ("config", "studied_case")}


def check_recorded_preview(calculated, recorded):
    preview = calculated["preview"]
    prefix.require(
        preview["guards"] == recorded["guards"]
        and preview["advanced"] == recorded["advanced"]
        and preview["selected_index"] == recorded["selected_index"]
        and preview["preview_proposal"]["selected_index"] == recorded["preview_proposal"]["selected_index"]
        and preview["preview_proposal"]["advanced"] == recorded["preview_proposal"]["advanced"],
        "reconstructed original preview decision differs",
    )
    for key in (
        "reference",
        "original_reference",
        "position_span_m",
        "rotation_span_rad",
        "measured_position_error_m",
        "measured_rotation_error_rad",
    ):
        np.testing.assert_allclose(preview[key], recorded[key], atol=1e-10, rtol=0)
    for key in ("reference", "original_reference"):
        np.testing.assert_allclose(
            preview["preview_proposal"][key], recorded["preview_proposal"][key], atol=1e-10, rtol=0
        )


def analyze_one(poses, openings, offset, measured_tcp, measured_opening, motor, baseline, recorded):
    latch = recorded["latch"]
    prefix.require(
        recorded["command"] == baseline and type(latch["closing_episode"]) is bool,
        "original completed gripper decision required",
    )
    candidate = propose_deferred_close(
        poses, openings, offset, measured_tcp, measured_opening, motor, baseline, latch["closing_episode"]
    )
    checked = independent_proposal(
        poses, openings, offset, measured_tcp, measured_opening, motor, baseline, latch["closing_episode"]
    )
    difference = verify_proposal(candidate, checked)
    check_recorded_preview(candidate, recorded["preview"])
    np.testing.assert_allclose(latch["measured_opening"], measured_opening, atol=1e-10, rtol=0)
    np.testing.assert_allclose(latch["motor_opening"], candidate["motor_opening"], atol=1e-10, rtol=0)
    return candidate, difference


def shadow_case(item, chunks, rows, tool):
    triggers, reasons, maximum = [], Counter(), 0.0
    for step, row in enumerate(rows):
        query, offset = divmod(step, 8)
        chunk = chunks[query]
        poses, openings = np.asarray(chunk["effective_targets_tcp"]), np.asarray(chunk["openings"])
        measured_tcp = np.asarray(row["measured_source"]) @ tool
        baseline = float(row["command"][6])
        candidate, difference = analyze_one(
            poses,
            openings,
            offset,
            measured_tcp,
            row["opening_before"],
            row["motor_target_before"],
            baseline,
            row["gripper"],
        )
        maximum = max(maximum, difference)
        reasons["closing_command"] += int(baseline == 1.0)
        reasons["deferred"] += int(candidate["deferred"])
        if candidate["deferred"]:
            after = row["opening"] if item["family"] == "correct_reference" else row["opening_after"]
            target = float(openings[offset])
            triggers.append(
                dict(
                    step=step,
                    query=query,
                    offset=offset,
                    baseline_command=baseline,
                    proposed_command=candidate["command"],
                    measured_opening=float(row["opening_before"]),
                    motor_opening=candidate["motor_opening"],
                    raw_opening_target=target,
                    predicted_rise=candidate["predicted_rise"],
                    baseline_opening_after=float(after),
                    baseline_opening_error=float(after - target),
                    baseline_success=bool(row["success"]),
                    baseline_any_object_grasp=any(bool(obj["grasp_contact"]) for obj in row["objects"].values()),
                )
            )
    prefix.require(len(rows) == item["steps"], "complete original case denominator")
    return dict(
        steps=len(rows),
        closing_commands=reasons["closing_command"],
        deferred=reasons["deferred"],
        triggers=triggers,
        max_independent_difference=maximum,
    )


def shadow_studied(inputs):
    case = inputs["studied_case"]
    selected = studied_binding.selected_case(case, "latest")
    original, check = (
        prefix.read(inputs[key]) for key in ("studied_producer_branch_ref", "studied_checker_branch_ref")
    )
    periods, replays = original["runtime"]["continuation"]["periods"], check["runtime"]["continuation"]["periods"]
    prefix.require(len(periods) == len(replays) == 8, "complete independent studied physical branch")
    measured_tcp = np.asarray(case["saved"]["source_pose"][-1]) @ case["tool"]
    measured_opening = float(case["saved"]["opening"][-1])
    motor = np.asarray(case["rows"][-1]["motor_target_after"])
    triggers, maximum = [], 0.0
    for offset, (period, replay) in enumerate(zip(periods, replays, strict=True)):
        np.testing.assert_array_equal(period["command"], replay["command"])
        np.testing.assert_allclose(period["actual_state"], replay["actual_state"], atol=1e-10, rtol=0)
        original_decision = period["decision"]["gripper_decision"]
        candidate, difference = analyze_one(
            selected["targets"],
            selected["openings"],
            offset,
            measured_tcp,
            measured_opening,
            motor,
            float(period["command"][6]),
            original_decision,
        )
        maximum = max(maximum, difference)
        score = period["completed_feedback"]["scores"]
        if candidate["deferred"]:
            triggers.append(
                dict(
                    step=period["step"],
                    offset=offset,
                    baseline_command=float(period["command"][6]),
                    proposed_command=candidate["command"],
                    measured_opening=measured_opening,
                    motor_opening=candidate["motor_opening"],
                    raw_opening_target=float(selected["openings"][offset]),
                    predicted_rise=candidate["predicted_rise"],
                    baseline_opening_after=float(score["gripper"]["opening"]),
                    baseline_opening_error=float(score["opening_errors"]["latest"]),
                    baseline_success=period["task_success"],
                    baseline_target_grasp=period["target_grasp"],
                )
            )
        measured_tcp = np.asarray(score["actual_tcp"])
        measured_opening = float(score["gripper"]["opening"])
        motor = np.asarray(score["gripper"]["motor_target"])
    return dict(steps=8, deferred=len(triggers), triggers=triggers, max_independent_difference=maximum)


def dry_run():
    before = verify_qa_sources()
    inputs = load_inputs()
    receipt, sources = input_receipt(inputs), source_refs()
    prefix.require(snapshot(REPO)["sources"] == before, "dry sources changed")
    print(
        publish(
            DRY,
            dict(
                complete=True,
                passed=True,
                inputs=receipt,
                source_refs=sources,
                world_model_calls=0,
                physics_steps=0,
                fresh_cases=0,
            ),
        )
    )


def worker(index):
    prefix.environment()
    registration = prefix.read(prefix.reference(ROOT / "registration.json"))
    prefix.require(
        registration["source_refs"] == source_refs() and type(index) is int and 0 <= index <= 30,
        "frozen worker source and case",
    )
    if index < 30:
        item = registration["inputs"]["cases"][index]
        chunks, rows, tool = binding.read_case(item)
        summary = shadow_case(item, chunks, rows, tool)
        for ref in item["refs"].values():
            prefix.checked(ref)
        output = dict(
            complete=True,
            passed=True,
            case_id=item["case_id"],
            family=item["family"],
            input_refs=item["refs"],
            **summary,
        )
    else:
        inputs = load_inputs()
        prefix.require(input_receipt(inputs) == registration["inputs"], "frozen studied case inputs")
        output = dict(
            complete=True,
            passed=True,
            case_id="task1_init0_s43_terminal058_latest",
            family="studied",
            **shadow_studied(inputs),
        )
    print(publish(ROOT / f"case_{index:02d}.json", output), flush=True)


def execute():
    before = verify_qa_sources()
    inputs = load_inputs()
    dry = prefix.read(prefix.reference(DRY))
    prefix.require(
        dry["complete"] is dry["passed"] is True
        and dry["inputs"] == input_receipt(inputs)
        and dry["source_refs"] == source_refs(),
        "matching dry inputs",
    )
    prefix.require(not ROOT.exists(), "new non-overwriting shadow output required")
    ROOT.mkdir(parents=True)
    registration_ref = publish(
        ROOT / "registration.json", dict(inputs=input_receipt(inputs), source_refs=source_refs())
    )
    specs = [
        dict(
            key=f"case_{i:02d}",
            argv=[sys.executable, "-m", "scripts.wam.tracking.preview.shadow_defer_close_release", "--worker", str(i)],
            log=ROOT / f"case_{i:02d}.log",
        )
        for i in range(31)
    ]
    lifecycle = run_owned_workers(specs, cwd=REPO, timeout_seconds=inputs["config"]["worker_timeout_seconds"])
    for worker in lifecycle["workers"]:
        worker["log_ref"] = prefix.reference(worker.pop("log"))
    workers_ref = publish(ROOT / "workers.json", lifecycle)
    result = dict(
        complete=False,
        passed=False,
        registration_ref=registration_ref,
        workers_ref=workers_ref,
        world_model_calls=0,
        physics_steps=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        registered_feedback_phase.require_closed_workers(lifecycle, [spec["key"] for spec in specs])
        refs = [prefix.reference(ROOT / f"case_{i:02d}.json") for i in range(31)]
        cases = [prefix.read(ref) for ref in refs]
        prefix.require(
            all(case["complete"] is case["passed"] is True for case in cases)
            and sum(case["steps"] for case in cases[:18]) == 4907
            and sum(case["steps"] for case in cases[18:30]) == 7200
            and cases[30]["steps"] == 8,
            "all registered shadow denominators required",
        )
        fresh_inputs = load_inputs()
        prefix.require(
            input_receipt(fresh_inputs) == input_receipt(inputs)
            and source_refs() == dry["source_refs"]
            and snapshot(REPO)["sources"] == before,
            "shadow sources and inputs changed",
        )
        result.update(
            complete=True,
            passed=True,
            case_refs=refs,
            correct_steps=4907,
            development_steps=7200,
            studied_steps=8,
            correct_deferred=sum(case["deferred"] for case in cases[:18]),
            development_deferred=sum(case["deferred"] for case in cases[18:30]),
            studied_deferred=cases[30]["deferred"],
            max_independent_difference=max(case["max_independent_difference"] for case in cases),
            cases=[{key: case[key] for key in ("case_id", "family", "steps", "deferred")} for case in cases],
            evidence_boundary="Original baseline observations only; proposals were not executed and do not prove grasp retention or physical tracking.",
        )
    finally:
        print(publish(ROOT / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--worker", type=int)
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    elif args.execute:
        execute()
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
