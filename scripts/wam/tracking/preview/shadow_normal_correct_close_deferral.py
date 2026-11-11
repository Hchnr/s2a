"""Audit close deferral on all ordered normal-gripper correct-reference histories."""

import argparse
import sys
from pathlib import Path

from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import registered_feedback_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.wam.tracking.preview import bind_bounded_release_shadow as reader
from scripts.wam.tracking.preview import shadow_defer_close_release as shadow

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/gripper/defer_close_normal_correct_shadow_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/gripper/defer_close_normal_correct_shadow_20261111.md"
ROOT = REPO / "outputs/defer_close_normal_correct_shadow_20261111"
DRY = REPO / "outputs/defer_close_normal_correct_shadow_dry_20261111/result.json"
QA = REPO / "outputs/qa/defer_close_normal_correct_shadow_v2_20261111/result.json"
ORACLE_ROOT = REPO / "outputs/tracking_correct_reference_ordered_20261027"


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            REPO / "plans/v0.0.1/tracking/gripper/defer_close_normal_shadow_qa_repair_20261111.md",
            Path(__file__),
            Path(reader.__file__),
            Path(shadow.__file__),
            REPO / "src/s2a/research/gripper/defer_close_release.py",
            REPO / "scripts/verification/reference_ports/check_defer_close_release.py",
            REPO / "tests/research/gripper/test_defer_close_normal_correct_shadow.py",
        )
    ]


def read_config():
    prefix.environment()
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    fixed = dict(
        schema="s2a.defer-close-normal-correct-shadow.v1",
        output="outputs/defer_close_normal_correct_shadow_20261111",
        correct_cases=18,
        correct_steps=4907,
        max_workers=18,
        worker_timeout_seconds=600,
        world_model_calls=0,
        physics_steps=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    prefix.require(
        all(config.get(key) == value and type(config.get(key)) is type(value) for key, value in fixed.items()),
        "The fixed normal-controller shadow protocol is required",
    )
    return ref, config


def configured_ref(config, name):
    ref = {"path": str(REPO / config[name]), "sha256": config[name + "_sha256"]}
    prefix.checked(ref)
    return ref


def load_inputs():
    config_ref, config = read_config()
    fixed = {
        name: configured_ref(config, name)
        for name in ("ordered_qualification", "prepared", "prior_release_floor_shadow")
    }
    closed, prepared, prior = (
        prefix.read(fixed[name]) for name in ("ordered_qualification", "prepared", "prior_release_floor_shadow")
    )
    prefix.require(
        closed["complete"] is closed["passed"] is closed["performance_qualified"] is True
        and closed["qualified_cases"] == len(closed["cases"]) == 18
        and closed["candidate_physics_steps"] == closed["replay_physics_steps"] == 4907
        and closed["prepared_ref"] == fixed["prepared"]
        and prepared["complete"] is prepared["passed"] is True
        and len(prepared["profiles"]) == 18
        and prior["complete"] is prior["passed"] is True
        and prior["correct_steps"] == 4907,
        "Complete independently audited ordered matrix and fixed prior shadow required",
    )
    cases = []
    for index, (item, profile) in enumerate(zip(closed["cases"], prepared["profiles"], strict=True)):
        prefix.require(
            item["case_index"] == profile["case_index"] == index, "The original complete case order is fixed"
        )
        native, robot, labels = (
            prefix.read(ref)
            for ref in (item["verified_ref"], item["audit_refs"]["robot_inputs"], item["audit_refs"]["labels_images"])
        )
        result = prefix.read(native["result_ref"])
        count = item["physics_steps"]
        prefix.require(
            native["complete"] is native["passed"] is native["performance_qualified"] is True
            and robot["complete"] is robot["passed"] is labels["complete"] is labels["passed"] is True
            and native["result_ref"] == robot["result_ref"] == labels["result_ref"]
            and result["complete"] is result["passed"] is result["performance_qualified"] is True
            and result["case_index"] == index
            and result["policy"] == "candidate"
            and result["physics_steps"] == robot["rows"] == profile["profile"]["physics_steps"] == count,
            "Each normal-gripper case needs its original execution and three independent audits",
        )
        directory = ORACLE_ROOT / f"case{index:02d}_candidate"
        prefix.require(
            Path(native["result_ref"]["path"]) == directory / "result.json"
            and Path(robot["rows_ref"]["path"]) == directory / "steps.json",
            "Original ordered result and audited rows required",
        )
        refs = dict(
            result=native["result_ref"],
            rows=robot["rows_ref"],
            chunks=prefix.reference(directory / "chunks.json"),
            tool=profile["tool_ref"],
            trajectory=profile["profile"]["trajectory_ref"],
            native_check=item["verified_ref"],
            robot_audit=item["audit_refs"]["robot_inputs"],
            label_audit=item["audit_refs"]["labels_images"],
        )
        for ref in refs.values():
            prefix.checked(ref)
        prefix.require(result["method_contract"]["tool_ref"] == refs["tool"], "Original TCP tool identity required")
        cases.append(
            dict(
                family="correct_reference",
                index=index,
                case_id=f"correct_{index:02d}",
                steps=count,
                format=profile["profile"]["format"],
                refs=refs,
            )
        )
    prefix.require(
        sum(item["steps"] for item in cases) == 4907, "All 4907 original periods must remain in the denominator"
    )
    return dict(config_ref=config_ref, config=config, refs=fixed, cases=cases)


def input_receipt(inputs):
    return {key: inputs[key] for key in ("config_ref", "refs", "cases")}


def verify_qa_sources():
    qa = prefix.read(prefix.reference(QA))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True
        and qa["all_tests_passed"] is True
        and qa["skipped_count"] == 0,
        "Complete zero-skip CPU regression required",
    )
    before = snapshot(REPO)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "CPU regression source identity")
    return before


def aggregate(results):
    prefix.require(
        len(results) == 18
        and all(
            result["complete"] is result["passed"] is True and result["index"] == i for i, result in enumerate(results)
        )
        and sum(result["steps"] for result in results) == 4907,
        "All original cases and periods must close",
    )
    return dict(
        correct_cases=len(results),
        correct_steps=sum(result["steps"] for result in results),
        deferred=sum(result["deferred"] for result in results),
        cases_with_deferral=sum(result["deferred"] > 0 for result in results),
        max_independent_difference=max(result["max_independent_difference"] for result in results),
        triggers=[dict(case_id=result["case_id"], **trigger) for result in results for trigger in result["triggers"]],
    )


def worker(index):
    prefix.environment()
    registration = prefix.read(prefix.reference(ROOT / "registration.json"))
    prefix.require(
        type(index) is int and 0 <= index < 18 and registration["source_refs"] == source_refs(),
        "Frozen worker identity required",
    )
    item = registration["inputs"]["cases"][index]
    chunks, rows, tool = reader.read_case(item)
    data = shadow.shadow_case(item, chunks, rows, tool)
    for trigger in data["triggers"]:
        objects = rows[trigger["step"]]["objects"]
        trigger["grasped_objects"] = sorted(name for name, obj in objects.items() if obj["grasp_contact"])
    for ref in item["refs"].values():
        prefix.checked(ref)
    prefix.require(source_refs() == registration["source_refs"], "Worker sources changed")
    print(
        publish(
            ROOT / f"case_{index:02d}" / "result.json",
            dict(complete=True, passed=True, index=index, case_id=item["case_id"], input_refs=item["refs"], **data),
        ),
        flush=True,
    )


def dry_run():
    before = verify_qa_sources()
    inputs, sources = load_inputs(), source_refs()
    prefix.require(snapshot(REPO)["sources"] == before, "Dry-run sources changed")
    print(
        publish(
            DRY,
            dict(
                complete=True,
                passed=True,
                inputs=input_receipt(inputs),
                source_refs=sources,
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
            ),
        ),
        flush=True,
    )


def execute():
    before = verify_qa_sources()
    inputs, sources = load_inputs(), source_refs()
    dry = prefix.read(prefix.reference(DRY))
    prefix.require(
        dry["complete"] is dry["passed"] is True
        and dry["inputs"] == input_receipt(inputs)
        and dry["source_refs"] == sources,
        "Frozen dry-run identity required",
    )
    ROOT.mkdir(parents=True, exist_ok=False)
    registration = publish(ROOT / "registration.json", dict(inputs=input_receipt(inputs), source_refs=sources))
    specs = [
        dict(
            key=f"case_{i:02d}",
            argv=[
                sys.executable,
                "-m",
                "scripts.wam.tracking.preview.shadow_normal_correct_close_deferral",
                "--worker",
                str(i),
            ],
            log=ROOT / f"case_{i:02d}" / "worker.log",
        )
        for i in range(18)
    ]
    lifecycle = run_owned_workers(specs, cwd=REPO, timeout_seconds=inputs["config"]["worker_timeout_seconds"])
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    workers = publish(ROOT / "workers.json", lifecycle)
    result = dict(
        complete=False,
        passed=False,
        registration_ref=registration,
        workers_ref=workers,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        registered_feedback_phase.require_closed_workers(lifecycle, [f"case_{i:02d}" for i in range(18)])
        refs = [prefix.reference(ROOT / f"case_{i:02d}" / "result.json") for i in range(18)]
        values = aggregate([prefix.read(ref) for ref in refs])
        for item in inputs["cases"]:
            for ref in item["refs"].values():
                prefix.checked(ref)
        for ref in inputs["refs"].values():
            prefix.checked(ref)
        prefix.require(
            source_refs() == sources and snapshot(REPO)["sources"] == before, "Sources changed during shadow execution"
        )
        result.update(complete=True, passed=True, sources_unchanged=True, case_refs=refs, **values)
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
