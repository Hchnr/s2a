"""Reconstruct recorded placement and contacts without advancing physics."""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

from s2a.geometry import inverse
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result
from s2a.research.native_runtime.placement_conditions import placement_conditions
from s2a.research.native_runtime.saved_contacts import target_contacts
from s2a.research.native_runtime.saved_geometry_v2 import reconstruct_endpoint
from s2a.research.wam.training.native.native_training import state_sha256
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import prepare_reset_correct_reference_cohort as preparation
from scripts.simulation.runtime import probe_bowl_native_predicate as geometry
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_correct_reference as previous
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.verification.trajectory_search import audit_seed_policy_first_difference as divergence

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/joint_control/seed_policy_contact_reconstruction_20261116.json"
PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/seed_policy_contact_reconstruction_20261116.md"
INCIDENT = ROOT / "reports/tracking/joint_control/seed_policy_contact_qa_incident_20261116.md"
OUTPUT = ROOT / "outputs/seed_policy_contact_reconstruction_20261116"
DRY = ROOT / "outputs/seed_policy_contact_reconstruction_dry_20261116.json"
QA = ROOT / "outputs/qa/seed_policy_contact_reconstruction_final_20261116/result.json"
MODULE = "scripts.simulation.runtime.reconstruct_seed_policy_contacts"
FIRST = dict(
    path="outputs/seed_policy_first_difference_20261116/result.json",
    sha256="5b9505f95cf2b10d952a99c8ab6a0f8e14f5f7cc072ee5dc59a7faf796c080c4",
)
MAPPING = dict(
    path="outputs/correct_reference_failure_diagnosis_20261116/result.json",
    sha256="fb37341ecfae367a5e8b7c3ace7c6b64b1d82541311675b857f159767945d897",
)
TASKS = {
    6: dict(
        target="cream_cheese_1_main", support="akita_black_bowl_1_main", goal="(On cream_cheese_1 akita_black_bowl_1)"
    ),
    8: dict(target="akita_black_bowl_1_main", support="plate_1_main", goal="(On akita_black_bowl_1 plate_1)"),
    9: dict(
        target="wine_bottle_1_main",
        support="wine_rack_1_main",
        site="wine_rack_1_top_region",
        goal="(On wine_bottle_1 wine_rack_1_top_region)",
    ),
}
PAD_LEFT = "gripper0_finger1_pad_collision"
PAD_RIGHT = "gripper0_finger2_pad_collision"


def validate_config(config):
    expected = dict(
        schema="s2a.seed-policy-contact-reconstruction.v1",
        output="outputs/seed_policy_contact_reconstruction_20261116",
        fast_cohort_audit=divergence.FAST,
        bounded_cohort_audit=divergence.BOUNDED,
        first_difference_audit=FIRST,
        object_mapping_audit=MAPPING,
        candidate_cases=18,
        fast_saved_endpoints=4778,
        bounded_saved_endpoints=4680,
        max_workers=18,
        max_models=18,
        max_data=18,
        max_forwards=9458,
        state_tolerance=1e-10,
        tcp_tolerance_m=1e-9,
        worker_timeout_seconds=1800,
        physics_steps=0,
        render_contexts=0,
        world_model_calls=0,
        gpu_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    if config != expected or any(type(config.get(key)) is not type(value) for key, value in expected.items()):
        raise ValueError("The fixed sources, case roster and static geometry budgets are required")


def source_refs():
    return divergence.source_refs() + [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            INCIDENT,
            Path(__file__).resolve(),
            ROOT / "src/s2a/research/native_runtime/placement_conditions.py",
            ROOT / "src/s2a/research/native_runtime/saved_contacts.py",
            ROOT / "src/s2a/research/native_runtime/saved_geometry_v2.py",
            ROOT / "scripts/simulation/runtime/probe_bowl_native_predicate.py",
            ROOT / "tests/research/native_runtime/test_placement_conditions.py",
            ROOT / "tests/research/native_runtime/test_seed_policy_contact_reconstruction.py",
        )
    ]


def inputs():
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    validate_config(config)
    _old_config, old_identity, audits = divergence.inputs()
    first_ref, mapping_ref = (
        preparation.configured_ref(config[key]) for key in ("first_difference_audit", "object_mapping_audit")
    )
    first, mapping = prefix.read(first_ref), prefix.read(mapping_ref)
    group = prefix.read(audits[0]["producer_ref"])
    cohort = prefix.read(group["registration_ref"])["inputs"]["cohort"]
    prefix.require(
        old_identity["fast_ref"] == preparation.configured_ref(config["fast_cohort_audit"])
        and old_identity["bounded_ref"] == preparation.configured_ref(config["bounded_cohort_audit"])
        and first["complete"] is first["passed"] is first["sources_unchanged"] is True
        and len(first["cases"]) == len(mapping["cases"]) == len(cohort["cases"]) == 18
        and mapping["complete"] is mapping["passed"] is mapping["source_unchanged"] is True
        and sum(row["summary"]["completed_periods"] for row in audits[0]["cases"]) == config["fast_saved_endpoints"]
        and sum(row["summary"]["completed_periods"] for row in audits[1]["cases"]) == config["bounded_saved_endpoints"]
        and all(
            a["case"] == b["case"] == c["spec"]
            for a, b, c in zip(audits[0]["cases"], mapping["cases"], cohort["cases"], strict=True)
        ),
        "All four closed sources must bind the same original 18-case cohort",
    )
    identity = dict(
        config_ref=config_ref,
        fast_ref=old_identity["fast_ref"],
        bounded_ref=old_identity["bounded_ref"],
        first_ref=first_ref,
        mapping_ref=mapping_ref,
        cohort=cohort,
    )
    return config, identity, audits, first, mapping


def first_true(values):
    return next((index for index, value in enumerate(values) if value), None)


def initial_wave_ready(lifecycle):
    """Only a fully reaped, successful three-task wave permits further queries."""
    workers = lifecycle.get("workers", [])
    return bool(
        lifecycle.get("launch_error") is None
        and len(workers) == 3
        and {row.get("key") for row in workers} == {"3", "7", "12"}
        and all(
            row.get("returncode") == 0
            and row.get("actual_wait") is row.get("reaped") is True
            and row.get("signal_sent") is False
            for row in workers
        )
    )


def pair_metrics(left, right):
    """Compare only actual common endpoints and keep missing future separate."""
    count = min(len(left), len(right))
    if not count:
        return dict(common_endpoints=0, object_max_m=None, tcp_max_m=None, opening_max=None)
    a, b = left[:count], right[:count]
    obj = np.linalg.norm(
        np.asarray([r["target_world_m"] for r in a]) - np.asarray([r["target_world_m"] for r in b]), axis=1
    )
    tcp = np.linalg.norm(np.asarray([r["tcp_world_m"] for r in a]) - np.asarray([r["tcp_world_m"] for r in b]), axis=1)
    opening = np.abs(np.asarray([r["opening"] for r in a]) - np.asarray([r["opening"] for r in b]))
    return dict(
        common_endpoints=count,
        object_max_m=float(np.max(obj)),
        object_first_above_1mm=first_true(obj > 0.001),
        object_first_above_10mm=first_true(obj > 0.01),
        tcp_max_m=float(np.max(tcp)),
        opening_max=float(np.max(opening)),
        first_grasp_difference=first_true([x["native_grasp"] != y["native_grasp"] for x, y in zip(a, b, strict=True)]),
        first_support_contact_difference=first_true(
            [
                x["conditions"]["support_contact"] != y["conditions"]["support_contact"]
                for x, y in zip(a, b, strict=True)
            ]
        ),
        first_spatial_difference=first_true(
            [
                x["conditions"]["spatial_satisfied"] != y["conditions"]["spatial_satisfied"]
                for x, y in zip(a, b, strict=True)
            ]
        ),
        first_success_difference=first_true(
            [x["native_success"] != y["native_success"] for x, y in zip(a, b, strict=True)]
        ),
    )


def summary(rows):
    goal_mismatch = [row["step"] for row in rows if row["conditions"]["reconstructed_success"] != row["native_success"]]
    grasp_mismatch = [row["step"] for row in rows if row["bilateral"] != row["native_grasp"]]
    return dict(
        endpoints=len(rows),
        native_success_count=sum(row["native_success"] for row in rows),
        native_grasp_count=sum(row["native_grasp"] for row in rows),
        left_pad_contact_count=sum(row["left_pad_contact"] for row in rows),
        right_pad_contact_count=sum(row["right_pad_contact"] for row in rows),
        support_contact_count=sum(row["conditions"]["support_contact"] for row in rows),
        spatial_satisfied_count=sum(row["conditions"]["spatial_satisfied"] for row in rows),
        first_success=first_true([row["native_success"] for row in rows]),
        first_grasp=first_true([row["native_grasp"] for row in rows]),
        first_support_contact=first_true([row["conditions"]["support_contact"] for row in rows]),
        goal_label_mismatch_steps=goal_mismatch,
        grasp_label_mismatch_steps=grasp_mismatch,
        interpretation_eligible=not goal_mismatch and not grasp_mismatch,
        max_tcp_reconstruction_difference_m=max((row["tcp_reconstruction_difference_m"] for row in rows), default=None),
        max_object_mapping_difference_m=max((row["object_mapping_difference_m"] for row in rows), default=None),
        max_state_normalization_difference=max((row["state_normalization_difference"] for row in rows), default=None),
    )


def check_loaded_native_refs(branches, mujoco):
    """Match the used MuJoCo binaries and native predicate code to both executions."""
    names = (
        "libmujoco.so.2.3.7",
        "_functions.cpython-311-x86_64-linux-gnu.so",
        "_structs.cpython-311-x86_64-linux-gnu.so",
    )
    sources = ("base_object_states.py", "site_object.py", "sim_utils.py", "manipulation_env.py", "panda_gripper.py")
    collected = []
    for branch in branches:
        matches = []
        for key, required in (("loaded_libraries", names), ("loaded_sources", sources)):
            records = branch["runtime"][key]
            for name in required:
                found = [item for item in records if Path(item["path"]).name == name]
                prefix.require(len(found) == 1, f"Exactly one recorded native dependency is required: {name}")
                prefix.checked(found[0])
                matches.append(dict(path=found[0]["path"], sha256=found[0]["sha256"]))
        collected.append(matches)
    prefix.require(collected[0] == collected[1], "Both original runs must use the same native geometry dependencies")
    module_dir = Path(mujoco.__file__).resolve().parent
    prefix.require(
        mujoco.mj_versionString() == "2.3.7"
        and all(Path(ref["path"]).resolve().is_relative_to(module_dir) for ref in collected[0][:3]),
        "The current reconstruction must import the same pinned MuJoCo installation",
    )
    return collected[0]


def reconstruct_case(index, config, identity, audits, first, mapping, counts):
    prefix.require(os.environ.get("MUJOCO_GL") == "disable", "Static geometry must disable all render backends")
    import mujoco

    a, b = (audit["cases"][index] for audit in audits)
    branches = [prefix.read(row["producer_ref"]) for row in (a, b)]
    period_sets = [previous.periods(branch) for branch in branches]
    case_budget = a["summary"]["completed_periods"] + b["summary"]["completed_periods"]
    prefix.require(
        [len(rows) for rows in period_sets] == [a["summary"]["completed_periods"], b["summary"]["completed_periods"]]
        and 0 < case_budget <= 2 * a["case"]["periods"],
        "Each unique case must be bounded by its actual recorded endpoints",
    )
    spec = a["case"]
    prefix.require(spec == b["case"] == identity["cohort"]["cases"][index]["spec"], "Case identities must match")
    prefix.require(
        branches[0]["runtime"]["reset_endpoint_comparison"] == branches[1]["runtime"]["reset_endpoint_comparison"],
        "Both branches must share the same reset",
    )
    native_refs = check_loaded_native_refs(branches, mujoco)
    first_case = prefix.read(first["cases"][str(index)]["result_ref"])
    position_mapping = mapping["cases"][index]["metrics"]["state_position_mapping"]
    columns = position_mapping["columns"]
    prefix.require(
        first_case["case"] == spec
        and position_mapping["maximum_residual"] <= config["state_tolerance"]
        and len(columns) == 3,
        "Audited first difference and object mapping must match",
    )
    shared = preparation.shared_inputs()
    prefix.require(
        shared["receipt"] == identity["cohort"]["shared_receipt"], "Shared frozen simulator assets must match"
    )
    case = preparation.case_inputs(identity["cohort"]["cases"][index], shared)
    xml = case["record_xml"]
    xml_sha = hashlib.sha256(xml.encode()).hexdigest()
    prefix.require(
        all(
            branch["runtime"]["model_records"][2]["stage"] == "record_xml"
            and branch["runtime"]["model_records"][2]["xml_sha256"] == xml_sha
            for branch in branches
        ),
        "The same executed model XML is required",
    )
    task = TASKS[spec["task_id"]]
    for branch in branches:
        bddl_ref = branch["runtime"]["bddl_ref"]
        text = " ".join(prefix.checked(bddl_ref).read_text().split())
        prefix.require(task["goal"] in text, "The original task predicate must match the registered rule")
    target_names = geometry.contact_geom_names(xml, task["target"])
    support_names = set(geometry.contact_geom_names(xml, task["support"]))
    counts["models_attempted"] += 1
    model = mujoco.MjModel.from_xml_string(xml)
    counts["models_completed"] += 1
    counts["data_attempted"] += 1
    data = mujoco.MjData(model)
    counts["data_completed"] += 1
    prefix.require((model.nq, model.nv, model.na) == (41, 37, 0), "The recorded simulator state layout must match")

    def lookup(kind, name):
        return int(mujoco.mj_name2id(model, kind, name))

    site = lookup(mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")
    target = lookup(mujoco.mjtObj.mjOBJ_BODY, task["target"])
    support = lookup(mujoco.mjtObj.mjOBJ_BODY, task["support"])
    goal_site = lookup(mujoco.mjtObj.mjOBJ_SITE, task["site"]) if "site" in task else None
    prefix.require(
        min(site, target, support) >= 0 and (goal_site is None or goal_site >= 0),
        "Named controller and placement geometry must exist",
    )
    reset, controller = case["reset"], case["case"]
    expected_site = np.asarray(reset["expected"]["controller_site_pose"], dtype=float)
    expected_source = np.asarray(reset["expected"]["source_pose"], dtype=float)
    site_from_tcp = inverse(expected_site) @ expected_source @ np.asarray(controller["tool"], dtype=float)
    original_forward = mujoco.mj_forward

    def counted_forward(m, d):
        prefix.require(
            counts["forwards_attempted"] < case_budget, "The case's recorded endpoint query budget is exhausted"
        )
        counts["forwards_attempted"] += 1
        original_forward(m, d)
        counts["forwards_completed"] += 1

    originals = {name: getattr(mujoco, name) for name in ("mj_step", "mj_step1", "mj_step2") if hasattr(mujoco, name)}

    def no_step(*_args, **_kwargs):
        raise ValueError("Recorded-state geometry cannot advance physics")

    for name in originals:
        setattr(mujoco, name, no_step)
    records = []
    try:
        for periods in period_sets:
            rows = []
            for row in periods:
                step = row["step"]
                state = np.asarray(row["actual_state"], dtype=float)
                prefix.require(
                    state_sha256(state) == row["endpoint_receipt"]["sim_state_sha256"], "Saved endpoint SHA must match"
                )
                point = reconstruct_endpoint(
                    model,
                    data,
                    state,
                    site_id=site,
                    body_id=target,
                    site_from_tcp=site_from_tcp,
                    forward=counted_forward,
                )
                contacts = target_contacts(
                    model,
                    data,
                    mujoco=mujoco,
                    body_id=target,
                    target_names=target_names,
                    left_pad=PAD_LEFT,
                    right_pad=PAD_RIGHT,
                )
                support_pairs = [item for item in contacts["other"] if item["other_geom"] in support_names]
                goal = placement_conditions(
                    spec["task_id"],
                    point["body_origin_world_m"],
                    np.asarray(data.xpos[support]).copy(),
                    bool(support_pairs),
                    site_position=None if goal_site is None else np.asarray(data.site_xpos[goal_site]).copy(),
                    site_rotation=None
                    if goal_site is None
                    else np.asarray(data.site_xmat[goal_site]).reshape(3, 3).copy(),
                    site_size=None if goal_site is None else np.asarray(model.site_size[goal_site]).copy(),
                )
                desired = case["timeline"].window(step)[0][0, :3, 3]
                tcp_diff = float(np.max(np.abs(desired - point["tcp_world_m"] - np.asarray(row["tcp_error"][:3]))))
                obj_diff = float(np.max(np.abs(point["body_origin_world_m"] - state[columns])))
                prefix.require(
                    tcp_diff <= config["tcp_tolerance_m"] and obj_diff <= config["state_tolerance"],
                    "Reconstructed TCP and mapped object positions must match",
                )
                rows.append(
                    dict(
                        step=step,
                        state_sha256=row["endpoint_receipt"]["sim_state_sha256"],
                        target_world_m=point["body_origin_world_m"].tolist(),
                        tcp_world_m=point["tcp_world_m"].tolist(),
                        opening=float(row["completed_feedback"]["score"]["opening"]),
                        native_success=row["task_success"],
                        native_grasp=row["target_grasp"],
                        left_pad_contact=bool(contacts["left_pad"]),
                        right_pad_contact=bool(contacts["right_pad"]),
                        bilateral=contacts["bilateral"],
                        support_pairs=support_pairs,
                        conditions=goal,
                        tcp_reconstruction_difference_m=tcp_diff,
                        object_mapping_difference_m=obj_diff,
                        state_normalization_difference=point["state_change"]["max_absolute_change"],
                    )
                )
            records.append(rows)
    finally:
        for name, value in originals.items():
            setattr(mujoco, name, value)
    prefix.require(
        [len(rows) for rows in records] == [a["summary"]["completed_periods"], b["summary"]["completed_periods"]],
        "Every actually saved endpoint must be queried",
    )
    summaries = [summary(rows) for rows in records]
    return dict(
        case=spec,
        fast_qualified=a["qualified"],
        bounded_qualified=b["qualified"],
        first_action_difference=first_case["first_action_difference"],
        fast_ref=a["producer_ref"],
        bounded_ref=b["producer_ref"],
        xml_sha256=xml_sha,
        native_refs=native_refs,
        target_geom_count=len(target_names),
        support_geom_count=len(support_names),
        summaries=dict(fast=summaries[0], bounded=summaries[1]),
        pair=pair_metrics(*records),
        interpretation_eligible=all(row["interpretation_eligible"] for row in summaries),
        endpoints=dict(fast=records[0], bounded=records[1]),
    )


def worker(index):
    prefix.environment()
    prefix.require(os.environ.get("MUJOCO_GL") == "disable", "Static geometry must disable rendering")
    counts = dict(
        models_attempted=0,
        models_completed=0,
        data_attempted=0,
        data_completed=0,
        forwards_attempted=0,
        forwards_completed=0,
    )
    result = dict(
        complete=False,
        passed=False,
        case_index=index,
        counters=counts,
        physics_steps=0,
        render_contexts=0,
        world_model_calls=0,
        gpu_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
        config, identity, audits, first, mapping = inputs()
        prefix.require(
            0 <= index < config["candidate_cases"]
            and identity == registration["inputs"]
            and source_refs() == registration["source_refs"],
            "Only the registered case and sources are admissible",
        )
        original.require_no_simulator_import()
        result.update(reconstruct_case(index, config, identity, audits, first, mapping, counts))
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        print(publish_result(OUTPUT / f"case{index:02d}/result.json", result), flush=True)


def execute():
    prefix.environment()
    prefix.require(os.environ.get("MUJOCO_GL") == "disable", "Static geometry must disable rendering")
    original.require_no_simulator_import()
    qa_ref, dry_ref = prefix.reference(QA), prefix.reference(DRY)
    qa, dry = prefix.read(qa_ref), prefix.read(dry_ref)
    before, sources = snapshot(ROOT)["sources"], source_refs()
    config, identity, _audits, _first, _mapping = inputs()
    prefix.require(
        qa["complete"]
        is qa["passed"]
        is qa["all_tests_passed"]
        is qa["source_unchanged"]
        is qa["actual_wait"]
        is qa["reaped"]
        is True
        and qa["configured_parallelism"] == 64
        and qa["skipped_count"] == 0
        and before == prefix.read(qa["source_before_ref"])["sources"]
        and dry["complete"] is dry["passed"] is True
        and dry["inputs"] == identity
        and dry["source_refs"] == sources
        and dry["models"] == dry["forwards"] == dry["physics_steps"] == 0,
        "Matching source-bound 64-worker QA and a zero-model dry run are required",
    )
    OUTPUT.mkdir(parents=True, exist_ok=False)
    registration_ref = publish_result(
        OUTPUT / "registration.json",
        dict(inputs=identity, source_refs=sources, qa_ref=qa_ref, dry_ref=dry_ref, automatic_retry=False),
    )
    specs = [
        dict(
            key=str(i), argv=[sys.executable, "-m", MODULE, "--worker", str(i)], log=OUTPUT / f"case{i:02d}/worker.log"
        )
        for i in range(config["candidate_cases"])
    ]
    initial = {"3", "7", "12"}
    first_wave = run_owned_workers(
        [spec for spec in specs if spec["key"] in initial], cwd=ROOT, timeout_seconds=config["worker_timeout_seconds"]
    )
    ready = initial_wave_ready(first_wave)
    second_wave = (
        run_owned_workers(
            [spec for spec in specs if spec["key"] not in initial],
            cwd=ROOT,
            timeout_seconds=config["worker_timeout_seconds"],
        )
        if ready
        else dict(workers=[], launch_error="Initial three-case wave did not close")
    )
    lifecycle = dict(
        workers=first_wave["workers"] + second_wave["workers"],
        launch_error=first_wave["launch_error"] or second_wave["launch_error"],
        initial_wave_keys=sorted(initial),
        second_wave_started=ready,
    )
    for child in lifecycle["workers"]:
        child["log_ref"] = prefix.reference(child.pop("log"))
    lifecycle_ref = publish_result(OUTPUT / "workers.json", lifecycle)
    result = dict(
        complete=False,
        passed=False,
        interpretation_eligible=False,
        cases={},
        registration_ref=registration_ref,
        lifecycle_ref=lifecycle_ref,
        physics_steps=0,
        render_contexts=0,
        world_model_calls=0,
        gpu_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        counters = []
        for index in range(config["candidate_cases"]):
            row = result["cases"][str(index)] = dict(status="failed", counters=None)
            path = OUTPUT / f"case{index:02d}/result.json"
            if path.is_file():
                ref = prefix.reference(path)
                data = prefix.read(ref)
                row.update(available_ref=ref, counters=data["counters"])
                counters.append(data["counters"])
                if data.get("complete") is data.get("passed") is True:
                    row.update(status="closed", result_ref=ref)
        keys = (
            "models_attempted",
            "models_completed",
            "data_attempted",
            "data_completed",
            "forwards_attempted",
            "forwards_completed",
        )
        totals = {key: sum(row[key] for row in counters) for key in keys}
        result.update(observed_resource_lower_bound=totals, resource_counts_complete=len(counters) == 18)
        prefix.require(
            totals["models_attempted"] <= config["max_models"]
            and totals["data_attempted"] <= config["max_data"]
            and totals["forwards_attempted"] <= config["max_forwards"],
            "Observed query budget exceeded",
        )
        prefix.require(
            lifecycle["launch_error"] is None
            and len(lifecycle["workers"]) == 18
            and all(
                w["returncode"] == 0 and w["actual_wait"] is w["reaped"] is True and not w["signal_sent"]
                for w in lifecycle["workers"]
            )
            and all(row["status"] == "closed" for row in result["cases"].values()),
            "All fixed workers must close and be reaped",
        )
        rows = [prefix.read(result["cases"][str(i)]["result_ref"]) for i in range(18)]
        _config, after, _audits, _first, _mapping = inputs()
        prefix.require(
            after == identity and source_refs() == sources and snapshot(ROOT)["sources"] == before,
            "Sources and inputs must remain unchanged",
        )
        prefix.require(
            totals["models_attempted"]
            == totals["models_completed"]
            == totals["data_attempted"]
            == totals["data_completed"]
            == 18
            and totals["forwards_attempted"] == totals["forwards_completed"] == config["max_forwards"],
            "All registered static queries must close",
        )
        result.update(
            complete=True,
            passed=True,
            sources_unchanged=True,
            interpretation_eligible=all(row["interpretation_eligible"] for row in rows),
            goal_label_mismatch_count=sum(
                len(s["goal_label_mismatch_steps"]) for row in rows for s in row["summaries"].values()
            ),
            grasp_label_mismatch_count=sum(
                len(s["grasp_label_mismatch_steps"]) for row in rows for s in row["summaries"].values()
            ),
        )
    except Exception as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
    print(publish_result(OUTPUT / "result.json", result), flush=True)
    if not result["passed"]:
        raise SystemExit("The registered reconstruction did not close; no automatic retry")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--worker", type=int)
    args = parser.parse_args()
    prefix.environment()
    prefix.require(os.environ.get("MUJOCO_GL") == "disable", "Static geometry must disable rendering")
    if args.dry:
        original.require_no_simulator_import()
        _config, identity, _audits, _first, _mapping = inputs()
        print(
            publish_result(
                DRY,
                dict(
                    complete=True,
                    passed=True,
                    inputs=identity,
                    source_refs=source_refs(),
                    models=0,
                    forwards=0,
                    physics_steps=0,
                    render_contexts=0,
                    world_model_calls=0,
                    gpu_calls=0,
                    fresh_cases=0,
                ),
            ),
            flush=True,
        )
    elif args.execute:
        execute()
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
