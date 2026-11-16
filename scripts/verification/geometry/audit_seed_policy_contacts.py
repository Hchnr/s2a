"""Independently check recorded contact summaries without MuJoCo queries."""

import argparse
import math
import statistics
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result
from s2a.research.wam.training.native.native_training import state_sha256
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import prepare_reset_correct_reference_cohort as preparation
from scripts.simulation.runtime import probe_bowl_native_predicate as geometry
from scripts.simulation.runtime import reconstruct_seed_policy_contacts as source
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_correct_reference as previous
from scripts.simulation.runtime import run_reset_start_feedback as original

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/joint_control/seed_policy_contact_independent_audit_20261116.json"
PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/seed_policy_contact_independent_audit_20261116.md"
OUTPUT = ROOT / "outputs/seed_policy_contact_independent_audit_20261116"
DRY = ROOT / "outputs/seed_policy_contact_independent_audit_dry_20261116.json"
QA = ROOT / "outputs/qa/seed_policy_contact_independent_audit_final_20261116/result.json"
MODULE = "scripts.verification.geometry.audit_seed_policy_contacts"
RECONSTRUCTION = dict(
    path="outputs/seed_policy_contact_reconstruction_20261116/result.json",
    sha256="498b3d32cb40e1aa65b61e27c1ca091472b4a5bbab07e8b12dbda44c937ee00d",
)


def validate_config(config):
    expected = dict(
        schema="s2a.seed-policy-contact-independent-audit.v1",
        output="outputs/seed_policy_contact_independent_audit_20261116",
        reconstruction=RECONSTRUCTION,
        cases=18,
        endpoints=9458,
        max_workers=18,
        worker_timeout_seconds=1200,
        tolerance=1e-10,
        models=0,
        forwards=0,
        physics_steps=0,
        world_model_calls=0,
        gpu_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    if config != expected or any(type(config.get(k)) is not type(v) for k, v in expected.items()):
        raise ValueError("The fixed reconstruction identity, tolerance and zero-query budget are required")


def source_refs():
    return source.source_refs() + [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__).resolve(),
            ROOT / "tests/research/native_runtime/test_seed_policy_contact_independent_audit.py",
        )
    ]


def inputs():
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    validate_config(config)
    recon_ref = preparation.configured_ref(config["reconstruction"])
    recon = prefix.read(recon_ref)
    registration = prefix.read(recon["registration_ref"])
    _old, previous_identity, audits, first, mapping = source.inputs()
    prefix.require(
        recon["complete"] is recon["passed"] is recon["sources_unchanged"] is True
        and recon["resource_counts_complete"] is True
        and recon["observed_resource_lower_bound"]["forwards_completed"] == config["endpoints"]
        and len(recon["cases"]) == config["cases"]
        and all(row["status"] == "closed" for row in recon["cases"].values())
        and registration["inputs"] == previous_identity
        and registration["source_refs"] == source.source_refs(),
        "The completed reconstruction and its original source identities must remain intact",
    )
    return (
        config,
        dict(config_ref=config_ref, reconstruction_ref=recon_ref, previous_identity=previous_identity),
        recon,
        audits,
        first,
        mapping,
    )


def scalar_conditions(task, row, site_size, tolerance):
    """Recompute saved condition booleans using scalar arithmetic only."""
    c = row["conditions"]
    pairs = row["support_pairs"]
    if type(pairs) is not list or type(c["support_contact"]) is not bool or bool(pairs) != c["support_contact"]:
        raise ValueError("Support contact must equal the retained contact list")
    if task in (6, 8):
        distance, height = float(c["xy_distance_m"]), float(c["height_difference_m"])
        if (
            not all(math.isfinite(x) for x in (distance, height))
            or distance < 0
            or abs(float(c["xy_clearance_m"]) - (0.03 - distance)) > tolerance
        ):
            raise ValueError("Object placement distances and clearances must be finite and consistent")
        spatial = height >= 0 and distance < 0.03
    elif task == 9:
        if site_size is None or len(site_size) != 3:
            raise ValueError("The original wine rack site dimensions are required")
        xyz, widths = list(map(float, c["site_local_m"])), list(map(float, site_size))
        if len(xyz) != 3 or not all(math.isfinite(x) for x in xyz + widths) or min(widths) <= 0:
            raise ValueError("Finite wine site coordinates and dimensions are required")
        clear = (
            widths[0] - abs(xyz[0]),
            widths[1] - abs(xyz[1]),
            xyz[2] - (widths[2] - 0.005),
            widths[2] + 0.1 - xyz[2],
        )
        recorded = (*c["xy_clearance_m"], c["lower_height_clearance_m"], c["upper_height_clearance_m"])
        if len(recorded) != 4 or any(
            abs(float(x) - float(y)) > tolerance for x, y in zip(clear, recorded, strict=True)
        ):
            raise ValueError("Wine site boundary clearances must match the original dimensions")
        spatial = all(x > 0 for x in clear)
    else:
        raise ValueError("An original placement task is required")
    expected = spatial and bool(pairs)
    if c["spatial_satisfied"] is not spatial or c["reconstructed_success"] is not expected:
        raise ValueError("Saved placement conditions must match the independently calculated predicate")
    return expected


def first(values):
    return next((index for index, value in enumerate(values) if value), None)


def compare_rows(left, right):
    """Independently recompute common-prefix differences from scalar values."""
    count = min(len(left), len(right))
    if not count:
        return dict(common_endpoints=0, object_max_m=None, tcp_max_m=None, opening_max=None)
    a, b = left[:count], right[:count]
    objects = [math.dist(x["target_world_m"], y["target_world_m"]) for x, y in zip(a, b, strict=True)]
    tcps = [math.dist(x["tcp_world_m"], y["tcp_world_m"]) for x, y in zip(a, b, strict=True)]
    opening = [abs(x["opening"] - y["opening"]) for x, y in zip(a, b, strict=True)]
    return dict(
        common_endpoints=count,
        object_max_m=max(objects),
        object_first_above_1mm=first([v > 0.001 for v in objects]),
        object_first_above_10mm=first([v > 0.01 for v in objects]),
        tcp_max_m=max(tcps),
        opening_max=max(opening),
        first_grasp_difference=first([x["native_grasp"] != y["native_grasp"] for x, y in zip(a, b, strict=True)]),
        first_support_contact_difference=first(
            [
                x["conditions"]["support_contact"] != y["conditions"]["support_contact"]
                for x, y in zip(a, b, strict=True)
            ]
        ),
        first_spatial_difference=first(
            [
                x["conditions"]["spatial_satisfied"] != y["conditions"]["spatial_satisfied"]
                for x, y in zip(a, b, strict=True)
            ]
        ),
        first_success_difference=first([x["native_success"] != y["native_success"] for x, y in zip(a, b, strict=True)]),
    )


def tail_summary(rows, expected_periods, task):
    if len(rows) < 40:
        return dict(available=False, full_trajectory=len(rows) == expected_periods, actual_periods=len(rows))
    tail = rows[-40:]
    result = dict(
        available=True,
        full_trajectory=len(rows) == expected_periods,
        actual_periods=len(rows),
        support_contact=sum(row["conditions"]["support_contact"] for row in tail),
        spatial_satisfied=sum(row["conditions"]["spatial_satisfied"] for row in tail),
        native_success=sum(row["native_success"] for row in tail),
        native_grasp=sum(row["native_grasp"] for row in tail),
    )
    if task == 9:
        for axis in (0, 1):
            values = [row["conditions"]["xy_clearance_m"][axis] for row in tail]
            result[f"axis_{axis}_clearance_m"] = dict(
                minimum=min(values),
                median=statistics.median(values),
                maximum=max(values),
                outside=sum(v <= 0 for v in values),
            )
        for key in ("lower_height_clearance_m", "upper_height_clearance_m"):
            values = [row["conditions"][key] for row in tail]
            result[key] = dict(
                minimum=min(values),
                median=statistics.median(values),
                maximum=max(values),
                outside=sum(v <= 0 for v in values),
            )
    else:
        for key in ("xy_clearance_m", "height_difference_m"):
            values = [row["conditions"][key] for row in tail]
            result[key] = dict(minimum=min(values), median=statistics.median(values), maximum=max(values))
    return result


def audit_case(index, config, identity, reconstruction, mapping):
    item = prefix.read(reconstruction["cases"][str(index)]["result_ref"])
    branches = [prefix.read(item[key]) for key in ("fast_ref", "bounded_ref")]
    case_identity = identity["previous_identity"]["cohort"]["cases"][index]
    shared = preparation.shared_inputs()
    prefix.require(
        shared["receipt"] == identity["previous_identity"]["cohort"]["shared_receipt"],
        "Frozen source assets must match",
    )
    case = preparation.case_inputs(case_identity, shared)
    spec = case["spec"]
    prefix.require(spec == item["case"] == mapping["cases"][index]["case"], "The original case identity is required")
    xml = ET.fromstring(case["record_xml"])
    task = source.TASKS[spec["task_id"]]
    target_names = set(geometry.contact_geom_names(case["record_xml"], task["target"]))
    support_names = set(geometry.contact_geom_names(case["record_xml"], task["support"]))
    site_size = None
    if spec["task_id"] == 9:
        sites = [site for site in xml.iter("site") if site.get("name") == task["site"]]
        prefix.require(len(sites) == 1, "One original wine placement site is required")
        site_size = [float(x) for x in sites[0].get("size").split()]
    columns = mapping["cases"][index]["metrics"]["state_position_mapping"]["columns"]
    checked, tails = 0, {}
    for label, branch in zip(("fast", "bounded"), branches, strict=True):
        original_rows, rows = previous.periods(branch), item["endpoints"][label]
        prefix.require(
            len(original_rows) == len(rows) == item["summaries"][label]["endpoints"],
            "Every saved endpoint must be audited",
        )
        for step, (old, row) in enumerate(zip(original_rows, rows, strict=True)):
            state = old["actual_state"]
            prefix.require(
                row["step"] == old["step"] == step
                and row["state_sha256"] == old["endpoint_receipt"]["sim_state_sha256"] == state_sha256(state)
                and row["native_success"] is old["task_success"]
                and row["native_grasp"] is old["target_grasp"]
                and abs(row["opening"] - old["completed_feedback"]["score"]["opening"]) <= config["tolerance"]
                and all(
                    abs(float(row["target_world_m"][j]) - float(state[columns[j]])) <= config["tolerance"]
                    for j in range(3)
                )
                and all(
                    pair["target_geom"] in target_names
                    and pair["other_geom"] in support_names
                    and math.isfinite(pair["distance_m"])
                    for pair in row["support_pairs"]
                ),
                "State identity, object columns, labels and retained target/support pairs must match",
            )
            desired = case["timeline"].window(step)[0][0, :3, 3]
            prefix.require(
                all(
                    abs(float(desired[j]) - float(row["tcp_world_m"][j]) - float(old["tcp_error"][j])) <= 1e-9
                    for j in range(3)
                ),
                "TCP error must match the original reference and observation",
            )
            goal = scalar_conditions(spec["task_id"], row, site_size, config["tolerance"])
            prefix.require(
                goal is row["native_success"]
                and (row["left_pad_contact"] and row["right_pad_contact"]) is row["bilateral"] is row["native_grasp"],
                "Reconstructed labels must match the original observations",
            )
            checked += 1
        tails[label] = tail_summary(rows, spec["periods"], spec["task_id"])
    pair = compare_rows(item["endpoints"]["fast"], item["endpoints"]["bounded"])
    for key, value in pair.items():
        expected = item["pair"][key]
        prefix.require(
            (value == expected)
            if value is None or isinstance(value, int)
            else abs(value - expected) <= config["tolerance"],
            f"Common-prefix comparison differs: {key}",
        )
    return dict(
        case=spec,
        checked_endpoints=checked,
        fast_qualified=item["fast_qualified"],
        bounded_qualified=item["bounded_qualified"],
        pair=pair,
        tails=tails,
        site_size=site_size,
        reconstruction_case_ref=reconstruction["cases"][str(index)]["result_ref"],
    )


def worker(index):
    prefix.environment()
    registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
    config, identity, reconstruction, _audits, _first, mapping = inputs()
    prefix.require(
        0 <= index < config["cases"]
        and identity == registration["inputs"]
        and source_refs() == registration["source_refs"],
        "One registered case and source set are required",
    )
    original.require_no_simulator_import()
    value = audit_case(index, config, identity, reconstruction, mapping)
    value.update(
        complete=True,
        passed=True,
        models=0,
        forwards=0,
        physics_steps=0,
        world_model_calls=0,
        gpu_calls=0,
        fresh_cases=0,
    )
    print(publish_result(OUTPUT / f"case{index:02d}/result.json", value), flush=True)


def execute():
    prefix.environment()
    original.require_no_simulator_import()
    qa_ref, dry_ref = prefix.reference(QA), prefix.reference(DRY)
    qa, dry = prefix.read(qa_ref), prefix.read(dry_ref)
    before, sources = snapshot(ROOT)["sources"], source_refs()
    config, identity, _recon, _audits, _first, _mapping = inputs()
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
        "Source-matched CPU QA and zero-model dry must pass",
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
        for i in range(config["cases"])
    ]
    lifecycle = run_owned_workers(specs, cwd=ROOT, timeout_seconds=config["worker_timeout_seconds"])
    for child in lifecycle["workers"]:
        child["log_ref"] = prefix.reference(child.pop("log"))
    lifecycle_ref = publish_result(OUTPUT / "workers.json", lifecycle)
    result = dict(
        complete=False,
        passed=False,
        cases={},
        registration_ref=registration_ref,
        lifecycle_ref=lifecycle_ref,
        models=0,
        forwards=0,
        physics_steps=0,
        world_model_calls=0,
        gpu_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        for i in range(config["cases"]):
            path = OUTPUT / f"case{i:02d}/result.json"
            row = result["cases"][str(i)] = dict(status="failed")
            if path.is_file():
                ref = prefix.reference(path)
                value = prefix.read(ref)
                row["available_ref"] = ref
                if value.get("complete") is value.get("passed") is True:
                    row.update(status="closed", result_ref=ref)
        prefix.require(
            lifecycle["launch_error"] is None
            and len(lifecycle["workers"]) == config["cases"]
            and all(
                w["returncode"] == 0 and w["actual_wait"] is w["reaped"] is True and not w["signal_sent"]
                for w in lifecycle["workers"]
            )
            and all(row["status"] == "closed" for row in result["cases"].values()),
            "Every independent audit worker must close and be reaped",
        )
        rows = [prefix.read(result["cases"][str(i)]["result_ref"]) for i in range(config["cases"])]
        _config, after, *_ = inputs()
        prefix.require(
            sum(r["checked_endpoints"] for r in rows) == config["endpoints"]
            and after == identity
            and source_refs() == sources
            and snapshot(ROOT)["sources"] == before,
            "All endpoints and source identities must close",
        )
        result.update(
            complete=True,
            passed=True,
            sources_unchanged=True,
            checked_endpoints=sum(r["checked_endpoints"] for r in rows),
        )
    except Exception as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
    print(publish_result(OUTPUT / "result.json", result), flush=True)
    if not result["passed"]:
        raise SystemExit("The independent audit did not close; no automatic retry")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--worker", type=int)
    args = parser.parse_args()
    prefix.environment()
    original.require_no_simulator_import()
    if args.dry:
        _config, identity, *_ = inputs()
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
