"""Decompose a saved native task predicate with non-stepping CPU queries."""

import argparse
import hashlib
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from s2a.geometry import inverse, validate_transform
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.native_runtime.saved_geometry_v2 import reconstruct_endpoint
from s2a.research.wam.training.native.native_training import state_sha256
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import adaptive_live_bowl_audit as history
from scripts.simulation.runtime import adaptive_live_bowl_recovery_audit as comparison
from scripts.simulation.runtime import adaptive_live_bowl_supervisor as supervisor
from scripts.simulation.runtime import adaptive_long_final_endpoint_admission as previous
from scripts.simulation.runtime import adaptive_long_final_endpoint_owner as previous_owner
from scripts.simulation.runtime import registered_feedback_phase as lifecycle
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/bowl_native_predicate_probe_20261113.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/bowl_native_predicate_probe_20261113.md"
OUTPUT = ROOT / "outputs/bowl_native_predicate_probe_20261113"
QA = ROOT / "outputs/qa/bowl_native_predicate_probe_accounted_20261113/result.json"
DRY = ROOT / "outputs/bowl_native_predicate_probe_dry_accounted_20261113/result.json"
MODULE = "scripts.simulation.runtime.probe_bowl_native_predicate"
CONFIG_SHA = "b95eaf1ea7a48188197b33eab0546b3529631138667bf7677e66b81aca04d3e8"
PLAN_SHA = "1a5fdb688c9afbdbf5a5eb1247505168a4eca6cd7744e4ad6063e05d04d3c6ff"
CHECKER = {
    "path": str(previous.OUTPUT / "checker/result.json"),
    "sha256": "53424f2e0ea3bbc89c40d515a585c7e74d7995a38c48e0fcb9e406ad4d85063a",
}
QA_DEVELOPMENT = {
    "path": str(ROOT / "outputs/qa/bowl_predicate_development_20261113/result.json"),
    "sha256": "3e949658cee1cb294a9889faaec071446b5b7338a5e7c82104cb4862e5c1cc6a",
}
QA_INITIAL = {
    "path": str(ROOT / "outputs/qa/bowl_native_predicate_probe_20261113/result.json"),
    "sha256": "d630271cf1e268ee492e2a44503463e3d2b8182e30d4cc686a3a7f908cb5f9a2",
}
SYNTHETIC_NODE = (
    "tests/research/native_runtime/test_saved_geometry_v2.py::"
    "test_real_cpu_mujoco_normalization_is_recorded_without_physics_step"
)
NATIVE_SOURCES = {
    "object_state_source_sha256": ROOT
    / "outputs/native_runtime_path_snapshot_20261029/LIBERO/libero/libero/envs/object_states/base_object_states.py",
    "predicate_source_sha256": ROOT
    / "outputs/native_runtime_path_snapshot_20261029/LIBERO/libero/libero/envs/predicates/base_predicates.py",
    "contact_source_sha256": ROOT / "outputs/libero_env311/lib/python3.11/site-packages/robosuite/utils/sim_utils.py",
    "geom_group_source_sha256": ROOT
    / "outputs/libero_env311/lib/python3.11/site-packages/robosuite/utils/mjcf_utils.py",
}
REQUIRED_QA_TARGETS = tuple(
    dict.fromkeys(
        (
            *previous_owner.REQUIRED_QA_TARGETS,
            "tests/research/native_runtime/test_saved_geometry_v2.py",
            "tests/research/native_runtime/test_probe_bowl_native_predicate.py",
        )
    )
)


def contact_geom_names(xml, body_name):
    """Apply the pinned robosuite object-subtree group rule to recorded XML."""
    root = ET.fromstring(xml)
    bodies = [node for node in root.iter("body") if node.get("name") == body_name]
    prefix.require(len(bodies) == 1, "Exactly one recorded object body is required")
    names = [node.get("name") for node in bodies[0].iter("geom") if node.get("group") in (None, "0")]
    prefix.require(
        names and all(type(name) is str and name for name in names) and len(set(names)) == len(names),
        "Named, unique recorded contact geoms are required",
    )
    return names


def on_conditions(bowl, plate, contact_pairs, *, xy_limit):
    """Apply the saved native On conditions to one reconstructed state."""
    bowl, plate = np.asarray(bowl, dtype=float), np.asarray(plate, dtype=float)
    prefix.require(
        bowl.shape == plate.shape == (3,)
        and np.isfinite(bowl).all()
        and np.isfinite(plate).all()
        and type(contact_pairs) is list
        and type(xy_limit) is float
        and xy_limit > 0,
        "Finite object positions, contact pairs and a positive limit are required",
    )
    xy = float(np.linalg.norm(bowl[:2] - plate[:2]))
    height, touching, centered = bool(plate[2] <= bowl[2]), bool(contact_pairs), bool(xy < xy_limit)
    return dict(
        bowl_world_m=bowl,
        plate_world_m=plate,
        xy_distance_m=xy,
        bowl_above_plate_m=float(bowl[2] - plate[2]),
        height_condition=height,
        contact_condition=touching,
        xy_condition=centered,
        reconstructed_on=height and touching and centered,
    )


def verify_synthetic_qa(qa, *, parallelism):
    """Count the one actual MuJoCo synthetic model/forward in a completed QA."""
    calls = [
        report
        for worker in qa["workers"]
        for report in worker["test_reports"]
        if report["nodeid"] == SYNTHETIC_NODE and report["when"] == "call"
    ]
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is True
        and qa["actual_wait"] is qa["reaped"] is qa["source_unchanged"] is True
        and qa["configured_parallelism"] == qa["execution_parallelism"] == parallelism
        and qa["signal_sent"] is False
        and len(calls) == 1
        and calls[0]["outcome"] == "passed"
        and calls[0]["wasxfail"] is None,
        "Exactly one real synthetic MuJoCo test and closed QA workers are required",
    )
    return 1


def source_refs():
    paths = (
        CONFIG,
        PLAN,
        Path(__file__).resolve(),
        ROOT / "src/s2a/research/native_runtime/saved_geometry_v2.py",
        *(ROOT / path for path in REQUIRED_QA_TARGETS),
    )
    refs = previous_owner.source_refs()
    known = {ref["path"]: ref["sha256"] for ref in refs}
    for path in paths:
        ref = prefix.reference(path)
        if ref["path"] in known:
            prefix.require(known[ref["path"]] == ref["sha256"], "Conflicting source identities")
        else:
            refs.append(ref)
            known[ref["path"]] = ref["sha256"]
    return refs


def load_inputs():
    """Bind the real producer, complete checker, native predicate and XML."""
    prefix.environment()
    refs = {
        "config": {"path": str(CONFIG), "sha256": CONFIG_SHA},
        "plan": {"path": str(PLAN), "sha256": PLAN_SHA},
        "qa_development": QA_DEVELOPMENT,
        "qa_initial": QA_INITIAL,
    }
    config = prefix.read(refs["config"])
    prefix.checked(refs["plan"])
    prefix.require(
        config["schema"] == "s2a.bowl-native-predicate-probe.v1"
        and (ROOT / config["output"]).resolve() == OUTPUT
        and (ROOT / config["qa"]).resolve() == QA
        and (ROOT / config["dry"]).resolve() == DRY
        and config["endpoint_first"] == 108
        and config["endpoint_last"] == 147
        and config["max_forwards"] == 40
        and config["max_static_models"] == config["max_data"] == 1
        and config["known_prior_diagnostic_static_models"] == 2
        and config["current_qa_synthetic_models"] == config["current_qa_synthetic_forwards"] == 3
        and config["new_physics_steps"] == config["new_contexts"] == config["new_model_requests"] == 0
        and config["fresh_cases"] == 0
        and config["automatic_retry"] is False,
        "The exact one-model, non-stepping diagnostic limits are required",
    )
    prefix.require(
        verify_synthetic_qa(prefix.read(QA_DEVELOPMENT), parallelism=32)
        + verify_synthetic_qa(prefix.read(QA_INITIAL), parallelism=64)
        == 2,
        "Both previously completed synthetic QA model/forward calls must be retained",
    )
    inputs, checked = previous.load_inputs(), prefix.read(CHECKER)
    producer = prefix.read(inputs["producer"]["branch_ref"])
    branch = prefix.read(checked["branch_ref"])
    workers = prefix.read(checked["workers_ref"])
    prefix.require(
        inputs["producer"]["complete"] is inputs["producer"]["passed"] is True
        and checked["complete"] is checked["passed"] is checked["sources_unchanged"] is True
        and checked["control_qualified"] is False
        and checked["automatic_retry"] is False
        and checked["branch_ref"]["sha256"] == config["checker_branch_sha256"]
        and CHECKER["sha256"] == config["checker_sha256"]
        and inputs["refs"]["long_producer"]["sha256"] == config["producer_sha256"]
        and checked["counts"]["control_steps"] == 147
        and checked["counts"]["adaptive_steps"] == 40
        and checked["model_requests"] == checked["denoise_calls"] == 0,
        "The completed producer and independent replay must match the registered case",
    )
    lifecycle.require_closed_workers(workers, ["checker"])
    prefix.checked(workers["workers"][0]["log_ref"])
    differences = comparison.compare_branches(producer, branch)
    prefix.require(all(value == 0 for value in differences.values()), "All saved physical samples must match exactly")
    previous_owner.previous.compare_captures(producer, branch)
    preceding, adaptive = history.periods(branch, "checker")
    prefix.require(
        len(preceding) == 107
        and len(adaptive) == 40
        and [row["step"] + 1 for row in adaptive] == list(range(108, 148))
        and branch["case"]["case_id"] == config["case_id"]
        and inputs["config"]["seed"] == config["seed"],
        "Exactly 40 ordered adaptive states from the registered case are required",
    )
    case = inputs["long_inputs"]["prior"]["base"]["source"]["previous"]["base"]["case"]
    xml = case["record_xml"]
    prefix.require(hashlib.sha256(xml.encode()).hexdigest() == config["record_xml_sha256"], "Recorded XML changed")
    prefix.require(
        any(
            row.get("stage") == "record_xml" and row.get("xml_sha256") == config["record_xml_sha256"]
            for row in branch["runtime"]["model_records"]
        ),
        "The checker must have used the same recorded XML",
    )
    bowl = contact_geom_names(xml, config["bowl_body"])
    plate = contact_geom_names(xml, config["plate_body"])
    prefix.require(
        len(bowl) == config["bowl_contact_geoms"] == 40
        and len(plate) == config["plate_contact_geoms"] == 10
        and not set(bowl) & set(plate),
        "The registered bowl and plate contact geom sets must match",
    )
    bddl = branch["runtime"]["bddl_ref"]
    prefix.require(bddl["sha256"] == config["bddl_sha256"], "The native task BDDL must remain fixed")
    text = prefix.checked(bddl).read_text()
    prefix.require("(And (On akita_black_bowl_1 plate_1))" in text, "The registered native On goal is required")
    refs.update(
        checker=CHECKER, checker_branch=checked["branch_ref"], checker_workers=checked["workers_ref"], bddl=bddl
    )
    refs.update({key: {"path": str(path), "sha256": config[key]} for key, path in NATIVE_SOURCES.items()})
    consumed = dict(inputs["source_sha256"])
    for ref in refs.values():
        prefix.checked(ref, consumed)
    return dict(
        config=config,
        refs=refs,
        previous_inputs=inputs,
        checked=checked,
        producer=producer,
        branch=branch,
        adaptive=adaptive,
        case=case,
        bowl_geoms=bowl,
        plate_geoms=plate,
        source_sha256=consumed,
    )


def input_receipt(inputs):
    return dict(
        refs=inputs["refs"],
        previous=previous.input_receipt(inputs["previous_inputs"]),
        bowl_geoms=inputs["bowl_geoms"],
        plate_geoms=inputs["plate_geoms"],
        source_sha256=inputs["source_sha256"],
    )


def dry_run():
    inputs = load_inputs()
    prefix.require("mujoco" not in sys.modules, "Dry admission cannot import MuJoCo")
    return publish(
        DRY,
        dict(
            complete=True,
            passed=True,
            inputs=input_receipt(inputs),
            source_refs=source_refs(),
            static_models=0,
            data=0,
            forwards=0,
            physics_steps=0,
            contexts=0,
            model_requests=0,
            automatic_retry=False,
        ),
    )


def check_qa_and_dry(inputs):
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    before = snapshot(ROOT)["sources"]
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["configured_parallelism"] == qa["execution_parallelism"] == 64
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and qa["aggregate_node_partition_exact"] is True
        and not qa["collection"]["collection_errors"]
        and set(REQUIRED_QA_TARGETS).issubset(set(qa["targets"]))
        and prefix.read(qa["source_before_ref"])["sources"] == before
        and dry["complete"] is dry["passed"] is True
        and dry["automatic_retry"] is False
        and dry["inputs"] == input_receipt(inputs)
        and dry["source_refs"] == source_refs()
        and dry["static_models"] == dry["data"] == dry["forwards"] == dry["physics_steps"] == 0
        and dry["contexts"] == dry["model_requests"] == 0,
        "Matching 64-worker regression and zero-model dry run are required",
    )
    prefix.require(
        verify_synthetic_qa(qa, parallelism=64) == 1
        and inputs["config"]["current_qa_synthetic_models"] == inputs["config"]["current_qa_synthetic_forwards"] == 3,
        "All three current synthetic QA models and forwards must be accounted for",
    )
    return before


def query_saved_states(inputs, counters, *, mujoco):
    """Reconstruct 40 saved states; never dispatch an action or advance time."""
    config, case = inputs["config"], inputs["case"]
    counters["model_attempts"] += 1
    model = mujoco.MjModel.from_xml_string(case["record_xml"])
    counters["models"] += 1
    counters["data_attempts"] += 1
    data = mujoco.MjData(model)
    counters["data"] += 1
    prefix.require(
        dict(nq=model.nq, nv=model.nv, na=model.na) == config["model_shape"] == dict(nq=41, nv=37, na=0),
        "The recorded MuJoCo model shape must match",
    )
    site = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, config["tcp_site"]))
    bowl = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config["bowl_body"]))
    plate = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config["plate_body"]))
    prefix.require(min(site, bowl, plate) >= 0, "The registered site and object bodies must exist")
    bowl_geoms, plate_geoms = set(inputs["bowl_geoms"]), set(inputs["plate_geoms"])
    prefix.require(
        all(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0 for name in bowl_geoms | plate_geoms),
        "Every registered collision geom must exist in the compiled model",
    )
    expected_site = validate_transform(np.asarray(case["expected"]["controller_site_pose"], dtype=float))
    expected_source = validate_transform(np.asarray(case["expected"]["source_pose"], dtype=float))
    tool = validate_transform(np.asarray(case["tool"], dtype=float))
    site_from_tcp = validate_transform(inverse(expected_site) @ expected_source @ tool)
    captures = {row["frame_index"]: row for row in inputs["branch"]["runtime"]["adaptive_checker_captures"]}
    forward = mujoco.mj_forward

    def counted_forward(actual_model, actual_data):
        prefix.require(counters["forwards_attempted"] < config["max_forwards"], "The forward budget is exhausted")
        counters["forwards_attempted"] += 1
        forward(actual_model, actual_data)
        counters["forwards_completed"] += 1

    originals = {}
    for name in ("mj_step", "mj_step1", "mj_step2"):
        if hasattr(mujoco, name):
            originals[name] = getattr(mujoco, name)

            def reject(*_args, **_kwargs):
                raise ValueError("A saved-state diagnostic cannot advance physics")

            setattr(mujoco, name, reject)
    rows, anchors = [], []
    try:
        for period in inputs["adaptive"]:
            step = period["step"] + 1
            state = np.asarray(period["actual_state"], dtype=np.float64)
            prefix.require(
                state_sha256(state) == period["endpoint_receipt"]["sim_state_sha256"],
                "Every actual saved state must match its endpoint digest",
            )
            geometry = reconstruct_endpoint(
                model,
                data,
                state,
                site_id=site,
                body_id=bowl,
                site_from_tcp=site_from_tcp,
                forward=counted_forward,
            )
            pairs = []
            for index in range(data.ncon):
                contact = data.contact[index]
                first = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
                second = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
                if (first in bowl_geoms and second in plate_geoms) or (second in bowl_geoms and first in plate_geoms):
                    distance = float(contact.dist)
                    prefix.require(np.isfinite(distance), "Contact distances must be finite")
                    pairs.append(dict(geom1=first, geom2=second, distance_m=distance))
            point = on_conditions(
                geometry["body_origin_world_m"], data.xpos[plate], pairs, xy_limit=config["xy_limit_m"]
            )
            point.update(
                endpoint_step=step,
                state_sha256=period["endpoint_receipt"]["sim_state_sha256"],
                native_success=bool(period["task_success"]),
                native_grasp=bool(period["target_grasp"]),
                contact_pairs=pairs,
                contact_count=len(pairs),
                state_change=geometry["state_change"],
            )
            rows.append(point)
            if step in captures:
                captured_tcp = (np.asarray(captures[step]["current_source_pose"], dtype=float) @ tool)[:3, 3]
                delta = float(np.max(np.abs(geometry["tcp_world_m"] - captured_tcp)))
                prefix.require(delta <= config["tcp_anchor_tolerance_m"], "Reconstructed TCP differs from capture")
                anchors.append(dict(endpoint_step=step, max_absolute_difference_m=delta))
        prefix.require(
            len(rows) == counters["forwards_attempted"] == counters["forwards_completed"] == 40
            and [row["endpoint_step"] for row in rows] == list(range(108, 148))
            and [row["endpoint_step"] for row in anchors] == [111, 119, 120, 128, 136, 144]
            and counters["models"] == counters["data"] == 1
            and counters["physics_steps"] == counters["contexts"] == counters["model_requests"] == 0,
            "The completed non-stepping work and capture anchors must match the registered scope",
        )
        mismatch = [row["endpoint_step"] for row in rows if row["native_success"] != row["reconstructed_on"]]
        return dict(
            rows=rows,
            capture_anchor_checks=anchors,
            all_native_labels_match=not mismatch,
            label_mismatch_endpoints=mismatch,
            interpretation_eligible=not mismatch,
            native_failed_endpoints=[row["endpoint_step"] for row in rows if not row["native_success"]],
            reconstructed_failed_endpoints=[row["endpoint_step"] for row in rows if not row["reconstructed_on"]],
            max_state_change=max(row["state_change"]["max_absolute_change"] for row in rows),
            object_and_task_information_used_online=False,
        )
    finally:
        for name, original in originals.items():
            setattr(mujoco, name, original)


def worker():
    """Publish actual model and forward counts even when a query fails."""
    directory = OUTPUT / "worker"
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        automatic_retry=False,
        counters=dict(
            model_attempts=0,
            models=0,
            data_attempts=0,
            data=0,
            forwards_attempted=0,
            forwards_completed=0,
            physics_steps=0,
            contexts=0,
            model_requests=0,
        ),
    )
    try:
        prefix.environment()
        inputs = load_inputs()
        registered = prefix.read(prefix.reference(OUTPUT / "registration.json"))
        prefix.require(
            registered["inputs"] == input_receipt(inputs) and registered["source_refs"] == source_refs(),
            "The worker requires exact registered inputs and sources",
        )
        import mujoco

        result["mujoco_version"] = mujoco.__version__
        result["diagnosis"] = query_saved_states(inputs, result["counters"], mujoco=mujoco)
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def run():
    """Run the preregistered CPU query once and retain incomplete costs."""
    prefix.environment()
    inputs = load_inputs()
    before, sources = check_qa_and_dry(inputs), source_refs()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    directory = OUTPUT / "worker"
    directory.mkdir()
    result = dict(complete=False, passed=False, control_qualified=False, automatic_retry=False)
    workers = None
    try:
        result["registration_ref"] = publish(
            OUTPUT / "registration.json", dict(inputs=input_receipt(inputs), source_refs=sources)
        )
        environment = physical._cpu_environment(directory)
        spec = dict(key="worker", argv=[sys.executable, "-m", MODULE, "--worker"], log=directory / "worker.log")
        with physical._temporary_environment(environment):
            workers = run_owned_workers([spec], cwd=ROOT, timeout_seconds=900)
        result["workers_ref"] = publish(OUTPUT / "workers.json", supervisor._record_worker_logs(workers))
        lifecycle.require_closed_workers(workers, ["worker"])
        result["worker_ref"] = prefix.reference(directory / "result.json")
        actual = prefix.read(result["worker_ref"])
        prefix.require(
            actual["complete"] is actual["passed"] is True
            and actual["counters"]["models"] == actual["counters"]["data"] == 1
            and actual["counters"]["forwards_attempted"] == actual["counters"]["forwards_completed"] == 40
            and actual["counters"]["physics_steps"]
            == actual["counters"]["contexts"]
            == actual["counters"]["model_requests"]
            == 0,
            "The completed static query must match its actual registered work",
        )
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(source_refs() == sources and snapshot(ROOT)["sources"] == before, "Probe sources changed")
        result.update(
            complete=True,
            passed=True,
            sources_unchanged=True,
            counters=actual["counters"],
            all_native_labels_match=actual["diagnosis"]["all_native_labels_match"],
            interpretation_eligible=actual["diagnosis"]["interpretation_eligible"],
        )
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        path = directory / "result.json"
        matched = False
        try:
            matched = source_refs() == sources
        except Exception as error:
            result["source_check_error"] = dict(type=type(error).__name__, message=str(error))
        result["partial_source_match"] = matched
        final = (
            matched
            and workers is not None
            and len(workers.get("workers", [])) == 1
            and all(workers["workers"][0].get(key) is True for key in ("actual_wait", "reaped"))
        )
        if path.is_file():
            try:
                result["available_worker_ref"] = prefix.reference(path)
                partial = prefix.read(result["available_worker_ref"])["counters"] if final else {}
            except Exception as error:
                result["partial_receipt_error"] = dict(type=type(error).__name__, message=str(error))
                partial = {}
        else:
            partial = {}
        bounds = {}
        for name, attempted, completed, limit in (
            ("models", "model_attempts", "models", 1),
            ("data", "data_attempts", "data", 1),
            ("forwards", "forwards_attempted", "forwards_completed", 40),
        ):
            a, c = partial.get(attempted), partial.get(completed)
            if type(a) is type(c) is int and 0 <= c <= a <= limit:
                bounds[name] = dict(lower=c, upper=a)
            else:
                bounds[name] = dict(lower=0, upper=limit)
        result["static_work_bounds"] = bounds
        result["previous_diagnostic_static_models"] = 2
        result["current_qa_synthetic_models"] = 3
        result["current_qa_synthetic_forwards"] = 3
        result["cumulative_diagnostic_static_models"] = dict(
            lower=2 + bounds["models"]["lower"], upper=2 + bounds["models"]["upper"]
        )
        result["identified_static_models_including_current_qa"] = dict(
            lower=5 + bounds["models"]["lower"], upper=5 + bounds["models"]["upper"]
        )
        result["earlier_qa_static_models_known"] = False
        result["cumulative_physical_bounds"] = inputs["checked"]["cumulative_physical_bounds"]
        result["cumulative_model_bounds"] = inputs["checked"]["cumulative_model_bounds"]
        print(publish(OUTPUT / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(dry_run(), flush=True)
    elif args.run:
        run()
    else:
        worker()


if __name__ == "__main__":
    main()
