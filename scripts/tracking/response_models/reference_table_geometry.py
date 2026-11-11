"""Check generated TCP targets against table geometry without advancing physics."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from s2a.geometry import inverse, pose_error, transform, validate_transform
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.wam.training.native.native_training import state_sha256
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import causal_stream_live_48_replay as references
from scripts.simulation.runtime import prepare_quality_rate_repair_recorded_feedback as recorded
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.simulation.runtime import run_causal_wam_two_window_feedback as historical
from scripts.tracking.response_models import replay_contact_torque as contact

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/response_models/reference_table_geometry_20261111.json"
CONFIG_SHA = "8082ca539b528f77f25ee3edccc4e9fa6482e770d2a6d633ef938c1901ae8cd6"
PLAN = ROOT / "plans/v0.0.1/tracking/response_models/reference_table_geometry_20261111.md"
OUTPUT = ROOT / "outputs/reference_table_geometry_20261111"
DRY = ROOT / "outputs/reference_table_geometry_dry_20261111/result.json"
QA = ROOT / "outputs/qa/reference_table_geometry_20261111/result.json"
MODULE = "scripts.tracking.response_models.reference_table_geometry"


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__).resolve(),
            Path(contact.__file__).resolve(),
            ROOT / "src/s2a/research/wam/tracking/preview/development_reference_geometry.py",
            ROOT / "src/s2a/research/wam/tracking/preview/development_reference_geometry_roundoff.py",
            ROOT / "tests/research/response_models/test_reference_table_geometry.py",
        )
    ]


def load_inputs():
    prefix.environment()
    inputs = contact.load_inputs()
    consumed = dict(inputs["replayed_contact_torque"]["source_sha256"])
    config = prefix.read({"path": str(CONFIG), "sha256": CONFIG_SHA}, consumed)
    prefix.require(
        config["schema"] == "s2a.reference-table-geometry.v1"
        and config["output"] == str(OUTPUT.relative_to(ROOT))
        and config["case_index"] == 8
        and config["seed"] == 42
        and (config["first_step"], config["last_step"], config["executed_periods"]) == (80, 126, 46)
        and config["max_models"] == 1
        and config["max_forward_queries"] == 94
        and config["physics_steps"] == config["world_model_calls"] == config["fresh_cases"] == 0
        and config["automatic_retry"] is False,
        "The exact saved-reference geometry budget is required",
    )
    refs = {
        name: {"path": str(ROOT / config[name]), "sha256": config[name + "_sha256"]}
        for name in ("producer", "checker", "contact_audit", "contact_counts")
    }
    producer, checker, audit, correction = (prefix.read(refs[name], consumed) for name in refs)
    prefix.require(
        producer["complete"] is producer["passed"] is checker["complete"] is checker["passed"] is True
        and checker["candidate_modules_unloaded"] is True
        and audit["complete"] is audit["passed"] is correction["complete"] is correction["passed"] is True
        and audit["control_qualified"] is correction["control_qualified"] is False
        and correction["legacy_ref"] == refs["contact_audit"],
        "The replayed feedback and corrected force audit are required",
    )
    actual, checked = recorded.periods(producer), recorded.periods(checker)
    prefix.require(len(actual) == len(checked) == 46, "All original feedback periods are required")
    for i, (left, right) in enumerate(zip(actual, checked, strict=True)):
        prefix.require(left["step"] == right["step"] == 80 + i, "Consecutive independently replayed actions required")
        historical.compare_first_period(left, right)
    old = inputs["old"]["old"]["base"]["previous"]
    initial = np.asarray(old["saved"]["sim_states"][-1]).copy()
    captures = producer["runtime"]["streaming_captures"]
    prefix.require(
        len(captures) == 6 and state_sha256(initial) == captures[0]["current_endpoint"]["sim_state_sha256"],
        "The step80 state must match the first actual capture",
    )
    for capture, exchange in zip(captures, producer["runtime"]["streaming_model_exchanges"], strict=True):
        for ref in (capture["canvas_png"], exchange["request_ref"], exchange["response_ref"]):
            prefix.checked(ref, consumed)
    image_root = Path(refs["producer"]["path"]).parents[2] / "producer/simulation/runtime"
    chunks, _sources = references.decode_actual_references(inputs, producer, image_root=image_root, max_windows=6)
    prefix.require(len(chunks) == 6, "Six already-generated reference chunks are required")
    states = [initial] + [np.asarray(row["actual_state"]).copy() for row in actual]
    targets = [np.asarray(chunks[(step - 80) // 8][(step - 80) % 8]).copy() for step in range(80, 127)]
    for index, state in enumerate(states[1:]):
        prefix.require(
            state_sha256(state) == actual[index]["endpoint_receipt"]["sim_state_sha256"], "Saved state identity"
        )
    inputs["reference_table_geometry"] = {
        "config": config,
        "refs": refs,
        "states": states,
        "targets": targets,
        "source_sha256": consumed,
    }
    return inputs


def input_receipt(inputs):
    item = inputs["reference_table_geometry"]
    return {
        "prior": contact.input_receipt(inputs),
        "config_ref": {"path": str(CONFIG), "sha256": CONFIG_SHA},
        "refs": item["refs"],
        "source_sha256": item["source_sha256"],
    }


def matching_contacts(data, geom1, geom2):
    """Return only penetrating or touching contacts for a fixed geometry pair."""
    return [
        float(item.dist)
        for item in data.contact[: data.ncon]
        if {int(item.geom1), int(item.geom2)} == {geom1, geom2} and float(item.dist) <= 0
    ]


def inspect_target(probe, state, target_tcp, site_from_tcp, table, finger):
    """Compare current and target-aligned finger geometry at fixed joint state."""
    target_tcp = validate_transform(np.asarray(target_tcp, dtype=float))
    site_from_tcp = validate_transform(np.asarray(site_from_tcp, dtype=float))
    probe.restore(state)
    data = probe.data
    current_site = probe.site_transform()
    current_tcp = current_site @ site_from_tcp
    finger_before = transform(data.geom_xpos[finger].copy(), data.geom_xmat[finger].reshape(3, 3).copy())
    table_position, table_rotation = data.geom_xpos[table].copy(), data.geom_xmat[table].copy()
    qpos, qvel, timestamp = data.qpos.copy(), data.qvel.copy(), float(data.time)
    actual = matching_contacts(data, table, finger)
    desired_site = target_tcp @ inverse(site_from_tcp)
    expected_finger = desired_site @ inverse(current_site) @ finger_before
    probe.place(desired_site)
    candidate = matching_contacts(data, table, finger)
    finger_after = transform(data.geom_xpos[finger].copy(), data.geom_xmat[finger].reshape(3, 3).copy())
    np.testing.assert_allclose(finger_after, expected_finger, atol=1e-9, rtol=0)
    np.testing.assert_allclose(probe.site_transform() @ site_from_tcp, target_tcp, atol=1e-9, rtol=0)
    np.testing.assert_allclose(data.geom_xpos[table], table_position, atol=1e-12, rtol=0)
    np.testing.assert_allclose(data.geom_xmat[table], table_rotation, atol=1e-12, rtol=0)
    np.testing.assert_allclose(data.qpos, qpos, atol=1e-12, rtol=0)
    np.testing.assert_allclose(data.qvel, qvel, atol=1e-12, rtol=0)
    prefix.require(float(data.time) == timestamp, "A static geometry query must not advance physical time")
    return {
        "current_tcp_error": pose_error(target_tcp, current_tcp),
        "current_finger_table_contact_distances_m": actual,
        "target_proxy_finger_table_contact_distances_m": candidate,
        "target_proxy_finger_table_intersection": bool(candidate),
        "finger_pose_transform_max_difference": float(np.max(np.abs(finger_after - expected_finger))),
        "qpos_max_difference": float(np.max(np.abs(data.qpos - qpos))),
        "table_position_max_difference": float(np.max(np.abs(data.geom_xpos[table] - table_position))),
    }


def analyze(inputs, stats):
    """Permit only the explicitly bounded forwards; reject physical integration."""
    import mujoco

    from s2a.research.wam.tracking.preview.development_reference_geometry_roundoff import RoundoffBoundedProbe

    originals = {name: getattr(mujoco, name) for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward")}

    def prohibited(*_args, **_kwargs):
        stats["physics_steps_attempted"] += 1
        raise RuntimeError("Static reference geometry cannot integrate physics")

    def forward(*args, **kwargs):
        stats["forward_queries_attempted"] += 1
        prefix.require(stats["forward_queries_attempted"] <= 94, "The 94-query budget is exhausted")
        value = originals["mj_forward"](*args, **kwargs)
        stats["forward_queries_completed"] += 1
        return value

    try:
        for name in ("mj_step", "mj_step1", "mj_step2"):
            setattr(mujoco, name, prohibited)
        mujoco.mj_forward = forward
        old = inputs["old"]["old"]["base"]["previous"]
        item = inputs["reference_table_geometry"]
        config = item["config"]
        stats["models_attempted"] += 1
        model = mujoco.MjModel.from_xml_string(old["record_xml"])
        stats["models_completed"] += 1
        target_object = inputs["old"]["old"]["base"]["config"]["target_object"]
        probe = RoundoffBoundedProbe(model, target_object, budget=94)
        prefix.require(
            old["controller"]["contract"]["controller_site"] == "gripper0_grip_site", "Controller site identity"
        )
        site_from_tcp = inverse(np.asarray(old["controller"]["source_from_site"])) @ old["tool"]
        table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, config["table_geom"])
        finger = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, config["finger_geom"])
        prefix.require(min(table, finger) >= 0 and table != finger, "Registered table and finger geoms required")
        rows = []
        for offset, (state, target) in enumerate(zip(item["states"], item["targets"], strict=True)):
            step = 80 + offset
            row = inspect_target(probe, state, target, site_from_tcp, table, finger)
            row.update(
                step=step, generated_step=80 + 8 * (offset // 8), executed=step < 126, state_sha256=state_sha256(state)
            )
            rows.append(row)
        prefix.require(
            probe.queries == stats["forward_queries_completed"] == 94, "Every fixed geometry query must complete"
        )
        return {
            "periods": rows,
            "max_object_rotation_change": probe.max_object_rotation_change,
            "max_qpos_change_during_placement": probe.max_qpos_change_during_placement,
            "scope": "current gripper configuration rigidly placed at each generated target; no IK or dynamics",
        }
    finally:
        for name, function in originals.items():
            setattr(mujoco, name, function)


def worker():
    prefix.environment()
    prefix.require(os.environ.get("MUJOCO_GL") == "disable", "No graphics context may be initialized")
    inputs = load_inputs()
    registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
    prefix.require(
        registration["inputs"] == input_receipt(inputs) and registration["source_refs"] == source_refs(),
        "Worker identity changed",
    )
    stats = {
        "models_attempted": 0,
        "models_completed": 0,
        "forward_queries_attempted": 0,
        "forward_queries_completed": 0,
        "physics_steps_attempted": 0,
    }
    result = {
        "complete": False,
        "passed": False,
        "stats": stats,
        "world_model_calls": 0,
        "fresh_cases": 0,
        "control_qualified": False,
        "automatic_retry": False,
    }
    try:
        result["analysis"] = analyze(inputs, stats)
        prefix.require(
            stats["models_completed"] == 1
            and stats["forward_queries_completed"] == 94
            and stats["physics_steps_attempted"] == 0,
            "The zero-physics geometry budget must hold",
        )
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(OUTPUT / "worker/result.json", result), flush=True)


def dry_run():
    inputs = load_inputs()
    print(
        publish(
            DRY,
            {
                "complete": True,
                "passed": True,
                "inputs": input_receipt(inputs),
                "source_refs": source_refs(),
                "models": 0,
                "forward_queries": 0,
                "physics_steps": 0,
                "world_model_calls": 0,
                "fresh_cases": 0,
            },
        ),
        flush=True,
    )


def execute():
    inputs = load_inputs()
    sources, before = source_refs(), snapshot(ROOT)["sources"]
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and not qa["collection"]["collection_errors"]
        and prefix.read(qa["source_before_ref"])["sources"] == before
        and dry["complete"] is dry["passed"] is True
        and dry["inputs"] == input_receipt(inputs)
        and dry["source_refs"] == sources
        and dry["models"] == dry["forward_queries"] == dry["physics_steps"] == 0,
        "Source-matched 32-worker QA and zero-work dry are required",
    )
    OUTPUT.mkdir(exist_ok=False)
    (OUTPUT / "worker").mkdir()
    registration = publish(OUTPUT / "registration.json", {"inputs": input_receipt(inputs), "source_refs": sources})
    result = {
        "complete": False,
        "passed": False,
        "registration_ref": registration,
        "world_model_calls": 0,
        "fresh_cases": 0,
        "control_qualified": False,
        "automatic_retry": False,
    }
    try:
        env = physical._cpu_environment(OUTPUT / "worker")
        env["MUJOCO_GL"] = "disable"
        spec = {"key": "geometry", "argv": [sys.executable, "-m", MODULE, "--worker"], "log": OUTPUT / "worker.log"}
        with physical._temporary_environment(env):
            lifecycle = run_owned_workers(
                [spec], cwd=ROOT, timeout_seconds=inputs["reference_table_geometry"]["config"]["worker_timeout_seconds"]
            )
        result["workers_ref"] = publish(OUTPUT / "workers.json", lifecycle)
        prefix.require(
            lifecycle["launch_error"] is None
            and len(lifecycle["workers"]) == 1
            and lifecycle["workers"][0]["actual_wait"] is lifecycle["workers"][0]["reaped"] is True
            and lifecycle["workers"][0]["returncode"] == 0
            and lifecycle["workers"][0]["signal_sent"] is False,
            "The geometry worker must actually exit and be reaped",
        )
        ref = prefix.reference(OUTPUT / "worker/result.json")
        branch = prefix.read(ref)
        prefix.require(
            branch["complete"] is branch["passed"] is True and len(branch["analysis"]["periods"]) == 47,
            "The complete fixed geometry roster is required",
        )
        prefix.require(
            source_refs() == sources and snapshot(ROOT)["sources"] == before, "Geometry source identity changed"
        )
        for path, digest in inputs["reference_table_geometry"]["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        result.update(complete=True, passed=True, sources_unchanged=True, branch_ref=ref)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(OUTPUT / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    elif args.execute:
        execute()
    else:
        worker()


if __name__ == "__main__":
    main()
