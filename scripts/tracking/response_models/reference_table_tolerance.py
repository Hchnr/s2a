"""Bound table collision for saved TCP targets over the original pose tolerances."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from s2a.geometry import inverse, transform
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.joint_control.tracking.pose_box_collision_certificate import fixed_vertex_box_certificate
from s2a.research.wam.training.native.native_training import state_sha256
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.tracking.response_models import reference_table_geometry as geometry

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/response_models/reference_table_tolerance_20261111.json"
CONFIG_SHA = "c32952d60f35b820945d3d2f2557e5534f87c91771f1c60cb8e36415790e0a90"
PLAN = ROOT / "plans/v0.0.1/tracking/response_models/reference_table_tolerance_20261111.md"
OUTPUT = ROOT / "outputs/reference_table_tolerance_20261111"
DRY = ROOT / "outputs/reference_table_tolerance_dry_20261111/result.json"
QA = ROOT / "outputs/qa/reference_table_tolerance_20261111/result.json"
MODULE = "scripts.tracking.response_models.reference_table_tolerance"


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__).resolve(),
            Path(geometry.__file__).resolve(),
            ROOT / "src/s2a/research/joint_control/tracking/pose_box_collision_certificate.py",
            ROOT / "src/s2a/research/wam/tracking/preview/development_reference_geometry.py",
            ROOT / "src/s2a/research/wam/tracking/preview/development_reference_geometry_roundoff.py",
            ROOT / "tests/research/response_models/test_reference_table_tolerance.py",
            ROOT / "tests/research/joint_control/tracking/test_pose_box_collision_certificate.py",
        )
    ]


def load_inputs():
    inputs = geometry.load_inputs()
    consumed = dict(inputs["reference_table_geometry"]["source_sha256"])
    config = prefix.read({"path": str(CONFIG), "sha256": CONFIG_SHA}, consumed)
    prefix.require(
        config["schema"] == "s2a.reference-table-tolerance.v1"
        and config["output"] == str(OUTPUT.relative_to(ROOT))
        and config["case_index"] == 8
        and config["seed"] == 42
        and config["steps"] == list(range(120, 127))
        and config["position_tolerance_m"] == 0.01
        and config["rotation_tolerance_rad"] == 0.05
        and config["certificate_margin_m"] == 1e-9
        and config["max_models"] == 1
        and config["max_forward_queries"] == 14
        and config["physics_steps"] == config["world_model_calls"] == config["fresh_cases"] == 0
        and config["automatic_retry"] is False,
        "The exact tolerance-certificate scope and static budget are required",
    )
    refs = {
        key: {"path": str(ROOT / config[key]), "sha256": config[key + "_sha256"]}
        for key in ("geometry_audit", "geometry_worker")
    }
    audit, worker = (prefix.read(refs[key], consumed) for key in refs)
    prefix.require(
        audit["complete"] is audit["passed"] is worker["complete"] is worker["passed"] is True
        and audit["control_qualified"] is worker["control_qualified"] is False
        and audit["summary"]["target_intersection_steps"] == config["steps"]
        and len(worker["analysis"]["periods"]) == 47,
        "The source-pinned geometry and independent audit are required",
    )
    rows = worker["analysis"]["periods"][40:47]
    prefix.require([row["step"] for row in rows] == config["steps"], "The seven registered target rows are required")
    inputs["reference_table_tolerance"] = {
        "config": config,
        "refs": refs,
        "previous_rows": rows,
        "source_sha256": consumed,
    }
    return inputs


def input_receipt(inputs):
    item = inputs["reference_table_tolerance"]
    return {
        "prior": geometry.input_receipt(inputs),
        "config_ref": {"path": str(CONFIG), "sha256": CONFIG_SHA},
        "refs": item["refs"],
        "source_sha256": item["source_sha256"],
    }


def native_geom_pose(data, geom):
    return transform(data.geom_xpos[geom].copy(), data.geom_xmat[geom].reshape(3, 3).copy())


def evaluate_target(probe, state, target_tcp, site_from_tcp, table, finger, vertices, config, prior_row):
    """Evaluate every compiled collision vertex after the registered static placement."""
    probe.restore(state)
    data = probe.data
    table_pose = native_geom_pose(data, table)
    qpos, qvel, timestamp = data.qpos.copy(), data.qvel.copy(), float(data.time)
    probe.place(target_tcp @ inverse(site_from_tcp))
    np.testing.assert_allclose(native_geom_pose(data, table), table_pose, atol=1e-12, rtol=0)
    np.testing.assert_allclose(data.qpos, qpos, atol=1e-12, rtol=0)
    np.testing.assert_allclose(data.qvel, qvel, atol=1e-12, rtol=0)
    prefix.require(float(data.time) == timestamp, "Static geometry must not advance physical time")
    measured_contacts = geometry.matching_contacts(data, table, finger)
    np.testing.assert_allclose(
        sorted(measured_contacts),
        sorted(prior_row["target_proxy_finger_table_contact_distances_m"]),
        atol=1e-10,
        rtol=0,
    )
    finger_pose = native_geom_pose(data, finger)
    world_vertices = vertices @ finger_pose[:3, :3].T + finger_pose[:3, 3]
    proof = fixed_vertex_box_certificate(
        world_vertices,
        target_tcp,
        table_pose,
        probe.model.geom_size[table],
        position_tolerance=config["position_tolerance_m"],
        rotation_tolerance=config["rotation_tolerance_rad"],
        margin=config["certificate_margin_m"],
    )
    prefix.require(
        not proof["certified"] or bool(measured_contacts),
        "A certified intersecting vertex requires the recorded collision pair",
    )
    return {
        "target_proxy_finger_table_contact_distances_m": measured_contacts,
        "certificate": proof,
        "state_sha256": state_sha256(state),
        "table_pose": table_pose,
        "target_tcp": target_tcp,
        "finger_geom_pose": finger_pose,
        "qpos_change_magnitude": float(np.max(np.abs(data.qpos - qpos))),
    }


def analyze(inputs, stats):
    import mujoco

    from s2a.research.wam.tracking.preview.development_reference_geometry_roundoff import RoundoffBoundedProbe

    originals = {name: getattr(mujoco, name) for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward")}

    def prohibited(*_args, **_kwargs):
        stats["physics_steps_attempted"] += 1
        raise RuntimeError("The tolerance certificate cannot integrate physics")

    def forward(*args, **kwargs):
        stats["forward_queries_attempted"] += 1
        prefix.require(stats["forward_queries_attempted"] <= 14, "The 14-forward budget is exhausted")
        value = originals["mj_forward"](*args, **kwargs)
        stats["forward_queries_completed"] += 1
        return value

    try:
        for name in ("mj_step", "mj_step1", "mj_step2"):
            setattr(mujoco, name, prohibited)
        mujoco.mj_forward = forward
        old = inputs["old"]["old"]["base"]["previous"]
        item = inputs["reference_table_tolerance"]
        data = inputs["reference_table_geometry"]
        config = item["config"]
        stats["models_attempted"] += 1
        model = mujoco.MjModel.from_xml_string(old["record_xml"])
        stats["models_completed"] += 1
        probe = RoundoffBoundedProbe(model, inputs["old"]["old"]["base"]["config"]["target_object"], budget=14)
        prefix.require(
            old["controller"]["contract"]["controller_site"] == "gripper0_grip_site", "Controller site identity"
        )
        table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, config["table_geom"])
        finger = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, config["finger_geom"])
        prefix.require(
            min(table, finger) >= 0
            and model.geom_type[table] == mujoco.mjtGeom.mjGEOM_BOX
            and model.geom_type[finger] == mujoco.mjtGeom.mjGEOM_MESH
            and (
                (model.geom_contype[table] & model.geom_conaffinity[finger])
                or (model.geom_contype[finger] & model.geom_conaffinity[table])
            ),
            "The actual enabled box/mesh collision pair is required",
        )
        mesh = int(model.geom_dataid[finger])
        prefix.require(mesh >= 0, "The finger must reference a compiled collision mesh")
        address, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        vertices = np.asarray(model.mesh_vert[address : address + count], dtype=float).copy()
        prefix.require(
            vertices.shape == (count, 3) and count >= 4 and np.isfinite(vertices).all(), "Finite compiled mesh vertices"
        )
        site_from_tcp = inverse(np.asarray(old["controller"]["source_from_site"])) @ old["tool"]
        rows = []
        for step, previous in zip(config["steps"], item["previous_rows"], strict=True):
            index = step - 80
            row = evaluate_target(
                probe,
                data["states"][index],
                data["targets"][index],
                site_from_tcp,
                table,
                finger,
                vertices,
                config,
                previous,
            )
            row.update(step=step, generated_step=120, executed=step < 126)
            rows.append(row)
        prefix.require(
            probe.queries == stats["forward_queries_completed"] == 14, "Every registered static query must complete"
        )
        return {
            "periods": rows,
            "mesh_vertices": count,
            "box_half_size_m": model.geom_size[table].copy(),
            "max_object_rotation_change": probe.max_object_rotation_change,
            "max_qpos_change_during_placement": probe.max_qpos_change_during_placement,
            "scope": "fixed current gripper configuration and one compiled finger collision mesh against a fixed table box",
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
            and stats["forward_queries_completed"] == 14
            and stats["physics_steps_attempted"] == 0,
            "The registered static budget must hold",
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
        spec = {"key": "certificate", "argv": [sys.executable, "-m", MODULE, "--worker"], "log": OUTPUT / "worker.log"}
        with physical._temporary_environment(env):
            lifecycle = run_owned_workers(
                [spec],
                cwd=ROOT,
                timeout_seconds=inputs["reference_table_tolerance"]["config"]["worker_timeout_seconds"],
            )
        result["workers_ref"] = publish(OUTPUT / "workers.json", lifecycle)
        prefix.require(
            lifecycle["launch_error"] is None
            and len(lifecycle["workers"]) == 1
            and lifecycle["workers"][0]["actual_wait"] is lifecycle["workers"][0]["reaped"] is True
            and lifecycle["workers"][0]["returncode"] == 0
            and lifecycle["workers"][0]["signal_sent"] is False,
            "The certificate worker must actually exit and be reaped",
        )
        ref = prefix.reference(OUTPUT / "worker/result.json")
        branch = prefix.read(ref)
        prefix.require(
            branch["complete"] is branch["passed"] is True and len(branch["analysis"]["periods"]) == 7,
            "The seven registered certificates are required",
        )
        prefix.require(
            source_refs() == sources and snapshot(ROOT)["sources"] == before, "Certificate source identity changed"
        )
        for path, digest in inputs["reference_table_tolerance"]["source_sha256"].items():
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
