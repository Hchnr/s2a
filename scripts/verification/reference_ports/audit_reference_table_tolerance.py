"""Independently reconstruct saved mesh witnesses and their pose-tolerance bounds."""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

from s2a.geometry import inverse, transform, validate_transform
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.wam.training.native.native_training import state_sha256
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.tracking.response_models import reference_table_tolerance as previous

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/response_models/reference_table_tolerance_audit_20261111.json"
CONFIG_SHA = "1f3025299577eea0909aef7121f5ecc96572ec340f1bf75d556629c717d8939c"
PLAN = ROOT / "plans/v0.0.1/tracking/response_models/reference_table_tolerance_audit_20261111.md"
OUTPUT = ROOT / "outputs/reference_table_tolerance_audit_20261111"
DRY = ROOT / "outputs/reference_table_tolerance_audit_dry_20261111/result.json"
QA = ROOT / "outputs/qa/reference_table_tolerance_audit_20261111/result.json"
MODULE = "scripts.verification.reference_ports.audit_reference_table_tolerance"


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__).resolve(),
            Path(previous.__file__).resolve(),
            ROOT / "src/s2a/research/joint_control/tracking/pose_box_collision_certificate.py",
            ROOT / "tests/research/response_models/test_reference_table_tolerance_audit.py",
        )
    ]


def load_inputs():
    prefix.environment()
    inputs = previous.load_inputs()
    consumed = dict(inputs["reference_table_tolerance"]["source_sha256"])
    config = prefix.read({"path": str(CONFIG), "sha256": CONFIG_SHA}, consumed)
    prefix.require(
        config["schema"] == "s2a.reference-table-tolerance-audit.v1"
        and config["output"] == str(OUTPUT.relative_to(ROOT))
        and config["steps"] == list(range(120, 127))
        and config["position_tolerance_m"] == 0.01
        and config["rotation_tolerance_rad"] == 0.05
        and config["certificate_margin_m"] == 1e-9
        and config["max_models"] == 1
        and config["max_forward_queries"] == 7
        and config["physics_steps"] == config["world_model_calls"] == config["fresh_cases"] == 0
        and config["automatic_retry"] is False,
        "The exact independent certificate audit budget is required",
    )
    refs = {key: {"path": str(ROOT / config[key]), "sha256": config[key + "_sha256"]} for key in ("owner", "worker")}
    owner, worker_result = (prefix.read(refs[key], consumed) for key in refs)
    prefix.require(
        owner["complete"] is owner["passed"] is owner["sources_unchanged"] is True
        and worker_result["complete"] is worker_result["passed"] is True
        and owner["branch_ref"] == refs["worker"]
        and owner["control_qualified"] is worker_result["control_qualified"] is False
        and [row["step"] for row in worker_result["analysis"]["periods"]] == config["steps"]
        and worker_result["stats"]["models_completed"] == 1
        and worker_result["stats"]["forward_queries_completed"] == 14
        and worker_result["stats"]["physics_steps_attempted"] == 0,
        "The complete source-pinned original static study is required",
    )
    inputs["reference_table_tolerance_audit"] = {
        "config": config,
        "refs": refs,
        "original": worker_result,
        "source_sha256": consumed,
    }
    return inputs


def input_receipt(inputs):
    item = inputs["reference_table_tolerance_audit"]
    return {
        "prior": previous.input_receipt(inputs),
        "config_ref": {"path": str(CONFIG), "sha256": CONFIG_SHA},
        "refs": item["refs"],
        "source_sha256": item["source_sha256"],
    }


def target_finger_pose(current_site, current_finger, target_tcp, site_from_tcp):
    """Reconstruct target finger placement without moving the robot base."""
    return (
        validate_transform(target_tcp)
        @ inverse(site_from_tcp)
        @ inverse(current_site)
        @ validate_transform(current_finger)
    )


def independent_scores(vertices, finger_pose, tcp_pose, table_pose, half_size, position_tolerance, rotation_tolerance):
    """Recompute each point's box-face slack using scalar operations."""
    points = np.asarray(vertices, dtype=float)
    finger_pose, tcp_pose, table_pose = map(validate_transform, (finger_pose, tcp_pose, table_pose))
    half_size = np.asarray(half_size, dtype=float)
    prefix.require(
        points.ndim == 2
        and points.shape[1] == 3
        and len(points) > 0
        and half_size.shape == (3,)
        and np.isfinite(points).all()
        and np.isfinite(half_size).all()
        and np.all(half_size > 0)
        and math.isfinite(position_tolerance)
        and position_tolerance >= 0
        and math.isfinite(rotation_tolerance)
        and 0 <= rotation_tolerance <= math.pi,
        "Finite mesh, positive box and bounded pose tolerances are required",
    )
    results = []
    for point in points:
        world = finger_pose[:3, :3] @ point + finger_pose[:3, 3]
        local = table_pose[:3, :3].T @ (world - table_pose[:3, 3])
        face = [float(half_size[i] - abs(local[i])) for i in range(3)]
        radius = math.dist(world.tolist(), tcp_pose[:3, 3].tolist())
        bound = position_tolerance + 2 * radius * math.sin(rotation_tolerance / 2)
        results.append(
            {
                "world": world,
                "table_local": local,
                "face_margins_m": face,
                "inside_margin_m": min(face),
                "tcp_radius_m": radius,
                "motion_bound_m": bound,
                "certificate_slack_m": min(face) - bound,
            }
        )
    return results


def verify_certificate(points, saved, *, margin):
    index = saved["vertex_index"]
    prefix.require(
        type(index) is int and 0 <= index < len(points) and saved["vertices_checked"] == len(points),
        "Witness index and full mesh",
    )
    witness = points[index]
    best = max(point["certificate_slack_m"] for point in points)
    prefix.require(
        abs(witness["certificate_slack_m"] - best) <= 1e-12, "The saved witness must maximize the independent bound"
    )
    for key in ("inside_margin_m", "tcp_radius_m", "motion_bound_m", "certificate_slack_m"):
        np.testing.assert_allclose(saved[key], witness[key], atol=1e-12, rtol=0)
    np.testing.assert_allclose(saved["vertex_world"], witness["world"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(saved["vertex_table_local"], witness["table_local"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(saved["face_margins_m"], witness["face_margins_m"], atol=1e-12, rtol=0)
    prefix.require(
        saved["required_margin_m"] >= margin
        and saved["fixed_gripper_configuration"] is True
        and saved["unproven_means_collision_free"] is False
        and saved["certified"] is (witness["certificate_slack_m"] > saved["required_margin_m"]),
        "The independent strict certificate decision must match",
    )
    return {
        "witness_index": index,
        "independent_slack_m": witness["certificate_slack_m"],
        "inside_margin_m": witness["inside_margin_m"],
        "motion_bound_m": witness["motion_bound_m"],
        "certified": saved["certified"],
    }


def analyze(inputs, stats):
    import mujoco

    originals = {name: getattr(mujoco, name) for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward")}

    def prohibited(*_args, **_kwargs):
        stats["physics_steps_attempted"] += 1
        raise RuntimeError("The independent certificate audit cannot integrate physics")

    def forward(*args, **kwargs):
        stats["forward_queries_attempted"] += 1
        prefix.require(stats["forward_queries_attempted"] <= 7, "The seven-forward budget is exhausted")
        value = originals["mj_forward"](*args, **kwargs)
        stats["forward_queries_completed"] += 1
        return value

    try:
        for name in ("mj_step", "mj_step1", "mj_step2"):
            setattr(mujoco, name, prohibited)
        mujoco.mj_forward = forward
        old = inputs["old"]["old"]["base"]["previous"]
        item, source = inputs["reference_table_tolerance_audit"], inputs["reference_table_geometry"]
        config, original = item["config"], item["original"]["analysis"]
        stats["models_attempted"] += 1
        model = mujoco.MjModel.from_xml_string(old["record_xml"])
        stats["models_completed"] += 1
        data = mujoco.MjData(model)
        table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_collision")
        finger = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "gripper0_finger2_collision")
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, old["controller"]["contract"]["controller_site"])
        prefix.require(
            min(table, finger, site) >= 0
            and model.geom_type[table] == mujoco.mjtGeom.mjGEOM_BOX
            and model.geom_type[finger] == mujoco.mjtGeom.mjGEOM_MESH
            and (
                (model.geom_contype[table] & model.geom_conaffinity[finger])
                or (model.geom_contype[finger] & model.geom_conaffinity[table])
            ),
            "The original enabled table/finger geometry and controller site are required",
        )
        mesh = int(model.geom_dataid[finger])
        prefix.require(mesh >= 0, "The finger must own a compiled mesh")
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        vertices = np.asarray(model.mesh_vert[start : start + count], dtype=float).copy()
        prefix.require(
            vertices.shape == (count, 3) and count == original["mesh_vertices"] and np.isfinite(vertices).all(),
            "Original compiled mesh",
        )
        np.testing.assert_allclose(model.geom_size[table], original["box_half_size_m"], atol=0, rtol=0)
        site_from_tcp = inverse(np.asarray(old["controller"]["source_from_site"])) @ old["tool"]
        rows, max_transform = [], 0.0
        for step, saved in zip(config["steps"], original["periods"], strict=True):
            index = step - 80
            state, target = np.asarray(source["states"][index]), np.asarray(source["targets"][index])
            prefix.require(
                state.shape == (1 + model.nq + model.nv + model.na,) and model.na == 0, "Exact native state layout"
            )
            data.time, data.qpos[:], data.qvel[:] = state[0], state[1 : 1 + model.nq], state[1 + model.nq :]
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(data.qpos, state[1 : 1 + model.nq], atol=1e-12, rtol=0)
            np.testing.assert_allclose(data.qvel, state[1 + model.nq :], atol=1e-12, rtol=0)
            prefix.require(
                float(data.time) == float(state[0]) and state_sha256(state) == saved["state_sha256"],
                "Original state and clock",
            )
            current_site = transform(data.site_xpos[site].copy(), data.site_xmat[site].reshape(3, 3).copy())
            current_finger = transform(data.geom_xpos[finger].copy(), data.geom_xmat[finger].reshape(3, 3).copy())
            table_pose = transform(data.geom_xpos[table].copy(), data.geom_xmat[table].reshape(3, 3).copy())
            finger_pose = target_finger_pose(current_site, current_finger, target, site_from_tcp)
            np.testing.assert_allclose(table_pose, saved["table_pose"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(target, saved["target_tcp"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(finger_pose, saved["finger_geom_pose"], atol=1e-9, rtol=0)
            max_transform = max(max_transform, float(np.max(np.abs(finger_pose - saved["finger_geom_pose"]))))
            points = independent_scores(
                vertices,
                finger_pose,
                target,
                table_pose,
                model.geom_size[table],
                config["position_tolerance_m"],
                config["rotation_tolerance_rad"],
            )
            checked = verify_certificate(points, saved["certificate"], margin=config["certificate_margin_m"])
            checked.update(step=step, executed=step < 126, generated_step=120)
            rows.append(checked)
        prefix.require(stats["forward_queries_completed"] == 7, "All seven independent forwards must complete")
        return {
            "periods": rows,
            "mesh_vertices": count,
            "max_target_finger_transform_difference": max_transform,
            "independent_base_moved": False,
            "scope": "fixed recorded gripper joints against fixed table box",
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
            and stats["forward_queries_completed"] == 7
            and stats["physics_steps_attempted"] == 0,
            "The independent static budget must hold",
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
        spec = {
            "key": "certificate_audit",
            "argv": [sys.executable, "-m", MODULE, "--worker"],
            "log": OUTPUT / "worker.log",
        }
        with physical._temporary_environment(env):
            lifecycle = run_owned_workers(
                [spec],
                cwd=ROOT,
                timeout_seconds=inputs["reference_table_tolerance_audit"]["config"]["worker_timeout_seconds"],
            )
        result["workers_ref"] = publish(OUTPUT / "workers.json", lifecycle)
        prefix.require(
            lifecycle["launch_error"] is None
            and len(lifecycle["workers"]) == 1
            and lifecycle["workers"][0]["actual_wait"] is lifecycle["workers"][0]["reaped"] is True
            and lifecycle["workers"][0]["returncode"] == 0
            and lifecycle["workers"][0]["signal_sent"] is False,
            "The certificate auditor must actually exit and be reaped",
        )
        ref = prefix.reference(OUTPUT / "worker/result.json")
        branch = prefix.read(ref)
        prefix.require(
            branch["complete"] is branch["passed"] is True and len(branch["analysis"]["periods"]) == 7,
            "The seven independent comparisons are required",
        )
        prefix.require(
            source_refs() == sources and snapshot(ROOT)["sources"] == before, "Audit source identity changed"
        )
        for path, digest in inputs["reference_table_tolerance_audit"]["source_sha256"].items():
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
