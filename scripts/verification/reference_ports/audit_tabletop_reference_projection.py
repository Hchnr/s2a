"""Independently reconstruct and audit saved tabletop projection decisions."""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import inverse, transform
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.wam.training.native.native_training import state_sha256
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.tracking.response_models import tabletop_reference_projection as previous

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/response_models/tabletop_reference_projection_audit_20261111.json"
CONFIG_SHA = "06c765ef8cc82d1185fdd7aa18d9925fafa3fd59fdfccacdbb5dbc638815c3b8"
PLAN = ROOT / "plans/v0.0.1/tracking/response_models/tabletop_reference_projection_audit_20261111.md"
OUTPUT = ROOT / "outputs/tabletop_reference_projection_audit_20261111"
DRY = ROOT / "outputs/tabletop_reference_projection_audit_dry_20261111/result.json"
QA = ROOT / "outputs/qa/tabletop_reference_projection_audit_20261111/result.json"
MODULE = "scripts.verification.reference_ports.audit_tabletop_reference_projection"


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__).resolve(),
            Path(previous.__file__).resolve(),
            ROOT / "src/s2a/research/joint_control/tracking/tabletop_reference_projection.py",
            ROOT / "tests/research/response_models/test_tabletop_reference_projection_audit.py",
        )
    ]


def load_inputs():
    inputs = previous.load_inputs()
    consumed = dict(inputs["tabletop_reference_projection"]["source_sha256"])
    config = prefix.read({"path": str(CONFIG), "sha256": CONFIG_SHA}, consumed)
    prefix.require(
        config["schema"] == "s2a.tabletop-reference-projection-audit.v1"
        and config["output"] == str(OUTPUT.relative_to(ROOT))
        and (config["first_step"], config["last_step"]) == (80, 126)
        and config["max_models"] == 1
        and config["max_forward_queries"] == 47
        and config["physics_steps"] == config["world_model_calls"] == config["fresh_cases"] == 0
        and config["automatic_retry"] is False,
        "The original fixed-state, independent audit budget is required",
    )
    refs = {key: {"path": str(ROOT / config[key]), "sha256": config[key + "_sha256"]} for key in ("owner", "worker")}
    owner, worker_result = (prefix.read(refs[key], consumed) for key in refs)
    prefix.require(
        owner["complete"] is owner["passed"] is owner["sources_unchanged"] is True
        and worker_result["complete"] is worker_result["passed"] is True
        and owner["branch_ref"] == refs["worker"]
        and owner["control_qualified"] is worker_result["control_qualified"] is False
        and [row["step"] for row in worker_result["analysis"]["periods"]] == list(range(80, 127))
        and worker_result["stats"]["models_completed"] == 1
        and worker_result["stats"]["forward_queries_completed"] == 188
        and worker_result["stats"]["physics_steps_attempted"] == 0,
        "The complete source-pinned static projection study is required",
    )
    inputs["tabletop_reference_projection_audit"] = {
        "config": config,
        "refs": refs,
        "original": worker_result,
        "source_sha256": consumed,
    }
    return inputs


def input_receipt(inputs):
    item = inputs["tabletop_reference_projection_audit"]
    return {
        "prior": previous.input_receipt(inputs),
        "config_ref": {"path": str(CONFIG), "sha256": CONFIG_SHA},
        "refs": item["refs"],
        "source_sha256": item["source_sha256"],
    }


def scalar_projection(vertices, target, table, size, config, variant):
    """Recompute bounds from individual coordinates without the candidate function."""
    points, bounds, worlds = [], [], []
    for vertex in vertices:
        world = [sum(float(target[i, j]) * float(vertex[j]) for j in range(3)) + float(target[i, 3]) for i in range(3)]
        local = [sum(float(table[i, j]) * (world[i] - float(table[i, 3])) for i in range(3)) for j in range(3)]
        radius = math.sqrt(sum(float(value) ** 2 for value in vertex))
        distance = (
            variant["position_uncertainty_m"]
            + 2 * radius * math.sin(variant["rotation_uncertainty_rad"] / 2)
            + variant["shape_uncertainty_m"]
        )
        worlds.append(world)
        points.append(local)
        bounds.append(distance)
    scale = max(
        1.0,
        *(abs(value) for point in worlds for value in point),
        *(abs(float(x)) for x in table[:3, 3]),
        *(float(x) for x in size),
    )
    guard = max(config["numerical_margin_m"], 64 * np.finfo(float).eps * scale)
    overlap = all(
        min(point[axis] - distance for point, distance in zip(points, bounds, strict=True))
        <= float(size[axis]) + config["clearance_m"] + guard
        and max(point[axis] + distance for point, distance in zip(points, bounds, strict=True))
        >= -float(size[axis]) - config["clearance_m"] - guard
        for axis in (0, 1)
    )
    lower = min(point[2] - distance for point, distance in zip(points, bounds, strict=True)) - float(size[2])
    required = max(0.0, config["clearance_m"] + guard - lower) if overlap else 0.0
    accepted = required <= config["max_translation_m"]
    return {
        "overlap": overlap,
        "original_lower": lower,
        "max_displacement": max(bounds),
        "guard": guard,
        "required": required,
        "accepted": accepted,
        "status": "correction_exceeds_limit" if not accepted else ("projected" if required > 0 else "unchanged"),
    }


def check_result(saved, independently_computed, target, table, current):
    """Check both target offset and current error; refusal never becomes a target."""
    value = independently_computed
    prefix.require(
        saved["accepted"] is value["accepted"]
        and saved["status"] == value["status"]
        and saved["uncertainty_expanded_xy_overlap"] is value["overlap"]
        and saved["conditional_table_clearance_certified"] is value["accepted"]
        and saved["reference_rotation_change_rad"] == 0,
        "Independent projection decision mismatch",
    )
    for actual, expected in (
        (saved["original_height_lower_bound_m"], value["original_lower"]),
        (saved["maximum_vertex_displacement_bound_m"], value["max_displacement"]),
        (saved["numerical_guard_m"], value["guard"]),
        (saved["required_translation_m"], value["required"]),
    ):
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    if value["accepted"]:
        expected = target.copy()
        expected[:3, 3] += value["required"] * table[:3, 2]
        np.testing.assert_allclose(saved["projected_tcp"], expected, atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            saved["projected_height_lower_bound_m"], value["original_lower"] + value["required"], atol=1e-12, rtol=0
        )
        prefix.require(
            saved["projected_proxy_finger_table_contact_distances_m"] == [], "Saved static contact must be empty"
        )
        error = np.r_[
            expected[:3, 3] - current[:3, 3], Rotation.from_matrix(expected[:3, :3] @ current[:3, :3].T).as_rotvec()
        ]
        np.testing.assert_allclose(saved["current_tcp_error_to_projected"], error, atol=1e-12, rtol=0)
    else:
        prefix.require(
            saved["projected_tcp"]
            is saved["projected_height_lower_bound_m"]
            is saved["projected_proxy_finger_table_contact_distances_m"]
            is saved["current_tcp_error_to_projected"]
            is None,
            "Refusal cannot contain a projected target or a contact/pose result",
        )


def analyze(inputs, stats):
    import mujoco

    originals = {name: getattr(mujoco, name) for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward")}

    def prohibited(*_args, **_kwargs):
        stats["physics_steps_attempted"] += 1
        raise RuntimeError("The independent audit cannot integrate physics")

    def forward(*args, **kwargs):
        stats["forward_queries_attempted"] += 1
        prefix.require(stats["forward_queries_attempted"] <= 47, "The 47-forward budget is exhausted")
        value = originals["mj_forward"](*args, **kwargs)
        stats["forward_queries_completed"] += 1
        return value

    try:
        for name in ("mj_step", "mj_step1", "mj_step2"):
            setattr(mujoco, name, prohibited)
        mujoco.mj_forward = forward
        old = inputs["old"]["old"]["base"]["previous"]
        item, source = inputs["tabletop_reference_projection_audit"], inputs["reference_table_geometry"]
        config, original = inputs["tabletop_reference_projection"]["config"], item["original"]["analysis"]
        stats["models_attempted"] += 1
        model = mujoco.MjModel.from_xml_string(old["record_xml"])
        stats["models_completed"] += 1
        data = mujoco.MjData(model)
        table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, config["table_geom"])
        finger = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, config["finger_geom"])
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
        mesh_vertices = np.asarray(model.mesh_vert[start : start + count], dtype=float).copy()
        prefix.require(
            count == original["mesh_vertex_count"] and np.isfinite(mesh_vertices).all(), "Original compiled mesh"
        )
        site_from_tcp = inverse(np.asarray(old["controller"]["source_from_site"])) @ old["tool"]
        rows, maximum = [], 0.0
        for index, (state, target, saved) in enumerate(
            zip(source["states"], source["targets"], original["periods"], strict=True)
        ):
            state, target = np.asarray(state), np.asarray(target)
            prefix.require(
                state.shape == (1 + model.nq + model.nv + model.na,) and model.na == 0, "Exact native state layout"
            )
            data.time, data.qpos[:], data.qvel[:] = state[0], state[1 : 1 + model.nq], state[1 + model.nq :]
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(data.qpos, state[1 : 1 + model.nq], atol=1e-12, rtol=0)
            np.testing.assert_allclose(data.qvel, state[1 + model.nq :], atol=1e-12, rtol=0)
            step = 80 + index
            prefix.require(
                float(data.time) == float(state[0])
                and state_sha256(state) == saved["state_sha256"]
                and saved["step"] == step
                and saved["executed"] is (step < 126),
                "Original state, step and clock",
            )
            current_site = transform(data.site_xpos[site].copy(), data.site_xmat[site].reshape(3, 3).copy())
            current = current_site @ site_from_tcp
            finger_pose = transform(data.geom_xpos[finger].copy(), data.geom_xmat[finger].reshape(3, 3).copy())
            table_pose = transform(data.geom_xpos[table].copy(), data.geom_xmat[table].reshape(3, 3).copy())
            reconstructed = np.array(
                [
                    current[:3, :3].T @ (finger_pose[:3, :3] @ point + finger_pose[:3, 3] - current[:3, 3])
                    for point in mesh_vertices
                ]
            )
            np.testing.assert_allclose(target, saved["target_tcp"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(current, saved["current_tcp"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(table_pose, saved["table_pose"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(model.geom_size[table], saved["table_half_size_m"], atol=0, rtol=0)
            np.testing.assert_allclose(reconstructed, saved["vertices_tcp"], atol=1e-12, rtol=0)
            maximum = max(maximum, float(np.max(np.abs(reconstructed - saved["vertices_tcp"]))))
            error = np.r_[
                target[:3, 3] - current[:3, 3], Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec()
            ]
            np.testing.assert_allclose(saved["current_tcp_error_to_original"], error, atol=1e-12, rtol=0)
            candidates = {}
            for variant in config["candidates"]:
                computed = scalar_projection(reconstructed, target, table_pose, model.geom_size[table], config, variant)
                check_result(saved["candidates"][variant["name"]], computed, target, table_pose, current)
                candidates[variant["name"]] = {
                    "status": computed["status"],
                    "reference_translation_m": computed["required"],
                }
            rows.append(dict(step=step, executed=step < 126, candidates=candidates))
        prefix.require(stats["forward_queries_completed"] == 47, "All independent static forwards must complete")
        return dict(
            periods=rows,
            mesh_vertex_count=count,
            max_tcp_vertex_difference=maximum,
            independent_base_moved=False,
            scope="fixed recorded finger mesh and calibrated table only; no physics or feedback rollout",
        )
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
    stats = dict(
        models_attempted=0,
        models_completed=0,
        forward_queries_attempted=0,
        forward_queries_completed=0,
        physics_steps_attempted=0,
    )
    result = dict(
        complete=False,
        passed=False,
        stats=stats,
        world_model_calls=0,
        fresh_cases=0,
        control_qualified=False,
        automatic_retry=False,
    )
    try:
        result["analysis"] = analyze(inputs, stats)
        prefix.require(
            stats["models_completed"] == 1
            and stats["forward_queries_completed"] == 47
            and stats["physics_steps_attempted"] == 0,
            "Independent audit budget must hold",
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
            dict(
                complete=True,
                passed=True,
                inputs=input_receipt(inputs),
                source_refs=source_refs(),
                models=0,
                forward_queries=0,
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
            ),
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
    registration = publish(OUTPUT / "registration.json", dict(inputs=input_receipt(inputs), source_refs=sources))
    result = dict(
        complete=False,
        passed=False,
        registration_ref=registration,
        world_model_calls=0,
        fresh_cases=0,
        control_qualified=False,
        automatic_retry=False,
    )
    try:
        env = physical._cpu_environment(OUTPUT / "worker")
        env["MUJOCO_GL"] = "disable"
        spec = dict(key="audit", argv=[sys.executable, "-m", MODULE, "--worker"], log=OUTPUT / "worker.log")
        with physical._temporary_environment(env):
            lifecycle = run_owned_workers(
                [spec],
                cwd=ROOT,
                timeout_seconds=inputs["tabletop_reference_projection_audit"]["config"]["worker_timeout_seconds"],
            )
        result["workers_ref"] = publish(OUTPUT / "workers.json", lifecycle)
        prefix.require(
            lifecycle["launch_error"] is None
            and len(lifecycle["workers"]) == 1
            and lifecycle["workers"][0]["actual_wait"] is lifecycle["workers"][0]["reaped"] is True
            and lifecycle["workers"][0]["returncode"] == 0
            and lifecycle["workers"][0]["signal_sent"] is False,
            "Audit worker must exit and be reaped",
        )
        ref = prefix.reference(OUTPUT / "worker/result.json")
        branch = prefix.read(ref)
        prefix.require(
            branch["complete"] is branch["passed"] is True and len(branch["analysis"]["periods"]) == 47,
            "Complete independent audit required",
        )
        prefix.require(
            source_refs() == sources and snapshot(ROOT)["sources"] == before, "Audit source identity changed"
        )
        for path, digest in inputs["tabletop_reference_projection_audit"]["source_sha256"].items():
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
