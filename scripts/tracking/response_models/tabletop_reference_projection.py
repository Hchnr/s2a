"""Evaluate a bounded table projection on previously recorded development states."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from s2a.geometry import inverse, pose_error, transform
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.joint_control.tracking.tabletop_reference_projection import (
    collision_vertices_in_tcp,
    project_tabletop_reference,
)
from s2a.research.wam.training.native.native_training import state_sha256
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.tracking.response_models import reference_table_geometry as geometry

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/response_models/tabletop_reference_projection_20261111.json"
CONFIG_SHA = "ca3916bb1e730791e15da79695e2632b13d03b244c0fa6f7e1a09f298164d85a"
PLAN = ROOT / "plans/v0.0.1/tracking/response_models/tabletop_reference_projection_20261111.md"
OUTPUT = ROOT / "outputs/tabletop_reference_projection_20261111"
DRY = ROOT / "outputs/tabletop_reference_projection_dry_20261111/result.json"
QA = ROOT / "outputs/qa/tabletop_reference_projection_20261111/result.json"
MODULE = "scripts.tracking.response_models.tabletop_reference_projection"


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__).resolve(),
            Path(geometry.__file__).resolve(),
            ROOT / "src/s2a/research/joint_control/tracking/tabletop_reference_projection.py",
            ROOT / "src/s2a/research/wam/tracking/preview/development_reference_geometry.py",
            ROOT / "src/s2a/research/wam/tracking/preview/development_reference_geometry_roundoff.py",
            ROOT / "tests/research/response_models/test_tabletop_reference_projection_study.py",
            ROOT / "tests/research/joint_control/tracking/test_tabletop_reference_projection.py",
        )
    ]


def load_inputs():
    inputs = geometry.load_inputs()
    consumed = dict(inputs["reference_table_geometry"]["source_sha256"])
    config = prefix.read({"path": str(CONFIG), "sha256": CONFIG_SHA}, consumed)
    prefix.require(
        config["schema"] == "s2a.tabletop-reference-projection.v1"
        and config["output"] == str(OUTPUT.relative_to(ROOT))
        and config["case_index"] == 8
        and config["seed"] == 42
        and (config["first_step"], config["last_step"]) == (80, 126)
        and (config["table_geom"], config["finger_geom"]) == ("table_collision", "gripper0_finger2_collision")
        and (config["clearance_m"], config["numerical_margin_m"], config["max_translation_m"]) == (0.001, 1e-9, 0.05)
        and config["candidates"]
        == [
            {
                "name": "nominal",
                "position_uncertainty_m": 0.0,
                "rotation_uncertainty_rad": 0.0,
                "shape_uncertainty_m": 0.0,
            },
            {
                "name": "pose_bounded",
                "position_uncertainty_m": 0.01,
                "rotation_uncertainty_rad": 0.05,
                "shape_uncertainty_m": 0.0,
            },
        ]
        and config["max_models"] == 1
        and config["max_forward_queries"] == 188
        and config["physics_steps"] == config["world_model_calls"] == config["fresh_cases"] == 0
        and config["automatic_retry"] is False,
        "The fixed projection candidates, original sources, and static budget are required",
    )
    refs = {
        key: {"path": str(ROOT / config[key]), "sha256": config[key + "_sha256"]}
        for key in ("geometry_audit", "tolerance_audit")
    }
    audit, tolerance = (prefix.read(refs[key], consumed) for key in refs)
    prefix.require(
        audit["complete"] is audit["passed"] is tolerance["complete"] is tolerance["passed"] is True
        and audit["control_qualified"] is tolerance["control_qualified"] is False
        and audit["summary"]["target_intersection_steps"] == list(range(120, 127)),
        "The independent fixed-reference geometry audits must pass",
    )
    owner = prefix.read(audit["owner_ref"], consumed)
    prefix.require(owner["complete"] is owner["passed"] is True, "Original geometry owner receipt required")
    previous = prefix.read(owner["branch_ref"], consumed)
    prefix.require(
        previous["complete"] is previous["passed"] is True and len(previous["analysis"]["periods"]) == 47,
        "All 47 earlier static geometry rows must be available",
    )
    inputs["tabletop_reference_projection"] = {
        "config": config,
        "refs": refs,
        "previous_rows": previous["analysis"]["periods"],
        "source_sha256": consumed,
    }
    return inputs


def input_receipt(inputs):
    item = inputs["tabletop_reference_projection"]
    return {
        "prior": geometry.input_receipt(inputs),
        "config_ref": {"path": str(CONFIG), "sha256": CONFIG_SHA},
        "refs": item["refs"],
        "source_sha256": item["source_sha256"],
    }


def geom_pose(data, geom):
    return transform(data.geom_xpos[geom].copy(), data.geom_xmat[geom].reshape(3, 3).copy())


def inspect_state(probe, state, target, site_from_tcp, table, finger, mesh_vertices, config, prior_row):
    """Use only current geometry and one already-generated reference per decision."""
    probe.restore(state)
    data = probe.data
    current_tcp = probe.site_transform() @ site_from_tcp
    table_pose = geom_pose(data, table)
    finger_pose = geom_pose(data, finger)
    vertices = collision_vertices_in_tcp(mesh_vertices, current_tcp, finger_pose)
    actual_vertices = mesh_vertices @ finger_pose[:3, :3].T + finger_pose[:3, 3]
    np.testing.assert_allclose(
        vertices @ current_tcp[:3, :3].T + current_tcp[:3, 3], actual_vertices, atol=1e-9, rtol=0
    )
    qpos, qvel, timestamp = data.qpos.copy(), data.qvel.copy(), float(data.time)

    def place_and_check(tcp):
        probe.place(tcp @ inverse(site_from_tcp))
        np.testing.assert_allclose(probe.site_transform() @ site_from_tcp, tcp, atol=1e-9, rtol=0)
        np.testing.assert_allclose(geom_pose(data, table), table_pose, atol=1e-12, rtol=0)
        np.testing.assert_allclose(data.qpos, qpos, atol=1e-12, rtol=0)
        np.testing.assert_allclose(data.qvel, qvel, atol=1e-12, rtol=0)
        prefix.require(float(data.time) == timestamp, "Static placement must not advance physical time")
        actual_pose = geom_pose(data, finger)
        placed = mesh_vertices @ actual_pose[:3, :3].T + actual_pose[:3, 3]
        expected = vertices @ tcp[:3, :3].T + tcp[:3, 3]
        np.testing.assert_allclose(placed, expected, atol=1e-9, rtol=0)
        return geometry.matching_contacts(data, table, finger)

    original_contacts = place_and_check(target)
    np.testing.assert_allclose(
        sorted(original_contacts),
        sorted(prior_row["target_proxy_finger_table_contact_distances_m"]),
        atol=1e-10,
        rtol=0,
    )
    candidates = {}
    for variant in config["candidates"]:
        answer = project_tabletop_reference(
            target,
            vertices,
            table_pose,
            probe.model.geom_size[table],
            position_uncertainty=variant["position_uncertainty_m"],
            rotation_uncertainty=variant["rotation_uncertainty_rad"],
            shape_uncertainty=variant["shape_uncertainty_m"],
            clearance=config["clearance_m"],
            numerical_margin=config["numerical_margin_m"],
            max_translation=config["max_translation_m"],
        )
        if answer["accepted"]:
            projected = answer["projected_tcp"]
            contacts = place_and_check(projected)
            prefix.require(not contacts, "An accepted static target must have no table/finger contact")
            answer["current_tcp_error_to_projected"] = pose_error(projected, current_tcp)
            answer["projected_proxy_finger_table_contact_distances_m"] = contacts
        else:
            answer["current_tcp_error_to_projected"] = None
            answer["projected_proxy_finger_table_contact_distances_m"] = None
        candidates[variant["name"]] = answer
    return {
        "state_sha256": state_sha256(state),
        "current_tcp": current_tcp,
        "target_tcp": target,
        "vertices_tcp": vertices,
        "table_pose": table_pose,
        "table_half_size_m": probe.model.geom_size[table].copy(),
        "current_tcp_error_to_original": pose_error(target, current_tcp),
        "original_proxy_finger_table_contact_distances_m": original_contacts,
        "candidates": candidates,
    }


def analyze(inputs, stats):
    import mujoco

    from s2a.research.wam.tracking.preview.development_reference_geometry_roundoff import RoundoffBoundedProbe

    originals = {name: getattr(mujoco, name) for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward")}

    def prohibited(*_args, **_kwargs):
        stats["physics_steps_attempted"] += 1
        raise RuntimeError("Static projection cannot integrate physics")

    def forward(*args, **kwargs):
        stats["forward_queries_attempted"] += 1
        prefix.require(stats["forward_queries_attempted"] <= 188, "The 188-forward budget is exhausted")
        result = originals["mj_forward"](*args, **kwargs)
        stats["forward_queries_completed"] += 1
        return result

    try:
        for name in ("mj_step", "mj_step1", "mj_step2"):
            setattr(mujoco, name, prohibited)
        mujoco.mj_forward = forward
        old = inputs["old"]["old"]["base"]["previous"]
        item = inputs["tabletop_reference_projection"]
        data = inputs["reference_table_geometry"]
        config = item["config"]
        stats["models_attempted"] += 1
        model = mujoco.MjModel.from_xml_string(old["record_xml"])
        stats["models_completed"] += 1
        probe = RoundoffBoundedProbe(model, inputs["old"]["old"]["base"]["config"]["target_object"], budget=188)
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
            "An enabled actual box/mesh collision pair is required",
        )
        mesh = int(model.geom_dataid[finger])
        prefix.require(mesh >= 0, "The finger must reference a compiled collision mesh")
        address, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        vertices = np.asarray(model.mesh_vert[address : address + count], dtype=float).copy()
        prefix.require(
            vertices.shape == (count, 3) and count >= 4 and np.isfinite(vertices).all(), "Compiled mesh vertices"
        )
        site_from_tcp = inverse(np.asarray(old["controller"]["source_from_site"])) @ old["tool"]
        rows = []
        for offset, (state, target, previous) in enumerate(
            zip(data["states"], data["targets"], item["previous_rows"], strict=True)
        ):
            step = 80 + offset
            prefix.require(
                previous["step"] == step and previous["state_sha256"] == state_sha256(state), "Saved state identity"
            )
            row = inspect_state(probe, state, target, site_from_tcp, table, finger, vertices, config, previous)
            row.update(step=step, generated_step=80 + 8 * (offset // 8), executed=step < 126)
            rows.append(row)
        accepted = sum(int(candidate["accepted"]) for row in rows for candidate in row["candidates"].values())
        prefix.require(
            probe.queries == stats["forward_queries_completed"] == 94 + accepted, "Every static query must complete"
        )
        return {
            "periods": rows,
            "mesh_vertex_count": count,
            "max_object_rotation_change": probe.max_object_rotation_change,
            "max_qpos_change_during_placement": probe.max_qpos_change_during_placement,
            "scope": "recorded states and references only; fixed finger shape and static proxy placements, no feedback rollout",
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
            and stats["forward_queries_completed"] <= 188
            and stats["physics_steps_attempted"] == 0,
            "The static budget must hold",
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
        spec = dict(key="projection", argv=[sys.executable, "-m", MODULE, "--worker"], log=OUTPUT / "worker.log")
        with physical._temporary_environment(env):
            lifecycle = run_owned_workers(
                [spec],
                cwd=ROOT,
                timeout_seconds=inputs["tabletop_reference_projection"]["config"]["worker_timeout_seconds"],
            )
        result["workers_ref"] = publish(OUTPUT / "workers.json", lifecycle)
        prefix.require(
            lifecycle["launch_error"] is None
            and len(lifecycle["workers"]) == 1
            and lifecycle["workers"][0]["actual_wait"] is lifecycle["workers"][0]["reaped"] is True
            and lifecycle["workers"][0]["returncode"] == 0
            and lifecycle["workers"][0]["signal_sent"] is False,
            "Worker must exit and be reaped",
        )
        ref = prefix.reference(OUTPUT / "worker/result.json")
        branch = prefix.read(ref)
        prefix.require(
            branch["complete"] is branch["passed"] is True and len(branch["analysis"]["periods"]) == 47,
            "Complete static roster required",
        )
        prefix.require(
            source_refs() == sources and snapshot(ROOT)["sources"] == before, "Projection source identity changed"
        )
        for path, digest in inputs["tabletop_reference_projection"]["source_sha256"].items():
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
