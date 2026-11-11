"""Study saved feedback endpoints against two fixed boxes without physical integration."""

import argparse
import hashlib
from pathlib import Path

import numpy as np

from s2a.geometry import inverse, transform
from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.joint_control.tracking.parallel_box_reference_projection import (
    project_parallel_boxes_reference,
    single_slide_shape_bound,
)
from s2a.research.joint_control.tracking.tabletop_reference_projection import collision_vertices_in_tcp
from s2a.research.wam.training.native.native_training import state_sha256
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_early_tabletop_live_feedback as live
from scripts.tracking.response_models import reference_table_geometry as geometry

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/response_models/parallel_boxes_shape_bound_20261111.json"
CONFIG_SHA = "58748c928510b9cd8cc3428e48acff8163e037f35361030ec1e69da08cb4f3c7"
PLAN = ROOT / "plans/v0.0.1/tracking/response_models/parallel_boxes_shape_bound_20261111.md"
OUTPUT = ROOT / "outputs/parallel_boxes_shape_bound_20261111"
DRY = ROOT / "outputs/parallel_boxes_shape_bound_dry_20261111/result.json"
QA = ROOT / "outputs/qa/parallel_boxes_study_20261111/result.json"


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__).resolve(),
            ROOT / "src/s2a/research/joint_control/tracking/parallel_box_reference_projection.py",
            ROOT / "src/s2a/research/joint_control/tracking/tabletop_reference_projection.py",
            ROOT / "tests/research/response_models/test_parallel_boxes_shape_bound_study.py",
            ROOT / "tests/research/joint_control/tracking/test_parallel_box_reference_projection.py",
        )
    ]


def load_inputs():
    inputs = geometry.load_inputs()
    consumed = dict(inputs["reference_table_geometry"]["source_sha256"])
    config_ref = {"path": str(CONFIG), "sha256": CONFIG_SHA}
    config = prefix.read(config_ref, consumed)
    prefix.require(
        config["schema"] == "s2a.parallel-boxes-shape-bound-study.v1"
        and config["output"] == str(OUTPUT.relative_to(ROOT))
        and (config["case_index"], config["seed"], config["first_step"], config["periods"]) == (8, 42, 120, 8)
        and [item["mode"] for item in config["branches"]] == ["original", "pose_bounded"]
        and (config["table_geom"], config["stove_geom"], config["finger_geom"], config["finger_joint"])
        == ("table_collision", "flat_stove_1_collision_burner", "gripper0_finger2_collision", "gripper0_finger_joint2")
        and (config["position_uncertainty_m"], config["rotation_uncertainty_rad"]) == (0.01, 0.05)
        and (config["clearance_m"], config["numerical_margin_m"], config["max_translation_m"]) == (0.001, 1e-9, 0.05)
        and config["shape_modes"] == ["zero", "joint_range"]
        and (config["max_models"], config["max_forward_queries"]) == (1, 16)
        and config["physics_steps"] == config["world_model_calls"] == config["fresh_cases"] == 0
        and config["automatic_retry"] is False,
        "The exact diagnostic candidates and static budget are required",
    )
    owner_ref = {"path": str(ROOT / config["prior_owner"]), "sha256": config["prior_owner_sha256"]}
    owner = prefix.read(owner_ref, consumed)
    prefix.require(
        owner["complete"] is owner["passed"] is owner["sources_unchanged"] is True
        and owner["control_qualified"] is False,
        "A completed independent prior feedback owner is required",
    )
    branches = {}
    previous = inputs["old"]["old"]["base"]["previous"]
    xml_sha = hashlib.sha256(previous["record_xml"].encode()).hexdigest()
    for item in config["branches"]:
        mode = item["mode"]
        refs = {
            phase: {"path": str(ROOT / item[phase]), "sha256": item[phase + "_sha256"]}
            for phase in ("producer", "checker")
        }
        prefix.require(
            owner["producer_refs"][mode] == refs["producer"] and owner["checker_refs"][mode] == refs["checker"],
            "Branch identities must match the original owner",
        )
        producer, checker = (prefix.read(refs[phase], consumed) for phase in ("producer", "checker"))
        prefix.require(
            producer["complete"] is producer["passed"] is checker["complete"] is checker["passed"] is True
            and producer["mode"] == checker["mode"] == mode
            and producer["runtime"]["model_records"][-1]["xml_sha256"] == xml_sha
            and checker["runtime"]["model_records"][-1]["xml_sha256"] == xml_sha,
            "Both branches must use the same verified mapped native model",
        )
        live.check_pair(producer, checker)
        continuation = producer["runtime"]["continuation"]
        periods = continuation["probe"]["periods"]
        pre = continuation["pre_model_replay"]["probe"]["periods"]
        prefix.require(
            [row["step"] for row in periods] == list(range(120, 128))
            and pre[-1]["step"] == 119
            and state_sha256(np.asarray(pre[-1]["actual_state"], dtype=float)) == continuation["model_state_sha256"]
            and len(continuation["inner_samples"]) == 200,
            "Eight complete, branch-bound periods and their matching initial state are required",
        )
        branches[mode] = {"refs": refs, "producer": producer, "checker": checker}
    return {
        "config": config,
        "config_ref": config_ref,
        "owner_ref": owner_ref,
        "branches": branches,
        "old": inputs,
        "source_sha256": consumed,
    }


def input_receipt(inputs):
    return {
        "prior": geometry.input_receipt(inputs["old"]),
        "config_ref": inputs["config_ref"],
        "owner_ref": inputs["owner_ref"],
        "branches": {mode: branch["refs"] for mode, branch in inputs["branches"].items()},
        "source_sha256": inputs["source_sha256"],
    }


def body_path(model, body):
    path, seen = [], set()
    while body:
        if body < 0 or body >= model.nbody or body in seen:
            raise ValueError("Invalid native body ancestry")
        seen.add(body)
        path.append(body)
        body = int(model.body_parentid[body])
    return [*path, 0]


def chain_joints(model, bodies):
    return [
        joint
        for body in bodies
        for joint in range(int(model.body_jntadr[body]), int(model.body_jntadr[body] + model.body_jntnum[body]))
    ]


def single_slide_topology(model, finger_body, site_body, slide, slide_type):
    """Verify relative kinematics, rather than inferring a bound from joint names."""
    finger, site = body_path(model, finger_body), body_path(model, site_body)
    common = next(body for body in finger if body in set(site))
    finger_joints = chain_joints(model, finger[: finger.index(common)])
    site_joints = chain_joints(model, site[: site.index(common)])
    if (
        finger_joints != [slide]
        or site_joints
        or int(model.jnt_type[slide]) != slide_type
        or not bool(model.jnt_limited[slide])
        or not np.isclose(np.linalg.norm(model.jnt_axis[slide]), 1.0, atol=1e-12, rtol=0)
    ):
        raise ValueError("The TCP-relative geometry must contain exactly one limited unit slide")
    interval = np.asarray(model.jnt_range[slide], dtype=float)
    if interval.shape != (2,) or not np.isfinite(interval).all() or interval[0] > interval[1]:
        raise ValueError("The declared slide interval must be finite and ordered")
    return {
        "common_ancestor_body_id": common,
        "finger_chain_body_ids": finger[: finger.index(common)],
        "site_chain_body_ids": site[: site.index(common)],
        "joint_id": slide,
        "joint_interval_m": interval,
    }


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
                native_models=0,
                forward_queries=0,
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
                automatic_retry=False,
            ),
        ),
        flush=True,
    )


def analyze(inputs, stats):
    previous = inputs["old"]["old"]["old"]["base"]["previous"]
    cv2 = previous["cv2_admission"]
    restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
    import mujoco

    originals = {name: getattr(mujoco, name) for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward")}

    def forbidden(*_args, **_kwargs):
        stats["physics_steps_attempted"] += 1
        raise RuntimeError("Static analysis cannot integrate physics")

    def forward(*args, **kwargs):
        stats["forward_queries_attempted"] += 1
        prefix.require(stats["forward_queries_attempted"] <= 16, "Static forward budget exhausted")
        value = originals["mj_forward"](*args, **kwargs)
        stats["forward_queries_completed"] += 1
        return value

    try:
        for name in ("mj_step", "mj_step1", "mj_step2"):
            setattr(mujoco, name, forbidden)
        mujoco.mj_forward = forward
        stats["models_attempted"] += 1
        model = mujoco.MjModel.from_xml_string(previous["record_xml"])
        stats["models_completed"] += 1
        data = mujoco.MjData(model)
        config = inputs["config"]
        names = (config["table_geom"], config["stove_geom"], config["finger_geom"])
        table, stove, finger = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in names]
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, config["finger_joint"])
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, previous["controller"]["contract"]["controller_site"])
        prefix.require(
            min(table, stove, finger, joint, site) >= 0, "All original native geometries and joints must exist"
        )
        prefix.require(
            model.geom_type[table] == model.geom_type[stove] == mujoco.mjtGeom.mjGEOM_BOX
            and model.geom_type[finger] == mujoco.mjtGeom.mjGEOM_MESH,
            "Both environment bodies are boxes and finger2 is a native collision mesh",
        )
        for geom in (table, stove):
            prefix.require(
                not chain_joints(model, body_path(model, int(model.geom_bodyid[geom]))[:-1]),
                "Both boxes must be fixed to the world",
            )
            prefix.require(
                (model.geom_contype[geom] & model.geom_conaffinity[finger])
                or (model.geom_contype[finger] & model.geom_conaffinity[geom]),
                "Each studied collision pair must be enabled",
            )
        topology = single_slide_topology(
            model, int(model.geom_bodyid[finger]), int(model.site_bodyid[site]), joint, int(mujoco.mjtJoint.mjJNT_SLIDE)
        )
        mesh = int(model.geom_dataid[finger])
        prefix.require(mesh >= 0, "The actual finger mesh is required")
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        vertices_geom = np.asarray(model.mesh_vert[start : start + count], dtype=float).copy()
        prefix.require(
            count >= 4 and vertices_geom.shape == (count, 3) and np.isfinite(vertices_geom).all(),
            "Finite native collision vertices",
        )
        site_from_tcp = inverse(np.asarray(previous["controller"]["source_from_site"])) @ np.asarray(previous["tool"])
        box_anchors, rows = None, []
        for mode, branch in inputs["branches"].items():
            continuation = branch["producer"]["runtime"]["continuation"]
            periods = continuation["probe"]["periods"]
            states = [continuation["pre_model_replay"]["probe"]["periods"][-1]["actual_state"]] + [
                row["actual_state"] for row in periods[:-1]
            ]
            for index, (row, state) in enumerate(zip(periods, states, strict=True)):
                state = np.asarray(state, dtype=float)
                prefix.require(
                    state.shape == (1 + model.nq + model.nv + model.na,) and np.isfinite(state).all(),
                    "Full finite native state required",
                )
                data.time = state[0]
                data.qpos[:] = state[1 : 1 + model.nq]
                data.qvel[:] = state[1 + model.nq : 1 + model.nq + model.nv]
                if model.na:
                    data.act[:] = state[-model.na :]
                mujoco.mj_forward(model, data)

                def geom_pose(identifier):
                    return transform(data.geom_xpos[identifier].copy(), data.geom_xmat[identifier].reshape(3, 3).copy())

                site_pose = transform(data.site_xpos[site].copy(), data.site_xmat[site].reshape(3, 3).copy())
                tcp = site_pose @ site_from_tcp
                vertices = collision_vertices_in_tcp(vertices_geom, tcp, geom_pose(finger))
                decision = row["decision"]
                np.testing.assert_allclose(tcp, decision["decision_tcp"], atol=1e-9, rtol=0)
                np.testing.assert_allclose(vertices, decision["decision_vertices_tcp"], atol=1e-9, rtol=0)
                boxes = [
                    (name, geom_pose(geom), model.geom_size[geom].copy())
                    for name, geom in (("table", table), ("stove", stove))
                ]
                if box_anchors is None:
                    box_anchors = [(name, pose.copy(), size.copy()) for name, pose, size in boxes]
                for (_, pose, size), (_, anchor_pose, anchor_size) in zip(boxes, box_anchors, strict=True):
                    np.testing.assert_allclose(pose, anchor_pose, atol=1e-12, rtol=0)
                    np.testing.assert_array_equal(size, anchor_size)
                q = float(data.qpos[int(model.jnt_qposadr[joint])])
                lower, upper = (float(v) for v in topology["joint_interval_m"])
                bounds = {"zero": 0.0}
                if lower <= q <= upper:
                    bounds["joint_range"] = single_slide_shape_bound(q, lower, upper)
                else:
                    bounds["joint_range"] = None
                candidates = {}
                for shape_mode, shape_bound in bounds.items():
                    candidates[shape_mode] = (
                        None
                        if shape_bound is None
                        else project_parallel_boxes_reference(
                            decision["original_target_tcp"],
                            vertices,
                            boxes,
                            position_uncertainty=config["position_uncertainty_m"],
                            rotation_uncertainty=config["rotation_uncertainty_rad"],
                            shape_uncertainty=shape_bound,
                            clearance=config["clearance_m"],
                            numerical_margin=config["numerical_margin_m"],
                            max_translation=config["max_translation_m"],
                        )
                    )
                samples = continuation["inner_samples"][25 * index : 25 * (index + 1)]
                displacements = [
                    sample["bounded_sample_conditions"]["max_material_vertex_displacement_tcp_m"] for sample in samples
                ]
                rows.append(
                    {
                        "mode": mode,
                        "step": row["step"],
                        "state_sha256": state_sha256(state),
                        "slide_position_m": q,
                        "slide_range_assumption_available": bounds["joint_range"] is not None,
                        "max_observed_shape_change_m": max(displacements),
                        "observed_shape_bound_met_samples": None
                        if bounds["joint_range"] is None
                        else sum(value <= bounds["joint_range"] for value in displacements),
                        "candidates": candidates,
                    }
                )
        return {"topology": topology, "finger_vertices": count, "box_anchors": box_anchors, "rows": rows}
    finally:
        for name, original in originals.items():
            setattr(mujoco, name, original)


def execute():
    inputs = load_inputs()
    before = snapshot(ROOT)["sources"]
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
        and dry["source_refs"] == source_refs()
        and dry["native_models"] == dry["forward_queries"] == dry["physics_steps"] == 0,
        "Source-matched 32-worker tests and a zero-native dry are required",
    )
    OUTPUT.mkdir(exist_ok=False)
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
        control_qualified=False,
        automatic_retry=False,
        world_model_calls=0,
        fresh_cases=0,
        inputs=input_receipt(inputs),
        source_refs=source_refs(),
        qa_ref=prefix.reference(QA),
        dry_ref=prefix.reference(DRY),
        stats=stats,
    )
    try:
        result["analysis"] = analyze(inputs, stats)
        prefix.require(
            stats
            == dict(
                models_attempted=1,
                models_completed=1,
                forward_queries_attempted=16,
                forward_queries_completed=16,
                physics_steps_attempted=0,
            ),
            "Only the registered static budget may execute",
        )
        result["sources_unchanged"] = snapshot(ROOT)["sources"] == before
        prefix.require(result["sources_unchanged"], "Sources changed during the study")
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(OUTPUT / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    dry_run() if args.dry else execute()


if __name__ == "__main__":
    main()
