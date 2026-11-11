"""Compare directional and isotropic slide bounds on saved decision states."""

import argparse
from pathlib import Path

import numpy as np

from s2a.geometry import finite, inverse, transform
from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.joint_control.tracking.directional_slide_box_projection import (
    project_directional_slide_boxes_reference,
)
from s2a.research.joint_control.tracking.tabletop_reference_projection import (
    _strict_pose,
    collision_vertices_in_tcp,
)
from s2a.research.wam.training.native.native_training import state_sha256
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.response_models import parallel_boxes_shape_bound_study as prior

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/response_models/directional_slide_boxes_20261111.json"
CONFIG_SHA = "70c9f67baf9dbeb0f1023a3356d647e1e7a33eddd30bd2e2d8be085a13fed7e7"
PLAN = ROOT / "plans/v0.0.1/tracking/response_models/directional_slide_boxes_20261111.md"
OUTPUT = ROOT / "outputs/directional_slide_boxes_20261111"
DRY = ROOT / "outputs/directional_slide_boxes_dry_20261111/result.json"
QA = ROOT / "outputs/qa/directional_slide_study_20261111/result.json"


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__).resolve(),
            Path(prior.__file__).resolve(),
            ROOT / "src/s2a/research/joint_control/tracking/directional_slide_box_projection.py",
            ROOT / "src/s2a/research/joint_control/tracking/parallel_box_reference_projection.py",
            ROOT / "src/s2a/research/joint_control/tracking/tabletop_reference_projection.py",
            ROOT / "tests/research/joint_control/tracking/test_directional_slide_box_projection.py",
            ROOT / "tests/research/response_models/test_directional_slide_boxes_study.py",
        )
    ]


def load_inputs():
    base = prior.load_inputs()
    consumed = dict(base["source_sha256"])
    config_ref = {"path": str(CONFIG), "sha256": CONFIG_SHA}
    config = prefix.read(config_ref, consumed)
    prefix.require(
        config["schema"] == "s2a.directional-slide-boxes-study.v1"
        and config["output"] == str(OUTPUT.relative_to(ROOT))
        and (config["case_index"], config["seed"], config["first_step"], config["periods"]) == (8, 42, 120, 8)
        and (config["table_geom"], config["stove_geom"], config["finger_geom"], config["finger_joint"])
        == ("table_collision", "flat_stove_1_collision_burner", "gripper0_finger2_collision", "gripper0_finger_joint2")
        and (config["position_uncertainty_m"], config["rotation_uncertainty_rad"]) == (0.01, 0.05)
        and config["additional_shape_uncertainty_m"] == 0.0
        and (config["clearance_m"], config["numerical_margin_m"], config["max_translation_m"]) == (0.001, 1e-9, 0.05)
        and (config["max_models"], config["max_forward_queries"]) == (1, 16)
        and config["physics_steps"] == config["world_model_calls"] == config["fresh_cases"] == 0
        and config["automatic_retry"] is False,
        "The exact prior states, directional assumptions and static budget are required",
    )
    previous_ref = {"path": str(ROOT / config["previous_study"]), "sha256": config["previous_study_sha256"]}
    previous = prefix.read(previous_ref, consumed)
    prefix.require(
        previous["complete"] is previous["passed"] is previous["sources_unchanged"] is True
        and previous["control_qualified"] is False
        and previous["inputs"] == prior.input_receipt(base)
        and previous["stats"]
        == dict(
            models_attempted=1,
            models_completed=1,
            forward_queries_attempted=16,
            forward_queries_completed=16,
            physics_steps_attempted=0,
        )
        and len(previous["analysis"]["rows"]) == 16,
        "The exact source-pinned isotropic study and its complete endpoints are required",
    )
    for ref in previous["source_refs"]:
        prefix.checked(ref, consumed)
    return {
        "config": config,
        "config_ref": config_ref,
        "previous_ref": previous_ref,
        "previous": previous,
        "base": base,
        "source_sha256": consumed,
    }


def input_receipt(inputs):
    return {
        "config_ref": inputs["config_ref"],
        "previous_ref": inputs["previous_ref"],
        "prior": prior.input_receipt(inputs["base"]),
        "source_sha256": inputs["source_sha256"],
    }


def slide_axis_in_tcp(tcp_pose, axis_world):
    """Convert the currently measured unit native joint axis into TCP coordinates."""
    tcp, axis = _strict_pose(tcp_pose), finite(axis_world, (3,))
    if not np.isclose(np.linalg.norm(axis), 1.0, atol=1e-12, rtol=0):
        raise ValueError("The native world slide axis must have unit length")
    return tcp[:3, :3].T @ axis


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
    previous = inputs["base"]["old"]["old"]["old"]["base"]["previous"]
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
        prefix.require(min(table, stove, finger, joint, site) >= 0, "All original geometries and joints must exist")
        prefix.require(
            model.geom_type[table] == model.geom_type[stove] == mujoco.mjtGeom.mjGEOM_BOX
            and model.geom_type[finger] == mujoco.mjtGeom.mjGEOM_MESH,
            "Both environment geometries must be boxes and finger2 must be a mesh",
        )
        for geom in (table, stove):
            prefix.require(
                not prior.chain_joints(model, prior.body_path(model, int(model.geom_bodyid[geom]))[:-1]),
                "Both boxes must remain world fixed",
            )
            prefix.require(
                (model.geom_contype[geom] & model.geom_conaffinity[finger])
                or (model.geom_contype[finger] & model.geom_conaffinity[geom]),
                "Each collision pair must be enabled",
            )
        topology = prior.single_slide_topology(
            model, int(model.geom_bodyid[finger]), int(model.site_bodyid[site]), joint, int(mujoco.mjtJoint.mjJNT_SLIDE)
        )
        prior_topology = inputs["previous"]["analysis"]["topology"]
        prefix.require(
            all(
                topology[key] == prior_topology[key]
                for key in ("common_ancestor_body_id", "finger_chain_body_ids", "site_chain_body_ids", "joint_id")
            )
            and np.array_equal(topology["joint_interval_m"], prior_topology["joint_interval_m"]),
            "The native slide topology must match the previous study",
        )
        mesh = int(model.geom_dataid[finger])
        prefix.require(mesh >= 0, "The actual finger collision mesh is required")
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        vertices_geom = np.asarray(model.mesh_vert[start : start + count], dtype=float).copy()
        prefix.require(
            count == inputs["previous"]["analysis"]["finger_vertices"]
            and count >= 4
            and np.isfinite(vertices_geom).all(),
            "Identical finite native collision vertices",
        )
        site_from_tcp = inverse(np.asarray(previous["controller"]["source_from_site"])) @ np.asarray(previous["tool"])
        baseline = {(row["mode"], row["step"]): row for row in inputs["previous"]["analysis"]["rows"]}
        prefix.require(len(baseline) == 16, "All prior endpoints must be unique")
        rows = []
        for mode, branch in inputs["base"]["branches"].items():
            continuation = branch["producer"]["runtime"]["continuation"]
            periods = continuation["probe"]["periods"]
            states = [continuation["pre_model_replay"]["probe"]["periods"][-1]["actual_state"]] + [
                row["actual_state"] for row in periods[:-1]
            ]
            for row, state in zip(periods, states, strict=True):
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
                for (name, pose, size), (old_name, old_pose, old_size) in zip(
                    boxes, inputs["previous"]["analysis"]["box_anchors"], strict=True
                ):
                    prefix.require(name == old_name, "Fixed box identity")
                    np.testing.assert_allclose(pose, old_pose, atol=1e-12, rtol=0)
                    np.testing.assert_array_equal(size, old_size)
                old = baseline[(mode, row["step"])]
                q = float(data.qpos[int(model.jnt_qposadr[joint])])
                prefix.require(
                    state_sha256(state) == old["state_sha256"] and q == old["slide_position_m"],
                    "Identical recorded state and slide coordinate required",
                )
                axis_world = data.xaxis[joint].copy()
                axis_tcp = slide_axis_in_tcp(tcp, axis_world)
                lower, upper = (float(v) for v in topology["joint_interval_m"])
                spherical = old["candidates"]["joint_range"]
                prefix.require((spherical is not None) == (lower <= q <= upper), "Same joint range admission")
                directional = None
                if spherical is not None:
                    directional = project_directional_slide_boxes_reference(
                        decision["original_target_tcp"],
                        vertices,
                        axis_tcp,
                        q,
                        lower,
                        upper,
                        boxes,
                        position_uncertainty=config["position_uncertainty_m"],
                        rotation_uncertainty=config["rotation_uncertainty_rad"],
                        additional_shape_uncertainty=config["additional_shape_uncertainty_m"],
                        clearance=config["clearance_m"],
                        numerical_margin=config["numerical_margin_m"],
                        max_translation=config["max_translation_m"],
                    )
                    for name in ("table", "stove"):
                        prefix.require(
                            directional["boxes"][name]["required_translation_m"]
                            <= spherical["boxes"][name]["required_translation_m"] + 1e-10,
                            "The directional bound must not exceed the prior spherical bound",
                        )
                rows.append(
                    {
                        "mode": mode,
                        "step": row["step"],
                        "state_sha256": old["state_sha256"],
                        "slide_position_m": q,
                        "slide_axis_world": axis_world,
                        "slide_axis_tcp": axis_tcp,
                        "spherical": spherical,
                        "directional": directional,
                    }
                )
        return {"topology": topology, "box_anchors": inputs["previous"]["analysis"]["box_anchors"], "rows": rows}
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
