"""Audit every saved development chunk against its own fixed table geometry."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from s2a.geometry import inverse, transform
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.joint_control.tracking.slide_range_box_collision_certificate import (
    slide_range_vertex_box_certificate,
)
from s2a.research.joint_control.tracking.tabletop_reference_projection import collision_vertices_in_tcp
from s2a.research.wam.tracking.preview.development_reference_geometry import migrate_xml
from s2a.research.wam.training.native.native_training import state_sha256
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.simulation.runtime.box_face_feedback_measurement import _body_path, _joints, checked_slide_topology
from scripts.tracking.response_models.early_box_face_bowl_effect import check_forward_qpos
from scripts.wam.tracking.preview import diagnose_development_reference_geometry as historical

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/response_models/multicase_slide_table_collision_20261111.json"
CONFIG_SHA = "c9152158d29afff9c582635e1e26da296ecbb095d2d39cdfe8cd06d63707c22f"
PLAN = ROOT / "plans/v0.0.1/tracking/response_models/multicase_slide_table_collision_20261111.md"
OUTPUT = ROOT / "outputs/multicase_slide_table_collision_20261111"
DRY = ROOT / "outputs/multicase_slide_table_collision_dry_20261111/result.json"
QA = ROOT / "outputs/qa/multicase_slide_table_collision_20261111/result.json"
MODULE = "scripts.tracking.response_models.multicase_slide_table_collision"


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__).resolve(),
            Path(historical.__file__).resolve(),
            ROOT / "src/s2a/research/joint_control/tracking/slide_range_box_collision_certificate.py",
            ROOT / "src/s2a/research/wam/tracking/preview/development_reference_geometry.py",
            ROOT / "scripts/simulation/runtime/box_face_feedback_measurement.py",
            ROOT / "scripts/tracking/response_models/early_box_face_bowl_effect.py",
            ROOT / "tests/research/joint_control/tracking/test_slide_range_box_collision_certificate.py",
            ROOT / "tests/research/response_models/test_multicase_slide_table_collision.py",
        )
    ]


def validate_config(config):
    prefix.require(
        config["schema"] == "s2a.multicase-slide-table-collision.v1"
        and config["output"] == str(OUTPUT.relative_to(ROOT))
        and config["case_indices"] == list(range(12))
        and config["tasks"] == [8, 9, 6]
        and config["sampling_seeds"] == [42, 43]
        and config["variants"] == ["raw", "terminal058"]
        and (config["queries_per_case"], config["targets_per_query"]) == (75, 8)
        and (config["table_geom"], config["finger_geom"], config["finger_joint"])
        == ("table_collision", "gripper0_finger2_collision", "gripper0_finger_joint2")
        and config["finger_joint_range_m"] == [-0.04, 0.0]
        and (
            config["position_tolerance_m"],
            config["rotation_tolerance_rad"],
            config["additional_shape_uncertainty_m"],
            config["numerical_margin_m"],
            config["quaternion_atol"],
        )
        == (0.01, 0.05, 0.0, 1e-9, 1e-12)
        and (config["max_models"], config["max_forward_queries"], config["max_certificate_calculations"])
        == (12, 900, 14400)
        and config["physics_steps"] == config["world_model_calls"] == config["fresh_cases"] == 0
        and config["automatic_retry"] is False,
        "The exact twelve-case source, target roster and static budgets are required",
    )


def load_inputs(*, verify_assets):
    config_ref = {"path": str(CONFIG), "sha256": CONFIG_SHA}
    config = prefix.read(config_ref)
    validate_config(config)
    refs = {
        key: {"path": str(ROOT / config[key]), "sha256": config[key + "_sha256"]}
        for key in ("review", "asset_snapshot", "geometry_summary")
    }
    review, assets, geometry = (prefix.read(refs[key]) for key in ("review", "asset_snapshot", "geometry_summary"))
    prefix.require(
        review["complete"] is review["passed"] is True
        and len(review["cases"]) == len(geometry["cases"]) == 12
        and geometry["prior_full_independent_review_ref"] == refs["review"]
        and len(assets["assets"]) == 96
        and assets["file_reference_count"] == 99,
        "The complete reviewed development matrix and fixed asset closure are required",
    )
    for index, case in enumerate(review["cases"]):
        name, task = historical.expected_case(index)
        prefix.require(
            case["case"]["case_index"] == index
            and case["case"]["case_id"] == name
            and case["mechanical"]["steps"] == 600
            and case["mechanical"]["chunks"] == 75
            and geometry["cases"][index]["case"]["case_id"] == name
            and task == config["tasks"][index // 4],
            "Fixed case identity and complete historical runs",
        )
    if verify_assets:
        paths = historical.asset_paths(assets)
    else:
        paths = {asset["declared_path"]: asset["snapshot_ref"]["path"] for asset in assets["assets"]}
        prefix.require(len(paths) == 96, "The fixed asset roster must not contain duplicate paths")
    return {
        "config": config,
        "config_ref": config_ref,
        "refs": refs,
        "review": review,
        "assets": assets,
        "paths": paths,
    }


def input_receipt(inputs):
    return {
        "config_ref": inputs["config_ref"],
        "refs": inputs["refs"],
        "asset_refs": [asset["snapshot_ref"] for asset in inputs["assets"]["assets"]],
    }


def case_inputs(index, inputs):
    name, task_id = historical.expected_case(index)
    reviewed = inputs["review"]["cases"][index]
    source_map, consumed = inputs["review"]["source_and_artifact_sha256"], {}
    case_dir = historical.BASE / name
    worker_path = historical.verified_input(case_dir / "worker_result.json", source_map, consumed)
    worker = json.loads(worker_path.read_bytes())
    prefix.require(
        worker["complete"] is worker["passed"] is True and worker["physics_steps"] == 600,
        "A complete original physical worker is required",
    )
    folder = Path(worker["physical_case_dir"])
    prefix.require(
        folder == Path(reviewed["mechanical"]["case"]) and folder.is_relative_to(case_dir),
        "The reviewed physical case path must match",
    )
    files = {
        key: historical.verified_input(folder / value, source_map, consumed)
        for key, value in (
            ("manifest", "manifest.json"),
            ("xml", "model.xml"),
            ("chunks", "chunks.json"),
            ("controller", "controller.json"),
        )
    }
    manifest = json.loads(files["manifest"].read_bytes())
    prefix.require(manifest["record"]["task_id"] == task_id, "The original task identity must match")
    files["tool"] = historical.verified_input(Path(manifest["runtime_config"]["tool_contract"]), source_map, consumed)
    files["arrays"] = historical.verified_input(Path(worker["arrays_ref"]["path"]), source_map, consumed)
    prefix.require(
        historical.safe_ref(files["arrays"]) == worker["arrays_ref"], "The saved state array identity must match"
    )
    controller, tool = (
        json.loads(files["controller"].read_bytes()),
        np.asarray(json.loads(files["tool"].read_bytes())["source_from_tcp"], dtype=float),
    )
    chunks = json.loads(files["chunks"].read_bytes())
    prefix.require(
        len(chunks) == 75 and [row["step"] for row in chunks] == list(range(0, 600, 8)),
        "All original chunk times must match",
    )
    return {
        "case": reviewed["case"],
        "files": files,
        "controller": controller,
        "tool": tool,
        "chunks": chunks,
        "source_sha256": consumed,
    }


def assess_chunk(targets, vertices, axis, q, interval, table_pose, table_size, config):
    targets = np.asarray(targets, dtype=float)
    if targets.shape != (8, 4, 4) or not np.isfinite(targets).all():
        raise ValueError("Exactly eight finite already-generated targets are required")
    options = dict(
        position_tolerance=config["position_tolerance_m"],
        rotation_tolerance=config["rotation_tolerance_rad"],
        additional_shape_uncertainty=config["additional_shape_uncertainty_m"],
        numerical_margin=config["numerical_margin_m"],
    )
    rows = []
    for cursor, target in enumerate(targets):
        args = (target, vertices, axis, q)
        fixed = slide_range_vertex_box_certificate(*args, q, q, table_pose, table_size, **options)
        whole = slide_range_vertex_box_certificate(*args, *interval, table_pose, table_size, **options)
        if interval[0] <= q <= interval[1]:
            prefix.require(
                not whole["certified"] or fixed["certified"],
                "A full-range certificate must imply its fixed-current subset",
            )
        rows.append({"cursor": cursor, "target_tcp": target, "fixed_current": fixed, "full_joint_range": whole})
    return rows


def analyze_case(index, inputs, stats):
    import mujoco

    config, case = inputs["config"], case_inputs(index, inputs)
    originals = {name: getattr(mujoco, name) for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward")}

    def forbidden(*_args, **_kwargs):
        stats["physics_steps_attempted"] += 1
        raise RuntimeError("Development reference analysis cannot integrate physics")

    def forward(*args, **kwargs):
        stats["forward_queries_attempted"] += 1
        prefix.require(stats["forward_queries_attempted"] <= 75, "The per-case forward budget is exhausted")
        value = originals["mj_forward"](*args, **kwargs)
        stats["forward_queries_completed"] += 1
        return value

    try:
        for name in ("mj_step", "mj_step1", "mj_step2"):
            setattr(mujoco, name, forbidden)
        mujoco.mj_forward = forward
        stats["models_attempted"] += 1
        xml = migrate_xml(case["files"]["xml"].read_bytes(), inputs["paths"])
        model = mujoco.MjModel.from_xml_string(xml.decode("utf-8"))
        stats["models_completed"] += 1
        data = mujoco.MjData(model)
        table, finger = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, config[key]) for key in ("table_geom", "finger_geom")
        ]
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, config["finger_joint"])
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, case["controller"]["contract"]["controller_site"])
        prefix.require(
            min(table, finger, joint, site) >= 0
            and model.geom_type[table] == mujoco.mjtGeom.mjGEOM_BOX
            and model.geom_type[finger] == mujoco.mjtGeom.mjGEOM_MESH,
            "The original table, finger mesh, joint and controller site are required",
        )
        prefix.require(
            not _joints(model, _body_path(model, int(model.geom_bodyid[table]))[:-1])
            and (
                (model.geom_contype[table] & model.geom_conaffinity[finger])
                or (model.geom_contype[finger] & model.geom_conaffinity[table])
            ),
            "A world-fixed enabled table/finger collision pair is required",
        )
        interval = checked_slide_topology(
            model, int(model.geom_bodyid[finger]), int(model.site_bodyid[site]), joint, int(mujoco.mjtJoint.mjJNT_SLIDE)
        )
        np.testing.assert_array_equal(interval, config["finger_joint_range_m"])
        mesh = int(model.geom_dataid[finger])
        prefix.require(mesh >= 0, "The original finger collision mesh is required")
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        mesh_vertices = np.asarray(model.mesh_vert[start : start + count], dtype=float).copy()
        prefix.require(
            count >= 4 and mesh_vertices.shape == (count, 3) and np.isfinite(mesh_vertices).all(),
            "Finite compiled collision vertices required",
        )
        site_from_tcp = inverse(np.asarray(case["controller"]["source_from_site"])) @ case["tool"]
        with np.load(case["files"]["arrays"], allow_pickle=False) as archive:
            states = archive["sim_states"].copy()
            prefix.require(
                len(states) == 601 and archive["endpoint_step"].tolist() == list(range(601)),
                "All 601 saved endpoint states required",
            )
            np.testing.assert_array_equal(archive["sim_time"], states[:, 0])
        rows, table_anchor, max_qpos_change = [], None, 0.0
        for index, chunk in enumerate(case["chunks"]):
            step = 8 * index
            state = np.asarray(states[step], dtype=float)
            prefix.require(
                state.shape == (1 + model.nq + model.nv + model.na,) and np.isfinite(state).all(),
                "The complete finite query-time state is required",
            )
            data.time, data.qpos[:], data.qvel[:] = (
                state[0],
                state[1 : 1 + model.nq],
                state[1 + model.nq : 1 + model.nq + model.nv],
            )
            if model.na:
                data.act[:] = state[-model.na :]
            before_q, before_v, before_time = data.qpos.copy(), data.qvel.copy(), float(data.time)
            mujoco.mj_forward(model, data)
            normalized = check_forward_qpos(
                model,
                before_q,
                data.qpos,
                free_type=int(mujoco.mjtJoint.mjJNT_FREE),
                ball_type=int(mujoco.mjtJoint.mjJNT_BALL),
            )
            max_qpos_change = max(max_qpos_change, normalized["maximum_qpos_change"])
            np.testing.assert_array_equal(data.qvel, before_v)
            prefix.require(float(data.time) == before_time, "A static query must not advance time")

            def geom_pose(identifier):
                return transform(data.geom_xpos[identifier].copy(), data.geom_xmat[identifier].reshape(3, 3).copy())

            site_pose = transform(data.site_xpos[site].copy(), data.site_xmat[site].reshape(3, 3).copy())
            tcp = site_pose @ site_from_tcp
            vertices = collision_vertices_in_tcp(mesh_vertices, tcp, geom_pose(finger))
            axis = tcp[:3, :3].T @ data.xaxis[joint]
            q = float(data.qpos[int(model.jnt_qposadr[joint])])
            table_pose, table_size = geom_pose(table), model.geom_size[table].copy()
            if table_anchor is None:
                table_anchor = (table_pose.copy(), table_size.copy())
            np.testing.assert_allclose(table_pose, table_anchor[0], atol=1e-12, rtol=0)
            np.testing.assert_array_equal(table_size, table_anchor[1])
            targets = np.asarray(chunk["effective_targets_tcp"], dtype=float)
            prefix.require(
                targets.shape == (40, 4, 4) and np.isfinite(targets).all(), "The original H40 targets are required"
            )
            certificates = assess_chunk(targets[:8], vertices, axis, q, interval, table_pose, table_size, config)
            stats["certificate_calculations"] += 16
            rows.append(
                {
                    "query": index,
                    "step": step,
                    "state_sha256": state_sha256(state),
                    "measured_joint_m": q,
                    "measured_joint_in_range": bool(interval[0] <= q <= interval[1]),
                    "vertices_tcp": vertices,
                    "slide_axis_tcp": axis,
                    "targets": certificates,
                }
            )
            if (index + 1) % 15 == 0:
                print(
                    json.dumps(
                        {
                            "case": case["case"]["case_index"],
                            "queries_completed": index + 1,
                            "models": stats["models_completed"],
                            "forwards": stats["forward_queries_completed"],
                        }
                    ),
                    flush=True,
                )
        for path, digest in case["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(
            stats
            == dict(
                models_attempted=1,
                models_completed=1,
                forward_queries_attempted=75,
                forward_queries_completed=75,
                certificate_calculations=1200,
                physics_steps_attempted=0,
            ),
            "Every case must use its exact registered static budget",
        )
        return {
            "case": case["case"],
            "input_sha256": case["source_sha256"],
            "table_pose": table_anchor[0],
            "table_half_size_m": table_anchor[1],
            "finger_joint_range_m": interval,
            "max_quaternion_qpos_change": max_qpos_change,
            "queries": rows,
        }
    finally:
        for name, original in originals.items():
            setattr(mujoco, name, original)


def worker(index):
    directory = OUTPUT / f"case_{index:02d}"
    temp, cache = directory / "tmp", directory / "cache"
    temp.mkdir()
    cache.mkdir()
    os.environ.update(TMPDIR=str(temp), XDG_CACHE_HOME=str(cache), NUMBA_CACHE_DIR=str(cache / "numba"))
    tempfile.tempdir = None
    prefix.environment()
    inputs = load_inputs(verify_assets=False)
    registration = prefix.read(prefix.reference(directory / "registration.json"))
    prefix.require(
        registration["case_index"] == index
        and registration["inputs"] == input_receipt(inputs)
        and registration["source_refs"] == source_refs(),
        "Worker identity and source registration must match",
    )
    stats = dict(
        models_attempted=0,
        models_completed=0,
        forward_queries_attempted=0,
        forward_queries_completed=0,
        certificate_calculations=0,
        physics_steps_attempted=0,
    )
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        case_index=index,
        automatic_retry=False,
        world_model_calls=0,
        fresh_cases=0,
        stats=stats,
    )
    try:
        result["analysis"] = analyze_case(index, inputs, stats)
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def case_summary(case):
    queries = case["analysis"]["queries"]
    prefix.require(
        len(queries) == 75 and all(len(row["targets"]) == 8 for row in queries),
        "Complete query and target roster required",
    )
    targets = [target for query in queries for target in query["targets"]]
    return {
        "case": case["analysis"]["case"],
        "targets": len(targets),
        "queries": len(queries),
        "fixed_current_certified": sum(row["fixed_current"]["certified"] for row in targets),
        "full_joint_range_certified": sum(row["full_joint_range"]["certified"] for row in targets),
        "queries_with_full_range_certificate": sum(
            any(row["full_joint_range"]["certified"] for row in query["targets"]) for query in queries
        ),
        "queries_current_joint_outside_range": sum(not query["measured_joint_in_range"] for query in queries),
        "full_range_by_cursor": [
            sum(query["targets"][cursor]["full_joint_range"]["certified"] for query in queries) for cursor in range(8)
        ],
    }


def dry_run():
    inputs = load_inputs(verify_assets=True)
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
                certificate_calculations=0,
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
                automatic_retry=False,
            ),
        ),
        flush=True,
    )


def execute():
    inputs = load_inputs(verify_assets=True)
    before, sources = snapshot(ROOT)["sources"], source_refs()
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
        and dry["native_models"] == dry["forward_queries"] == dry["physics_steps"] == 0,
        "Source-matched 32-worker QA and zero-model dry are required",
    )
    OUTPUT.mkdir(exist_ok=False)
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        automatic_retry=False,
        world_model_calls=0,
        fresh_cases=0,
        inputs=input_receipt(inputs),
        source_refs=sources,
        qa_ref=prefix.reference(QA),
        dry_ref=prefix.reference(DRY),
    )
    try:
        specs = []
        for index in inputs["config"]["case_indices"]:
            directory = OUTPUT / f"case_{index:02d}"
            directory.mkdir()
            publish(
                directory / "registration.json",
                dict(case_index=index, inputs=input_receipt(inputs), source_refs=sources),
            )
            specs.append(
                dict(
                    key=f"case_{index:02d}",
                    argv=[sys.executable, "-m", MODULE, "--worker", str(index)],
                    log=directory / "worker.log",
                )
            )
        env = physical._cpu_environment(OUTPUT)
        with physical._temporary_environment(env):
            lifecycle = run_owned_workers(specs, cwd=ROOT, timeout_seconds=inputs["config"]["worker_timeout_seconds"])
        result["workers_ref"] = publish(OUTPUT / "workers.json", lifecycle)
        prefix.require(
            lifecycle["launch_error"] is None
            and len(lifecycle["workers"]) == 12
            and all(
                row["actual_wait"] is row["reaped"] is True and row["returncode"] == 0 and row["signal_sent"] is False
                for row in lifecycle["workers"]
            ),
            "Every case worker must exit and be reaped",
        )
        refs = [
            prefix.reference(OUTPUT / f"case_{index:02d}/result.json") for index in inputs["config"]["case_indices"]
        ]
        cases = [prefix.read(ref) for ref in refs]
        prefix.require(
            all(
                case["complete"] is case["passed"] is True and case["case_index"] == index
                for index, case in enumerate(cases)
            ),
            "Every registered case must complete",
        )
        totals = {
            key: sum(case["stats"][key] for case in cases)
            for key in (
                "models_completed",
                "forward_queries_completed",
                "certificate_calculations",
                "physics_steps_attempted",
            )
        }
        prefix.require(
            totals
            == dict(
                models_completed=12,
                forward_queries_completed=900,
                certificate_calculations=14400,
                physics_steps_attempted=0,
            ),
            "The full fixed static budget must match",
        )
        prefix.require(
            historical.asset_paths(inputs["assets"]) == inputs["paths"]
            and source_refs() == sources
            and snapshot(ROOT)["sources"] == before,
            "Inputs, assets or sources changed during the analysis",
        )
        result.update(
            complete=True,
            passed=True,
            sources_unchanged=True,
            case_refs=refs,
            summaries=[case_summary(case) for case in cases],
            totals=totals,
        )
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        partial = []
        for index in inputs["config"]["case_indices"]:
            path = OUTPUT / f"case_{index:02d}/result.json"
            if path.is_file():
                ref = prefix.reference(path)
                branch = prefix.read(ref)
                partial.append(dict(case_index=index, ref=ref, stats=branch.get("stats")))
        result["partial_cases"] = partial
        raise
    finally:
        print(publish(OUTPUT / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--worker", type=int, choices=range(12))
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    elif args.execute:
        execute()
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
