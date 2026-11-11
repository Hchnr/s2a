"""Compare bounded static model witnesses at the audited step125 stop."""

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

from s2a.controllers.osc import OSCInterface
from s2a.geometry import inverse, pose_error, transform
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.references.generated_twist import reference_world_twist
from s2a.research.joint_control.preview.interior_joint_preview import build_joint_preview
from s2a.research.joint_control.tracking.early_quality_diagnostic import independently_check_witness
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import registered_feedback_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.tracking.trajectory_search import cached_live_stop125_admission as admission

ROOT, OUTPUT, QA, DRY = admission.ROOT, admission.OUTPUT, admission.QA, admission.DRY
MODULE = "scripts.tracking.trajectory_search.run_cached_live_stop125_feasibility"


def source_refs():
    paths = (
        admission.CONFIG,
        admission.PLAN,
        Path(__file__).resolve(),
        ROOT / "scripts/tracking/trajectory_search/cached_live_stop125_admission.py",
        ROOT / "src/s2a/research/trajectory_search/quality_capped_inner_joint_plan.py",
        ROOT / "src/s2a/research/trajectory_search/past_inner_error_joint_plan.py",
        ROOT / "src/s2a/research/joint_control/tracking/early_quality_diagnostic.py",
        ROOT / "tests/research/native_runtime/test_cached_live_stop125_admission.py",
        ROOT / "tests/research/native_runtime/test_cached_live_stop125_feasibility.py",
    )
    return [prefix.reference(path) for path in paths]


def dry_run():
    inputs = admission.load_inputs()
    print(
        publish(
            DRY,
            {
                "complete": True,
                "passed": True,
                "inputs": admission.input_receipt(inputs),
                "source_refs": source_refs(),
                "model_initializations": 0,
                "forward_queries": 0,
                "optimizer_calls": 0,
                "physics_steps": 0,
                "world_model_calls": 0,
                "automatic_retry": False,
            },
        ),
        flush=True,
    )


def analyze_candidates(
    context, stopped, targets, anchor, stats, *, quality_solver, joint_solver, auditor, results=None
):
    """Run fixed diagnostic branches using no task-success or future state."""
    tool = context["site_from_tcp"]
    twists = np.asarray([reference_world_twist(targets, index, anchor, 0.05)["twist"] for index in range(5, 13)])
    desired = targets[5:13]
    previous = stopped["previous"][:6]
    args = (
        context["model"],
        context["velocity"],
        stopped["shifted_seed"],
        previous,
        context["interface"],
        context["site"],
        tool,
        desired,
        twists,
        context["limits"][:, 0],
        context["limits"][:, 1],
    )
    results = {} if results is None else results
    warm = None
    for label in ("quality_past", "quality_nominal", "joint_past", "quality_past_warm"):
        if label == "quality_past_warm" and warm is None:
            results[label] = {"skipped": True, "reason": "joint_past_no_verified_witness"}
            continue
        past = np.zeros((25, 7)) if label == "quality_nominal" else stopped["past_inner_error_rad"]
        solver = joint_solver if label == "joint_past" else quality_solver
        extra = {"initial_guess": warm} if label == "quality_past_warm" else {}
        stats["optimizer_calls_attempted"] += 1
        prefix.require(stats["optimizer_calls_attempted"] <= 4, "Static optimizer call budget exhausted")
        before = stats["optimizer_responses_completed"]
        started = time.monotonic()
        solved = solver(*args, past_inner_error_rad=past, **extra)
        stats["optimizer_calls_completed"] += 1
        responses = int(solved["response_evaluations"])
        prefix.require(
            0 <= responses <= 160 and stats["optimizer_responses_completed"] - before == responses,
            "Actual solver responses must match each registered 160-response limit",
        )
        results[label] = {key: value for key, value in solved.items() if key != "rollout"}
        results[label]["independent_verification"] = None
        verification = None
        if solved["witness_found"] is True:
            stats["independent_endpoint_queries_charged"] += 200
            stats["independent_transition_steps_charged"] += 2600
            prefix.require(
                stats["independent_endpoint_queries_charged"] <= 800
                and stats["independent_transition_steps_charged"] <= 10400,
                "Independent witness audit budget exhausted",
            )
            verification = auditor(
                solved,
                context["mass"],
                context["jacobian"],
                context["contract"]["kp"],
                context["contract"]["kd"],
                context["q"],
                context["velocity"],
                context["initial_joint"],
                context["interface"],
                previous,
                context["site"],
                tool,
                desired[0],
                context["limits"][:, 0],
                context["limits"][:, 1],
                past_inner_error_rad=past,
                enforce_pose_caps=label != "joint_past",
            )
            prefix.require(
                verification["passed"] is True
                and verification["independent_endpoint_queries"] == 200
                and verification["independent_transition_steps"] == 2600,
                "Every claimed witness requires the independently bounded check",
            )
            if label == "joint_past":
                warm = np.asarray(solved["planned_commands"]).copy()
        results[label].update(independent_verification=verification, elapsed_seconds=time.monotonic() - started)
    return results


def analyze(inputs, stats, partial):
    """Restore one frozen model without allowing any native integration."""
    import mujoco

    from s2a.research.trajectory_search import past_inner_error_joint_plan as joint
    from s2a.research.trajectory_search import quality_capped_inner_joint_plan as quality

    def prohibited(*_args, **_kwargs):
        stats["physics_steps_attempted"] += 1
        raise RuntimeError("Static stop-state diagnosis forbids physical integration")

    originals = {name: getattr(mujoco, name) for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward")}
    original_quality, original_joint = quality.rollout_joint_preview, joint.rollout_joint_preview

    def guarded_forward(*args, **kwargs):
        stats["forward_queries_attempted"] += 1
        prefix.require(stats["forward_queries_attempted"] <= 1, "Static forward budget exhausted")
        value = originals["mj_forward"](*args, **kwargs)
        stats["forward_queries_completed"] += 1
        return value

    def counted(function):
        def rollout(*args, **kwargs):
            stats["optimizer_responses_attempted"] += 1
            prefix.require(stats["optimizer_responses_attempted"] <= 640, "Static model response budget exhausted")
            value = function(*args, **kwargs)
            stats["optimizer_responses_completed"] += 1
            return value

        return rollout

    try:
        for name in ("mj_step", "mj_step1", "mj_step2"):
            setattr(mujoco, name, prohibited)
        mujoco.mj_forward = guarded_forward
        quality.rollout_joint_preview, joint.rollout_joint_preview = counted(original_quality), counted(original_joint)
        stats["model_initializations_attempted"] += 1
        model = mujoco.MjModel.from_xml_string(inputs["record_xml"])
        stats["model_initializations_completed"] += 1
        data = mujoco.MjData(model)
        state = inputs["stop_inputs"]["state"]
        prefix.require(len(state) == 1 + model.nq + model.nv + model.na and model.na == 0, "Native state layout")
        data.time = state[0]
        data.qpos[:] = state[1 : 1 + model.nq]
        data.qvel[:] = state[1 + model.nq :]
        mujoco.mj_forward(model, data)
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{i}") for i in range(1, 8)]
        prefix.require(min(joints) >= 0, "Seven fixed robot joints are required")
        qids = np.asarray([model.jnt_qposadr[j] for j in joints], dtype=int)
        dofs = np.asarray([model.jnt_dofadr[j] for j in joints], dtype=int)
        np.testing.assert_array_equal(qids, np.arange(7))
        np.testing.assert_allclose(data.qpos[qids], state[1 : 1 + model.nq][qids], atol=1e-12, rtol=0)
        np.testing.assert_allclose(data.qvel[dofs], state[1 + model.nq :][dofs], atol=1e-12, rtol=0)
        prefix.require(float(data.time) == float(state[0]), "Static forward must not advance time")
        restored_difference = float(np.max(np.abs(data.qpos - state[1 : 1 + model.nq])))
        contract = inputs["controller"]["contract"]
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, contract["controller_site"])
        prefix.require(site_id >= 0, "The fixed controller site is required")
        site = transform(data.site_xpos[site_id].copy(), data.site_xmat[site_id].reshape(3, 3).copy())
        mass = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, mass, data.qM)
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, site_id)
        mass, jacobian = mass[np.ix_(dofs, dofs)], np.vstack((jp, jr))[:, dofs]
        q, velocity = data.qpos[qids].copy(), data.qvel[dofs].copy()
        limits = np.asarray([model.jnt_range[j] for j in joints]).copy()
        interface = OSCInterface(*(contract[key] for key in ("input_min", "input_max", "output_min", "output_max")))
        preview = build_joint_preview(
            mass, jacobian, contract["kp"], contract["kd"], q, inputs["initial_joint"], null_kp=10.0, dt=0.002
        )
        site_from_tcp = inverse(np.asarray(inputs["controller"]["source_from_site"])) @ inputs["tool"]
        current_error = pose_error(inputs["targets"][5], site @ site_from_tcp)
        context = {
            "model": preview,
            "velocity": velocity,
            "interface": interface,
            "site": site,
            "site_from_tcp": site_from_tcp,
            "limits": limits,
            "mass": mass,
            "jacobian": jacobian,
            "q": q,
            "contract": contract,
            "initial_joint": inputs["initial_joint"],
        }
        partial.update(
            stop_step=125,
            generated_step=120,
            reference_cursor=5,
            seed_scope="shifted last completed plan; original failed baseline was not recorded",
            current_tcp_error=current_error,
            current_position_error_m=float(np.linalg.norm(current_error[:3])),
            current_rotation_error_rad=float(np.linalg.norm(current_error[3:])),
            minimum_current_joint_margin_rad=float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q))),
            max_qpos_restore_difference=restored_difference,
            candidates={},
            scope="frozen local model only; no physical dispatch or control qualification",
        )
        analyze_candidates(
            context,
            inputs["stop_inputs"],
            inputs["targets"],
            inputs["anchor"],
            stats,
            quality_solver=quality.solve_interior_plan,
            joint_solver=joint.solve_interior_plan,
            auditor=independently_check_witness,
            results=partial["candidates"],
        )
        return partial
    finally:
        quality.rollout_joint_preview, joint.rollout_joint_preview = original_quality, original_joint
        for name, function in originals.items():
            setattr(mujoco, name, function)


def new_stats():
    return dict.fromkeys(
        (
            "model_initializations_attempted",
            "model_initializations_completed",
            "forward_queries_attempted",
            "forward_queries_completed",
            "optimizer_calls_attempted",
            "optimizer_calls_completed",
            "optimizer_responses_attempted",
            "optimizer_responses_completed",
            "independent_endpoint_queries_charged",
            "independent_transition_steps_charged",
            "physics_steps_attempted",
        ),
        0,
    )


def check_admission(inputs):
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and not qa["collection"]["collection_errors"]
        and qa["configured_parallelism"] == 32,
        "Complete 32-worker CPU QA without skips or collection errors is required",
    )
    before = snapshot(ROOT)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "QA source identity changed")
    prefix.require(
        dry["complete"] is dry["passed"] is True
        and dry["inputs"] == admission.input_receipt(inputs)
        and dry["source_refs"] == source_refs()
        and dry["model_initializations"] == dry["forward_queries"] == dry["optimizer_calls"] == 0
        and dry["physics_steps"] == dry["world_model_calls"] == 0,
        "Matching zero-work dry admission is required",
    )
    return before


def worker():
    def interrupted(signum, _frame):
        raise InterruptedError(f"Worker received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    stats = new_stats()
    result = {
        "complete": False,
        "passed": False,
        "stats": stats,
        "analysis": {},
        "world_model_calls": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
    }
    try:
        prefix.environment()
        prefix.require(os.environ.get("MUJOCO_GL") == "disable", "Static worker must not initialize a graphics context")
        registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
        inputs = admission.load_inputs()
        prefix.require(
            registration["inputs"] == admission.input_receipt(inputs) and registration["source_refs"] == source_refs(),
            "Worker inputs and source identity must match the owner registration",
        )
        analyze(inputs, stats, result["analysis"])
        prefix.require(
            stats["model_initializations_attempted"] == stats["model_initializations_completed"] == 1
            and stats["forward_queries_attempted"] == stats["forward_queries_completed"] == 1
            and 3 <= stats["optimizer_calls_completed"] == stats["optimizer_calls_attempted"] <= 4
            and stats["optimizer_responses_attempted"] == stats["optimizer_responses_completed"] <= 640
            and stats["independent_endpoint_queries_charged"] <= 800
            and stats["independent_transition_steps_charged"] <= 10400
            and stats["physics_steps_attempted"] == 0,
            "Static work must remain within the registered budget",
        )
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(registration["source_refs"] == source_refs(), "Source files changed during the static worker")
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(OUTPUT / "worker/result.json", result), flush=True)


def execute():
    prefix.environment()
    prefix.require(not OUTPUT.exists(), "A new output is required; automatic retries are prohibited")
    inputs = admission.load_inputs()
    before = check_admission(inputs)
    sources = source_refs()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    directory = OUTPUT / "worker"
    directory.mkdir()
    registration = publish(
        OUTPUT / "registration.json",
        {
            "inputs": admission.input_receipt(inputs),
            "source_refs": sources,
            "qa_ref": prefix.reference(QA),
            "dry_ref": prefix.reference(DRY),
            "automatic_retry": False,
        },
    )
    result = {
        "complete": False,
        "passed": False,
        "registration_ref": registration,
        "sources_unchanged": False,
        "worker_ref": None,
        "workers_ref": None,
        "automatic_retry": False,
    }
    try:
        env = physical._cpu_environment(directory)
        env["MUJOCO_GL"] = "disable"
        spec = {
            "key": "stop125_static",
            "argv": [sys.executable, "-m", MODULE, "--worker"],
            "log": directory / "worker.log",
        }
        with physical._temporary_environment(env):
            lifecycle = run_owned_workers([spec], cwd=ROOT, timeout_seconds=inputs["config"]["worker_timeout_seconds"])
        result["workers_ref"] = publish(OUTPUT / "workers.json", lifecycle)
        registered_feedback_phase.require_closed_workers(lifecycle, ["stop125_static"])
        result["worker_ref"] = prefix.reference(directory / "result.json")
        worker_result = prefix.read(result["worker_ref"])
        prefix.require(worker_result["complete"] is worker_result["passed"] is True, "Static worker did not complete")
        result["stats"] = worker_result["stats"]
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        result["sources_unchanged"] = before == snapshot(ROOT)["sources"] and sources == source_refs()
        prefix.require(result["sources_unchanged"], "Sources changed during the static diagnosis")
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        worker_path = directory / "result.json"
        if worker_path.is_file() and result["worker_ref"] is None:
            result["worker_ref"] = prefix.reference(worker_path)
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
