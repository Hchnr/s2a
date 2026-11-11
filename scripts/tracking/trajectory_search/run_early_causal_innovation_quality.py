"""Compare raw and causal-innovation quality at the completed step121 state."""

import argparse
import os
import signal
from pathlib import Path

import numpy as np

from s2a.controllers.osc import OSCInterface
from s2a.geometry import inverse, pose_error, transform
from s2a.infrastructure.publication import publish_result as publish
from s2a.references.generated_twist import reference_world_twist
from s2a.research.joint_control.preview.interior_joint_preview import build_joint_preview
from s2a.research.joint_control.tracking.early_quality_diagnostic import independently_check_witness
from s2a.research.trajectory_search import innovation_capped_quality_solver as innovation_solver
from s2a.research.trajectory_search import quality_rate_repair_solver as raw_solver
from scripts.simulation.runtime import prepare_quality_rate_repair_recorded_feedback as physical
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import audit_cached_live_stop125_constraints as audit
from scripts.tracking.trajectory_search import early_causal_innovation_admission as admission
from scripts.tracking.trajectory_search import run_cached_live_stop125_feasibility as base

ROOT, OUTPUT, QA, DRY = admission.ROOT, admission.OUTPUT, admission.QA, admission.DRY
MODULE = "scripts.tracking.trajectory_search.run_early_causal_innovation_quality"


def source_refs():
    return base.source_refs() + [
        prefix.reference(path)
        for path in (
            admission.CONFIG,
            admission.PLAN,
            Path(__file__).resolve(),
            ROOT / "scripts/tracking/trajectory_search/early_causal_innovation_admission.py",
            ROOT / "src/s2a/research/trajectory_search/innovation_capped_quality_solver.py",
            ROOT / "src/s2a/research/trajectory_search/quality_rate_repair_solver.py",
            ROOT / "src/s2a/research/trajectory_search/rate_boundary_repair.py",
            ROOT / "tests/research/native_runtime/test_early_causal_innovation_admission.py",
            ROOT / "tests/research/native_runtime/test_early_causal_innovation_quality.py",
            ROOT / "tests/research/trajectory_search/test_innovation_capped_quality_solver.py",
        )
    ]


def run_branches(context, stopped, targets, anchor, stats, *, raw, innovation, auditor, results):
    """Evaluate both registered branches with only completed causal inputs."""
    cursor = 2
    previous, tool = stopped["previous"][:6], context["site_from_tcp"]
    desired = targets[cursor : cursor + 8]
    twists = np.asarray(
        [reference_world_twist(targets, index, anchor, 0.05)["twist"] for index in range(cursor, cursor + 8)]
    )
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
    for label, solve in (("raw_quality", raw), ("innovation_quality", innovation)):
        options = {"past_inner_error_rad": stopped["past_inner_error_rad"]}
        if label == "innovation_quality":
            options.update(
                translation_innovation=stopped["translation_innovation"],
                rotation_discrepancy=stopped["rotation_discrepancy"],
            )
        stats["optimizer_calls_attempted"] += 1
        prefix.require(stats["optimizer_calls_attempted"] <= 2, "Two registered optimizer calls")
        before = stats["optimizer_responses_completed"]
        solved = solve(*args, **options)
        stats["optimizer_calls_completed"] += 1
        prefix.require(
            0 <= solved["response_evaluations"] <= 160
            and solved["response_evaluations"] == stats["optimizer_responses_completed"] - before,
            "Each branch must fit its actual 160-response budget",
        )
        row = {key: value for key, value in solved.items() if key != "rollout"}
        results[label] = row
        row.update(certified_model_witness=False, independent_verification=None)
        if solved["witness_found"] is not True:
            continue
        stats["independent_endpoint_queries_charged"] += 200
        stats["independent_transition_steps_charged"] += 2600
        prefix.require(
            stats["independent_endpoint_queries_charged"] <= 400
            and stats["independent_transition_steps_charged"] <= 5200,
            "Two bounded independent witness audits",
        )
        try:
            checked = auditor(
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
                past_inner_error_rad=stopped["past_inner_error_rad"],
                enforce_pose_caps=True,
            )
            corrected = innovation_solver.corrected_error(
                checked["first_tcp_error"], stopped["translation_innovation"], stopped["rotation_discrepancy"]
            )
            prefix.require(
                checked["passed"] is checked["first_pose_caps_satisfied"] is checked["first_pose_caps_enforced"] is True
                and checked["independent_endpoint_queries"] == 200
                and checked["independent_transition_steps"] == 2600,
                "Original independent quality and safety audit required",
            )
            if label == "innovation_quality":
                np.testing.assert_allclose(
                    corrected, solved["innovation_corrected_first_tcp_error"], atol=1e-10, rtol=0
                )
                prefix.require(
                    np.linalg.norm(corrected[:3]) <= 0.01 + 1e-10 and np.linalg.norm(corrected[3:]) <= 0.05 + 1e-10,
                    "Independent innovation-corrected quality gates required",
                )
        except (AssertionError, ValueError) as error:
            row["independent_verification"] = {"passed": False, "error": str(error)}
        else:
            row.update(
                certified_model_witness=True, independent_verification=checked, independent_corrected_error=corrected
            )
    return results


def analyze(inputs, stats, partial):
    """Restore a single static model and forbid every native integration entry."""
    import mujoco

    from s2a.research.trajectory_search import quality_capped_inner_joint_plan as quality

    originals = {name: getattr(mujoco, name) for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward")}
    original_rollout = quality.rollout_joint_preview

    def prohibited(*_args, **_kwargs):
        stats["physics_steps_attempted"] += 1
        raise RuntimeError("Static early quality search forbids physical integration")

    def forward(*args, **kwargs):
        stats["forward_queries_attempted"] += 1
        prefix.require(stats["forward_queries_attempted"] <= 1, "One static forward is allowed")
        value = originals["mj_forward"](*args, **kwargs)
        stats["forward_queries_completed"] += 1
        return value

    def rollout(*args, **kwargs):
        stats["optimizer_responses_attempted"] += 1
        prefix.require(stats["optimizer_responses_attempted"] <= 320, "The shared response budget is exhausted")
        value = original_rollout(*args, **kwargs)
        stats["optimizer_responses_completed"] += 1
        return value

    try:
        for name in ("mj_step", "mj_step1", "mj_step2"):
            setattr(mujoco, name, prohibited)
        mujoco.mj_forward = forward
        quality.rollout_joint_preview = rollout
        stats["model_initializations_attempted"] += 1
        model = mujoco.MjModel.from_xml_string(inputs["record_xml"])
        stats["model_initializations_completed"] += 1
        data, state = mujoco.MjData(model), inputs["stop_inputs"]["state"]
        prefix.require(
            len(state) == 1 + model.nq + model.nv + model.na and model.na == 0, "Original native state layout"
        )
        data.time, data.qpos[:], data.qvel[:] = state[0], state[1 : 1 + model.nq], state[1 + model.nq :]
        mujoco.mj_forward(model, data)
        prefix.require(float(data.time) == float(state[0]), "Static forward must not advance time")
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{i}") for i in range(1, 8)]
        prefix.require(min(joints) >= 0, "Seven fixed robot joints are required")
        qids = np.asarray([model.jnt_qposadr[j] for j in joints], dtype=int)
        dofs = np.asarray([model.jnt_dofadr[j] for j in joints], dtype=int)
        np.testing.assert_array_equal(qids, np.arange(7))
        np.testing.assert_allclose(data.qpos[qids], state[1 : 1 + model.nq][qids], atol=1e-12, rtol=0)
        np.testing.assert_allclose(data.qvel[dofs], state[1 + model.nq :][dofs], atol=1e-12, rtol=0)
        contract = inputs["controller"]["contract"]
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, contract["controller_site"])
        prefix.require(site_id >= 0, "The fixed controller site is required")
        site = transform(data.site_xpos[site_id].copy(), data.site_xmat[site_id].reshape(3, 3).copy())
        mass, jp, jr = np.zeros((model.nv, model.nv)), np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_fullM(model, mass, data.qM)
        mujoco.mj_jacSite(model, data, jp, jr, site_id)
        mass, jacobian = mass[np.ix_(dofs, dofs)], np.vstack((jp, jr))[:, dofs]
        q, velocity = data.qpos[qids].copy(), data.qvel[dofs].copy()
        limits = np.asarray([model.jnt_range[j] for j in joints]).copy()
        interface = OSCInterface(*(contract[key] for key in ("input_min", "input_max", "output_min", "output_max")))
        model_preview = build_joint_preview(
            mass, jacobian, contract["kp"], contract["kd"], q, inputs["initial_joint"], null_kp=10.0, dt=0.002
        )
        tool = inverse(np.asarray(inputs["controller"]["source_from_site"])) @ inputs["tool"]
        error = pose_error(inputs["targets"][2], site @ tool)
        context = {
            "model": model_preview,
            "velocity": velocity,
            "interface": interface,
            "site": site,
            "site_from_tcp": tool,
            "limits": limits,
            "mass": mass,
            "jacobian": jacobian,
            "q": q,
            "contract": contract,
            "initial_joint": inputs["initial_joint"],
        }
        partial.update(
            decision_step=122,
            last_completed_step=121,
            generated_step=120,
            reference_cursor=2,
            current_tcp_error=error,
            current_position_error_m=float(np.linalg.norm(error[:3])),
            current_rotation_error_rad=float(np.linalg.norm(error[3:])),
            minimum_current_joint_margin_rad=float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q))),
            max_qpos_restore_difference=float(np.max(np.abs(data.qpos - state[1 : 1 + model.nq]))),
            candidates={},
            scope="frozen local model only; no physical dispatch or control qualification",
        )
        run_branches(
            context,
            inputs["stop_inputs"],
            inputs["targets"],
            inputs["anchor"],
            stats,
            raw=raw_solver.solve_interior_plan,
            innovation=innovation_solver.solve_interior_plan,
            auditor=independently_check_witness,
            results=partial["candidates"],
        )
        return partial
    finally:
        quality.rollout_joint_preview = original_rollout
        for name, function in originals.items():
            setattr(mujoco, name, function)


def worker():
    def interrupted(signum, _frame):
        raise InterruptedError(f"Worker received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    stats = base.new_stats()
    result = dict(complete=False, passed=False, stats=stats, analysis={}, world_model_calls=0, fresh_cases=0)
    try:
        prefix.environment()
        prefix.require(os.environ.get("MUJOCO_GL") == "disable", "No graphics context may be initialized")
        registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
        inputs = admission.load_inputs()
        prefix.require(
            registration["inputs"] == admission.input_receipt(inputs) and registration["source_refs"] == source_refs(),
            "Worker inputs and sources must match registration",
        )
        analyze(inputs, stats, result["analysis"])
        # Read the later physical result only after both searches have finished.
        historical = prefix.read(admission.PRODUCER)
        row = next(item for item in physical.periods(historical) if item["step"] == 122)
        result["post_search_recorded_comparison"] = {
            "step": 122,
            "predicted_tcp_error": row["decision"]["quality"]["first_tcp_error"],
            "actual_tcp_error": row["tcp_error"],
            "actual_pose_caps_satisfied": row["pose_caps_satisfied"],
            "used_as_search_input": False,
        }
        prefix.require(
            stats["model_initializations_attempted"] == stats["model_initializations_completed"] == 1
            and stats["forward_queries_attempted"] == stats["forward_queries_completed"] == 1
            and stats["optimizer_calls_attempted"] == stats["optimizer_calls_completed"] == 2
            and stats["optimizer_responses_attempted"] == stats["optimizer_responses_completed"] <= 320
            and stats["independent_endpoint_queries_charged"] <= 400
            and stats["independent_transition_steps_charged"] <= 5200
            and stats["physics_steps_attempted"] == 0,
            "The registered two-branch static budget must hold",
        )
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(registration["source_refs"] == source_refs(), "Sources changed during early quality search")
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(OUTPUT / "worker/result.json", result), flush=True)


def dry_run():
    inputs = admission.load_inputs()
    print(
        publish(
            DRY,
            dict(
                complete=True,
                passed=True,
                inputs=admission.input_receipt(inputs),
                source_refs=source_refs(),
                model_initializations=0,
                forward_queries=0,
                optimizer_calls=0,
                physics_steps=0,
                world_model_calls=0,
            ),
        ),
        flush=True,
    )


def execute():
    check = audit.isolated(
        base.check_admission, {"QA": QA, "DRY": DRY, "admission": admission, "source_refs": source_refs}
    )
    owner = audit.isolated(
        base.execute,
        {
            "OUTPUT": OUTPUT,
            "QA": QA,
            "DRY": DRY,
            "MODULE": MODULE,
            "admission": admission,
            "source_refs": source_refs,
            "check_admission": check,
        },
    )
    owner()


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
