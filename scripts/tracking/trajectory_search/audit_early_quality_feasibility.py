"""Audit quality-constrained plans at all saved early-feedback states."""

import argparse
import ctypes
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

from s2a.controllers.osc import OSCInterface
from s2a.geometry import inverse, transform
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.references.generated_twist import reference_world_twist
from s2a.research.joint_control.limits.joint_limit_preview_objective import preview_objective
from s2a.research.joint_control.preview.interior_joint_preview import build_joint_preview, rollout_joint_preview
from s2a.research.joint_control.tracking.early_quality_diagnostic import (
    completed_error_profile,
    independently_check_witness,
    state_before_period,
)
from s2a.research.trajectory_search.quality_capped_inner_joint_plan import solve_interior_plan
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import compare_early_warning_tracking as early
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/early_quality_feasibility_20261110.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/early_quality_feasibility_20261110.md"
CORE = REPO / "src/s2a/research/joint_control/tracking/early_quality_diagnostic.py"
CORE_TEST = REPO / "tests/research/joint_control/tracking/test_early_quality_diagnostic.py"
INPUT_TEST = REPO / "tests/research/native_runtime/test_early_quality_feasibility_inputs.py"
QA = REPO / "outputs/qa/early_quality_feasibility_v3_20261110/result.json"
DRY = REPO / "outputs/early_quality_feasibility_dry_v2_20261110/result.json"
OUTPUT = REPO / "outputs/early_quality_feasibility_v2_20261110"
OSMESA = {
    "path": str(REPO / "outputs/native_runtime_path_snapshot_20261029/systemlibs/libOSMesa.so.8.0.0"),
    "sha256": "40e28f0f15c4cd13ad3c353509eb2fad79d8041528c59dc56d6e68fae8d1677a",
}
PRODUCER = {
    "path": str(REPO / "outputs/early_refiner_feedback_20261110/producer/result.json"),
    "sha256": "2e900f506f7bda364348822e2891d3c1be4191bb71703590cc8704a4300c669b",
}
CHECKER = {
    "path": str(REPO / "outputs/early_refiner_feedback_20261110/checker/result.json"),
    "sha256": "7062266a8a6eff29ff43fb339ddb578769150a4d35e279e6c2e5f496022f1404",
}
BRANCHES = tuple((index, offset) for index in (2, 6) for offset in range(8))


def source_refs():
    paths = (
        Path(__file__),
        CONFIG,
        PLAN,
        CORE,
        CORE_TEST,
        INPUT_TEST,
        Path(early.__file__),
        REPO / "src/s2a/research/trajectory_search/quality_capped_inner_joint_plan.py",
        REPO / "tests/research/trajectory_search/test_quality_capped_inner_joint_plan.py",
        REPO / "src/s2a/research/response_models/causal/causal_inner_error.py",
        REPO / "src/s2a/research/joint_control/limits/joint_preview_constraint_audit.py",
        REPO / "src/s2a/infrastructure/owned_processes.py",
    )
    return [prefix.reference(path) for path in paths]


def branch_key(index, offset):
    prefix.require((index, offset) in BRANCHES, "Fixed saved feedback state")
    return f"case_{index:02d}_offset_{offset:02d}"


def load_inputs():
    config_ref = prefix.reference(CONFIG)
    c = prefix.read(config_ref)
    prefix.require(
        c["schema"] == "s2a.early-quality-feasibility.v1"
        and c["cases"] == [2, 6]
        and c["sampling_seed"] == 43
        and c["offsets"] == list(range(8))
        and c["control_dt_s"] == 0.05
        and c["inner_dt_s"] == 0.002
        and c["joint_buffer_rad"] == 0.05
        and c["braking_time_s"] == 0.1
        and c["position_cap_m"] == 0.01
        and c["rotation_cap_rad"] == 0.05
        and c["arm_rate_limit"] == 0.25
        and c["max_models"] == 16
        and c["max_forward_queries"] == 72
        and c["max_optimizer_calls"] == 30
        and c["max_optimizer_responses"] == 4800
        and c["max_independent_endpoint_queries"] == 6000
        and c["max_independent_transition_steps"] == 78000
        and c["worker_timeout_seconds"] == 1800
        and c["physics_steps"] == c["world_model_calls"] == c["fresh_cases"] == 0
        and c["automatic_retry"] is False,
        "Fixed scope, constraints and budgets",
    )
    producer, checker = prefix.read(PRODUCER), prefix.read(CHECKER)
    prefix.require(
        producer["complete"] is producer["passed"] is producer["sources_unchanged"] is True
        and checker["complete"] is checker["passed"] is checker["sources_unchanged"] is True,
        "Closed feedback and independent replay",
    )
    inputs = early.load_inputs()
    refs = inputs["source_sha256"]
    refs[str(CONFIG)], refs[PRODUCER["path"]], refs[CHECKER["path"]] = (
        config_ref["sha256"],
        PRODUCER["sha256"],
        CHECKER["sha256"],
    )
    prefix.checked(OSMESA)
    refs[OSMESA["path"]] = OSMESA["sha256"]
    branches = {}
    for owner in (producer, checker):
        for ref in owner["branch_refs"]:
            data = prefix.read(ref)
            refs[ref["path"]] = ref["sha256"]
            key = (data["phase"], data["case_index"], data["branch"])
            prefix.require(key not in branches and data["complete"] is data["passed"] is True, "Unique closed branch")
            branches[key] = data
    for case in inputs["cases"]:
        index = case["spec"]["case_index"]
        prefix.require(case["config"]["osmesa_library"] == OSMESA["path"], "Registered software GL library")
        produced, checked = branches[("producer", index, "refiner")], branches[("checker", index, "refiner")]
        periods = produced["runtime"]["continuation"]["periods"]
        audited = checked["runtime"]["continuation"]["periods"]
        prefix.require(
            len(periods) == len(audited) == 8
            and checked["candidate_policy_imported"] is False
            and checked["preview_module_imported"] is False,
            "Eight independently replayed actions",
        )
        for left, right in zip(periods, audited, strict=True):
            np.testing.assert_array_equal(left["command"], right["command"])
            np.testing.assert_array_equal(left["actual_state"], right["actual_state"])
        case["feedback_periods"] = periods
        case["initial_joint"] = np.asarray(produced["runtime"]["runtime_initial_joint"], dtype=float)
    inputs["quality_config"] = c
    return inputs


def preload_software_gl(case, source_sha256):
    """Load the verified software library before importing MuJoCo; no context."""
    ref = prefix.reference(case["config"]["osmesa_library"])
    prefix.require(ref == OSMESA and source_sha256.get(ref["path"]) == ref["sha256"], "Bound software GL identity")
    os.environ.update(MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa")
    ctypes.CDLL(ref["path"], mode=ctypes.RTLD_GLOBAL)
    return ref


def restored_context(model, data, case, periods, index, stats):
    import mujoco

    start = case["spec"]["prefix_steps"]
    state, previous, plan = state_before_period(
        case["saved"]["sim_states"][-1], case["commands"][-1], periods, start, index
    )
    prefix.require(state.shape == (1 + model.nq + model.nv + model.na,), "Original MuJoCo state layout")
    data.time = state[0]
    data.qpos[:] = state[1 : 1 + model.nq]
    data.qvel[:] = state[1 + model.nq : 1 + model.nq + model.nv]
    if model.na:
        data.act[:] = state[-model.na :]
    stats["forward_queries_attempted"] += 1
    mujoco.mj_forward(model, data)
    stats["forward_queries_completed"] += 1
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{i}") for i in range(1, 8)]
    prefix.require(min(joints) >= 0, "Seven robot joints")
    qids = np.asarray([model.jnt_qposadr[j] for j in joints], dtype=int)
    dofs = np.asarray([model.jnt_dofadr[j] for j in joints], dtype=int)
    np.testing.assert_array_equal(qids, np.arange(7))
    np.testing.assert_allclose(data.qpos[qids], state[1 : 1 + model.nq][qids], atol=1e-12, rtol=0)
    np.testing.assert_allclose(data.qvel[dofs], state[1 + model.nq : 1 + model.nq + model.nv][dofs], atol=1e-12, rtol=0)
    prefix.require(float(data.time) == float(state[0]), "Forward must not advance time")
    stats["max_qpos_restore_difference"] = max(
        stats["max_qpos_restore_difference"], float(np.max(np.abs(data.qpos - state[1 : 1 + model.nq])))
    )
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, case["controller"]["contract"]["controller_site"])
    prefix.require(sid >= 0, "Fixed controller site")
    site = transform(data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy())
    if index == 0:
        np.testing.assert_allclose(site, case["saved"]["controller_site_pose"][-1], atol=1e-8, rtol=0)
    mass = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, mass, data.qM)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jp, jr, sid)
    mass, jacobian = mass[np.ix_(dofs, dofs)], np.vstack((jp, jr))[:, dofs]
    q, velocity = data.qpos[qids].copy(), data.qvel[dofs].copy()
    limits = np.asarray([model.jnt_range[j] for j in joints]).copy()
    contract = case["controller"]["contract"]
    interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
    preview = build_joint_preview(
        mass, jacobian, contract["kp"], contract["kd"], q, case["initial_joint"], null_kp=10.0, dt=0.002
    )
    deltas = np.asarray([interface.decode(command) for command in plan])
    rollout = rollout_joint_preview(preview, velocity, deltas, site, limits[:, 0], limits[:, 1], enforce_limits=False)
    stats["baseline_rollouts"] += 1
    return {
        "state": state,
        "previous": previous,
        "plan": plan,
        "site": site,
        "mass": mass,
        "jacobian": jacobian,
        "q": q,
        "velocity": velocity,
        "limits": limits,
        "contract": contract,
        "interface": interface,
        "model": preview,
        "rollout": rollout,
    }


def analyze_state(case, offset, stats):
    import mujoco

    def prohibited(*_args, **_kwargs):
        raise RuntimeError("Static quality audit forbids physical integration")

    for name in ("mj_step", "mj_step1", "mj_step2"):
        setattr(mujoco, name, prohibited)
    stats["model_initializations_attempted"] += 1
    model = mujoco.MjModel.from_xml_string(case["record_xml"])
    stats["model_initializations_completed"] += 1
    data = mujoco.MjData(model)
    periods = case["feedback_periods"]
    cache = {}

    def context(k):
        if k not in cache:
            cache[k] = restored_context(model, data, case, periods, k, stats)
        return cache[k]

    start = case["spec"]["prefix_steps"]
    profile = completed_error_profile(
        periods, start, offset, lambda k: context(k)["rollout"]["inner_joint_position"][:25]
    )
    current = context(offset)
    site_from_tcp = inverse(np.asarray(case["controller"]["source_from_site"])) @ case["tool"]
    cursor = case["intervention_spec"]["cursor"] + offset
    targets = case["targets"][cursor : cursor + 8]
    twists = np.asarray(
        [reference_world_twist(case["targets"], i, case["anchor"], 0.05)["twist"] for i in range(cursor, cursor + 8)]
    )
    score = preview_objective(current["rollout"], site_from_tcp, targets, twists)
    predicted_error = score["residual"][:6] / np.array([100.0] * 3 + [20.0] * 3)
    args = (
        current["model"],
        current["velocity"],
        current["plan"],
        current["previous"][:6],
        current["interface"],
        current["site"],
        site_from_tcp,
        targets,
        twists,
        current["limits"][:, 0],
        current["limits"][:, 1],
    )
    output = {}
    for label, error in (("nominal", np.zeros((25, 7))), ("past_error", profile["inner_position_error_rad"])):
        if label == "past_error" and offset == 0:
            output[label] = {
                "same_as": "nominal",
                "witness_found": output["nominal"]["witness_found"],
                "status": output["nominal"]["status"],
                "optimizer_calls": 0,
            }
            continue
        stats["optimizer_calls_attempted"] += 1
        started = time.monotonic()
        solved = solve_interior_plan(*args, past_inner_error_rad=error)
        stats["optimizer_calls_completed"] += 1
        stats["optimizer_responses"] += int(solved["response_evaluations"])
        prefix.require(solved["response_evaluations"] <= 160, "Original per-search response budget")
        verification = None
        if solved["witness_found"]:
            verification = independently_check_witness(
                solved,
                current["mass"],
                current["jacobian"],
                current["contract"]["kp"],
                current["contract"]["kd"],
                current["q"],
                current["velocity"],
                case["initial_joint"],
                current["interface"],
                current["previous"][:6],
                current["site"],
                site_from_tcp,
                targets[0],
                current["limits"][:, 0],
                current["limits"][:, 1],
                past_inner_error_rad=error,
            )
            stats["independent_endpoint_queries"] += verification["independent_endpoint_queries"]
            stats["independent_transition_steps"] += verification["independent_transition_steps"]
        result = {k: v for k, v in solved.items() if k != "rollout"}
        result.update(
            optimizer_calls=1, independent_verification=verification, elapsed_seconds=time.monotonic() - started
        )
        output[label] = result
    # The just-completed physical outcome is first read here, after both candidate searches.
    actual = periods[offset]
    true_inner = np.asarray([row["q"] for row in actual["actual_inner"]])
    predicted_inner = current["rollout"]["inner_joint_position"][:25]
    return {
        "case_index": case["spec"]["case_index"],
        "offset": offset,
        "step": start + offset,
        "past_completed_periods": profile["completed_periods"],
        "past_error_available": profile["available"],
        "past_inner_error_rad": profile["inner_position_error_rad"],
        "baseline_model_first_tcp_error": predicted_error,
        "baseline_actual_first_tcp_error": actual["tcp_error"],
        "baseline_model_vs_actual_inner_max_abs_by_joint": np.max(np.abs(predicted_inner - true_inner), axis=0),
        "baseline_actual_inner_min_margin_rad": actual["minimum_inner_margin_rad"],
        "baseline_actual_pose_caps_satisfied": actual["pose_caps_satisfied"],
        "candidates": output,
        "actual_outcome_read_after_candidate_search": True,
    }


def worker(index, offset):
    def interrupted(signum, _frame):
        raise InterruptedError(f"Worker received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
    inputs = load_inputs()
    prefix.require(
        inputs["source_sha256"] == registration["source_sha256"]
        and inputs["asset_refs"] == registration["asset_refs"]
        and source_refs() == registration["source_refs"],
        "Frozen source and input identities",
    )
    case = next(item for item in inputs["cases"] if item["spec"]["case_index"] == index)
    directory = OUTPUT / branch_key(index, offset)
    stats = {
        "model_initializations_attempted": 0,
        "model_initializations_completed": 0,
        "forward_queries_attempted": 0,
        "forward_queries_completed": 0,
        "optimizer_calls_attempted": 0,
        "optimizer_calls_completed": 0,
        "optimizer_responses": 0,
        "baseline_rollouts": 0,
        "independent_endpoint_queries": 0,
        "independent_transition_steps": 0,
        "max_qpos_restore_difference": 0.0,
        "physics_steps": 0,
    }
    result = {
        "complete": False,
        "passed": False,
        "case_index": index,
        "offset": offset,
        "stats": stats,
        "world_model_calls": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
    }
    try:
        result["osmesa_ref"] = preload_software_gl(case, inputs["source_sha256"])
        result["analysis"] = analyze_state(case, offset, stats)
        prefix.require(
            stats["model_initializations_completed"] == 1
            and stats["forward_queries_completed"] == offset + 1
            and stats["optimizer_calls_completed"] == (1 if offset == 0 else 2)
            and stats["optimizer_responses"] <= 160 * stats["optimizer_calls_completed"],
            "Worker budgets",
        )
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def execute():
    prefix.require(not OUTPUT.exists(), "New output only; no automatic retry")
    dry_ref, qa_ref = prefix.reference(DRY), prefix.reference(QA)
    dry, qa = prefix.read(dry_ref), prefix.read(qa_ref)
    prefix.require(
        dry["complete"] is dry["passed"] is True and dry["models"] == 0 and dry["physics_steps"] == 0,
        "Zero-model dry closed",
    )
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "CPU QA closed",
    )
    before = snapshot(REPO)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "Full QA source identity")
    sources, inputs = source_refs(), load_inputs()
    prefix.require(
        dry["source_refs"] == sources
        and dry["source_sha256"] == inputs["source_sha256"]
        and dry["asset_refs"] == inputs["asset_refs"],
        "Dry source/input identity",
    )
    registration = publish(
        OUTPUT / "registration.json",
        {
            "dry_ref": dry_ref,
            "qa_ref": qa_ref,
            "producer_ref": PRODUCER,
            "checker_ref": CHECKER,
            "source_refs": sources,
            "source_sha256": inputs["source_sha256"],
            "asset_refs": inputs["asset_refs"],
            "branches": [list(b) for b in BRANCHES],
            "max_models": 16,
            "max_forward_queries": 72,
            "max_optimizer_calls": 30,
            "max_optimizer_responses": 4800,
            "max_independent_endpoint_queries": 6000,
            "max_independent_transition_steps": 78000,
            "physics_steps": 0,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
    )
    specs = [
        {
            "key": branch_key(index, offset),
            "argv": [
                sys.executable,
                "-m",
                "scripts.tracking.trajectory_search.audit_early_quality_feasibility",
                "--worker",
                str(index),
                str(offset),
            ],
            "log": OUTPUT / branch_key(index, offset) / "worker.log",
        }
        for index, offset in BRANCHES
    ]
    lifecycle = run_owned_workers(specs, cwd=REPO, timeout_seconds=1800)
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    workers_ref = publish(OUTPUT / "workers.json", lifecycle)
    result = {
        "complete": False,
        "passed": False,
        "registration_ref": registration,
        "workers_ref": workers_ref,
        "physics_steps": 0,
        "world_model_calls": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
    }
    try:
        prefix.require(
            not lifecycle["launch_error"]
            and len(lifecycle["workers"]) == 16
            and all(w["returncode"] == 0 and not w["signal_sent"] for w in lifecycle["workers"]),
            "All sixteen workers closed",
        )
        refs = [prefix.reference(OUTPUT / branch_key(i, k) / "result.json") for i, k in BRANCHES]
        rows = [prefix.read(ref) for ref in refs]
        prefix.require(all(row["complete"] is row["passed"] is True for row in rows), "Complete state audits")
        prefix.require(
            all(prefix.reference(path)["sha256"] == digest for path, digest in inputs["source_sha256"].items())
            and source_refs() == sources
            and snapshot(REPO)["sources"] == before,
            "Sources and inputs unchanged",
        )
        totals = {
            name: sum(row["stats"][name] for row in rows)
            for name in (
                "model_initializations_completed",
                "forward_queries_completed",
                "optimizer_calls_completed",
                "optimizer_responses",
                "baseline_rollouts",
                "independent_endpoint_queries",
                "independent_transition_steps",
            )
        }
        prefix.require(
            totals["model_initializations_completed"] == 16
            and totals["forward_queries_completed"] == 72
            and totals["optimizer_calls_completed"] == 30
            and totals["optimizer_responses"] <= 4800
            and totals["baseline_rollouts"] == 72
            and totals["independent_endpoint_queries"] <= 6000
            and totals["independent_transition_steps"] <= 78000,
            "Fixed aggregate budgets",
        )
        result.update(complete=True, passed=True, sources_unchanged=True, case_refs=refs, **totals)
    finally:
        print(publish(OUTPUT / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--worker", nargs=2, type=int, metavar=("CASE", "OFFSET"))
    args = parser.parse_args()
    prefix.environment()
    if args.dry_run:
        prefix.require(not DRY.exists(), "New dry output only")
        inputs = load_inputs()
        print(
            publish(
                DRY,
                {
                    "complete": True,
                    "passed": True,
                    "source_refs": source_refs(),
                    "source_sha256": inputs["source_sha256"],
                    "asset_refs": inputs["asset_refs"],
                    "branches": [list(x) for x in BRANCHES],
                    "models": 0,
                    "physics_steps": 0,
                    "world_model_calls": 0,
                    "fresh_cases": 0,
                },
            ),
            flush=True,
        )
    elif args.execute:
        execute()
    else:
        index, offset = args.worker
        branch_key(index, offset)
        worker(index, offset)


if __name__ == "__main__":
    main()
