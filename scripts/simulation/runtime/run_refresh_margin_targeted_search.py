"""Run a registered static refresh-margin search at one replayed development state."""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

from s2a.controllers.osc import OSCInterface
from s2a.geometry import inverse, transform
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result, result_json
from s2a.references.generated_twist import reference_world_twist
from s2a.research.joint_control.preview.interior_joint_preview import build_joint_preview
from s2a.research.joint_control.tracking.full_horizon_quality_audit import audit_full_horizon_witness
from s2a.research.trajectory_search.refresh_margin_targeted_search import (
    MAX_RESPONSES,
    candidate_is_eligible,
    solve_targeted_refresh,
)
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import bounded_reference_choice as references
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_refresh_margin_physical as physical

ROOT = prefix.REPO
PLAN = ROOT / "plans/v0.0.1/tracking/references/refresh_margin_targeted_search_20261116.md"
OUTPUT = ROOT / "outputs/refresh_margin_targeted_search_20261116"
DRY = ROOT / "outputs/refresh_margin_targeted_search_dry_v2_20261116.json"
QA = ROOT / "outputs/qa/refresh_margin_targeted_final_v2_20261116/result.json"
MODULE = "scripts.simulation.runtime.run_refresh_margin_targeted_search"
PRODUCER = {
    "path": str(ROOT / "outputs/refresh_margin_physical_20261115/producer/result.json"),
    "sha256": "1efc6e7e845226d3517350a6e7676ab9c23d438297d3e8181a84e7b6fffc274e",
}
CHECKER = {
    "path": str(ROOT / "outputs/refresh_margin_physical_20261115/checker/result.json"),
    "sha256": "c6543b5314e6aefd9cd8bf9eada111c86c6a0f8436a3ac73b2b5b8be42df0b68",
}


def source_refs():
    return physical.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__).resolve(),
            PLAN,
            ROOT / "src/s2a/research/trajectory_search/refresh_margin_targeted_search.py",
            ROOT / "tests/research/trajectory_search/test_refresh_margin_targeted_search.py",
        )
    ]


def load_sample():
    """Admit only the completed state and the decision available before dispatch."""
    data = physical.prepare()
    groups = {name: prefix.read(ref) for name, ref in (("producer", PRODUCER), ("checker", CHECKER))}
    for group in groups.values():
        prefix.require(
            group["complete"] is group["passed"] is group["sources_unchanged"] is True
            and all(
                worker["returncode"] == 0 and worker["actual_wait"] is worker["reaped"] is True
                for worker in group["lifecycle"]["workers"]
            ),
            "Both registered physical phases must have closed and been reaped",
        )
    prefix.require(
        all(value == 0 for value in groups["checker"]["counts"]["replay_max_difference"].values()),
        "The independent replay must match the completed physical producer",
    )
    registration = prefix.read(groups["producer"]["registration_ref"])
    prefix.require(registration["identity"] == data["identity"], "The original physical inputs must remain identical")
    producer = prefix.read(groups["producer"]["branch_ref"])
    runtime = producer["runtime"]
    continuation = runtime["continuation"]
    prior = continuation["second_window"]["periods"][-1]
    tail = continuation["tail"]["segments"][0]["continuation"]["periods"]
    prefix.require(prior["step"] == 15 and len(tail) == 1 and tail[0]["step"] == 16, "Exact decision boundary")
    decision = tail[0]["decision"]
    state = np.asarray(prior["actual_state"], dtype=float)
    digest = hashlib.sha256(np.asarray(state, dtype=np.float64).tobytes()).hexdigest()
    capture = data["captures"][1]
    prefix.require(
        digest == prior["endpoint_receipt"]["sim_state_sha256"] == capture["current_endpoint"]["sim_state_sha256"]
        and capture["frame_index"] == 16,
        "The saved current state must match the completed endpoint and reference image",
    )
    quality = decision["quality"]
    verified = quality["verification"]
    prefix.require(
        verified["passed"] is verified["prefix_pose_caps_satisfied"] is True
        and verified["hard_pose_prefix"] == 2
        and quality["past_error"]["future_guarantee"] is False,
        "The pre-dispatch plan must retain its independent quality and causal-history audit",
    )
    case = data["inputs"]["case"]
    poses, _openings = references.saved_reference(capture, data["exchanges"][1], data["responses"][1], case["tool"], 16)
    anchor = np.asarray(capture["current_source_pose"]) @ np.asarray(case["tool"])
    sample = dict(
        state=state,
        state_sha256=digest,
        previous=np.asarray(prior["command"][:6], dtype=float),
        seed=np.asarray(decision["refined_plan"], dtype=float),
        initial_joint=np.asarray(runtime["runtime_initial_joint"], dtype=float),
        past_inner_error_rad=np.asarray(quality["past_error"]["inner_position_error_rad"], dtype=float),
        expected_tcp_errors=np.asarray(verified["tcp_errors"], dtype=float),
        expected_future_slack=verified["future_quality_slack"],
        anchor=anchor,
        targets=np.asarray(poses[:8], dtype=float),
        twists=np.asarray([reference_world_twist(poses, index, anchor, 0.05)["twist"] for index in range(8)]),
        controller=case["controller"],
        tool=case["tool"],
        record_xml=case["record_xml"],
    )
    np.testing.assert_array_equal(sample["seed"][0], np.asarray(tail[0]["command"][:6]))
    return sample, dict(
        producer_ref=PRODUCER,
        checker_ref=CHECKER,
        producer_branch_ref=groups["producer"]["branch_ref"],
        checker_branch_ref=groups["checker"]["branch_ref"],
        original_identity=data["identity"],
        sample_sha256=hashlib.sha256(result_json(sample).encode()).hexdigest(),
        state_sha256=digest,
        generated_step=16,
        decision_step=16,
        last_completed_step=15,
        future_actual_state_as_input=False,
    )


def dry():
    prefix.environment()
    _sample, identity = load_sample()
    print(
        publish_result(
            DRY, dict(complete=True, passed=True, identity=identity, source_refs=source_refs(), physics_steps=0)
        ),
        flush=True,
    )


def analyze(sample, counters):
    """Restore a single state while prohibiting native integration."""
    import mujoco

    names = ("mj_step", "mj_step1", "mj_step2", "mj_forward")
    originals = {name: getattr(mujoco, name) for name in names}

    def forbidden(*_args, **_kwargs):
        counters["physics_steps_attempted"] += 1
        raise RuntimeError("This static search cannot advance physics")

    def forward(*args, **kwargs):
        counters["forward_attempted"] += 1
        prefix.require(counters["forward_attempted"] <= 1, "Only one native forward is allowed")
        result = originals["mj_forward"](*args, **kwargs)
        counters["forward_completed"] += 1
        return result

    def response(event):
        prefix.require(event in ("attempted", "completed"), "Only explicit model-response events are allowed")
        counters[f"responses_{event}"] += 1
        prefix.require(counters["responses_attempted"] <= MAX_RESPONSES, "The extra response budget is fixed")
        prefix.require(counters["responses_completed"] <= counters["responses_attempted"], "Response count order")

    try:
        for name in names[:3]:
            setattr(mujoco, name, forbidden)
        mujoco.mj_forward = forward
        counters["native_models_attempted"] += 1
        model = mujoco.MjModel.from_xml_string(sample["record_xml"])
        counters["native_models_completed"] += 1
        data, state = mujoco.MjData(model), sample["state"]
        prefix.require(len(state) == 1 + model.nq + model.nv + model.na and model.na == 0, "Native state layout")
        data.time, data.qpos[:], data.qvel[:] = state[0], state[1 : 1 + model.nq], state[1 + model.nq :]
        mujoco.mj_forward(model, data)
        prefix.require(float(data.time) == float(state[0]), "The native forward must not advance the clock")
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{i}") for i in range(1, 8)]
        prefix.require(min(joints) >= 0, "Seven registered robot joints are required")
        qids = np.asarray([model.jnt_qposadr[joint] for joint in joints], dtype=int)
        dofs = np.asarray([model.jnt_dofadr[joint] for joint in joints], dtype=int)
        np.testing.assert_array_equal(qids, np.arange(7))
        contract = sample["controller"]["contract"]
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, contract["controller_site"])
        prefix.require(site_id >= 0, "The registered controller site must exist")
        site = transform(data.site_xpos[site_id].copy(), data.site_xmat[site_id].reshape(3, 3).copy())
        tool = inverse(np.asarray(sample["controller"]["source_from_site"])) @ np.asarray(sample["tool"])
        np.testing.assert_allclose(site @ tool, sample["anchor"], atol=1e-9, rtol=0)
        mass, jp, jr = np.zeros((model.nv, model.nv)), np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_fullM(model, mass, data.qM)
        mujoco.mj_jacSite(model, data, jp, jr, site_id)
        mass, jacobian = mass[np.ix_(dofs, dofs)], np.vstack((jp, jr))[:, dofs]
        q, velocity = data.qpos[qids].copy(), data.qvel[dofs].copy()
        limits = np.asarray([model.jnt_range[joint] for joint in joints])
        interface = OSCInterface(*(contract[key] for key in ("input_min", "input_max", "output_min", "output_max")))
        counters["preview_models_attempted"] += 1
        preview = build_joint_preview(
            mass, jacobian, contract["kp"], contract["kd"], q, sample["initial_joint"], null_kp=10.0, dt=0.002
        )
        counters["preview_models_completed"] += 1
        counters["optimizer_calls_attempted"] += 1
        solved = solve_targeted_refresh(
            preview,
            velocity,
            sample["seed"],
            sample["previous"],
            interface,
            site,
            tool,
            sample["targets"],
            sample["twists"],
            limits[:, 0],
            limits[:, 1],
            past_inner_error_rad=sample["past_inner_error_rad"],
            expected_seed_tcp_errors=sample["expected_tcp_errors"],
            expected_baseline_future_slack=sample["expected_future_slack"],
            on_response=response,
        )
        counters["optimizer_calls_completed"] += 1
        prefix.require(
            solved["response_attempts"] == counters["responses_attempted"]
            and solved["response_completions"] == counters["responses_completed"],
            "Optimizer and actual response counters must agree",
        )
        audit = None
        candidate = solved["candidate"]
        if candidate is not None:
            counters["audits_attempted"] += 1
            counters["audit_endpoint_queries_charged"] += 200
            counters["audit_transition_steps_charged"] += 2600
            audit = audit_full_horizon_witness(
                candidate,
                mass,
                jacobian,
                contract["kp"],
                contract["kd"],
                q,
                velocity,
                sample["initial_joint"],
                interface,
                sample["previous"],
                site,
                tool,
                sample["targets"],
                limits[:, 0],
                limits[:, 1],
                past_inner_error_rad=sample["past_inner_error_rad"],
                enforce_pose_caps=False,
            )
            errors = np.asarray(audit["tcp_errors"])
            future = max(
                0.0,
                float(np.max(np.linalg.norm(errors[2:, :3], axis=1) / 0.01 - 1)),
                float(np.max(np.linalg.norm(errors[2:, 3:], axis=1) / 0.05 - 1)),
            )
            np.testing.assert_allclose(future, candidate["future_quality_slack"], atol=1e-10, rtol=0)
            prefix.require(
                audit["passed"] is True
                and candidate_is_eligible(errors, future, solved["baseline_future_slack"], {"accepted": True}),
                "The independent witness must satisfy both fixed and added conditions",
            )
            counters["audits_completed"] += 1
        return dict(solver=solved, independent_audit=audit, certified_model_witness=audit is not None)
    finally:
        for name, function in originals.items():
            setattr(mujoco, name, function)


def worker():
    counters = dict.fromkeys(
        (
            "native_models_attempted",
            "native_models_completed",
            "preview_models_attempted",
            "preview_models_completed",
            "forward_attempted",
            "forward_completed",
            "physics_steps_attempted",
            "optimizer_calls_attempted",
            "optimizer_calls_completed",
            "responses_attempted",
            "responses_completed",
            "audits_attempted",
            "audits_completed",
            "audit_endpoint_queries_charged",
            "audit_transition_steps_charged",
        ),
        0,
    )
    result = dict(complete=False, passed=False, counters=counters, physics_steps=0, world_model_calls=0, fresh_cases=0)
    try:
        prefix.environment()
        prefix.require(
            os.environ.get("MUJOCO_GL") == "disable", "Static analysis requires CPU MuJoCo without rendering"
        )
        registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
        sample, identity = load_sample()
        prefix.require(
            registration["identity"] == identity and registration["source_refs"] == source_refs(), "Frozen inputs"
        )
        result["analysis"] = analyze(sample, counters)
        prefix.require(
            counters["native_models_attempted"] == counters["native_models_completed"] == 1
            and counters["preview_models_attempted"] == counters["preview_models_completed"] == 1
            and counters["forward_attempted"] == counters["forward_completed"] == 1
            and counters["optimizer_calls_attempted"] == counters["optimizer_calls_completed"] == 1
            and counters["responses_attempted"] == counters["responses_completed"] <= MAX_RESPONSES
            and counters["physics_steps_attempted"] == 0
            and counters["audits_attempted"] == counters["audits_completed"] <= 1,
            "The separately registered static resource budget must hold",
        )
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        print(publish_result(OUTPUT / "worker/result.json", result), flush=True)


def execute():
    prefix.environment()
    prefix.require(os.environ.get("MUJOCO_GL") == "disable", "Static analysis requires CPU MuJoCo without rendering")
    qa, ready = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True
        and ready["complete"] is ready["passed"] is True
        and ready["physics_steps"] == 0,
        "The full CPU regression and zero-physics preflight must pass",
    )
    before = snapshot(ROOT)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "CPU QA must bind the current sources")
    _sample, identity = load_sample()
    sources = source_refs()
    prefix.require(ready["identity"] == identity and ready["source_refs"] == sources, "Preflight identities changed")
    prefix.require(not OUTPUT.exists(), "The registered output directory must be new")
    registration_ref = publish_result(
        OUTPUT / "registration.json",
        dict(identity=identity, source_refs=sources, qa_ref=prefix.reference(QA), dry_ref=prefix.reference(DRY)),
    )
    lifecycle = run_owned_workers(
        [dict(key="static", argv=[sys.executable, "-m", MODULE, "--worker"], log=OUTPUT / "worker.log")],
        cwd=ROOT,
        timeout_seconds=1800,
    )
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    result = dict(
        complete=False, passed=False, registration_ref=registration_ref, lifecycle=lifecycle, automatic_retry=False
    )
    try:
        prefix.require(
            lifecycle["launch_error"] is None
            and len(lifecycle["workers"]) == 1
            and lifecycle["workers"][0]["returncode"] == 0
            and lifecycle["workers"][0]["actual_wait"] is lifecycle["workers"][0]["reaped"] is True,
            "The static worker must exit and be reaped",
        )
        worker_ref = prefix.reference(OUTPUT / "worker/result.json")
        saved = prefix.read(worker_ref)
        prefix.require(saved["complete"] is saved["passed"] is True, "The worker must close its budget")
        prefix.require(
            snapshot(ROOT)["sources"] == before and load_sample()[1] == identity and source_refs() == sources,
            "Sources and inputs changed",
        )
        result.update(
            complete=True,
            passed=True,
            sources_unchanged=True,
            worker_ref=worker_ref,
            certified_model_witness=saved["analysis"]["certified_model_witness"],
            counters=saved["counters"],
        )
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        print(publish_result(OUTPUT / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.dry:
        dry()
    elif args.execute:
        execute()
    else:
        worker()


if __name__ == "__main__":
    main()
