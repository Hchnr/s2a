"""Audit branch-local shifted plans at shared-budget feedback stops."""

import argparse
from pathlib import Path
from types import FunctionType

import numpy as np

from s2a.geometry import inverse, pose_error
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.joint_control.tracking.full_horizon_quality_audit import audit_full_horizon_witness
from s2a.research.joint_control.tracking.plan_model_drift import shifted_completed_plan
from s2a.research.trajectory_search.feasible_start_restoration import domain_check, linear_domain
from scripts.simulation.runtime import prepare_task2_shared_shift as binding
from scripts.simulation.runtime import registered_static_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import audit_early_quality_feasibility as context
from scripts.tracking.trajectory_search import audit_task2_shifted_seed as previous

REPO = prefix.REPO
QA = REPO / "outputs/qa/task2_shared_shift_qualification_20261111/result.json"
DRY = REPO / "outputs/task2_shared_shift_qualification_dry_20261111/result.json"
OUTPUT = REPO / "outputs/task2_shared_shift_qualification_20261111"
INDICES = (10, 11)


def source_refs():
    return (
        binding.source_refs()
        + previous.source_refs()
        + [
            prefix.reference(path)
            for path in (Path(__file__), REPO / "tests/research/native_runtime/test_task2_shared_shift_audit.py")
        ]
    )


def qualify(errors, safety_passed):
    """Separate first-pose eligibility from the existing two-point result."""
    result = previous.qualify(errors, safety_passed)
    result["model_first_pose_candidate"] = bool(safety_passed and result["point_pose_caps_satisfied"][0])
    return result


def analyze(case, inputs, stats):
    context.preload_software_gl(case, inputs["source_sha256"])
    import mujoco

    def prohibited(*_args, **_kwargs):
        raise RuntimeError("Static shifted-plan qualification forbids physical integration")

    for name in ("mj_step", "mj_step1", "mj_step2"):
        setattr(mujoco, name, prohibited)
    stats["model_initializations_attempted"] += 1
    model = mujoco.MjModel.from_xml_string(case["record_xml"])
    stats["model_initializations_completed"] += 1
    data = mujoco.MjData(model)
    periods = case["feedback_periods"]
    last = periods[-1]
    shifted = shifted_completed_plan(last)
    np.testing.assert_array_equal(shifted, case["stop_inputs"]["shifted"])
    decision = dict(step=case["stop_spec"]["stop_step"], decision=dict(refined_plan=shifted))
    current = context.restored_context(model, data, case, periods + [decision], len(periods), stats)
    np.testing.assert_array_equal(current["state"], last["actual_state"])
    np.testing.assert_array_equal(current["previous"], last["command"])
    np.testing.assert_array_equal(current["plan"], shifted)
    np.testing.assert_allclose(current["q"], last["actual_inner"][-1]["q"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(current["velocity"], last["actual_inner"][-1]["qvel"], atol=1e-10, rtol=0)
    cursor = case["intervention_spec"]["cursor"] + len(periods)
    targets = np.asarray(case["targets"][cursor : cursor + 8])
    prefix.require(targets.shape == (8, 4, 4), "Eight already-generated reference points")
    tool = inverse(np.asarray(case["controller"]["source_from_site"])) @ case["tool"]
    errors = np.asarray(
        [pose_error(target, pose @ tool) for target, pose in zip(targets, current["rollout"]["pose"], strict=True)]
    )
    profile = case["stop_inputs"]["profile"]
    domain = linear_domain(
        current["model"],
        shifted,
        current["previous"][:6],
        current["interface"],
        current["limits"][:, 0],
        current["limits"][:, 1],
        profile,
        current["rollout"],
    )
    check = domain_check(shifted, domain, current["interface"])
    candidate = dict(
        witness_found=True,
        planned_commands=shifted,
        rollout=current["rollout"],
        first_tcp_error=errors[0],
        tcp_errors=errors,
        terminal_normals=domain["terminal_normals"],
    )
    stats["independent_audits_attempted"] += 1
    stats["independent_endpoint_queries_charged"] += 200
    stats["independent_transition_steps_charged"] += 2600
    try:
        audited = audit_full_horizon_witness(
            candidate,
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
            tool,
            targets,
            current["limits"][:, 0],
            current["limits"][:, 1],
            past_inner_error_rad=profile,
            enforce_pose_caps=False,
            enforce_braking=True,
        )
        prefix.require(
            audited["passed"] is True
            and audited["independent_endpoint_queries"] == 200
            and audited["independent_transition_steps"] == 2600,
            "Complete independent model safety audit",
        )
        np.testing.assert_allclose(audited["tcp_errors"], errors, atol=1e-10, rtol=0)
        stats["independent_audits_completed"] += 1
        audit = dict(status="passed", verification=audited)
    except (AssertionError, ValueError) as error:
        audit = dict(status="rejected", error=dict(type=type(error).__name__, message=str(error)))
    quality = qualify(errors, check["accepted"] and audit["status"] == "passed")
    return dict(
        first_target_cursor=cursor,
        shifted_commands=shifted,
        tcp_errors=errors,
        linear_safety=check,
        terminal_normals=domain["terminal_normals"],
        terminal_scope="derived from the shifted completed plan, not the stopped search seed",
        independent_audit=audit,
        quality=quality,
        scope="static frozen-model assessment; no new action or physical feasibility claim",
    )


def isolated_worker():
    old = previous.worker
    namespace = old.__globals__.copy()
    namespace.update(OUTPUT=OUTPUT, load_inputs=binding.load_inputs, input_receipt=binding.input_receipt)
    namespace.update(source_refs=source_refs, analyze=analyze)
    function = FunctionType(old.__code__, namespace, old.__name__, old.__defaults__, old.__closure__)
    function.__kwdefaults__ = dict(old.__kwdefaults__ or {})
    return function


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--worker", type=int, choices=INDICES)
    args = parser.parse_args()
    prefix.environment()
    if args.dry_run:
        inputs = binding.load_inputs()
        print(
            publish(
                DRY,
                dict(
                    complete=True,
                    passed=True,
                    inputs=binding.input_receipt(inputs),
                    source_refs=source_refs(),
                    model_initializations=0,
                    physics_steps=0,
                ),
            )
        )
    elif args.run:
        registered_static_phase.execute_static_phase(
            root=OUTPUT,
            qa_path=QA,
            dry_path=DRY,
            indices=INDICES,
            worker_module="scripts.tracking.trajectory_search.audit_task2_shared_shift",
            load_inputs=binding.load_inputs,
            input_receipt=binding.input_receipt,
            source_refs=source_refs,
            summarize=previous.summarize,
        )
    else:
        isolated_worker()(args.worker)


if __name__ == "__main__":
    main()
