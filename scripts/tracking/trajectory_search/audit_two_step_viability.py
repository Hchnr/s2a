"""Audit two-point viability at four already-replayed pre-stop decisions."""

import argparse

import numpy as np

from s2a.geometry import inverse, pose_error
from s2a.infrastructure.publication import publish_result as publish
from s2a.references.generated_twist import reference_world_twist
from s2a.research.joint_control.tracking.two_step_quality_audit import audit_two_step_witness
from s2a.research.trajectory_search.feasible_start_restoration import domain_check, linear_domain
from s2a.research.trajectory_search.two_step_viability import solve_two_step_viability
from scripts.simulation.runtime import prepare_two_step_viability as binding
from scripts.simulation.runtime import registered_static_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import audit_early_quality_feasibility as previous
from scripts.tracking.trajectory_search import compare_multitask_feasibility_first as comparison

REPO = prefix.REPO
QA = REPO / "outputs/qa/two_step_viability_20261111/result.json"
DRY = REPO / "outputs/two_step_viability_dry_20261111/result.json"
OUTPUT = REPO / "outputs/two_step_viability_20261111"
INDICES = (2, 6, 10, 11)


def analyze(case, inputs, stats):
    previous.preload_software_gl(case, inputs["source_sha256"])
    import mujoco

    def prohibited(*_args, **_kwargs):
        raise RuntimeError("Static two-step feasibility forbids physical integration")

    for name in ("mj_step", "mj_step1", "mj_step2"):
        setattr(mujoco, name, prohibited)
    stats["model_initializations_attempted"] += 1
    model = mujoco.MjModel.from_xml_string(case["record_xml"])
    stats["model_initializations_completed"] += 1
    data = mujoco.MjData(model)
    decision = case["decision_inputs"]
    current = previous.restored_context(model, data, case, decision["periods"], decision["offset"], stats)
    np.testing.assert_array_equal(current["plan"], decision["recorded_plan"])
    site_from_tcp = inverse(np.asarray(case["controller"]["source_from_site"])) @ case["tool"]
    cursor = case["intervention_spec"]["cursor"] + decision["offset"]
    targets = case["targets"][cursor : cursor + 8]
    twists = np.asarray(
        [reference_world_twist(case["targets"], i, case["anchor"], 0.05)["twist"] for i in range(cursor, cursor + 8)]
    )
    prefix.require(targets.shape == (8, 4, 4) and twists.shape == (8, 6), "Already-generated eight-point reference")
    seed_error = pose_error(targets[0], current["rollout"]["pose"][0] @ site_from_tcp)
    np.testing.assert_allclose(seed_error, decision["recorded_model_first_error"], atol=1e-8, rtol=0)
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
    fingerprint = comparison.input_fingerprint(args, decision["profile"])
    seed_domain = linear_domain(
        current["model"],
        current["plan"],
        current["previous"][:6],
        current["interface"],
        current["limits"][:, 0],
        current["limits"][:, 1],
        decision["profile"],
        current["rollout"],
    )
    joint_value = seed_domain["joint_a"] @ current["plan"].ravel()
    joint_residual = np.minimum(joint_value - seed_domain["joint_lo"], seed_domain["joint_hi"] - joint_value)
    seed_safety = dict(
        check=domain_check(current["plan"], seed_domain, current["interface"]),
        minimum_tightened_inner_residual_rad=float(np.min(joint_residual[: 200 * 7])),
        minimum_braking_residual_rad=float(np.min(joint_residual[200 * 7 :])),
    )

    def response(event):
        stats[f"solver_responses_{event}"] += 1
        prefix.require(stats["solver_responses_attempted"] <= 320, "Per-solve model response budget")

    stats["solver_calls_attempted"] += 1
    stats["start_lp_budget_charged"] += 2
    solved = solve_two_step_viability(*args, past_inner_error_rad=decision["profile"], on_response=response)
    stats["solver_calls_completed"] += 1
    stats["start_lp_actual"] += solved["initialization"]["start_lp_attempts"]
    prefix.require(
        stats["solver_responses_attempted"]
        == stats["solver_responses_completed"]
        == solved["response_evaluations"]
        <= 320
        and 0 <= solved["initialization"]["start_lp_attempts"] <= 2
        and comparison.input_fingerprint(args, decision["profile"]) == fingerprint,
        "One shared response cache and unchanged causal inputs",
    )
    candidate = solved.get("best_joint_safe")
    audit = dict(status="not_attempted")
    if candidate is not None:
        stats["audit_attempts"] += 1
        stats["audit_endpoint_queries_charged"] += 200
        stats["audit_transition_steps_charged"] += 2600
        try:
            verified = audit_two_step_witness(
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
                site_from_tcp,
                targets,
                current["limits"][:, 0],
                current["limits"][:, 1],
                past_inner_error_rad=decision["profile"],
            )
            prefix.require(
                verified["passed"] is verified["prefix_pose_caps_satisfied"] is True
                and verified["independent_endpoint_queries"] == 200
                and verified["independent_transition_steps"] == 2600,
                "Independent two-point and full safety audit",
            )
            stats["audit_passed"] += 1
            audit = dict(status="passed", verification=verified)
        except (AssertionError, ValueError) as error:
            audit = dict(status="rejected", error={"type": type(error).__name__, "message": str(error)})
    return dict(
        case=case["decision_spec"],
        profile_scope=decision["profile_scope"],
        first_target_cursor=cursor,
        input_sha256=fingerprint,
        seed_safety=seed_safety,
        seed_model_first_error=seed_error,
        seed_recorded_model_difference=float(np.max(abs(seed_error - decision["recorded_model_first_error"]))),
        seed_observed_first_error=decision["observed_first_error"],
        seed_model_actual_first_difference=decision["observed_first_error"] - seed_error,
        solver=solved,
        independent_audit=audit,
        model_prefix_witness=bool(candidate is not None and audit["status"] == "passed"),
        scope="static frozen-model comparison; no new physical action or full-horizon qualification",
    )


def worker(index):
    prefix.environment()
    registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
    inputs = binding.load_inputs()
    prefix.require(
        binding.input_receipt(inputs) == registration["inputs"]
        and binding.source_refs() == registration["source_refs"],
        "Frozen pre-stop inputs and source identity",
    )
    case = next(item for item in inputs["cases"] if item["decision_spec"]["case_index"] == index)
    stats = dict(
        model_initializations_attempted=0,
        model_initializations_completed=0,
        forward_queries_attempted=0,
        forward_queries_completed=0,
        baseline_rollouts=0,
        max_qpos_restore_difference=0.0,
        solver_calls_attempted=0,
        solver_calls_completed=0,
        solver_responses_attempted=0,
        solver_responses_completed=0,
        start_lp_budget_charged=0,
        start_lp_actual=0,
        audit_attempts=0,
        audit_passed=0,
        audit_endpoint_queries_charged=0,
        audit_transition_steps_charged=0,
    )
    result = dict(complete=False, passed=False, control_qualified=False, case=case["decision_spec"], stats=stats)
    try:
        data = analyze(case, inputs, stats)
        prefix.require(
            stats["model_initializations_attempted"]
            == stats["model_initializations_completed"]
            == stats["forward_queries_attempted"]
            == stats["forward_queries_completed"]
            == stats["baseline_rollouts"]
            == stats["solver_calls_attempted"]
            == stats["solver_calls_completed"]
            == 1
            and stats["start_lp_budget_charged"] == 2
            and 0 <= stats["start_lp_actual"] <= 2
            and stats["audit_attempts"] in (0, 1)
            and stats["audit_endpoint_queries_charged"] == 200 * stats["audit_attempts"]
            and stats["audit_transition_steps_charged"] == 2600 * stats["audit_attempts"],
            "Bounded per-case static and independent audit costs",
        )
        result.update(complete=True, passed=True, data=data)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(OUTPUT / f"case_{index:02d}" / "result.json", result), flush=True)


def summarize(cases, config):
    keys = (
        "model_initializations_completed",
        "forward_queries_completed",
        "baseline_rollouts",
        "solver_calls_attempted",
        "solver_calls_completed",
        "solver_responses_attempted",
        "solver_responses_completed",
        "start_lp_budget_charged",
        "start_lp_actual",
        "audit_attempts",
        "audit_passed",
        "audit_endpoint_queries_charged",
        "audit_transition_steps_charged",
    )
    counts = {key: sum(item["stats"][key] for item in cases) for key in keys}
    prefix.require(
        counts["model_initializations_completed"]
        == counts["forward_queries_completed"]
        == counts["baseline_rollouts"]
        == counts["solver_calls_attempted"]
        == counts["solver_calls_completed"]
        == 4
        and counts["solver_responses_attempted"]
        == counts["solver_responses_completed"]
        <= config["max_solver_responses"]
        and counts["start_lp_actual"] <= counts["start_lp_budget_charged"] == config["max_start_lp_calls"]
        and counts["audit_endpoint_queries_charged"] <= config["max_audit_endpoint_queries"]
        and counts["audit_transition_steps_charged"] <= config["max_audit_transition_steps"],
        "Registered four-case model and audit budgets",
    )
    return counts


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
                    source_refs=binding.source_refs(),
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
            worker_module="scripts.tracking.trajectory_search.audit_two_step_viability",
            load_inputs=binding.load_inputs,
            input_receipt=binding.input_receipt,
            source_refs=binding.source_refs,
            summarize=summarize,
        )
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
