"""Independently audit the saved residual-correction QP and all eleven calls.

No correction builder or solver is used here. The native replay route and its
full-response comparisons are inherited from the previously closed checker.
"""

import argparse
import json
import os

import numpy as np
from accepted_point_sqp_math_audit import audit_qp_certificate
from check_tracking_v5_accepted_point_sqp import case_engine
from check_tracking_v5_prefix_trust_control import compare_contract
from check_tracking_v5_saved_direction_backtracking import NUMERIC_TOLERANCE, TORQUE_TOLERANCE, native_math_proof
from check_tracking_v5_tail_translation_reserve_v2 import semantic_compare
from check_tracking_v5_terminal_secant_probe import verify_response_trace
from directional_derivative_audit import _response, array, boolean, exact_array, integer
from tail_translation_reserve_audit import _comparison, _comparison_report
from terminal_secant_audit import _tree

from s2a.accepted_point_sqp_evidence import (
    CPU_ENVIRONMENT,
    PREFIX_CHECK_REF,
    REPO,
    finish,
    model_registration,
    read,
    reference,
    remember,
    require,
)
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.preview_result_publication import publish_result

ROOT = REPO / "outputs/tracking_v5_constraint_residual_repair_check_20261029"
ALPHAS = tuple(2.0**-index for index in range(9))
FIELDS = (
    "commands",
    "rollout",
    "audit",
    "tcp_errors",
    "normalized_squared_errors",
    "safety_slack",
    "prefix_length",
    "maximum_normalized_squared_error",
    "maximum_pose_cap_violation",
    "pose_loss",
    "feasible",
)


def _squared(entry):
    error = array(entry["tcp_errors"], "complete pose errors", (12, 6))
    squared = np.column_stack(
        (np.sum(error[:, :3] ** 2, axis=1) / 0.01**2, np.sum(error[:, 3:] ** 2, axis=1) / 0.05**2)
    )
    if "normalized_squared_errors" in entry:
        exact_array("original normalized pose algebra", squared, entry["normalized_squared_errors"])
    return squared


def audit_repair_problem(bundle, saved_problem=None):
    """Rebuild every QP row from the bound historical stencil, without a solve."""
    base, trial = bundle["base"], bundle["trial"]
    x, y = (array(entry["commands"], "original commands", (12, 6)) for entry in (base, trial))
    previous = array(bundle["source"]["previous_command"], "previous")[:6]
    require(previous.shape == (6,), "Completed previous command")
    lower, upper = (array(bundle["control"][key], key, (6,)) for key in ("input_min", "input_max"))
    require(np.all(lower < upper), "Strict original command bounds")
    global_lower, global_upper = np.tile(lower, (12, 1)), np.tile(upper, (12, 1))
    global_lower[0] = np.maximum(global_lower[0], previous - 0.25)
    global_upper[0] = np.minimum(global_upper[0], previous + 0.25)
    stage_lower, stage_upper = np.maximum(global_lower, x - 0.005), np.minimum(global_upper, x + 0.005)
    exact_array("original stage lower", stage_lower, bundle["phase"]["stage_lower_command"])
    exact_array("original stage upper", stage_upper, bundle["phase"]["stage_upper_command"])
    for commands in (x, y):
        require(
            np.all(commands >= stage_lower - 1e-12) and np.all(commands <= stage_upper + 1e-12),
            "Base/trial inside registered stage",
        )
        require(
            np.all(np.abs(np.diff(np.vstack((previous, commands)), axis=0)) <= 0.25 + 1e-12),
            "Base/trial original rate admission",
        )
    s0, st = (array(entry["safety_slack"], "full original safety", (4369,)) for entry in (base, trial))
    q0, qt = _squared(base), _squared(trial)
    require(np.all(s0 >= 1e-8) and np.all(q0[:9] <= 1), "Qualified original anchor")
    base_target, trial_target = float(np.max(q0[9])), float(np.max(qt[9]))
    require(trial_target < base_target, "Observed cursor9 descent")
    threshold = base_target - 0.5 * (base_target - trial_target)
    nominal = array(bundle["nominal_deltas"], "signed original stencil", (72,))
    expected_nominal = np.full(72, 1e-4)
    expected_nominal[[2, 3, 4]] = -1e-4
    exact_array("Registered signs, not reconstructed effective step", expected_nominal, nominal)
    probes = bundle["probes"]
    require(len(probes) == 72, "All seventy-two original probes")
    Js, Jq, actual = np.empty((4369, 72)), np.empty((24, 72)), []
    for axis, probe in enumerate(probes):
        command = array(probe["commands"], "one original FD command", (12, 6))
        difference = command.ravel() - x.ravel()
        require(np.all(difference[np.arange(72) != axis] == 0), "One registered FD axis only")
        tolerance = 4 * np.finfo(float).eps * max(1.0, abs(x.ravel()[axis]), abs(command.ravel()[axis]))
        require(
            difference[axis] != 0 and abs(difference[axis] - nominal[axis]) <= tolerance, "Original nominal signed step"
        )
        require(
            np.all(command >= stage_lower - 1e-12) and np.all(command <= stage_upper + 1e-12),
            "FD remains in original stage",
        )
        Js[:, axis] = (array(probe["safety_slack"], "FD safety", (4369,)) - s0) / nominal[axis]
        Jq[:, axis] = (_squared(probe).ravel() - q0.ravel()) / nominal[axis]
        actual.append(float(difference[axis]))
    command_lower = np.maximum.reduce((stage_lower, np.tile(lower, (12, 1)), y - 0.0025))
    command_upper = np.minimum.reduce((stage_upper, np.tile(upper, (12, 1)), y + 0.0025))
    require(np.all(command_lower <= command_upper), "Nonempty fixed correction box")
    identity = np.eye(72)
    difference, center = identity - np.eye(72, k=-6), np.r_[previous, np.zeros(66)]
    blocks = (
        ("safety", Js, st - 1e-8),
        ("prefix", -Jq[:18], 0.9999 - qt[:9].ravel()),
        ("target", -Jq[18:20], threshold - qt[9]),
        ("rate_lower", difference, 0.25 + difference @ y.ravel() - center),
        ("rate_upper", -difference, 0.25 - difference @ y.ravel() + center),
        ("command_lower", identity, (y - command_lower).ravel()),
        ("command_upper", -identity, (command_upper - y).ravel()),
    )
    matrices, offsets, row_blocks, count = [], [], {}, 0
    for name, matrix, offset in blocks:
        matrices.append(np.column_stack((matrix, np.zeros(len(offset)))))
        offsets.append(offset)
        row_blocks[name] = [count, count + len(offset)]
        count += len(offset)
    matrices.append(np.r_[np.zeros(72), 1.0][None])
    offsets.append(np.zeros(1))
    row_blocks["epigraph"] = [count, count + 1]
    expected = dict(
        base_commands=x,
        trial_commands=y,
        previous=previous,
        input_min=lower,
        input_max=upper,
        stage_lower=stage_lower,
        stage_upper=stage_upper,
        command_lower=command_lower,
        command_upper=command_upper,
        delta_lower=command_lower - y,
        delta_upper=command_upper - y,
        radius=0.0025,
        safety_buffer=1e-8,
        prefix_optimization_margin=1e-4,
        base_target=base_target,
        trial_target=trial_target,
        threshold=threshold,
        base_safety=s0,
        trial_safety=st,
        base_normalized_squared_errors=q0,
        trial_normalized_squared_errors=qt,
        nominal_deltas=nominal,
        actual_deltas=np.asarray(actual),
        Js=Js,
        Jq=Jq,
        A=np.vstack(matrices),
        b=np.concatenate(offsets),
        B=identity,
        g=np.zeros(72),
        initial_t=0.0,
        row_blocks=row_blocks,
        solver_iteration_limit=100,
        certificate_tolerance=1e-6,
        new_response_evaluations=0,
        empirical_error_bound_used=False,
    )
    if saved_problem is not None:
        _tree("complete independently reconstructed problem", expected, saved_problem, exact_array)
    return expected


def audit_command_plan(problem, qp, plan):
    """No clipping, alternative scale, or duplicate request can enter the ray."""
    integer(plan["planned_request_count"], 9, "Fixed nine-alpha ray")
    if qp["qualified"] is False:
        boolean(plan["admitted"], False, "Unqualified QP cannot call a model")
        require(plan["reason"] == "qp_unqualified" and plan["commands"] == [], "Explicit zero-response rejection")
        return dict(admitted=False, passed=True)
    boolean(qp["qualified"], True, "Strict QP flag")
    correction = array(qp["point"], "QP correction", (73,))[:72].reshape(12, 6)
    y = problem["trial_commands"]
    require(len(plan["commands"]) == 9, "All fixed alphas before any model")
    seen, all_admitted = {y.tobytes()}, True
    for index, (alpha, row) in enumerate(zip(ALPHAS, plan["commands"], strict=True)):
        integer(row["alpha_index"], index, "Fixed alpha index")
        require(type(row["alpha"]) is float and row["alpha"] == alpha, "Fixed dyadic alpha")
        command = y + alpha * correction
        actual = command - y
        exact_array("unclipped corrected command", command, row["commands"])
        exact_array("actual correction", actual, row["actual_delta"])
        gates = dict(
            box=bool(
                np.all(command >= problem["input_min"] - 1e-12) and np.all(command <= problem["input_max"] + 1e-12)
            ),
            rate=bool(np.all(np.abs(np.diff(np.vstack((problem["previous"], command)), axis=0)) <= 0.25 + 1e-12)),
            stage=bool(
                np.all(command >= problem["stage_lower"] - 1e-12) and np.all(command <= problem["stage_upper"] + 1e-12)
            ),
            radius=bool(np.max(np.abs(actual)) <= 0.0025 + 1e-12),
            nonzero=bool(np.any(actual != 0)),
            unique=command.tobytes() not in seen,
        )
        _tree("fixed ray admission", gates, row["admission"], exact_array)
        all_admitted = all_admitted and all(gates.values())
        seen.add(command.tobytes())
    boolean(plan["admitted"], all_admitted, "Complete preflight admission")
    require(
        plan["reason"] == ("admitted" if all_admitted else "fixed_ray_preflight_failed"), "Explicit preflight result"
    )
    return dict(admitted=all_admitted, passed=True)


def _classify(safe, squared, joint, threshold):
    gates = dict(
        joint_feasible=bool(joint),
        safety_buffer_satisfied=bool(np.all(safe >= 1e-8)),
        prefix9_preserved=bool(np.all(squared[:9] <= 1)),
        descent_preserved=bool(np.all(squared[9] <= threshold)),
    )
    return dict(
        **gates,
        prefix_optimization_margin_satisfied=bool(np.all(squared[:9] <= 0.9999)),
        repair_eligible=all(gates.values()),
        feasible=bool(joint and np.all(squared <= 1)),
    )


def audit_repair_ledger(answer, *, trial, problem, command_plan, compare=semantic_compare):
    """Check all saved responses and selection before native construction."""
    require(command_plan["admitted"] is True, "Responses only follow a fully admitted plan")
    entries, requests = answer["evaluated_candidates"], answer["request_ledger"]
    require(len(entries) == len(requests) == len(answer["classifications"]) == 11, "All eleven billed responses")
    integer(answer["response_evaluations"], 11, "All complete calls")
    integer(answer["unique_command_count"], 10, "Ten commands and forced trial repeat")
    boolean(answer["control_qualified"], False, "No physical qualification")
    commands = [
        problem["trial_commands"],
        *[array(row["commands"], "fixed command", (12, 6)) for row in command_plan["commands"]],
        problem["trial_commands"],
    ]
    require(len({c.tobytes() for c in commands}) == 10, "Only final repeat duplicates")
    classifications = []
    for index, (request, entry, command) in enumerate(zip(requests, entries, commands, strict=True)):
        integer(request["request_index"], index, "Fixed request order")
        require(
            request["role"] == ("trial" if index == 0 else "trial_repeat" if index == 10 else "correction"),
            "Fixed request role",
        )
        require(request["alpha"] == (None if index in (0, 10) else ALPHAS[index - 1]), "Fixed request alpha")
        exact_array("request command", command, request["commands"])
        require(request["command_key"] == command.tobytes().hex(), "Exact request command key")
        boolean(request["completed"], True, "Every request completed")
        safe, squared = _response(
            entry, command, problem["previous"], problem["input_min"], problem["input_max"], compare, index
        )
        classification = _classify(safe, squared, entry["audit"]["feasible"], problem["threshold"])
        _tree("strict saved classification", classification, answer["classifications"][index], compare)
        classifications.append(classification)
    require(set(FIELDS) <= set(trial), "All historical numerical fields")
    history = _comparison(answer["historical_comparison"], {key: trial[key] for key in FIELDS}, entries[0], 0, compare)
    repeat = _comparison(answer["repeatability"], entries[0], entries[-1], 10, compare)
    boolean(answer["historical_exact"], history["exact"], "Saved historical exact gate")
    boolean(answer["repeat_exact"], repeat["exact"], "Own-route trial repeat exact gate")
    selected = (
        next((i for i in range(1, 10) if classifications[i]["repair_eligible"]), None)
        if history["exact"] and repeat["exact"]
        else None
    )
    _tree("first eligible response", selected, answer["selected_response_index"], compare)
    witnesses = [i for i, row in enumerate(classifications) if row["feasible"]]
    _tree("all witness indices", witnesses, answer["witness_indices"], compare)
    boolean(answer["witness_found"], bool(witnesses), "Full original witness")
    return dict(
        passed=True,
        response_evaluations=11,
        unique_command_count=10,
        historical_exact=history["exact"],
        repeat_exact=repeat["exact"],
        selected_response_index=selected,
        witness_indices=witnesses,
        full_responses_independently_replayed_here=False,
        control_qualified=False,
    )


def verify_replayed_repair(answer, rollouts, metrics, threshold):
    require(len(rollouts) == len(metrics) == 11, "All eleven independent fresh responses")
    classifications = []
    for index, metric in enumerate(metrics):
        safe = array(metric["safety_slack"], "fresh full safety", (4369,))
        squared = array(metric["normalized_squared_errors"], "fresh full pose caps", (12, 2))
        boolean(metric["joint_feasible"], metric["joint_feasible"], "Strict fresh joint flag")
        current = _classify(safe, squared, metric["joint_feasible"], threshold)
        _tree("fresh strict classification", current, answer["classifications"][index], exact_array)
        classifications.append(current)
    repeat = _comparison_report(
        dict(rollout=rollouts[0], metrics=metrics[0]), dict(rollout=rollouts[-1], metrics=metrics[-1]), 10
    )
    boolean(answer["repeat_exact"], repeat["exact"], "Fresh full repeat exact gate")
    selected = (
        next((i for i in range(1, 10) if classifications[i]["repair_eligible"]), None)
        if answer["historical_exact"] and repeat["exact"]
        else None
    )
    _tree("fresh first eligible response", selected, answer["selected_response_index"], exact_array)
    return dict(
        passed=True,
        response_evaluations=11,
        selected_response_index=selected,
        repeat_exact=repeat["exact"],
        strict_disagreements=[],
        control_qualified=False,
        historical_scope="Exact saved-route historical fields; independent native route keeps registered numerical tolerance",
    )


def audit_execution_costs(producer, *, admitted):
    integer(producer["qp_solves"], 1, "Exactly one original QP")
    count = 11 if admitted else 0
    for key, expected in (
        ("prediction_calls", count),
        ("response_evaluations", count),
        ("unique_command_count", 10 if admitted else 0),
        ("numerical_inner_steps", 300 * count),
        ("query_count", 1 + 312 * count if admitted else 0),
        ("partial_response_queries", 0),
        ("model_construction_attempts", int(admitted)),
        ("model_constructions_completed", int(admitted)),
    ):
        integer(producer[key], expected, key)
    for key in ("physics_steps", "world_model_calls", "fresh_cases", "actual_consumed_reference_count"):
        integer(producer[key], 0, key)
    boolean(producer["control_qualified"], False, "No physical qualification")
    boolean(producer["automatic_retry"], False, "No automatic retry")
    if not admitted:
        require(
            producer["answer"] is None and producer["witness_indices"] == [] and producer["witness_found"] is False,
            "No invented zero-response trajectory evidence",
        )
    return dict(
        passed=True,
        response_evaluations=count,
        query_count=1 + 312 * count if admitted else 0,
        full_responses_independently_replayed_here=False,
    )


def main(dry=False):
    from probe_tracking_v5_constraint_residual_repair import INPUT, PLAN, load_inputs
    from probe_tracking_v5_constraint_residual_repair import ROOT as PRODUCER_ROOT
    from probe_tracking_v5_terminal_only_tail_repair import load_environment_guard

    strict_native_environment()
    require(all(os.environ.get(key) == value for key, value in CPU_ENVIRONMENT.items()), "Fixed single-thread CPU")
    bundle, refs = load_inputs()
    problem = audit_repair_problem(bundle)
    require(problem["threshold"] == 1.3145080074672406, "Preregistered half-descent threshold")
    guard = load_environment_guard(refs)
    original, source_reg = bundle["source"], bundle["source_registration"]
    _, expected_assets = model_registration(source_reg, refs)
    proof = native_math_proof()
    remember(proof["source_ref"], refs)
    prefix_check = read(remember(PREFIX_CHECK_REF, refs))
    prefix_check_reg = read(remember(prefix_check["registration_ref"], refs))
    finish(refs)
    if dry:
        print(
            json.dumps(dict(passed=True, dry=True, qp_solves=0, responses=0, queries=0, source_count=len(refs))),
            flush=True,
        )
        return
    require(not ROOT.exists(), "Exclusive checker output; no overwrite or retry")
    producer_ref = remember(reference(PRODUCER_ROOT / "verified.json"), refs)
    producer = read(producer_ref)
    require(producer["complete"] is True and producer["passed"] is True, "Normally completed producer")
    registration = read(remember(producer["registration_ref"], refs))
    require(
        registration["experiment"] == "constraint_residual_repair"
        and registration["input_ref"] == reference(INPUT)
        and registration["plan_ref"] == reference(PLAN)
        and registration["base_response_index"] == 3557
        and registration["trial_response_index"] == 3639
        and registration["seed"] == 42
        and registration["response_budget"] == 11
        and registration["qp_budget"] == 1
        and registration["qp_iteration_budget"] == 100
        and registration["automatic_retry"] is False,
        "Exact bounded registered experiment",
    )
    for ref in registration["source_refs"] + registration["input_refs"] + producer["source_refs_unchanged"]:
        remember(ref, refs)
    audit_repair_problem(bundle, producer["problem"])
    qp_check = audit_qp_certificate(producer["qp"], problem["B"], problem["g"], problem["A"], problem["b"])
    plan_check = audit_command_plan(problem, producer["qp"], producer["command_plan"])
    cost_check = audit_execution_costs(producer, admitted=plan_check["admitted"])
    compare_contract(expected_assets, producer["asset_loading_proof"], name="producer_assets")
    ledger = (
        audit_repair_ledger(
            producer["answer"], trial=bundle["trial"], problem=problem, command_plan=producer["command_plan"]
        )
        if plan_check["admitted"]
        else None
    )
    finish(refs)
    registered = publish_result(
        ROOT / "registration.json",
        dict(
            experiment="constraint_residual_repair",
            producer_ref=producer_ref,
            producer_registration_ref=producer["registration_ref"],
            input_ref=reference(INPUT),
            seed=42,
            input_refs=list(refs.values()),
            numerical_math_proof=proof,
            numeric_tolerance=NUMERIC_TOLERANCE,
            torque_tolerance=TORQUE_TOLERANCE,
            expected_responses=11 if plan_check["admitted"] else 0,
            automatic_retry=False,
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        experiment="constraint_residual_repair",
        registration_ref=registered,
        qp_check=qp_check,
        plan_check=plan_check,
        cost_check=cost_check,
        ledger_check=ledger,
        qp_solves=0,
        response_evaluations=0,
        verified_responses=0,
        unique_command_count=0,
        repeated_responses_checked=0,
        numerical_inner_steps=0,
        mj_forward_queries=0,
        query_count=0,
        max_abs_difference={},
        max_jit_vs_python_goal_difference=0.0,
        jit_python_goal_differences=0,
        boundary_classification_disagreements=[],
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        actual_consumed_reference_count=0,
        control_qualified=False,
        full_responses_independently_replayed_here=False,
    )

    def compare(label, actual, saved, tolerance=NUMERIC_TOLERANCE):
        difference = semantic_compare(label, actual, saved, tolerance)
        result["max_abs_difference"][label] = max(result["max_abs_difference"].get(label, 0.0), difference)
        return difference

    try:
        if plan_check["admitted"]:
            result["environment_admission"] = restore_cv2_environment(
                guard["exact_probe_ref"], guard["runtime_source_refs"]
            )
            os.environ.pop("MUJOCO_GL", None)
            os.environ.pop("PYOPENGL_PLATFORM", None)
            reconstruct, compare_rollout, classify, check_audit, current, compiled_goal = case_engine(
                "prefix", original, source_reg, prefix_check_reg, result, refs, compare
            )
            compare_contract(current, producer["current_quantities"], compare=compare, name="new_current")
            rollouts, metrics = [], []

            def retain_replay(commands):
                rollout, differences = reconstruct(commands)
                rollouts.append(rollout)
                return rollout, differences

            def retain_classification(commands, rollout):
                metric = classify(commands, rollout)
                metrics.append(metric)
                return metric

            verify_response_trace(
                producer["answer"],
                retain_replay,
                compare_rollout,
                retain_classification,
                check_audit,
                result,
                compare=compare,
            )
            result["independent_repair_check"] = verify_replayed_repair(
                producer["answer"], rollouts, metrics, problem["threshold"]
            )
            require(bool(compiled_goal.nopython_signatures), "Actual native quaternion compilation")
            result["native_quaternion_jit_signatures"] = [str(value) for value in compiled_goal.nopython_signatures]
            integer(result["verified_responses"], 11, "All responses replayed")
            integer(result["unique_command_count"], 10, "All unique commands")
            result["full_responses_independently_replayed_here"] = True
        finish(refs)
        strict_native_environment()
        result.update(complete=True, passed=True, source_refs_unchanged=list(refs.values()))
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        result["query_count"] = result["mj_forward_queries"]
        publish_result(ROOT / "verified.json", result)
    print(json.dumps(dict(passed=True, verified_responses=result["verified_responses"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true")
    main(parser.parse_args().dry)
