"""Replay the saved terminal-only candidate repair under every original gate.

The input loader is shared with the producer. Grid construction, saved-value
selection, and fresh numerical classification are checked independently.
"""

import argparse
import json
import os

import numpy as np
from check_tracking_v5_accepted_point_sqp import case_engine
from check_tracking_v5_prefix_trust_control import compare_contract
from check_tracking_v5_saved_direction_backtracking import NUMERIC_TOLERANCE, TORQUE_TOLERANCE, native_math_proof
from check_tracking_v5_tail_translation_reserve_v2 import semantic_compare
from check_tracking_v5_terminal_secant_probe import verify_response_trace
from directional_derivative_audit import _response, array, boolean, exact_array, integer
from tail_translation_reserve_audit import (
    AXES,
    STEPS,
    _admission,
    _comparison,
    _comparison_report,
    _displacement,
    _grid_specs,
    audit_tail_translation_grid,
    compare_tail_causal_rollouts,
)
from terminal_secant_audit import _clip_flags, _terminal, _tree

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
from s2a.terminal_development_execution import CONTROL_REF

ROOT = REPO / "outputs/tracking_v5_terminal_only_tail_repair_check_20261029"


def audit_repair_ledger(answer, *, historical_anchor, compare=semantic_compare):
    """Audit14 fully replayed saved records; scientific failure remains valid."""
    require(answer["method"] == "fixed_terminal_only_tail_repair", "Fixed restoration protocol")
    grid = answer["grid"]
    grid_check = audit_tail_translation_grid(grid, compare=compare)
    for key in ("horizon", "tail_period_index", "axis_names"):
        _tree("answer/grid " + key, grid[key], answer[key], compare)
    for key in ("anchor_commands", "previous_command", "input_min", "input_max", "signed_steps"):
        exact_array("answer/grid " + key, grid[key], answer[key])
    anchor = array(answer["anchor_commands"], "anchor", (12, 6))
    previous = array(answer["previous_command"], "previous", (6,))
    lower, upper = (array(answer[k], k, (6,)) for k in ("input_min", "input_max"))
    exact_array("historical anchor commands", historical_anchor["commands"], anchor)
    for key, value in (
        ("max_responses", 14),
        ("planned_request_count", 14),
        ("response_evaluations", 14),
        ("unique_command_count", 13),
        ("total_iterations", 0),
        ("required_prefix", 9),
    ):
        integer(answer[key], value, key)
    require(answer["safety_buffer"] == 1e-8, "Original fixed safety buffer")
    require(
        not any(
            key in answer
            for key in (
                "prior_max_abs_residual",
                "reserve_target",
                "terminal_threshold",
                "reserve_factor",
                "sufficient_reserve_restored",
            )
        ),
        "No empirical E17 qualification",
    )
    boolean(answer["control_qualified"], False, "No control qualification")
    boolean(answer["empirical_error_bound_used"], False, "No empirical error bound")
    requests, responses = answer["request_ledger"], answer["evaluated_candidates"]
    require(len(requests) == len(responses) == 14, "All fourteen billed complete calls")
    specs = [(None, None, anchor.copy(), np.zeros_like(anchor)), *_grid_specs(anchor)]
    specs.append((None, None, anchor.copy(), np.zeros_like(anchor)))
    terminal, safety, squared, keys, offset = [], [], [], [], None
    for index, (request, entry, spec) in enumerate(zip(requests, responses, specs, strict=True)):
        axis, si, command, nominal = spec
        step = None if si is None else STEPS[si]
        expected = dict(
            request_index=index,
            role="anchor" if index == 0 else ("anchor_repeat" if index == 13 else "grid"),
            axis_index=axis,
            axis_name=None if axis is None else AXES[axis],
            step_index=si,
            signed_step=step,
            response_index=index,
            calls_before=index,
            calls_after=index + 1,
            charged=True,
            outcome="repeated" if index == 13 else "computed",
            repeat_of=0 if index == 13 else None,
        )
        for key, value in expected.items():
            require(type(request[key]) is type(value) and request[key] == value, "Fixed request " + key)
        _displacement("request", command, anchor, nominal, request, step, compare)
        _admission("request.admission", command, previous, lower, upper, request["admission"], compare)
        s, q = _response(entry, command, previous, lower, upper, compare, index)
        current_offset, value = _terminal(entry)
        require(offset is None or offset == current_offset, "Fixed terminal row layout")
        offset = current_offset
        safety.append(s)
        squared.append(q)
        terminal.append(value)
        keys.append(command.tobytes())
    require(len(set(keys)) == 13 and keys[0] == keys[-1], "Mandatory repeat only, always billed")
    integer(answer["terminal_row_index"], offset, "terminal row index")
    compare("terminal_values", terminal, answer["terminal_values"])
    repeat = _comparison(answer["repeatability"], responses[0], responses[-1], 13, compare)
    historical = answer["historical_anchor_comparison"]
    integer(historical["new_response_index"], 0, "new response index")
    fields = (
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
    require(historical["compared_fields"] == list(fields), "All original numerical response fields compared")
    require(
        historical["ignored_historical_fields"] == sorted(set(historical_anchor) - set(fields)),
        "Explicit extra historical metadata",
    )
    history = _comparison(
        dict(initial_response_index=0, repeated_response_index=1, **historical["comparison"]),
        {key: historical_anchor[key] for key in fields},
        responses[0],
        1,
        compare,
    )
    causal, clipping = [], []
    require(len(answer["causal_checks"]) == 14, "Every causal comparison recorded")
    base_flags = _clip_flags(responses[0], 12)
    for index, entry in enumerate(responses):
        saved = answer["causal_checks"][index]
        expected = compare_tail_causal_rollouts(responses[0]["rollout"], entry["rollout"], index)
        require(saved == expected, "Complete exact causal difference report")
        causal.append(expected)
        changed = int(np.count_nonzero(_clip_flags(entry, 12) != base_flags))
        clipping.append(dict(response_index=index, exact=changed == 0, changed_count=changed, period_count=12))
    _tree("clipping checks", clipping, answer["clipping_checks"], compare)
    candidate_checks = []
    for index in range(1, 13):
        entry, request, current = responses[index], requests[index], causal[index]
        joint = entry["audit"]["feasible"]
        buffer = bool(np.all(safety[index] >= 1e-8))
        prefix = bool(np.all(squared[index][:9] <= 1.0))
        conditions = (
            (joint, "joint_audit"),
            (buffer, "safety_buffer"),
            (prefix, "prefix9"),
            (current["prefix_exact"], "causal_prefix"),
            (current["rotation_goals_exact"], "rotation_goals"),
            (history["exact"], "historical_anchor"),
            (repeat["exact"], "anchor_repeat"),
        )
        reasons = [name for passed, name in conditions if not passed]
        candidate_checks.append(
            dict(
                request_index=index,
                response_index=index,
                axis_index=request["axis_index"],
                axis_name=request["axis_name"],
                step_index=request["step_index"],
                signed_step=request["signed_step"],
                actual_displacement_linf=float(np.max(np.abs(array(entry["commands"], "command") - anchor))),
                terminal=terminal[index],
                terminal_improvement=terminal[index] - terminal[0],
                other_safety_min=float(np.min(safety[index][:offset])),
                safety_min=float(np.min(safety[index])),
                joint_feasible=joint,
                safety_buffer_satisfied=buffer,
                prefix9_preserved=prefix,
                causal_exact=current["passed"],
                historical_exact=history["exact"],
                repeat_exact=repeat["exact"],
                repair_eligible=not reasons,
                rejection_reasons=reasons,
            )
        )
    _tree("candidate_checks", candidate_checks, answer["candidate_checks"], compare)
    ranked = sorted(
        (row for row in candidate_checks if row["repair_eligible"]),
        key=lambda row: (-row["terminal"], row["actual_displacement_linf"], row["request_index"]),
    )
    ranking = [row["response_index"] for row in ranked]
    require(
        isinstance(answer["eligible_ranking"], list)
        and all(type(value) is int for value in answer["eligible_ranking"])
        and answer["eligible_ranking"] == ranking,
        "Strict eligible ranking",
    )
    best = ranked[0] if ranked and ranked[0]["terminal"] > terminal[0] else None
    selected_index = None if best is None else best["response_index"]
    _tree("selected response index", selected_index, answer["selected_response_index"], compare)
    summary = None
    if best is None:
        require(answer["selected"] is None, "No selected trajectory without strict improvement")
    else:
        chosen = responses[selected_index]
        # The selection copies a saved response; numeric tolerance must not hide
        # even a sub-tolerance alteration to its reported strict classifications.
        _tree("selected full response", chosen, answer["selected"], compare)
        exact_array("selected command bytes", chosen["commands"], answer["selected"]["commands"])
        _response(
            answer["selected"],
            array(chosen["commands"], "selected command"),
            previous,
            lower,
            upper,
            compare,
            selected_index,
        )
        summary = dict(best)
        summary.update(
            prefix_length=chosen["prefix_length"],
            tcp_errors=chosen["tcp_errors"],
            normalized_squared_errors=chosen["normalized_squared_errors"],
            feasible=chosen["feasible"],
        )
    _tree("selected summary", summary, answer["selected_summary"], compare)
    boolean(answer["improvement_found"], best is not None, "Strict terminal improvement")
    all_causal = all(row["passed"] for row in causal)
    boolean(answer["causal_checks_all_passed"], all_causal, "All causal checks")
    return dict(
        passed=True,
        response_evaluations=14,
        unique_command_count=13,
        grid_check=grid_check,
        repeatability=repeat,
        historical_anchor_exact=history["exact"],
        causal_checks=causal,
        causal_checks_all_passed=all_causal,
        eligible_ranking=ranking,
        selected_response_index=selected_index,
        selected_summary=summary,
        improvement_found=best is not None,
        witness_indices=[i for i, entry in enumerate(responses) if entry["feasible"]],
        full_responses_independently_replayed_here=False,
        control_qualified=False,
    )


def verify_replayed_repair(answer, rollouts, metrics):
    """Reclassify within independent arithmetic; never reuse saved strict labels."""
    require(len(rollouts) == len(metrics) == len(answer["causal_checks"]) == 14, "All fourteen fresh replays")
    require(len(answer["candidate_checks"]) == 12, "All twelve repair candidates")
    causal = []
    for index, rollout in enumerate(rollouts):
        current = compare_tail_causal_rollouts(rollouts[0], rollout, index)
        for field in ("prefix_exact", "rotation_goals_exact", "passed"):
            require(current[field] is answer["causal_checks"][index][field], "Fresh causal strict flag: " + field)
        causal.append(current)
    repeat = _comparison_report(
        dict(rollout=rollouts[0], metrics=metrics[0]),
        dict(rollout=rollouts[-1], metrics=metrics[-1]),
        13,
    )
    require(repeat["exact"] is answer["repeatability"]["exact"], "Fresh complete anchor repeat strict flag")
    history = answer["historical_anchor_comparison"]["comparison"]["exact"]
    require(type(history) is bool, "Strict saved-route historical gate")
    terminals = [float(np.asarray(row["safety_slack"])[4368]) for row in metrics]
    candidates = []
    for index in range(1, 13):
        metric = metrics[index]
        safe = np.asarray(metric["safety_slack"], dtype=float)
        squared = np.asarray(metric["normalized_squared_errors"], dtype=float)
        require(safe.shape == (4369,) and np.isfinite(safe).all(), "Full finite independent safety vector")
        require(squared.shape == (12, 2) and np.isfinite(squared).all(), "Full finite independent pose errors")
        row = dict(
            response_index=index,
            joint_feasible=metric["joint_feasible"],
            safety_buffer_satisfied=bool(np.all(safe >= 1e-8)),
            prefix9_preserved=bool(np.all(squared[:9] <= 1.0)),
            causal_exact=causal[index]["passed"],
            historical_exact=history,
            repeat_exact=repeat["exact"],
        )
        row["repair_eligible"] = all(value for key, value in row.items() if key != "response_index")
        for key, value in row.items():
            if key != "response_index":
                require(
                    type(value) is bool and value is answer["candidate_checks"][index - 1][key],
                    "Independent strict repair flag: " + key,
                )
        row["terminal"] = terminals[index]
        candidates.append(row)
    improving = [row for row in candidates if row["repair_eligible"] and row["terminal"] > terminals[0]]
    improvement = bool(improving)
    require(improvement is answer["improvement_found"], "Independent strict improvement classification")
    selected = answer["selected_response_index"]
    if selected is None:
        require(not improvement, "No omitted independently improving candidate")
    else:
        require(type(selected) is int and 1 <= selected <= 12, "One fixed selected response")
        require(candidates[selected - 1] in improving, "Selected response independently remains an eligible repair")
    return dict(
        passed=True,
        response_evaluations=14,
        causal_checks=causal,
        repeatability=repeat,
        candidate_checks=candidates,
        selected_response_index=selected,
        improvement_found=improvement,
        strict_disagreements=[],
        ranking_scope="Exact ordering uses independently audited saved-precision values in ledger_check",
        historical_scope="Exact history is checked within the saved production route; cross-route comparison keeps original tolerance",
        empirical_error_bound_used=False,
        control_qualified=False,
    )


def main(dry=False):
    from probe_tracking_v5_terminal_only_tail_repair import INPUT, load_inputs
    from probe_tracking_v5_terminal_only_tail_repair import ROOT as PRODUCER_ROOT

    strict_native_environment()
    require(all(os.environ.get(key) == value for key, value in CPU_ENVIRONMENT.items()), "Fixed single-thread CPU")
    bundle, refs = load_inputs()
    original, source_reg = bundle["source"], bundle["source_registration"]
    proof = native_math_proof()
    remember(proof["source_ref"], refs)
    prefix_check = read(remember(PREFIX_CHECK_REF, refs))
    prefix_check_reg = read(remember(prefix_check["registration_ref"], refs))
    _, expected_assets = model_registration(source_reg, refs)
    grid_check = audit_tail_translation_grid(bundle["grid"], compare=semantic_compare)
    finish(refs)
    if dry:
        print(
            json.dumps(dict(dry_run=True, passed=True, source_count=len(refs), new_responses=0, new_queries=0)),
            flush=True,
        )
        return
    require(not ROOT.exists(), "Exclusive independent replay output; no overwrite or retry")
    producer_ref = remember(reference(PRODUCER_ROOT / "verified.json"), refs)
    producer = read(producer_ref)
    require(producer["complete"] is True and producer["passed"] is True, "Completed producer before independent replay")
    registration = read(remember(producer["registration_ref"], refs))
    require(
        registration["experiment"] == "terminal_only_tail_repair"
        and registration["input_ref"] == reference(INPUT)
        and registration["seed"] == 42
        and registration["response_budget"] == 14
        and registration["automatic_retry"] is False,
        "Exact producer experiment, fixed input and response budget",
    )
    for ref in registration["source_refs"] + registration["input_refs"] + producer["source_refs_unchanged"]:
        remember(ref, refs)
    require(registration["historical_response_index"] == 3543, "One previously selected historical ray")
    answer = producer["answer"]
    require(answer["grid"] == registration["grid"] == bundle["grid"], "Complete fixed input grid binding")
    exact_array("historical_commands", answer["anchor_commands"], bundle["historical_anchor"]["commands"])
    exact_array(
        "completed367_previous", np.asarray(original["previous_command"], dtype=float)[:6], answer["previous_command"]
    )
    control = read(remember(source_reg["control_ref"], refs))["contract"]
    for field in ("input_min", "input_max"):
        exact_array("original_" + field, control[field], answer[field])
    compare_contract(expected_assets, producer["asset_loading_proof"], name="producer_assets")
    # Reject forged ledger/selection before constructing any native dynamics.
    ledger_check = audit_repair_ledger(answer, historical_anchor=bundle["historical_anchor"])
    require(
        producer["response_evaluations"] == 14 and producer["unique_command_count"] == 13, "All billed producer calls"
    )
    finish(refs)
    registered = publish_result(
        ROOT / "registration.json",
        dict(
            experiment="terminal_only_tail_repair",
            producer_ref=producer_ref,
            producer_registration_ref=producer["registration_ref"],
            input_ref=reference(INPUT),
            historical_response_index=3543,
            seed=42,
            input_refs=list(refs.values()),
            numerical_math_proof=proof,
            numeric_tolerance=NUMERIC_TOLERANCE,
            torque_tolerance=TORQUE_TOLERANCE,
            expected_responses=14,
            expected_unique_commands=13,
            expected_inner_steps=4200,
            expected_queries=4369,
            automatic_retry=False,
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        experiment="terminal_only_tail_repair",
        registration_ref=registered,
        response_evaluations=0,
        verified_responses=0,
        unique_command_count=0,
        repeated_responses_checked=0,
        numerical_inner_steps=0,
        mj_forward_queries=0,
        query_count=0,
        total_iterations=0,
        max_abs_difference={},
        max_jit_vs_python_goal_difference=0.0,
        jit_python_goal_differences=0,
        boundary_classification_disagreements=[],
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        actual_consumed_reference_count=0,
        control_qualified=False,
        empirical_error_bound_used=False,
    )

    def compare(label, actual, saved, tolerance=NUMERIC_TOLERANCE):
        difference = semantic_compare(label, actual, saved, tolerance)
        result["max_abs_difference"][label] = max(result["max_abs_difference"].get(label, 0.0), difference)
        return difference

    try:
        guard = read(remember(CONTROL_REF, refs))["cv2_guard_registration"]
        result["environment_admission"] = restore_cv2_environment(
            guard["exact_probe_ref"], guard["runtime_source_refs"]
        )
        for ref in [guard["exact_probe_ref"], *guard["runtime_source_refs"]]:
            remember(ref, refs)
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
            answer, retain_replay, compare_rollout, retain_classification, check_audit, result, compare=compare
        )
        result["grid_check"], result["ledger_check"] = grid_check, ledger_check
        try:
            result["independent_repair_check"] = verify_replayed_repair(answer, rollouts, metrics)
        except ValueError as error:
            result["boundary_classification_disagreements"].append(
                dict(scope="fresh repair/causality/repeat", error=str(error))
            )
            raise
        require(bool(compiled_goal.nopython_signatures), "Actual native quaternion compilation")
        result["native_quaternion_jit_signatures"] = [str(value) for value in compiled_goal.nopython_signatures]
        require(
            result["verified_responses"] == 14 and result["unique_command_count"] == 13, "No omitted response or repeat"
        )
        finish(refs)
        strict_native_environment()
        result.update(complete=True, passed=True, source_refs_unchanged=list(refs.values()))
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        result["query_count"] = result["mj_forward_queries"]
        publish_result(ROOT / "verified.json", result)
    print(json.dumps(dict(complete=True, passed=True, verified_responses=14)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true")
    main(parser.parse_args().dry)
