"""Independently replay the registered H12 common-warm baseline or trust search.

Dynamics are the previously audited independent native-query closure. All unique
trace commands are reconstructed, but only initial/selected/closest records have
archived full rollouts to compare. Candidate request and stage bookkeeping is
audited independently after strict numerical classification. This module does
not import a production solver, scorer, constraint audit, or experiment runner.
"""

import argparse
import hashlib
import json
import os

import numpy as np
from check_tracking_v5_prefix_preserving_trajectory_control import (
    audit_prefix_search_ledger,
    finite_max_abs_difference,
    require_absolute_tolerance,
)
from check_tracking_v5_saved_direction_backtracking import (
    NUMERIC_TOLERANCE,
    TORQUE_TOLERANCE,
    case_engine,
    native_math_proof,
)
from check_tracking_v5_saved_direction_backtracking_v2 import compare_contract as compare_strings

from s2a.artifacts import REPO
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.preview_result_publication import publish_result as publish
from s2a.saved_direction_backtracking_evidence import require
from s2a.terminal_development_execution import CONTROL_REF

CASES = ("baseline", "candidate")
TRACE_FIELDS = {"commands", "safety_min", "joint_feasible", "tcp_errors", "prefix_length", "feasible"}
RADII = (0.01, 0.005, 0.0025)


def numerical_compare(name, actual, saved, tolerance=NUMERIC_TOLERANCE):
    difference = finite_max_abs_difference(actual, saved, name=name)
    require_absolute_tolerance(difference, tolerance, name=name)
    return difference


def compare_contract(actual, saved, *, compare=numerical_compare, name="contract"):
    """Recurse locally so string metadata never reaches a numeric comparator."""
    if isinstance(actual, dict):
        require(isinstance(saved, dict) and set(actual) == set(saved), f"Complete {name} fields")
        for key in actual:
            compare_contract(actual[key], saved[key], compare=compare, name=f"{name}.{key}")
    elif isinstance(actual, (list, tuple)) and any(isinstance(item, dict) for item in actual):
        require(type(saved) is type(actual) and len(saved) == len(actual), f"Complete {name} sequence")
        for index, (value, archived) in enumerate(zip(actual, saved, strict=True)):
            compare_contract(value, archived, compare=compare, name=f"{name}[{index}]")
    elif actual is None or isinstance(actual, (str, bool, int)):
        require(type(saved) is type(actual) and saved == actual, f"Exact {name} label")
    else:
        values, archived = np.asarray(actual), np.asarray(saved)
        if values.dtype.kind in "US":
            compare_strings(actual, saved, compare=compare, name=name)
        elif values.dtype.kind in "biu":
            allowed = archived.dtype.kind == "b" if values.dtype.kind == "b" else archived.dtype.kind in "iu"
            require(
                allowed and values.shape == archived.shape and np.array_equal(values, archived),
                f"Exact {name} labels",
            )
        else:
            compare(name, actual, saved)


def command_key(commands, horizon):
    array = np.asarray(commands, dtype=float)
    require(array.shape == (horizon, 6) and np.all(np.isfinite(array)), "Finite full-horizon command matrix")
    return array.tobytes().hex()


def normalized_errors(errors, horizon):
    errors = np.asarray(errors, dtype=float)
    require(errors.shape == (horizon, 6) and np.all(np.isfinite(errors)), "Finite full-horizon TCP errors")
    normalized = np.column_stack(
        (np.sum(errors[:, :3] ** 2, axis=1) / 0.01**2, np.sum(errors[:, 3:] ** 2, axis=1) / 0.05**2)
    )
    require(np.all(np.isfinite(normalized)), "Finite normalized squared errors")
    return normalized


def continuous_prefix(normalized):
    failures = np.flatnonzero(np.any(np.asarray(normalized) > 1.0, axis=1))
    return int(failures[0]) if len(failures) else len(normalized)


def validate_trace_entry(saved, metrics, *, compare=numerical_compare):
    """Require strict classifications on both saved and independent arithmetic."""
    require(set(saved) == TRACE_FIELDS, "Complete compact trace fields")
    horizon = len(saved["commands"])
    squared = normalized_errors(saved["tcp_errors"], horizon)
    prefix = continuous_prefix(squared)
    independent_prefix = continuous_prefix(metrics["normalized_squared_errors"])
    require(
        type(saved["prefix_length"]) is int
        and saved["prefix_length"] == prefix == independent_prefix == metrics["prefix_length"]
        and saved["joint_feasible"] is metrics["joint_feasible"]
        and saved["feasible"] is metrics["feasible"]
        and saved["feasible"] is bool(saved["joint_feasible"] and prefix == horizon),
        "Strict saved and independent prefix/joint/witness labels agree without relaxed caps",
    )
    compare("trace_tcp_errors", metrics["tcp_errors"], saved["tcp_errors"])
    compare("trace_normalized_squared_errors", metrics["normalized_squared_errors"], squared)
    compare("trace_safety_minimum", metrics["safety_min"], saved["safety_min"])
    return squared


def check_full_candidate(saved, commands, rollout, metrics, compare_rollout, check_audit, *, compare):
    require(
        command_key(saved["commands"], len(commands)) == command_key(commands, len(commands)),
        "Full candidate binds this exact unique response",
    )
    compare_rollout(rollout, saved["rollout"])
    check_audit(saved["audit"], metrics)
    for key in (
        "tcp_errors",
        "normalized_squared_errors",
        "safety_slack",
        "maximum_normalized_squared_error",
        "maximum_pose_cap_violation",
        "pose_loss",
    ):
        compare("full_" + key, metrics[key], saved[key])
    require(
        type(saved["prefix_length"]) is int
        and saved["prefix_length"] == metrics["prefix_length"]
        and saved["feasible"] is metrics["feasible"],
        "Strict full-candidate eligibility",
    )


def trace_ranks(answer, classifications, *, compare=numerical_compare):
    """Recompute exact saved ordering after independent numerical comparison."""
    horizon = answer["horizon"]
    require(type(horizon) is int and 1 <= horizon <= 16, "Explicit finite horizon")
    trace = answer["evaluated_candidates"]
    require(len(trace) == len(classifications) >= 1, "Independent classification for every unique response")
    squared, keys, prefixes = [], [], []
    for saved, metrics in zip(trace, classifications, strict=True):
        squared.append(validate_trace_entry(saved, metrics, compare=compare))
        keys.append(command_key(saved["commands"], horizon))
        prefixes.append(saved["prefix_length"])
    require(len(keys) == len(set(keys)), "Every command-byte response is charged only once")
    witness = [(float(np.max(q)), float(np.sum(q)), i) for i, q in enumerate(squared)]
    ordinary = [
        (-prefix, float(np.max(q[prefix])) if prefix < horizon else 0.0, float(np.sum(q)), i)
        for i, (q, prefix) in enumerate(zip(squared, prefixes, strict=True))
    ]

    def best(indices):
        require(bool(indices), "A retained eligible candidate exists")
        complete = [i for i in indices if trace[i]["feasible"]]
        return min(complete, key=lambda i: witness[i]) if complete else min(indices, key=lambda i: ordinary[i])

    return squared, keys, prefixes, best


def audit_trust_search_ledger(answer, classifications, *, compare=numerical_compare):
    """Check recorded requests and trust stages, never infer unrecorded calls.

    Numeric classification must already be independently reconstructed. Ranking uses
    the checked saved arithmetic and global first-response order. Derivative groups
    are checked against each stage's actual bounds; an exhausted budget may leave
    the final group incomplete, but cannot create a free response or acceptance.
    """
    squared, keys, prefixes, best = trace_ranks(answer, classifications, compare=compare)
    trace, horizon = answer["evaluated_candidates"], answer["horizon"]
    budget, iteration_budget, stage_budget = answer["max_evaluations"], answer["max_iterations"], answer["max_stages"]
    require(
        all(type(value) is int and value >= 1 for value in (budget, iteration_budget, stage_budget))
        and len(trace) == answer["response_evaluations"] <= budget,
        "Finite shared response, iteration, and stage budgets",
    )
    require(command_key(answer["initial"]["commands"], horizon) == keys[0], "Initial is response zero")
    require(command_key(answer["used_start"], horizon) == keys[0], "Initial response uses exact supplied start")
    require(trace[0]["joint_feasible"] is True, "Safe warm required")
    if not trace[0]["feasible"]:
        require(trace[0]["safety_min"] >= 1e-8 and classifications[0]["safety_min"] >= 1e-8, "Warm buffer is strict")
    safe = [i for i, row in enumerate(trace) if row["joint_feasible"]]
    witnesses = [i for i in safe if trace[i]["feasible"]]
    selected = best(safe)
    require(command_key(answer["selected"]["commands"], horizon) == keys[selected], "Final global safe ranking")
    require(
        answer["witness_found"] is bool(witnesses)
        and answer["status"] == ("witness" if witnesses else "finite_search_without_witness")
        and answer["initial_prefix_length"] == prefixes[0]
        and answer["selected_prefix_length"] == prefixes[selected] >= prefixes[0],
        "Final strict witness and retained-prefix labels",
    )
    phases, requests = answer["phases"], answer["request_ledger"]
    require(len(phases) == answer["total_stages"] <= stage_budget, "Declared finite stage count")
    lo, hi = np.asarray(answer["lower_command"]), np.asarray(answer["upper_command"])
    require(lo.shape == hi.shape == (horizon, 6) and np.all(lo <= hi), "Finite original command bounds")
    require(np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)), "Finite original command bounds")
    roles = {
        "warm",
        "stage_anchor",
        "objective",
        "constraint",
        "gradient_base",
        "gradient_probe",
        "jacobian_base",
        "jacobian_probe",
        "callback",
        "final",
    }
    seen, used_responses, request_checks = {}, 0, []
    stage_touched = {index: set() for index in range(len(phases))}
    stage_callbacks = {index: 0 for index in range(len(phases))}
    for index, request in enumerate(requests):
        require(
            type(request["request_index"]) is int
            and request["request_index"] == index
            and request["role"] in roles
            and type(request["response_evaluations_before"]) is int
            and type(request["response_evaluations_after"]) is int
            and request["response_evaluations_before"] == used_responses,
            "Contiguous recorded requests and charged-response count",
        )
        stage, role, key = request["stage_index"], request["role"], request["command_key"]
        require(isinstance(key, str), "Explicit float64 command-byte key")
        try:
            raw = bytes.fromhex(key)
            commands = np.frombuffer(raw, dtype=np.float64).reshape(horizon, 6)
        except (ValueError, TypeError) as error:
            raise ValueError("Invalid finite full command-byte key") from error
        require(command_key(commands, horizon) == key, "Canonical finite float64 byte key")
        if index == 0:
            require(stage is None and role == "warm" and key == keys[0], "One initial warm request")
            lower, upper = lo, hi
        else:
            require(
                type(stage) is int and 0 <= stage < len(phases) and role != "warm", "Declared stage and request role"
            )
            phase = phases[stage]
            require(
                phase["requests_before"] <= index < phase["requests_after"], "Request belongs to its stage interval"
            )
            lower, upper = np.asarray(phase["stage_lower_command"]), np.asarray(phase["stage_upper_command"])
        in_bounds = bool(np.all(commands >= lower - 1e-12) and np.all(commands <= upper + 1e-12))
        outcome = request["outcome"]
        if outcome == "bounds_rejected":
            raise ValueError("A completed solver answer cannot contain an unhandled bounds rejection")
        else:
            require(in_bounds, "Every new or cached request respects its current stage bounds")
            if outcome == "cache_hit":
                require(
                    key in seen and type(request["response_index"]) is int and request["response_index"] == seen[key],
                    "Cache hit references an earlier exact response",
                )
                require(request["new_response_index"] is None, "Cache hit is not charged again")
            elif outcome == "new_response":
                require(key not in seen and used_responses < budget, "New response requires remaining shared budget")
                require(
                    type(request["response_index"]) is type(request["new_response_index"]) is int
                    and request["response_index"] == request["new_response_index"] == used_responses
                    and used_responses < len(keys)
                    and key == keys[used_responses],
                    "New responses match exact global trace order",
                )
                seen[key] = used_responses
                used_responses += 1
            elif outcome == "budget_rejected":
                require(key not in seen and used_responses == budget, "Only an uncached request can exhaust the budget")
                require(
                    request["response_index"] is request["new_response_index"] is None,
                    "No response after budget exhaustion",
                )
                require(index == len(requests) - 1, "Budget rejection is the final request")
            else:
                raise ValueError("Unknown request outcome")
        require(request["response_evaluations_after"] == used_responses, "Exact request response accounting")
        if outcome in ("new_response", "cache_hit") and stage is not None:
            stage_touched[stage].add(request["response_index"])
        if role == "callback":
            stage_callbacks[stage] += 1
        if not role.endswith("_probe"):
            require(
                request["axis"] is None and request["nominal_delta"] is None,
                "Only finite-difference probes carry an axis",
            )
        request_checks.append((commands, in_bounds))
    require(used_responses == len(trace), "No unrequested or uncharged trace response")
    require(bool(requests) and requests[0]["outcome"] == "new_response", "Warm is charged once")

    step = answer["finite_difference_step"]
    require(step == 1e-4, "Registered absolute finite-difference step")
    for index, request in enumerate(requests):
        role = request["role"]
        if role not in ("gradient_base", "jacobian_base"):
            continue
        if request["outcome"] == "budget_rejected":
            continue
        stage = request["stage_index"]
        base = request_checks[index][0].ravel()
        upper = np.asarray(phases[stage]["stage_upper_command"]).ravel()
        lower = np.asarray(phases[stage]["stage_lower_command"]).ravel()
        for axis in range(6 * horizon):
            probe_index = index + axis + 1
            require(
                probe_index < len(requests),
                "Every started derivative has a complete or explicitly rejected probe group",
            )
            probe = requests[probe_index]
            require(
                probe["stage_index"] == stage
                and probe["role"] == role.replace("_base", "_probe")
                and probe["axis"] == axis,
                "Contiguous coordinate derivative group in the same stage",
            )
            delta = step if base[axis] + step <= upper[axis] else -step
            expected = base.copy()
            expected[axis] += delta
            require(
                probe["nominal_delta"] == delta
                and np.all(expected >= lower - 1e-12)
                and np.all(expected <= upper + 1e-12)
                and command_key(expected.reshape(horizon, 6), horizon) == probe["command_key"],
                "Stage-local one-sided finite differences use the exact nominal step",
            )
            if probe["outcome"] == "budget_rejected":
                break
    for index, request in enumerate(requests):
        if request["role"].endswith("_probe"):
            axis = request["axis"]
            require(
                type(axis) is int and 0 <= axis < horizon * 6 and index > axis, "Valid finite-difference coordinate"
            )
            base = requests[index - axis - 1]
            require(
                base["role"] == request["role"].replace("_probe", "_base")
                and base["stage_index"] == request["stage_index"],
                "Every probe has its recorded preceding derivative base",
            )

    current, used_iterations, previous_requests, previous_responses, radius_index = 0, 0, 1, 1, 0
    accepted_indices, phase_checks, accepted_specs = [0], [], [(0, 0, None)]
    for index, phase in enumerate(phases):
        require(
            index < stage_budget
            and phase["stage_index"] == index
            and phase["anchor_before_index"] == current
            and phase["requests_before"] == previous_requests
            and phase["evaluations_before"] == previous_responses,
            "Stage starts from the preceding accepted anchor and contiguous ledger boundaries",
        )
        end, response_end = phase["requests_after"], phase["evaluations_after"]
        require(type(end) is int and previous_requests < end <= len(requests), "Finite stage request interval")
        require(
            requests[previous_requests]["role"] == "stage_anchor"
            and requests[previous_requests]["command_key"] == keys[current]
            and requests[previous_requests]["outcome"] == "cache_hit"
            and sum(row["role"] == "stage_anchor" for row in requests[previous_requests:end]) == 1,
            "Each stage explicitly touches its own cached anchor once at entry",
        )
        require(
            response_end
            == (requests[end - 1]["response_evaluations_after"] if end > previous_requests else previous_responses),
            "Stage response boundary agrees with its request ledger",
        )
        require(
            phase["phase"] == "prefix_trust"
            and phase["response_budget_remaining_before"] == budget - previous_responses
            and phase["response_budget_remaining_after"] == budget - response_end,
            "Exact remaining shared response budget",
        )
        require(
            phase["radius_index"] == radius_index
            and phase["radius"] == RADII[radius_index]
            and phase["max_iterations"] == min(5, iteration_budget - used_iterations)
            and previous_responses < budget
            and used_iterations < iteration_budget,
            "Registered radius schedule and only remaining iteration budget",
        )
        anchor = np.asarray(trace[current]["commands"])
        require(
            np.array_equal(np.asarray(phase["stage_lower_command"]), np.maximum(lo, anchor - RADII[radius_index]))
            and np.array_equal(np.asarray(phase["stage_upper_command"]), np.minimum(hi, anchor + RADII[radius_index])),
            "Trust bounds are the exact intersection around the accepted anchor",
        )
        cursor = prefixes[current]
        require(
            cursor < horizon
            and phase["prefix_before"] == phase["target_cursor"] == phase["cursor"] == cursor
            and command_key(phase["initial_commands"], horizon) == keys[current],
            "Each stage targets only its accepted anchor's next failing cursor",
        )
        compare("stage_initial_safety", trace[current]["safety_min"], phase["initial_safety_min"])
        compare(
            "stage_initial_epigraph",
            max(0.0, float(squared[current][cursor].max()) - 1.0) + 1e-6,
            phase["initial_epigraph"],
        )
        callbacks, iterations = phase["callback_count"], phase["iterations"]
        require(
            type(callbacks) is int
            and type(iterations) is int
            and callbacks == stage_callbacks[index]
            and 0 <= callbacks <= iterations <= phase["max_iterations"]
            and type(phase["optimizer_returned"]) is bool
            and type(phase["success"]) is bool,
            "Callbacks and stages charge their actual finite iteration counts",
        )
        if phase["optimizer_returned"]:
            require(
                type(phase["optimizer_iterations"]) is int
                and 0 <= phase["optimizer_iterations"] <= phase["max_iterations"],
                "Integral bounded optimizer nit",
            )
            require(
                iterations == max(callbacks, phase["optimizer_iterations"]), "Normal return charges max(callbacks,nit)"
            )
            final_requests = [row for row in requests[previous_requests:end] if row["role"] == "final"]
            require(
                len(final_requests) == 1
                and final_requests[0] is requests[end - 1]
                and final_requests[0]["command_key"] == command_key(phase["final_commands"], horizon)
                and np.isfinite(phase["final_epigraph"])
                and phase["final_epigraph"] >= 0,
                "Normal optimizer return retains and evaluates the exact finite final point",
            )
        else:
            require(
                iterations == callbacks and "optimizer_iterations" not in phase, "Interrupted stage charges callbacks"
            )
            require(
                not any(row["role"] == "final" for row in requests[previous_requests:end]),
                "Interrupted stage has no fabricated final return",
            )
        if callbacks:
            last_callback = [row for row in requests[previous_requests:end] if row["role"] == "callback"][-1]
            require(
                command_key(phase["last_callback_commands"], horizon) == last_callback["command_key"]
                and np.isfinite(phase["last_callback_epigraph"])
                and phase["last_callback_epigraph"] >= 0,
                "Last callback snapshot matches the actual recorded callback request",
            )
        else:
            require(
                "last_callback_commands" not in phase and "last_callback_epigraph" not in phase,
                "No invented callback snapshot",
            )
        touched = sorted(stage_touched[index])
        require(
            phase["touched_response_indices"] == touched, "Stage candidates are exactly completed touched responses"
        )
        eligible, decisions = [], []
        before_target = float(squared[current][cursor].max())
        for candidate in touched:
            row, reasons = trace[candidate], []
            if not row["feasible"]:
                if not row["joint_feasible"]:
                    reasons.append("joint_or_command")
                if prefixes[candidate] < cursor:
                    reasons.append("prefix_lost")
                if row["safety_min"] < 1e-8:
                    reasons.append("safety_buffer")
                if prefixes[candidate] <= cursor and before_target - float(squared[candidate][cursor].max()) <= 1e-12:
                    reasons.append("no_target_descent")
            if not reasons:
                require(
                    row["feasible"] or classifications[candidate]["safety_min"] >= 1e-8,
                    "Independently checked accepted safety buffer",
                )
                eligible.append(candidate)
            decisions.append(
                dict(response_index=candidate, eligible=not reasons, selected=False, rejection_reasons=reasons)
            )
        chosen = best(eligible) if eligible else current
        accepted = bool(eligible)
        reason = (
            (
                "witness"
                if trace[chosen]["feasible"]
                else "prefix_growth"
                if prefixes[chosen] > cursor
                else "target_descent"
            )
            if accepted
            else None
        )
        for decision in decisions:
            decision["selected"] = bool(accepted and decision["response_index"] == chosen)
            if decision["eligible"] and not decision["selected"]:
                decision["rejection_reasons"] = ["not_selected_by_rank"]
        compare_contract(decisions, phase["candidate_decisions"], name="candidate_decisions")
        compare_contract(
            [row for row in decisions if not row["selected"]], phase["rejected_candidates"], name="rejected_candidates"
        )
        require(
            phase["accepted"] is accepted
            and phase["acceptance_reason"] == reason
            and phase["anchor_after_index"] == chosen
            and phase["prefix_after"] == prefixes[chosen]
            and command_key(phase["selected_commands"], horizon) == keys[chosen],
            "Exact nonlinear acceptance and candidate ranking",
        )
        next_radius = 0.01 if accepted else RADII[radius_index + 1] if radius_index + 1 < len(RADII) else None
        require(phase["radius_after"] == next_radius, "Accepted steps reset radius; rejected stages shrink or stop")
        if phase["status"] == "witness_found":
            require(
                phase["optimizer_returned"] is False
                and phase["success"] is False
                and callbacks >= 1
                and trace[chosen]["feasible"]
                and requests[end - 1]["role"] == "callback",
                "Callback interruption requires a complete touched witness",
            )
        elif phase["status"] == "response_budget":
            require(
                response_end == budget
                and phase["success"] is False
                and index == len(phases) - 1
                and requests[end - 1]["outcome"] == "budget_rejected",
                "Budget interruption keeps all completed responses and is the final stage",
            )
        else:
            require(
                type(phase["status"]) is int
                and phase["optimizer_returned"] is True
                and phase["success"] is (phase["status"] == 0),
                "Normal optimizer status never substitutes for acceptance or a witness",
            )
        if accepted:
            accepted_indices.append(chosen)
            accepted_specs.append((chosen, current, index))
        used_iterations += iterations
        phase_checks.append(
            dict(
                stage_index=index,
                selected_trace_index=chosen,
                accepted=accepted,
                radius=phase["radius"],
                touched_response_indices=touched,
            )
        )
        current, previous_requests, previous_responses = chosen, end, response_end
        radius_index = 0 if accepted else radius_index + 1
        if index + 1 < len(phases):
            require(
                not trace[current]["feasible"] and radius_index < len(RADII),
                "Only unfinished valid-radius stages continue",
            )
    require(
        previous_requests == len(requests) and previous_responses == len(trace),
        "All recorded work belongs to warm or one stage",
    )
    require(used_iterations == answer["total_iterations"] <= iteration_budget, "Total iteration budget is never reset")
    require(
        [anchor["response_index"] for anchor in answer["accepted_anchors"]] == accepted_indices,
        "Only eligible selected stages produce anchors",
    )
    for anchor, (chosen, before, stage) in zip(answer["accepted_anchors"], accepted_specs, strict=True):
        cursor = (
            phases[stage]["target_cursor"]
            if stage is not None
            else None
            if trace[chosen]["feasible"]
            else prefixes[chosen]
        )
        before_max = float(squared[before][cursor].max()) if cursor is not None else None
        after_max = float(squared[chosen][cursor].max()) if cursor is not None else None
        expected = dict(
            is_initial=stage is None,
            stage_index=stage,
            response_index=chosen,
            previous_response_index=before if stage is not None else None,
            commands=trace[chosen]["commands"],
            prefix_length=prefixes[chosen],
            joint_feasible=trace[chosen]["joint_feasible"],
            safety_min=trace[chosen]["safety_min"],
            feasible=trace[chosen]["feasible"],
            tcp_errors=trace[chosen]["tcp_errors"],
            normalized_squared_errors=squared[chosen],
            stage_target_cursor=cursor,
            stage_target_max_before=before_max,
            stage_target_max_after=after_max,
            target_improvement=before_max - after_max if cursor is not None else None,
            radius=phases[stage]["radius"] if stage is not None else None,
            acceptance_reason=phases[stage]["acceptance_reason"]
            if stage is not None
            else "initial_witness"
            if trace[chosen]["feasible"]
            else "warm",
        )
        require(set(anchor) == set(expected) | {"audit"}, "Complete accepted-anchor evidence fields")
        compare_contract(expected, {key: anchor[key] for key in expected}, compare=compare, name="accepted_anchor")
    if witnesses:
        termination = "witness"
    elif len(trace) >= budget:
        termination = "response_budget"
    elif used_iterations >= iteration_budget:
        termination = "iteration_budget"
    elif len(phases) >= stage_budget:
        termination = "stage_budget"
    else:
        require(radius_index == len(RADII), "No budget remains implicit when finite radius retries stop")
        termination = "trust_region_stalled"
    require(
        answer["termination_reason"] == termination, "Termination follows witnessed progress and actual budget evidence"
    )
    return dict(
        selected_trace_index=selected,
        joint_feasible_indices=safe,
        witness_indices=witnesses,
        initial_prefix_length=prefixes[0],
        selected_prefix_length=prefixes[selected],
        total_iterations=used_iterations,
        phase_checks=phase_checks,
        accepted_anchor_indices=accepted_indices,
        request_count=len(requests),
        termination_reason=termination,
    )


def main(case, dry=False):
    from s2a.prefix_trust_control_evidence import (
        CPU_ENVIRONMENT,
        NATIVE_PYTHON,
        PLAN,
        PREFIX_CHECK_REF,
        SOURCES,
        closed_owner,
        finish,
        load_prerequisites,
        read,
        reference,
        remember,
    )

    require(case in CASES, "Exactly one preregistered A/B case")
    strict_native_environment()
    require(
        {key: os.environ.get(key) for key in CPU_ENVIRONMENT} == CPU_ENVIRONMENT,
        "Registered CPU-only single-thread environment",
    )
    original, prefix_reg, backtracking, refs = load_prerequisites()
    proof = native_math_proof()
    remember(proof["source_ref"], refs)
    warm = np.asarray(backtracking["cases"]["prefix"]["entries"][55]["request"]["commands"], dtype=float)
    warm_key = command_key(warm, 12)
    prefix_check = read(remember(PREFIX_CHECK_REF, refs))
    prefix_check_reg = read(remember(prefix_check["registration_ref"], refs))
    finish(refs)
    if dry:
        print(
            json.dumps(
                dict(
                    ready=True,
                    dry=True,
                    case_name=case,
                    source_refs=len(refs),
                    warm_sha256=hashlib.sha256(warm.tobytes()).hexdigest(),
                    new_responses=0,
                    new_queries=0,
                    numerical_math_proof=proof,
                )
            ),
            flush=True,
        )
        return

    output = REPO / f"outputs/tracking_v5_prefix_trust_{case}_check_20261028"
    require(not output.exists(), "New independent result path; no overwrite or retry")
    owner_ref = reference(REPO / f"outputs/tracking_v5_prefix_trust_{case}_execution_20261028/result.json")
    owner, producer = closed_owner(owner_ref, refs)
    producer_ref = reference(REPO / f"outputs/tracking_v5_prefix_trust_{case}_20261028/verified.json")
    require(owner["result_ref"] == producer_ref, "Closed owner binds the intended new producer")
    launch = read(remember(owner["owned_command"]["launch"], refs))
    require(
        launch["command"]
        == [str(NATIVE_PYTHON), str(REPO / "scripts/probe_tracking_v5_prefix_trust_control.py"), "--case", case],
        "Only the registered producer command and case",
    )
    registration = read(remember(producer["registration_ref"], refs))
    require(
        registration["case_name"] == producer["case_name"] == case
        and registration["plan_ref"] == reference(PLAN)
        and registration["seed"] == 42,
        "Fixed common-warm case, preregistration, and seed",
    )
    require(
        registration["backtracking_ref"]
        == reference(REPO / "outputs/tracking_v5_saved_direction_backtracking_20261028/verified.json"),
        "The previously qualified common warm has the exact registered source",
    )
    for ref in registration["source_refs"] + registration["input_refs"]:
        remember(ref, refs)
    answer = producer["answer"]
    require(command_key(producer["warm"], 12) == warm_key, "Both arms use the exact common warm bytes")
    for key in ("supplied_warm", "used_start"):
        require(command_key(answer[key], 12) == warm_key, f"No clipping or replacement of {key}")
    compare_contract(np.zeros((12, 6)), answer["start_roundoff_clip"], name="no_warm_clip")
    require(
        answer["horizon"] == answer["pose_caps_horizon"] == 12
        and answer["all_pose_caps_enforced"] is True
        and answer["max_evaluations"] == 4000
        and answer["max_iterations"] == 60
        and answer["finite_difference_step"] == 1e-4
        and answer["optimization_safety_buffer"] == 1e-8,
        "Original complete H12 caps and fixed search budgets",
    )
    if case == "candidate":
        require(answer["max_stages"] == 60, "Registered finite stage budget")
    compare_contract(original["terminal_normals"], answer["terminal_normals"], name="fixed_terminal")
    compare_contract(original["past_inner_error_rad"], answer["past_inner_error_rad"], name="zero_history")
    compare_contract(original["answer"]["lower_command"], answer["lower_command"], name="base_lower_bounds")
    compare_contract(original["answer"]["upper_command"], answer["upper_command"], name="base_upper_bounds")
    trace = answer["evaluated_candidates"]
    n = len(trace)
    require(1 <= n <= 4000, "Registered finite full response count")
    require(
        producer["prediction_calls"] == producer["response_evaluations"] == answer["response_evaluations"] == n
        and producer["total_iterations"] == answer["total_iterations"] <= 60
        and producer["numerical_inner_steps"] == 300 * n
        and producer["query_count"] == 1 + 312 * n,
        "Every unique complete response is charged exactly once",
    )
    require(
        producer["control_qualified"] is False
        and all(producer[key] == 0 for key in ("physics_steps", "world_model_calls", "fresh_cases")),
        "Numerical search does not qualify actual control or task performance",
    )
    new_registration = publish(
        output / "registration.json",
        dict(
            case_name=case,
            producer_ref=producer_ref,
            producer_owner_ref=owner_ref,
            producer_registration_ref=producer["registration_ref"],
            plan_ref=reference(PLAN),
            seed=42,
            source_refs=[reference(REPO / path) for path in SOURCES],
            input_refs=list(refs.values()),
            numerical_math_proof=proof,
            numeric_tolerance=NUMERIC_TOLERANCE,
            torque_tolerance=TORQUE_TOLERANCE,
            expected_responses=n,
            expected_inner_steps=300 * n,
            expected_queries=1 + 312 * n,
            evidence_scope="All compact trace errors and labels independently reconstructed; full archived q/v/FK/torque/slacks compared only for initial/selected/closest",
            automatic_retry=False,
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        case_name=case,
        registration_ref=new_registration,
        response_evaluations=0,
        verified_responses=0,
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
        control_qualified=False,
    )

    def compare(label, actual, saved, tolerance=NUMERIC_TOLERANCE):
        difference = numerical_compare(label, actual, saved, tolerance)
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
            "prefix", original, prefix_reg, prefix_check_reg, result, refs, compare
        )
        compare_contract(current, producer["current_quantities"], compare=compare, name="new_current")
        full = {key: answer[key] for key in ("initial", "selected", "closest_safety_attempt")}
        full_keys = {key: command_key(value["commands"], 12) for key, value in full.items()}
        checked_full, classifications, saved_safety_ranks = set(), [], []
        anchors = answer["accepted_anchors"] if case == "candidate" else []
        checked_anchors = set()
        previous = np.asarray(original["previous_command"][:6], dtype=float)
        control = read(remember(prefix_reg["control_ref"], refs))["contract"]
        lower, upper = np.asarray(control["input_min"]), np.asarray(control["input_max"])
        for index, saved in enumerate(trace):
            commands = np.asarray(saved["commands"], dtype=float)
            key = command_key(commands, 12)
            rollout, _ = reconstruct(commands)
            result["response_evaluations"] += 1
            result["query_count"] = result["mj_forward_queries"]
            metrics = classify(commands, rollout)
            try:
                normalized = validate_trace_entry(saved, metrics, compare=compare)
            except ValueError:
                result["boundary_classification_disagreements"].append(
                    dict(
                        trace_index=index,
                        producer_prefix=saved["prefix_length"],
                        independent_prefix=metrics["prefix_length"],
                        producer_joint=saved["joint_feasible"],
                        independent_joint=metrics["joint_feasible"],
                        producer_feasible=saved["feasible"],
                        independent_feasible=metrics["feasible"],
                        producer_normalized=normalized_errors(saved["tcp_errors"], 12),
                        independent_normalized=metrics["normalized_squared_errors"],
                        evidence="Numerical check rejected; no inference of actual physical violation or search success",
                    )
                )
                raise
            classifications.append(
                {
                    key: metrics[key]
                    for key in (
                        "joint_feasible",
                        "feasible",
                        "prefix_length",
                        "normalized_squared_errors",
                        "tcp_errors",
                        "safety_min",
                    )
                }
            )
            rate = np.diff(np.vstack((previous, commands)), axis=0)
            worst = min(
                saved["safety_min"],
                float(np.min(commands - lower)),
                float(np.min(upper - commands)),
                float(np.min(0.25 - rate)),
                float(np.min(0.25 + rate)),
            )
            saved_safety_ranks.append((max(0.0, -worst), float(np.sum(normalized)), index))
            for label, expected in full_keys.items():
                if key == expected:
                    check_full_candidate(
                        full[label], commands, rollout, metrics, compare_rollout, check_audit, compare=compare
                    )
                    checked_full.add(label)
            for anchor_index, anchor in enumerate(anchors):
                if anchor["response_index"] != index:
                    continue
                require(command_key(anchor["commands"], 12) == key, "Accepted anchor binds exact response bytes")
                check_audit(anchor["audit"], metrics)
                for field in ("tcp_errors", "normalized_squared_errors", "safety_min"):
                    compare("anchor_" + field, metrics[field], anchor[field])
                require(
                    anchor["joint_feasible"] is metrics["joint_feasible"]
                    and anchor["feasible"] is metrics["feasible"]
                    and type(anchor["prefix_length"]) is int
                    and anchor["prefix_length"] == metrics["prefix_length"],
                    "Independent strict accepted-anchor classification",
                )
                checked_anchors.add(anchor_index)
            result["verified_responses"] += 1
            require(
                result["numerical_inner_steps"] == 300 * (index + 1)
                and result["mj_forward_queries"] == 1 + 312 * (index + 1),
                "Every trace response independently resets and completes all 12 predicted periods",
            )
            if (index + 1) % 25 == 0 or index + 1 == n:
                print(json.dumps(dict(case_name=case, checked=index + 1, total=n)), flush=True)
        require(checked_full == set(full), "Every retained full candidate is present in the unique trace")
        require(
            checked_anchors == set(range(len(anchors))),
            "Every accepted anchor has an independently checked complete audit",
        )
        closest_index = min(range(n), key=lambda i: saved_safety_ranks[i])
        require(
            command_key(trace[closest_index]["commands"], 12) == full_keys["closest_safety_attempt"],
            "Exact saved minimum-violation then full-loss closest candidate ranking",
        )
        result["ledger_check"] = (
            audit_prefix_search_ledger(answer, classifications, compare=compare)
            if case == "baseline"
            else audit_trust_search_ledger(answer, classifications, compare=compare)
        )
        require(bool(compiled_goal.nopython_signatures), "Native quaternion goal actually compiled")
        result["native_quaternion_jit_signatures"] = [str(value) for value in compiled_goal.nopython_signatures]
        for ref in producer["source_refs_unchanged"]:
            remember(ref, refs)
        finish(refs)
        strict_native_environment()
        result.update(
            complete=True,
            passed=True,
            full_candidates_checked=sorted(checked_full),
            accepted_anchor_audits_checked=len(checked_anchors),
            closest_trace_index=closest_index,
            source_refs_unchanged=list(refs.values()),
            witness_found=bool(result["ledger_check"]["witness_indices"]),
            evidence="Independent local numerical and search-ledger qualification only; no physical control or task qualification",
        )
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["query_count"] = result["mj_forward_queries"]
        publish(output / "verified.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument(
        "--dry", action="store_true", help="Read source/closed evidence only; create no output or response"
    )
    arguments = parser.parse_args()
    main(arguments.case, arguments.dry)
