"""Independently reconstruct every prefix-preserving trajectory and stage ledger.

The closed previous H12 audit is a prerequisite, including its preserved earlier
aborted attempt. Native dynamics and every original constraint are recomputed
without importing the production predictor, optimizer or constraint audit.
"""

import json
import os
from pathlib import Path

import numpy as np
from check_tracking_v5_state_updated_response import native_functions
from scipy.spatial.transform import Rotation

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.preview_result_publication import publish_result as publish
from s2a.terminal_development_execution import CONTROL_REF

ROOT = REPO / "outputs/tracking_v5_prefix_preserving_trajectory_control_check_20261028"
PRODUCER = REPO / "outputs/tracking_v5_prefix_preserving_trajectory_control_20261028/verified.json"
NUMERIC_TOLERANCE = 1e-10
TORQUE_TOLERANCE = 1e-8
HORIZON = 12


def finite_max_abs_difference(actual, expected, *, name="array"):
    """Return max absolute error for finite, same-shaped arrays (empty gives0)."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(
        actual.shape == expected.shape and np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)),
        f"Finite matching shape: {name}",
    )
    return float(np.max(abs(actual - expected))) if actual.size else 0.0


def require_absolute_tolerance(difference, tolerance, *, name="array"):
    """Apply the original finite absolute comparison, with zero relative tolerance."""
    require(
        np.isfinite(difference) and np.isfinite(tolerance) and tolerance >= 0 and 0 <= difference <= tolerance,
        f"{name}: maximum absolute difference {difference!r} exceeds tolerance {tolerance!r}",
    )


def require_finite_array_equal(actual, expected, *, name="array"):
    """Require every finite same-shaped coordinate to be exactly unchanged."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(
        actual.shape == expected.shape
        and np.all(np.isfinite(actual))
        and np.all(np.isfinite(expected))
        and np.array_equal(actual, expected),
        f"Every quantity query preserves all finite {name} coordinates exactly",
    )


def audit_prefix_search_ledger(answer, classifications, *, compare=None):
    """Audit selection and stage accounting after independent numerical replay.

    ``classifications`` supplies each replayed trajectory's original eligibility,
    continuous prefix, normalized errors and nonlinear safety minimum. Exact
    ordering uses the checked archived error arithmetic, preserving first-query
    ties despite harmless sub-tolerance differences in independent dynamics.
    No production solver, predictor, model or constraint-audit code is imported.
    """
    if compare is None:

        def compare(name, actual, expected):
            difference = finite_max_abs_difference(actual, expected, name=name)
            require_absolute_tolerance(difference, NUMERIC_TOLERANCE, name=name)

    trace = answer["evaluated_candidates"]
    horizon = answer["horizon"]
    budget, iterations_budget = answer["max_evaluations"], answer["max_iterations"]
    require(type(horizon) is int and 1 <= horizon <= 16, "Explicit finite trajectory horizon")
    require(
        type(budget) is int and budget >= 1 and type(iterations_budget) is int and iterations_budget >= 1,
        "Explicit integral response and iteration budgets",
    )
    require(
        1 <= len(trace) == len(classifications) == answer["response_evaluations"] <= budget,
        "One independent classification per uniquely charged complete response",
    )
    normalized, prefixes, command_keys, witness_ranks, prefix_ranks = [], [], [], [], []
    for index, (saved, independent) in enumerate(zip(trace, classifications, strict=True)):
        commands = np.asarray(saved["commands"], dtype=float)
        errors = np.asarray(saved["tcp_errors"], dtype=float)
        require(
            commands.shape == (horizon, 6)
            and np.all(np.isfinite(commands))
            and errors.shape == (horizon, 6)
            and np.all(np.isfinite(errors)),
            "Finite full commands and TCP errors in every trace entry",
        )
        squared = np.stack(
            (np.sum(errors[:, :3] ** 2, axis=1) / 0.01**2, np.sum(errors[:, 3:] ** 2, axis=1) / 0.05**2), axis=1
        )
        require(np.all(np.isfinite(squared)), "Finite normalized pose arithmetic")
        failures = np.flatnonzero(np.any(squared > 1.0, axis=1))
        prefix = int(failures[0]) if len(failures) else horizon
        independent_squared = np.asarray(independent["normalized_squared_errors"])
        require(
            independent_squared.shape == squared.shape and np.all(np.isfinite(independent_squared)),
            "Finite independently reconstructed normalized errors",
        )
        independent_failed = np.flatnonzero(np.any(independent_squared > 1.0, axis=1))
        independent_prefix = int(independent_failed[0]) if len(independent_failed) else horizon
        compare("ledger_safety_minimum", independent["safety_min"], saved["safety_min"])
        require(
            type(saved["prefix_length"]) is int
            and saved["prefix_length"] == independent["prefix_length"] == independent_prefix == prefix
            and saved["joint_feasible"] is independent["joint_feasible"]
            and saved["feasible"] is independent["feasible"]
            and saved["feasible"] is bool(saved["joint_feasible"] and prefix == horizon),
            "Strict archived and independent prefix/eligibility classifications agree",
        )
        normalized.append(squared)
        prefixes.append(prefix)
        command_keys.append(commands.tobytes())
        loss = float(np.sum(squared))
        witness_ranks.append((float(np.max(squared)), loss, index))
        next_error = float(np.max(squared[prefix])) if prefix < horizon else 0.0
        prefix_ranks.append((-prefix, next_error, loss, index))
    require(len(set(command_keys)) == len(command_keys), "Command-byte cache charges each trajectory only once")

    def best(indices):
        require(bool(indices), "At least one already audited safe candidate is available")
        witnesses = [index for index in indices if trace[index]["feasible"]]
        return (
            min(witnesses, key=lambda index: witness_ranks[index])
            if witnesses
            else min(indices, key=lambda index: prefix_ranks[index])
        )

    def lookup(commands, end, *, name):
        values = np.asarray(commands, dtype=float)
        require(values.shape == (horizon, 6) and np.all(np.isfinite(values)), f"Finite {name} commands")
        key = values.tobytes()
        require(key in command_keys[:end], f"{name} binds an already completed cached response")
        return command_keys.index(key)

    require(lookup(answer["initial"]["commands"], 1, name="initial") == 0, "Initial entry is the first response")
    require_finite_array_equal(trace[0]["commands"], answer["used_start"], name="unique warm start")
    require(
        type(answer["initial_prefix_length"]) is int and answer["initial_prefix_length"] == prefixes[0],
        "Reported initial continuous prefix is exact",
    )
    require(trace[0]["joint_feasible"] is True, "Repair starts from an already fully safe warm")
    if not trace[0]["feasible"]:
        require(trace[0]["safety_min"] >= 1e-8, "Incomplete warm has the registered nonlinear optimization buffer")
    safe = [index for index, saved in enumerate(trace) if saved["joint_feasible"]]
    witnesses = [index for index in safe if trace[index]["feasible"]]
    selected = best(safe)
    require(answer["selected"] is not None, "The safe initial warm guarantees a retained safe selection")
    require(lookup(answer["selected"]["commands"], len(trace), name="selected") == selected, "Exact final ranking")
    require(
        type(answer["selected_prefix_length"]) is int
        and answer["selected_prefix_length"] == prefixes[selected] >= prefixes[0],
        "Final selection retains the entire initially qualified continuous prefix",
    )
    require(
        answer["witness_found"] is bool(witnesses)
        and answer["status"] == ("witness" if witnesses else "finite_search_without_witness"),
        "Only complete full-audit pose witnesses qualify, regardless of optimizer success",
    )

    phases, phase_checks = answer["phases"], []
    used, previous_end, expected_start = 0, 1, 0
    if trace[0]["feasible"]:
        require(not phases and len(trace) == 1, "A complete warm returns after one response and no optimization")
    else:
        require(bool(phases), "An incomplete safe warm starts exactly one repair chain")
    for phase_index, phase in enumerate(phases):
        before, after = phase["evaluations_before"], phase["evaluations_after"]
        require(
            type(before) is int
            and type(after) is int
            and 1 <= before <= after <= len(trace)
            and before == previous_end,
            "All phases consume one contiguous response ledger",
        )
        require(
            before < budget and used < iterations_budget and phase["max_iterations"] == iterations_budget - used,
            "Each stage receives only the remaining shared iteration budget",
        )
        require(
            phase["response_budget_remaining_before"] == budget - before
            and phase["response_budget_remaining_after"] == budget - after,
            "Phase response-budget accounting matches exact ledger boundaries",
        )
        initial_index = lookup(phase["initial_commands"], before, name="phase initial")
        require(
            initial_index == expected_start, "Phase starts from the preceding boundary's deterministic best candidate"
        )
        cursor = prefixes[initial_index]
        require(
            phase["phase"] == "prefix"
            and type(phase["prefix_before"]) is int
            and phase["prefix_before"] == phase["target_cursor"] == phase["cursor"] == cursor < horizon,
            "Only the next failing cursor receives an epigraph; no safety restoration or repeated stage",
        )
        compare("phase_initial_safety", classifications[initial_index]["safety_min"], phase["initial_safety_min"])
        initial_epigraph = max(0.0, float(np.max(normalized[initial_index][cursor])) - 1.0) + 1e-6
        compare("phase_initial_epigraph", initial_epigraph, phase["initial_epigraph"])
        callbacks, count = phase["callback_count"], phase["iterations"]
        require(
            type(callbacks) is int
            and type(count) is int
            and 0 <= callbacks <= count <= phase["max_iterations"]
            and type(phase["optimizer_returned"]) is bool
            and type(phase["success"]) is bool,
            "Finite stage callback/iteration accounting and explicit optimizer return status",
        )
        if phase["optimizer_returned"]:
            nit = phase["optimizer_iterations"]
            require(
                type(nit) is int and 0 <= nit <= phase["max_iterations"] and count == max(callbacks, nit),
                "Normal-return stage charges the larger of callbacks and optimizer nit",
            )
            final_commands = np.asarray(phase["final_commands"], dtype=float)
            require(
                final_commands.shape == (horizon, 6)
                and np.all(np.isfinite(final_commands))
                and np.isfinite(phase["final_epigraph"])
                and phase["final_epigraph"] >= 0,
                "Normal optimizer return retains finite full commands and nonnegative epigraph",
            )
            if phase["status"] != "response_budget":
                lookup(final_commands, after, name="optimizer final")
        else:
            nit = None
            require(
                count == callbacks and "optimizer_iterations" not in phase, "Interrupted stage charges its callbacks"
            )
        if callbacks:
            lookup(phase["last_callback_commands"], after, name="last callback")
            require(
                np.isfinite(phase["last_callback_epigraph"]) and phase["last_callback_epigraph"] >= 0,
                "Last completed callback retains its finite nonnegative epigraph",
            )
        else:
            require(
                "last_callback_commands" not in phase and "last_callback_epigraph" not in phase,
                "No callback means no invented callback snapshot",
            )
        eligible = [index for index in range(after) if trace[index]["joint_feasible"] and prefixes[index] >= cursor]
        end_index = best(eligible)
        require(
            lookup(phase["selected_commands"], after, name="phase selected") == end_index
            and phase["prefix_after"] == prefixes[end_index],
            "Stage boundary chooses the longest safe prefix with the registered precise tie rules",
        )
        if phase["status"] == "prefix_advanced":
            require(
                phase["optimizer_returned"] is False
                and phase["success"] is False
                and callbacks >= 1
                and prefixes[end_index] > cursor,
                "Prefix interruption occurs only after a counted callback and an actually longer safe prefix",
            )
        elif phase["status"] == "response_budget":
            require(
                after == budget and phase["success"] is False and phase_index == len(phases) - 1,
                "A response-budget interruption retains all completed probes and cannot start another phase",
            )
        else:
            require(
                type(phase["status"]) is int
                and phase["optimizer_returned"] is True
                and phase["success"] is (phase["status"] == 0),
                "Normal optimizer status is recorded consistently but never substitutes for a witness",
            )
        used += count
        previous_end, expected_start = after, end_index
        phase_checks.append(
            dict(
                index=phase_index,
                initial_trace_index=initial_index,
                selected_trace_index=end_index,
                prefix_before=cursor,
                prefix_after=prefixes[end_index],
                evaluations_before=before,
                evaluations_after=after,
                callback_count=callbacks,
                optimizer_iterations=nit,
                iterations=count,
                status=phase["status"],
            )
        )
        if phase_index + 1 < len(phases):
            require(
                cursor < prefixes[end_index] < horizon
                and after < budget
                and used < iterations_budget
                and (phase["status"] == "prefix_advanced" or phase["optimizer_returned"] is True),
                "Only strict progress at an optimizer boundary can begin the next stage",
            )
    require(previous_end == len(trace), "Every response belongs to initial warm or exactly one declared stage")
    require(used == answer["total_iterations"] <= iterations_budget, "Shared iteration total is exact")
    if witnesses:
        termination = "witness"
    elif len(trace) == budget:
        termination = "response_budget"
    elif phases[-1]["prefix_after"] <= phases[-1]["prefix_before"]:
        termination = "no_prefix_progress"
    else:
        require(used >= iterations_budget, "A growing unfinished prefix with remaining budget must continue")
        termination = "iteration_budget"
    require(answer["termination_reason"] == termination, "Termination reason follows actual prefix and budget evidence")
    return dict(
        selected_trace_index=selected,
        joint_feasible_indices=safe,
        witness_indices=witnesses,
        initial_prefix_length=prefixes[0],
        selected_prefix_length=prefixes[selected],
        total_iterations=used,
        phase_checks=phase_checks,
        termination_reason=termination,
    )


def main():
    strict_native_environment()
    require(not ROOT.exists(), "No overwrite or automatic retry")
    references = {}

    def remember(ref):
        checked(ref)
        require(ref["path"] not in references or references[ref["path"]] == ref, "Unique source fingerprint")
        references[ref["path"]] = ref
        return ref

    def load(ref):
        return read(remember(ref))

    def closed_owner(ref, expected_result=None):
        owner = load(ref)
        require(owner["complete"] is owner["passed"] is True, "Closed prerequisite owner")
        launch = load(owner["owned_command"]["launch"])
        exited = load(owner["owned_command"]["exit"])
        released = load(owner["owned_command"]["release"])
        require(
            exited["actual_wait"] is True
            and exited["returncode"] == 0
            and released["reaped"] is True
            and released["returncode"] == 0
            and released["signal_sent"] is False
            and released["original_error"] is None,
            "Prerequisite actual wait0/reap0 without signal or original error",
        )
        require(
            released["pid"] == launch["child"]["pid"] and released["start_ticks"] == launch["child"]["start_ticks"],
            "Same owned process birth through release",
        )
        require(
            launch["selected_environment"]["CUDA_VISIBLE_DEVICES"] == ""
            and launch["selected_environment"]["LD_LIBRARY_PATH"] == ""
            and launch["automatic_retry"] is False
            and exited["automatic_retry"] is False,
            "CPU-only prerequisite without automatic retry",
        )
        if expected_result is not None:
            require(owner["result_ref"] == expected_result, "Owner binds exact registered result")
        value = load(owner["result_ref"])
        require(value["complete"] is value["passed"] is True, "Prerequisite result passed")
        return owner, value

    producer_ref = remember(reference(PRODUCER))
    producer_owner_ref = remember(
        reference(REPO / "outputs/tracking_v5_prefix_preserving_trajectory_control_execution_20261028/result.json")
    )
    producer_owner, producer = closed_owner(producer_owner_ref, producer_ref)
    owner_reg = load(producer_owner["registration_ref"])
    for path, sha256 in owner_reg["source_sha256"].items():
        remember(dict(path=path, sha256=sha256))
    producer_launch = load(producer_owner["owned_command"]["launch"])
    require(
        producer_launch["command"][-1]
        == str(REPO / "scripts/probe_tracking_v5_prefix_preserving_trajectory_control.py"),
        "Owner actually ran the new trajectory producer",
    )
    require(
        producer["physics_steps"] == producer["world_model_calls"] == producer["fresh_cases"] == 0
        and producer["control_qualified"] is False,
        "Numerical search only, no action dispatch or physical qualification",
    )
    reg = load(producer["registration_ref"])
    require(
        reg["seed"] == 42
        and reg["endpoint"] == 368
        and reg["cursor"] == 0
        and reg["horizon"] == HORIZON
        and reg["target_cursors"] == list(range(HORIZON))
        and reg["initial_history_completed_periods"] == 0
        and reg["automatic_retry"] is False,
        "Single registered actual368 H12 roster and empty causal history",
    )
    require(
        reg["search"]["total_iterations"] == 60
        and reg["search"]["expected_initial_prefix"] == 6
        and reg["search"]["response_budget"] == 4000
        and reg["search"]["difference_step"] == 1e-4
        and reg["search"]["optimization_safety_buffer"] == 1e-8
        and reg["search"]["ftol"] == 1e-12
        and reg["search"]["starts"] == 1,
        "Fixed single-start trajectory search budget and difference rule",
    )
    require(
        reg["search"]["safety_policy"]
        == "No restoration phase; repair requires full-safe warm with minimum nonlinear slack>=1e-8"
        and reg["search"]["prefix_policy"]
        == "Original caps only, all72 commands remain free; no locked commands or old residual values"
        and reg["search"]["selection_policy"]
        == "Longest continuous prefix, next failure max normalized square, full squared sum, first evaluation"
        and reg["search"]["witness_selection_policy"]
        == "Global max normalized square, full squared sum, first evaluation"
        and reg["search"]["phase_policy"]
        == "Advance at callback or normal optimizer-return boundary only after full audit and strictly longer continuous prefix",
        "Original safe warm, free commands and deterministic prefix/witness selection",
    )
    for name, value in dict(
        rate=0.25, joint_margin=0.05, braking_time=0.1, position_cap=0.01, rotation_cap=0.05, pose_caps_horizon=HORIZON
    ).items():
        require(reg["constraints"][name] == value, "Original trajectory conditions")
    normals = np.eye(7)[6:]
    np.testing.assert_array_equal(reg["constraints"]["terminal_normals"], normals)
    for ref in reg["source_refs"] + reg["native_source_refs"]:
        remember(ref)
    _, stop_check = closed_owner(reg["prior_stop_check_owner_ref"])
    stop_check_reg = load(stop_check["registration_ref"])
    require(stop_check_reg["producer_ref"] == reg["prior_stop_ref"], "Same closed stopped-state attempt")
    _, stop = closed_owner(stop_check_reg["producer_owner_ref"], reg["prior_stop_ref"])
    require(
        stop["answer"]["witness_found"] is False and stop_check["witness_found"] is False,
        "Transition after the registered failed stopped-state search",
    )
    stop_reg = load(stop["registration_ref"])
    early = load(reg["early_numeric_ref"])
    require(early["complete"] is early["passed"] is True, "Completed prior early-state warm source")
    early_reg = load(early["registration_ref"])
    _, early_check = closed_owner(reg["early_independent_owner_ref"])
    require(
        load(early_check["registration_ref"])["producer_ref"] == reg["early_numeric_ref"],
        "Independent checker binds exactly the earlier H12 warm",
    )
    require(reg["original_producer_ref"] == early_reg["producer_ref"], "Original physical branch source")
    original = load(reg["original_producer_ref"])
    require(original["complete"] is original["passed"] is True, "Original actual branch complete")
    original_reg = load(original["registration_ref"])
    require(
        reg["physical_independent_owner_ref"] == early_reg["independent_owner_ref"],
        "Same independent original physical replay owner",
    )
    _, physical_check = closed_owner(reg["physical_independent_owner_ref"])
    require(
        load(physical_check["registration_ref"])["producer_ref"] == reg["original_producer_ref"],
        "Exact original physical replay",
    )
    branches = [b for b in original["branches"] if b["branch"] == "causal_error"]
    require(len(branches) == 1, "Exactly one original causal_error branch")
    require(
        reg["states_ref"] == early_reg["states_ref"] == branches[0]["states_ref"],
        "State368 from the independently audited causal_error branch",
    )
    for name in ("xml_ref", "control_ref", "tool_ref", "chunks_ref"):
        require(reg[name] == early_reg[name], "Unchanged early-state geometry/control/available reference")
    require(
        reg["manifest_ref"] == original_reg["manifest_ref"]
        and reg["original_rows_ref"] == original_reg["rows_ref"]
        and reg["initial_ref"] == original_reg["braking_numeric_ref"],
        "Original controller history, execution and nullspace target sources",
    )
    require(reg["chunks_ref"] == original_reg["chunks_ref"], "Reference generated by this same original run")
    require(
        reg["native_source_refs"] == stop_reg["native_source_refs"]
        and reg["native_transform_ref"] == stop_reg["native_transform_ref"],
        "Same pinned native controller and JIT quaternion sources",
    )
    entries = [e for e in early["entries"] if e["step"] == 368 and e["horizon"] == HORIZON]
    require(len(entries) == 1, "Unique prior H12 actual368 warm trajectory")
    entry = entries[0]
    require(entry["cursor"] == 0 and entry["target_cursors"] == list(range(HORIZON)), "Same warm target roster")
    warm = np.asarray(entry["answer"]["planned_commands"], dtype=float)
    require(warm.shape == (HORIZON, 6), "Exact prior H12 planned_commands, no other start")
    np.testing.assert_array_equal(entry["answer"]["terminal_normals"], normals)
    profile = np.zeros((25, 7))
    np.testing.assert_array_equal(entry["causal_history"], profile)
    with np.load(checked(remember(reg["states_ref"])), allow_pickle=False) as archive:
        state = archive["sim_states"][368].copy()
    with checked(remember(reg["original_rows_ref"])).open() as stream:
        for index, line in enumerate(stream):
            if index == 367:
                previous_row = json.loads(line)
                break
        else:
            raise ValueError("Missing completed actual367 controller state")
    previous = np.asarray(previous_row["command"], dtype=float)
    np.testing.assert_array_equal(previous, entry["previous_command"])
    previous_site = np.asarray(previous_row["qualified_forward_input"]["x_site_pose"])[0]
    latch = np.asarray(previous_row["controller_goal_after"])
    require(
        previous.shape == (7,) and previous_site.shape == (4, 4) and latch.shape == (3, 3),
        "Actual previous command, SITE and native orientation goal",
    )
    chunk = load(reg["chunks_ref"])[46]
    require(chunk["step"] == 368, "Chunk46 is already generated at current368")
    desired = np.asarray(chunk["effective_targets_tcp"])[0:HORIZON].copy()
    require(desired.shape == (HORIZON, 4, 4), "All twelve available desired TCP poses")
    control_record, tool = load(reg["control_ref"]), load(reg["tool_ref"])
    control = control_record["contract"]
    require(control["control_freq"] == 20 and control["model_timestep"] == 0.002, "Original control timing")
    site_from_tcp = np.linalg.solve(np.asarray(control_record["source_from_site"]), np.asarray(tool["source_from_tcp"]))
    initial = np.asarray(load(reg["initial_ref"])["initial_joint"])
    kp, kd = np.asarray(control["kp"]), np.asarray(control["kd"])
    input_min, input_max = np.asarray(control["input_min"]), np.asarray(control["input_max"])
    output_min, output_max = np.asarray(control["output_min"]), np.asarray(control["output_max"])
    previous_delta = output_min + (np.clip(previous[:6], input_min, input_max) - input_min) * (
        (output_max - output_min) / (input_max - input_min)
    )
    require(np.any(previous_delta[3:] != 0), "Nonzero367 reset makes historical latch independently identifiable")
    manifest = load(reg["manifest_ref"])
    control_refs = [r for r in reg["native_source_refs"] if r["path"].endswith("/utils/control_utils.py")]
    require(len(control_refs) == 1, "Single original native control utility source")
    functions, control_locations = native_functions(
        control_refs[0], ("orientation_error", "opspace_matrices", "nullspace_torques")
    )
    transform_ref = remember(reg["native_transform_ref"])
    require(transform_ref == stop_check_reg["native_transform_source_ref"], "Previously checked native JIT source")
    transformations, transform_locations = native_functions(
        transform_ref, ("axisangle2quat", "quat2mat"), include_eps=True
    )
    for ref in [*reg["native_source_refs"], transform_ref]:
        require(manifest["execution_sha256"][ref["path"]] == ref["sha256"], "Actual original native execution source")
    helper_ref = remember(reference(REPO / "scripts/check_tracking_v5_state_updated_response.py"))
    require(helper_ref == stop_check_reg["ast_helper_ref"], "Unchanged independently executed AST helper")
    checker_ref = remember(reference(Path(__file__)))
    answer, trace = producer["answer"], producer["answer"]["evaluated_candidates"]
    require(1 <= len(trace) == answer["response_evaluations"] <= 4000, "All evaluated trajectories, within budget")
    prior_check_owner_ref = remember(reg["prior_trajectory_check_owner_ref"])
    require(
        prior_check_owner_ref
        == reference(
            REPO / "outputs/tracking_v5_state_updated_trajectory_control_v2_check_execution_20261028/result.json"
        ),
        "Exactly the closed complete v2 trajectory audit is the prerequisite",
    )
    prior_check_owner, prior_check = closed_owner(prior_check_owner_ref, reg["prior_trajectory_check_ref"])
    prior_check_reg = load(prior_check["registration_ref"])
    prior_owner_reg = load(prior_check_owner["registration_ref"])
    for path, sha256 in prior_owner_reg["source_sha256"].items():
        remember(dict(path=path, sha256=sha256))
    prior_ref = remember(reg["prior_trajectory_ref"])
    require(
        prior_ref == reference(REPO / "outputs/tracking_v5_state_updated_trajectory_control_20261028/verified.json")
        and prior_check_reg["producer_ref"] == prior_ref
        and prior_check["trace_count"] == 4000
        and prior_check["numerical_inner_steps"] == 1200000
        and prior_check["mj_forward_queries"] == 1248001
        and prior_check["witness_found"] is False
        and prior_check["control_qualified"] is False,
        "Original4000 trajectories and1200000 numerical steps independently closed without a witness",
    )
    for ref in prior_check["source_refs_unchanged"]:
        remember(ref)
    _, prior = closed_owner(prior_check_reg["producer_owner_ref"], prior_ref)
    prior_reg = load(prior["registration_ref"])
    require(prior["answer"]["witness_found"] is False, "The previous finite H12 failure remains preserved")
    for name in (
        "early_numeric_ref",
        "early_independent_owner_ref",
        "physical_independent_owner_ref",
        "original_producer_ref",
        "states_ref",
        "original_rows_ref",
        "xml_ref",
        "control_ref",
        "tool_ref",
        "chunks_ref",
        "native_source_refs",
        "native_transform_ref",
        "initial_ref",
    ):
        require(reg[name] == prior_reg[name], "Unchanged original warm, current state and native dynamics sources")
    require(
        owner_reg["prior_full_trajectory_audit_owner_ref"] == prior_check_owner_ref,
        "New owner admission binds the same closed complete original audit",
    )
    # Preserve the old unsuccessful500-trajectory audit through the already
    # completed v2 audit's exact evidence, without treating it as a new run.
    abort_fields = (
        "prior_aborted_owner_ref",
        "prior_aborted_check_registration_ref",
        "prior_aborted_source_ref",
        "prior_aborted_launch_ref",
        "prior_aborted_exit_ref",
        "prior_aborted_release_ref",
        "prior_aborted_command_log_ref",
        "explicit_stop_decision_ref",
    )
    preserved_abort_refs = {name: remember(prior_check_reg[name]) for name in abort_fields}
    aborted_owner = load(preserved_abort_refs["prior_aborted_owner_ref"])
    aborted_reg = load(preserved_abort_refs["prior_aborted_check_registration_ref"])
    aborted_exit = load(preserved_abort_refs["prior_aborted_exit_ref"])
    aborted_release = load(preserved_abort_refs["prior_aborted_release_ref"])
    aborted_decision = load(preserved_abort_refs["explicit_stop_decision_ref"])
    require(
        aborted_owner["complete"] is False
        and aborted_owner["passed"] is False
        and aborted_reg["producer_ref"] == prior_ref
        and aborted_exit["actual_wait"] is True
        and aborted_exit["returncode"] == -15
        and aborted_release["reaped"] is True
        and aborted_release["returncode"] == -15
        and aborted_decision["last_progress"]
        == dict(checked_trajectories=500, total_trajectories=4000, numerical_inner_steps=150000),
        "Preserved old abort remains explicitly incomplete and actually terminated/reaped",
    )
    fixture_migrations = prior_owner_reg["explicit_test_source_migrations"]
    expected_paths = {
        str(REPO / "tests" / name)
        for name in ("test_native_training_continuation_preparation.py", "test_native_training_runner.py")
    }
    require(
        len(fixture_migrations) == 2 and {item["path"] for item in fixture_migrations} == expected_paths,
        "Only the two previously independently audited fixture migrations are allowed",
    )
    allowed_migrations = {}
    for migration in fixture_migrations:
        require(
            migration["before_commit"] == "9db6efa"
            and migration["after_commit"] == "2f09739"
            and migration["path"] == migration["after_ref"]["path"],
            "Exact preserved fixture repair identities",
        )
        remember(migration["after_ref"])
        allowed_migrations[migration["path"]] = migration
    used_migrations = reg["explicit_historical_source_migrations"]
    require(
        len(used_migrations) == 2 and {item["path"] for item in used_migrations} == expected_paths,
        "Producer records exactly the two actual historical physical-owner fixture migrations",
    )
    physical_owner = load(reg["physical_independent_owner_ref"])
    physical_sources = load(physical_owner["registration_ref"])["source_sha256"]
    for migration in used_migrations:
        require(
            migration == dict(owner_ref=reg["physical_independent_owner_ref"], **allowed_migrations[migration["path"]])
            and physical_sources[migration["path"]] == migration["before_sha256"],
            "Each used migration binds the original physical owner's exact historical source digest",
        )
    for path, digest in physical_sources.items():
        if path not in expected_paths:
            remember(dict(path=path, sha256=digest))
    version_plan_ref = remember(reference(REPO / "plans/v0.0.1/prefix_preserving_trajectory_control_20261028.md"))
    require(producer["numerical_inner_steps"] == len(trace) * HORIZON * 25, "Producer trajectory/inner-step accounting")
    registration = publish(
        ROOT / "registration.json",
        dict(
            producer_ref=producer_ref,
            producer_registration_ref=producer["registration_ref"],
            producer_owner_ref=producer_owner_ref,
            prior_stop_check_owner_ref=reg["prior_stop_check_owner_ref"],
            early_independent_owner_ref=reg["early_independent_owner_ref"],
            physical_independent_owner_ref=reg["physical_independent_owner_ref"],
            source_ref=checker_ref,
            ast_helper_ref=helper_ref,
            native_control_source_ref=control_refs[0],
            native_transform_source_ref=transform_ref,
            native_control_function_lines=control_locations,
            native_transform_function_lines=transform_locations,
            seed=42,
            endpoint=368,
            cursor=0,
            horizon=HORIZON,
            target_cursors=list(range(HORIZON)),
            expected_trace_count=len(trace),
            prior_full_trajectory_check_owner_ref=prior_check_owner_ref,
            prior_full_trajectory_check_ref=reg["prior_trajectory_check_ref"],
            prior_full_trajectory_ref=prior_ref,
            preserved_prior_abort_refs=preserved_abort_refs,
            explicit_historical_source_migrations=used_migrations,
            version_plan_ref=version_plan_ref,
            comparison_policy="Same finite/shape/absolute1e-10 or torque1e-8 checks; compare max absolute difference directly, no success-path testing assertion overhead",
            query_integrity_policy="Every query still checks all scalar qpos/qvel exactly and every normalized quaternion within the original tolerance",
            initial_history_completed_periods=0,
            numeric_tolerance=NUMERIC_TOLERANCE,
            torque_tolerance=TORQUE_TOLERANCE,
            dynamics="Independent resetData/current fixed fingers/predicted arm q-v/mj_forward; no mj_step",
            recurrence="Independent native AST OSC and implicit-damping Euler, carry predicted q/v and JIT goal through12x25 steps",
            independence="No import of new trajectory, response, goal, query, audit, history or optimizer implementations",
            history="Only current368 state and completed367 goal; zero own past-error profile, no future observations or labels",
            trace="Recompute every trajectory including difference probes; box checked, rate violations explicitly classified",
            constraints="All12 TCP caps, all300 inner margins, all12 braking endpoints, final +e7 terminal; original tolerances",
            repair="Preserve all qualified continuous-prefix caps; only next point receives the stage epigraph",
            stage_accounting="One shared response ledger and60 iterations; callback count versus normal-return nit, deterministic cached prefix advancement",
            qualification="Numerical reconstruction only; passed does not authorize physical action or certify control/task/fresh",
            automatic_retry=False,
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        registration_ref=registration,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        control_qualified=False,
        trace_checks=[],
        boundary_classification_disagreements=[],
        numerical_inner_steps=0,
        mj_forward_queries=0,
        max_abs_difference={},
        max_jit_vs_python_goal_difference=0.0,
        jit_python_goal_differences=0,
    )

    def compare(name, actual, expected, tolerance=NUMERIC_TOLERANCE):
        difference = finite_max_abs_difference(actual, expected, name=name)
        result["max_abs_difference"][name] = max(result["max_abs_difference"].get(name, 0.0), difference)
        require_absolute_tolerance(difference, tolerance, name=name)
        return difference

    try:
        guard = load(CONTROL_REF)["cv2_guard_registration"]
        result["environment_admission"] = restore_cv2_environment(
            guard["exact_probe_ref"], guard["runtime_source_refs"]
        )
        for ref in [guard["exact_probe_ref"], *guard["runtime_source_refs"]]:
            remember(ref)
        os.environ.pop("MUJOCO_GL", None)
        os.environ.pop("PYOPENGL_PLATFORM", None)
        import mujoco
        import numba

        quat2mat_jit = numba.njit(cache=False)(transformations["quat2mat"])
        require(hasattr(quat2mat_jit, "nopython_signatures"), "Native quaternion path must enable actual JIT")
        model = mujoco.MjModel.from_xml_path(str(checked(remember(reg["xml_ref"]))))
        data = mujoco.MjData(model)
        require(
            model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER and model.opt.timestep == 0.002,
            "Native Euler2ms model",
        )
        result["native_runtime"] = dict(mujoco=mujoco.__version__, numpy=np.__version__, numba=numba.__version__)
        require(result["native_runtime"] == stop_check["native_runtime"], "Same verified native numerical runtime")
        joints = np.asarray(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{k}") for k in range(1, 8)]
        )
        require(
            np.all(joints >= 0) and np.all(model.jnt_type[joints] == mujoco.mjtJoint.mjJNT_HINGE), "Seven arm joints"
        )
        qids, dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, control["controller_site"])
        require(sid >= 0, "Native control SITE")
        fingers = {}
        for k in (1, 2):
            name = f"gripper0_finger_joint{k}"
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            require(jid >= 0 and model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE, "Two scalar finger coordinates")
            fingers[name] = (int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid]))
        aids = []
        for joint in joints:
            matched = np.flatnonzero((model.actuator_trnid[:, 0] == joint) & (model.actuator_trntype == 0))
            require(len(matched) == 1, "One direct arm motor")
            aids.append(matched[0])
        aids = np.asarray(aids)
        np.testing.assert_array_equal(model.actuator_gear[aids], np.tile([1, 0, 0, 0, 0, 0], (7, 1)))
        require(
            np.all(model.actuator_gaintype[aids] == 0)
            and np.all(model.actuator_gainprm[aids, 0] == 1)
            and np.all(model.actuator_biastype[aids] == 0)
            and np.all(model.actuator_dyntype[aids] == 0)
            and np.all(model.actuator_ctrllimited[aids])
            and not np.any(model.actuator_forcelimited[aids]),
            "Native unit torque actuators",
        )
        torque_limits, limits = model.actuator_ctrlrange[aids].copy(), model.jnt_range[joints].copy()
        lower, upper = limits[:, 0], limits[:, 1]
        quaternion_blocks = []
        for joint in range(model.njnt):
            kind, address = int(model.jnt_type[joint]), int(model.jnt_qposadr[joint])
            if kind == int(mujoco.mjtJoint.mjJNT_FREE):
                quaternion_blocks.append(np.arange(address + 3, address + 7))
            elif kind == int(mujoco.mjtJoint.mjJNT_BALL):
                quaternion_blocks.append(np.arange(address, address + 4))
        quaternion_ids = np.concatenate(quaternion_blocks) if quaternion_blocks else np.empty(0, dtype=int)
        scalar_ids = np.setdiff1d(np.arange(model.nq), quaternion_ids)

        def query(q, velocity, fixed):
            mujoco.mj_resetData(model, data)
            for name, (qi, vi) in fingers.items():
                data.qpos[qi], data.qvel[vi] = fixed[name]
            data.qpos[qids], data.qvel[dofs] = q, velocity
            before_q, before_v = data.qpos.copy(), data.qvel.copy()
            mujoco.mj_forward(model, data)
            result["mj_forward_queries"] += 1
            require_finite_array_equal(data.qpos[scalar_ids], before_q[scalar_ids], name="scalar qpos")
            for indices in quaternion_blocks:
                compare(
                    "quaternion_normalization",
                    data.qpos[indices],
                    before_q[indices] / np.linalg.norm(before_q[indices]),
                )
            require_finite_array_equal(data.qvel, before_v, name="qvel")
            require(data.time == 0, "Quantity query never integrates time")
            mass = np.zeros((model.nv, model.nv))
            mujoco.mj_fullM(model, mass, data.qM)
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jp, jr, sid)
            pose = np.eye(4)
            pose[:3, 3], pose[:3, :3] = data.site_xpos[sid], data.site_xmat[sid].reshape(3, 3)
            damping = model.dof_damping[dofs].copy()
            return dict(
                mass=mass[np.ix_(dofs, dofs)],
                jacobian=np.vstack((jp, jr))[:, dofs],
                pose=pose,
                bias=data.qfrc_bias[dofs].copy(),
                damping=damping,
                passive_extra=data.qfrc_passive[dofs].copy() + damping * velocity,
            )

        def reconstruct(q0, v0, fixed, command, latched_rotation_goal=None):
            command = np.asarray(command, dtype=float)
            require(command.shape == (6,) and np.all(np.isfinite(command)), "Finite six-dimensional arm command")
            delta = output_min + (np.clip(command, input_min, input_max) - input_min) * (
                (output_max - output_min) / (input_max - input_min)
            )
            start = query(q0, v0, fixed)
            goal = start["pose"].copy()
            goal[:3, 3] += delta[:3]
            difference = 0.0
            if np.any(delta[3:] != 0):
                quaternion = transformations["axisangle2quat"](delta[3:])
                increment = quat2mat_jit(quaternion)
                python_increment = transformations["quat2mat"](quaternion)
                difference = float(np.max(abs(increment - python_increment)))
                result["max_jit_vs_python_goal_difference"] = max(
                    result["max_jit_vs_python_goal_difference"], difference
                )
                result["jit_python_goal_differences"] += int(difference > 0)
                goal[:3, :3] = increment @ start["pose"][:3, :3]
            else:
                latch = np.asarray(latched_rotation_goal)
                require(
                    latch.shape == (3, 3) and np.all(np.isfinite(latch)),
                    "Exact-zero command needs known historical orientation goal",
                )
                goal[:3, :3] = latch
            q, velocity = q0.copy(), v0.copy()
            records = []
            for inner in range(25):
                current = start if inner == 0 else query(q, velocity, fixed)
                mass, jacobian = current["mass"], current["jacobian"]
                _, lambda_pos, lambda_ori, null = functions["opspace_matrices"](
                    mass, jacobian, jacobian[:3], jacobian[3:]
                )
                rotation_error = functions["orientation_error"](goal[:3, :3], current["pose"][:3, :3])
                force = kp[:3] * (goal[:3, 3] - current["pose"][:3, 3]) - kd[:3] * (jacobian[:3] @ velocity)
                moment = kp[3:] * rotation_error - kd[3:] * (jacobian[3:] @ velocity)
                requested = jacobian.T @ np.r_[lambda_pos @ force, lambda_ori @ moment]
                requested += functions["nullspace_torques"](mass, null, initial, q, velocity, joint_kp=10)
                requested += current["bias"]
                applied = np.minimum(torque_limits[:, 1], np.maximum(torque_limits[:, 0], requested))
                velocity = np.linalg.solve(
                    mass + np.diag(0.002 * current["damping"]),
                    mass @ velocity + 0.002 * (applied - current["bias"] + current["passive_extra"]),
                )
                q = q + 0.002 * velocity
                records.append(
                    dict(
                        joint_position=q.copy(),
                        joint_velocity=velocity.copy(),
                        requested_torque=requested,
                        applied_torque=applied,
                        clipped=requested != applied,
                        mass_change=float(np.max(abs(mass - start["mass"]))),
                        jacobian_change=float(np.max(abs(jacobian - start["jacobian"]))),
                    )
                )
            endpoint = query(q, velocity, fixed)
            result["numerical_inner_steps"] += 25
            return (
                dict(
                    **{key: np.asarray([record[key] for record in records]) for key in records[0]},
                    goal=goal,
                    endpoint_pose=endpoint["pose"],
                    endpoint_twist=endpoint["jacobian"] @ velocity,
                    update_matrices=True,
                    dynamics_queries=26,
                ),
                start,
                difference,
            )

        def compare_response(actual, archived):
            require(set(actual) == set(archived), "Complete native-goal response fields")
            require(
                archived["update_matrices"] is True and archived["dynamics_queries"] == 26, "25x2ms updated response"
            )
            for key, value in actual.items():
                if key in ("update_matrices", "dynamics_queries"):
                    continue
                if key == "clipped":
                    np.testing.assert_array_equal(value, archived[key])
                else:
                    compare(
                        "response_" + key,
                        value,
                        archived[key],
                        TORQUE_TOLERANCE if "torque" in key else NUMERIC_TOLERANCE,
                    )

        require(state.shape == (1 + model.nq + model.nv,), "Complete actual368 state vector")
        q0, v0 = state[1 + qids].copy(), state[1 + model.nq + dofs].copy()
        fixed = {name: [state[1 + qi], state[1 + model.nq + vi]] for name, (qi, vi) in fingers.items()}
        np.testing.assert_array_equal(q0, entry["current_q"])
        np.testing.assert_array_equal(v0, entry["current_velocity"])
        require(set(fixed) == set(producer["fixed_gripper_state"]), "Exactly two current fixed finger states")
        for name in fixed:
            np.testing.assert_array_equal(fixed[name], producer["fixed_gripper_state"][name])
        current = query(q0, v0, fixed)
        compare("earlier_current_site", current["pose"], entry["current_site"])
        require(set(current) == set(producer["current_quantities"]), "All current robot quantities")
        for name, value in current.items():
            compare("current_" + name, value, producer["current_quantities"][name])
        for name, value in dict(
            current_q=q0,
            current_velocity=v0,
            initial_joint=initial,
            kp=kp,
            kd=kd,
            torque_limits=torque_limits,
            joint_limits=limits,
            site_from_tcp=site_from_tcp,
            desired_tcp=desired,
            previous_command=previous,
            previous_site=previous_site,
            previous_physical_delta=previous_delta,
            latched_rotation_goal=latch,
            warm=warm,
            past_inner_error_rad=profile,
            terminal_normals=normals,
        ).items():
            compare("input_" + name, value, producer[name])
        prior_goal = quat2mat_jit(transformations["axisangle2quat"](previous_delta[3:])) @ previous_site[:3, :3]
        compare("actual367_latched_goal", prior_goal, latch)
        history = producer["history_before_search"]
        require(
            history["completed_periods"] == 0
            and history["last_completed_step"] is None
            and history["future_guarantee"] is False,
            "No native-goal response observations exist before this branch starts at368",
        )
        np.testing.assert_array_equal(history["inner_position_error_rad"], profile)
        np.testing.assert_array_equal(answer["past_inner_error_rad"], profile)
        np.testing.assert_array_equal(answer["terminal_normals"], normals)
        require(
            answer["horizon"] == answer["pose_caps_horizon"] == HORIZON and answer["all_pose_caps_enforced"] is True,
            "All twelve TCP caps and original final +e7 terminal remain witness requirements",
        )
        require(
            answer["pose_cap_enforcement_scope"]
            == "All H caps are required for a witness; stages hard-constrain only the attained prefix",
            "Whole-window qualification is distinct from each stage's hard prefix and relaxed target point",
        )
        lo, hi = np.tile(input_min, (HORIZON, 1)), np.tile(input_max, (HORIZON, 1))
        lo[0], hi[0] = np.maximum(lo[0], previous[:6] - 0.25), np.minimum(hi[0], previous[:6] + 0.25)
        require(np.all(hi - lo > 1e-4), "Every normalized-command difference probe has room inside box")
        require(
            np.all(warm >= lo - 1e-12)
            and np.all(warm <= hi + 1e-12)
            and np.all(abs(np.diff(np.vstack((previous[:6], warm)), axis=0)) <= 0.25 + 1e-12),
            "Registered warm satisfies original box/rate before any search",
        )
        start = np.clip(warm, lo, hi)
        np.testing.assert_array_equal(answer["supplied_warm"], warm)
        np.testing.assert_array_equal(answer["used_start"], start)
        np.testing.assert_array_equal(answer["start_roundoff_clip"], start - warm)
        require(np.max(abs(start - warm)) <= 1e-12, "Only admitted warm bound roundoff clipped")
        compare("command_lower", lo, answer["lower_command"])
        compare("command_upper", hi, answer["upper_command"])

        def reconstruct_trajectory(commands):
            q, velocity, inherited_goal = q0.copy(), v0.copy(), latch.copy()
            periods, differences = [], []
            for command in commands:
                response, _, difference = reconstruct(q, velocity, fixed, command, inherited_goal)
                periods.append(response)
                differences.append(difference)
                q = response["joint_position"][-1].copy()
                velocity = response["joint_velocity"][-1].copy()
                inherited_goal = response["goal"][:3, :3].copy()
            require(len(periods) == HORIZON, "Exactly twelve autonomous predicted periods")
            return dict(
                inner_joint_position=np.concatenate([v["joint_position"] for v in periods]),
                inner_joint_velocity=np.concatenate([v["joint_velocity"] for v in periods]),
                joint_position=np.asarray([v["joint_position"][-1] for v in periods]),
                joint_velocity=np.asarray([v["joint_velocity"][-1] for v in periods]),
                pose=np.asarray([v["endpoint_pose"] for v in periods]),
                twist=np.asarray([v["endpoint_twist"] for v in periods]),
                goals=np.asarray([v["goal"] for v in periods]),
                periods=periods,
            ), differences

        def compare_rollout(actual, archived):
            require(set(actual) == set(archived), "Every aggregated trajectory field is present")
            for name in actual:
                if name != "periods":
                    compare("trajectory_" + name, actual[name], archived[name])
            require(len(archived["periods"]) == HORIZON, "All twelve full period responses retained")
            for response, saved in zip(actual["periods"], archived["periods"], strict=True):
                compare_response(response, saved)

        # Written independently from the shared production constraint audit.
        margin = np.tile(0.05 + profile.max(axis=0), (25 * HORIZON, 1))
        margin[:25] = 0.05 + profile
        safety_names = ("inner_lower", "inner_upper", "braking_lower", "braking_upper", "terminal")

        def command_groups(commands):
            change = np.diff(np.vstack((previous[:6], commands)), axis=0)
            return dict(
                box_lower=(commands - input_min, 1e-12),
                box_upper=(input_max - commands, 1e-12),
                rate_lower=(0.25 + change, 1e-12),
                rate_upper=(0.25 - change, 1e-12),
            )

        def classify(commands, rollout):
            inner_q = rollout["inner_joint_position"]
            braking = rollout["joint_position"] + 0.1 * rollout["joint_velocity"]
            groups = dict(
                **command_groups(commands),
                inner_lower=(inner_q - lower - margin, 1e-10),
                inner_upper=(upper - margin - inner_q, 1e-10),
                braking_lower=(braking - lower - 0.05, 1e-10),
                braking_upper=(upper - 0.05 - braking, 1e-10),
                terminal=(-normals @ rollout["joint_velocity"][-1], 1e-10),
            )
            require(all(np.all(np.isfinite(v)) for v, _ in groups.values()), "Finite independent constraint slacks")
            safe = all(np.all(slack >= -tol) for slack, tol in groups.values())
            safety = np.concatenate([groups[name][0].ravel() for name in safety_names])
            tcp = rollout["pose"] @ site_from_tcp
            rotation = Rotation.from_matrix(desired[:, :3, :3] @ np.swapaxes(tcp[:, :3, :3], -1, -2)).as_rotvec()
            errors = np.concatenate((desired[:, :3, 3] - tcp[:, :3, 3], rotation), axis=1)
            normalized = np.stack(
                (np.sum(errors[:, :3] ** 2, axis=1) / 0.01**2, np.sum(errors[:, 3:] ** 2, axis=1) / 0.05**2), axis=1
            )
            require(np.all(np.isfinite(normalized)), "Finite independently reconstructed normalized pose errors")
            failed = np.flatnonzero(np.any(normalized > 1.0, axis=1))
            prefix = int(failed[0]) if len(failed) else HORIZON
            return dict(
                groups=groups,
                safety_slack=safety,
                safety_min=float(np.min(safety)),
                all_constraint_min=float(min(np.min(v) for v, _ in groups.values())),
                joint_feasible=bool(safe),
                tcp_errors=errors,
                normalized_squared_errors=normalized,
                prefix_length=prefix,
                maximum_normalized_squared_error=float(np.max(normalized)),
                maximum_pose_cap_violation=float(max(0.0, np.max(np.sqrt(normalized)) - 1.0)),
                pose_loss=float(np.sum(normalized)),
                feasible=bool(safe and np.all(normalized <= 1.0)),
            )

        def check_audit(archived, metrics):
            require(archived["horizon"] == HORIZON and archived["joint_count"] == 7, "H12 seven-joint audit")
            require(archived["activity_tolerance"] == 1e-8, "Original activity threshold")
            compare("audit_margin", margin, archived["position_margin_rad"])
            require(set(archived["groups"]) == set(metrics["groups"]), "All nine audit constraint groups")
            nonterminal = set(metrics["groups"]) - {"terminal"}
            require(
                set(archived["first_period_minima"]) == set(archived["future_period_minima"]) == nonterminal,
                "Complete first/future grouped minima",
            )
            for name, (slack, tolerance) in metrics["groups"].items():
                group = archived["groups"][name]
                saved = np.asarray(group["slack"])
                compare("audit_" + name, slack, saved)
                require(group["tolerance"] == tolerance, "Original feasibility tolerance")
                compare("audit_minimum_" + name, np.min(slack), group["minimum"])
                # Exact activity/violation labels refer to the saved arithmetic;
                # the slacks and actual eligibility are independently recomputed.
                for field, indices in dict(
                    active_indices=np.argwhere(saved <= 1e-8), violated_indices=np.argwhere(saved < -tolerance)
                ).items():
                    np.testing.assert_array_equal(np.asarray(group[field], dtype=int).reshape(-1, saved.ndim), indices)
                if name != "terminal":
                    count = 25 if name.startswith("inner_") else 1
                    compare("audit_first_" + name, np.min(slack[:count]), archived["first_period_minima"][name])
                    compare("audit_future_" + name, np.min(slack[count:]), archived["future_period_minima"][name])
            require(archived["feasible"] is metrics["joint_feasible"], "Independent full trajectory joint eligibility")

        full_entries = {
            name: answer[name] for name in ("initial", "selected", "closest_safety_attempt") if answer[name] is not None
        }
        require(
            "initial" in full_entries and "closest_safety_attempt" in full_entries, "Initial and closest fully retained"
        )
        wanted = {np.asarray(v["commands"], dtype=float).tobytes() for v in full_entries.values()}
        retained, trace_keys, trace_key_set = {}, [], set()
        classified, archived_safety_rank = [], []
        result["trace_rate_violations"] = 0
        for index, item in enumerate(trace):
            commands = np.asarray(item["commands"], dtype=float)
            require(
                commands.shape == (HORIZON, 6)
                and np.all(np.isfinite(commands))
                and np.all(commands >= lo - 1e-12)
                and np.all(commands <= hi + 1e-12),
                "Every trace command inside registered box and first-period bounds",
            )
            key = commands.tobytes()
            require(key not in trace_key_set, "Unique exact-byte trajectory accounting")
            trace_keys.append(key)
            trace_key_set.add(key)
            rollout, differences = reconstruct_trajectory(commands)
            metrics = classify(commands, rollout)
            compare("trace_tcp_errors", metrics["tcp_errors"], item["tcp_errors"])
            compare("trace_safety_minimum", metrics["safety_min"], item["safety_min"])
            rates_ok = all(np.all(metrics["groups"][name][0] >= -1e-12) for name in ("rate_lower", "rate_upper"))
            result["trace_rate_violations"] += int(not rates_ok)
            saved_errors = np.asarray(item["tcp_errors"])
            normalized = np.stack(
                (
                    np.sum(saved_errors[:, :3] ** 2, axis=1) / 0.01**2,
                    np.sum(saved_errors[:, 3:] ** 2, axis=1) / 0.05**2,
                ),
                axis=1,
            )
            saved_failed = np.flatnonzero(np.any(normalized > 1.0, axis=1))
            saved_prefix = int(saved_failed[0]) if len(saved_failed) else HORIZON
            if (
                saved_prefix != metrics["prefix_length"]
                or item["prefix_length"] != metrics["prefix_length"]
                or item["feasible"] is not metrics["feasible"]
            ):
                result["boundary_classification_disagreements"].append(
                    dict(
                        trace_index=index,
                        producer_declared_prefix=item["prefix_length"],
                        producer_error_prefix=saved_prefix,
                        producer_feasible=item["feasible"],
                        producer_joint_feasible=item["joint_feasible"],
                        producer_normalized_squared_errors=normalized,
                        independent_prefix=metrics["prefix_length"],
                        independent_feasible=metrics["feasible"],
                        independent_joint_feasible=metrics["joint_feasible"],
                        independent_normalized_squared_errors=metrics["normalized_squared_errors"],
                        maximum_normalized_difference=float(
                            np.max(abs(normalized - metrics["normalized_squared_errors"]))
                        ),
                        maximum_tcp_error_difference=float(np.max(abs(saved_errors - metrics["tcp_errors"]))),
                        evidence="Boundary classification disagreement / independent qualification not passed; no physical-state or search-success inference",
                    )
                )
            require(
                item["joint_feasible"] is metrics["joint_feasible"]
                and item["feasible"] is metrics["feasible"]
                and type(item["prefix_length"]) is int
                and item["prefix_length"] == metrics["prefix_length"],
                "Every evaluated trajectory independently satisfies the same strict prefix and eligibility classification",
            )
            require(
                saved_prefix == item["prefix_length"], "Strict prefix independently agrees with saved error arithmetic"
            )
            loss = float(np.sum(normalized))
            # The production closest rank includes box/rate as well as nonlinear
            # safety. Reconstruct it from checked saved arithmetic to preserve ties.
            saved_worst = min(
                float(item["safety_min"]), *(float(np.min(v)) for v, _ in command_groups(commands).values())
            )
            archived_safety_rank.append((max(0.0, -saved_worst), loss, index))
            classified.append(
                {
                    k: metrics[k]
                    for k in (
                        "joint_feasible",
                        "feasible",
                        "safety_min",
                        "normalized_squared_errors",
                        "all_constraint_min",
                        "prefix_length",
                    )
                }
            )
            if key in wanted:
                retained[key] = (rollout, metrics)
            result["trace_checks"].append(
                dict(
                    index=index,
                    joint_feasible=metrics["joint_feasible"],
                    feasible=metrics["feasible"],
                    safety_min=metrics["safety_min"],
                    all_constraint_min=metrics["all_constraint_min"],
                    rate_feasible=bool(rates_ok),
                    normalized_squared_errors=metrics["normalized_squared_errors"],
                    prefix_length=metrics["prefix_length"],
                    native_jit_vs_python_goal_difference=max(differences),
                )
            )
            if (index + 1) % 100 == 0 or index + 1 == len(trace):
                print(
                    json.dumps(
                        dict(
                            checked_trajectories=index + 1,
                            total_trajectories=len(trace),
                            numerical_inner_steps=result["numerical_inner_steps"],
                        )
                    ),
                    flush=True,
                )
        require(wanted <= set(retained), "All fully retained candidates occur in the actual response trace")
        np.testing.assert_array_equal(trace[0]["commands"], start)
        for name, saved in full_entries.items():
            key = np.asarray(saved["commands"], dtype=float).tobytes()
            rollout, metrics = retained[key]
            compare_rollout(rollout, saved["rollout"])
            check_audit(saved["audit"], metrics)
            for field in (
                "tcp_errors",
                "normalized_squared_errors",
                "safety_slack",
                "maximum_pose_cap_violation",
                "maximum_normalized_squared_error",
                "pose_loss",
            ):
                compare("full_" + field, metrics[field], saved[field])
            require(saved["feasible"] is metrics["feasible"], "Full candidate eligibility agrees")
            require(saved["prefix_length"] == metrics["prefix_length"], "Full candidate continuous prefix agrees")
            if name == "initial":
                require(key == trace_keys[0], "Initial is exactly the one registered warm start")
        require(
            answer["max_evaluations"] == 4000
            and answer["max_iterations"] == 60
            and answer["finite_difference_step"] == 1e-4
            and answer["optimization_safety_buffer"] == 1e-8,
            "Published fixed trajectory budget and difference arithmetic",
        )
        require(
            answer["initial_prefix_length"] == classified[0]["prefix_length"] == 6
            and answer["selected_prefix_length"] >= 6,
            "The same warm begins with six qualified consecutive points and final selection retains them",
        )
        require_finite_array_equal(
            answer["initial"]["commands"], prior["answer"]["initial"]["commands"], name="unchanged original warm"
        )
        compare("original_warm_tcp_errors", answer["initial"]["tcp_errors"], prior["answer"]["initial"]["tcp_errors"])
        ledger = audit_prefix_search_ledger(answer, classified, compare=compare)
        safe, witnesses = ledger["joint_feasible_indices"], ledger["witness_indices"]
        result["selected_trace_index"] = ledger["selected_trace_index"]
        result["phase_checks"] = ledger["phase_checks"]
        result["initial_prefix_length"] = ledger["initial_prefix_length"]
        result["selected_prefix_length"] = ledger["selected_prefix_length"]
        result["termination_reason"] = ledger["termination_reason"]
        closest = min(range(len(trace)), key=lambda i: archived_safety_rank[i])
        np.testing.assert_array_equal(answer["closest_safety_attempt"]["commands"], trace[closest]["commands"])
        result["closest_safety_trace_index"] = closest
        used = ledger["total_iterations"]
        require(
            result["numerical_inner_steps"] == len(trace) * HORIZON * 25 <= 1200000,
            "Every unique complete trajectory was independently reconstructed",
        )
        require(
            result["mj_forward_queries"] == 1 + len(trace) * HORIZON * 26,
            "Independent current query plus every period's25 inner and endpoint queries",
        )
        require(bool(quat2mat_jit.nopython_signatures), "Native quaternion path really compiled in nopython JIT")
        for ref in references.values():
            checked(ref)
        strict_native_environment()
        result.update(
            complete=True,
            passed=True,
            endpoint=368,
            horizon=HORIZON,
            target_cursors=list(range(HORIZON)),
            initial_history_completed_periods=0,
            actual_history_labels=0,
            trace_count=len(trace),
            predicted_control_periods=len(trace) * HORIZON,
            full_candidates_checked=list(full_entries),
            unique_full_candidates=len(wanted),
            joint_feasible_trajectories=len(safe),
            witness_trajectories=len(witnesses),
            native_quaternion_jit_signatures=[str(v) for v in quat2mat_jit.nopython_signatures],
            witness_found=bool(witnesses),
            total_iterations=used,
            past_inner_error_rad=profile,
            terminal_normals=normals,
            source_refs_unchanged=list(references.values()),
            evidence="Every full causal trajectory independently reconstructed; no physical control/task/fresh qualification",
        )
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        publish(ROOT / "verified.json", result)


if __name__ == "__main__":
    main()
