"""Independently audit one fixed H13 reference-phase case, including rejected warm starts.

The numerical query and native OSC recurrence are derived from the separately
executed prefix checker. No production phase helper, solver, response, dynamics
query or constraint-audit implementation is imported.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from check_tracking_v5_prefix_preserving_trajectory_control import (
    audit_prefix_search_ledger,
    finite_max_abs_difference,
    require_absolute_tolerance,
    require_finite_array_equal,
)
from check_tracking_v5_state_updated_response import native_functions
from scipy.spatial.transform import Rotation

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.preview_result_publication import publish_result as publish
from s2a.terminal_development_execution import CONTROL_REF

HORIZON = 13
SOURCE_HORIZON = 12
NUMERIC_TOLERANCE = 1e-10
TORQUE_TOLERANCE = 1e-8


def reconstruct_phase_case(name, targets, openings, warm12):
    """Build the registered reference and command midpoint without production helpers."""
    require(name in ("early", "late"), "Exactly one registered early/late case")
    targets, openings, warm12 = np.asarray(targets), np.asarray(openings), np.asarray(warm12)
    require(
        targets.ndim == 3
        and targets.shape[1:] == (4, 4)
        and len(targets) >= 13
        and openings.shape == (len(targets),)
        and warm12.shape == (12, 6)
        and np.all(np.isfinite(targets))
        and np.all(np.isfinite(openings))
        and np.all((openings >= 0) & (openings <= 1))
        and np.all(np.isfinite(warm12)),
        "Complete finite generated targets, openings and original H12 warm",
    )
    left = 5 if name == "early" else 8
    insert = left + 1
    phases = np.insert(np.arange(12, dtype=float), insert, left + 0.5)
    midpoint = np.eye(4)
    midpoint[:3, 3] = (targets[left, :3, 3] + targets[left + 1, :3, 3]) / 2
    # A world-frame relative rotation gives the shortest midpoint on SO(3).
    increment = Rotation.from_matrix(targets[left + 1, :3, :3] @ targets[left, :3, :3].T).as_rotvec()
    midpoint[:3, :3] = Rotation.from_rotvec(increment / 2).as_matrix() @ targets[left, :3, :3]
    desired = np.insert(targets[:12], insert, midpoint, axis=0)
    opening = np.insert(openings[:12], insert, (openings[left] + openings[left + 1]) / 2)
    warm = np.insert(warm12, insert, (warm12[left] + warm12[left + 1]) / 2, axis=0)
    node_indices = np.delete(np.arange(13), insert)
    return dict(
        name=name,
        phases=phases,
        desired_tcp=desired,
        opening=opening,
        original_timed_tcp=targets[:13].copy(),
        original_timed_opening=openings[:13].copy(),
        warm=warm,
        source_warm12=warm12.copy(),
        used_start=warm.copy(),
        start_roundoff_clip=np.zeros_like(warm),
        original_timed_cursors=np.arange(13),
        original_node_cursors=np.arange(12),
        original_node_endpoint_indices=node_indices,
        unconsumed_suffix=np.arange(12, len(targets)),
        physical_time_s=0.05 * np.arange(1, 14),
        reference_time_s=0.05 * (phases + 1),
        duration_s=0.65,
        control_dt_s=0.05,
        scheduled_original_node_count=12,
        scheduled_final_original_cursor=11,
        actual_consumed_reference_count=0,
    )


def check_command_admission(case, previous, input_min, input_max):
    """Require the registered source/derived commands and exact solver start bytes."""
    previous = np.asarray(previous)
    input_min, input_max = np.asarray(input_min), np.asarray(input_max)
    require(
        previous.shape == input_min.shape == input_max.shape == (6,)
        and np.all(np.isfinite(previous))
        and np.all(np.isfinite(input_min))
        and np.all(np.isfinite(input_max))
        and np.all(input_min < input_max),
        "Original finite command interface and completed previous command",
    )
    for name in ("source_warm12", "warm"):
        commands = np.asarray(case[name], dtype=float)
        require(
            np.all(commands >= input_min - 1e-12)
            and np.all(commands <= input_max + 1e-12)
            and np.all(abs(np.diff(np.vstack((previous, commands)), axis=0)) <= 0.25 + 1e-12),
            "Registered source and derived warm satisfy the unchanged box/rate",
        )
    lo, hi = np.tile(input_min, (13, 1)), np.tile(input_max, (13, 1))
    lo[0], hi[0] = np.maximum(lo[0], previous - 0.25), np.minimum(hi[0], previous + 0.25)
    require(np.all(hi - lo > 1e-4), "Nondegenerate original finite-difference box")
    warm = np.asarray(case["warm"], dtype=float)
    require(np.clip(warm, lo, hi).tobytes() == warm.tobytes(), "Solver clipping must preserve every warm byte")
    return lo, hi


def score_phase_contracts(site_poses, case, site_from_tcp):
    """Recompute original timed, fixed retimed and scheduled original-node errors."""
    tcp = np.asarray(site_poses) @ np.asarray(site_from_tcp)
    require(tcp.shape == (13, 4, 4) and np.all(np.isfinite(tcp)), "All thirteen finite predicted TCP poses")

    def metric(desired, actual):
        rotation = Rotation.from_matrix(desired[:, :3, :3] @ actual[:, :3, :3].swapaxes(-1, -2)).as_rotvec()
        errors = np.c_[desired[:, :3, 3] - actual[:, :3, 3], rotation]
        normalized = np.column_stack(((errors[:, :3] ** 2).sum(1) / 0.01**2, (errors[:, 3:] ** 2).sum(1) / 0.05**2))
        require(np.all(np.isfinite(normalized)), "Finite dual-contract normalized pose scores")
        passed = np.all(normalized <= 1, axis=1)
        failures = np.flatnonzero(~passed)
        prefix = int(failures[0]) if len(failures) else len(passed)
        return dict(
            tcp_errors=errors,
            normalized_squared_errors=normalized,
            pose_cap_pass=passed,
            all_pose_caps_satisfied=bool(np.all(passed)),
            prefix_length=prefix,
            maximum_normalized_squared_error=float(np.max(normalized)),
            maximum_pose_cap_violation=float(max(0, np.max(np.sqrt(normalized)) - 1)),
            pose_loss=float(np.sum(normalized)),
        )

    original = metric(np.asarray(case["original_timed_tcp"]), tcp)
    retimed = metric(np.asarray(case["desired_tcp"]), tcp)
    indices = np.asarray(case["original_node_endpoint_indices"], dtype=int)
    nodes = metric(np.asarray(case["original_timed_tcp"])[:12], tcp[indices])
    return dict(
        predicted_tcp=tcp,
        original_timed=original,
        retimed_path=retimed,
        original_nodes=nodes,
        predicted_qualified_original_node_prefix=nodes["prefix_length"],
        predicted_final_qualified_original_cursor=nodes["prefix_length"] - 1,
        actual_consumed_reference_count=0,
    )


def compare_contract(actual, saved, *, compare=None, name="contract"):
    """Compare every contract field, preserving strict integer/boolean labels."""
    if compare is None:

        def compare(label, left, right):
            difference = finite_max_abs_difference(left, right, name=label)
            require_absolute_tolerance(difference, NUMERIC_TOLERANCE, name=label)

    if isinstance(actual, dict):
        require(isinstance(saved, dict) and set(actual) == set(saved), f"Complete {name} fields")
        for key in actual:
            compare_contract(actual[key], saved[key], compare=compare, name=name + "." + key)
    elif isinstance(actual, (list, tuple)) and any(isinstance(item, dict) for item in actual):
        require(isinstance(saved, (list, tuple)) and len(saved) == len(actual), f"Complete {name} sequence")
        for index, (item, archived) in enumerate(zip(actual, saved, strict=True)):
            compare_contract(item, archived, compare=compare, name=f"{name}[{index}]")
    elif actual is None or isinstance(actual, (str, bool, int)):
        require(type(saved) is type(actual) and saved == actual, f"Exact {name} label")
    else:
        array, archived = np.asarray(actual), np.asarray(saved)
        if array.dtype.kind in "biu":
            kind_ok = archived.dtype.kind == "b" if array.dtype.kind == "b" else archived.dtype.kind in "iu"
            require(
                kind_ok and array.shape == archived.shape and np.array_equal(array, archived), f"Exact {name} labels"
            )
        else:
            compare(name, actual, saved)


def warm_admission_status(metrics):
    """A complete witness bypasses only the extra optimization safety buffer."""
    complete = bool(metrics["joint_feasible"] and metrics["feasible"])
    admitted = bool(complete or (metrics["joint_feasible"] and metrics["safety_min"] >= 1e-8))
    return dict(
        joint_feasible=bool(metrics["joint_feasible"]),
        safety_min=float(metrics["safety_min"]),
        complete_witness=complete,
        admitted=admitted,
        status="complete_witness" if complete else "admitted" if admitted else "warm_admission_failed",
        rejection_reason=None
        if admitted
        else "joint_or_command_audit_failed"
        if not metrics["joint_feasible"]
        else "optimization_safety_buffer_not_met",
    )


def audit_case_accounting(producer, classifications, *, compare=None):
    """Validate admission and response counts after independent trajectory reconstruction."""
    trace, answer = producer["trace"], producer["answer"]
    require(len(trace) == len(classifications) == producer["response_evaluations"], "Complete case response ledger")
    require(1 <= len(trace) <= 4000, "Admitted commands require at least the full warm response")
    admission = warm_admission_status(classifications[0])
    compare_contract(admission, producer["warm_admission"], compare=compare, name="warm_admission")
    require(producer["numerical_inner_steps"] == 325 * len(trace), "Every H13 response has325 inner steps")
    require(producer["query_count"] == 1 + 338 * len(trace), "One current query plus338 per unique full response")
    if not admission["admitted"]:
        require(
            answer is None
            and len(trace) == producer["prediction_calls"] == 1
            and producer["cache_hits"] == 0
            and producer["total_iterations"] == 0
            and producer["phases"] == []
            and producer["status"] == "warm_admission_failed"
            and producer["witness_found"] is False,
            "Rejected dynamic warm retains exactly one response and starts no optimizer",
        )
        return dict(
            termination_reason="warm_admission_failed",
            selected_trace_index=None,
            joint_feasible_indices=[0] if admission["joint_feasible"] else [],
            witness_indices=[],
            initial_prefix_length=classifications[0]["prefix_length"],
            selected_prefix_length=None,
            total_iterations=0,
            phase_checks=[],
        )
    require(isinstance(answer, dict), "An admitted warm has the original complete solver answer")
    require(
        producer["prediction_calls"] == len(trace) + 1
        and producer["cache_hits"] == 1
        and producer["total_iterations"] == answer["total_iterations"]
        and producer["witness_found"] is answer["witness_found"],
        "Preflight and solver initial share exactly one cached response",
    )
    compare_contract(answer["phases"], producer["phases"], compare=compare, name="mirrored_phases")
    require(len(answer["evaluated_candidates"]) == len(trace), "No omitted or invented solver responses")
    for saved, row in zip(answer["evaluated_candidates"], trace, strict=True):
        for key in ("commands", "safety_min", "joint_feasible", "tcp_errors", "prefix_length", "feasible"):
            compare_contract(saved[key], row[key], compare=compare, name="mirrored_trace." + key)
    return audit_prefix_search_ledger(answer, classifications, compare=compare)


def audit_command_rejection(producer, case, previous, input_min, input_max):
    """A command-only rejection must be independently justified before any query."""
    try:
        check_command_admission(case, previous, input_min, input_max)
    except ValueError as error:
        reason = str(error)
    else:
        raise ValueError("A valid registered warm cannot be labelled command_admission_failed")
    require(
        producer["status"] == "command_admission_failed"
        and producer["case_contract"]
        is producer["answer"]
        is producer["warm_preflight"]
        is producer["warm_admission"]
        is None
        and producer["command_admission"]["admitted"] is False
        and isinstance(producer["command_admission"]["reason"], str)
        and bool(producer["command_admission"]["reason"])
        and all(
            type(producer[name]) is int and producer[name] == 0
            for name in (
                "response_evaluations",
                "total_iterations",
                "prediction_calls",
                "cache_hits",
                "query_count",
                "numerical_inner_steps",
            )
        )
        and producer["trace"] == producer["phases"] == []
        and producer["dual_metrics"] == {}
        and producer["witness_found"] is False,
        "Command rejection must retain zero queries/responses/iterations and no fabricated dynamic record",
    )
    return reason


def main(case_name):
    strict_native_environment()
    require(case_name in ("early", "late"), "Explicit paired case")
    root = REPO / f"outputs/tracking_v5_paired_reference_phase_{case_name}_check_20261028"
    producer_path = REPO / f"outputs/tracking_v5_paired_reference_phase_{case_name}_20261028/verified.json"
    owner_path = REPO / f"outputs/tracking_v5_paired_reference_phase_{case_name}_execution_20261028/result.json"
    require(not root.exists(), "No overwrite or automatic retry")
    references = {}

    def remember(ref):
        checked(ref)
        require(ref["path"] not in references or references[ref["path"]] == ref, "Unique source fingerprint")
        references[ref["path"]] = ref
        return ref

    def load(ref):
        return read(remember(ref))

    def closed_owner(ref, expected=None):
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
            and released["original_error"] is None
            and released["pid"] == launch["child"]["pid"]
            and released["start_ticks"] == launch["child"]["start_ticks"]
            and launch["selected_environment"]["CUDA_VISIBLE_DEVICES"] == ""
            and launch["selected_environment"]["LD_LIBRARY_PATH"] == ""
            and launch["automatic_retry"] is False
            and exited["automatic_retry"] is False,
            "Same CPU child actually waited/reaped0 without signal or retry",
        )
        if expected is not None:
            require(owner["result_ref"] == expected, "Owner binds this exact result")
        value = load(owner["result_ref"])
        require(value["complete"] is value["passed"] is True, "Prerequisite result passed")
        return owner, value

    producer_ref, producer_owner_ref = remember(reference(producer_path)), remember(reference(owner_path))
    producer_owner, producer = closed_owner(producer_owner_ref, producer_ref)
    owner_reg = load(producer_owner["registration_ref"])
    for path, sha256 in owner_reg["source_sha256"].items():
        remember(dict(path=path, sha256=sha256))
    launch = load(producer_owner["owned_command"]["launch"])
    require(
        launch["command"]
        == [
            str(REPO / "outputs/libero_env311/bin/python"),
            str(REPO / "scripts/probe_tracking_v5_paired_reference_phase.py"),
            "--case",
            case_name,
        ],
        "Owner executed exactly this registered case",
    )
    require(
        producer["physics_steps"] == producer["world_model_calls"] == producer["fresh_cases"] == 0
        and producer["control_qualified"] is False,
        "Numerical case cannot dispatch or qualify physical control",
    )
    reg = load(producer["registration_ref"])
    prefix_owner_ref = remember(
        reference(
            REPO / "outputs/tracking_v5_prefix_preserving_trajectory_control_check_execution_20261028/result.json"
        )
    )
    prefix_owner, prefix_check = closed_owner(prefix_owner_ref)
    prefix_check_reg = load(prefix_check["registration_ref"])
    require(
        prefix_check["trace_count"] == 4000
        and prefix_check["numerical_inner_steps"] == 1200000
        and prefix_check["mj_forward_queries"] == 1248001
        and prefix_check["total_iterations"] == 54
        and prefix_check["initial_prefix_length"] == 6
        and prefix_check["selected_prefix_length"] == 9
        and prefix_check["witness_found"] is False
        and prefix_check["control_qualified"] is False,
        "Complete independently closed fixed H12 prefix prerequisite",
    )
    for ref in prefix_check["source_refs_unchanged"]:
        remember(ref)
    for path, digest in load(prefix_owner["registration_ref"])["source_sha256"].items():
        remember(dict(path=path, sha256=digest))
    prefix_ref = remember(prefix_check_reg["producer_ref"])
    _, prefix = closed_owner(prefix_check_reg["producer_owner_ref"], prefix_ref)
    source_reg = load(prefix["registration_ref"])
    frozen_checker_ref = remember(reference(REPO / "scripts/check_tracking_v5_prefix_preserving_trajectory_control.py"))
    require(
        frozen_checker_ref == prefix_check_reg["source_ref"], "Imported ledger is the exact closed independent checker"
    )
    phase_ref = remember(reference(REPO / "outputs/tracking_v5_saved_trajectory_phase_20261028/result.json"))
    phase = load(phase_ref)
    phase_reg = load(phase["registration_ref"])
    require(
        phase["complete"] is phase["passed"] is True
        and phase["control_qualified"] is phase["retiming_qualified"] is False
        and prefix_ref in phase["input_refs_before"],
        "Closed saved-geometry diagnosis binds this prefix without retiming qualification",
    )
    for ref in phase["source_refs_after"] + phase["input_refs_after"] + phase_reg["source_refs"]:
        remember(ref)
    # New producer schema bindings are checked below, independently of its helper.
    require(reg["seed"] == 42 and reg["endpoint"] == 368 and reg["horizon"] == HORIZON, "Registered actual368 H13 case")
    for ref in reg["source_refs"]:
        remember(ref)
    early = load(source_reg["early_numeric_ref"])
    entries = [e for e in early["entries"] if e["step"] == 368 and e["horizon"] == SOURCE_HORIZON]
    require(len(entries) == 1, "Original H12 warm source remains H12")
    entry = entries[0]
    source_warm = np.asarray(entry["answer"]["planned_commands"], dtype=float)
    require(source_warm.shape == (12, 6), "One original twelve-command source warm")
    require_finite_array_equal(source_warm, prefix["answer"]["initial"]["commands"], name="original H12 warm")
    normals, profile = np.eye(7)[6:], np.zeros((25, 7))
    require_finite_array_equal(entry["causal_history"], profile, name="empty current causal history")
    require_finite_array_equal(entry["answer"]["terminal_normals"], normals, name="original terminal normals")
    with np.load(checked(remember(source_reg["states_ref"])), allow_pickle=False) as archive:
        state = archive["sim_states"][368].copy()
    with checked(remember(source_reg["original_rows_ref"])).open() as stream:
        for index, line in enumerate(stream):
            if index == 367:
                previous_row = json.loads(line)
                break
        else:
            raise ValueError("Missing completed actual367 controller state")
    previous = np.asarray(previous_row["command"], dtype=float)
    previous_site = np.asarray(previous_row["qualified_forward_input"]["x_site_pose"])[0]
    latch = np.asarray(previous_row["controller_goal_after"])
    require_finite_array_equal(previous, entry["previous_command"], name="completed367 command")
    chunk = load(source_reg["chunks_ref"])[46]
    require(chunk["step"] == 368, "Only the already generated chunk46 conditions both cases")
    targets = np.asarray(chunk["effective_targets_tcp"])
    openings = np.asarray(chunk["openings"])
    case = reconstruct_phase_case(case_name, targets, openings, source_warm)
    require(
        len(targets) == 40 and np.all(openings[:13] == 1), "Forty-node generated reference with unchanged local opening"
    )
    desired = case["desired_tcp"]
    control_record, tool = load(source_reg["control_ref"]), load(source_reg["tool_ref"])
    control = control_record["contract"]
    require(control["control_freq"] == 20 and control["model_timestep"] == 0.002, "Fixed native timing")
    site_from_tcp = np.linalg.solve(np.asarray(control_record["source_from_site"]), np.asarray(tool["source_from_tcp"]))
    initial = np.asarray(load(source_reg["initial_ref"])["initial_joint"])
    kp, kd = np.asarray(control["kp"]), np.asarray(control["kd"])
    input_min, input_max = np.asarray(control["input_min"]), np.asarray(control["input_max"])
    output_min, output_max = np.asarray(control["output_min"]), np.asarray(control["output_max"])
    previous_delta = output_min + (np.clip(previous[:6], input_min, input_max) - input_min) * (
        (output_max - output_min) / (input_max - input_min)
    )
    require(np.any(previous_delta[3:] != 0), "Completed367 nonzero reset identifies the historical latch")
    command_error = None
    try:
        lo, hi = check_command_admission(case, previous[:6], input_min, input_max)
    except ValueError as error:
        command_error = str(error)
    registration = publish(
        root / "registration.json",
        dict(
            producer_ref=producer_ref,
            producer_owner_ref=producer_owner_ref,
            producer_registration_ref=producer["registration_ref"],
            prior_prefix_check_owner_ref=prefix_owner_ref,
            prior_prefix_check_ref=prefix_owner["result_ref"],
            prior_prefix_ref=prefix_ref,
            phase_diagnostic_ref=phase_ref,
            independent_ledger_source_ref=frozen_checker_ref,
            source_ref=remember(reference(Path(__file__))),
            version_plan_ref=remember(reference(REPO / "plans/v0.0.1/paired_reference_phase_20261028.md")),
            seed=42,
            endpoint=368,
            case_name=case_name,
            horizon=HORIZON,
            source_horizon=SOURCE_HORIZON,
            initial_history_completed_periods=0,
            numeric_tolerance=NUMERIC_TOLERANCE,
            torque_tolerance=TORQUE_TOLERANCE,
            independence="Independent fixed phase/midpoint/scoring, native AST/JIT query and full recurrence; no production helper/solver/response/audit import",
            query_accounting="325 numerical inner steps and338 forward queries per unique full H13; one current query for admitted commands",
            contracts="Original timed cursor0..12 versus fixed13 phases and12 scheduled original nodes; no actual consumption",
            qualification="Numerical reconstruction only; no physical control, retiming, task or fresh qualification",
            automatic_retry=False,
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        registration_ref=registration,
        case_name=case_name,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        control_qualified=False,
        retiming_qualified=False,
        actual_consumed_reference_count=0,
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
        require(
            producer["case_name"] == reg["case_name"] == case_name
            and reg["prior_prefix_check_owner_ref"] == prefix_owner_ref
            and reg["prior_prefix_check_ref"] == prefix_owner["result_ref"]
            and reg["prior_prefix_ref"] == prefix_ref
            and reg["saved_phase_diagnostic_ref"] == phase_ref
            and reg["cursor"] == reg["initial_history_completed_periods"] == 0
            and reg["automatic_retry"] is False,
            "This case binds the exact closed prefix and phase prerequisites",
        )
        for name in (
            "prior_trajectory_check_owner_ref",
            "prior_trajectory_check_ref",
            "prior_trajectory_ref",
            "early_numeric_ref",
            "early_independent_owner_ref",
            "physical_independent_owner_ref",
            "original_producer_ref",
            "manifest_ref",
            "states_ref",
            "original_rows_ref",
            "initial_ref",
            "xml_ref",
            "control_ref",
            "tool_ref",
            "chunks_ref",
            "native_source_refs",
            "native_transform_ref",
        ):
            require(reg[name] == source_reg[name], "Original causal and native source bindings remain unchanged")
        compare_contract(
            dict(
                response_budget=4000,
                total_iterations=60,
                difference_step=1e-4,
                optimization_safety_buffer=1e-8,
                ftol=1e-12,
                starts=1,
                restoration=False,
            ),
            reg["search"],
            compare=compare,
            name="registered_search",
        )
        compare_contract(
            dict(
                rate=0.25,
                joint_margin=0.05,
                braking_time=0.1,
                position_cap=0.01,
                rotation_cap=0.05,
                pose_caps_horizon=13,
                terminal_normals=normals,
            ),
            reg["constraints"],
            compare=compare,
            name="registered_constraints",
        )
        require(
            producer["actual_consumed_reference_count"] == reg["actual_consumed_reference_count"] == 0
            and producer["retimed_control_qualified"] is producer["original_timed_control_qualified"] is False,
            "Only scheduled/predicted progress, no consumed original node or control qualification",
        )
        for name in (
            "current_q",
            "current_velocity",
            "fixed_gripper_state",
            "initial_joint",
            "kp",
            "kd",
            "torque_limits",
            "joint_limits",
            "site_from_tcp",
            "previous_command",
            "previous_site",
            "previous_physical_delta",
            "latched_rotation_goal",
            "past_inner_error_rad",
            "terminal_normals",
        ):
            compare_contract(prefix[name], producer[name], compare=compare, name="closed_input." + name)
        for name, value in dict(
            source_warm12=source_warm,
            source_targets_tcp=targets,
            source_openings=openings,
            actual367_motor_target_after=previous_row["motor_target_after"],
            actual367_motor_opening_after=previous_row["motor_opening_after"],
            actual367_gripper_command=float(previous[6]),
        ).items():
            compare_contract(value, producer[name], compare=compare, name="source_input." + name)
        compare_contract(
            dict(
                completed_periods=0, last_completed_step=None, inner_position_error_rad=profile, future_guarantee=False
            ),
            producer["history_before_search"],
            compare=compare,
            name="zero_causal_history",
        )
        if command_error is not None:
            command_error = audit_command_rejection(producer, case, previous[:6], input_min, input_max)
            compare_contract(
                prefix["current_quantities"],
                producer["current_quantities"],
                compare=compare,
                name="closed_current_quantities",
            )
            for ref in references.values():
                checked(ref)
            strict_native_environment()
            result.update(
                complete=True,
                passed=True,
                termination_reason="command_admission_failed",
                command_admission_error=command_error,
                trace_count=0,
                witness_found=False,
                source_refs_unchanged=list(references.values()),
            )
            return
        compare_contract(
            dict(admitted=True, reason=None), producer["command_admission"], compare=compare, name="command_admission"
        )
        compare_contract(case, producer["case_contract"], compare=compare, name="registered_case")
        for field in ("warm", "source_warm12", "used_start"):
            require(
                np.asarray(case[field], dtype=float).tobytes()
                == np.asarray(producer["case_contract"][field], dtype=float).tobytes(),
                "Every registered warm and start byte is unchanged",
            )
        stop_owner, stop_check = closed_owner(source_reg["prior_stop_check_owner_ref"])
        stop_check_reg = load(stop_check["registration_ref"])
        control_refs = [r for r in source_reg["native_source_refs"] if r["path"].endswith("/utils/control_utils.py")]
        require(len(control_refs) == 1, "Single pinned native OSC source")
        functions, control_locations = native_functions(
            remember(control_refs[0]), ("orientation_error", "opspace_matrices", "nullspace_torques")
        )
        transform_ref = remember(source_reg["native_transform_ref"])
        require(
            transform_ref == stop_check_reg["native_transform_source_ref"],
            "Same independently checked native goal source",
        )
        transformations, transform_locations = native_functions(
            transform_ref, ("axisangle2quat", "quat2mat"), include_eps=True
        )
        manifest = load(source_reg["manifest_ref"])
        for ref in [*source_reg["native_source_refs"], transform_ref]:
            remember(ref)
            require(
                manifest["execution_sha256"][ref["path"]] == ref["sha256"], "Original native execution source unchanged"
            )
        helper_ref = remember(reference(REPO / "scripts/check_tracking_v5_state_updated_response.py"))
        require(helper_ref == prefix_check_reg["ast_helper_ref"], "Same independent AST extraction source")
        result["native_function_sources"] = dict(
            control_ref=control_refs[0],
            transform_ref=transform_ref,
            ast_helper_ref=helper_ref,
            control_locations=control_locations,
            transform_locations=transform_locations,
        )
        reg = source_reg
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
            require(len(periods) == HORIZON, "Exactly thirteen autonomous predicted periods")
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
            require(len(archived["periods"]) == HORIZON, "All thirteen full period responses retained")
            for response, saved in zip(actual["periods"], archived["periods"], strict=True):
                compare_response(response, saved)

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
            require(archived["horizon"] == HORIZON and archived["joint_count"] == 7, "H13 seven-joint audit")
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

        require(state.shape == (1 + model.nq + model.nv,), "Complete actual368 state vector")
        q0, v0 = state[1 + qids].copy(), state[1 + model.nq + dofs].copy()
        fixed = {name: [state[1 + qi], state[1 + model.nq + vi]] for name, (qi, vi) in fingers.items()}
        require_finite_array_equal(q0, entry["current_q"], name="actual368 arm position")
        require_finite_array_equal(v0, entry["current_velocity"], name="actual368 arm velocity")
        compare_contract(fixed, producer["fixed_gripper_state"], compare=compare, name="actual368 fingers")
        current = query(q0, v0, fixed)
        compare("earlier_current_site", current["pose"], entry["current_site"])
        compare_contract(current, producer["current_quantities"], compare=compare, name="current_quantities")
        for name, value in dict(
            current_q=q0,
            current_velocity=v0,
            initial_joint=initial,
            kp=kp,
            kd=kd,
            torque_limits=torque_limits,
            joint_limits=limits,
            site_from_tcp=site_from_tcp,
            previous_command=previous,
            previous_site=previous_site,
            previous_physical_delta=previous_delta,
            latched_rotation_goal=latch,
            past_inner_error_rad=profile,
            terminal_normals=normals,
        ).items():
            compare("input_" + name, value, producer[name])
        prior_goal = quat2mat_jit(transformations["axisangle2quat"](previous_delta[3:])) @ previous_site[:3, :3]
        compare("actual367_latched_goal", prior_goal, latch)
        margin = np.tile(0.05 + profile.max(axis=0), (25 * HORIZON, 1))
        margin[:25] = 0.05 + profile
        safety_names = ("inner_lower", "inner_upper", "braking_lower", "braking_upper", "terminal")
        answer, trace = producer["answer"], producer["trace"]
        require(isinstance(trace, list) and 1 <= len(trace) <= 4000, "Complete bounded case trace")
        require(isinstance(producer["warm_preflight"], dict), "Retained full H13 preflight response")
        full_entries = {"warm": producer["warm_preflight"]}
        if answer is not None:
            require(isinstance(answer, dict), "Original solver answer or explicit admission rejection")
            full_entries.update({name: answer[name] for name in ("initial", "selected", "closest_safety_attempt")})
            require(
                all(value is not None for value in full_entries.values()),
                "Safe warm guarantees all three full candidates",
            )
            require(
                answer["horizon"] == answer["pose_caps_horizon"] == HORIZON
                and answer["all_pose_caps_enforced"] is True
                and answer["pose_cap_enforcement_scope"]
                == "All H caps are required for a witness; stages hard-constrain only the attained prefix"
                and answer["max_evaluations"] == 4000
                and answer["max_iterations"] == 60
                and answer["finite_difference_step"] == 1e-4
                and answer["optimization_safety_buffer"] == 1e-8,
                "Unchanged frozen solver and all13-point witness contract",
            )
            compare("answer_profile", profile, answer["past_inner_error_rad"])
            compare("answer_terminal", normals, answer["terminal_normals"])
            compare("command_lower", lo, answer["lower_command"])
            compare("command_upper", hi, answer["upper_command"])
            for field in ("supplied_warm", "used_start"):
                require(
                    np.asarray(answer[field], dtype=float).tobytes() == case["warm"].tobytes(),
                    "Original warm bytes reach the solver unchanged",
                )
            require_finite_array_equal(answer["start_roundoff_clip"], np.zeros((13, 6)), name="no start clipping")
        require(
            set(producer["dual_metrics"]) == set(full_entries),
            "Dual metrics for warm and every retained full candidate",
        )
        wanted = {np.asarray(record["commands"], dtype=float).tobytes() for record in full_entries.values()}
        retained, trace_keys, trace_key_set, classified, archived_safety_rank = {}, [], set(), [], []
        result["trace_rate_violations"] = 0
        for index, item in enumerate(trace):
            commands = np.asarray(item["commands"], dtype=float)
            require(
                commands.shape == (13, 6)
                and np.all(np.isfinite(commands))
                and np.all(commands >= lo - 1e-12)
                and np.all(commands <= hi + 1e-12),
                "Every full trace command remains within the fixed box",
            )
            key = commands.tobytes()
            require(key not in trace_key_set, "Every exact-byte trajectory is uniquely charged")
            if index == 0:
                require(
                    key == case["warm"].tobytes(), "The only preflight and first response is the exact registered warm"
                )
            trace_keys.append(key)
            trace_key_set.add(key)
            rollout, differences = reconstruct_trajectory(commands)
            metrics = classify(commands, rollout)
            dual = score_phase_contracts(rollout["pose"], case, site_from_tcp)
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
            failures = np.flatnonzero(np.any(normalized > 1, axis=1))
            saved_prefix = int(failures[0]) if len(failures) else HORIZON
            disagreement = (
                saved_prefix != metrics["prefix_length"]
                or item["prefix_length"] != metrics["prefix_length"]
                or item["joint_feasible"] is not metrics["joint_feasible"]
                or item["feasible"] is not metrics["feasible"]
            )
            for contract in ("original_timed", "retimed_path", "original_nodes"):
                saved_metric = item["dual_metrics"][contract]
                independent_metric = dual[contract]
                if (
                    saved_metric["prefix_length"] != independent_metric["prefix_length"]
                    or saved_metric["all_pose_caps_satisfied"] is not independent_metric["all_pose_caps_satisfied"]
                    or not np.array_equal(saved_metric["pose_cap_pass"], independent_metric["pose_cap_pass"])
                ):
                    disagreement = True
            if disagreement:
                result["boundary_classification_disagreements"].append(
                    dict(
                        trace_index=index,
                        producer_prefix=item["prefix_length"],
                        independent_prefix=metrics["prefix_length"],
                        producer_joint_feasible=item["joint_feasible"],
                        independent_joint_feasible=metrics["joint_feasible"],
                        producer_feasible=item["feasible"],
                        independent_feasible=metrics["feasible"],
                        producer_normalized_squared_errors=normalized,
                        independent_normalized_squared_errors=metrics["normalized_squared_errors"],
                        producer_dual_metrics=item["dual_metrics"],
                        independent_dual_metrics=dual,
                        maximum_tcp_error_difference=float(np.max(abs(saved_errors - metrics["tcp_errors"]))),
                        evidence="Strict classification disagreement; numerical qualification fails, without a physical-limit or search-success inference",
                    )
                )
            require(not disagreement, "All original strict classifications agree, without an isclose relaxation")
            compare_contract(dual, item["dual_metrics"], compare=compare, name="trace_dual_metrics")
            loss = float(np.sum(normalized))
            saved_worst = min(
                float(item["safety_min"]), *(float(np.min(v)) for v, _ in command_groups(commands).values())
            )
            archived_safety_rank.append((max(0.0, -saved_worst), loss, index))
            classified.append(
                {
                    name: metrics[name]
                    for name in (
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
                retained[key] = (rollout, metrics, dual)
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
                    dual_metrics=dual,
                    native_jit_vs_python_goal_difference=max(differences),
                )
            )
            if (index + 1) % 100 == 0 or index + 1 == len(trace):
                print(
                    json.dumps(
                        dict(
                            case_name=case_name,
                            checked_trajectories=index + 1,
                            total_trajectories=len(trace),
                            numerical_inner_steps=result["numerical_inner_steps"],
                        )
                    ),
                    flush=True,
                )
        require(wanted <= set(retained), "Every fully retained candidate belongs to the actual unique response ledger")
        for name, saved in full_entries.items():
            key = np.asarray(saved["commands"], dtype=float).tobytes()
            rollout, metrics, dual = retained[key]
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
            require(
                saved["feasible"] is metrics["feasible"] and saved["prefix_length"] == metrics["prefix_length"],
                "Full candidate strict eligibility agrees",
            )
            compare_contract(dual, producer["dual_metrics"][name], compare=compare, name="full_dual_metrics." + name)
            if name in ("warm", "initial"):
                require(key == trace_keys[0], "Preflight and solver initial bind the same first unique response")
        ledger = audit_case_accounting(producer, classified, compare=compare)
        closest = None
        if answer is not None:
            closest = min(range(len(trace)), key=lambda index: archived_safety_rank[index])
            require(
                np.asarray(answer["closest_safety_attempt"]["commands"], dtype=float).tobytes() == trace_keys[closest],
                "Exact closest-safety ranking includes original box and rate",
            )
            require(producer["status"] == answer["status"], "Top-level status preserves the solver result")
        require(
            result["numerical_inner_steps"] == producer["numerical_inner_steps"] == len(trace) * 325 <= 1300000
            and result["mj_forward_queries"] == producer["query_count"] == 1 + len(trace) * 338,
            "One current query plus every uniquely charged complete H13 reconstruction",
        )
        require(bool(quat2mat_jit.nopython_signatures), "Quaternion path really compiled in native nopython JIT")
        for ref in producer["source_refs_unchanged"]:
            remember(ref)
        for ref in references.values():
            checked(ref)
        strict_native_environment()
        result.update(
            complete=True,
            passed=True,
            endpoint=368,
            horizon=13,
            source_horizon=12,
            case_contract=case,
            initial_history_completed_periods=0,
            actual_history_labels=0,
            trace_count=len(trace),
            predicted_control_periods=13 * len(trace),
            full_candidates_checked=list(full_entries),
            unique_full_candidates=len(wanted),
            joint_feasible_trajectories=len(ledger["joint_feasible_indices"]),
            witness_trajectories=len(ledger["witness_indices"]),
            witness_found=bool(ledger["witness_indices"]),
            selected_trace_index=ledger["selected_trace_index"],
            closest_safety_trace_index=closest,
            initial_prefix_length=ledger["initial_prefix_length"],
            selected_prefix_length=ledger["selected_prefix_length"],
            total_iterations=ledger["total_iterations"],
            phase_checks=ledger["phase_checks"],
            termination_reason=ledger["termination_reason"],
            warm_admission=warm_admission_status(classified[0]),
            native_quaternion_jit_signatures=[str(value) for value in quat2mat_jit.nopython_signatures],
            source_refs_unchanged=list(references.values()),
            evidence="Every fixed-case response reconstructed independently; nominal-time and retimed path remain distinct, without physical/control qualification",
        )
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        publish(root / "verified.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("early", "late"), required=True)
    main(parser.parse_args().case)
