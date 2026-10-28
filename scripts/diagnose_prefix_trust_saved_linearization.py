"""Audit saved H12 finite differences without a solver or new model response.

The latest completed same-stage derivative is an observable ledger association,
not a reconstruction of SLSQP's complete QP. The separate stage-anchor model is
only a fixed-base comparison. No simulator or production helper is imported.
"""

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / "plans/v0.0.1/prefix_trust_saved_linearization_20261029.md"
HORIZON, WIDTH, TARGET, STEP = 12, 72, 9, 1e-4
ROLES = {
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


def require(condition, message):
    if not condition:
        raise ValueError(message)


def array(value, shape):
    value = np.asarray(value, dtype=np.float64)
    require(value.shape == shape and np.all(np.isfinite(value)), f"Finite array of shape {shape} required")
    return value


def command_key(value):
    return array(value, (HORIZON, 6)).tobytes().hex()


def decode_key(key):
    require(isinstance(key, str), "String float64 command key required")
    try:
        commands = np.frombuffer(bytes.fromhex(key), dtype=np.float64).reshape(HORIZON, 6)
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid full command byte key") from error
    require(command_key(commands) == key, "Canonical finite command bytes required")
    return commands


def normalized_errors(errors):
    errors = array(errors, (HORIZON, 6))
    q = np.stack((np.sum(errors[:, :3] ** 2, axis=1) / 0.01**2, np.sum(errors[:, 3:] ** 2, axis=1) / 0.05**2), axis=1)
    return array(q, (HORIZON, 2))


def prefix_length(q):
    failures = np.flatnonzero(np.any(q > 1.0, axis=1))
    return int(failures[0]) if len(failures) else HORIZON


def integer(value, lower, upper, label):
    require(type(value) is int and lower <= value <= upper, label)
    return value


def validate_ledger(producer):
    """Validate saved request identity/accounting; do not certify dynamics."""
    require(
        producer["complete"] is producer["passed"] is True and producer["case_name"] == "candidate",
        "One closed candidate producer archive required",
    )
    answer = producer["answer"]
    require(answer["horizon"] == HORIZON and answer["finite_difference_step"] == STEP, "Registered H12 and FD step")
    trace, phases, ledger = answer["evaluated_candidates"], answer["phases"], answer["request_ledger"]
    require(isinstance(trace, list) and trace and isinstance(phases, list) and isinstance(ledger, list), "Saved arrays")
    require(
        len(trace) == producer["response_evaluations"] == answer["response_evaluations"],
        "Complete saved response count",
    )
    keys, qvalues = [], []
    for row in trace:
        keys.append(command_key(row["commands"]))
        q = normalized_errors(row["tcp_errors"])
        require(
            type(row["prefix_length"]) is int and row["prefix_length"] == prefix_length(q), "Strict saved pose prefix"
        )
        require(type(row["joint_feasible"]) is bool and type(row["feasible"]) is bool, "Boolean saved audit labels")
        require(row["feasible"] == (row["joint_feasible"] and prefix_length(q) == HORIZON), "Saved full-witness label")
        require(np.isfinite(row["safety_min"]), "Finite saved safety minimum")
        qvalues.append(q)
    require(len(set(keys)) == len(keys), "Each trace row is one unique command byte string")
    lower, upper = array(answer["lower_command"], (HORIZON, 6)), array(answer["upper_command"], (HORIZON, 6))
    require(np.all(lower <= upper), "Ordered original bounds")
    next_request = 1
    for stage, phase in enumerate(phases):
        require(type(phase["stage_index"]) is int and phase["stage_index"] == stage, "Contiguous stages")
        require(phase["requests_before"] == next_request, "Contiguous stage request intervals")
        next_request = integer(phase["requests_after"], next_request + 1, len(ledger), "Nonempty stage interval")
        prefix = integer(phase["prefix_before"], 0, HORIZON - 1, "An incomplete hard prefix")
        require(phase["target_cursor"] == prefix, "Stage target is its first failing cursor")
        require(np.isfinite(phase["radius"]) and phase["radius"] > 0, "Finite positive saved radius")
        lo, hi = array(phase["stage_lower_command"], (HORIZON, 6)), array(phase["stage_upper_command"], (HORIZON, 6))
        require(np.all(hi - lo > STEP) and np.all(lo >= lower) and np.all(hi <= upper), "Admitted stage bounds")
        anchor = integer(phase["anchor_before_index"], 0, len(trace) - 1, "Existing anchor response")
        require(command_key(phase["initial_commands"]) == keys[anchor], "Exact stage anchor command bytes")
        require(prefix_length(qvalues[anchor]) == prefix, "Anchor has the declared contiguous prefix")
    require(next_request == len(ledger), "Every non-warm request belongs to exactly one stage")
    seen, commands = {}, []
    for index, request in enumerate(ledger):
        require(
            type(request["request_index"]) is int and request["request_index"] == index, "Contiguous request indices"
        )
        require(request["role"] in ROLES, "Known request role")
        integer(request["response_evaluations_before"], len(seen), len(seen), "Exact prior response count")
        key = request["command_key"]
        command = decode_key(key)
        commands.append(command)
        if index == 0:
            require(request["stage_index"] is None and request["role"] == "warm", "Initial warm request")
            lo, hi = lower, upper
        else:
            stage = integer(request["stage_index"], 0, len(phases) - 1, "Known stage")
            phase = phases[stage]
            require(phase["requests_before"] <= index < phase["requests_after"], "Same-stage request interval")
            require(request["role"] != "warm", "Only one warm request")
            lo, hi = np.asarray(phase["stage_lower_command"]), np.asarray(phase["stage_upper_command"])
        require(np.all(command >= lo - 1e-12) and np.all(command <= hi + 1e-12), "Saved request respects stage bounds")
        outcome = request["outcome"]
        if outcome == "new_response":
            response = integer(request["response_index"], len(seen), len(seen), "New response order")
            require(
                key not in seen
                and response < len(trace)
                and keys[response] == key
                and type(request["new_response_index"]) is int
                and request["new_response_index"] == response,
                "New request matches its exact trace bytes",
            )
            seen[key] = response
        elif outcome == "cache_hit":
            require(
                key in seen
                and type(request["response_index"]) is int
                and request["response_index"] == seen[key]
                and request["new_response_index"] is None,
                "Cache request references an earlier exact response",
            )
        else:
            require(
                outcome == "budget_rejected"
                and index == len(ledger) - 1
                and key not in seen
                and len(seen) == answer["max_evaluations"]
                and request["response_index"] is request["new_response_index"] is None,
                "Only a final explicit budget rejection can lack a response",
            )
        integer(request["response_evaluations_after"], len(seen), len(seen), "Exact subsequent response count")
        if not request["role"].endswith("_probe"):
            require(request["axis"] is request["nominal_delta"] is None, "Only FD probes carry axis/delta")
    require(len(seen) == len(trace) and ledger[0]["outcome"] == "new_response", "Every saved response was requested")
    for stage, phase in enumerate(phases):
        first, end = phase["requests_before"], phase["requests_after"]
        require(
            ledger[first]["role"] == "stage_anchor"
            and ledger[first]["response_index"] == phase["anchor_before_index"]
            and sum(row["role"] == "stage_anchor" for row in ledger[first:end]) == 1,
            "Each stage starts with its exact already available anchor",
        )
        callbacks = [index for index in range(first, end) if ledger[index]["role"] == "callback"]
        require(
            type(phase["callback_count"]) is int and phase["callback_count"] == len(callbacks), "Saved callback count"
        )
        if callbacks:
            require(
                command_key(phase["last_callback_commands"]) == ledger[callbacks[-1]]["command_key"],
                "Saved final callback matches the ledger",
            )
    return answer, keys, qvalues, commands


def finite_difference_groups(answer, qvalues, commands):
    ledger, phases = answer["request_ledger"], answer["phases"]
    groups, covered = [], set()
    for index, request in enumerate(ledger):
        if request["role"] not in ("gradient_base", "jacobian_base"):
            continue
        stage = request["stage_index"]
        group = dict(
            group_index=len(groups),
            stage_index=stage,
            role=request["role"],
            base_ledger_index=index,
            base_response_index=request["response_index"],
            probe_ledger_indices=[],
            probe_response_indices=[],
            nominal_deltas=[],
            complete=False,
            end_ledger_index=index,
        )
        groups.append(group)
        if request["outcome"] == "budget_rejected":
            group["incomplete_reason"] = "base_budget_rejected"
            continue
        base = commands[index].ravel()
        hi = np.asarray(phases[stage]["stage_upper_command"]).ravel()
        columns = []
        for axis in range(WIDTH):
            probe_index = index + axis + 1
            require(probe_index < len(ledger), "Truncated FD group lacks an explicit budget rejection")
            probe = ledger[probe_index]
            require(
                probe["stage_index"] == stage
                and probe["role"] == request["role"].replace("_base", "_probe")
                and type(probe["axis"]) is int
                and probe["axis"] == axis,
                "Complete contiguous same-stage axis-ordered FD group required",
            )
            delta = STEP if base[axis] + STEP <= hi[axis] else -STEP
            expected = base.copy()
            expected[axis] += delta
            require(
                probe["nominal_delta"] == delta and expected.tobytes().hex() == probe["command_key"],
                "FD probe differs from its exact nominal one-coordinate command",
            )
            covered.add(probe_index)
            group["probe_ledger_indices"].append(probe_index)
            group["probe_response_indices"].append(probe["response_index"])
            group["nominal_deltas"].append(delta)
            group["end_ledger_index"] = probe_index
            if probe["outcome"] == "budget_rejected":
                group["incomplete_reason"] = "probe_budget_rejected"
                break
            columns.append(
                (qvalues[probe["response_index"]].ravel() - qvalues[request["response_index"]].ravel()) / delta
            )
        else:
            group["complete"] = True
            group["q_base"] = qvalues[request["response_index"]].ravel()
            group["jacobian"] = array(np.asarray(columns).T, (24, WIDTH))
    require(
        covered == {i for i, row in enumerate(ledger) if row["role"].endswith("_probe")},
        "Every saved FD probe belongs to exactly one validated group",
    )
    return groups


def model_comparison(group, callback_index, phase, answer, qvalues, commands):
    ledger = answer["request_ledger"]
    q_base = qvalues[group["base_response_index"]]
    q_actual = qvalues[ledger[callback_index]["response_index"]]
    displacement = commands[callback_index].ravel() - commands[group["base_ledger_index"]].ravel()
    q_linear = (q_base.ravel() + group["jacobian"] @ displacement).reshape(HORIZON, 2)
    array(q_linear, (HORIZON, 2))
    residual = q_actual - q_linear
    prefix, target = phase["prefix_before"], phase["target_cursor"]
    result = dict(
        fd_group_index=group["group_index"],
        base_ledger_index=group["base_ledger_index"],
        fd_end_ledger_index=group["end_ledger_index"],
        base_response_index=group["base_response_index"],
        zero_step=bool(np.all(displacement == 0)),
        command_bytes_equal=ledger[callback_index]["command_key"] == ledger[group["base_ledger_index"]]["command_key"],
        step_linf=float(np.max(abs(displacement))),
        step_l2=float(np.linalg.norm(displacement)),
        q_base=q_base.ravel(),
        q_linear=q_linear.ravel(),
        q_actual=q_actual.ravel(),
        residual=residual.ravel(),
        prefix_q_base=q_base[:prefix],
        prefix_q_linear=q_linear[:prefix],
        prefix_q_actual=q_actual[:prefix],
        prefix_linear_max_q_minus_one=float(np.max(q_linear[:prefix] - 1.0)) if prefix else None,
        prefix_actual_max_q_minus_one=float(np.max(q_actual[:prefix] - 1.0)) if prefix else None,
        prefix_residual_max_abs=float(np.max(abs(residual[:prefix]))) if prefix else None,
        prefix_linear_strict_pass=bool(np.all(q_linear[:prefix] <= 1.0)),
        prefix_actual_strict_pass=bool(np.all(q_actual[:prefix] <= 1.0)),
        linear_negative_component_count=int(np.sum(q_linear < 0)),
    )
    for label, cursor in (("original_target", TARGET), ("stage_target", target)):
        base, linear, actual = (float(np.max(q[cursor])) for q in (q_base, q_linear, q_actual))
        result[label] = dict(
            cursor=cursor,
            q_base=q_base[cursor],
            q_linear=q_linear[cursor],
            q_actual=q_actual[cursor],
            base_score=base,
            linear_score=linear,
            actual_score=actual,
            predicted_improvement=base - linear,
            actual_improvement=base - actual,
        )
    return result


def summarize(callbacks):
    models = [row["latest_complete_fd"] for row in callbacks if row["latest_complete_fd"] is not None]
    nonzero = [row for row in models if not row["zero_step"]]
    result = dict(
        callback_count=len(callbacks),
        matched_count=len(models),
        zero_step_count=len(models) - len(nonzero),
        nonzero_step_count=len(nonzero),
    )
    for label, values in (("all_matched", models), ("nonzero_steps", nonzero)):
        metrics = dict(
            strict_linear_prefix_pass_count=sum(row["prefix_linear_strict_pass"] for row in values),
            strict_actual_prefix_pass_count=sum(row["prefix_actual_strict_pass"] for row in values),
        )
        for key in ("prefix_linear_max_q_minus_one", "prefix_actual_max_q_minus_one", "prefix_residual_max_abs"):
            scalars = [row[key] for row in values if row[key] is not None]
            metrics[key] = dict(min=min(scalars), max=max(scalars)) if scalars else None
        result[label] = metrics
    return result


def diagnose_saved_linearization(producer):
    answer, keys, qvalues, commands = validate_ledger(producer)
    groups = finite_difference_groups(answer, qvalues, commands)
    callbacks, gaps = [], []
    for index, request in enumerate(answer["request_ledger"]):
        if request["role"] != "callback":
            continue
        stage = request["stage_index"]
        phase = answer["phases"][stage]
        available = [
            group
            for group in groups
            if group["complete"] and group["stage_index"] == stage and group["end_ledger_index"] < index
        ]
        anchored = [group for group in available if group["base_response_index"] == phase["anchor_before_index"]]
        row = dict(
            stage_index=stage,
            radius=phase["radius"],
            ledger_index=index,
            response_index=request["response_index"],
            outcome=request["outcome"],
            prefix_before=phase["prefix_before"],
            stage_target_cursor=phase["target_cursor"],
            original_target_cursor=TARGET,
            stage_anchor_response_index=phase["anchor_before_index"],
            latest_complete_fd=None,
            stage_anchor_fd=None,
        )
        if request["response_index"] is None:
            gaps.append(dict(ledger_index=index, reason="callback_budget_rejected_no_actual_response"))
        else:
            actual = answer["evaluated_candidates"][request["response_index"]]
            row.update(
                actual_q=qvalues[request["response_index"]].ravel(),
                actual_prefix_length=actual["prefix_length"],
                actual_joint_feasible=actual["joint_feasible"],
                actual_safety_min=actual["safety_min"],
            )
            for key, candidates, choose in (("latest_complete_fd", available, -1), ("stage_anchor_fd", anchored, 0)):
                if candidates:
                    row[key] = model_comparison(candidates[choose], index, phase, answer, qvalues, commands)
                else:
                    gaps.append(dict(ledger_index=index, reason=f"no_prior_same_stage_{key}"))
        callbacks.append(row)
    stages = [
        dict(
            stage_index=index,
            radius=phase["radius"],
            anchor_response_index=phase["anchor_before_index"],
            requests_before=phase["requests_before"],
            requests_after=phase["requests_after"],
            summary=summarize([row for row in callbacks if row["stage_index"] == index]),
        )
        for index, phase in enumerate(answer["phases"])
    ]
    return dict(
        schema_version="s2a.saved-prefix-linearization.v1",
        horizon=HORIZON,
        command_axes=WIDTH,
        normalized_components=24,
        component_order="cursor0 position, cursor0 rotation, ..., cursor11 rotation",
        residual_definition="q_actual - q_linear; negative linear components are retained",
        pose_caps=dict(position_m=0.01, rotation_rad=0.05),
        finite_difference_step=STEP,
        complete_fd_group_count=sum(group["complete"] for group in groups),
        incomplete_fd_group_count=sum(not group["complete"] for group in groups),
        saved_response_count=len(keys),
        saved_request_count=len(answer["request_ledger"]),
        groups=groups,
        callbacks=callbacks,
        stages=stages,
        summary=summarize(callbacks),
        data_gaps=gaps,
        evidence_scope="Arithmetic diagnosis of producer-saved responses; independent dynamics qualification is not asserted",
        latest_model_scope="Latest fully completed same-stage FD group preceding each callback; not a complete QP reconstruction",
        anchor_model_scope="Earliest completed same-stage group at its anchor, available before callback; not necessarily the optimizer current model",
        safety_scope="Saved scalar safety minimum only; no full safety Jacobian or joint-specific attribution",
        floating_point_scope="Raw signed q-1 and residuals; no diagnostic tolerance changes any strict q<=1 classification",
        zero_step_scope="Retained in all_matched; excluded from nonzero_steps residual summary",
        independent_numerical_qualified=False,
        control_qualified=False,
        task_qualified=False,
        new_responses=0,
        query_count=0,
        numerical_inner_steps=0,
        physics_steps=0,
        model_calls=0,
        world_model_calls=0,
        fresh_cases=0,
    )


def reference(path):
    path = Path(path).resolve()
    require(path.is_relative_to(REPO), "Evidence and output must stay in the current repository")
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main(input_path, input_sha256, output):
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "", "Explicit CPU environment"
    )
    output = Path(output).resolve()
    require(output.is_relative_to(REPO) and not output.exists(), "Exclusive new repository JSON output")
    input_ref, source_ref, plan_ref = reference(input_path), reference(__file__), reference(PLAN)
    require(input_ref["sha256"] == input_sha256, "Input SHA differs from supplied registration")
    payload = Path(input_ref["path"]).read_bytes()
    require(hashlib.sha256(payload).hexdigest() == input_sha256, "Input changed during read")
    result = diagnose_saved_linearization(json.loads(payload))
    for ref in (input_ref, source_ref, plan_ref):
        require(reference(ref["path"]) == ref, "Input/source/plan changed during diagnosis")
    result.update(
        complete=True,
        passed=True,
        input_ref=input_ref,
        source_ref=source_ref,
        plan_ref=plan_ref,
        runtime=dict(python=platform.python_version(), numpy=np.__version__),
    )

    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False, default=convert)
        stream.write("\n")
    print(
        json.dumps(
            dict(complete=True, passed=True, output_ref=reference(output), summary=result["summary"], new_responses=0)
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.input, args.input_sha256, args.output)
