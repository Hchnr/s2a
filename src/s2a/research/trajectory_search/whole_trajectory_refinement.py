"""Nonlinear H12 refinement that protects every measured reference component."""

import hashlib

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from s2a.geometry import finite
from s2a.research.trajectory_search.cap_distance_objective import cap_loss


def loss_weights(q):
    """Derivative of the original distance-to-cap loss in squared-error units."""
    q = finite(q, (24,))
    if np.any(q < 0):
        raise ValueError("Squared errors cannot be negative")
    result = np.zeros(24)
    outside = q > 1
    result[outside] = 0.5 * (1 - 1 / np.sqrt(q[outside]))
    return result


def common_start(warm, previous, input_min, input_max, root_lower, root_upper):
    """Use one byte-identical warm point in both registered command domains."""
    warm = finite(warm, (12, 6)).copy()
    previous, lo, hi = (finite(v, (6,)) for v in (previous, input_min, input_max))
    lo, hi = np.tile(lo, (12, 1)), np.tile(hi, (12, 1))
    lo[0], hi[0] = np.maximum(lo[0], previous - 0.25), np.minimum(hi[0], previous + 0.25)
    lower, upper = np.maximum(lo, finite(root_lower, (12, 6))), np.minimum(hi, finite(root_upper, (12, 6)))
    if np.any(lower >= upper):
        raise ValueError("Common root domain must be nondegenerate")
    start = np.clip(warm, lower, upper)
    if np.max(np.abs(start - warm)) > 1e-12:
        raise ValueError("Common warm may receive only roundoff correction")
    if np.any(np.abs(np.diff(np.vstack((previous, start)), axis=0)) > 0.25 + 1e-12):
        raise ValueError("Common warm must satisfy original rate constraints")
    return start


def refine(
    evaluate, warm, previous, lower, upper, *, max_evaluations=640, max_iterations=8, ledger=None, event_sink=None
):
    """Search one fixed command domain; all candidate gates use actual responses."""
    warm, previous = finite(warm, (12, 6)).copy(), finite(previous, (6,)).copy()
    lo, hi = finite(lower, (12, 6)).ravel().copy(), finite(upper, (12, 6)).ravel().copy()
    if type(max_evaluations) is not int or not 1 <= max_evaluations <= 640:
        raise ValueError("Bounded response budget")
    if type(max_iterations) is not int or not 1 <= max_iterations <= 8:
        raise ValueError("Bounded iteration budget")
    if np.any(hi - lo <= 1e-4) or np.any(warm.ravel() < lo) or np.any(warm.ravel() > hi):
        raise ValueError("Exact box-admitted warm and nondegenerate FD domain required")
    D = np.eye(72) - np.eye(72, k=-6)
    center = np.r_[previous, np.zeros(66)]
    if np.any(np.abs(D @ warm.ravel() - center) > 0.25 + 1e-12):
        raise ValueError("Original first and adjacent rate constraints")
    state = {} if ledger is None else ledger
    state.update(
        used_warm=warm,
        lower=lo,
        upper=hi,
        max_evaluations=max_evaluations,
        max_iterations=max_iterations,
        requests=[],
        accesses=[],
        derivatives=[],
        objective_calls=[],
        gradient_calls=[],
        callbacks=[],
        response_evaluations=0,
        status="running",
        solver_success=False,
        selected_index=None,
        eligible_indices=[],
        witness_indices=[],
    )
    cache, entries = {}, []
    derivative_key, derivative_value, derivative_index = None, None, None

    def emit(kind, **values):
        if event_sink is not None:
            event_sink(dict(kind=kind, **values))

    class BudgetExhausted(Exception):
        pass

    def get(x, role):
        x = finite(x, (72,)).copy()
        key = x.tobytes()
        if key not in cache:
            if len(state["requests"]) >= max_evaluations:
                raise BudgetExhausted
            if np.any(x < lo - 1e-12) or np.any(x > hi + 1e-12):
                raise ValueError("Out-of-box response request")
            index = len(state["requests"])
            identity = hashlib.sha256(key).hexdigest()
            row = dict(
                index=index,
                commands=x.reshape(12, 6),
                command_sha256=identity,
                role=role,
                completed=False,
                rate_admitted=bool(np.all(np.abs(D @ x - center) <= 0.25 + 1e-12)),
            )
            state["requests"].append(row)
            emit("request", index=index, command_sha256=identity, role=role, rate_admitted=row["rate_admitted"])
            entry = evaluate(x.reshape(12, 6).copy())
            if not np.array_equal(finite(entry["commands"], (12, 6)), x.reshape(12, 6)):
                raise ValueError("Exact command identity required")
            finite(entry["safety_slack"], (4369,))
            q = finite(entry["normalized_squared_errors"], (12, 2))
            if np.any(q < 0):
                raise ValueError("Nonnegative full reference errors")
            cache[key] = len(entries)
            entries.append(entry)
            row["completed"] = True
            state["response_evaluations"] = len(entries)
            emit("response_completed", index=index, command_sha256=identity)
        index = cache[key]
        state["accesses"].append(dict(index=index, role=role))
        return index, entries[index]

    _, initial = get(warm.ravel(), "initial")
    initial_q = np.asarray(initial["normalized_squared_errors"]).ravel()
    if not initial["audit"]["feasible"] or np.any(np.asarray(initial["safety_slack"]) < 1e-8):
        raise ValueError("Initial actual response must pass original buffered safety")
    ceiling = np.maximum(1.0, initial_q)
    state["q_ceiling"] = ceiling

    def objective(x):
        index, entry = get(x, "objective")
        value = cap_loss(np.asarray(entry["normalized_squared_errors"]).ravel())
        state["objective_calls"].append(dict(index=index, value=value))
        return value

    def derivatives(x):
        nonlocal derivative_key, derivative_value, derivative_index
        key = np.asarray(x).tobytes()
        if key != derivative_key:
            base_index, base = get(x, "derivative_base")
            record = dict(base_index=base_index, probes=[], complete=False)
            derivative_index = len(state["derivatives"])
            state["derivatives"].append(record)
            emit("derivative_start", derivative_index=derivative_index, base_index=base_index)
            ds, dq = [], []
            for axis in range(72):
                delta = 1e-4 if x[axis] + 1e-4 <= hi[axis] else -1e-4
                probe = np.asarray(x).copy()
                probe[axis] += delta
                index, entry = get(probe, "derivative_probe")
                ds.append((np.asarray(entry["safety_slack"]) - np.asarray(base["safety_slack"])) / delta)
                dq.append(
                    (
                        np.asarray(entry["normalized_squared_errors"]).ravel()
                        - np.asarray(base["normalized_squared_errors"]).ravel()
                    )
                    / delta
                )
                row = dict(axis=axis, nominal_delta=delta, index=index)
                record["probes"].append(row)
                emit("derivative_probe", derivative_index=derivative_index, **row)
            Js, Jq = np.asarray(ds).T, np.asarray(dq).T
            derivative_key, derivative_value = key, (Js, Jq)
            record.update(
                complete=True,
                safety_jacobian_sha256=hashlib.sha256(Js.tobytes()).hexdigest(),
                q_jacobian_sha256=hashlib.sha256(Jq.tobytes()).hexdigest(),
            )
            emit(
                "derivative_complete",
                derivative_index=derivative_index,
                safety_jacobian_sha256=record["safety_jacobian_sha256"],
                q_jacobian_sha256=record["q_jacobian_sha256"],
            )
        return derivative_value

    def gradient(x):
        index, entry = get(x, "gradient")
        _, Jq = derivatives(x)
        value = loss_weights(np.asarray(entry["normalized_squared_errors"]).ravel()) @ Jq
        state["gradient_calls"].append(dict(index=index, derivative_index=derivative_index, gradient=value.copy()))
        return value

    def constraint(x):
        _, entry = get(x, "constraint")
        return np.r_[
            np.asarray(entry["safety_slack"]) - 1e-8, ceiling - np.asarray(entry["normalized_squared_errors"]).ravel()
        ]

    def jacobian(x):
        Js, Jq = derivatives(x)
        return np.vstack((Js, -Jq))

    def callback(x):
        index, _ = get(x, "callback")
        state["callbacks"].append(index)
        emit("callback", index=index, count=len(state["callbacks"]))

    try:
        answer = minimize(
            objective,
            warm.ravel(),
            jac=gradient,
            method="SLSQP",
            bounds=list(zip(lo, hi, strict=True)),
            constraints=[
                LinearConstraint(D, center - 0.25, center + 0.25),
                dict(type="ineq", fun=constraint, jac=jacobian),
            ],
            callback=callback,
            options=dict(maxiter=max_iterations, ftol=1e-12, disp=False),
        )
        state.update(
            solver_success=bool(answer.success),
            status=int(answer.status),
            message=str(answer.message),
            iterations=int(answer.nit),
            solver_return_index=get(answer.x, "solver_return")[0],
        )
    except BudgetExhausted:
        state.update(status="response_budget", iterations=len(state["callbacks"]))
    for index, entry in enumerate(entries):
        q = np.asarray(entry["normalized_squared_errors"]).ravel()
        if entry["audit"]["feasible"] and np.all(np.asarray(entry["safety_slack"]) >= 1e-8) and np.all(q <= ceiling):
            state["eligible_indices"].append(index)
            if np.all(q <= 1):
                state["witness_indices"].append(index)
    candidates = state["witness_indices"] or state["eligible_indices"]
    if candidates:
        state["selected_index"] = min(
            candidates,
            key=lambda i: (
                cap_loss(np.asarray(entries[i]["normalized_squared_errors"]).ravel()),
                float(np.max(entries[i]["normalized_squared_errors"])),
                i,
            ),
        )
    state["complete"] = True
    emit(
        "solver_closed", status=state["status"], iterations=state["iterations"], selected_index=state["selected_index"]
    )
    return state
