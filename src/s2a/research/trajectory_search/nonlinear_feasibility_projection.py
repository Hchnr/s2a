"""Finite nonlinear safety projection of a complete H12 command trajectory."""

import hashlib

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from s2a.geometry import finite


def project(evaluate, warm, previous, input_min, input_max, *, max_evaluations=320, max_iterations=4, ledger=None):
    """Numerical unsafe iterates are observations; only full buffered audits qualify."""
    warm = finite(warm, (12, 6)).copy()
    previous, lo, hi = (finite(v, (6,)) for v in (previous, input_min, input_max))
    if not np.all(lo < hi) or type(max_evaluations) is not int or not 1 <= max_evaluations <= 320:
        raise ValueError("Strict input domain and bounded response budget")
    if type(max_iterations) is not int or not 1 <= max_iterations <= 4:
        raise ValueError("Bounded iteration budget")
    lower, upper = np.tile(lo, 12), np.tile(hi, 12)
    lower[:6] = np.maximum(lower[:6], previous - 0.25)
    upper[:6] = np.minimum(upper[:6], previous + 0.25)
    if np.any(upper - lower <= 1e-4):
        raise ValueError("Nondegenerate finite difference box")
    start = np.clip(warm.ravel(), lower, upper)
    if np.max(np.abs(start - warm.ravel())) > 1e-12:
        raise ValueError("Only registered roundoff repair of supplied warm")
    D = np.eye(72) - np.eye(72, k=-6)
    center = np.r_[previous, np.zeros(66)]
    if np.max(np.abs(D @ start - center)) > 0.25 + 1e-12:
        raise ValueError("Warm must satisfy original interperiod rates")
    state = {} if ledger is None else ledger
    state.update(
        supplied_warm=warm,
        used_warm=start.reshape(12, 6),
        warm_roundoff=start - warm.ravel(),
        lower=lower,
        upper=upper,
        requests=[],
        accesses=[],
        derivatives=[],
        callbacks=[],
        objective_calls=[],
        gradient_calls=[],
        response_evaluations=0,
        max_evaluations=max_evaluations,
        max_iterations=max_iterations,
        solver_success=False,
        status="running",
        selected_index=None,
        safe_indices=[],
        witness_indices=[],
    )
    cache, entries = {}, []
    derivative_key, derivative_value = None, None

    class BudgetExhausted(Exception):
        pass

    def get(x, role):
        x = finite(x, (72,)).copy()
        key = x.tobytes()
        if key not in cache:
            if len(state["requests"]) >= max_evaluations:
                raise BudgetExhausted
            if np.any(x < lower - 1e-12) or np.any(x > upper + 1e-12):
                raise ValueError("Out-of-box response request")
            row = dict(
                index=len(state["requests"]),
                commands=x.reshape(12, 6),
                role=role,
                completed=False,
                rate_admitted=bool(np.all(np.abs(D @ x - center) <= 0.25 + 1e-12)),
            )
            state["requests"].append(row)
            entry = evaluate(x.reshape(12, 6).copy())
            if not np.array_equal(finite(entry["commands"], (12, 6)), x.reshape(12, 6)):
                raise ValueError("Response command identity")
            finite(entry["safety_slack"], (4369,))
            cache[key] = len(entries)
            entries.append(entry)
            row["completed"] = True
            state["response_evaluations"] = len(entries)
        index = cache[key]
        state["accesses"].append(dict(index=index, role=role))
        return index, entries[index]

    def objective(x):
        d = x - warm.ravel()
        value = float(0.5 * (d @ d))
        state["objective_calls"].append(dict(commands=np.asarray(x).reshape(12, 6).copy(), value=value))
        return value

    def gradient(x):
        value = x - warm.ravel()
        state["gradient_calls"].append(dict(commands=np.asarray(x).reshape(12, 6).copy(), gradient=value.copy()))
        return value

    def constraint(x):
        return np.asarray(get(x, "constraint")[1]["safety_slack"]) - 1e-8

    def jacobian(x):
        nonlocal derivative_key, derivative_value
        key = np.asarray(x).tobytes()
        if key != derivative_key:
            base_index, base = get(x, "derivative_base")
            record = dict(base_index=base_index, probes=[], complete=False)
            state["derivatives"].append(record)
            columns = []
            for axis in range(72):
                delta = 1e-4 if x[axis] + 1e-4 <= upper[axis] else -1e-4
                probe = np.asarray(x).copy()
                probe[axis] += delta
                index, entry = get(probe, "derivative_probe")
                columns.append((np.asarray(entry["safety_slack"]) - np.asarray(base["safety_slack"])) / delta)
                record["probes"].append(dict(axis=axis, nominal_delta=delta, index=index))
            derivative_value = np.asarray(columns).T
            derivative_key = key
            record.update(complete=True, jacobian_sha256=hashlib.sha256(derivative_value.tobytes()).hexdigest())
        return derivative_value

    def callback(x):
        state["callbacks"].append(get(x, "callback")[0])

    get(start, "initial")
    try:
        answer = minimize(
            objective,
            start,
            jac=gradient,
            method="SLSQP",
            bounds=list(zip(lower, upper, strict=True)),
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
    for i, entry in enumerate(entries):
        if entry["audit"]["feasible"] and np.all(np.asarray(entry["safety_slack"]) >= 1e-8):
            state["safe_indices"].append(i)
            if np.all(np.asarray(entry["normalized_squared_errors"]) <= 1):
                state["witness_indices"].append(i)
    candidates = state["witness_indices"] or state["safe_indices"]
    if candidates:
        state["selected_index"] = min(
            candidates, key=lambda i: (float(np.sum((np.asarray(entries[i]["commands"]) - warm) ** 2)), i)
        )
    state["complete"] = True
    return state
