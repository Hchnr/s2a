"""Reconstruct full-trajectory refinement decisions from saved actual responses."""

import hashlib

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def independent_loss(q):
    q = np.asarray(q, dtype=float).ravel()
    require(q.shape == (24,) and np.isfinite(q).all() and np.all(q >= 0), "Finite original 24 squared errors")
    r = np.zeros(24)
    outside = q > 1
    r[outside] = (q[outside] - 1) / (np.sqrt(q[outside]) + 1)
    return float(0.5 * np.sum(r**2))


def audit_refinement(ledger, entries, previous, lower, upper, events):
    requests = ledger["requests"]
    require(
        ledger["complete"]
        and len(requests) == len(entries) == ledger["response_evaluations"]
        and len(entries) <= ledger["max_evaluations"] <= 640,
        "Complete bounded response accounting",
    )
    require(ledger["iterations"] <= ledger["max_iterations"] <= 8, "Bounded optimizer iterations")
    np.testing.assert_array_equal(np.asarray(ledger["lower"]).reshape(12, 6), lower)
    np.testing.assert_array_equal(np.asarray(ledger["upper"]).reshape(12, 6), upper)
    np.testing.assert_array_equal(entries[0]["commands"], ledger["used_warm"])
    ceiling = np.maximum(1.0, np.asarray(entries[0]["normalized_squared_errors"]).ravel())
    np.testing.assert_array_equal(ceiling, ledger["q_ceiling"])
    seen = set()
    for i, (row, entry) in enumerate(zip(requests, entries, strict=True)):
        x = np.asarray(entry["commands"])
        require(x.shape == (12, 6) and row["index"] == i and row["completed"], "Complete ordered response")
        np.testing.assert_array_equal(row["commands"], x)
        identity = hashlib.sha256(x.ravel().tobytes()).hexdigest()
        require(identity == row["command_sha256"] and identity not in seen, "Unique command-byte identity")
        seen.add(identity)
        require(np.all(x >= np.asarray(lower) - 1e-12) and np.all(x <= np.asarray(upper) + 1e-12), "Original box")
        rate = bool(np.all(np.abs(np.diff(np.vstack((previous, x)), axis=0)) <= 0.25 + 1e-12))
        require(rate == row["rate_admitted"], "Original first/adjacent rate labels")
        error = np.asarray(entry["tcp_errors"])
        q = np.column_stack((np.sum(error[:, :3] ** 2, axis=1) / 0.01**2, np.sum(error[:, 3:] ** 2, axis=1) / 0.05**2))
        np.testing.assert_array_equal(q, entry["normalized_squared_errors"])
    for access in ledger["accesses"]:
        require(0 <= access["index"] < len(entries), "Cache access references actual charged response")
    derivative_matrices = []
    hi = np.asarray(upper).ravel()
    for record in ledger["derivatives"]:
        base = entries[record["base_index"]]
        x = np.asarray(base["commands"]).ravel()
        ds, dq = [], []
        for axis, probe in enumerate(record["probes"]):
            delta = 1e-4 if x[axis] + 1e-4 <= hi[axis] else -1e-4
            require(probe["axis"] == axis and probe["nominal_delta"] == delta, "Every signed probe and order")
            other = entries[probe["index"]]
            expected = x.copy()
            expected[axis] += delta
            np.testing.assert_array_equal(expected.reshape(12, 6), other["commands"])
            ds.append((np.asarray(other["safety_slack"]) - np.asarray(base["safety_slack"])) / delta)
            dq.append(
                (
                    np.asarray(other["normalized_squared_errors"]).ravel()
                    - np.asarray(base["normalized_squared_errors"]).ravel()
                )
                / delta
            )
        if record["complete"]:
            require(len(ds) == 72, "Every derivative column is charged")
            Js, Jq = np.asarray(ds).T, np.asarray(dq).T
            require(
                hashlib.sha256(Js.tobytes()).hexdigest() == record["safety_jacobian_sha256"]
                and hashlib.sha256(Jq.tobytes()).hexdigest() == record["q_jacobian_sha256"],
                "Fresh derivative identity",
            )
            derivative_matrices.append(Jq)
        else:
            require(ledger["status"] == "response_budget" and len(ds) < 72, "Only budget may truncate a derivative")
            derivative_matrices.append(None)
    for call in ledger["objective_calls"]:
        require(
            call["value"] == independent_loss(entries[call["index"]]["normalized_squared_errors"]),
            "Actual complete objective",
        )
    for call in ledger["gradient_calls"]:
        record = ledger["derivatives"][call["derivative_index"]]
        require(record["complete"] and record["base_index"] == call["index"], "Gradient uses current base")
        q = np.asarray(entries[call["index"]]["normalized_squared_errors"]).ravel()
        weights = np.zeros(24)
        outside = q > 1
        weights[outside] = 0.5 * (1 - 1 / np.sqrt(q[outside]))
        np.testing.assert_array_equal(weights @ derivative_matrices[call["derivative_index"]], call["gradient"])
    require(
        len(ledger["callbacks"]) <= ledger["max_iterations"]
        and all(0 <= i < len(entries) for i in ledger["callbacks"]),
        "Paid bounded callback records",
    )
    eligible = []
    witnesses = []
    for i, entry in enumerate(entries):
        q = np.asarray(entry["normalized_squared_errors"]).ravel()
        if entry["audit"]["feasible"] and np.all(np.asarray(entry["safety_slack"]) >= 1e-8) and np.all(q <= ceiling):
            eligible.append(i)
            if np.all(q <= 1):
                witnesses.append(i)
    require(
        ledger["eligible_indices"] == eligible and ledger["witness_indices"] == witnesses, "Actual strict acceptance"
    )
    candidates = witnesses or eligible
    selected = (
        min(
            candidates,
            key=lambda i: (
                independent_loss(entries[i]["normalized_squared_errors"]),
                float(np.max(entries[i]["normalized_squared_errors"])),
                i,
            ),
        )
        if candidates
        else None
    )
    require(ledger["selected_index"] == selected, "Independent full-trajectory ranking")
    starts = [e for e in events if e["kind"] == "request"]
    completed = [e for e in events if e["kind"] == "response_completed"]
    require(len(starts) == len(completed) == len(requests), "Durable completed request accounting")
    for i, row in enumerate(requests):
        require(
            starts[i]["index"] == completed[i]["index"] == i
            and starts[i]["command_sha256"] == completed[i]["command_sha256"] == row["command_sha256"]
            and starts[i]["rate_admitted"] == row["rate_admitted"]
            and starts[i]["role"] == row["role"],
            "Durable request and response identity",
        )
    require(sum(e["kind"] == "solver_closed" for e in events) == 1, "Durable terminal state")
    return dict(
        passed=True,
        responses=len(entries),
        eligible=len(eligible),
        witnesses=len(witnesses),
        selected_index=selected,
        complete_derivatives=sum(r["complete"] for r in ledger["derivatives"]),
    )
