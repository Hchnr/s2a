"""Independent accounting, derivative and eligibility audit; no optimizer imported."""

import hashlib

import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def audit_projection(ledger, entries, previous, input_min, input_max):
    requests = ledger["requests"]
    require(
        len(requests) == len(entries) == ledger["response_evaluations"] <= ledger["max_evaluations"] <= 320,
        "Every actual response charged",
    )
    require(ledger["complete"] and ledger["iterations"] <= ledger["max_iterations"] <= 4, "Complete bounded solver")
    warm = np.asarray(ledger["supplied_warm"])
    lo, hi = np.asarray(ledger["lower"]), np.asarray(ledger["upper"])
    lower, upper = np.tile(input_min, 12), np.tile(input_max, 12)
    lower[:6] = np.maximum(lower[:6], np.asarray(previous) - 0.25)
    upper[:6] = np.minimum(upper[:6], np.asarray(previous) + 0.25)
    np.testing.assert_array_equal(lo, lower)
    np.testing.assert_array_equal(hi, upper)
    used = np.clip(warm.ravel(), lo, hi)
    require(np.max(np.abs(used - warm.ravel())) <= 1e-12, "Roundoff-only warm repair")
    np.testing.assert_array_equal(used.reshape(12, 6), ledger["used_warm"])
    np.testing.assert_array_equal(used - warm.ravel(), ledger["warm_roundoff"])
    seen = set()
    for i, (request, entry) in enumerate(zip(requests, entries, strict=True)):
        x = np.asarray(entry["commands"])
        require(request["completed"] and request["index"] == i and x.shape == (12, 6), "Exact request identity")
        np.testing.assert_array_equal(x, request["commands"])
        require(
            request["rate_admitted"] == bool(np.all(np.abs(np.diff(np.vstack((previous, x)), axis=0)) <= 0.25 + 1e-12)),
            "Actual rate admission label",
        )
        require(
            x.tobytes() not in seen and np.all(x.ravel() >= lo - 1e-12) and np.all(x.ravel() <= hi + 1e-12),
            "Unique in-box response",
        )
        seen.add(x.tobytes())
    np.testing.assert_array_equal(entries[0]["commands"], ledger["used_warm"])
    for access in ledger["accesses"]:
        require(0 <= access["index"] < len(entries), "Cache access references paid response")
    for record in ledger["derivatives"]:
        base = entries[record["base_index"]]
        x = np.asarray(base["commands"]).ravel()
        columns = []
        for axis, probe in enumerate(record["probes"]):
            require(
                probe["axis"] == axis and probe["nominal_delta"] == (1e-4 if x[axis] + 1e-4 <= hi[axis] else -1e-4),
                "Fixed signed stencil order",
            )
            other = entries[probe["index"]]
            expected = x.copy()
            expected[axis] += probe["nominal_delta"]
            np.testing.assert_array_equal(expected.reshape(12, 6), other["commands"])
            columns.append(
                (np.asarray(other["safety_slack"]) - np.asarray(base["safety_slack"])) / probe["nominal_delta"]
            )
        if record["complete"]:
            require(len(columns) == 72, "Full refreshed Jacobian")
            J = np.asarray(columns).T
            require(
                hashlib.sha256(J.tobytes()).hexdigest() == record["jacobian_sha256"], "Independent derivative identity"
            )
        else:
            require(
                ledger["status"] == "response_budget" and len(columns) < 72, "Only charged budget may truncate stencil"
            )
    for call in ledger["objective_calls"]:
        d = (np.asarray(call["commands"]) - warm).ravel()
        require(float(0.5 * (d @ d)) == call["value"], "Exact minimum-distance objective")
    for call in ledger["gradient_calls"]:
        np.testing.assert_array_equal((np.asarray(call["commands"]) - warm).ravel(), call["gradient"])
    safe = [
        i for i, e in enumerate(entries) if e["audit"]["feasible"] and np.all(np.asarray(e["safety_slack"]) >= 1e-8)
    ]
    witnesses = [i for i in safe if np.all(np.asarray(entries[i]["normalized_squared_errors"]) <= 1)]
    require(ledger["safe_indices"] == safe and ledger["witness_indices"] == witnesses, "Original actual buffered gates")
    candidates = witnesses or safe
    selected = (
        min(candidates, key=lambda i: (float(np.sum((np.asarray(entries[i]["commands"]) - warm) ** 2)), i))
        if candidates
        else None
    )
    require(ledger["selected_index"] == selected, "Independently selected full candidate")
    return dict(
        passed=True,
        responses=len(entries),
        safe=len(safe),
        witnesses=len(witnesses),
        selected_index=selected,
        full_derivatives=sum(r["complete"] for r in ledger["derivatives"]),
    )
