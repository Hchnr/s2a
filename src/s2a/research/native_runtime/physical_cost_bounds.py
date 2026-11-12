"""Bound actual physical costs without inventing missing worker receipts."""

_FIELDS = ("control_steps", "native_steps", "models", "contexts")


def physical_cost_bounds(runtime, *, limits, receipt_final):
    """Return observed lower and registered upper bounds for one worker.

    A final receipt must already be bound to the owned, reaped worker by its
    caller. Completed counters are lower bounds; attempted counters are upper
    bounds only for such a final receipt. Missing counters or a missing receipt
    retain the registered budget as the upper bound.
    """
    if (
        type(limits) is not dict
        or set(limits) != set(_FIELDS)
        or any(type(value) is not int or value < 0 for value in limits.values())
        or type(receipt_final) is not bool
        or (runtime is not None and type(runtime) is not dict)
    ):
        raise ValueError("Explicit nonnegative physical limits and a receipt-final flag are required")
    runtime = {} if runtime is None else runtime
    counters = runtime.get("counters", {})
    if type(counters) is not dict:
        raise ValueError("Runtime counters must be a mapping")
    entries = {
        "control_steps": (runtime.get("control_steps_attempted"), runtime.get("control_steps_completed")),
        "native_steps": counters.get("mj_step"),
        "models": counters.get("model_construct"),
        "contexts": counters.get("context"),
    }
    result = {}
    for name, entry in entries.items():
        if name != "control_steps" and entry is not None:
            if type(entry) is not dict:
                raise ValueError(f"{name} counter must be a mapping")
            entry = (entry.get("attempted"), entry.get("completed"))
        if entry is None or entry == (None, None):
            result[name] = {"lower": 0, "upper": limits[name], "evidence": "not_recorded"}
            continue
        attempted, completed = entry
        if type(attempted) is not int or type(completed) is not int or not 0 <= completed <= attempted <= limits[name]:
            raise ValueError(f"{name} counter exceeds its registered budget or is incomplete")
        result[name] = {
            "lower": completed,
            "upper": attempted if receipt_final else limits[name],
            "evidence": "final_receipt" if receipt_final else "partial_snapshot",
        }
    return result
