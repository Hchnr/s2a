"""Independent terminal capture audit, retaining the original full point checker."""

import hashlib
from pathlib import Path

import verify_wam_clean_training_capture as CAPTURE
from verify_wam_clean_training_state import point_math, require, zero_sources


def step_head(heads, index):
    if type(index) is not int or not 0 <= index < 35:
        raise ValueError("Actual point index in [0,34] required")
    if heads is None:
        return None
    if not isinstance(heads, tuple) or len(heads) != 2:
        raise ValueError("Explicit independent baseline/terminal numerical head pair required")
    return heads[1 if index == 34 else 0]


def compact_points(
    evidence,
    ledger,
    header,
    directory,
    trace,
    *,
    head,
    historical=None,
    vision_shape=CAPTURE.VISION,
    time_vision_shape=(6000,),
):
    require(ledger["nodes"] == len(ledger["points"]) == len(trace["steps"]) == 35, "full35 state points")
    chain = hashlib.sha256(CAPTURE.canonical(header))
    first, changed = None, 0
    for i, (ref, step) in enumerate(zip(ledger["points"], trace["steps"], strict=True)):
        record = evidence.json(ref)
        require(
            Path(ref["path"]).resolve() == directory / f"compact/step{i:02d}.json"
            and record["request_ref"] == ledger["request_ref"],
            "state point path/order/request",
        )
        arrays = evidence.arrays(record["arrays_ref"], directory / f"compact/step{i:02d}.npz")
        chain.update(CAPTURE.canonical({k: record[k] for k in ("metadata", "source_arrays", "compact_arrays")}))
        require(chain.hexdigest() == record["source_stream_sha256"], "state compact chain")
        changed += point_math(
            record,
            arrays,
            header,
            step,
            head=step_head(head, i),
            vision_shape=vision_shape,
            time_vision_shape=time_vision_shape,
        )
        if historical is not None:
            summaries = zero_sources(
                evidence, record, historical["points"][i], Path(historical["points"][0]["path"]).parent, i
            )
            if "source_summary_witnesses" in historical:
                require(
                    summaries == historical["source_summary_witnesses"][i],
                    "zero producer sidecar differs from actual037 arrays",
                )
        if i == 0:
            first = dict(record, _sample_active=arrays["sample_active10_fp32"])
    require(chain.hexdigest() == ledger["source_stream_sha256"], "state final source hash chain")
    return first, changed
