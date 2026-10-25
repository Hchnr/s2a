"""Independent061 full compact math with an explicit per-node head pair."""

import hashlib
from pathlib import Path

import numpy as np
from verify_wam_clean_training_capture import VISION, canonical, require, verify_point
from verify_wam_terminal_points import step_head


def audit_points(
    evidence,
    ledger,
    header,
    directory,
    steps,
    *,
    head=None,
    vision_shape=VISION,
    time_vision_shape=(6000,),
    purpose="training",
):
    require(ledger["nodes"] == len(ledger["points"]) == len(steps) == 35, "complete35 compact points required")
    stream = hashlib.sha256(canonical(header))
    first, changed = None, 0
    for index, (point_ref, step) in enumerate(zip(ledger["points"], steps, strict=True)):
        record = evidence.json(point_ref)
        require(
            Path(point_ref["path"]).resolve() == directory / f"compact/step{index:02d}.json"
            and record["request_ref"] == ledger["request_ref"]
            and step["step_index"] == index,
            "compact step path/request/order",
        )
        arrays = evidence.arrays(record["arrays_ref"], directory / f"compact/step{index:02d}.npz")
        stream.update(canonical({k: record[k] for k in ("metadata", "source_arrays", "compact_arrays")}))
        require(stream.hexdigest() == record["source_stream_sha256"], "compact source-summary chain")
        changed += verify_point(
            record,
            arrays,
            header,
            step,
            head=step_head(head, index),
            vision_shape=vision_shape,
            time_vision_shape=time_vision_shape,
            purpose=purpose,
        )
        if index == 0:
            first = dict(record, _sample_active=arrays["sample_active10_fp32"])
    require(stream.hexdigest() == ledger["source_stream_sha256"], "final compact source-summary chain")
    return first, 35, changed


def head_parameters(value):
    require(
        set(value) == {"schema_version", "coefficients", "center", "scale", "ridge"}
        and value["schema_version"] == "s2a.clean-residual-readout.v1",
        "061 parameter schema",
    )
    coefficient, center = np.asarray(value["coefficients"], np.float64), np.asarray(value["center"], np.float64)
    scale = value["scale"]
    require(
        coefficient.shape == (1291, 10)
        and center.shape == (1290,)
        and np.isfinite(coefficient).all()
        and np.isfinite(center).all(),
        "061 parameter layout",
    )
    require(
        type(scale) in (int, float) and np.isfinite(scale) and scale > 0 and value["ridge"] == 0.1 / 1290,
        "061 frozen scale/ridge",
    )
    return coefficient, center, float(scale)
