"""Independent038 full-source label/normalization/raw-C0/augmented-QR fit audit.

This program does not import the producer, its label/normalization factory,
CleanResidualReadout or production streaming fit. It reparses saved actual
arrays after the capture/normal-exit gates, then uses independent augmented QR.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from s2a.clean_training_fit_contract import (
    FEATURE_COLUMNS,
    FIT_SCHEMA,
    VERIFICATION_SCHEMA,
    canonical_bytes,
    complete,
    cpu_environment,
    load_context,
    publish,
    reference,
    require,
)
from s2a.geometry import validate_transform
from s2a.streaming_ridge_qr import independent_ridge

DOMAIN = "franka_panda__parallel_gripper/eef20/observed_local_delta/libero_closed_pad_observed_endpoint_eef10_v1"
ACTIVE = np.arange(128) < 10
COMPACT_ARRAYS = {
    "sample_active10_fp32",
    "raw_velocity_active10",
    "deployed_velocity_active10",
    "c0_active10",
    "hidden",
    "projection_ids",
    "clean_residual_active10",
    "velocity_delta_active10",
    "time_vision",
    "time_action",
    "condition_vision",
    "condition_action",
    "action_dim_mask",
}


def bits(left, right, message):
    left, right = np.asarray(left), np.asarray(right)
    require(left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes(), message)


def array_archive(evidence, ref):
    path = evidence.path(ref)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    require(
        all(value.dtype.kind in "biuf" and np.isfinite(value).all() for value in arrays.values()),
        "nonfinite saved arrays",
    )
    return arrays


def independent_label(context, anchor, normalization):
    """Direct measured SE3 right-local labels and separate FP32 affine operations."""
    case = context.evidence.json(anchor["case_ref"])
    source_identity = anchor["source_identity"]
    require(
        all(case["record"][k] == source_identity[k] for k in ("task_index", "demo", "initial_state_sha256"))
        and case["arrays"] == anchor["declared_trajectory_ref"]
        and case["arrays"]["sha256"] == anchor["trajectory_ref"]["sha256"],
        "independent label source differs",
    )
    start, count = anchor["anchor_frame"], anchor["native_steps"] + 81
    require(start in case["record"]["anchor_frames"] and start + 40 < count, "independent H40 window is incomplete")
    path = context.evidence.path(anchor["trajectory_ref"])
    keys = (
        "source_pose",
        "actual_source_site_pose",
        "actual_eef_body_pose",
        "opening",
        "sim_time",
        "state_mask",
        "action_mask",
        "action_valid",
        "action128",
        "state128",
        "endpoint_step",
    )
    values = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in keys:
            array = archive[key]
            require(
                case["array_index"][key] == dict(shape=list(array.shape), dtype=str(array.dtype))
                and len(array) == count,
                "independent actual trajectory layout differs",
            )
            length = 40 if key in ("action_mask", "action_valid", "action128") else 41
            values[key] = array[start : start + length].copy()
    source, site, body = (
        validate_transform(values[key]) for key in ("source_pose", "actual_source_site_pose", "actual_eef_body_pose")
    )
    require(
        source.shape == site.shape == body.shape == (41, 4, 4)
        and np.allclose(source[:, :3, 3], site[:, :3, 3], rtol=0, atol=1e-10)
        and np.allclose(source[:, :3, :3], body[:, :3, :3], rtol=0, atol=1e-10),
        "independent source SITE/BODY mismatch",
    )
    require(
        np.array_equal(values["endpoint_step"], np.arange(start, start + 41))
        and values["sim_time"][0] == anchor["current_endpoint"]["sim_time"]
        and np.allclose(np.diff(values["sim_time"]), 0.05, rtol=0, atol=1e-9),
        "independent measured time mismatch",
    )
    for key, rows in (("state_mask", 41), ("action_mask", 40)):
        require(
            values[key].dtype == np.bool_ and np.array_equal(values[key], np.broadcast_to(ACTIVE, (rows, 128))),
            "independent active mask mismatch",
        )
    require(
        values["action_valid"].dtype == np.bool_
        and values["action_valid"].shape == (40,)
        and values["action_valid"].all(),
        "independent label contains invalid/padded rows",
    )
    opening = values["opening"]
    require(
        opening.shape == (41,) and np.isfinite(opening).all() and np.all((opening >= 0) & (opening <= 1)),
        "independent measured opening differs",
    )
    tcp = source @ validate_transform(anchor["tool_source_from_tcp"])
    left_inverse = np.broadcast_to(np.eye(4), (40, 4, 4)).copy()
    left_inverse[:, :3, :3] = tcp[:-1, :3, :3].transpose(0, 2, 1)
    left_inverse[:, :3, 3] = -(left_inverse[:, :3, :3] @ tcp[:-1, :3, 3, None])[..., 0]
    delta = left_inverse @ tcp[1:]
    physical = np.zeros((40, 128), np.float64)
    physical[:, :3], physical[:, 3:6], physical[:, 6:9], physical[:, 9] = (
        delta[:, :3, 3],
        delta[:, :3, 0],
        delta[:, :3, 1],
        opening[1:],
    )
    require(
        np.allclose(physical, values["action128"], rtol=0, atol=1e-12),
        "independent measured delta/next opening differs",
    )
    state = np.zeros((41, 128), np.float64)
    state[:, :3], state[:, 3:6], state[:, 6:9], state[:, 9] = tcp[:, :3, 3], tcp[:, :3, 0], tcp[:, :3, 1], opening
    require(np.array_equal(state, values["state128"]), "independent measured TCP state differs")
    require(
        normalization["domain"] == DOMAIN
        and normalization["normalization_id"] == 31
        and normalization["projection_id"] == 0,
        "independent normalization domain differs",
    )
    transform = normalization["transform"]
    mask = np.asarray(transform["normalization_mask"])
    scale, bias = np.asarray(transform["scale"], np.float32), np.asarray(transform["bias"], np.float32)
    require(
        mask.dtype == np.bool_
        and mask.shape == scale.shape == bias.shape == (128,)
        and not mask[10:].any()
        and np.isfinite(scale).all()
        and np.isfinite(bias).all(),
        "independent normalization transform differs",
    )
    low, high = np.float32(transform["clip_min"]), np.float32(transform["clip_max"])
    require(np.isfinite([low, high]).all() and low < high, "independent clip bounds differ")
    values32 = physical.astype(np.float32)
    product = np.multiply(values32[:, mask], scale[mask], dtype=np.float32)
    affine = np.add(product, bias[mask], dtype=np.float32)
    clips = (affine < low) | (affine > high)
    values32[:, mask] = np.minimum(np.maximum(affine, low), high)
    require(np.isfinite(values32).all(), "independent FP32 labels overflowed")
    return values32[:, :10].copy(), dict(
        anchor_name=anchor["name"],
        clipped_count=int(clips.sum()),
        trajectory_ref=anchor["trajectory_ref"],
        all40_measured_labels=True,
    )


def load_head(evidence, ref):
    params = evidence.json(ref)
    require(
        set(params) == {"schema_version", "coefficients", "center", "scale", "ridge"}
        and params["schema_version"] == "s2a.clean-residual-readout.v1"
        and params["ridge"] == 0.1,
        "independent parameter schema/ridge differs",
    )
    coefficients, center = np.asarray(params["coefficients"], np.float64), np.asarray(params["center"], np.float64)
    scale = params["scale"]
    require(
        coefficients.shape == (1291, 10)
        and center.shape == (1290,)
        and np.isfinite(coefficients).all()
        and np.isfinite(center).all()
        and type(scale) is float
        and np.isfinite(scale)
        and scale > 0,
        "independent parameter geometry/value differs",
    )
    return coefficients, center, scale


def independent_points(context, entry):
    """Reparse compact hashes/chain/arrays without the producer compact reader."""
    evidence = context.evidence
    ledger = evidence.json(entry["compact_ledger_ref"])
    require(
        ledger["schema_version"] == "s2a.clean-compact-ledger.v1"
        and ledger["complete"] is True
        and ledger["nodes"] == len(ledger["points"]) == 35
        and ledger["future_labels_read"] == 0,
        "independent compact ledger is incomplete",
    )
    header = evidence.json(ledger["request_ref"])
    require(
        header["schema_version"] == "s2a.clean-compact-request.v1" and header["labels_accepted"] is False,
        "independent compact header differs",
    )
    digest = hashlib.sha256(canonical_bytes(header))
    for step, point_ref in enumerate(ledger["points"]):
        point = evidence.json(point_ref)
        require(
            point["schema_version"] == "s2a.clean-compact-point.v1" and point["request_ref"] == ledger["request_ref"],
            "independent point source differs",
        )
        digest.update(canonical_bytes({key: point[key] for key in ("metadata", "source_arrays", "compact_arrays")}))
        require(digest.hexdigest() == point["source_stream_sha256"], "independent compact chain differs")
        arrays = array_archive(evidence, point["arrays_ref"])
        require(
            set(arrays) == COMPACT_ARRAYS and set(point["compact_arrays"]) == COMPACT_ARRAYS,
            "independent compact array roster differs",
        )
        for name, value in arrays.items():
            summary = dict(
                shape=list(value.shape),
                storage_dtype=value.dtype.str,
                snapshot_bytes=value.nbytes,
                snapshot_sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
            )
            require(summary == point["compact_arrays"][name], "independent compact stored bytes differ")
        metadata = point["metadata"]
        sigma = metadata["sigma_actual"]
        require(
            metadata["step_index"] == step
            and (sigma, metadata["network_timestep"]) == tuple(header["expected_grid"][step])
            and type(sigma) is float
            and sigma > 0
            and float(np.float32(sigma)) == sigma,
            "independent actual sigma/step/time differs",
        )
        require(
            arrays["hidden"].shape == (40, 1280)
            and arrays["sample_active10_fp32"].dtype == np.float32
            and arrays["sample_active10_fp32"].shape == arrays["raw_velocity_active10"].shape == (40, 10)
            and arrays["raw_velocity_active10"].dtype == np.float32
            and arrays["action_dim_mask"].dtype == np.bool_
            and np.array_equal(arrays["action_dim_mask"], ACTIVE),
            "independent compact active/hidden layout differs",
        )
        require(
            arrays["projection_ids"].shape == (40,)
            and arrays["projection_ids"].dtype == np.int64
            and not arrays["projection_ids"].any()
            and not arrays["condition_action"].any(),
            "independent compact projection/privileged action condition differs",
        )
        yield header["request"], metadata, arrays
    require(digest.hexdigest() == ledger["source_stream_sha256"], "independent final compact chain differs")


def independent_point_rows(x0, metadata, arrays, *, head=None):
    sigma = metadata["sigma_actual"]
    require(
        type(sigma) is float and np.isfinite(sigma) and sigma > 0 and float(np.float32(sigma)) == sigma,
        "independent target requires actual positive FP32 sigma",
    )
    require(
        x0.dtype == np.float32 and x0.shape == (40, 10) and np.isfinite(x0).all(),
        "independent x0 must be normalized FP32 active10",
    )
    c0 = arrays["sample_active10_fp32"].astype(np.float64) - sigma * arrays["raw_velocity_active10"].astype(np.float64)
    bits(c0, arrays["c0_active10"], "independent C0 was not built from raw velocity")
    feature = np.concatenate((arrays["hidden"].astype(np.float64), c0), axis=1)
    if head is None:
        require(not arrays["clean_residual_active10"].any(), "D0 contains an adapted residual")
    else:
        coefficient, center, scale = head
        residual = np.column_stack(((feature - center) / scale, np.ones(40))) @ coefficient
        require(
            np.allclose(residual, arrays["clean_residual_active10"], rtol=1e-9, atol=1e-11),
            "independent captured residual differs from frozen W1",
        )
    return feature, x0.astype(np.float64) - c0


def independent_initial_pairs(context):
    if context.rounds == ("D0",):
        return
    for a, b in zip(*(collection["requests"] for collection in context.collections), strict=True):
        left, right = context.evidence.json(a["initial_ref"]), context.evidence.json(b["initial_ref"])
        require(
            all(left[key] == right[key] for key in ("name", "pair_id", "sampling_seed", "caption", "text_tokens")),
            "independent D0/D1 online initial conditions differ",
        )
        x, y = array_archive(context.evidence, left["arrays"]), array_archive(context.evidence, right["arrays"])
        require(
            set(x) == set(y) == {"initial_noise", "condition_mask", "first_condition_latent"},
            "independent D0/D1 initial roster differs",
        )
        for key in x:
            require(x[key].dtype == y[key].dtype == np.float32, "independent initial state is not actual FP32")
            bits(x[key], y[key], "independent D0/D1 complete initial bits differ")


def independent_block_factory(context, label_receipts):
    anchors = {a["name"]: a for a in context.prepared["training_anchors"]}
    require(
        len(label_receipts) == 150 and len({r["anchor_name"] for r in label_receipts}) == 150,
        "producer post-capture label receipts are incomplete",
    )
    receipts = {r["anchor_name"]: r for r in label_receipts}
    require(set(receipts) == set(anchors), "producer label sources include missing/validation anchors")
    normal = context.evidence.json(context.collections[0]["normalization_ref"])
    labels, independent_receipts = {}, {}

    def factory():
        rows = 0
        for round_name, collection in zip(context.rounds, context.collections, strict=True):
            head = None if round_name == "D0" else load_head(context.evidence, collection["readout_ref"])
            for expected, entry in zip(context.prepared["training_requests"], collection["requests"], strict=True):
                name = expected["anchor_name"]
                if name not in labels:
                    x0, receipt = independent_label(context, anchors[name], normal)
                    primary = receipts[name]
                    require(
                        primary["case_ref"] == anchors[name]["case_ref"]
                        and primary["trajectory_ref"] == anchors[name]["trajectory_ref"]
                        and primary["source_identity"] == anchors[name]["source_identity"]
                        and primary["anchor_frame"] == anchors[name]["anchor_frame"]
                        and primary["labels_opened_after_all_capture_exits"] is True
                        and primary["clipped_count"] == receipt["clipped_count"],
                        "producer label reconstruction receipt differs",
                    )
                    saved = array_archive(context.evidence, primary["normalized_x0_ref"])
                    require(set(saved) == {"normalized_x0_active10"}, "producer label archive fields differ")
                    bits(x0, saved["normalized_x0_active10"], "independent normalized FP32 target differs")
                    labels[name], independent_receipts[name] = x0, receipt
                features, targets = [], []
                for identity, metadata, arrays in independent_points(context, entry):
                    require(
                        identity["request_name"] == metadata["request_name"] == expected["name"]
                        and identity["noise_seed"] == metadata["noise_seed"] == expected["sampling_seed"]
                        and identity["anchor_id"] == name
                        and identity["round_name"] == round_name
                        and identity["purpose"] == metadata["purpose"] == "training"
                        and identity["readout_sha256"]
                        == (None if head is None else collection["readout_ref"]["sha256"]),
                        "independent compact training policy/identity differs",
                    )
                    feature, target = independent_point_rows(labels[name], metadata, arrays, head=head)
                    features.append(feature)
                    targets.append(target)
                require(len(features) == 35, "independent fit missed positive-sigma nodes")
                rows += 35 * 40
                # All hierarchy levels have fixed equal cardinality. Union
                # assigns exactly half mass to each complete420000-row round.
                yield (
                    np.concatenate(features),
                    np.concatenate(targets),
                    np.full(1400, 1.0 / (420000 * len(context.rounds))),
                )
        require(rows == context.expected_rows and len(labels) == 150, "independent full row denominator differs")

    return factory, independent_receipts


def verify_fit(prepared_ref, registration_ref, fit_ref, *, stage, output):
    cpu_environment()
    from s2a.clean_training_fit_contract import Evidence

    initial = Evidence()
    produced = initial.json(fit_ref)
    complete(produced, FIT_SCHEMA, "production fit did not complete")
    require(
        produced["stage"] == stage
        and produced["prepared_ref"] == prepared_ref
        and produced["execution_registration_ref"] == registration_ref
        and produced["producer_fits"] == 1,
        "production fit stage/registration differs",
    )
    context = load_context(
        prepared_ref,
        registration_ref,
        stage=stage,
        collection_refs=produced["collection_refs"],
        capture_audit_refs=produced["capture_audit_refs"],
        caller_sources=(
            __file__,
            Path(__file__).resolve().parents[1] / "src/s2a/clean_training_fit_contract.py",
            Path(__file__).resolve().parents[1] / "src/s2a/streaming_ridge_qr.py",
            Path(__file__).resolve().parents[1] / "src/s2a/geometry.py",
        ),
    )
    context.evidence.inputs.update(initial.inputs)
    output = Path(output).resolve()
    require(
        output.is_relative_to(Path(context.prepared["root"]).resolve()) and not output.exists(),
        "independent fit refuses prior/outside output",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(
        schema_version=VERIFICATION_SCHEMA,
        stage=stage,
        complete=False,
        passed=False,
        prepared_ref=prepared_ref,
        execution_registration_ref=registration_ref,
        fit_ref=fit_ref,
        params_ref=produced["params_ref"],
        independent_reconstructions=0,
        new_model_calls=0,
        physics_steps=0,
        rtol=1e-9,
        atol=1e-11,
        automatic_retry=False,
    )
    try:
        independent_initial_pairs(context)
        factory, receipts = independent_block_factory(context, produced["label_reconstruction_receipts"])
        qr = independent_ridge(
            factory,
            expected_rows=context.expected_rows,
            feature_columns=FEATURE_COLUMNS,
            target_columns=10,
            ridge=0.1,
            max_block_rows=1400,
        )
        result["independent_reconstructions"] = 1
        coefficients, center, scale = load_head(context.evidence, produced["params_ref"])
        for name, actual, expected in (
            ("coefficients", coefficients, qr.coefficients),
            ("center", center, qr.center),
            ("scale", np.asarray(scale), np.asarray(qr.scale)),
        ):
            require(np.allclose(actual, expected, rtol=1e-9, atol=1e-11), f"independent QR {name} differs")
        primary = produced["fit_metadata"]
        require(
            primary["rows"] == produced["actual_rows"] == qr.rows == context.expected_rows
            and primary["feature_columns"] == list(FEATURE_COLUMNS)
            and primary["replay_sha256"]["features"] == qr.replay_sha256[0]
            and primary["replay_sha256"]["target_clean_residual"] == qr.replay_sha256[1]
            and primary["replay_sha256"]["weights"] == qr.replay_sha256[2],
            "independent all-row identity/hash differs",
        )
        context.evidence.finish()
        result.update(
            complete=True,
            passed=True,
            actual_rows=qr.rows,
            qr_metadata=qr.metadata(),
            label_reconstruction_receipts=list(receipts.values()),
            input_sha256=dict(context.evidence.inputs),
            full_training_row_reconstruction=True,
            raw_c0_used_for_D1=True,
        )
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        publish(output, result)
    return reference(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--execution-registration", required=True)
    parser.add_argument("--execution-registration-sha256", required=True)
    parser.add_argument("--fit", required=True)
    parser.add_argument("--fit-sha256", required=True)
    parser.add_argument("--stage", choices=("W1", "W2"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    args = parser.parse_args()
    value = verify_fit(
        dict(path=args.prepared, sha256=args.prepared_sha256),
        dict(path=args.execution_registration, sha256=args.execution_registration_sha256),
        dict(path=args.fit, sha256=args.fit_sha256),
        stage=args.stage,
        output=args.output,
    )
    print(json.dumps(value), flush=True)


if __name__ == "__main__":
    main()
