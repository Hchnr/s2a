"""038 one fixed W1/W2 production fit, after complete capture exit/audit gates.

No model is loaded. This CPU entry opens actual034 labels only after every
required D0/D1 capture and normal-release receipt has passed admission.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from s2a.clean_training_data import clean_targets, normalize_active10, post_capture_labels
from s2a.clean_training_fit_contract import (
    FEATURE_COLUMNS,
    FIT_SCHEMA,
    cpu_environment,
    fresh_output,
    load_context,
    publish,
    reference,
    require,
)
from s2a.wam_clean_compact import iter_compact_points
from s2a.wam_clean_readout import CleanResidualReadout
from s2a.wam_clean_streaming import CleanFitBlock, CleanFitLayout, fit_clean_residual_streaming


def _identity(anchor):
    identity = anchor["source_identity"]
    return identity["task_index"], identity["demo"], identity["initial_state_sha256"]


def load_label(context, anchor, normalization):
    """Only called post-admission; reconstruct and cross-check one real H40 window."""
    evidence = context.evidence
    case = evidence.json(anchor["case_ref"])
    record = case["record"]
    require(
        tuple(record[k] for k in ("task_index", "demo", "initial_state_sha256")) == _identity(anchor)
        and case["arrays"] == anchor["declared_trajectory_ref"]
        and case["arrays"]["sha256"] == anchor["trajectory_ref"]["sha256"],
        "034 label source identity differs",
    )
    path = evidence.path(anchor["trajectory_ref"])
    start, count = anchor["anchor_frame"], anchor["native_steps"] + 81
    require(start in record["anchor_frames"] and start + 40 < count, "034 anchor lacks all41 actual endpoints")
    endpoint_keys = (
        "source_pose",
        "actual_source_site_pose",
        "actual_eef_body_pose",
        "opening",
        "sim_time",
        "state_mask",
    )
    with np.load(path, allow_pickle=False) as archive:
        values = {}
        for key in (*endpoint_keys, "action_valid", "action_mask", "action128", "state128", "endpoint_step"):
            array = archive[key]
            require(
                case["array_index"][key] == dict(shape=list(array.shape), dtype=str(array.dtype)),
                "034 actual label array layout differs from its audited index",
            )
            require(len(array) == count, "034 endpoint denominator differs")
            values[key] = array[
                start : start + (40 if key in ("action_valid", "action_mask", "action128") else 41)
            ].copy()
    require(
        np.array_equal(values["endpoint_step"], np.arange(start, start + 41))
        and values["sim_time"][0] == anchor["current_endpoint"]["sim_time"],
        "034 current/future time binding differs",
    )
    rebuilt = post_capture_labels(
        values["source_pose"],
        values["opening"],
        values["sim_time"],
        source_from_tcp=anchor["tool_source_from_tcp"],
        actual_source_site_poses=values["actual_source_site_pose"],
        actual_eef_body_poses=values["actual_eef_body_pose"],
        state_mask=values["state_mask"],
        action_valid=values["action_valid"],
        action_mask=values["action_mask"],
    )
    require(
        np.array_equal(rebuilt["physical_action128"], values["action128"]),
        "034 saved labels differ from measured local endpoints",
    )
    tcp = rebuilt["tcp_pose"]
    states = np.zeros((41, 128), np.float64)
    states[:, :3], states[:, 3:6], states[:, 6:9], states[:, 9] = (
        tcp[:, :3, 3],
        tcp[:, :3, 0],
        tcp[:, :3, 1],
        values["opening"],
    )
    require(np.array_equal(states, values["state128"]), "034 saved state differs from measured TCP/opening")
    normalized = normalize_active10(rebuilt["physical_action128"], **normalization)
    return normalized["normalized_active10"], dict(
        anchor_name=anchor["name"],
        source_identity=anchor["source_identity"],
        anchor_frame=start,
        case_ref=anchor["case_ref"],
        trajectory_ref=anchor["trajectory_ref"],
        valid_steps=40,
        endpoint_count=41,
        physical_dt=0.05,
        clipped_count=normalized["clipped_count"],
        clipped_per_active_slot=normalized["clipped_per_active_slot"].tolist(),
        transform_sha256=normalized["transform_sha256"],
        terminal_labels="actual measured motion and next opening; no synthetic zero padding",
    )


def point_rows(x0, metadata, arrays, *, head=None):
    target = clean_targets(
        x0,
        arrays["sample_active10_fp32"],
        arrays["raw_velocity_active10"],
        float(metadata["sigma_actual"]),
        velocity_semantics="raw",
    )
    require(np.array_equal(target["raw_c0_active10"], arrays["c0_active10"]), "saved C0 differs from raw capture")
    features = np.column_stack((arrays["hidden"].astype(np.float64), target["raw_c0_active10"]))
    if head is None:
        require(not arrays["clean_residual_active10"].any(), "D0 raw capture contains a residual")
    residual = np.zeros((40, 10)) if head is None else head.predict_clean_residual(features)
    require(
        np.allclose(arrays["clean_residual_active10"], residual, rtol=1e-9, atol=1e-11),
        "capture residual does not come from the fixed raw/W1 policy",
    )
    return features, target["target_clean_residual"]


def check_initial_pairs(context):
    if context.rounds != ("D0", "D1"):
        return
    for left, right in zip(context.collections[0]["requests"], context.collections[1]["requests"], strict=True):
        a, b = context.evidence.json(left["initial_ref"]), context.evidence.json(right["initial_ref"])
        require(
            all(a[key] == b[key] for key in ("name", "pair_id", "sampling_seed", "caption", "text_tokens")),
            "D0/D1 initial conditioning identity differs",
        )
        with (
            np.load(context.evidence.path(a["arrays"]), allow_pickle=False) as aa,
            np.load(context.evidence.path(b["arrays"]), allow_pickle=False) as bb,
        ):
            require(
                set(aa.files) == set(bb.files) == {"initial_noise", "condition_mask", "first_condition_latent"},
                "D0/D1 initial array roster differs",
            )
            for key in aa.files:
                x, y = aa[key], bb[key]
                require(
                    x.dtype == y.dtype == np.float32 and x.shape == y.shape and x.tobytes() == y.tobytes(),
                    "D0/D1 complete initial FP32 noise/condition differs",
                )


def block_factory(context, labels):
    """Replay all35x40 rows per request, in complete D0 then D1 order."""
    expected_requests = context.prepared["training_requests"]
    normal = context.evidence.json(context.collections[0]["normalization_ref"])
    require(
        set(normal) == {"domain", "normalization_id", "projection_id", "transform"},
        "normalization receipt fields differ",
    )
    layout = CleanFitLayout(FEATURE_COLUMNS, "active10")
    weight = 1.0 / context.expected_rows
    anchors = {a["name"]: a for a in context.prepared["training_anchors"]}

    def replay():
        rows = 0
        for round_name, collection in zip(context.rounds, context.collections, strict=True):
            head = (
                None
                if round_name == "D0"
                else CleanResidualReadout.from_parameters(context.evidence.json(collection["readout_ref"]))
            )
            for expected, entry in zip(expected_requests, collection["requests"], strict=True):
                anchor_name = expected["anchor_name"]
                if anchor_name not in labels:
                    labels[anchor_name] = load_label(context, anchors[anchor_name], normal)
                x0, _ = labels[anchor_name]
                ledger = context.evidence.json(entry["compact_ledger_ref"])
                header = context.evidence.json(ledger["request_ref"])
                for point_ref in ledger["points"]:
                    context.evidence.path(context.evidence.json(point_ref)["arrays_ref"])
                identity = header["request"]
                require(
                    identity["request_name"] == expected["name"]
                    and identity["anchor_id"] == anchor_name
                    and identity["noise_seed"] == expected["sampling_seed"]
                    and identity["purpose"] == "training"
                    and identity["round_name"] == round_name
                    and identity["readout_sha256"] == (None if head is None else collection["readout_ref"]["sha256"]),
                    "compact request is outside the frozen training identity/policy",
                )
                features, targets = [], []
                for step, (metadata, arrays) in enumerate(iter_compact_points(entry["compact_ledger_ref"])):
                    require(
                        metadata["step_index"] == step
                        and metadata["request_name"] == expected["name"]
                        and metadata["noise_seed"] == expected["sampling_seed"],
                        "compact step identity differs",
                    )
                    feature, target = point_rows(x0, metadata, arrays, head=head)
                    features.append(feature)
                    targets.append(target)
                require(len(features) == 35, "all35 actual positive-sigma steps must enter fit")
                rows += 1400
                yield CleanFitBlock(np.concatenate(features), np.concatenate(targets), np.full(1400, weight), layout)
        require(rows == context.expected_rows and len(labels) == 150, "full row/anchor denominator differs")

    return replay, layout


def solve_blocks(factory, layout, *, expected_rows):
    """One production solve; the CLI fixes1290 feature columns and full row count."""
    fit = fit_clean_residual_streaming(factory, layout=layout, ridge=0.1)
    require(fit.rows == expected_rows and fit.readout.ridge == 0.1, "production fit dropped or added rows")
    return fit


def run_fit(prepared_ref, registration_ref, *, stage, collection_refs, capture_audit_refs, output_dir):
    cpu_environment()
    context = load_context(
        prepared_ref,
        registration_ref,
        stage=stage,
        collection_refs=collection_refs,
        capture_audit_refs=capture_audit_refs,
        caller_sources=(
            __file__,
            *(
                Path(__file__).resolve().parents[1] / "src/s2a" / name
                for name in (
                    "clean_training_fit_contract.py",
                    "clean_training_data.py",
                    "wam_clean_streaming.py",
                    "wam_clean_readout.py",
                    "wam_clean_compact.py",
                    "wam_clean_capture.py",
                    "geometry.py",
                )
            ),
        ),
    )
    directory = fresh_output(output_dir, context)
    result = dict(
        schema_version=FIT_SCHEMA,
        stage=stage,
        complete=False,
        passed=False,
        prepared_ref=prepared_ref,
        execution_registration_ref=registration_ref,
        collection_refs=list(collection_refs),
        capture_audit_refs=list(capture_audit_refs),
        rounds=list(context.rounds),
        producer_fits=0,
        fit_attempts=0,
        new_model_calls=0,
        physics_steps=0,
        fresh_cases=0,
        automatic_retry=False,
        label_reconstruction_receipts=[],
    )
    try:
        check_initial_pairs(context)
        labels = {}
        factory, layout = block_factory(context, labels)
        result["fit_attempts"] = 1
        fit = solve_blocks(factory, layout, expected_rows=context.expected_rows)
        result["producer_fits"] = 1
        label_dir = directory / "post_capture_labels"
        label_dir.mkdir()
        for index, (name, (x0, receipt)) in enumerate(labels.items()):
            array_path = label_dir / f"anchor{index:03d}.npz"
            with array_path.open("xb") as stream:
                np.savez(stream, normalized_x0_active10=x0)
            receipt = dict(receipt, normalized_x0_ref=reference(array_path), labels_opened_after_all_capture_exits=True)
            result["label_reconstruction_receipts"].append(receipt)
        params_ref = publish(directory / "parameters.json", fit.readout.to_parameters())
        context.evidence.finish()
        result.update(
            complete=True,
            passed=True,
            params_ref=params_ref,
            fit_metadata=fit.fit_metadata(),
            actual_rows=fit.rows,
            label_anchor_count=150,
            input_sha256=dict(context.evidence.inputs),
        )
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        publish(directory / "result.json", result)
    return reference(directory / "result.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--execution-registration", required=True)
    parser.add_argument("--execution-registration-sha256", required=True)
    parser.add_argument("--stage", choices=("W1", "W2"), required=True)
    parser.add_argument("--collection", nargs=2, action="append", required=True, metavar=("PATH", "SHA256"))
    parser.add_argument("--capture-audit", nargs=2, action="append", required=True, metavar=("PATH", "SHA256"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    args = parser.parse_args()
    value = run_fit(
        dict(path=args.prepared, sha256=args.prepared_sha256),
        dict(path=args.execution_registration, sha256=args.execution_registration_sha256),
        stage=args.stage,
        collection_refs=[dict(path=p, sha256=s) for p, s in args.collection],
        capture_audit_refs=[dict(path=p, sha256=s) for p, s in args.capture_audit],
        output_dir=args.output_dir,
    )
    print(json.dumps(value), flush=True)


if __name__ == "__main__":
    main()
