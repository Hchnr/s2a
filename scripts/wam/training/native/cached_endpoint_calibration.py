"""Fit on D0 terminal proxies, then score frozen parameters on saved raw responses."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from s2a.geometry import pose9_to_transform, pose_error
from s2a.references.action_reference import decode_action_targets
from s2a.research.wam.readout.aggregate_state_metrics import diagnosis
from s2a.research.wam.training.cached_endpoint_calibration import (
    apply_calibration,
    features,
    fit_opening,
    fit_pose,
    training_targets,
)
from s2a.research.wam.training.clean.clean_training_aggregation import admit_aggregation
from s2a.research.wam.training.clean.clean_training_data import VALIDATION_DEMO_KEYS
from s2a.research.wam.training.clean.clean_training_runtime import publish, reference
from scripts.verification.wam.training.verify_wam_clean_training_capture import Evidence, require

CONFIG = Path("configs/wam/training/cached_endpoint_calibration_20261111.json")
SOURCE = Path("src/s2a/research/wam/training/cached_endpoint_calibration.py")
PLAN = Path("plans/v0.0.1/wam/training/cached_endpoint_calibration_20261111.md")
CHECKER = Path("scripts/verification/wam/training/verify_cached_endpoint_calibration.py")


def load_config(evidence):
    require(all(os.environ.get(k) == "" for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")), "Explicit CPU required")
    config_ref = reference(CONFIG)
    config = evidence.json(config_ref)
    require(
        config["schema"] == "s2a.cached-endpoint-calibration.v1"
        and config["training_round"] == "D0"
        and config["training_demos"] == 30
        and config["training_anchors"] == 150
        and config["training_requests"] == 300
        and config["validation_demos"] == 6
        and config["validation_anchors"] == 24
        and config["validation_requests"] == 48
        and config["horizon"] == 40
        and config["dt_s"] == 0.05
        and config["seed"] == 42
        and config["automatic_retry"] is config["fresh_authorized"] is False,
        "registered experiment identity differs",
    )
    return config, config_ref


def checked_config_ref(config, key):
    path = Path(config[key]).resolve()
    return {"path": str(path), "sha256": config[key + "_sha256"]}


def raw_c0_action(evidence, request, normal):
    ledger = evidence.json(request["compact_ledger_ref"])
    require(ledger["complete"] and len(ledger["points"]) == 35, "complete D0 ledger required")
    point = evidence.json(ledger["points"][-1])
    meta = point["metadata"]
    require(
        meta["step_index"] == 34
        and meta["request_name"] == request["name"]
        and meta["noise_seed"] == request["sampling_seed"]
        and not meta["feature_terms"],
        "terminal sample identity differs",
    )
    with np.load(evidence.path(point["arrays_ref"]), allow_pickle=False) as z:
        raw = z["sample_active10_fp32"].astype(np.float64) - meta["sigma_actual"] * z["raw_velocity_active10"].astype(
            np.float64
        )
        require(raw.shape == (40, 10) and np.array_equal(raw, z["c0_active10"]), "D0 C0 reconstruction differs")
    transform = normal["transform"]
    mask = np.asarray(transform["normalization_mask"][:10], dtype=bool)
    scale, bias = (np.asarray(transform[k][:10], dtype=float) for k in ("scale", "bias"))
    require(normal["normalization_id"] == 31 and np.all(scale[mask] != 0), "normalization identity differs")
    raw[:, mask] = (raw[:, mask] - bias[mask]) / scale[mask]
    action = np.zeros((40, 128))
    action[:, :10] = raw
    return action


def training_archive(evidence, anchor):
    with np.load(evidence.path(anchor["trajectory_ref"]), allow_pickle=False) as z:
        data = {k: z[k].copy() for k in ("state128", "sim_states", "sim_time", "state_mask", "action_valid")}
    start = anchor["anchor_frame"]
    endpoint = anchor["current_endpoint"]
    states = data["state128"][start : start + 41]
    require(
        states.shape == (41, 128)
        and np.isfinite(states).all()
        and data["action_valid"][start : start + 40].all()
        and data["state_mask"][start : start + 41, :10].all()
        and np.allclose(np.diff(data["sim_time"][start : start + 41]), 0.05, atol=1e-9, rtol=0)
        and data["sim_time"][start] == endpoint["sim_time"]
        and hashlib.sha256(data["sim_states"][start].tobytes()).hexdigest() == endpoint["sim_state_sha256"]
        and hashlib.sha256(data["sim_states"][0].tobytes()).hexdigest()
        == anchor["source_identity"]["initial_state_sha256"]
        and np.all((states[:, 9] >= 0) & (states[:, 9] <= 1)),
        "training current/future state, mask, clock or source differs",
    )
    return states


def fit():
    evidence = Evidence()
    config, config_ref = load_config(evidence)
    output = Path(config["output"])
    require(not output.exists(), "new immutable output root required")
    registration_ref = checked_config_ref(config, "fit_registration")
    registration = evidence.json(registration_ref)
    _, prepared, _, collections = admit_aggregation(registration, read_json=evidence.json)
    training = collections[0]
    require(training["round_name"] == "D0" and len(training["requests"]) == 300, "exact D0 collection required")
    normal = evidence.json(training["normalization_ref"])
    anchors = {a["name"]: a for a in prepared["training_anchors"]}
    identities = {(a["source_identity"]["task_index"], a["source_identity"]["demo"]) for a in anchors.values()}
    require(len(anchors) == 150 and len(identities) == 30 and not identities & VALIDATION_DEMO_KEYS, "training split")
    archive_cache, counts = {}, {}
    pose_x, pose_y, gripper_x, raw_open, truth_open = [], [], [], [], []
    for request in training["requests"]:
        anchor = anchors[request["anchor_name"]]
        name = anchor["name"]
        if name not in archive_cache:
            archive_cache[name] = training_archive(evidence, anchor)
        states = archive_cache[name]
        counts[name] = counts.get(name, 0) + 1
        action = raw_c0_action(evidence, request, normal)
        measured = pose9_to_transform(states[0, :9])
        truth = pose9_to_transform(states[1:, :9])
        predicted, opening = decode_action_targets(measured, action, horizon=40)
        px, gx = features(measured, predicted, opening, states[0, 9], config)
        pose_x.append(px)
        pose_y.append(training_targets(measured, predicted, truth, config))
        gripper_x.append(gx)
        raw_open.append(opening)
        truth_open.append(states[1:, 9])
    require(len(archive_cache) == 150 and set(counts.values()) == {2}, "two D0 seeds per anchor required")
    px, py, gx, opening, truth = map(np.concatenate, (pose_x, pose_y, gripper_x, raw_open, truth_open))
    pose_coefficients = fit_pose(px, py, config["pose_ridge"])
    opening_coefficients, opening_result = fit_opening(
        gx, opening, truth, config["opening_l1"], config["opening_solver_time_limit_s"]
    )
    source_refs = [reference(SOURCE), reference(Path(__file__)), reference(PLAN), reference(CHECKER)]
    for ref in source_refs:
        evidence.path(ref)
    evidence.finish()
    candidate = dict(
        schema="s2a.cached-endpoint-candidate.v1",
        complete=True,
        config_ref=config_ref,
        registration_ref=registration_ref,
        source_refs=source_refs,
        pose_coefficients=pose_coefficients.tolist(),
        opening_coefficients=opening_coefficients.tolist(),
        opening_lp=opening_result,
        training_identities=sorted([list(identity) for identity in identities]),
        training_anchor_names=sorted(anchors),
        training_requests=len(training["requests"]),
        training_rows=len(px),
        terminal_proxy="D0 ideal FP64 C0 at node 34, not final sampler output",
        validation_future_reads=0,
        new_model_calls=0,
        physics_steps=0,
        fresh_cases=0,
        input_sha256=evidence.inputs,
    )
    print(json.dumps(publish(output / "candidate.json", candidate)), flush=True)


def endpoint_errors(truth, poses, opening, truth_opening):
    error = pose_error(truth, poses)
    return dict(
        position=np.linalg.norm(error[:, :3], axis=1).tolist(),
        rotation=np.linalg.norm(error[:, 3:], axis=1).tolist(),
        opening=(opening - truth_opening).tolist(),
    )


def rename_candidate(value):
    if isinstance(value, dict):
        return {("endpoint_calibration" if k == "aggregate056" else k): rename_candidate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rename_candidate(v) for v in value]
    if value == "aggregate056":
        return "endpoint_calibration"
    return value


def validate():
    evidence = Evidence()
    config, config_ref = load_config(evidence)
    root = Path(config["output"])
    candidate_ref = reference(root / "candidate.json")
    candidate = evidence.json(candidate_ref)
    require(
        candidate["schema"] == "s2a.cached-endpoint-candidate.v1"
        and candidate["complete"]
        and candidate["config_ref"] == config_ref
        and candidate["training_requests"] == 300
        and candidate["training_rows"] == 12000
        and candidate["validation_future_reads"] == 0,
        "frozen training candidate required",
    )
    for ref in candidate["source_refs"]:
        evidence.path(ref)
    score_ref = checked_config_ref(config, "validation_score")
    score = evidence.json(score_ref)
    numeric = evidence.json(score["numerical_ref"])
    require(
        score["schema_version"] == "s2a.aggregate-state-score.v1"
        and score["complete"] is score["passed"] is numeric["complete"] is numeric["passed"] is True
        and numeric["numerical_contract_passed"] is numeric["lifecycle_passed"] is True
        and len(numeric["generated_refs"]) == 96,
        "original complete validation evidence required",
    )
    prepared = evidence.json(score["prepared_ref"])
    current = evidence.json(prepared["current_inputs_ref"])
    anchors = {a["name"]: a for a in current["current_inputs"]}
    rows = {r["name"]: r for r in score["state_rows"]}
    require(len(anchors) == 24 and len(rows) == 96, "original validation denominator differs")
    archives, scoring_rows = {}, []
    for item in numeric["generated_refs"]:
        if item["variant"] != "raw":
            continue
        row = rows[item["name"]]
        anchor = anchors[item["anchor_name"]]
        episode = row["episode"]
        if episode not in archives:
            with np.load(evidence.path(score["future_archive_refs"][episode]), allow_pickle=False) as z:
                archives[episode] = {k: z[k].copy() for k in ("state128", "sim_states", "sim_time", "action_valid")}
        data = archives[episode]
        frame = anchor["anchor_frame"]
        states = data["state128"][frame : frame + 41]
        require(
            states.shape == (41, 128)
            and np.array_equal(states[0], anchor["measured_state128"])
            and hashlib.sha256(data["sim_states"][frame].tobytes()).hexdigest() == anchor["current_sim_state_sha256"]
            and data["sim_time"][frame] == anchor["current_time_s"]
            and data["action_valid"][frame : frame + 40].all()
            and np.allclose(np.diff(data["sim_time"][frame : frame + 41]), 0.05, atol=1e-9, rtol=0),
            "validation current/future binding differs",
        )
        response = evidence.json(item["response_ref"])
        action = np.asarray(response["outputs"][0]["content"]["action"])
        measured = pose9_to_transform(states[0, :9])
        truth = pose9_to_transform(states[1:, :9])
        raw, opening = decode_action_targets(measured, action, horizon=40)
        old_errors = endpoint_errors(truth, raw, opening, states[1:, 9])
        require(
            all(np.allclose(old_errors[k], row["errors"]["wam"][k], atol=1e-12, rtol=0) for k in old_errors),
            "raw reconstruction differs from registered validation score",
        )
        calibrated, adjusted_opening = apply_calibration(
            measured,
            raw,
            opening,
            states[0, 9],
            candidate["pose_coefficients"],
            candidate["opening_coefficients"],
            config,
        )
        corrected_row = dict(row)
        corrected_row["name"] = row["name"] + "_endpoint_calibration"
        corrected_row["variant"] = "aggregate056"  # Internal label required by the original fixed gate.
        corrected_row["errors"] = dict(row["errors"])
        corrected_row["errors"]["wam"] = endpoint_errors(truth, calibrated, adjusted_opening, states[1:, 9])
        scoring_rows.extend((row, corrected_row))
    require(len(archives) == 6 and len(scoring_rows) == 96, "all 48 paired raw cases required")
    validation_ids = {(int(name[4]), "demo_" + name.rsplit("_", 1)[1]) for name in archives}
    require(
        validation_ids == VALIDATION_DEMO_KEYS
        and not validation_ids & set(map(tuple, candidate["training_identities"])),
        "validation split",
    )
    groups, gate = diagnosis(scoring_rows)
    evidence.finish()
    result = dict(
        schema="s2a.cached-endpoint-validation.v1",
        complete=True,
        candidate_ref=candidate_ref,
        score_ref=score_ref,
        requests=48,
        validation_anchors=len(anchors),
        validation_episodes=sorted(archives),
        state_metrics=rename_candidate(groups),
        original_054_gate_on_saved_responses=gate,
        state_rows=rename_candidate(scoring_rows),
        eligible_for_development=False,
        new_model_calls=0,
        physics_steps=0,
        fresh_cases=0,
        input_sha256=evidence.inputs,
        evidence_boundary="Offline postprocessing of saved raw responses; not new WAM sampling, closed-loop tracking, grasp or task success.",
    )
    print(json.dumps(publish(root / "validation.json", result)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "validate"))
    args = parser.parse_args()
    {"fit": fit, "validate": validate}[args.phase]()


if __name__ == "__main__":
    main()
