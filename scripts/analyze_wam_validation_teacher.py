"""Evaluate fixed052 residuals on qualified synchronous validation captures.

Future labels enter only this offline diagnostic. No fitting or generation.
"""

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from s2a.clean_training_data import normalize_active10


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(os.environ.get(k) != "" for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")):
        raise ValueError("Explicit CPU environment required")
    if args.output.exists():
        raise FileExistsError(args.output)
    consumed = {}

    def checked(ref):
        path = Path(ref["path"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != ref["sha256"]:
            raise ValueError(f"Input differs: {path}")
        consumed[str(path)] = digest
        return path

    def read(ref):
        return json.loads(checked(ref).read_text())

    score_payload = args.score.read_bytes()
    consumed[str(args.score)] = hashlib.sha256(score_payload).hexdigest()
    score = json.loads(score_payload)
    numerical = read(score["numerical_ref"])
    if not all(numerical[k] is True for k in ("passed", "complete", "numerical_contract_passed", "lifecycle_passed")):
        raise ValueError("Numerical and lifecycle qualification required before future access")
    if not score["passed"] or not score["complete"] or numerical["prepared_ref"] != score["prepared_ref"]:
        raise ValueError("Scoring identity differs")
    prepared = read(score["prepared_ref"])
    current = read(prepared["current_inputs_ref"])
    anchors = {a["name"]: a for a in current["current_inputs"]}
    params = read(numerical["candidate_params_ref"])
    center, coeff = np.asarray(params["center"]), np.asarray(params["coefficients"])
    sampling = read(numerical["sampling_ref"])
    prefix = read(numerical["prefix_failed_sampling_ref"])
    normal = read(sampling["normalization_ref"])
    generations = {
        g["name"]: g
        for source in (prefix, sampling)
        for g in source["generations"]
        if g["complete"] and g["variant"] in ("raw", "probe052")
    }
    if len(generations) != 96 or set(generations) != {g["name"] for g in numerical["generated_refs"]}:
        raise ValueError("Expected qualified 48 pairs")
    archives, labels = {}, {}
    stats = defaultdict(lambda: {"rows": 0, "base": np.zeros(10), "head": np.zeros(10)})
    for count, qualified in enumerate(numerical["generated_refs"]):
        generation = generations[qualified["name"]]
        if generation["response_ref"] != qualified["response_ref"]:
            raise ValueError("Generation identity differs")
        name = qualified["anchor_name"]
        anchor = anchors[name]
        episode = name.rsplit("_a", 1)[0]
        if episode not in archives:
            with np.load(checked(score["future_archive_refs"][episode]), allow_pickle=False) as z:
                archives[episode] = {
                    k: z[k].copy() for k in ("sim_time", "sim_states", "state128", "action128", "action_valid")
                }
        if name not in labels:
            data = archives[episode]
            frame = anchor["anchor_frame"]
            if (
                not np.array_equal(data["state128"][frame], anchor["measured_state128"])
                or hashlib.sha256(data["sim_states"][frame].tobytes()).hexdigest() != anchor["current_sim_state_sha256"]
                or data["sim_time"][frame] != anchor["current_time_s"]
                or not data["action_valid"][frame : frame + 40].all()
                or not np.allclose(np.diff(data["sim_time"][frame : frame + 41]), 0.05, rtol=0, atol=1e-9)
            ):
                raise ValueError("Current/future physical binding differs")
            labels[name] = normalize_active10(data["action128"][frame : frame + 40], **normal)[
                "normalized_active10"
            ].astype(np.float64)
        ledger = read(generation["compact_ledger_ref"])
        if not ledger["complete"] or len(ledger["points"]) != 35:
            raise ValueError("Incomplete capture")
        variant = qualified["variant"]
        for step, ref in enumerate(ledger["points"]):
            point = read(ref)
            if point["metadata"]["feature_terms"] or point["metadata"]["step_index"] != step:
                raise ValueError("Feature contract differs")
            with np.load(checked(point["arrays_ref"]), allow_pickle=False) as z:
                c0 = z["c0_active10"]
                x = np.column_stack((z["hidden"].astype(np.float64), c0))
                saved_residual = z["clean_residual_active10"]
            prediction = ((x - center) / params["scale"]) @ coeff[:-1] + coeff[-1]
            if variant == "probe052" and not np.allclose(prediction, saved_residual, rtol=1e-9, atol=1e-11):
                raise ValueError("Saved deployed prediction differs")
            target = labels[name] - c0
            error = target - prediction
            if not np.isfinite(error).all():
                raise ValueError("Nonfinite error")
            for group in ("all", f"step/{step}", f"task/{episode[4]}", f"seed/{qualified['sampling_seed']}", episode):
                stat = stats[f"{variant}/{group}"]
                stat["rows"] += 40
                stat["base"] += (target**2).sum(axis=0)
                stat["head"] += (error**2).sum(axis=0)
        if (count + 1) % 24 == 0:
            print(json.dumps({"requests": count + 1}), flush=True)
    groups = {}
    for key, stat in stats.items():
        groups[key] = {
            "rows": stat["rows"],
            "base_mse_active10": (stat["base"] / stat["rows"]).tolist(),
            "head_mse_active10": (stat["head"] / stat["rows"]).tolist(),
            "relative_mse_reduction": {
                component: float(1 - stat["head"][start:stop].sum() / stat["base"][start:stop].sum())
                for component, start, stop in (("xyz", 0, 3), ("rot6d", 3, 9), ("opening", 9, 10))
            },
        }
    if any(groups[f"{v}/all"]["rows"] != 67200 for v in ("raw", "probe052")):
        raise ValueError("Full validation node denominator differs")
    for path, sha in consumed.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError(f"Input changed: {path}")
    result = {
        "complete": True,
        "passed": True,
        "groups": groups,
        "future_cases_read": len(archives),
        "future_windows_decoded": len(labels),
        "input_sha256": consumed,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "normalization_source_sha256": hashlib.sha256(Path("src/s2a/clean_training_data.py").read_bytes()).hexdigest(),
        "boundary": "Normalized clean-estimate teacher error against synchronized native motion. Raw head corrections are offline counterfactuals. Not final generated-state error, robot tracking or task success; validation labels never used for fit.",
        "new_fits": 0,
        "new_model_calls": 0,
        "physics_steps": 0,
        "fresh_cases": 0,
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
