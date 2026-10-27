"""Offline physical endpoint error decomposition of completed056 validation."""

import argparse
import hashlib
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from verify_wam_clean_training_capture import Evidence, require

from s2a.clean_training_runtime import publish, reference
from s2a.reference_error_decomposition import decompose, position_summary


def opening_summary(error, predicted, truth):
    return dict(
        endpoints=len(error),
        bias=float(np.mean(error)),
        mae=float(np.mean(np.abs(error))),
        mse=float(np.mean(error**2)),
        predicted_mean=float(np.mean(predicted)),
        truth_mean=float(np.mean(truth)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        all(os.environ.get(k) == "" for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")),
        "Explicit CPU environment required",
    )
    e = Evidence()
    score_ref = reference(args.score)
    score = e.json(score_ref)
    numerical = e.json(score["numerical_ref"])
    require(
        score["schema_version"] == "s2a.aggregate-state-score.v1"
        and score["passed"]
        is score["complete"]
        is numerical["passed"]
        is numerical["complete"]
        is numerical["numerical_contract_passed"]
        is numerical["lifecycle_passed"]
        is True
        and len(numerical["generated_refs"]) == 96,
        "Qualified full056 score required",
    )
    p = e.json(score["prepared_ref"])
    require(numerical["prepared_ref"] == score["prepared_ref"], "Score/numerical identity differs")
    current = e.json(p["current_inputs_ref"])
    anchors = {a["name"]: a for a in current["current_inputs"]}
    rows = {r["name"]: r for r in score["state_rows"]}
    archives, groups = {}, defaultdict(list)
    for item in numerical["generated_refs"]:
        anchor = anchors[item["anchor_name"]]
        row = rows[item["name"]]
        episode = row["episode"]
        if episode not in archives:
            with np.load(e.path(score["future_archive_refs"][episode]), allow_pickle=False) as z:
                archives[episode] = {k: z[k].copy() for k in ("state128", "sim_states", "sim_time", "action_valid")}
        data = archives[episode]
        frame = anchor["anchor_frame"]
        require(
            np.array_equal(data["state128"][frame], anchor["measured_state128"])
            and hashlib.sha256(data["sim_states"][frame].tobytes()).hexdigest() == anchor["current_sim_state_sha256"]
            and data["sim_time"][frame] == anchor["current_time_s"]
            and data["action_valid"][frame : frame + 40].all()
            and np.allclose(np.diff(data["sim_time"][frame : frame + 41]), 0.05, atol=1e-9, rtol=0),
            "Measured current/future physical identity differs",
        )
        response = e.json(item["response_ref"])
        actions = np.asarray(response["outputs"][0]["content"]["action"])
        parts = decompose(data["state128"][frame : frame + 41], actions)
        require(
            np.allclose(
                np.linalg.norm(parts["total_cumulative"], axis=1), row["errors"]["wam"]["position"], rtol=0, atol=1e-12
            )
            and np.allclose(parts["opening_error"], row["errors"]["wam"]["opening"], rtol=0, atol=1e-12),
            "Physical endpoint errors differ from qualified score",
        )
        for horizon in (8, 16, 40):
            for label in ("pooled", f"task/{row['task_index']}", f"seed/{row['sampling_seed']}", episode):
                groups[f"{label}/h{horizon}/{item['variant']}"].append({k: v[:horizon] for k, v in parts.items()})
    result_groups = {}
    for key, items in groups.items():
        parts = {name: np.concatenate([item[name] for item in items]) for name in items[0]}
        error, predicted, truth = (parts[name] for name in ("opening_error", "predicted_opening", "truth_opening"))
        opening = {"all": opening_summary(error, predicted, truth)}
        # Privileged truth bins describe offline phase errors only. They are
        # neither success thresholds nor a proposed deployment switching rule.
        for label, mask in (
            ("truth_closed_le_0.1", truth <= 0.1),
            ("truth_open_ge_0.9", truth >= 0.9),
            ("truth_intermediate", (truth > 0.1) & (truth < 0.9)),
        ):
            if mask.any():
                opening[label] = opening_summary(error[mask], predicted[mask], truth[mask])
        result_groups[key] = dict(position=position_summary(parts), opening=opening)
    for h in (8, 16, 40):
        for variant in ("raw", "aggregate056"):
            measured = result_groups[f"pooled/h{h}/{variant}"]
            expected = score["state_metrics"]["pooled"][str(h)][variant]
            require(
                abs(measured["position"]["total_mse_m2"] - expected["position_rmse_m"] ** 2) < 1e-12
                and abs(measured["opening"]["all"]["mae"] - expected["opening_mae"]) < 1e-12,
                "Pooled physical metrics changed",
            )
    e.finish()
    publish(
        args.output,
        dict(
            complete=True,
            passed=True,
            score_ref=score_ref,
            groups=result_groups,
            requests=96,
            future_cases_read=len(archives),
            input_sha256=e.inputs,
            source_refs=[reference(Path(__file__)), reference(Path("src/s2a/reference_error_decomposition.py"))],
            evidence_boundary="Offline native future frames for algebraic error decomposition; not an online policy, new candidate or causal intervention. Cross term forbids independent percentage attribution.",
            new_fits=0,
            new_model_calls=0,
            physics_steps=0,
            fresh_cases=0,
        ),
    )


if __name__ == "__main__":
    main()
