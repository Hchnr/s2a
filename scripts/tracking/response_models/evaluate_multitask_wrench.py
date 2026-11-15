"""Cross-validate causal wrench corrections on recorded development actions."""

import argparse
from pathlib import Path

import numpy as np

from s2a.geometry import finite
from s2a.infrastructure.publication import publish_result
from s2a.research.response_models.causal.tcp_prediction_innovation import CausalTCPPredictionInnovation
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import replay_multitask_wrench as replay
from scripts.tracking.response_models import audit_causal_tcp_prediction as diagnostic

ROOT = prefix.REPO
OUTPUT = replay.OUTPUT / "cross_validation.json"
PLAN = ROOT / "plans/v0.0.1/tracking/response_models/wrench_multitask_replay_20261113.md"
RIDGE = 1.0


def prepare_periods(periods, trace):
    """Stage features at t before reading the outcome produced by action t."""
    endpoints = {row["endpoint_step"]: row for row in trace["rows"]}
    prefix.require(len(endpoints) == len(trace["rows"]) == len(periods) + 1, "One sensor sample per endpoint required")
    history, previous, result = CausalTCPPredictionInnovation(), None, []
    for step, period in enumerate(periods):
        prefix.require(period["step"] == step, "The complete ordered action stream is required")
        start, end = endpoints[step], endpoints[step + 1]
        receipt = period["endpoint_receipt"]
        prefix.require(
            receipt["endpoint_step"] == step + 1
            and receipt["sim_state_sha256"] == end["sim_state_sha256"]
            and receipt["sim_time"] == end["sim_time"]
            and start["sim_time"] < end["sim_time"],
            "Each observation must be aligned with the following physical outcome",
        )
        wrench = finite(start["wrench"]["world_at_control_site"], (6,)).copy()
        predicted = finite(period["decision"]["quality"]["first_tcp_error"], (6,)).copy()
        prefix.require(period["decision"]["quality"]["verification"]["passed"] is True, "Checked prediction required")
        staged = history.stage(step, predicted)
        features = None if previous is None else np.r_[previous, wrench]
        # No field from this outcome contributes to the staged inputs.
        actual = finite(period["tcp_error"], (6,)).copy()
        completed = history.complete(step, actual)
        if features is not None:
            result.append(
                dict(
                    step=step,
                    features=features,
                    raw=staged["raw"],
                    previous=staged["previous_innovation"],
                    actual=actual,
                    target=completed["position_innovation_m"],
                )
            )
        previous = completed["position_innovation_m"].copy()
    return result


def fit_ridge(features, target):
    """Fit one fixed, training-only standardized ridge with a free intercept."""
    features, target = np.asarray(features, dtype=float), np.asarray(target, dtype=float)
    if (
        features.ndim != 2
        or target.shape != (len(features), 3)
        or not len(features)
        or not np.isfinite(features).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("Finite paired feature rows and three-dimensional targets are required")
    mean, scale = features.mean(axis=0), features.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    center = target.mean(axis=0)
    standardized = (features - mean) / scale
    coefficients = np.linalg.solve(
        standardized.T @ standardized + RIDGE * np.eye(features.shape[1]), standardized.T @ (target - center)
    )
    return dict(mean=mean, scale=scale, center=center, coefficients=coefficients, ridge=RIDGE)


def predict_ridge(model, features):
    features = np.asarray(features, dtype=float)
    if features.shape[-1:] != model["mean"].shape or not np.isfinite(features).all():
        raise ValueError("Prediction features must match the fitted causal input")
    return (features - model["mean"]) / model["scale"] @ model["coefficients"] + model["center"]


def summarize(rows):
    result = {}
    for candidate in ("raw", "previous_innovation", "history_linear", "wrench_history_linear"):
        positions, rotations, predicted_safe, actual_safe = [], [], [], []
        for row in rows:
            predicted, actual = row[candidate], row["actual"]
            position, rotation = diagnostic._distance(predicted, actual)
            positions.append(position)
            rotations.append(rotation)
            predicted_safe.append(diagnostic._safe(predicted))
            actual_safe.append(diagnostic._safe(actual))
        result[candidate] = dict(
            periods=len(rows),
            false_safe=sum(p and not a for p, a in zip(predicted_safe, actual_safe, strict=True)),
            false_stop=sum(not p and a for p, a in zip(predicted_safe, actual_safe, strict=True)),
            position_prediction_rmse_m=float(np.sqrt(np.mean(np.square(positions)))),
            rotation_prediction_rmse_rad=float(np.sqrt(np.mean(np.square(rotations)))),
        )
    return result


def cross_validate(groups):
    """Hold out both seeds of one task and fit only the other two task groups."""
    prefix.require([group["case_index"] for group in groups] == list(replay.INDICES), "All six ordered cases required")
    prepared = {group["case_index"]: prepare_periods(group["periods"], group["trace"]) for group in groups}
    folds, cases, all_predictions = [], [], []
    for held_task in range(3):
        train = [row for index, rows in prepared.items() if index // 4 != held_task for row in rows]
        test_indices = [index for index in replay.INDICES if index // 4 == held_task]
        x = np.asarray([row["features"] for row in train])
        y = np.asarray([row["target"] for row in train])
        history_model, force_model = fit_ridge(x[:, :3], y), fit_ridge(x, y)
        folds.append(
            dict(
                held_task=held_task,
                train_case_indices=[index for index in replay.INDICES if index not in test_indices],
                test_case_indices=test_indices,
                train_periods=len(train),
                history_model=history_model,
                wrench_history_model=force_model,
            )
        )
        for index in test_indices:
            scored = []
            for row in prepared[index]:
                previous, raw = row["previous"], row["raw"]
                history = np.r_[raw[:3] + predict_ridge(history_model, row["features"][:3]), previous[3:]]
                wrench = np.r_[raw[:3] + predict_ridge(force_model, row["features"]), previous[3:]]
                scored.append(
                    dict(
                        step=row["step"],
                        raw=raw,
                        previous_innovation=previous,
                        history_linear=history,
                        wrench_history_linear=wrench,
                        actual=row["actual"],
                    )
                )
            cases.append(dict(case_index=index, summary=summarize(scored), periods=scored))
            all_predictions.extend(scored)
    return dict(folds=folds, cases=cases, summary=summarize(all_predictions))


def run():
    prefix.environment()
    review = prefix.read(replay.SOURCE)
    refs, groups = [replay.SOURCE], []
    for row in review["cases"]:
        index = row["case_index"]
        trace_ref = prefix.reference(replay.OUTPUT / f"case_{index:02d}/wrench_trace.json")
        trace, producer = prefix.read(trace_ref), prefix.read(row["producer_ref"])
        prefix.require(
            trace["complete"] is trace["passed"] is True
            and trace["case_index"] == index
            and trace["producer_ref"] == row["producer_ref"]
            and all(value == 0 for value in trace["independent_max_absolute_difference"].values()),
            "Each current-wrench trace must bind the independently replayed producer",
        )
        first, adaptive = replay.audit.periods(producer)
        periods = first + adaptive
        prefix.require(len(periods) == row["counts"]["control_steps"], "The full action denominator must agree")
        refs.extend((trace_ref, row["producer_ref"], trace["checker_ref"]))
        groups.append(dict(case_index=index, periods=periods, trace=trace))
    refs.extend((prefix.reference(PLAN), prefix.reference(Path(__file__).resolve())))
    result = cross_validate(groups)
    for ref in refs:
        prefix.checked(ref)
    return dict(
        complete=True,
        passed=True,
        control_qualified=False,
        retrospective=True,
        task_or_contact_features_used=False,
        task_identity_used_for_folds=True,
        source_refs=refs,
        new_physics_steps=0,
        new_model_requests=0,
        fresh_cases=0,
        fits=6,
        **result,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    print(publish_result(OUTPUT, run()), flush=True)


if __name__ == "__main__":
    main()
