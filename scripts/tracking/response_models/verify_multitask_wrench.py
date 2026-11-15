"""Independently reconstruct held-task wrench fits and their causal inputs."""

from pathlib import Path

import numpy as np

from s2a.infrastructure.publication import publish_result
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

ROOT = prefix.REPO
BASE = ROOT / "outputs/wrench_multitask_replay_20261113"
CROSS = {
    "path": str(BASE / "cross_validation.json"),
    "sha256": "cc5b29897300d5dc675798e99ce6c5a37b863f37ddac46591e60ac18d09fe51e",
}
PRIOR = {
    "path": str(ROOT / "outputs/multitask_causal_tcp_innovation_20261113/result.json"),
    "sha256": "62565e6ac178f87b5f3a497b279dd53d53a8f9ec9da17fd3750b1429aec834cc",
}
REVIEW = {
    "path": str(ROOT / "outputs/reset_multitask_adaptive_complete_audit_20261113/result.json"),
    "sha256": "7fe0f7d17ddea1913ffab2f5a4243c5b8e4b6a5def9633269ebec95efc82f8fe",
}
OUTPUT = BASE / "independent_verification.json"
INDICES = (0, 2, 4, 6, 8, 10)


def scalar_rows(branch, trace):
    """Build past-error and current-wrench inputs independently of the fit code."""
    sequence = branch["runtime"]["continuation"]
    periods = sequence["first_window"]["periods"] + [
        period for segment in sequence["adaptive"]["segments"] for period in segment["continuation"]["periods"]
    ]
    endpoints = {row["endpoint_step"]: row for row in trace["rows"]}
    prefix.require(len(endpoints) == len(periods) + 1, "Complete endpoint traces are required")
    result = []
    for index in range(1, len(periods)):
        previous, current = periods[index - 1], periods[index]
        prefix.require(
            previous["step"] == index - 1
            and current["step"] == index
            and previous["endpoint_receipt"]["sim_state_sha256"] == endpoints[index]["sim_state_sha256"]
            and current["endpoint_receipt"]["sim_state_sha256"] == endpoints[index + 1]["sim_state_sha256"],
            "Past and future state identities must match the actual action period",
        )
        old_prediction = np.asarray(previous["decision"]["quality"]["first_tcp_error"], dtype=float)
        old_actual = np.asarray(previous["tcp_error"], dtype=float)
        prediction = np.asarray(current["decision"]["quality"]["first_tcp_error"], dtype=float)
        actual = np.asarray(current["tcp_error"], dtype=float)
        wrench = np.asarray(endpoints[index]["wrench"]["world_at_control_site"], dtype=float)
        result.append(
            dict(
                step=index,
                features=np.r_[old_actual[:3] - old_prediction[:3], wrench],
                target=actual[:3] - prediction[:3],
                raw=prediction,
                actual=actual,
            )
        )
    return result


def independent_fit(features, target):
    """Use augmented least squares instead of the production normal equations."""
    mean, scale = features.mean(axis=0), features.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    center = target.mean(axis=0)
    standardized = (features - mean) / scale
    augmented = np.vstack((standardized, np.eye(features.shape[1])))
    response = np.vstack((target - center, np.zeros((features.shape[1], 3))))
    coefficients = np.linalg.lstsq(augmented, response, rcond=None)[0]
    return dict(mean=mean, scale=scale, center=center, coefficients=coefficients)


def main():
    prefix.environment()
    cross, prior, review = (prefix.read(ref) for ref in (CROSS, PRIOR, REVIEW))
    prefix.require(
        cross["complete"] is cross["passed"] is prior["complete"] is prior["passed"] is True, "Closed inputs required"
    )
    groups, refs = {}, [CROSS, PRIOR, REVIEW]
    prior_cases = {case["case_index"]: case for case in prior["cases"] if case["protocol"] == "adaptive"}
    reported = {case["case_index"]: case for case in cross["cases"]}
    max_difference = 0.0
    for case in review["cases"]:
        index = case["case_index"]
        trace_ref = prefix.reference(BASE / f"case_{index:02d}/wrench_trace.json")
        trace, branch = prefix.read(trace_ref), prefix.read(case["producer_ref"])
        prefix.require(trace["complete"] is trace["passed"] is True, "Completed independent replay required")
        refs.extend((trace_ref, case["producer_ref"]))
        groups[index] = scalar_rows(branch, trace)
        old_rows = {row["step"]: row for row in prior_cases[index]["rows"]}
        new_rows = {row["step"]: row for row in reported[index]["periods"]}
        for row in groups[index]:
            old, new = old_rows[row["step"]], new_rows[row["step"]]
            for actual, expected in (
                (row["raw"], old["raw"]["predicted_error"]),
                (row["actual"], old["actual_error"]),
                (new["previous_innovation"], old["previous_innovation"]["predicted_error"]),
            ):
                difference = float(np.max(np.abs(np.asarray(actual) - expected)))
                max_difference = max(max_difference, difference)
                prefix.require(difference <= 1e-12, "Original causal scoring must match")
    prefix.require(sorted(groups) == list(INDICES), "All six case groups are required")
    prediction_difference = 0.0
    for fold in cross["folds"]:
        task = fold["held_task"]
        train = [row for index, rows in groups.items() if index // 4 != task for row in rows]
        prefix.require(
            fold["train_case_indices"] == [index for index in INDICES if index // 4 != task]
            and fold["test_case_indices"] == [index for index in INDICES if index // 4 == task]
            and fold["train_periods"] == len(train),
            "Both seeds of the held task must be excluded from training",
        )
        x = np.asarray([row["features"] for row in train])
        y = np.asarray([row["target"] for row in train])
        for width, model_name, score_name in (
            (3, "history_model", "history_linear"),
            (9, "wrench_history_model", "wrench_history_linear"),
        ):
            independent = independent_fit(x[:, :width], y)
            claimed = fold[model_name]
            prefix.require(claimed["ridge"] == 1.0, "The predeclared regularization must remain fixed")
            for key, value in independent.items():
                difference = float(np.max(np.abs(value - np.asarray(claimed[key]))))
                max_difference = max(max_difference, difference)
                prefix.require(difference < 1e-9, "Independent standardized least-squares fit differs")
            for index in fold["test_case_indices"]:
                saved = {row["step"]: row for row in reported[index]["periods"]}
                for row in groups[index]:
                    estimate = (
                        (row["features"][:width] - independent["mean"])
                        / independent["scale"]
                        @ independent["coefficients"]
                        + independent["center"]
                        + row["raw"][:3]
                    )
                    difference = float(np.max(np.abs(estimate - np.asarray(saved[row["step"]][score_name])[:3])))
                    prediction_difference = max(prediction_difference, difference)
                    prefix.require(difference < 1e-9, "Independent held-task predictions differ")
    for name in ("raw", "previous_innovation"):
        expected = prior["protocol_summaries"]["adaptive"][name]
        actual = cross["summary"][name]
        prefix.require(
            all(actual[key] == expected[key] for key in ("periods", "false_safe", "false_stop"))
            and abs(actual["position_prediction_rmse_m"] - expected["position_prediction_rmse_m"]) < 1e-12,
            "Historical raw and causal-history baselines must remain unchanged",
        )
    for ref in refs:
        prefix.checked(ref)
    print(
        publish_result(
            OUTPUT,
            dict(
                complete=True,
                passed=True,
                source_refs=refs + [prefix.reference(Path(__file__).resolve())],
                held_task_folds=3,
                independent_least_squares_solves=6,
                max_fit_or_baseline_difference=max_difference,
                max_held_prediction_difference=prediction_difference,
                new_physics_steps=0,
                new_model_requests=0,
                fresh_cases=0,
            ),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
