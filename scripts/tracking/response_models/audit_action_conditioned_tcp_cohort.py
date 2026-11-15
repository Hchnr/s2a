"""Independently recompute retrospective TCP scores from the recorded periods."""

from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.infrastructure.publication import publish_result
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.response_models import run_action_conditioned_tcp_cohort as cohort

ROOT = prefix.REPO
OUTPUT = cohort.OUTPUT
QA = ROOT / "outputs/qa/action_conditioned_tcp_independent_audit_20261116/result.json"
METHODS = ("raw", "previous_innovation", "action_conditioned")


def verify_case(spec, periods, result):
    """Check every forecast against a separate augmented least-squares solve."""
    prefix.require(
        result["complete"] is result["passed"] is True
        and result["identity"] == spec
        and len(result["rows"]) == len(periods) == spec["expected_periods"],
        "The full original case and its score receipt are required",
    )
    history, totals = deque(maxlen=20), {}
    for name in METHODS:
        totals[name] = dict(periods=0, false_safe=0, false_stop=0, position_squared=0.0, rotation_squared=0.0)
    max_fit_difference = 0.0
    for step, (period, row) in enumerate(zip(periods, result["rows"], strict=True)):
        raw = np.asarray(period["decision"]["quality"]["first_tcp_error"], dtype=float)
        command = np.asarray(period["command"][:6], dtype=float)
        prefix.require(
            row["step"] == period["step"] == step
            and row["completed_history"] == step
            and row["fit_history"] == len(history)
            and row["history_available"] is bool(history)
            and row["ridge_fit_performed"] is bool(history)
            and np.array_equal(row["arm_command"], command),
            "The action and previous-history identities must match the original record",
        )
        expected = {"raw": raw, "previous_innovation": None, "action_conditioned": None}
        if history:
            previous_raw, previous_actual = history[-1][1], history[-1][2]
            previous_discrepancy = Rotation.from_rotvec(previous_actual[3:]).inv() * Rotation.from_rotvec(
                previous_raw[3:]
            )
            corrected_rotation = Rotation.from_rotvec(raw[3:]) * previous_discrepancy.inv()
            previous = np.r_[raw[:3] + previous_actual[:3] - previous_raw[:3], corrected_rotation.as_rotvec()]
            x = np.stack([item[0] for item in history])
            y = np.stack([item[2][:3] - item[1][:3] for item in history])
            mean_x, mean_y = np.mean(x, axis=0), np.mean(y, axis=0)
            # The augmented system is independent of the production normal-equation solve.
            augmented_x = np.vstack(((x - mean_x) / np.sqrt(len(x)), np.eye(6)))
            augmented_y = np.vstack(((y - mean_y) / np.sqrt(len(x)), np.zeros((6, 3))))
            weights = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0]
            conditioned = previous.copy()
            conditioned[:3] = raw[:3] + mean_y + (command - mean_x) @ weights
            expected.update(previous_innovation=previous, action_conditioned=conditioned)
        # Observe the endpoint only after the two historical predictions have been rebuilt.
        actual = np.asarray(period["tcp_error"], dtype=float)
        actual_safe = bool(np.linalg.norm(actual[:3]) <= 0.01 and np.linalg.norm(actual[3:]) <= 0.05)
        prefix.require(
            np.array_equal(actual, row["actual_error"])
            and actual_safe is period["pose_caps_satisfied"] is row["actual_pose_caps_satisfied"],
            "The measured endpoint and its physical gate must match the source",
        )
        for name in METHODS:
            saved, predicted = row[name], expected[name]
            if predicted is None:
                prefix.require(saved is None, "An empty history must not emit a corrected forecast")
                continue
            difference = float(np.max(np.abs(np.asarray(saved["predicted_error"]) - predicted)))
            prefix.require(difference <= 1e-11, "Independent forecast does not match the saved result")
            if name == "action_conditioned":
                max_fit_difference = max(max_fit_difference, difference)
            position = float(np.linalg.norm(predicted[:3] - actual[:3]))
            rotation = float((Rotation.from_rotvec(predicted[3:]).inv() * Rotation.from_rotvec(actual[3:])).magnitude())
            predicted_safe = bool(
                np.linalg.norm(predicted[:3]) <= 0.01 + 1e-10 and np.linalg.norm(predicted[3:]) <= 0.05 + 1e-10
            )
            prefix.require(
                abs(position - saved["position_prediction_error_m"]) <= 1e-11
                and abs(rotation - saved["rotation_prediction_error_rad"]) <= 1e-11
                and predicted_safe is saved["predicted_pose_caps_satisfied"],
                "Independent errors and predicted gate must match the saved result",
            )
            if history:
                item = totals[name]
                item["periods"] += 1
                item["false_safe"] += int(predicted_safe and not actual_safe)
                item["false_stop"] += int(not predicted_safe and actual_safe)
                item["position_squared"] += position**2
                item["rotation_squared"] += rotation**2
        history.append((command.copy(), raw.copy(), actual.copy()))
    for name, item in totals.items():
        saved = result["summary"]["methods"][name]
        prefix.require(
            item["periods"] == saved["periods"] == len(periods) - 1
            and item["false_safe"] == saved["false_safe"]
            and item["false_stop"] == saved["false_stop"]
            and abs(np.sqrt(item["position_squared"] / item["periods"]) - saved["position_prediction_rmse_m"]) <= 1e-11
            and abs(np.sqrt(item["rotation_squared"] / item["periods"]) - saved["rotation_prediction_rmse_rad"])
            <= 1e-11,
            "Independent per-case metrics must match the saved summary",
        )
    return dict(periods=len(periods), scored_periods=len(periods) - 1, max_fit_difference=max_fit_difference)


def run():
    prefix.environment()
    _config, roster, consumed = cohort.load_manifest()
    qa_ref = prefix.reference(QA)
    qa, before = prefix.read(qa_ref), snapshot(ROOT)["sources"]
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == 0
        and not qa["collection"]["collection_errors"]
        and before == prefix.read(qa["source_before_ref"])["sources"],
        "Source-matched CPU QA is required",
    )
    owner_ref = prefix.reference(OUTPUT / "result.json")
    owner = prefix.read(owner_ref)
    registration = prefix.read(owner["registration_ref"])
    prefix.require(
        owner["complete"] is owner["passed"] is True and registration["roster"] == roster, "A closed cohort is required"
    )
    for ref in registration["source_refs"]:
        prefix.checked(ref)
    checks = {}
    for spec in roster:
        key = spec["key"]
        try:
            case = prefix.read(owner["cases"][key]["result_ref"])
            periods, seen = cohort.load_periods(spec)
            checks[key] = dict(status="closed", **verify_case(spec, periods, case), input_sha256=seen)
        except Exception as error:
            checks[key] = dict(status="failed", error=dict(type=type(error).__name__, message=str(error)))
    _config, after_roster, after_consumed = cohort.load_manifest()
    unchanged = snapshot(ROOT)["sources"] == before and after_roster == roster and after_consumed == consumed
    for ref in registration["source_refs"]:
        prefix.checked(ref)
    passed = unchanged and len(checks) == 31 and all(item["status"] == "closed" for item in checks.values())
    result = dict(
        complete=True,
        passed=passed,
        source_unchanged=unchanged,
        control_qualified=False,
        owner_ref=owner_ref,
        qa_ref=qa_ref,
        cases=checks,
        world_model_calls=0,
        physics_steps=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    print(publish_result(OUTPUT / "independent_audit.json", result), flush=True)
    if not passed:
        raise SystemExit("Independent retrospective verification failed")


if __name__ == "__main__":
    run()
