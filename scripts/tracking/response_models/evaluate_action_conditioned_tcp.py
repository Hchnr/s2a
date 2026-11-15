"""Score action-conditioned TCP predictions using completed trajectory history."""

import numpy as np

from s2a.geometry import finite
from s2a.research.response_models.causal.action_conditioned_tcp_residual import CausalActionTCPResidual
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.response_models import audit_causal_tcp_prediction as diagnostic

METHODS = ("raw", "previous_innovation", "action_conditioned")


def evaluate(periods):
    """Stage every observed command before reading its resulting TCP error."""
    model, rows = CausalActionTCPResidual(), []
    for period in periods:
        quality = period["decision"]["quality"]
        prefix.require(quality["verification"]["passed"] is True, "An independently checked prediction is required")
        predicted = finite(quality["first_tcp_error"], (6,))
        command = finite(period["command"], (7,))
        forecast = model.stage(period["step"], predicted, command[:6])
        # A staged forecast cannot depend on this action's later measurement.
        actual = finite(period["tcp_error"], (6,))
        actual_safe = bool(np.linalg.norm(actual[:3]) <= 0.01 and np.linalg.norm(actual[3:]) <= 0.05)
        prefix.require(period["pose_caps_satisfied"] is actual_safe, "Saved physical pose gate changed")
        row = dict(
            step=period["step"],
            arm_command=command[:6].copy(),
            actual_error=actual.copy(),
            actual_pose_caps_satisfied=actual_safe,
            completed_history=forecast["completed_history"],
            history_available=forecast["history_available"],
            fit_history=forecast["fit_history"],
            ridge_fit_performed=forecast["ridge_fit_performed"],
        )
        for method in METHODS:
            value = forecast[method]
            if value is None:
                row[method] = None
                continue
            position, rotation = diagnostic._distance(value, actual)
            row[method] = dict(
                predicted_error=value,
                position_prediction_error_m=position,
                rotation_prediction_error_rad=rotation,
                predicted_pose_caps_satisfied=diagnostic._safe(value),
            )
        row["completed_innovation"] = model.complete(period["step"], actual)
        rows.append(row)
    return rows


def summarize(rows):
    """Compare all methods on the same completed-history denominator."""
    scored = [row for row in rows if row["history_available"]]
    prefix.require(scored, "At least one scored period with prior history is required")
    summaries = {}
    for method in METHODS:
        prefix.require(all(row[method] is not None for row in scored), "Every scored method must be present")
        safe = [row[method]["predicted_pose_caps_satisfied"] for row in scored]
        actual = [row["actual_pose_caps_satisfied"] for row in scored]
        errors_m = np.asarray([row[method]["position_prediction_error_m"] for row in scored])
        errors_rad = np.asarray([row[method]["rotation_prediction_error_rad"] for row in scored])
        summaries[method] = dict(
            periods=len(scored),
            predicted_safe=sum(safe),
            false_safe=sum(pred and not observed for pred, observed in zip(safe, actual, strict=True)),
            false_stop=sum(not pred and observed for pred, observed in zip(safe, actual, strict=True)),
            position_prediction_rmse_m=float(np.sqrt(np.mean(errors_m**2))),
            rotation_prediction_rmse_rad=float(np.sqrt(np.mean(errors_rad**2))),
        )
    return dict(
        recorded_periods=len(rows),
        scored_periods=len(scored),
        ridge_fits=sum(row["ridge_fit_performed"] for row in rows),
        methods=summaries,
    )
