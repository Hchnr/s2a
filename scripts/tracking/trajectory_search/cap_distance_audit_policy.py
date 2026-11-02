"""Independent cap-distance arithmetic, with no production-objective imports."""

import numpy as np

from scripts.tracking.trajectory_search.objective_search_audit import SumQAuditPolicy
from scripts.tracking.trajectory_search.state_vector_gn_search_audit import _metrics, _problem
from scripts.verification.trajectory_search.directional_derivative_audit import array, require

POLICY = "cap_distance_projection_gauss_newton_rationalized_float64"


def cap_loss(squared):
    values = array(squared, "original24 cap squared errors", (24,))
    require(np.all(values >= 0), "Nonnegative original cap squared errors")
    distances = np.zeros(24, dtype=np.float64)
    for index in range(24):
        if values[index] > 1.0:
            distances[index] = (values[index] - 1.0) / (np.sqrt(values[index]) + 1.0)
    loss = float(np.float64(0.5) * np.sum(distances**2, dtype=np.float64))
    require(np.isfinite(loss), "Finite independent cap loss")
    return loss


def projection_model(error, squared, raw_jacobian):
    """Reconstruct each 3D block from signed errors, never a saved projection."""
    residual = array(error, "signed error", (72,)).reshape(24, 3)
    values = array(squared, "original squared errors", (24,))
    require(np.all(values >= 0), "Nonnegative original cap squared errors")
    raw = array(raw_jacobian, "raw error derivative", (72, 72))
    future = np.repeat(np.arange(12), 6)[None, :] > np.repeat(np.arange(12), 6)[:, None]
    require(np.all(raw[future] == 0), "Raw model must already be strictly causal")
    r, projected = np.zeros(72), np.zeros((72, 72))
    for index in range(24):
        if values[index] <= 1.0:
            continue
        norm = np.sqrt(values[index])
        distance = (values[index] - 1.0) / (norm + 1.0)
        coefficient = distance / norm
        transform = coefficient * np.eye(3) + np.outer(residual[index], residual[index]) / norm**3
        rows = slice(3 * index, 3 * index + 3)
        r[rows] = coefficient * residual[index]
        projected[rows] = transform @ raw[rows]
    return r, projected


class CapDistanceAuditPolicy(SumQAuditPolicy):
    policy = POLICY

    @staticmethod
    def metrics(safety, squared, config):
        result = _metrics(safety, squared, config)
        result["cap_loss"] = cap_loss(squared)
        return result

    @staticmethod
    def merit_field(mode):
        return "violation" if mode == "restore" else "cap_loss"

    @staticmethod
    def reduction_scale(before):
        return before["cap_loss"]

    @staticmethod
    def problem(commands, safety, squared, error, Js, Jq, Je, previous, lower, upper, ceiling, radius, mode, config):
        B, g, A, b = _problem(
            commands, safety, squared, error, Js, Jq, Je, previous, lower, upper, ceiling, radius, mode, config
        )
        if mode == "pose":
            r, derivative = projection_model(error, squared, Je)
            B, g = np.eye(72) + derivative.T @ derivative, derivative.T @ r
        return B, g, A, b

    @staticmethod
    def prediction(before, safety, squared, error, Js, Jq, Je, step, config, mode, B, g):
        predicted = _metrics(safety + Js @ step, squared + Jq @ step, config)
        trial_error = error + Je @ step
        predicted["state_loss"] = float(0.5 * (trial_error @ trial_error))
        if mode == "restore":
            return predicted, before["violation"] - predicted["violation"]
        _, derivative = projection_model(error, squared, Je)
        response = derivative @ step
        decrease = float(-(g @ step) - np.float64(0.5) * (response @ response))
        predicted["cap_loss"] = before["cap_loss"] - decrease
        return predicted, decrease
