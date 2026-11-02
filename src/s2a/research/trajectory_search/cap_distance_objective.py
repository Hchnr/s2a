"""Full ball-projection GN objective, with an explicitly frozen float64 loss."""

import numpy as np

from s2a.geometry import finite
from s2a.research.trajectory_search.interior_trajectory_search import _jacobian, _outputs, _require
from s2a.research.trajectory_search.state_vector_gn_search import normalized_error_vector
from s2a.research.trajectory_search.state_vector_objective_scale import (
    _metrics,
    build_stage_problem,
    metrics,
)

POLICY = "cap_distance_projection_gauss_newton_rationalized_float64"


def cap_excess(squared):
    """Stable sqrt(q)-1 outside the original closed caps; no epsilon band."""
    q = finite(squared, (24,))
    _require(np.all(q >= 0), "Nonnegative original squared errors required")
    rho = np.sqrt(q)
    result = np.zeros(24, dtype=np.float64)
    outside = q > 1.0
    result[outside] = (q[outside] - 1.0) / (rho[outside] + 1.0)
    return result


def cap_loss(squared):
    excess = cap_excess(squared)
    value = np.float64(0.5) * np.sum(excess**2, dtype=np.float64)
    _require(np.isfinite(value), "Finite cap-distance loss required")
    return float(value)


def linearize(entry, raw_jacobian):
    """Rebuild full 3x3 projection blocks from raw signed E and causal Je."""
    raw = _jacobian(raw_jacobian, np.repeat(np.arange(12), 6))
    z = normalized_error_vector(entry).reshape(24, 3)
    _, squared = _outputs(entry)
    excess, rho = cap_excess(squared), np.sqrt(squared)
    residual, projected = np.zeros(72), np.zeros((72, 72))
    blocks = np.zeros((24, 3, 3))
    for index in np.flatnonzero(squared > 1.0):
        block = slice(3 * index, 3 * index + 3)
        beta = excess[index] / rho[index]
        blocks[index] = beta * np.eye(3) + np.outer(z[index], z[index]) / rho[index] ** 3
        residual[block] = beta * z[index]
        projected[block] = blocks[index] @ raw[block]
    return dict(
        residual=finite(residual, (72,)),
        jacobian=_jacobian(projected, np.repeat(np.arange(12), 6)),
        projection_blocks=blocks,
        squared=squared.copy(),
        loss=cap_loss(squared),
    )


class CapDistanceObjective:
    """Explicit objective policy; state, safety and raw secants stay in driver."""

    policy = POLICY

    @staticmethod
    def metrics(entry, config):
        result = metrics(entry, config)
        result["cap_loss"] = cap_loss(_outputs(entry)[1])
        return result

    @staticmethod
    def merit_field(mode):
        return "violation" if mode == "restore" else "cap_loss"

    @staticmethod
    def reduction_scale(before):
        return before["cap_loss"]

    @staticmethod
    def problem(anchor, Js, Jq, Je, previous, lower, upper, ceiling, radius, mode, config):
        result = build_stage_problem(anchor, Js, Jq, Je, previous, lower, upper, ceiling, radius, mode, config)
        if mode == "pose":
            model = linearize(anchor, Je)
            r, jacobian = model["residual"], model["jacobian"]
            result.update(B=np.eye(72) + jacobian.T @ jacobian, g=jacobian.T @ r, cap_model=model)
        return result

    @staticmethod
    def prediction(before, safety, squared, error, Js, Jq, Je, step, config, mode, problem):
        predicted = _metrics(safety + Js @ step, squared + Jq @ step, config, error + Je @ step)
        if mode == "restore":
            return predicted, before["violation"] - predicted["violation"]
        displacement = problem["cap_model"]["jacobian"] @ step
        reduction = float(-(problem["g"] @ step) - np.float64(0.5) * (displacement @ displacement))
        predicted["cap_loss"] = before["cap_loss"] - reduction
        return predicted, reduction
