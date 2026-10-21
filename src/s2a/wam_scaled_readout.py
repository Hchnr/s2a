"""027 train-only global centering/scaling of clean-action readout features."""

from dataclasses import dataclass

import numpy as np

from s2a.wam_flow_readout import FlowResidualReadout, sigma_rows
from s2a.wam_readout import ACTIVE_SLOTS, HIDDEN_SIZE, RIDGE, DomainContext, _bfloat16_rne, _matrix


def training_normalization(hidden, weights):
    """One weighted center and global vector L2 RMS; no column whitening."""
    hidden = _matrix(hidden, "training hidden")
    weights = np.asarray(weights, dtype=np.float64)
    if (
        weights.shape != (len(hidden),)
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
        or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError("requires positive training weights summing to one")
    center = np.sum(weights[:, None] * hidden, axis=0, dtype=np.float64)
    centered = hidden - center
    scale = float(np.sqrt(np.sum(weights * np.sum(centered**2, axis=1, dtype=np.float64), dtype=np.float64)))
    if not np.isfinite(center).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError("training center/global L2 RMS must be finite with positive scale")
    return center.copy(), scale


@dataclass(frozen=True)
class ScaledFlowReadout(FlowResidualReadout):
    center: np.ndarray
    scale: float

    def __post_init__(self):
        super().__post_init__()
        center = np.asarray(self.center, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        if (
            center.shape != (self.hidden_size,)
            or not np.isfinite(center).all()
            or scale.shape != ()
            or not np.isfinite(scale)
            or scale <= 0
        ):
            raise ValueError("requires finite matching center and positive scalar global scale")
        center = center.copy()
        center.setflags(write=False)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "scale", float(scale))

    @classmethod
    def zero(cls, hidden_size=HIDDEN_SIZE, *, center=None, scale=1.0):
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError("hidden size must be a positive integer")
        return cls(
            np.zeros((hidden_size + 1, ACTIVE_SLOTS)), np.zeros(hidden_size) if center is None else center, scale
        )

    @classmethod
    def fit(cls, hidden, sigma_actual, residual_target, weights):
        hidden = _matrix(hidden, "training hidden")
        target = _matrix(residual_target, "training residual", columns=ACTIVE_SLOTS)
        if len(target) != len(hidden):
            raise ValueError("fit hidden and residual rows must match")
        center, scale = training_normalization(hidden, weights)
        weights = np.asarray(weights, dtype=np.float64)
        normalized = (hidden - center) / scale
        phi = np.column_stack((normalized, np.ones(len(hidden))))
        design = -phi / sigma_rows(sigma_actual, len(hidden))
        normal = design.T @ (weights[:, None] * design) + RIDGE * np.eye(design.shape[1])
        fitted = np.linalg.solve(normal, design.T @ (weights[:, None] * target))
        return cls(fitted, center, scale)

    @classmethod
    def fit_bias(cls, *args, **kwargs):
        raise ValueError("027 reuses the admitted026 bias; a control refit is forbidden")

    def apply_torch(self, hidden, base, context, projection_ids, sigma_actual):
        import torch

        if not isinstance(context, DomainContext):
            raise ValueError("requires the full-domain request context")
        if not isinstance(base, torch.Tensor) or base.ndim != 2 or base.shape[1] != 128:
            raise ValueError("base must be N×128")
        if (
            not isinstance(projection_ids, torch.Tensor)
            or projection_ids.shape != (len(base),)
            or projection_ids.dtype != torch.long
            or not bool(torch.all(projection_ids == context.projection_id))
        ):
            raise ValueError("mixed or inconsistent projection IDs")
        if not context.is_libero:
            return base
        sigma = torch.tensor(sigma_rows(sigma_actual, len(base)), dtype=torch.float64, device=base.device)
        if self.is_zero:
            return base
        if (
            not isinstance(hidden, torch.Tensor)
            or hidden.shape != (len(base), self.hidden_size)
            or hidden.device != base.device
            or not hidden.is_floating_point()
            or not base.is_floating_point()
            or not bool(torch.isfinite(hidden).all())
            or not bool(torch.isfinite(base).all())
        ):
            raise ValueError("requires finite matching hidden/base tensors on one device")
        center = torch.tensor(self.center.copy(), dtype=torch.float64, device=base.device)
        normalized = (hidden.to(torch.float64) - center) / self.scale
        phi = torch.cat((normalized, torch.ones((len(base), 1), dtype=torch.float64, device=base.device)), dim=1)
        design = -phi / sigma
        coefficients = torch.tensor(self.weights.copy(), dtype=torch.float64, device=base.device)
        summed = base[:, :ACTIVE_SLOTS].to(torch.float64) + design @ coefficients
        active = _bfloat16_rne(summed) if base.dtype == torch.bfloat16 else summed.to(base.dtype)
        if not bool(torch.isfinite(active).all()):
            raise ValueError("scaled readout produced nonfinite deployment output")
        result = base.clone()
        result[:, :ACTIVE_SLOTS] = active
        return result
