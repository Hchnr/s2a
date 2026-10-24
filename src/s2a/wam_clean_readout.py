"""CPU primitives for an explicit clean-action residual head.

These routines do not load a model, choose training rows/noise, construct a time
basis, or qualify CUDA arithmetic. All calculations use NumPy FP64 after checking
that the supplied sampler state is genuinely stored as FP32. The caller remains
responsible for binding that state and sigma to the actual sampler invocation.
"""

from dataclasses import dataclass

import numpy as np

ACTIVE_SLOTS = 10
FULL_SLOTS = 128
PARAMETER_SCHEMA = "s2a.clean-residual-readout.v1"


def _array(values, name):
    values = np.asarray(values)
    if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return values


def _matrix(values, name, *, columns=None):
    values = _array(values, name)
    if values.ndim != 2 or min(values.shape) < 1 or (columns is not None and values.shape[1] != columns):
        shape = f"N x {columns}" if columns is not None else "a nonempty matrix"
        raise ValueError(f"{name} must be {shape}")
    return values.astype(np.float64, copy=True)


def _action(values, name, layout):
    if layout not in ("active10", "full128"):
        raise ValueError("layout must explicitly be active10 or full128")
    result = _matrix(values, name, columns=ACTIVE_SLOTS if layout == "active10" else FULL_SLOTS)
    if layout == "full128" and np.any(result[:, ACTIVE_SLOTS:] != 0):
        raise ValueError(f"{name} inactive118 slots must be exactly zero")
    return result


def _positive_scalar(value, name):
    value = _array(value, name)
    if value.ndim != 0 or value <= 0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return float(value)


def _sigma_rows(actual_sigma, rows):
    sigma = _array(actual_sigma, "actual sigma").astype(np.float64, copy=True)
    if sigma.ndim == 0:
        sigma = np.full((rows, 1), float(sigma))
    elif sigma.shape == (rows,):
        sigma = sigma[:, None]
    if sigma.shape != (rows, 1) or np.any(sigma <= 0):
        raise ValueError("actual sigma must be positive per row; terminal sigma=0 must skip the residual head")
    return sigma


def active10_from_full128(full):
    """Copy active slots only after rejecting nonzero inactive full-layout slots."""
    return _action(full, "full action", "full128")[:, :ACTIVE_SLOTS].copy()


def full128_from_active10(active):
    """Expand an active-layout array to FP64 with exactly zero inactive slots."""
    active = _action(active, "active action", "active10")
    full = np.zeros((len(active), FULL_SLOTS), dtype=np.float64)
    full[:, :ACTIVE_SLOTS] = active
    return full


def clean_estimate(sample_fp32, base_velocity, actual_sigma, *, layout):
    """Return C0 = actual sample - actual sigma * raw velocity in FP64.

    ``sample_fp32`` must be the original sampler action state, not the BF16
    packed network input or a reconstruction. Both arguments use the same
    explicit action layout. Zero terminal sigma is rejected: no head is called
    at that scheduler endpoint.
    """
    if np.asarray(sample_fp32).dtype != np.float32:
        raise ValueError("actual sampler sample must be stored as FP32")
    sample = _action(sample_fp32, "sampler sample", layout)
    base = _action(base_velocity, "raw base velocity", layout)
    if sample.shape != base.shape:
        raise ValueError("sampler sample and raw base velocity must have matching rows")
    result = sample - _sigma_rows(actual_sigma, len(sample)) * base
    if not np.isfinite(result).all():
        raise ValueError("clean estimate is nonfinite")
    if layout == "full128":
        result[:, ACTIVE_SLOTS:] = 0
    return result


def clean_residual_to_velocity(delta_clean, actual_sigma, *, layout):
    """Convert a desired clean correction to delta_v = -delta_clean / sigma.

    This returns a velocity correction, not an adapted total velocity. The
    caller adds it to the raw velocity and owns any deployment dtype rounding.
    """
    delta = _action(delta_clean, "clean residual", layout)
    result = -delta / _sigma_rows(actual_sigma, len(delta))
    if not np.isfinite(result).all():
        raise ValueError("velocity correction is nonfinite")
    if layout == "full128":
        result[:, ACTIVE_SLOTS:] = 0
    return result


def clean_features(hidden, c0_active, *, explicit_features=None):
    """Copy [hidden, C0_active10, optional caller-defined columns].

    No time basis, sigma weighting, or feature grouping is selected here. A
    caller may instead pass its own explicit matrix directly to ``fit`` and
    prediction, provided its column meaning/order stays fixed. A fitted model
    uses one center and one global vector L2 RMS over that supplied matrix.
    """
    hidden = _matrix(hidden, "hidden")
    c0 = _action(c0_active, "C0", "active10")
    blocks = [hidden, c0]
    if explicit_features is not None:
        blocks.append(_matrix(explicit_features, "explicit features"))
    if any(len(block) != len(hidden) for block in blocks):
        raise ValueError("all feature blocks must have matching rows")
    return np.concatenate(blocks, axis=1)


@dataclass(frozen=True, eq=False)
class CleanResidualReadout:
    """Immutable weighted-ridge clean residual predictor with training statistics.

    The objective is sum_i w_i ||phi_i W - residual_i||^2 + ridge ||W||^2,
    including the bias in regularization. Only explicitly supplied fit rows
    determine the center/scale; prediction never updates state. Targets and
    predictions are active10 clean residuals, never controller commands.
    """

    coefficients: np.ndarray
    center: np.ndarray
    scale: float
    ridge: float

    def __post_init__(self):
        coefficients = _matrix(self.coefficients, "coefficients", columns=ACTIVE_SLOTS)
        center = _array(self.center, "center").astype(np.float64, copy=True)
        if center.ndim != 1 or len(center) < 1 or coefficients.shape[0] != len(center) + 1:
            raise ValueError("center and coefficients must describe matching features plus one bias")
        scale = _positive_scalar(self.scale, "global scale")
        ridge = _positive_scalar(self.ridge, "ridge")
        coefficients.setflags(write=False)
        center.setflags(write=False)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "ridge", ridge)

    @classmethod
    def fit(cls, features, target_clean_residual, weights, *, ridge):
        """Fit only these rows; weights must be positive and sum to one.

        ``ridge`` has no default, so the experiment must choose it explicitly.
        No validation arrays or episode-based selection are accepted implicitly.
        """
        features = _matrix(features, "fit features")
        target = _action(target_clean_residual, "target clean residual", "active10")
        weights = _array(weights, "fit weights").astype(np.float64, copy=True)
        ridge = _positive_scalar(ridge, "ridge")
        if len(target) != len(features):
            raise ValueError("fit features and targets must have matching rows")
        if (
            weights.shape != (len(features),)
            or np.any(weights <= 0)
            or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-12)
        ):
            raise ValueError("fit weights must be positive and sum to one")
        center = np.sum(weights[:, None] * features, axis=0, dtype=np.float64)
        centered = features - center
        scale = np.sqrt(np.sum(weights * np.sum(centered**2, axis=1, dtype=np.float64), dtype=np.float64))
        scale = _positive_scalar(scale, "training global scale")
        phi = np.column_stack((centered / scale, np.ones(len(features))))
        normal = phi.T @ (weights[:, None] * phi) + ridge * np.eye(phi.shape[1])
        coefficients = np.linalg.solve(normal, phi.T @ (weights[:, None] * target))
        return cls(coefficients, center, scale, ridge)

    def predict_clean_residual(self, features):
        features = _matrix(features, "prediction features", columns=len(self.center))
        phi = np.column_stack(((features - self.center) / self.scale, np.ones(len(features))))
        result = phi @ self.coefficients
        if not np.isfinite(result).all():
            raise ValueError("predicted clean residual is nonfinite")
        return result

    def predict_velocity_delta(self, features, actual_sigma, *, layout):
        """Predict an active or zero-padded full velocity correction explicitly."""
        delta = self.predict_clean_residual(features)
        if layout == "full128":
            delta = full128_from_active10(delta)
        return clean_residual_to_velocity(delta, actual_sigma, layout=layout)

    def to_parameters(self):
        """Return independent JSON-serializable parameters without runtime state."""
        return dict(
            schema_version=PARAMETER_SCHEMA,
            coefficients=self.coefficients.tolist(),
            center=self.center.tolist(),
            scale=self.scale,
            ridge=self.ridge,
        )

    @classmethod
    def from_parameters(cls, parameters):
        expected = {"schema_version", "coefficients", "center", "scale", "ridge"}
        if not isinstance(parameters, dict) or set(parameters) != expected:
            raise ValueError("clean readout parameters must have exactly the registered fields")
        if parameters["schema_version"] != PARAMETER_SCHEMA:
            raise ValueError("unknown clean readout parameter schema")
        return cls(*(parameters[key] for key in ("coefficients", "center", "scale", "ridge")))
