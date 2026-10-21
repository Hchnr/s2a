"""Small frozen-backbone readout with an explicit full-domain request boundary.

This adapts generated velocity; it is not a robot response inverse. Torch is
optional and imported only by the deployment path. Fit/apply never read labels
or task conditions other than the arrays and domain context supplied by caller.
"""

import contextlib
import contextvars
from dataclasses import dataclass

import numpy as np

LIBERO_DOMAIN = "franka_panda__parallel_gripper/eef20/observed_local_delta/libero_closed_pad_observed_endpoint_eef10_v1"
NORMALIZATION_ID = 31
PROJECTION_ID = 0
HIDDEN_SIZE = 1280
ACTIVE_SLOTS = 10
ACTION_SIZE = 128
RIDGE = 0.1
ACTIVE_MASK = (True,) * ACTIVE_SLOTS + (False,) * (ACTION_SIZE - ACTIVE_SLOTS)


@dataclass(frozen=True)
class DomainContext:
    """One homogeneous request, established before the shared projection head."""

    domain: str
    normalization_id: int
    projection_id: int
    action_dim_mask: tuple

    def __post_init__(self):
        if not isinstance(self.domain, str) or not self.domain.strip():
            raise ValueError("requires the full nonempty action domain")
        for name in ("normalization_id", "projection_id"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        mask = tuple(self.action_dim_mask)
        if len(mask) != ACTION_SIZE or not all(isinstance(v, (bool, np.bool_)) for v in mask):
            raise ValueError("requires an explicit128-slot boolean dimension mask")
        object.__setattr__(self, "action_dim_mask", tuple(map(bool, mask)))
        named = self.domain == LIBERO_DOMAIN
        indexed = self.normalization_id == NORMALIZATION_ID
        if named != indexed or (named and (self.projection_id != PROJECTION_ID or mask != ACTIVE_MASK)):
            raise ValueError("full LIBERO domain, normalization31, projection0 and active10 must agree")

    @property
    def is_libero(self):
        return self.domain == LIBERO_DOMAIN


def _matrix(value, name, *, columns=None):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not result.shape[0] or not result.shape[1] or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite nonempty matrix")
    if columns is not None and result.shape[1] != columns:
        raise ValueError(f"{name} has the wrong width")
    return result.copy()


def features(hidden):
    """Fixed phi=[h/sqrt(D),1]; the registered deployment entry requires D=1280."""
    hidden = _matrix(hidden, "hidden")
    return np.column_stack((hidden / np.sqrt(hidden.shape[1]), np.ones(len(hidden))))


def _fit_inputs(hidden, residual_target, weights):
    phi = features(hidden)
    target = _matrix(residual_target, "residual target", columns=ACTIVE_SLOTS)
    weights = np.asarray(weights, dtype=np.float64)
    if (
        len(target) != len(phi)
        or weights.shape != (len(phi),)
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
        or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError("fit requires matching rows and positive weights summing to one")
    return phi, target, weights.copy()


def _bfloat16_rne(value):
    """Round FP64 directly to BF16, avoiding Torch CPU's FP32 intermediate.

    BF16 has seven explicit fraction bits and minimum subnormal spacing2^-133.
    Quantize in FP64 first; the final cast receives exactly representable BF16
    values. torch.round implements ties to even and preserves signed zero.
    """
    import torch

    if value.dtype != torch.float64:
        raise ValueError("BF16 single rounding requires an FP64 input")
    _, exponent = torch.frexp(value)
    spacing = torch.ldexp(torch.ones_like(value), torch.clamp(exponent - 8, min=-133))
    rounded = torch.round(value / spacing) * spacing
    return rounded.to(torch.bfloat16)


@dataclass(frozen=True)
class ResidualReadout:
    """Zero-prior ridge coefficients, including a regularized final bias row."""

    weights: np.ndarray

    def __post_init__(self):
        weights = _matrix(self.weights, "readout weights", columns=ACTIVE_SLOTS)
        if len(weights) < 2:
            raise ValueError("readout needs at least one hidden feature and a bias")
        weights.setflags(write=False)
        object.__setattr__(self, "weights", weights)

    @property
    def hidden_size(self):
        return len(self.weights) - 1

    @property
    def is_zero(self):
        return not np.any(self.weights)

    @classmethod
    def zero(cls, hidden_size=HIDDEN_SIZE):
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        return cls(np.zeros((hidden_size + 1, ACTIVE_SLOTS), dtype=np.float64))

    @classmethod
    def fit(cls, hidden, residual_target, weights):
        """One fixed ridge solve. Caller owns the episode/anchor/sigma split."""
        phi, target, weights = _fit_inputs(hidden, residual_target, weights)
        normal = phi.T @ (weights[:, None] * phi) + RIDGE * np.eye(phi.shape[1])
        rhs = phi.T @ (weights[:, None] * target)
        return cls(np.linalg.solve(normal, rhs))

    @classmethod
    def fit_bias(cls, hidden, residual_target, weights):
        """The registered constant-bias control uses the same target/weights/ridge."""
        phi, target, weights = _fit_inputs(hidden, residual_target, weights)
        result = np.zeros((phi.shape[1], ACTIVE_SLOTS), dtype=np.float64)
        result[-1] = (weights @ target) / (weights.sum() + RIDGE)
        return cls(result)

    def apply_torch(self, hidden, base, context, projection_ids):
        """Add only active10 in FP64, then cast once to the original base dtype.

        The shared projection ID alone never enables adaptation. Identity paths
        return the exact base object; all other118 slots remain byte-identical.
        """
        import torch

        if not isinstance(context, DomainContext):
            raise ValueError("requires an explicit full-domain request context")
        if not isinstance(base, torch.Tensor) or base.ndim != 2 or base.shape[1] != ACTION_SIZE:
            raise ValueError("base must be an N×128 tensor")
        if (
            not isinstance(projection_ids, torch.Tensor)
            or projection_ids.shape != (len(base),)
            or projection_ids.dtype != torch.long
            or not bool(torch.all(projection_ids == context.projection_id))
        ):
            raise ValueError("mixed or inconsistent per-token projection domain")
        if not context.is_libero or self.is_zero:
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
            raise ValueError("requires matching finite floating hidden/base tensors on one device")
        phi = torch.cat(
            (
                hidden.to(torch.float64) / np.sqrt(self.hidden_size),
                torch.ones((len(base), 1), dtype=torch.float64, device=base.device),
            ),
            dim=1,
        )
        coefficients = torch.tensor(self.weights.copy(), dtype=torch.float64, device=base.device)
        residual = phi @ coefficients
        summed = base[:, :ACTIVE_SLOTS].to(torch.float64) + residual
        active = _bfloat16_rne(summed) if base.dtype == torch.bfloat16 else summed.to(base.dtype)
        if not bool(torch.isfinite(active).all()):
            raise ValueError("readout generated nonfinite output after deployment quantization")
        result = base.clone()
        result[:, :ACTIVE_SLOTS] = active
        return result


class ReadoutHook:
    """Request-scoped hook around the original head; never edits head parameters."""

    def __init__(self, head, readout):
        if not isinstance(readout, ResidualReadout) or readout.hidden_size != HIDDEN_SIZE:
            raise ValueError("registered WAM hook requires hidden width1280")
        self.readout = readout
        self._context = contextvars.ContextVar(f"s2a_readout_context_{id(self)}", default=None)
        self._handle = head.register_forward_hook(self._forward, with_kwargs=True)

    @contextlib.contextmanager
    def request(self, context):
        if not isinstance(context, DomainContext) or self._context.get() is not None:
            raise ValueError("requires one non-nested homogeneous request context")
        token = self._context.set(context)
        try:
            yield self
        finally:
            self._context.reset(token)

    def _forward(self, module, args, kwargs, base):
        context = self._context.get()
        if context is None:
            raise ValueError("shared action head was called outside its request context")
        if kwargs or len(args) != 2:
            raise ValueError("registered head requires positional hidden and projection IDs")
        return self.readout.apply_torch(args[0], base, context, args[1])

    def remove(self):
        self._handle.remove()
