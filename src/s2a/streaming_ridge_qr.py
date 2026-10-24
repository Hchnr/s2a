"""Independent two-pass FP64 weighted ridge audit using augmented QR.

This module has no production-fit, feature-normalization, model or file imports.
The caller supplies all feature/target/weight rows twice, in the same order; only
block boundaries may change. No source rows survive the fit. The retained work
is O(D**2 + B*D + D*K), with D features plus bias, B bounded block rows and K
targets. All columns, including the final bias, receive the same ridge penalty.
"""

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular


def _require(value, message):
    if not value:
        raise ValueError(message)


def _positive(value, name):
    array = np.asarray(value)
    _require(
        array.ndim == 0 and array.dtype.kind in "fiu" and np.isfinite(array) and array > 0,
        f"{name} must be a finite positive scalar",
    )
    return float(array)


def _fp64(array, name):
    array = np.asarray(array)
    _require(array.dtype.kind in "fiu" and np.isfinite(array).all(), f"{name} contains invalid real values")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(array, dtype=np.float64, order="C", copy=True)
    _require(np.isfinite(result).all(), f"{name} cannot be represented as finite FP64")
    return result


@dataclass(frozen=True, eq=False)
class IndependentRidgeFit:
    coefficients: np.ndarray
    center: np.ndarray
    scale: float
    ridge: float
    rows: int
    feature_columns: tuple[str, ...]
    target_columns: int
    weight_sum: float
    block_counts: tuple[int, int]
    replay_sha256: tuple[str, str, str]

    def __post_init__(self):
        coefficients, center = self.coefficients.copy(), self.center.copy()
        coefficients.setflags(write=False)
        center.setflags(write=False)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "center", center)

    def metadata(self):
        """Return a serializable receipt; source/row identities are caller-owned."""
        return dict(
            schema_version="s2a.independent-streaming-ridge-qr.v1",
            method="two_pass_fp64_augmented_qr_no_normal_equations",
            rows=self.rows,
            feature_columns=list(self.feature_columns),
            target_columns=self.target_columns,
            ridge=self.ridge,
            weight_sum=self.weight_sum,
            scale=self.scale,
            feature_scaling="weighted_global_vector_l2_rms",
            bias_regularization="same_ridge_as_features",
            passes=2,
            block_counts=list(self.block_counts),
            replay_verified=True,
            replay_sha256=dict(zip(("features", "targets", "weights"), self.replay_sha256, strict=True)),
        )


class _Rows:
    def __init__(self, expected_rows, columns, targets, max_block_rows):
        self.expected_rows = expected_rows
        self.columns, self.targets, self.max_block_rows = columns, targets, max_block_rows
        self.rows = self.blocks = 0
        self.digests = [hashlib.sha256() for _ in range(3)]
        self.weight_sum = 0.0

    def consume(self, block):
        _require(
            isinstance(block, (tuple, list)) and len(block) == 3,
            "each block must explicitly contain features, targets, global_weights",
        )
        features, targets, weights = map(np.asarray, block)
        _require(features.ndim == 2 and features.shape[1] == self.columns, "feature column count/order differs")
        rows = len(features)
        _require(0 < rows <= self.max_block_rows, "empty or oversized block exceeds max_block_rows")
        _require(targets.shape == (rows, self.targets) and weights.shape == (rows,), "target/weight row layout differs")
        _require(self.rows + rows <= self.expected_rows, "stream exceeds expected_rows; duplicate/extra rows")
        features, targets, weights = (
            _fp64(value, name)
            for value, name in zip((features, targets, weights), ("features", "targets", "weights"), strict=True)
        )
        _require(np.all(weights > 0), "global row weights must be strictly positive")
        for digest, value in zip(self.digests, (features, targets, weights), strict=True):
            # Canonical row-major FP64 bytes make hashes independent of block
            # boundaries, original dtype, native endian and input strides.
            digest.update(np.asarray(value, dtype="<f8", order="C").tobytes(order="C"))
        self.rows += rows
        self.blocks += 1
        self.weight_sum += float(np.sum(weights, dtype=np.float64))
        _require(np.isfinite(self.weight_sum), "global weight sum overflowed")
        return features, targets, weights

    def finish(self):
        _require(self.rows == self.expected_rows, "stream row count differs from expected_rows; rows missing")
        _require(np.isclose(self.weight_sum, 1.0, rtol=0, atol=1e-12), "global row weights must sum to one")
        return tuple(digest.hexdigest() for digest in self.digests)


def independent_ridge(
    block_factory, *, expected_rows, feature_columns, target_columns=10, ridge=0.1, max_block_rows=1400
):
    """Read every row twice, then solve one independent augmented-QR objective.

    ``feature_columns`` is the ordered unique semantic column-name tuple. The
    center is ``sum(w*x)`` under globally unit row weights, including the allowed
    1e-12 sum tolerance. The scale is ``sqrt(sum(w*||x-center||**2))``. Stable
    weighted central moments avoid subtracting large uncentered square sums.
    Pass two begins with sqrt(ridge)*I; each bounded block QR updates only R and
    Q-transformed targets. No Gram matrix or normal-equation solve is formed.
    """
    _require(callable(block_factory), "block_factory must replay all rows on each invocation")
    _require(type(expected_rows) is int and expected_rows > 0, "expected_rows must be a positive integer")
    _require(
        isinstance(feature_columns, tuple)
        and feature_columns
        and all(isinstance(name, str) and name.strip() for name in feature_columns)
        and len(set(feature_columns)) == len(feature_columns),
        "feature_columns must be an ordered unique name tuple",
    )
    _require(type(target_columns) is int and target_columns > 0, "target_columns must be positive integer")
    _require(type(max_block_rows) is int and 0 < max_block_rows <= 1400, "max_block_rows must be1..1400")
    ridge = _positive(ridge, "ridge")
    columns = len(feature_columns)
    first = _Rows(expected_rows, columns, target_columns, max_block_rows)
    weighted_sum, mean = np.zeros(columns), np.zeros(columns)
    mass, centered_squares = 0.0, 0.0
    for block in block_factory():
        features, _, weights = first.consume(block)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            block_mass = float(weights.sum(dtype=np.float64))
            block_sum = np.sum(weights[:, None] * features, axis=0, dtype=np.float64)
            block_mean = block_sum / block_mass
            block_squares = float(np.sum(weights * np.sum((features - block_mean) ** 2, axis=1), dtype=np.float64))
            new_mass = mass + block_mass
            delta = block_mean - mean
            centered_squares += block_squares + float(np.dot(delta, delta)) * mass * block_mass / new_mass
            mean += delta * (block_mass / new_mass)
            weighted_sum += block_sum
            mass = new_mass
        _require(
            np.isfinite(weighted_sum).all() and np.isfinite(mean).all() and np.isfinite(centered_squares),
            "FP64 normalization statistics overflowed",
        )
    first_hashes = first.finish()
    center = weighted_sum
    with np.errstate(over="ignore", invalid="ignore"):
        scale = np.sqrt(centered_squares + mass * float(np.dot(mean - center, mean - center)))
    scale = _positive(scale, "global feature scale")
    dimensions = columns + 1
    factor = np.sqrt(ridge) * np.eye(dimensions, dtype=np.float64)
    projected = np.zeros((dimensions, target_columns), dtype=np.float64)
    second = _Rows(expected_rows, columns, target_columns, max_block_rows)
    for block in block_factory():
        features, targets, weights = second.consume(block)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            design = np.column_stack(((features - center) / scale, np.ones(len(features))))
            root_weights = np.sqrt(weights)[:, None]
            augmented = np.vstack((factor, root_weights * design))
            responses = np.vstack((projected, root_weights * targets))
        _require(np.isfinite(augmented).all() and np.isfinite(responses).all(), "FP64 augmented QR input overflowed")
        q, factor = np.linalg.qr(augmented, mode="reduced")
        projected = q.T @ responses
        _require(np.isfinite(factor).all() and np.isfinite(projected).all(), "FP64 QR factor/projection is nonfinite")
    second_hashes = second.finish()
    _require(first_hashes == second_hashes, "replay changed row order, features, targets or weights between passes")
    coefficients = solve_triangular(factor, projected, lower=False, check_finite=False)
    _require(np.isfinite(coefficients).all(), "FP64 triangular solution is nonfinite")
    return IndependentRidgeFit(
        coefficients,
        center,
        scale,
        ridge,
        expected_rows,
        feature_columns,
        target_columns,
        first.weight_sum,
        (first.blocks, second.blocks),
        first_hashes,
    )
