"""Two-pass CPU ridge fitting without retaining all clean-readout token rows.

The caller supplies a replayable block factory and the meaning/order of every
feature column. This module neither selects data nor constructs a time basis.
It returns the existing :class:`CleanResidualReadout` and uses its FP64 weighted
objective, one global vector L2 RMS, and regularized bias. Block reductions may
round differently from the direct fit; bitwise equality is not promised.
"""

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from s2a.wam_clean_readout import (
    ACTIVE_SLOTS,
    CleanResidualReadout,
    _action,
    _array,
    _matrix,
    _positive_scalar,
)

STREAMING_SCHEMA = "s2a.clean-residual-streaming-fit.v1"


@dataclass(frozen=True)
class CleanFitLayout:
    """Caller-defined feature order and explicit active clean-target layout."""

    feature_columns: tuple[str, ...]
    target_layout: str

    def __post_init__(self):
        if isinstance(self.feature_columns, str):
            raise ValueError("feature columns must be an explicit ordered sequence")
        try:
            columns = tuple(self.feature_columns)
        except TypeError as error:
            raise ValueError("feature columns must be an explicit ordered sequence") from error
        if not columns or any(not isinstance(x, str) or not x.strip() for x in columns):
            raise ValueError("feature columns must contain nonempty names")
        if len(set(columns)) != len(columns):
            raise ValueError("feature column names must be unique")
        if self.target_layout != "active10":
            raise ValueError("streaming target layout must explicitly be active10")
        object.__setattr__(self, "feature_columns", columns)


@dataclass(frozen=True, eq=False)
class CleanFitBlock:
    """One bounded batch; arrays are copied and validated when consumed.

    Weights are global row weights, not normalized separately in each block.
    The block factory must preserve all values and row order on its second
    invocation; it may change block boundaries. No arrays are kept by the fit.
    """

    features: np.ndarray
    target_clean_residual: np.ndarray
    weights: np.ndarray
    layout: CleanFitLayout


@dataclass(frozen=True, eq=False)
class StreamingCleanFit:
    """Fitted ordinary readout plus replay/normalization evidence, without rows."""

    readout: CleanResidualReadout
    layout: CleanFitLayout
    rows: int
    block_counts: tuple[int, int]
    weight_sum: float
    replay_sha256: tuple[str, str, str]

    def fit_metadata(self):
        """Independent JSON-ready metadata; source/row provenance is caller-owned."""
        return dict(
            schema_version=STREAMING_SCHEMA,
            rows=self.rows,
            feature_columns=list(self.layout.feature_columns),
            target_layout=self.layout.target_layout,
            compute_dtype="float64",
            feature_scaling="weighted_global_vector_l2_rms",
            bias_regularization="same_ridge_as_features",
            ridge=self.readout.ridge,
            weight_sum=self.weight_sum,
            passes=2,
            block_counts=list(self.block_counts),
            replay_verified=True,
            replay_sha256=dict(zip(("features", "target_clean_residual", "weights"), self.replay_sha256, strict=True)),
        )


class _Replay:
    def __init__(self, layout):
        self.layout = layout
        self.rows = 0
        self.blocks = 0
        self.digests = [hashlib.sha256() for _ in range(3)]

    def consume(self, block):
        if not isinstance(block, CleanFitBlock) or block.layout != self.layout:
            raise ValueError("each block must declare the exact fit feature/target layout")
        features = _matrix(block.features, "fit features", columns=len(self.layout.feature_columns))
        target = _action(block.target_clean_residual, "target clean residual", self.layout.target_layout)
        weights = _array(block.weights, "fit weights").astype(np.float64, copy=True)
        if len(target) != len(features):
            raise ValueError("fit features and targets must have matching rows")
        if weights.shape != (len(features),) or np.any(weights <= 0):
            raise ValueError("fit weights must be positive with one weight per row")
        for digest, values in zip(self.digests, (features, target, weights), strict=True):
            # Explicit endian/row order makes the evidence independent of input
            # strides, native endian, original dtype, and replay block boundaries.
            digest.update(np.asarray(values, dtype="<f8", order="C").tobytes(order="C"))
        self.rows += len(features)
        self.blocks += 1
        return features, target, weights

    def signature(self):
        return self.rows, tuple(digest.hexdigest() for digest in self.digests)


def _normalization(blocks, layout):
    replay = _Replay(layout)
    columns = len(layout.feature_columns)
    weight_sum = 0.0
    weighted_sum = np.zeros(columns, dtype=np.float64)
    mean = np.zeros(columns, dtype=np.float64)
    centered_sum_squares = 0.0
    for block in blocks:
        features, _, weights = replay.consume(block)
        block_weight = float(np.sum(weights, dtype=np.float64))
        block_sum = np.sum(weights[:, None] * features, axis=0, dtype=np.float64)
        block_mean = block_sum / block_weight
        block_ss = float(np.sum(weights * np.sum((features - block_mean) ** 2, axis=1), dtype=np.float64))
        next_weight = weight_sum + block_weight
        delta = block_mean - mean
        if weight_sum:
            centered_sum_squares += float(np.dot(delta, delta)) * weight_sum * block_weight / next_weight
        centered_sum_squares += block_ss
        mean += delta * (block_weight / next_weight)
        weighted_sum += block_sum
        weight_sum = next_weight
        if not (
            np.isfinite(weight_sum)
            and np.isfinite(weighted_sum).all()
            and np.isfinite(mean).all()
            and np.isfinite(centered_sum_squares)
        ):
            raise ValueError("training normalization statistics are nonfinite")
    if not replay.rows:
        raise ValueError("fit replay must contain nonempty blocks")
    if not np.isclose(weight_sum, 1, rtol=0, atol=1e-12):
        raise ValueError("global fit weights must sum to one")
    # The direct primitive uses sum(w*x), rather than sum(w*x)/sum(w),
    # for its center even inside the allowed weight-sum tolerance. Translate
    # the stable weighted central moment to precisely that center convention.
    center = weighted_sum
    centered_sum_squares += weight_sum * float(np.dot(mean - center, mean - center))
    scale = _positive_scalar(np.sqrt(centered_sum_squares), "training global scale")
    return replay, center, scale, weight_sum


def fit_clean_residual_streaming(
    replay_blocks: Callable[[], Iterable[CleanFitBlock]], *, layout: CleanFitLayout, ridge
):
    """Replay twice and solve weighted ridge once, returning the ordinary head.

    Peak retained storage is O(D**2 + B*D), where D is the explicitly named
    feature count and B the largest caller-provided block. No row-count budget
    or shuffling is chosen here. Positive weights must sum to one globally;
    ``ridge`` is explicit and also regularizes the final bias column.

    Pass one merges weighted central moments for stable global L2 RMS and
    computes the direct primitive's weighted center. Pass two accumulates the
    augmented Gram matrix/RHS. All three value streams must replay identically
    before the single solve. Thus validation cannot silently use changed rows,
    labels, weights, column order, or a one-shot exhausted iterator.
    """
    if not isinstance(layout, CleanFitLayout):
        raise ValueError("an explicit CleanFitLayout is required")
    if not callable(replay_blocks):
        raise ValueError("replay_blocks must be a callable returning the complete row stream")
    ridge = _positive_scalar(ridge, "ridge")
    first, center, scale, weight_sum = _normalization(replay_blocks(), layout)
    dimensions = len(center) + 1
    normal = np.zeros((dimensions, dimensions), dtype=np.float64)
    rhs = np.zeros((dimensions, ACTIVE_SLOTS), dtype=np.float64)
    second = _Replay(layout)
    for block in replay_blocks():
        features, target, weights = second.consume(block)
        phi = np.column_stack(((features - center) / scale, np.ones(len(features))))
        normal += phi.T @ (weights[:, None] * phi)
        rhs += phi.T @ (weights[:, None] * target)
        if not np.isfinite(normal).all() or not np.isfinite(rhs).all():
            raise ValueError("training normal equations are nonfinite")
    if first.signature() != second.signature():
        raise ValueError("fit replay changed features, targets, weights, or row order between passes")
    normal.flat[:: dimensions + 1] += ridge
    coefficients = np.linalg.solve(normal, rhs)
    readout = CleanResidualReadout(coefficients, center, scale, ridge)
    return StreamingCleanFit(
        readout,
        layout,
        first.rows,
        (first.blocks, second.blocks),
        weight_sum,
        first.signature()[1],
    )
