"""Synthetic independent weighted-LS checks for bounded-memory clean fitting."""

import json
import weakref

import numpy as np
import pytest

from s2a.wam_clean_readout import CleanResidualReadout
from s2a.wam_clean_streaming import CleanFitBlock, CleanFitLayout, fit_clean_residual_streaming


def layout(columns):
    return CleanFitLayout(tuple(f"feature:{i}" for i in range(columns)), "active10")


def factory(features, target, weights, sizes, *, declared=None):
    declared = declared or layout(features.shape[1])

    def replay():
        offset = 0
        for size in sizes:
            stop = min(offset + size, len(features))
            if stop > offset:
                yield CleanFitBlock(features[offset:stop], target[offset:stop], weights[offset:stop], declared)
            offset = stop
        if offset < len(features):
            yield CleanFitBlock(features[offset:], target[offset:], weights[offset:], declared)

    return replay


@pytest.mark.parametrize("sizes", [(1,) * 41, (3, 10, 2), (41,), (7, 19, 15)])
@pytest.mark.parametrize("weight_kind", ["uniform", "nonuniform"])
def test_stream_fit_matches_direct_and_independent_augmented_ls(sizes, weight_kind):
    rng = np.random.default_rng(42)
    features = rng.normal(size=(41, 7)) + np.linspace(-3, 5, 7)
    target = rng.normal(size=(41, 10)) + 2
    weights = np.ones(41) if weight_kind == "uniform" else np.geomspace(1e-3, 20, 41)
    weights /= weights.sum()
    originals = [v.copy() for v in (features, target, weights)]
    fitted = fit_clean_residual_streaming(factory(features, target, weights, sizes), layout=layout(7), ridge=0.1)
    direct = CleanResidualReadout.fit(features, target, weights, ridge=0.1)
    center = np.sum(weights[:, None] * features, axis=0)
    scale = np.sqrt(np.sum(weights[:, None] * (features - center) ** 2))
    phi = np.column_stack(((features - center) / scale, np.ones(len(features))))
    independent = np.linalg.lstsq(
        np.vstack((np.sqrt(weights)[:, None] * phi, np.sqrt(0.1) * np.eye(phi.shape[1]))),
        np.vstack((np.sqrt(weights)[:, None] * target, np.zeros((phi.shape[1], 10)))),
        rcond=None,
    )[0]
    for expected in (direct.coefficients, independent):
        np.testing.assert_allclose(fitted.readout.coefficients, expected, rtol=2e-12, atol=2e-14)
    np.testing.assert_allclose(fitted.readout.center, direct.center, rtol=1e-14, atol=2e-15)
    np.testing.assert_allclose(fitted.readout.scale, direct.scale, rtol=1e-14, atol=2e-15)
    prediction = rng.normal(size=(4, 7))
    np.testing.assert_allclose(
        fitted.readout.predict_clean_residual(prediction), direct.predict_clean_residual(prediction), atol=2e-14
    )
    for current, original in zip((features, target, weights), originals, strict=True):
        np.testing.assert_array_equal(current, original)
    metadata = json.loads(json.dumps(fitted.fit_metadata()))
    assert metadata["rows"] == 41 and metadata["passes"] == 2 and metadata["replay_verified"] is True
    assert metadata["feature_columns"] == list(layout(7).feature_columns)
    assert metadata["target_layout"] == "active10" and metadata["bias_regularization"] == "same_ridge_as_features"
    assert metadata["ridge"] == 0.1 and metadata["compute_dtype"] == "float64"


def test_replay_can_change_block_boundaries_strides_and_dtype_with_one_solve(monkeypatch):
    rng = np.random.default_rng(17)
    features, target = rng.normal(size=(23, 3)).astype(np.float32), rng.normal(size=(23, 10)).astype(np.float32)
    weights = np.full(23, 1 / 23)
    calls, solves = [], []
    original_solve = np.linalg.solve

    def solve(*args):
        solves.append(True)
        return original_solve(*args)

    def replay():
        calls.append(True)
        if len(calls) == 1:
            return factory(features, target, weights, (2, 8))()
        return factory(np.asfortranarray(features, dtype=np.float64), target, weights, (7, 1, 9))()

    monkeypatch.setattr(np.linalg, "solve", solve)
    result = fit_clean_residual_streaming(replay, layout=layout(3), ridge=0.1)
    assert len(calls) == 2 and len(solves) == 1 and result.block_counts == (3, 4)
    expected = CleanResidualReadout.fit(features, target, weights, ridge=0.1)
    np.testing.assert_allclose(result.readout.coefficients, expected.coefficients, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("changed", ["features", "targets", "weights", "row_order", "layout", "missing_rows"])
def test_second_pass_changed_stream_rejected_before_solve(changed, monkeypatch):
    features = np.arange(24, dtype=float).reshape(8, 3)
    target, weights = np.arange(80, dtype=float).reshape(8, 10), np.full(8, 1 / 8)
    calls = []

    def never_solve(*args):
        raise AssertionError("replay mismatches must be rejected before fitting")

    def replay():
        calls.append(True)
        x, y, w, declared = features.copy(), target.copy(), weights.copy(), layout(3)
        if len(calls) == 2:
            if changed == "features":
                x[0, 0] += 1e-12
            elif changed == "targets":
                y[0, 0] += 1e-12
            elif changed == "weights":
                w[:2] += [1e-12, -1e-12]
            elif changed == "row_order":
                x, y, w = x[::-1], y[::-1], w[::-1]
            elif changed == "layout":
                declared = CleanFitLayout(declared.feature_columns[::-1], "active10")
            else:
                x, y, w = x[:-1], y[:-1], w[:-1]
        yield CleanFitBlock(x, y, w, declared)

    monkeypatch.setattr(np.linalg, "solve", never_solve)
    with pytest.raises(ValueError, match="replay changed|exact fit.*layout"):
        fit_clean_residual_streaming(replay, layout=layout(3), ridge=0.1)


def test_weight_sum_tolerance_and_regularized_bias_follow_existing_fit():
    features = np.asarray([[10.0], [11.0], [12.0]])
    target = np.full((3, 10), 4.0)
    weights = np.full(3, (1 + 4e-13) / 3)
    fitted = fit_clean_residual_streaming(factory(features, target, weights, (1, 1)), layout=layout(1), ridge=0.1)
    direct = CleanResidualReadout.fit(features, target, weights, ridge=0.1)
    np.testing.assert_allclose(fitted.readout.center, direct.center, rtol=0, atol=2e-15)
    np.testing.assert_allclose(fitted.readout.scale, direct.scale, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(fitted.readout.coefficients, direct.coefficients, rtol=1e-12, atol=1e-14)
    assert np.all(fitted.readout.coefficients[-1] < 4)  # Bias receives the same ridge penalty.


@pytest.mark.parametrize(
    "bad",
    [
        "nan_features",
        "inf_targets",
        "nan_weights",
        "zero_weight",
        "negative_weight",
        "bad_sum",
        "bad_shape",
        "bad_rows",
    ],
)
def test_invalid_values_and_shapes_rejected(bad):
    x, y, w = np.arange(12, dtype=float).reshape(4, 3), np.ones((4, 10)), np.full(4, 0.25)
    if bad == "nan_features":
        x[0, 0] = np.nan
    elif bad == "inf_targets":
        y[0, 0] = np.inf
    elif bad == "nan_weights":
        w[0] = np.nan
    elif bad == "zero_weight":
        w[0] = 0
    elif bad == "negative_weight":
        w[0] = -1
    elif bad == "bad_sum":
        w *= 2
    elif bad == "bad_shape":
        w = w[:, None]
    else:
        y = y[:-1]
    with pytest.raises(ValueError):
        fit_clean_residual_streaming(lambda: [CleanFitBlock(x, y, w, layout(3))], layout=layout(3), ridge=0.1)


def test_explicit_layout_empty_one_shot_and_zero_scale_rejected():
    for columns in ("feature", (), ("a", "a"), ("",)):
        with pytest.raises(ValueError, match="feature"):
            CleanFitLayout(columns, "active10")
    with pytest.raises(ValueError, match="active10"):
        CleanFitLayout(("a",), "full128")
    with pytest.raises(ValueError, match="CleanFitLayout"):
        fit_clean_residual_streaming(lambda: [], layout=None, ridge=0.1)
    with pytest.raises(ValueError, match="callable"):
        fit_clean_residual_streaming([], layout=layout(1), ridge=0.1)
    with pytest.raises(ValueError, match="nonempty"):
        fit_clean_residual_streaming(lambda: [], layout=layout(1), ridge=0.1)
    block = CleanFitBlock(np.asarray([[0.0], [1.0]]), np.zeros((2, 10)), np.full(2, 0.5), layout(1))
    one_shot = iter([block])
    with pytest.raises(ValueError, match="replay changed"):
        fit_clean_residual_streaming(lambda: one_shot, layout=layout(1), ridge=0.1)
    with pytest.raises(ValueError, match="global scale"):
        fit_clean_residual_streaming(
            lambda: [CleanFitBlock(np.zeros((2, 1)), block.target_clean_residual, block.weights, layout(1))],
            layout=layout(1),
            ridge=0.1,
        )
    with pytest.raises(ValueError, match="ridge"):
        fit_clean_residual_streaming(lambda: [block], layout=layout(1), ridge=True)


def test_420k_synthetic_rows_replay_without_retaining_source_blocks():
    # The row count matches the planned order of magnitude, while a small D
    # keeps this a cheap CPU resource-contract test rather than a model fit.
    rows, block_rows = 420_000, 1024
    references, peak_live = [], []
    declared = layout(3)

    def replay():
        for start in range(0, rows, block_rows):
            index = np.arange(start, min(start + block_rows, rows), dtype=np.float64)
            features = np.column_stack((index % 23, index % 31, index % 47))
            references.append(weakref.ref(features))
            peak_live.append(sum(reference() is not None for reference in references))
            target = np.repeat((index % 5)[:, None], 10, axis=1)
            yield CleanFitBlock(features, target, np.full(len(index), 1 / rows), declared)

    fitted = fit_clean_residual_streaming(replay, layout=declared, ridge=0.1)
    assert fitted.rows == rows and fitted.block_counts == (411, 411)
    assert max(peak_live) <= 2 and all(reference() is None for reference in references)
    assert np.isfinite(fitted.readout.coefficients).all()
