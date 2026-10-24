"""Independent synthetic QR gold checks; no actual captures or labels are read."""

import gc
import hashlib
import weakref

import numpy as np
import pytest

from s2a import streaming_ridge_qr as audit


def gold(features, targets, weights, ridge):
    # Full small augmented least squares is the external mathematical oracle;
    # it never accumulates QR blocks or uses the audit's normalization code.
    center = np.sum(weights[:, None] * features, axis=0)
    scale = np.sqrt(np.sum(weights * np.linalg.norm(features - center, axis=1) ** 2))
    phi = np.column_stack(((features - center) / scale, np.ones(len(features))))
    matrix = np.concatenate((np.sqrt(weights)[:, None] * phi, np.sqrt(ridge) * np.eye(phi.shape[1])))
    rhs = np.concatenate((np.sqrt(weights)[:, None] * targets, np.zeros((phi.shape[1], targets.shape[1]))))
    coefficients = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    return coefficients, center, scale


def replay(features, targets, weights, sizes):
    calls = 0

    def factory():
        nonlocal calls
        block_size = sizes[min(calls, len(sizes) - 1)]
        calls += 1
        for start in range(0, len(features), block_size):
            yield (
                features[start : start + block_size],
                targets[start : start + block_size],
                weights[start : start + block_size],
            )

    return factory


@pytest.mark.parametrize("sizes", ((61, 61), (1, 7), (13, 3)))
@pytest.mark.parametrize("ridge", (0.1, 100000.0))
def test_partition_independent_qr_matches_full_augmented_lstsq_and_regularizes_bias(sizes, ridge):
    rng = np.random.default_rng(42)
    features = rng.normal(size=(61, 7)) + np.arange(7) * 20
    targets = rng.normal(size=(61, 3)) + 8
    weights = rng.uniform(0.1, 1, len(features))
    weights /= weights.sum()
    expected, center, scale = gold(features, targets, weights, ridge)
    columns = tuple(f"hidden_{i:04d}" for i in range(7))
    actual = audit.independent_ridge(
        replay(features, targets, weights, sizes),
        expected_rows=61,
        feature_columns=columns,
        target_columns=3,
        ridge=ridge,
    )
    np.testing.assert_allclose(actual.center, center, rtol=1e-14, atol=1e-13)
    np.testing.assert_allclose(actual.scale, scale, rtol=1e-14, atol=1e-13)
    np.testing.assert_allclose(actual.coefficients, expected, rtol=1e-10, atol=1e-12)
    assert not actual.center.flags.writeable and not actual.coefficients.flags.writeable
    assert actual.rows == 61 and actual.feature_columns == columns and actual.target_columns == 3
    assert actual.replay_sha256 == tuple(
        hashlib.sha256(np.asarray(value, dtype="<f8", order="C").tobytes()).hexdigest()
        for value in (features, targets, weights)
    )
    metadata = actual.metadata()
    assert metadata["bias_regularization"] == "same_ridge_as_features" and metadata["replay_verified"]
    assert metadata["block_counts"] == [(61 + size - 1) // size for size in sizes]
    if ridge > 1:
        assert np.linalg.norm(actual.coefficients[-1]) < 0.001


def test_near_collinearity_and_tiny_row_weights_have_stable_regularized_solution(monkeypatch):
    rng = np.random.default_rng(7)
    first = rng.normal(size=73)
    features = np.column_stack((first, first, first + 1e-13 * rng.normal(size=73), np.ones(73) * 1e4))
    targets = np.column_stack((first * 3, np.sin(first)))
    weights = np.geomspace(1e-20, 1, 73)
    weights /= weights.sum()
    expected, _, _ = gold(features, targets, weights, 0.1)

    def forbidden(*args, **kwargs):
        pytest.fail("an independent QR audit must not solve normal equations")

    monkeypatch.setattr(audit.np.linalg, "solve", forbidden)
    actual = audit.independent_ridge(
        replay(features, targets, weights, (8, 17)),
        expected_rows=73,
        feature_columns=("same_a", "same_b", "near_same", "constant"),
        target_columns=2,
    )
    np.testing.assert_allclose(actual.coefficients, expected, rtol=1e-10, atol=1e-11)


def test_canonical_row_hash_survives_changed_boundaries_strides_and_native_endian():
    features = np.arange(72, dtype=np.float64).reshape(24, 3) / 8
    targets = features[:, :2] / 4
    weights = np.full(24, 1 / 24)
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        if calls == 1:
            yield features, targets, weights
        else:
            for start in range(0, 24, 5):
                yield (
                    np.asfortranarray(features[start : start + 5]).astype(">f8"),
                    targets[start : start + 5].copy(order="F"),
                    weights[start : start + 5].astype(">f8"),
                )

    actual = audit.independent_ridge(factory, expected_rows=24, feature_columns=("x", "y", "z"), target_columns=2)
    assert actual.block_counts == (1, 5) and actual.rows == 24


@pytest.mark.parametrize("change", ("features", "targets", "weights", "reorder", "missing", "extra", "one_shot"))
def test_replay_mutations_or_wrong_row_counts_reject_before_the_final_solve(change, monkeypatch):
    features = np.arange(60, dtype=np.float64).reshape(20, 3)
    targets = features[:, :2].copy()
    weights = np.full(20, 0.05)
    calls = 0
    iterator = iter(((features, targets, weights),))

    def factory():
        nonlocal calls
        calls += 1
        if change == "one_shot":
            return iterator
        x, y, w = features.copy(), targets.copy(), weights.copy()
        if calls == 2:
            if change == "features":
                x[5, 1] += 1e-6
            elif change == "targets":
                y[5, 1] += 1e-6
            elif change == "weights":
                w[0] += 0.001
                w[1] -= 0.001
            elif change == "reorder":
                x, y, w = x[::-1], y[::-1], w[::-1]
            elif change == "missing":
                x, y, w = x[:-1], y[:-1], w[:-1]
            else:
                x, y, w = np.vstack((x, x[:1])), np.vstack((y, y[:1])), np.r_[w, w[:1]]
        return iter(((x, y, w),))

    monkeypatch.setattr(audit, "solve_triangular", lambda *a, **k: pytest.fail("rejected replay must never solve"))
    with pytest.raises(ValueError, match="replay changed|expected_rows"):
        audit.independent_ridge(factory, expected_rows=20, feature_columns=("x", "y", "z"), target_columns=2)


def test_bounded_qr_work_and_no_retained_source_rows(monkeypatch):
    references, shapes = [], []
    original_qr = audit.np.linalg.qr

    def observed_qr(matrix, *, mode):
        shapes.append(matrix.shape)
        assert matrix.shape[0] <= 1400 + 7 and matrix.shape[1] == 7
        return original_qr(matrix, mode=mode)

    def factory():
        for start in range(0, 5600, 1400):
            index = np.arange(start, start + 1400)[:, None]
            features = np.sin(index * np.arange(1, 7)[None, :] / 97)
            targets = features[:, :2].copy()
            weights = np.full(1400, 1 / 5600)
            references.extend(weakref.ref(a) for a in (features, targets, weights))
            yield features, targets, weights

    monkeypatch.setattr(audit.np.linalg, "qr", observed_qr)
    actual = audit.independent_ridge(
        factory, expected_rows=5600, feature_columns=tuple(f"h{i}" for i in range(6)), target_columns=2
    )
    gc.collect()
    assert len(shapes) == 4 and actual.block_counts == (4, 4)
    assert all(reference() is None for reference in references)
    assert actual.coefficients.shape == (7, 2) and actual.center.shape == (6,)


@pytest.mark.parametrize(
    "bad",
    (
        "nonfinite_x",
        "nonfinite_y",
        "nonfinite_w",
        "zero_w",
        "negative_w",
        "mass",
        "zero_scale",
        "layout",
        "empty",
        "oversized",
    ),
)
def test_invalid_values_weight_quality_shapes_and_limits_fail(bad):
    features = np.arange(60, dtype=np.float64).reshape(20, 3)
    targets = features[:, :2].copy()
    weights = np.full(20, 0.05)
    max_rows = 1400
    if bad == "nonfinite_x":
        features[1, 1] = np.nan
    elif bad == "nonfinite_y":
        targets[1, 1] = np.inf
    elif bad == "nonfinite_w":
        weights[1] = np.nan
    elif bad == "zero_w":
        weights[1] = 0
    elif bad == "negative_w":
        weights[1] = -0.05
    elif bad == "mass":
        weights *= 1.01
    elif bad == "zero_scale":
        features[:] = 0
    elif bad == "layout":
        targets = targets[:-1]
    elif bad == "empty":
        features, targets, weights = features[:0], targets[:0], weights[:0]
    else:
        max_rows = 10
    with pytest.raises(ValueError):
        audit.independent_ridge(
            lambda: iter(((features, targets, weights),)),
            expected_rows=20,
            feature_columns=("x", "y", "z"),
            target_columns=2,
            max_block_rows=max_rows,
        )


def test_weight_sum_tolerance_retains_the_explicit_sum_w_x_center():
    features = np.arange(60, dtype=np.float64).reshape(20, 3) / 10
    targets = features[:, :1] + 1
    weights = np.full(20, (1 + 5e-13) / 20)
    expected, center, scale = gold(features, targets, weights, 0.1)
    actual = audit.independent_ridge(
        replay(features, targets, weights, (4, 7)), expected_rows=20, feature_columns=("x", "y", "z"), target_columns=1
    )
    np.testing.assert_allclose(actual.center, center, rtol=0, atol=1e-14)
    np.testing.assert_allclose(actual.scale, scale, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(actual.coefficients, expected, rtol=1e-10, atol=1e-11)
    assert abs(actual.weight_sum - 1) > 1e-13


@pytest.mark.parametrize(
    "kwargs",
    (
        dict(expected_rows=0),
        dict(feature_columns=("x", "x", "z")),
        dict(feature_columns=["x", "y", "z"]),
        dict(target_columns=0),
        dict(ridge=0),
        dict(ridge=float("nan")),
        dict(max_block_rows=1401),
    ),
)
def test_explicit_fit_contract_rejects_invalid_configuration_before_reading_rows(kwargs):
    config = dict(expected_rows=20, feature_columns=("x", "y", "z"), target_columns=2)
    config.update(kwargs)

    def forbidden():
        pytest.fail("invalid fit contract must not consume rows")

    with pytest.raises(ValueError):
        audit.independent_ridge(forbidden, **config)
