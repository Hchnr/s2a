"""020 risks: integer clocks, causal copies, common endpoint geometry and SO(3)."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.temporal_ensemble import CausalTemporalEqualEnsemble, TemporalEqualConfig


def path(q, *, offset=0.0, rotation=None, length=40):
    poses = np.repeat(np.eye(4)[None], length, axis=0)
    poses[:, 0, 3] = np.arange(q + 1, q + length + 1) + offset
    if rotation is not None:
        poses[:, :3, :3] = rotation
    return poses, np.full(length, 0.5 + offset / 100)


def advance(ensemble, stop, factory=path):
    values = []
    while ensemble.next_step < stop:
        k = ensemble.next_step
        if k % 8 == 0:
            ensemble.push(k, *factory(k))
        values.append(ensemble.sample(k))
    return values


def test_absolute_endpoint_identity_expiration_and_short_tail():
    ensemble = CausalTemporalEqualEnsemble()
    # Like the oracle, all queries describe the same world trajectory; the last
    # four references shorten at the finite boundary, without padding targets.
    values = advance(ensemble, 64, lambda q: path(q, length=min(40, 64 - q)))
    for k, (target, opening, info) in enumerate(values):
        np.testing.assert_array_equal(target, path(k)[0][0])
        assert opening == 0.5 and info["identity_fast_path"]
        assert target[0, 3] == k + 1 and info["absolute_endpoint"] == k + 1
        assert all(c["query_step"] + c["endpoint_index"] + 1 == k + 1 for c in info["contributors"])
        assert all(c["weight"] == 1 / info["contributor_count"] for c in info["contributors"])
        assert info["contributor_count"] <= 5
    assert [c["query_step"] for c in values[39][2]["contributors"]] == [0, 8, 16, 24, 32]
    assert [c["query_step"] for c in values[40][2]["contributors"]] == [8, 16, 24, 32, 40]
    assert ensemble.active_query_steps == ()
    with pytest.raises(ValueError, match="missing"):
        ensemble.sample(64)


def test_mean_uses_same_contributors_and_keeps_old_world_anchor():
    ensemble = CausalTemporalEqualEnsemble()
    advance(ensemble, 8)
    ensemble.push(8, *path(8, offset=2.0))
    target, opening, info = ensemble.sample(8)
    assert target[0, 3] == 10 and opening == 0.51
    assert [(c["query_step"], c["endpoint_index"], c["weight"]) for c in info["contributors"]] == [
        (0, 8, 0.5),
        (8, 0, 0.5),
    ]
    assert [c["target_tcp"][0, 3] for c in info["contributors"]] == [9, 11]
    assert [c["opening"] for c in info["contributors"]] == [0.5, 0.52]
    assert not info["identity_fast_path"]


def test_saved_inputs_and_returned_metadata_are_copies_and_reset_isolates_cases():
    ensemble = CausalTemporalEqualEnsemble()
    poses, openings = path(0)
    ensemble.push(0, poses, openings)
    poses[:] = np.nan
    openings[:] = np.nan
    target, opening, info = ensemble.sample(0)
    target[:] = 99
    info["contributors"][0]["target_tcp"][:] = 99
    later, later_opening, _ = ensemble.sample(1)
    np.testing.assert_array_equal(later, path(1)[0][0])
    assert opening == later_opening == 0.5
    ensemble.reset()
    assert ensemble.next_step == 0 and ensemble.active_query_steps == ()
    with pytest.raises(ValueError, match="missing"):
        ensemble.sample(0)
    ensemble.push(0, *path(0))
    assert ensemble.sample(0)[0][0, 3] == 1


def test_clock_rejects_future_duplicate_skipped_and_missing_query_without_advancing():
    ensemble = CausalTemporalEqualEnsemble()
    for bad in (True, -1, 0.0, np.int64(0), 8):
        with pytest.raises(ValueError):
            ensemble.push(bad, *path(0))
    ensemble.push(0, *path(0))
    with pytest.raises(ValueError):
        ensemble.push(0, *path(0))
    for bad in (True, 1, 0.0):
        with pytest.raises(ValueError):
            ensemble.sample(bad)
    assert ensemble.next_step == 0
    for k in range(8):
        ensemble.sample(k)
    with pytest.raises(ValueError, match="missing"):
        ensemble.sample(8)
    with pytest.raises(ValueError):
        ensemble.push(16, *path(16))
    ensemble.push(8, *path(8))
    ensemble.sample(8)
    with pytest.raises(ValueError):
        ensemble.sample(8)


def test_expired_short_reference_fails_without_extrapolation():
    ensemble = CausalTemporalEqualEnsemble()
    ensemble.push(0, *path(0, length=1))
    ensemble.sample(0)
    with pytest.raises(ValueError, match="nonexpired"):
        ensemble.sample(1)


def test_near_identity_copies_latest_exactly_and_threshold_changes_are_rejected():
    ensemble = CausalTemporalEqualEnsemble()
    advance(ensemble, 8)
    poses, openings = path(8)
    poses[:, 2, 3] = 5e-13
    openings += 5e-13
    ensemble.push(8, poses, openings)
    target, opening, info = ensemble.sample(8)
    np.testing.assert_array_equal(target, poses[0])
    assert opening == openings[0] and info["identity_fast_path"]
    for changed in (
        dict(identity_atol=1e-10),
        dict(minimum_singular_value=1e-5),
        dict(horizon_steps=True),
        dict(query_interval_steps=16),
    ):
        with pytest.raises(ValueError):
            TemporalEqualConfig(**changed)


def noncommuting_sample(base=None):
    ensemble = CausalTemporalEqualEnsemble()
    rotations = [Rotation.from_rotvec(v).as_matrix() for v in ([0.5, 0, 0], [0, 0.7, 0], [0, 0, -0.4])]
    transform = np.eye(4) if base is None else base
    for k in range(17):
        if k % 8 == 0:
            poses, openings = path(k, offset=k / 8, rotation=rotations[k // 8])
            ensemble.push(k, transform @ poses, openings)
        value = ensemble.sample(k)
    return value, rotations


def test_noncommuting_rotations_use_chordal_projection_and_world_se3_equivariance():
    (pose, opening, info), rotations = noncommuting_sample()
    mean = np.mean(rotations, axis=0)
    # Check the polar optimum independently through R^T M symmetric positive.
    np.testing.assert_allclose(pose[:3, :3].T @ mean, (pose[:3, :3].T @ mean).T, atol=1e-14)
    assert np.linalg.eigvalsh(pose[:3, :3].T @ mean).min() > 0
    np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-14)
    assert np.linalg.det(pose[:3, :3]) == pytest.approx(1)
    assert len(info["contributors"]) == 3 and opening == pytest.approx(0.51)
    base = np.eye(4)
    base[:3, :3] = Rotation.from_rotvec([0.6, -0.3, 0.4]).as_matrix()
    base[:3, 3] = [2, -3, 5]
    (changed, changed_opening, changed_info), _ = noncommuting_sample(base)
    np.testing.assert_allclose(changed, base @ pose, atol=1e-14)
    assert changed_opening == opening
    assert [(c["query_step"], c["weight"]) for c in info["contributors"]] == [
        (c["query_step"], c["weight"]) for c in changed_info["contributors"]
    ]


@pytest.mark.parametrize("kind", ["singular", "negative_determinant"])
def test_ambiguous_rotation_means_fail_closed(kind):
    ensemble = CausalTemporalEqualEnsemble()
    stop = 8 if kind == "singular" else 16
    rotations = (
        [np.eye(3), Rotation.from_rotvec([0, 0, np.pi - 1e-8]).as_matrix()]
        if kind == "singular"
        else [Rotation.from_rotvec(v).as_matrix() for v in ([np.pi, 0, 0], [0, np.pi, 0], [0, 0, np.pi])]
    )
    for k in range(stop + 1):
        if k % 8 == 0:
            poses, openings = path(k)
            poses[stop - k, :3, :3] = rotations[k // 8]
            ensemble.push(k, poses, openings)
        if k < stop:
            ensemble.sample(k)
    with pytest.raises(ValueError, match="ambiguous"):
        ensemble.sample(stop)
    assert ensemble.next_step == stop


@pytest.mark.parametrize(
    "kind", ["nan_pose", "improper_rotation", "shape", "nan_opening", "range_opening", "empty", "too_long"]
)
def test_invalid_world_reference_rejected_before_clock_advances(kind):
    ensemble = CausalTemporalEqualEnsemble()
    poses, openings = path(0)
    if kind == "nan_pose":
        poses[0, 0, 0] = np.nan
    elif kind == "improper_rotation":
        poses[0, 0, 0] = -1
    elif kind == "shape":
        poses = poses[:, :3]
    elif kind == "nan_opening":
        openings[0] = np.nan
    elif kind == "range_opening":
        openings[0] = 1.001
    elif kind == "empty":
        poses, openings = poses[:0], openings[:0]
    else:
        poses, openings = np.repeat(np.eye(4)[None], 41, axis=0), np.zeros(41)
    with pytest.raises(ValueError):
        ensemble.push(0, poses, openings)
    assert ensemble.active_query_steps == () and ensemble.next_step == 0
