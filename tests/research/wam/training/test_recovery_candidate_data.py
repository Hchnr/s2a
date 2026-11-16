"""Recovery candidate identities, overlap and clipping must retain their semantics."""

import copy

import numpy as np
import pytest

from s2a.research.wam.training.clean.clean_training_data import DOMAIN
from s2a.research.wam.training.recovery_candidate_data import (
    candidate_seed,
    clipping_translation_effect,
    transition_multiplicities,
)


def normalization():
    return dict(
        domain=DOMAIN,
        normalization_id=31,
        projection_id=0,
        transform=dict(
            normalization_mask=[True] * 3 + [False] * 125,
            scale=[10.0] * 3 + [1.0] * 125,
            bias=[0.0] * 128,
            clip_min=-1.0,
            clip_max=1.0,
        ),
    )


def test_candidate_seed_separates_variants_roots_and_full_provenance():
    identity = dict(
        task_index=0,
        demo="demo_40",
        initial_state_sha256="1" * 64,
        case_index=0,
        variant="zero",
        anchor_step=89,
        image_sha256="2" * 64,
        trajectory_sha256="3" * 64,
    )
    original = candidate_seed(identity, 101)
    assert candidate_seed(identity, 101) == original
    assert candidate_seed(identity, 102)["sampling_seed"] != original["sampling_seed"]
    for key, value in (("case_index", 1), ("variant", "base_x"), ("image_sha256", "4" * 64)):
        changed = dict(identity, **{key: value})
        assert candidate_seed(changed, 101)["sampling_seed"] != original["sampling_seed"]
    with pytest.raises(ValueError, match="complete"):
        candidate_seed({key: value for key, value in identity.items() if key != "variant"}, 101)
    with pytest.raises(ValueError, match="explicit"):
        candidate_seed(dict(identity, trajectory_sha256="bad"), 101)
    with pytest.raises(ValueError, match="explicit"):
        candidate_seed(identity, 42)


def test_multiplicity_distinguishes_overlapping_windows_from_distinct_cases():
    counts = transition_multiplicities([(0, 0), (0, 2), (1, 0)], horizon=4)
    np.testing.assert_array_equal(counts, [[1, 1, 2, 2], [2, 2, 1, 1], [1, 1, 1, 1]])
    assert np.sum(1.0 / counts) == 10
    with pytest.raises(ValueError, match="Distinct"):
        transition_multiplicities([(0, 0), (0, 0)], horizon=4)
    with pytest.raises(ValueError, match="Distinct"):
        transition_multiplicities([(True, 0)], horizon=4)


def test_clipping_error_uses_measured_tcp_orientation_and_cancels():
    actions = np.zeros((40, 128), dtype=np.float64)
    actions[0, 0], actions[1, 0] = 0.2, -0.2
    rot = np.broadcast_to(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64), (40, 3, 3)).copy()
    result = clipping_translation_effect(actions, rot, normalization())
    assert int(result["clipped_mask"].sum()) == 2
    np.testing.assert_allclose(result["cumulative_base_translation_error"][0], [0, -0.1, 0], atol=1e-8)
    assert result["max_prefix_error_m"] == pytest.approx(0.1)
    assert result["final_error_m"] == pytest.approx(0.0, abs=1e-8)
    actions[:, 0] = 0.05
    result = clipping_translation_effect(actions, rot, normalization())
    assert not result["clipped_mask"].any()
    assert result["max_prefix_error_m"] == result["final_error_m"] == 0


def test_clipping_error_rejects_wrong_frame_or_normalization_contract():
    actions = np.zeros((40, 128))
    rot = np.broadcast_to(np.eye(3), (40, 3, 3)).copy()
    rot[0, 0, 0] = 2
    with pytest.raises(ValueError, match="proper measured TCP"):
        clipping_translation_effect(actions, rot, normalization())
    rot[0] = np.eye(3)
    invalid = copy.deepcopy(normalization())
    invalid["transform"]["normalization_mask"][1] = False
    with pytest.raises(ValueError, match="three-position mask"):
        clipping_translation_effect(actions, rot, invalid)
    invalid = copy.deepcopy(normalization())
    invalid["transform"]["scale"][0] = 0
    with pytest.raises(ValueError, match="Nonzero finite"):
        clipping_translation_effect(actions, rot, invalid)
