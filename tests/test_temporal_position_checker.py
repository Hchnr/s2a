"""Independent020 causal endpoint, geometry and retained-source contracts."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


@pytest.fixture
def checker(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "temporal_checker_test", root / "scripts/wam_temporal_position_checker.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def chunk(q, poses, openings, anchor=None):
    previous = np.eye(4) if anchor is None else anchor.copy()
    raw = np.zeros((len(poses), 128))
    for i, target in enumerate(poses):
        relative = np.linalg.inv(previous) @ target
        raw[i, :3] = relative[:3, 3]
        raw[i, 3:9] = np.r_[relative[:3, 0], relative[:3, 1]]
        raw[i, 9] = openings[i]
        previous = target
    return dict(step=q, anchor_tcp=np.eye(4) if anchor is None else anchor, raw128=raw, effective128=raw.copy())


def path(count, *, translation=0.0, angles=(0, 0, 0)):
    poses = np.repeat(np.eye(4)[None], count, axis=0)
    poses[:, 0, 3] = translation + np.arange(1, count + 1) * 0.001
    poses[:, :3, :3] = Rotation.from_rotvec(angles).as_matrix()
    return poses


def test_same_absolute_endpoint_indices_and_all_channel_weights(checker):
    first, second = path(40, angles=(0.3, 0, 0)), path(40, translation=0.1, angles=(0, 0.4, 0))
    chunks = [chunk(0, first, np.full(40, 0.2)), chunk(8, second, np.full(40, 0.8))]
    pose, opening, log = checker.reconstruct_ensemble(8, chunks)
    assert log["absolute_endpoint"] == 9 and log["contributor_count"] == 2
    assert [(c["query_step"], c["endpoint_index"], c["absolute_endpoint"]) for c in log["contributors"]] == [
        (0, 8, 9),
        (8, 0, 9),
    ]
    assert all(c["weight"] == 0.5 for c in log["contributors"])
    np.testing.assert_allclose(pose[:3, 3], (first[8, :3, 3] + second[0, :3, 3]) / 2, atol=1e-14)
    assert opening == pytest.approx(0.5)
    np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-14)
    assert np.linalg.det(pose[:3, :3]) == pytest.approx(1)
    actual_json = json.loads(json.dumps(log, default=lambda value: value.tolist()))
    checker.compare_ensemble_log(actual_json, log)


def test_expired_predictions_future_values_and_input_ownership(checker):
    chunks = [chunk(q, path(40, translation=q * 0.001), np.full(40, 0.5)) for q in range(0, 49, 8)]
    before = chunks[0]["raw128"].copy()
    result = checker.reconstruct_ensemble(40, chunks)
    assert [c["query_step"] for c in result[2]["contributors"]] == [8, 16, 24, 32, 40]
    chunks[-1]["raw128"][:] = np.nan  # query48 is not available at physical40
    again = checker.reconstruct_ensemble(40, chunks)
    np.testing.assert_array_equal(result[0], again[0])
    np.testing.assert_array_equal(chunks[0]["raw128"], before)
    result[0][:] = 99
    assert checker.reconstruct_ensemble(40, chunks)[0][3, 3] == 1


def test_identity_copies_latest_value_with_shortened_oracle_tail(checker):
    world = path(48, angles=(0.15, -0.3, 0.2))
    chunks = [chunk(0, world[:40], np.full(40, 0.7)), chunk(8, world[8:16], np.full(8, 0.7))]
    target, opening, log = checker.reconstruct_ensemble(15, chunks)
    assert log["identity_fast_path"] and log["minimum_singular_value"] is None
    latest = checker.decode(chunks[-1]["anchor_tcp"], chunks[-1]["raw128"])[7]
    np.testing.assert_array_equal(target, latest)
    assert opening == 0.7


def test_common_world_frame_equivariance_with_noncommuting_rotations(checker):
    poses = [path(40, translation=0.1, angles=(0.4, -0.2, 0.1)), path(40, translation=-0.2, angles=(-0.1, 0.3, 0.2))]
    opening = [np.full(40, 0.2), np.full(40, 0.8)]
    original = [chunk(q, value, aperture) for q, value, aperture in zip((0, 8), poses, opening, strict=True)]
    world = np.eye(4)
    world[:3, :3] = Rotation.from_euler("xyz", [0.6, -0.3, 0.8]).as_matrix()
    world[:3, 3] = [0.3, -0.1, 0.7]
    changed = [
        chunk(q, world @ value, aperture, world) for q, value, aperture in zip((0, 8), poses, opening, strict=True)
    ]
    a, ao, _ = checker.reconstruct_ensemble(8, original)
    b, bo, _ = checker.reconstruct_ensemble(8, changed)
    np.testing.assert_allclose(b, world @ a, atol=2e-14, rtol=0)
    assert ao == bo


def test_missing_query_and_ambiguous_rotation_are_rejected(checker):
    first = chunk(0, path(40), np.full(40, 0.5))
    with pytest.raises(AssertionError):
        checker.reconstruct_ensemble(8, [first])
    opposite = chunk(8, path(40, angles=(np.pi, 0, 0)), np.full(40, 0.5))
    with pytest.raises(AssertionError):
        checker.reconstruct_ensemble(8, [first, opposite])


def test_all_unrelated014_case_ast_and_entire_query_ast_remain_exact(checker):
    from audit_wam_position import Evidence

    proof = checker.source_proof(Evidence())
    assert proof["all_other_case_ast_nodes_exact"] and proof["literal_query_ast_exact"]
    assert not proof["production_temporal_module_imported"] and not proof["dynamic_patch_or_generated_function_used"]


def test_logged_indices_and_uniform_weights_are_exact_not_numeric_tolerances(checker):
    inputs = [chunk(0, path(40), np.full(40, 0.5))]
    _, _, expected = checker.reconstruct_ensemble(0, inputs)
    checker.compare_ensemble_log(copy.deepcopy(expected), expected)
    for group, field, value in (
        (None, "physical_step", 0.0),
        (None, "equal_weight", 1 - 1e-15),
        (0, "endpoint_index", 0.0),
        (0, "weight", 1 - 1e-15),
    ):
        saved = copy.deepcopy(expected)
        target = saved if group is None else saved["contributors"][group]
        target[field] = value
        with pytest.raises(AssertionError):
            checker.compare_ensemble_log(saved, expected)
