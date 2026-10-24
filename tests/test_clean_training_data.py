"""Synthetic038 identity/label contracts; no actual training futures are opened."""

import copy
import hashlib
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a import clean_training_data as data
from s2a.geometry import integrate_local, transform


def source(task, demo):
    return data.SourceIdentity(task, f"demo_{demo}", hashlib.sha256(f"{task}/{demo}".encode()).hexdigest())


@pytest.fixture
def roster_inputs():
    records = []
    for task in range(3):
        for demo in reversed(range(40, 50)):
            identity = source(task, demo)
            n = 90 + demo
            records.append(
                dict(
                    task_index=task,
                    demo=identity.demo,
                    initial_state_sha256=identity.initial_state_sha256,
                    native_steps=n,
                    anchor_frames=[j * (n - 40) // 3 for j in range(4)] + [n + 40],
                )
            )
    validation = tuple(source(task, int(demo[5:])) for task, demo in sorted(data.VALIDATION_DEMO_KEYS))
    fresh = tuple(source(task, demo) for task in range(3) for demo in range(200, 204))
    return records, dict(
        validation_sources=validation, fresh_sources=fresh, validation_demo_keys=tuple(data.VALIDATION_DEMO_KEYS)
    )


def test_all_metadata_anchors_seeds_and_numeric_order_are_fixed(roster_inputs):
    records, splits = roster_inputs
    anchors = data.build_training_roster(records, **splits)
    assert len(anchors) == 150 and anchors[0] == data.AnchorIdentity(source(0, 40), 0)
    assert anchors[5] == data.AnchorIdentity(source(0, 41), 0)
    assert anchors[4].anchor_frame == 170
    requests = data.noise_roster(anchors)
    assert len(requests) == len({r.sampling_seed for r in requests}) == 300
    assert not {r.sampling_seed for r in requests} & {42, 43}
    assert requests == data.noise_roster(reversed(anchors))
    first = requests[0]
    assert first.canonical_json == '{"anchor_frame":0,"demo":"demo_40","root_seed":101,"task":0}'
    digest = hashlib.sha256(first.canonical_json.encode("utf-8"))
    assert first.identity_sha256 == digest.hexdigest()
    assert first.sampling_seed == int.from_bytes(digest.digest()[:8], "big") % 2**63
    assert requests[1].root_seed == 102 and requests[2].anchor.anchor_frame == 30
    assert json.loads(json.dumps(first.as_dict()))["initial_state_sha256"] == source(0, 40).initial_state_sha256


@pytest.mark.parametrize("error", ("demo", "initial_alias", "duplicate", "old_terminal", "missing", "historical_demo"))
def test_training_split_and_complete_native_roster_reject_leaks(roster_inputs, error):
    records, splits = roster_inputs
    if error == "demo":
        records[0]["demo"] = "demo_16"
    elif error == "initial_alias":
        records[0]["initial_state_sha256"] = splits["fresh_sources"][0].initial_state_sha256
    elif error == "duplicate":
        records[1] = copy.deepcopy(records[0])
    elif error == "old_terminal":
        records[0]["anchor_frames"][-1] -= 40
    elif error == "missing":
        records.pop()
    else:
        splits["forbidden_demo_keys"] = [(0, "demo_49")]
    with pytest.raises(ValueError, match="leaks|duplicate|five anchors|all30"):
        data.build_training_roster(records, **splits)


def test_noise_collision_and_evaluation_seed_fail_without_repair(roster_inputs, monkeypatch):
    records, splits = roster_inputs
    anchors = data.build_training_roster(records, **splits)
    for seed in (42, 43, 555):
        monkeypatch.setattr(data, "derive_noise_seed", lambda anchor, root, seed=seed: seed)
        with pytest.raises(ValueError, match="collides or leaks"):
            data.noise_roster(anchors)


def test_unknown_validation_initial_identity_keeps_demo_and_known_alias_exclusions(roster_inputs):
    records, splits = roster_inputs
    splits["validation_sources"] = ()  # Do not fabricate an uncertified native SHA.
    assert len(data.build_training_roster(records, **splits)) == 150
    splits["forbidden_initial_state_sha256"] = [records[0]["initial_state_sha256"]]
    with pytest.raises(ValueError, match="leaks"):
        data.build_training_roster(records, **splits)
    splits.pop("forbidden_initial_state_sha256")
    splits["validation_demo_keys"] = splits["validation_demo_keys"][:-1]
    with pytest.raises(ValueError, match="six fixed validation"):
        data.build_training_roster(records, **splits)


def test_complete_d0_and_union_row_weights_include_all_terminal_rows(roster_inputs):
    records, splits = roster_inputs
    requests = data.noise_roster(data.build_training_roster(records, **splits))
    for rounds, total in ((("D0",), 420000), (("D0", "D1"), 840000)):
        weights = np.concatenate(
            [data.row_weights(start, min(start + 1400, total), rounds=rounds) for start in range(0, total, 1400)]
        )
        assert weights.shape == (total,)
        assert np.all(weights == 1 / total) and np.isclose(weights.sum(), 1, atol=1e-14, rtol=0)
        if len(rounds) == 2:
            assert np.isclose(weights[:420000].sum(), 0.5, atol=1e-14, rtol=0)
        assert data.row_identity(0, requests, rounds=rounds)["endpoint_index"] == 0
        last = data.row_identity(total - 1, requests, rounds=rounds)
        assert last["round"] == rounds[-1] and last["step_index"] == 34 and last["endpoint_index"] == 39
        assert last["root_seed"] == 102 and last["anchor_frame"] == 179
    assert data.row_identity(420000, requests, rounds=("D0", "D1"))["round"] == "D1"
    with pytest.raises(ValueError, match="complete D0"):
        data.row_weights(0, 40, rounds=("D1",))
    with pytest.raises(ValueError, match="complete ordered noise"):
        data.row_identity(0, tuple(reversed(requests)))


def window():
    # A rotating source and translated TCP expose the tool lever arm. The site
    # has a different rotation: using the controller-site orientation is wrong.
    t = np.arange(41, dtype=float)
    rotation = Rotation.from_euler("xyz", np.column_stack((0.01 * t, -0.005 * t, 0.02 * t))).as_matrix()
    source_pose = transform(np.column_stack((0.001 * t, np.sin(t / 5) * 0.002, t * 0.0003)), rotation)
    site = source_pose.copy()
    site[:, :3, :3] = rotation @ Rotation.from_euler("z", 0.3).as_matrix()
    body = source_pose.copy()
    body[:, :3, 3] += 0.1
    h = transform([0.03, 0.01, -0.004], Rotation.from_euler("y", 0.2).as_matrix())
    return dict(
        source_poses=source_pose,
        openings=0.2 + t * 0.01,
        sim_times=0.25 + t * 0.05,
        source_from_tcp=h,
        actual_source_site_poses=site,
        actual_eef_body_poses=body,
        state_mask=np.broadcast_to(data.ACTIVE_MASK, (41, 128)).copy(),
        action_mask=np.broadcast_to(data.ACTIVE_MASK, (40, 128)).copy(),
        action_valid=np.ones(40, bool),
    )


def test_post_capture_local_noncommuting_motion_tcp_lever_and_next_measured_opening():
    inputs = window()
    actual = data.post_capture_labels(**inputs)
    physical = actual["physical_action128"]
    tcp = inputs["source_poses"] @ inputs["source_from_tcp"]
    rebuilt = integrate_local(tcp[0], physical[:, :9])
    np.testing.assert_allclose(rebuilt, tcp, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(physical[:, 9], inputs["openings"][1:])
    assert not physical[:, 10:].any()
    source_local = np.linalg.inv(inputs["source_poses"][0]) @ inputs["source_poses"][1]
    assert not np.allclose(physical[0, :3], source_local[:3, 3], atol=1e-6)
    assert np.linalg.norm(physical[-1, :3]) > 0  # Real terminal drift is retained.
    wrong_left_delta = tcp[1] @ np.linalg.inv(tcp[0])
    assert not np.allclose(physical[0, :3], wrong_left_delta[:3, 3], atol=1e-6)


@pytest.mark.parametrize(
    "error", ("site_rotation", "wrong_position", "invalid_tail", "mask", "state_mask", "time", "command")
)
def test_post_capture_rejects_wrong_source_and_incomplete_physical_windows(error):
    inputs = window()
    if error == "site_rotation":
        inputs["source_poses"] = inputs["actual_source_site_poses"].copy()
    elif error == "wrong_position":
        inputs["source_poses"][:, 0, 3] += 0.01
    elif error == "invalid_tail":
        inputs["action_valid"][-1] = False
    elif error == "mask":
        inputs["action_mask"][-1, 9] = False
    elif error == "state_mask":
        inputs["state_mask"][-1] = False
    elif error == "time":
        inputs["sim_times"][-1] += 0.001
    else:
        inputs["openings"][-1] = -1  # A native -1 gripper command is not opening.
    with pytest.raises(ValueError, match="source|valid|mask|interval|opening"):
        data.post_capture_labels(**inputs)


def normalization():
    return dict(
        normalization_mask=[True, True, True] + [False] * 125,
        scale=[2.0] * 128,
        bias=[0.25] * 128,
        clip_min=-1.0,
        clip_max=1.0,
    )


def test_normalization31_is_fp32_masked_and_reports_every_clipped_label():
    physical = data.post_capture_labels(**window())["physical_action128"]
    physical[:, 0] = 2
    physical[0, 1] = 1.00000006
    transform = normalization()
    transform["scale"][1], transform["bias"][1] = 1.00000006, -1.00000012
    actual = data.normalize_active10(
        physical, transform=transform, domain=data.DOMAIN, normalization_id=31, projection_id=0
    )
    output = actual["normalized_active10"]
    assert output.dtype == np.float32 and output.shape == (40, 10)
    assert actual["clipped_count"] == 40 and actual["clipped_per_active_slot"][0] == 40
    np.testing.assert_array_equal(output[:, 3:], physical[:, 3:10].astype(np.float32))
    expected = np.float32(
        np.float32(np.float32(physical[0, 1]) * np.float32(transform["scale"][1])) + np.float32(transform["bias"][1])
    )
    assert output[0, 1] == expected
    assert output[0, 1] != np.float32(physical[0, 1] * transform["scale"][1] + transform["bias"][1])
    assert len(actual["transform_sha256"]) == 64


@pytest.mark.parametrize("error", ("domain", "normalization_id", "projection_id", "inactive", "transform_mask"))
def test_normalization_refuses_other_source_domains_or_silent_inactive_use(error):
    physical = data.post_capture_labels(**window())["physical_action128"]
    kwargs = dict(transform=normalization(), domain=data.DOMAIN, normalization_id=31, projection_id=0)
    if error in ("domain", "normalization_id", "projection_id"):
        kwargs[error] = "wrong" if error == "domain" else 1
    elif error == "inactive":
        physical[0, 10] = 1
    else:
        kwargs["transform"]["normalization_mask"][10] = True
    with pytest.raises(ValueError, match="required|inactive118"):
        data.normalize_active10(physical, **kwargs)


def test_d1_target_uses_raw_velocity_instead_of_subtracting_w1_twice():
    sample = np.full((40, 10), 0.75, dtype=np.float32)
    raw = np.full((40, 10), 0.5, dtype=np.float32)
    x0 = np.full((40, 10), 0.2, dtype=np.float32)
    sigma, w1_clean = 0.5, 0.125
    deployed = raw - w1_clean / sigma
    actual = data.clean_targets(x0, sample, raw, sigma, velocity_semantics="raw")
    np.testing.assert_array_equal(actual["raw_c0_active10"], sample.astype(np.float64) - sigma * raw)
    np.testing.assert_array_equal(actual["target_clean_residual"], x0.astype(np.float64) - 0.5)
    wrong_target = x0.astype(np.float64) - (sample - sigma * deployed)
    assert np.all(wrong_target != actual["target_clean_residual"])
    with pytest.raises(ValueError, match="raw velocity"):
        data.clean_targets(x0, sample, deployed, sigma, velocity_semantics="deployed")
    for bad_sigma in (0.0, -0.1, 0.1, 500.1):  # Last two are not actual FP32 scalar values.
        with pytest.raises(ValueError, match="actual positive observed FP32"):
            data.clean_targets(x0, sample, raw, bad_sigma, velocity_semantics="raw")
    with pytest.raises(ValueError, match="actual sampler sample"):
        data.clean_targets(x0, sample.astype(np.float64), raw, sigma, velocity_semantics="raw")
