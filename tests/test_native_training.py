"""030 geometry and causal collection control, without HDF5 or MuJoCo."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.geometry import transform
from s2a.native_training import (
    native_anchor_indices,
    native_command_schedule,
    state_sha256,
    synchronous_labels,
    validate_physical_source,
)

ROOT = Path(__file__).resolve().parents[1]


def collector():
    spec = importlib.util.spec_from_file_location(
        "native_collector_tests", ROOT / "scripts/collect_wam_native_training.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_labels_with_tcp_lever_arm_and_noncommuting_rotations():
    rotations = Rotation.from_euler("xyz", [[0.1, 0.3, 0.2], [0.7, -0.2, 0.5], [-0.4, 0.6, 0.9]]).as_matrix()
    source = transform([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.4, -0.1, 0.2]], rotations)
    h = transform([0.02, -0.04, 0.17], Rotation.from_euler("xyz", [0.4, 0.2, -0.3]).as_matrix())
    opening = np.array([0.13, 0.78, 0.24])
    labels = synchronous_labels(source, opening, h, [3.0, 3.05, 3.1])
    expected = source @ h
    np.testing.assert_allclose(labels["tcp_pose"], expected, rtol=0, atol=2e-15)
    assert np.linalg.norm(labels["action128"][0, :3]) > 0.01  # Pure source rotation moves the TCP.
    current = expected[0].copy()
    for index, action in enumerate(labels["action128"][:-1]):
        # Independent decoding: the two stored rotation columns reconstruct SO(3).
        x, y = action[3:6], action[6:9]
        delta = np.eye(4)
        delta[:3, :3] = np.column_stack((x, y, np.cross(x, y)))
        delta[:3, 3] = action[:3]
        current = current @ delta
        np.testing.assert_allclose(current, expected[index + 1], atol=3e-15, rtol=0)
        np.testing.assert_allclose(delta, np.linalg.inv(expected[index]) @ expected[index + 1], atol=1e-15)
    np.testing.assert_array_equal(labels["state128"][:, 9], opening)
    np.testing.assert_array_equal(labels["action128"][:-1, 9], opening[1:])
    assert not np.allclose(labels["action128"][:-1, 9], opening[:-1])
    assert not labels["action128"][:, 10:].any()
    assert not labels["state128"][:, 10:].any()
    assert labels["state128"].dtype == labels["action128"].dtype == np.float64
    assert labels["state_mask"][:, :10].all() and not labels["state_mask"][:, 10:].any()
    assert labels["action_mask"][:-1, :10].all() and not labels["action_mask"][-1].any()
    np.testing.assert_array_equal(labels["action_valid"], [True, True, False])
    assert not labels["action128"][-1].any()
    assert not np.shares_memory(labels["tcp_pose"], source)


def test_exact_clock_and_masks_reject_missing_or_duplicate_endpoint():
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    for times in ([0, 0.05, 0.15], [0, 0, 0.05], [0, 0.05, np.nan]):
        with pytest.raises(ValueError):
            synchronous_labels(poses, [0.2] * 3, np.eye(4), times)
    with pytest.raises(ValueError, match="opening"):
        synchronous_labels(poses, [0, 1.1, 0], np.eye(4), [0, 0.05, 0.1])


def test_all_five_anchor_windows_and_native_hold_are_fixed():
    rng = np.random.default_rng(42)
    for n in (43, 44, 77, 599, 600):
        native = rng.uniform(-1, 1, (n, 7)).astype(np.float32)
        original = native.copy()
        sent = native_command_schedule(native)
        expected = [int(np.floor(j * (n - 40) / 3)) for j in range(4)] + [n]
        assert native_anchor_indices(n) == expected
        assert len(set(expected)) == 5
        for frame in expected:
            assert len(np.arange(n + 41)[frame : frame + 41]) == 41
            assert frame + 39 < n + 40  # All 40 outgoing labels precede invalid final row.
        np.testing.assert_array_equal(sent[:n], native)
        assert not sent[n:, :6].any()
        np.testing.assert_array_equal(sent[n:, 6], np.full(40, native[-1, 6]))
        np.testing.assert_array_equal(native, original)
        assert len(sent) == n + 40 <= 640 and not np.shares_memory(sent, native)
    for n in (0, 42, 601, True, 43.5):
        with pytest.raises(ValueError):
            native_anchor_indices(n)
    with pytest.raises(ValueError):
        native_command_schedule(np.zeros((43, 8)))


def test_physical_source_uses_site_position_and_body_rotation():
    site = transform([0.3, 0.1, -0.2], Rotation.from_euler("z", 0.6).as_matrix())
    body = transform([0.4, -0.5, 0.7], Rotation.from_euler("x", 0.3).as_matrix())
    observed = body.copy()
    observed[:3, 3] = site[:3, 3]
    np.testing.assert_array_equal(validate_physical_source(observed, site, body), observed)
    assert not np.array_equal(observed[:3, :3], site[:3, :3])
    with pytest.raises(ValueError, match="actual grip-site/BODY"):
        validate_physical_source(site, site, body)


def test_physical_source_f64_observation_chain_rejects_nanometre_and_rotation_mismatch():
    body_rotation = Rotation.from_euler("xyz", [0.41, -0.67, 1.13])
    body = transform([0.4, -0.5, 0.7], body_rotation.as_matrix())
    site = transform([0.3, 0.1, -0.2], Rotation.from_euler("z", 0.6).as_matrix())
    # Actual robosuite single_arm.eef_quat -> get_body_xquat is F64 wxyz;
    # transform_utils.convert_quat only reorders it to xyzw, without mat2quat.
    actual_wxyz = body_rotation.as_quat()[[3, 0, 1, 2]]
    observed_xyzw = actual_wxyz[[1, 2, 3, 0]]
    observed = transform(site[:3, 3], Rotation.from_quat(observed_xyzw).as_matrix())
    assert observed_xyzw.dtype == np.float64
    np.testing.assert_allclose(validate_physical_source(observed, site, body), observed, atol=2e-15, rtol=0)
    wrong_position = observed.copy()
    wrong_position[0, 3] += 1e-9
    with pytest.raises(ValueError, match="actual grip-site/BODY"):
        validate_physical_source(wrong_position, site, body)
    wrong_rotation = observed.copy()
    wrong_rotation[:3, :3] = Rotation.from_rotvec([0, 0, 1e-9]).as_matrix() @ observed[:3, :3]
    with pytest.raises(ValueError, match="actual grip-site/BODY"):
        validate_physical_source(wrong_rotation, site, body)


def test_state_fingerprint_preserves_actual_bytes():
    state = np.array([0.0, -0.0, 0.1], dtype=np.float64)
    assert state_sha256(state) == hashlib.sha256(state.tobytes()).hexdigest()
    assert state_sha256(state) != state_sha256(np.array([0, 0, 0.1], dtype=np.float64))
    for invalid in (state.astype(np.float32), np.array([np.nan]), np.zeros((1, 3))):
        with pytest.raises(ValueError):
            state_sha256(invalid)


def test_worker_cpu_guard_refuses_overwrite_and_has_real_cli(tmp_path, monkeypatch):
    module = collector()
    for key, value in dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1").items():
        monkeypatch.setenv(key, value)
    assert module.cpu_environment()["CUDA_VISIBLE_DEVICES"] == ""
    monkeypatch.setenv("LD_LIBRARY_PATH", "/not/empty")
    with pytest.raises(ValueError, match="CPU environment"):
        module.cpu_environment()
    path = tmp_path / "receipt.json"
    module.write_once(path, {"value": 3})
    with pytest.raises(FileExistsError):
        module.write_once(path, {"value": 4})
    assert json.loads(path.read_text()) == {"value": 3}
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run(
        [sys.executable, "scripts/collect_wam_native_training.py", "--help"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0 and "--task-index" in completed.stdout
    assert "--prepared-sha256" in completed.stdout


def test_case_fixture_keeps_running_after_success_and_renders_only_current_anchors(tmp_path, monkeypatch):
    """Exercise scheduling with a synthetic endpoint clock, not a physics substitute."""
    module = collector()
    import s2a.libero_simulation as simulation

    native = np.zeros((43, 7), dtype=np.float32)
    native[:, 0] = np.linspace(0, 0.2, 43)
    native[:, -1] = -0.7
    expert = tmp_path / "expert.npy"
    np.save(expert, native)
    initial = np.array([0.0, 0.0], dtype=np.float64)
    record = dict(
        expert_path=str(expert),
        expert_sha256=module.sha256(expert),
        native_steps=43,
        anchor_frames=native_anchor_indices(43),
        task_id=8,
        task="synthetic scheduling fixture",
        initial_state_sha256=state_sha256(initial),
    )
    counter = dict(step=0, closed=False, observation_calls=[], rendered=[])

    class ClockEnvironment:
        # Actual LIBERO wrapper has action_dim only on its inner env.
        env = SimpleNamespace(action_dim=7, control_timestep=0.05)
        robots = [
            SimpleNamespace(
                eef_site_id=4,
                robot_model=SimpleNamespace(eef_name="eef_body"),
            )
        ]
        sim = SimpleNamespace(model=SimpleNamespace(site_id2name=lambda _: "grip_site", body_name2id=lambda _: 3))

        def step(self, actual):
            counter["step"] += 1
            assert actual.shape == (7,)
            return None  # Cached step observations are deliberately unused.

        def close(self):
            counter["closed"] = True

    def make_environment(policy, case_dir, received_record, trial):
        assert policy["camera_observations"] is False and policy["refresh_controller_after_restore"] is True
        assert trial == 0 and received_record is record
        return ClockEnvironment(), None

    def measured_endpoint(env, index):
        assert index == counter["step"]
        counter["observation_calls"].append(index)
        pose = np.eye(4)
        pose[0, 3] = index * 0.001
        state = np.array([index * 0.05, index * 0.001], dtype=np.float64)
        sample = dict(
            source_pose=pose,
            opening=np.float64(0.4),
            sim_time=np.float64(index * 0.05),
            sim_states=state,
            native_success=np.bool_(index == 1),  # Success at step1 must not stop the loop.
        )
        return sample, dict(endpoint_step=index, sim_state_sha256=state_sha256(state))

    def render(env, index, sample, receipt, case_dir):
        assert index == counter["step"] and sample["sim_time"] == index * 0.05
        counter["rendered"].append(index)
        return dict(frame_index=index)

    monkeypatch.setattr(simulation, "make_environment", make_environment)
    monkeypatch.setattr(
        simulation,
        "controller_contract",
        lambda _: (SimpleNamespace(eef_name="controller_site"), dict(control_freq=20)),
    )
    monkeypatch.setattr(module, "endpoint", measured_endpoint)
    monkeypatch.setattr(module, "render_anchor", render)
    prepared = dict(policy_config={}, tool_source_from_tcp=np.eye(4).tolist(), tool_contract_ref={"fixture": True})
    progress = dict(physics_steps=0, environment_initializations=0, current_canvas_images=0)
    ref = module.collect_case(prepared, {"fixture": True}, record, tmp_path / "case", progress)
    result = json.loads(Path(ref["path"]).read_text())
    assert result["complete"] is result["passed"] is True
    assert result["native_quality_passed"] is False
    assert counter["step"] == 83 and counter["observation_calls"] == list(range(84))
    assert counter["rendered"] == [0, 1, 2, 3, 43] and counter["closed"]
    assert progress == dict(physics_steps=83, environment_initializations=1, current_canvas_images=5)
    with np.load(result["arrays"]["path"], allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["sent_commands"][:43], native)
        assert not arrays["sent_commands"][43:, :6].any()
        assert len(arrays["state128"]) == 84 and not arrays["action_mask"][-1].any()


def test_run_startup_receipt_matches_owned_child_contract_before_collection(tmp_path, monkeypatch):
    """Run the real worker entry and deliberately stop before any physics call."""
    module = collector()
    for key, value in dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1").items():
        monkeypatch.setenv(key, value)
    prepared_path = tmp_path / "prepared.json"
    prepared_path.write_text("{}\n")
    prepared_sha = module.sha256(prepared_path)
    prepared = dict(
        schema_version="s2a.native-training-preparation.v1",
        seed=42,
        output_root=str(tmp_path / "new_output"),
        records=[dict(name=f"case{index}", task_index=0) for index in range(10)],
    )
    observed = []

    def read_prepared(path, digest):
        assert path == prepared_path and digest == prepared_sha
        return prepared

    def stop_before_collection(*args):
        observed.append(args[2]["name"])
        raise RuntimeError("intentional fixture stop before collection; zero physics")

    monkeypatch.setitem(sys.modules, "prepare_wam_native_training", SimpleNamespace(read_prepared=read_prepared))
    monkeypatch.setattr(module, "collect_case", stop_before_collection)
    code = module.run(SimpleNamespace(prepared=str(prepared_path), prepared_sha256=prepared_sha, task_index=0, seed=42))
    directory = Path(prepared["output_root"]) / "collection/task0"
    started = json.loads((directory / "worker_started.json").read_text())
    # These exact top-level keys are consumed by owned_child.observe_started.
    assert started["cuda"] == os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert started["ld"] == os.environ["LD_LIBRARY_PATH"] == ""
    assert started["actual_environment"] == dict(
        CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1"
    )
    assert started["pid"] == os.getpid() and started["ppid"] == os.getppid()
    assert started["argv"] == sys.argv and started["task_index"] == 0
    result = json.loads((directory / "result.json").read_text())
    assert code == 1 and observed == ["case0"]
    assert result["complete"] is result["passed"] is False and result["cases"] == []
    assert "zero physics" in result["error"]["message"]
    assert result["environment_initializations"] == result["physics_steps"] == result["current_canvas_images"] == 0
    assert result["wam_initializations"] == result["denoise_calls"] == result["fits"] == result["fresh_cases"] == 0
