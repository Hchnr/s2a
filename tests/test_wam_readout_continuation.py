"""025 resource continuation gates, without real feature/fit/model execution."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def driver(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("readout_continuation_test", scripts / "continue_wam_readout.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_original(tmp_path, driver, monkeypatch):
    root = tmp_path / "original"
    folder = root / "orchestration/features/model"
    (folder / "ownership").mkdir(parents=True)
    (root / "features").mkdir()
    prepared = tmp_path / "prepared.json"
    prepared.write_text("registered original")
    original = dict(output_root=str(root))
    identities = dict(
        pid=862631,
        start_ticks=170222073,
        ppid=862480,
        command=["original"],
        cwd="/original",
        environment=dict(CUDA_VISIBLE_DEVICES="6", LD_LIBRARY_PATH=""),
    )
    owner = dict(pid=862480, start_ticks=170221627)
    launch = dict(child=identities, selected_environment=dict(CUDA_VISIBLE_DEVICES="6", LD_LIBRARY_PATH=""))
    fail = dict(
        complete=False,
        passed=False,
        stage="features",
        prepared_ref=driver.frozen.reference(prepared),
        error=dict(type="TimeoutError", message="025 owned worker exited but GPU6 memory release was not observed"),
    )
    paths = {
        root / "orchestration/features/result.json": fail,
        folder / "launch.json": launch,
        folder / "exit.json": dict(actual_wait=True, automatic_retry=False, returncode=0),
        folder / "ownership/release.json": dict(
            pid=862631,
            start_ticks=170222073,
            reaped=True,
            returncode=0,
            signal_sent=False,
            original_error=None,
            only_owned_child_signalled=True,
        ),
        folder / "actual_child_identity.json": identities,
    }
    for path, value in paths.items():
        path.write_text(json.dumps(value))
    observer = tmp_path / "observer.json"
    observer.write_text(
        json.dumps(
            dict(
                passed=True,
                actual_model=identities,
                actual_owner=owner,
                launch=driver.frozen.reference(folder / "launch.json"),
                startup_identity=driver.frozen.reference(folder / "actual_child_identity.json"),
            )
        )
    )
    tool = tmp_path / "tool.json"
    tool.write_text(
        json.dumps(
            dict(
                tool_handle_terminal=True,
                tool_reported_exit_code=1,
                exec_session_id=35611,
                original_owner_result=driver.frozen.reference(root / "orchestration/features/result.json"),
            )
        )
    )
    features = dict(
        complete=True,
        passed=True,
        prepared_ref=driver.frozen.reference(prepared),
        denoise_calls=249,
        teacher_calls=[{}] * 144,
        zero_identity=[{}] * 3,
        all_original_parameters_frozen=True,
    )
    labels = [f"zero:{i}" for i in range(3) for _ in range(35)] + [f"teacher:{i}" for i in range(144)]
    features["denoise_ledger"] = [dict(index=i, label=label, completed=True) for i, label in enumerate(labels)]
    (root / "features/result.json").write_text(json.dumps(features))
    release_peer = tmp_path / "release_peer.json"
    release_peer.write_text(
        json.dumps(
            dict(
                passed=True,
                current_resource_release_verified=True,
                prepared=driver.frozen.reference(prepared),
                original_owner_passed=False,
                input_sha256={},
            )
        )
    )
    incident_peer = tmp_path / "incident_peer.json"
    incident_peer.write_text(
        json.dumps(
            dict(
                passed=True,
                original_owner_failure_preserved=True,
                complete_249_boundary_and144_artifact_roster_verified=True,
                zero_full128_exact_verified=3,
                all_teacher_bundle_refs_verified=144,
                input_sha256={},
            )
        )
    )
    for key, path in (
        ("LAUNCH_PROOF", observer),
        ("OWNER_EXIT", tool),
        ("RELEASE_PEER", release_peer),
        ("INCIDENT_PEER", incident_peer),
    ):
        monkeypatch.setattr(driver, key, path)
        monkeypatch.setattr(driver, key + "_SHA", driver.sha256(path))
    monkeypatch.setattr(driver, "terminal", lambda value: dict(expected=value, original_birth_terminal=True))
    return prepared, original, paths


def test_original_false_preserved_only_registered_timeout_admitted(driver, monkeypatch, tmp_path):
    prepared, original, paths = synthetic_original(tmp_path, driver, monkeypatch)
    proof = driver.original_evidence(prepared, original)
    assert proof["original_owner_passed"] is False
    owner_path = Path(original["output_root"]) / "orchestration/features/result.json"
    original_failure = paths[owner_path]
    assert driver.frozen.read(owner_path) == original_failure
    wrong = copy.deepcopy(original_failure)
    wrong["error"] = dict(type="ValueError", message="unrelated failure")
    owner_path.write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="only the original resource-observation timeout"):
        driver.original_evidence(prepared, original)


def test_incomplete249_or_non_normal_exit_cannot_continue(driver, monkeypatch, tmp_path):
    prepared, original, paths = synthetic_original(tmp_path, driver, monkeypatch)
    features_path = Path(original["output_root"]) / "features/result.json"
    features = driver.frozen.read(features_path)
    features["denoise_ledger"][-1]["completed"] = False
    features_path.write_text(json.dumps(features))
    with pytest.raises(ValueError, match="249 denoise ledger differs"):
        driver.original_evidence(prepared, original)
    model_exit = Path(original["output_root"]) / "orchestration/features/model/exit.json"
    model_exit.write_text(json.dumps(dict(actual_wait=True, automatic_retry=False, returncode=1)))
    with pytest.raises(ValueError, match="normal actual wait"):
        driver.original_evidence(prepared, original)


def test_current_gpu_identity_rejects_wrong_uuid_and_retains_busy_apps(driver, monkeypatch):
    replies = iter(["6, wrong-uuid, 0\n"])
    monkeypatch.setattr(driver.subprocess, "check_output", lambda *args, **kwargs: next(replies))
    with pytest.raises(ValueError, match="UUID differs"):
        driver.resource_observation()
    replies = iter([f"6, {driver.GPU_UUID}, 100\n", f"123, {driver.GPU_UUID}, 100\n"])
    result = driver.resource_observation()
    assert result["memory_mib"] == 100 and len(result["compute_apps"]) == 1


def test_live_original_birth_is_rejected_and_actual_exit_is_observed(driver):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="")
    with subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.3)"], env=environment) as child:
        expected = driver.birth(child.pid)
        with pytest.raises(ValueError, match="still live"):
            driver.terminal(expected)
        assert child.wait(timeout=3) == 0
        assert driver.terminal(expected)["original_birth_terminal"] is True


def test_registration_sha_and_continuation_sources_are_explicit(driver):
    args = SimpleNamespace(registration=driver.REGISTRATION, registration_sha256="0" * 64)
    with pytest.raises(ValueError, match="registration differs"):
        driver.new_sources(args)
    args.registration_sha256 = driver.REGISTRATION_SHA
    sources = driver.new_sources(args)
    assert str(driver.REGISTRATION) in sources
    assert str(Path(driver.__file__).resolve()) in sources
    assert str(Path(__file__).resolve()) in sources


def test_cli_has_no_feature_regeneration_stage(driver):
    result = subprocess.run(
        [sys.executable, driver.__file__, "--stage", "features"],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "invalid choice: 'features'" in result.stderr
