"""Explicit import admission, including one real CPU env311 OpenCV import."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from s2a.artifacts import REPO
from s2a.native_environment import (
    EnvironmentAdmissionError,
    environment_snapshot,
    restore_cv2_environment,
    strict_native_environment,
)


def ref(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def import_fixture(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    for key, value in dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1").items():
        monkeypatch.setenv(key, value)
    monkeypatch.delitem(sys.modules, "cv2", raising=False)
    package = tmp_path / "cv2"
    package.mkdir()
    for name in ("__init__.py", "config.py", "config-3.py"):
        (package / name).write_text(f"# synthetic source {name}\n")
    loader = ref(package / "__init__.py")
    sources = [loader, ref(package / "config.py"), ref(package / "config-3.py")]
    expected_ld = str(package / "../../lib64") + ":"
    probe = dict(
        schema_version="s2a.native-training-import-environment-probe.v1",
        before=environment_snapshot(),
        after_import_cv2=dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=expected_ld, cwd=str(REPO)),
        cv2_loader_ref=loader,
        cv2_version="fixture-version",
        reproduces_empty_ld_guard_rejection=True,
        new_environment_initializations=0,
        new_physics_steps=0,
        new_model_calls=0,
        new_hdf_reads=0,
    )
    probe_path = tmp_path / "probe.json"
    probe_path.write_text(json.dumps(probe))
    return dict(probe=ref(probe_path), sources=sources, loader=loader, expected_ld=expected_ld)


def fake_import(monkeypatch, fixture, *, mutate=None):
    def importer(name):
        assert name == "cv2"
        os.environ["LD_LIBRARY_PATH"] = fixture["expected_ld"]
        if mutate is not None:
            mutate()
        return SimpleNamespace(__file__=fixture["loader"]["path"], __version__="fixture-version")

    monkeypatch.setattr("s2a.native_environment.importlib.import_module", importer)


def test_exact_import_mutation_is_admitted_and_only_local_ld_restored(import_fixture, monkeypatch):
    fake_import(monkeypatch, import_fixture)
    result = restore_cv2_environment(import_fixture["probe"], import_fixture["sources"])
    assert result["complete"] is result["passed"] is result["restore_performed"] is True
    assert result["after_import_cv2"]["LD_LIBRARY_PATH"] == import_fixture["expected_ld"]
    assert result["before"] == result["after_restore"] == environment_snapshot()
    assert result["bytecode_after_restore"] == "1"
    assert result["source_before_sha256"] == result["source_after_sha256"]
    assert result["pid"] == os.getpid() and result["argv"] == sys.argv
    assert os.environ["LD_LIBRARY_PATH"] == "" and os.environ["CUDA_VISIBLE_DEVICES"] == ""


@pytest.mark.parametrize("changed", ["ld", "cuda", "cwd", "bytecode"])
def test_unknown_import_mutations_are_preserved_and_rejected(import_fixture, monkeypatch, tmp_path, changed):
    def mutate():
        if changed == "cwd":
            monkeypatch.chdir(tmp_path)
        else:
            key = dict(ld="LD_LIBRARY_PATH", cuda="CUDA_VISIBLE_DEVICES", bytecode="PYTHONDONTWRITEBYTECODE")[changed]
            monkeypatch.setenv(key, "unregistered")

    fake_import(monkeypatch, import_fixture, mutate=mutate)
    with pytest.raises(EnvironmentAdmissionError) as caught:
        restore_cv2_environment(import_fixture["probe"], import_fixture["sources"])
    receipt = caught.value.receipt
    assert receipt["complete"] is receipt["passed"] is receipt["restore_performed"] is False
    assert receipt["failure_environment"] == environment_snapshot()
    assert "after_restore" not in receipt
    assert os.environ["LD_LIBRARY_PATH"] != ""  # Unknown states are never silently repaired.


def test_loader_source_change_rejected_before_import_and_strict_guard_never_resets(import_fixture, monkeypatch):
    def forbidden_import(name):
        raise AssertionError("changed runtime source must reject before import")

    monkeypatch.setattr("s2a.native_environment.importlib.import_module", forbidden_import)
    Path(import_fixture["sources"][1]["path"]).write_text("changed source\n")
    with pytest.raises(EnvironmentAdmissionError, match="runtime source changed"):
        restore_cv2_environment(import_fixture["probe"], import_fixture["sources"])
    monkeypatch.setenv("LD_LIBRARY_PATH", "unknown")
    with pytest.raises(ValueError, match="strict CPU"):
        strict_native_environment()
    assert os.environ["LD_LIBRARY_PATH"] == "unknown"


def test_real_env311_cv2_import_matches_registered_probe_and_restores_cpu_environment(tmp_path):
    """Import-only owned subprocess: no HDF5, MuJoCo, LIBERO or model imports."""
    interpreter = REPO / "outputs/libero_env311/bin/python"
    probe_path = (
        REPO / "outputs/wam_stability_20261021/native_training_runtime_observer/cv2_import_environment_probe.json"
    )
    if not interpreter.exists() or not probe_path.exists():
        pytest.skip("registered native env311 import-only fixture not installed")
    output = tmp_path / "actual_import.json"
    code = """
import json
import sys
from pathlib import Path
from s2a.native_environment import digest, restore_cv2_environment
probe_path, output = map(Path, sys.argv[1:])
probe = json.loads(probe_path.read_text())
loader = Path(probe['cv2_loader_ref']['path'])
sources = [dict(path=str(p), sha256=digest(p)) for p in
           (loader, loader.parent/'config.py', loader.parent/'config-3.py')]
receipt = restore_cv2_environment(dict(path=str(probe_path), sha256=digest(probe_path)), sources)
assert not any(name == p or name.startswith(p+'.') for name in sys.modules
               for p in ('mujoco', 'robosuite', 'libero', 'h5py', 'torch'))
receipt['simulator_hdf_model_modules_loaded'] = []
with output.open('x') as stream:
    json.dump(receipt, stream)
"""
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        LD_LIBRARY_PATH="",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(REPO / "src"),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    child = subprocess.Popen(
        [str(interpreter), "-c", code, str(probe_path), str(output)],
        cwd=REPO,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = child.communicate(timeout=60)
    assert child.returncode == 0, (stdout, stderr)
    receipt = json.loads(output.read_text())
    assert receipt["complete"] is receipt["passed"] is receipt["restore_performed"] is True
    assert receipt["before"] == receipt["after_restore"]
    assert receipt["after_import_cv2"]["LD_LIBRARY_PATH"] != ""
    assert receipt["simulator_hdf_model_modules_loaded"] == []
    assert receipt["new_physics_steps"] == receipt["new_environment_initializations"] == receipt["new_model_calls"] == 0
