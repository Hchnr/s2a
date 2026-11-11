"""Verify the model Python's OpenCV mutation without running a GPU task."""

import hashlib
import os
import sys
from types import SimpleNamespace

import pytest

from scripts.wam.tracking.preview import causal_stream_live_admission as admission
from scripts.wam.tracking.preview import live_cv2_admission as cv2_admission


def test_actual_model_python_cv2_import_is_probed_in_a_cpu_child(tmp_path):
    inputs = admission.load_inputs()
    before = os.environ.get("LD_LIBRARY_PATH")
    result = cv2_admission.run_cpu_probe(inputs["prepared"]["model_config"]["model_python"], tmp_path)
    assert result["actual_wait"] and result["reaped"] and result["returncode"] == result["gpu_work"] == 0
    assert result["cuda_before"] == result["cuda_after"] == ""
    assert result["ld_before"] == "" and result["ld_after"]
    assert result["changes"]["LD_LIBRARY_PATH"] == ["", result["ld_after"]]
    assert len(result["sources"]) == 3 and os.environ.get("LD_LIBRARY_PATH") == before


def fake_probe(tmp_path, monkeypatch):
    directory = tmp_path / "cv2"
    directory.mkdir()
    paths = [directory / "__init__.py", directory / "config.py", directory / "config-3.py"]
    for index, path in enumerate(paths):
        path.write_text(f"source {index}\n")
    sources = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.delenv("QT_QPA_FONTDIR", raising=False)
    monkeypatch.delitem(sys.modules, "cv2", raising=False)
    probe = {
        "actual_wait": True,
        "reaped": True,
        "returncode": 0,
        "gpu_work": 0,
        "version": "test-version",
        "ld_after": "/known/lib:",
        "sources": sources,
        "changes": {"LD_LIBRARY_PATH": ["", "/known/lib:"], "QT_QPA_FONTDIR": [None, "/known/font"]},
    }

    def importer(_name):
        os.environ["LD_LIBRARY_PATH"] = "/known/lib:"
        os.environ["QT_QPA_FONTDIR"] = "/known/font"
        return SimpleNamespace(__file__=str(paths[0]), __version__="test-version")

    return probe, importer


def test_registered_mutation_is_restored_only_in_the_model_process(tmp_path, monkeypatch):
    probe, importer = fake_probe(tmp_path, monkeypatch)
    result = cv2_admission.admit_model_cv2(probe, gpu_uuid="GPU-test", importer=importer)
    assert result["complete"] and result["ld_restored"]
    assert os.environ["LD_LIBRARY_PATH"] == "" and os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-test"


def test_unregistered_import_mutation_is_rejected_without_silent_repair(tmp_path, monkeypatch):
    probe, importer = fake_probe(tmp_path, monkeypatch)
    probe["changes"]["LD_LIBRARY_PATH"][1] = "/unexpected:"
    with pytest.raises(ValueError, match="differently from its audited CPU probe"):
        cv2_admission.admit_model_cv2(probe, gpu_uuid="GPU-test", importer=importer)
    assert os.environ["LD_LIBRARY_PATH"] == "/known/lib:"
