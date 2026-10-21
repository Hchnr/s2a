"""029 guards reject invalid admission before any owned model launch."""

import argparse
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_wam_free_trajectory")


def args(tmp_path):
    p = tmp_path / "prepared.json"
    p.write_text("{}")
    return argparse.Namespace(
        stage="precision",
        prepared=p,
        prepared_sha256="h",
        source_peer=tmp_path / "peer.json",
        source_peer_sha256="p",
        dry_run=False,
    )


def test_cpu_owner_rejects_gpu_environment_before_reading_or_starting(module, tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.setattr(module, "read_prepared", lambda *a: pytest.fail("must reject before preparation"))
    with pytest.raises(ValueError, match="emptyCUDA"):
        module.execute(args(tmp_path))


def test_owner_rejects_unrelated_independent_preparation_before_launch(module, tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.setattr(module, "read_prepared", lambda *a: {"output_root": str(tmp_path / "outputs")})
    monkeypatch.setattr(
        module,
        "read",
        lambda *a: dict(
            schema_version="s2a.free-trajectory-independent-verification.v1",
            phase="prepare",
            passed=True,
            preparation_verified=True,
            prepared=dict(path="/wrong", sha256="wrong"),
            eligible_for_fit=False,
            physics_authorized=False,
        ),
    )
    monkeypatch.setattr(module, "owned_command", lambda *a, **k: pytest.fail("unrelated source must never launch"))
    with pytest.raises(ValueError, match="independent preparation"):
        module.execute(args(tmp_path))
    assert not (tmp_path / "outputs").exists()


def test_collection_missing_precision_rejected_before_stage_creation(module, tmp_path, monkeypatch):
    a = args(tmp_path)
    a.stage = "collection"
    root = tmp_path / "outputs"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.setattr(module, "read_prepared", lambda *a: {"output_root": str(root)})
    monkeypatch.setattr(
        module,
        "read",
        lambda *x: dict(
            schema_version="s2a.free-trajectory-independent-verification.v1",
            phase="prepare",
            passed=True,
            preparation_verified=True,
            prepared=module.reference(a.prepared),
            eligible_for_fit=False,
            physics_authorized=False,
            input_sha256={},
        ),
    )
    monkeypatch.setattr(module, "owned_command", lambda *a, **k: pytest.fail("missing qualification must never launch"))
    with pytest.raises(FileNotFoundError):
        module.execute(a)
    assert not root.exists()
