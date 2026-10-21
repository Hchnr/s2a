"""029 guards reject invalid admission before any owned model launch."""

import argparse
import importlib
import json
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


@pytest.mark.parametrize("backend_gold", [True, False])
def test_backend_specific_gold_controls_admission_with_cross_backend_difference(
    module, tmp_path, monkeypatch, backend_gold
):
    prepared_path = tmp_path / "prepared.json"
    prepared_path.write_text("{}")
    p, a = tmp_path / "precision.json", tmp_path / "audit.json"
    addendum = dict(path="/registered_addendum.md", sha256="pinned")
    prepared = dict(output_root=str(tmp_path), numerical_addendum_ref=addendum)
    precision = dict(
        schema_version="s2a.free-trajectory-synthetic-precision.v1",
        complete=True,
        passed=True,
        prepared_ref=module.reference(prepared_path),
        cases=[{}] * 20,
        backend_arithmetic_contract=module.BACKEND_CONTRACT,
        numerical_addendum_ref=addendum,
        cpu_cuda_full128_bitwise_equal=False,
    )
    p.write_text(json.dumps(precision))
    precision_sha = module.sha256(p)
    audit = dict(
        schema_version="s2a.free-trajectory-independent-verification.v1",
        passed=True,
        precision_verified=True,
        phase="precision",
        prepared=module.reference(prepared_path),
        input_sha256={str(p): precision_sha},
        backend_gold_verified=backend_gold,
        eligible_for_fit=False,
        physics_authorized=False,
    )
    monkeypatch.setattr(module, "read", lambda path, digest: precision if path == p else audit)
    monkeypatch.setattr(module, "owner_completed", lambda *args: {})
    if backend_gold:
        actual, _ = module.precision_admission(prepared, prepared_path, p, precision_sha, a, "a-sha")
        assert actual["cpu_cuda_full128_bitwise_equal"] is False
    else:
        with pytest.raises(ValueError, match="independent synthetic"):
            module.precision_admission(prepared, prepared_path, p, precision_sha, a, "a-sha")
