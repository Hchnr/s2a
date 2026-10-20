"""Git name canonicalization preserves one exact runtime and frozen physics checks."""

import hashlib
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def audit(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "long_commit_audit_test", root / "scripts/audit_wam_long_qualification_commit.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_short_and_full_identity_resolve_to_the_same_single_registered_commit(audit):
    assert audit.canonical_runtime_commit("d6fb4a6") == audit.RUNTIME_COMMIT
    assert audit.canonical_runtime_commit(audit.RUNTIME_COMMIT) == audit.RUNTIME_COMMIT
    with pytest.raises(ValueError, match="different registered source"):
        audit.canonical_runtime_commit("9fe490a")
    with pytest.raises(ValueError, match="missing runtime"):
        audit.canonical_runtime_commit("")


def test_physics_metric_and_prefix_helpers_are_the_unchanged_frozen_objects(audit):
    import audit_wam_long_qualification as original

    assert hashlib.sha256(audit.ORIGINAL_CHECKER.read_bytes()).hexdigest() == audit.ORIGINAL_CHECKER_SHA
    for name in ("native_check", "oracle_prefix", "factory_proof", "runtime_proof", "verify_case", "lines"):
        assert getattr(audit, name) is getattr(original, name)
