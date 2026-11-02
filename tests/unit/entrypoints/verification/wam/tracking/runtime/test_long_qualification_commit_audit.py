"""Git name canonicalization preserves one exact runtime and frozen physics checks."""

import hashlib
import importlib

import pytest

from tests.support.source_layout import current_module_name, historical_source_bytes


@pytest.fixture
def audit(monkeypatch):
    return importlib.import_module(current_module_name("scripts/audit_wam_long_qualification_commit.py"))


def test_short_and_full_identity_resolve_to_the_same_single_registered_commit(audit):
    assert audit.canonical_runtime_commit("d6fb4a6") == audit.RUNTIME_COMMIT
    assert audit.canonical_runtime_commit(audit.RUNTIME_COMMIT) == audit.RUNTIME_COMMIT
    with pytest.raises(ValueError, match="different registered source"):
        audit.canonical_runtime_commit("9fe490a")
    with pytest.raises(ValueError, match="missing runtime"):
        audit.canonical_runtime_commit("")


def test_physics_metric_and_prefix_helpers_are_the_unchanged_frozen_objects(audit):
    import scripts.verification.wam.physical_logs.audit_wam_long_qualification as original

    original_bytes = historical_source_bytes(
        dict(path=str(audit.ORIGINAL_CHECKER.relative_to(audit.REPO)), sha256=audit.ORIGINAL_CHECKER_SHA)
    )
    assert hashlib.sha256(original_bytes).hexdigest() == audit.ORIGINAL_CHECKER_SHA
    for name in ("native_check", "oracle_prefix", "factory_proof", "runtime_proof", "verify_case", "lines"):
        assert getattr(audit, name) is getattr(original, name)
