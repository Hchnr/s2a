"""An omitted duplicate does not permit missing or inconsistent plan identity."""

import hashlib
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def audit(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "long_plan_audit_test", root / "scripts/audit_wam_long_qualification_plan.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_review_binds_the_exact_oracle_plan_when_duplicate_is_absent(audit):
    name = str(audit.PLAN)
    digest = audit.sha256(audit.PLAN)
    launch = dict(execution_source_sha256={})
    review = dict(input_sha256={name: digest})
    result = audit.check_launch_plan_binding(launch, review, "oracle")
    assert result["sha256"] == digest
    assert result["source_review_binding"]
    assert result["oracle_duplicate_map_omission"]
    assert not result["execution_map_binding"]
    with pytest.raises(ValueError, match="only the known oracle"):
        audit.check_launch_plan_binding(launch, review, "native")


def test_absent_or_conflicting_identity_cannot_be_replaced_with_a_claim(audit):
    name = str(audit.PLAN)
    digest = audit.sha256(audit.PLAN)
    for review in ({}, {name: "0" * 64}):
        with pytest.raises(ValueError, match="review must bind"):
            audit.check_launch_plan_binding(
                dict(execution_source_sha256={name: digest}), dict(input_sha256=review), "oracle"
            )
    with pytest.raises(ValueError, match="execution plan fingerprint differs"):
        audit.check_launch_plan_binding(
            dict(execution_source_sha256={name: "0" * 64}), dict(input_sha256={name: digest}), "oracle"
        )
    result = audit.check_launch_plan_binding(
        dict(execution_source_sha256={name: digest}), dict(input_sha256={name: digest}), "native"
    )
    assert result["execution_map_binding"] and not result["oracle_duplicate_map_omission"]


def test_original_physics_helpers_and_both_prior_checker_bytes_stay_frozen(audit):
    import audit_wam_long_qualification as original

    for path, digest in (
        (audit.ORIGINAL_CHECKER, audit.ORIGINAL_CHECKER_SHA),
        (audit.PREVIOUS_CHECKER, audit.PREVIOUS_CHECKER_SHA),
    ):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    for name in ("native_check", "oracle_prefix", "factory_proof", "runtime_proof", "verify_case", "episode_check"):
        assert getattr(audit, name) is getattr(original, name)
