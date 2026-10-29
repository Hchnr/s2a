"""An omitted duplicate does not permit missing or inconsistent plan identity."""

import hashlib
import importlib

import pytest

from tests.support.source_layout import current_module_name, historical_source_bytes


@pytest.fixture
def audit(monkeypatch):
    return importlib.import_module(current_module_name("scripts/audit_wam_long_qualification_plan.py"))


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
    import scripts.verification.wam.physical_logs.audit_wam_long_qualification as original

    for path, digest in (
        (audit.ORIGINAL_CHECKER, audit.ORIGINAL_CHECKER_SHA),
        (audit.PREVIOUS_CHECKER, audit.PREVIOUS_CHECKER_SHA),
    ):
        original_bytes = historical_source_bytes(dict(path=str(path.relative_to(audit.REPO)), sha256=digest))
        assert hashlib.sha256(original_bytes).hexdigest() == digest
    for name in ("native_check", "oracle_prefix", "factory_proof", "runtime_proof", "verify_case", "episode_check"):
        assert getattr(audit, name) is getattr(original, name)
