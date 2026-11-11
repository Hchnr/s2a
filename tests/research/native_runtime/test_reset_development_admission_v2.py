"""The repair keeps the original budget and requires the complete image audit."""

import copy
import json

import pytest

from scripts.simulation.runtime import run_reset_development_admission as original
from scripts.simulation.runtime import run_reset_development_admission_v2 as v2


def test_v2_keeps_reset_budget_and_exact_audit_binding():
    config = json.loads(v2.CONFIG.read_text())
    v2.validate_config(config)
    for key, value in (
        ("canvas_torch_threads", 1),
        ("max_native_steps", 1),
        ("canvas_thread_audit_sha256", "wrong"),
        ("automatic_retry", True),
    ):
        with pytest.raises(ValueError):
            v2.validate_config(dict(config, **{key: value}))


def test_audit_requires_every_case_and_retains_original_failure():
    audit = dict(
        complete=True,
        passed=True,
        sources_unchanged=True,
        control_qualified=False,
        original_failure_not_reclassified=True,
        failure_ref={"sha256": v2.FAILURE_SHA},
        cases=[{"case": {"case_index": i}, "one_matches_saved": True, "two_matches_recorded": True} for i in range(12)],
        native_models=0,
        forward_queries=0,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        torch_threads_restored=1,
    )
    v2.validate_audit(audit)
    changed = copy.deepcopy(audit)
    changed["cases"][7]["two_matches_recorded"] = False
    with pytest.raises(ValueError, match="exact-pixel"):
        v2.validate_audit(changed)
    changed = dict(audit, original_failure_not_reclassified=False)
    with pytest.raises(ValueError, match="exact-pixel"):
        v2.validate_audit(changed)


def test_reused_owner_has_private_paths_and_leaves_original_globals_unchanged():
    cloned = v2.isolated_owner(original.execute)
    assert cloned.__code__ is original.execute.__code__
    assert cloned.__globals__["OUTPUT"] == v2.OUTPUT
    assert original.execute.__globals__["OUTPUT"] == original.OUTPUT
    assert cloned.__globals__["load_config"] is v2.load_config
