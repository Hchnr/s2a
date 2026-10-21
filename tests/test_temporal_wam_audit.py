"""020 must preserve first42 outputs while identifying the reference intervention."""

import copy
import hashlib
import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def audit(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("temporal_wam_audit_test", root / "scripts/audit_wam_temporal_wam.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pair(audit):
    request = {key: key for key in audit.FIRST_CONDITIONS}
    request["seed"] = 42
    variant = dict(
        name="position_hold", rotation="response", gripper="position_hold", governor=False, model_rotation_alpha=1.0
    )
    raw = np.zeros((40, 128)).tolist()
    old = dict(
        request=request,
        record=dict(name="same fixture"),
        variant=variant,
        controller=dict(actual_initial_physics_sha256="f" * 64),
        png=b"same pixels",
        normalized_action=raw,
        chunk=dict(
            step=0,
            anchor_tcp=np.eye(4).tolist(),
            anchor_opening=0.8,
            raw128=raw,
            effective128=raw,
            physics_unchanged=True,
            physics_before_sha256="f" * 64,
            physics_after_sha256="f" * 64,
            physics_before_time_s=0.0,
            physics_after_time_s=0.0,
        ),
        row=dict(
            measured_source=np.eye(4).tolist(),
            measured_twist=[0.0] * 6,
            opening_before=0.8,
            gripper_qpos_before=[0.03, -0.03],
            motor_target_before=[0.5, -0.5],
        ),
    )
    new = copy.deepcopy(old)
    new["variant"]["name"] = "temporal_equal"
    return old, new


def test_first_query_is_full42_regression_despite_the_new_variant_name(audit):
    old, new = pair(audit)
    result = audit.compare_first_inputs(old, new)
    assert result["first_query_full40x128_raw_exact"] and result["first_query_full40x128_normalized_exact"]
    assert result["variant_name_only_changed"] and result["actual_sampling_seed"] == 42


def test_no_seed_output_normalization_or_initial_state_substitution(audit):
    old, new = pair(audit)
    alternatives = []
    for key, value in (("seed", 43), ("seed", 42.0), ("guidance", 3)):
        bad = copy.deepcopy(new)
        bad["request"][key] = value
        alternatives.append(bad)
    for key in ("raw128", "effective128"):
        bad = copy.deepcopy(new)
        # Break deepcopy's shared list alias intentionally; each field is checked.
        bad["chunk"][key] = np.ones((40, 128)).tolist()
        alternatives.append(bad)
    bad = copy.deepcopy(new)
    bad["normalized_action"] = np.ones((40, 128)).tolist()
    alternatives.append(bad)
    bad = copy.deepcopy(new)
    bad["variant"]["model_rotation_alpha"] = 0.5
    alternatives.append(bad)
    bad = copy.deepcopy(new)
    bad["row"]["motor_target_before"] = [0.9, -0.9]
    alternatives.append(bad)
    for bad in alternatives:
        with pytest.raises(AssertionError):
            audit.compare_first_inputs(old, bad)


def test_new_coordinator_release_uses_real_ticks_and_wait_statuses(audit):
    roles = dict(model=dict(pid=10, start_ticks=100), matrix=dict(pid=11, start_ticks=101))
    released = dict(
        model_pid=10,
        model_start_ticks=100,
        model_returncode=-15,
        last_job_pid=11,
        last_job_returncode=0,
        cleanup_errors=[],
        only_owned_process_groups_stopped=True,
        automatic_retry=False,
        stop_evidence=dict(job=dict(pid=11), model=dict(pid=10)),
    )
    assert audit.check_release(released, roles)
    for key, value in (
        ("model_start_ticks", 999),
        ("model_returncode", -9),
        ("last_job_pid", 42),
        ("last_job_returncode", 1),
        ("cleanup_errors", ["failure"]),
        ("automatic_retry", True),
    ):
        bad = dict(released, **{key: value})
        with pytest.raises(AssertionError):
            audit.check_release(bad, roles)
    bad = copy.deepcopy(released)
    bad["stop_evidence"]["model"]["pid"] = 42
    with pytest.raises(AssertionError):
        audit.check_release(bad, roles)


def test_the_independent_ensemble_checker_is_frozen_and_reused(audit):
    import wam_temporal_position_checker as checker

    assert audit.verify_case is checker.verify_case and audit.verify_query is checker.verify_query
    original = subprocess.check_output(
        ["git", "show", "af147e1:scripts/wam_temporal_position_checker.py"], cwd=audit.REPO
    )
    proof = audit.source_proof(audit.Evidence())
    assert hashlib.sha256(original).hexdigest() == proof["new_checker_sha256"]
    assert proof["all_other_case_ast_nodes_exact"] and proof["literal_query_ast_exact"]
    assert not proof["production_temporal_module_imported"] and not proof["dynamic_patch_or_generated_function_used"]


def test_only_scope_independent_helpers_are_reused_from019(audit):
    import audit_wam_raw8_seed43 as previous

    for name in (
        "checkpoint_stats",
        "group_outcome",
        "http_budget",
        "opening_tracking",
        "optional_json",
        "service_artifacts",
    ):
        assert getattr(audit, name) is getattr(previous, name)
    assert audit.compare_first_inputs is not previous.compare_first_inputs
    assert audit.verify_case is not previous.verify_case
    assert audit.group_outcome([dict(status="pending")]) is None


def test_retained_suffix_usage_is_not_the_latest_query_clock(audit):
    chunks = [dict(step=q, raw128=[None] * 40) for q in range(0, 600, 8)]
    rows = []
    for step in range(600):
        queries = [q for q in range(0, step + 1, 8) if step - q < 40]
        rows.append(
            dict(
                step=step,
                temporal_ensemble=dict(
                    contributors=[
                        dict(query_step=q, endpoint_index=step - q, absolute_endpoint=step + 1, weight=1 / len(queries))
                        for q in queries
                    ]
                ),
            )
        )
    result = audit.ensemble_reference_usage(rows, chunks)
    assert result["physical_targets"] == 600
    assert result["recorded_contributor_endpoints"] == 2920
    assert result["unavailable_after_physical_budget"] == 80
    assert result["per_query"][-1]["recorded_contributor_endpoints"] == 8
    assert result["per_query"][0]["recorded_contributor_endpoints"] == 40
    for key, value in (("query_step", 600), ("absolute_endpoint", 999), ("weight", 1.0)):
        changed = copy.deepcopy(rows)
        changed[40]["temporal_ensemble"]["contributors"][0][key] = value
        with pytest.raises(AssertionError):
            audit.ensemble_reference_usage(changed, chunks)
