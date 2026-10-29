"""Original tolerant command admission with unchanged bytes and isolated v2 owners."""

import copy
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import test_directional_derivative_execution as prior
from test_accepted_point_sqp_execution import native_qa
from test_prefix_callback_backtracking_execution import write_json
from test_prefix_trust_control_execution import CountingDynamics, captured_rollout, source_fixture


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return SimpleNamespace(
        producer=importlib.import_module("probe_tracking_v5_directional_derivative_v2"),
        owner=importlib.import_module("execute_tracking_v5_directional_derivative_v2"),
        evidence=importlib.import_module("s2a.directional_derivative_v2_evidence"),
    )


@pytest.mark.parametrize("excess", [13 * np.spacing(0.25), 5e-13])
def test_tolerated_first_rate_boundary_keeps_signed_zero_and_all_command_bytes(modules, excess):
    source, anchor, interface = source_fixture()
    anchor[:, 1] = -0.1
    anchor[0, 0], anchor[0, 1] = 0.25 + excess, -0.25 - excess
    original = anchor.tobytes()
    previous = source["previous_command"][:6]
    lower = np.tile(interface.input_min, (12, 1))
    upper = np.tile(interface.input_max, (12, 1))
    lower[0], upper[0] = np.maximum(lower[0], previous - 0.25), np.minimum(upper[0], previous + 0.25)
    assert np.clip(anchor, lower, upper).tobytes() != original  # The old exact-clipping gate rejects this fixture.
    assert np.signbit(anchor[0, 3]) and 0 < np.max(np.abs(anchor[0] - previous)) - 0.25 < 1e-12
    admitted = modules.evidence.admit_anchor(source, anchor, interface)
    assert admitted.tobytes() == original
    dynamics, calls = CountingDynamics(), []
    result = modules.producer.run_probe(
        source,
        anchor,
        prior.directions(),
        interface,
        dynamics,
        rollout_fn=prior.complete_synthetic_rollout(source, calls),
    )
    assert result["prediction_calls"] == result["response_evaluations"] == len(calls) == 80
    assert result["unique_command_count"] == 79
    assert result["numerical_inner_steps"] == 24000 and result["query_count"] == 24961
    assert calls[0]["commands"].tobytes() == calls[-1]["commands"].tobytes() == original == anchor.tobytes()
    assert np.asarray(result["answer"]["anchor_commands"]).tobytes() == original
    assert result["answer"]["repeatability"]["exact"]


@pytest.mark.parametrize("group", ["first_rate", "later_rate", "box"])
def test_outside_original_tolerance_is_rejected_before_any_query(modules, group):
    source, anchor, interface = source_fixture()
    if group == "first_rate":
        anchor[0, 0] = 0.25 + 2e-12
    elif group == "later_rate":
        anchor[1, 0] = anchor[0, 0] + 0.25 + 2e-12
    else:
        source["previous_command"][0] = 0.9
        anchor[:, 0] = 0.9
        anchor[0, 0] = 1.0 + 2e-12
    original = anchor.tobytes()
    dynamics, calls = CountingDynamics(), []
    with pytest.raises(ValueError):
        modules.producer.run_probe(
            source,
            anchor,
            prior.directions(),
            interface,
            dynamics,
            rollout_fn=captured_rollout(source, calls),
        )
    assert dynamics.queries == 0 and calls == [] and anchor.tobytes() == original


def test_tolerated_later_rate_boundary_is_preserved_and_copy_owned(modules):
    source, anchor, interface = source_fixture()
    anchor[1, 0] = anchor[0, 0] + 0.25 + 5e-13
    original = anchor.tobytes()
    admitted = modules.evidence.admit_anchor(source, anchor, interface)
    assert admitted.tobytes() == original and admitted is not anchor
    admitted[0, 0] = 0.0
    assert anchor.tobytes() == original


@pytest.mark.parametrize("parallel,expected_calls", [(False, 80), (True, 28)])
def test_v2_keeps_default_fixed_grid_cache_and_final_reset(modules, parallel, expected_calls):
    prior.test_default_probe_bills_grid_cache_and_resets_final_repeat(modules, parallel, expected_calls)


@pytest.mark.parametrize(
    "defect,expected_calls,match",
    [
        ("after_repeat", 2, "after the final anchor repeat"),
        ("nonanchor_duplicate", 2, "Only the final anchor"),
        ("wrong_first", 0, "First billed response"),
        ("over_budget", 80, "fixed80 budget"),
    ],
)
def test_v2_preserves_raw_call_rejection_and_billing(modules, defect, expected_calls, match):
    prior.test_raw_predictor_rejects_unregistered_duplicate_budget_and_command_calls(
        modules,
        defect,
        expected_calls,
        match,
    )


@pytest.mark.parametrize("stage,timeout", [("diagnose", 600), ("check", 1800)])
def test_v2_owner_uses_only_new_entry_and_preserves_lifecycle(modules, monkeypatch, tmp_path, stage, timeout):
    prior.test_owner_fixed_cpu_timeout_and_complete_count_with_real_hash_lifecycle(
        modules,
        monkeypatch,
        tmp_path,
        stage,
        timeout,
    )
    assert modules.owner.entry_command(stage)[1].endswith("_tracking_v5_directional_derivative_v2.py")


@pytest.mark.parametrize("stage", ["diagnose", "check"])
def test_v2_owner_dry_never_launches(modules, monkeypatch, tmp_path, stage):
    prior.test_owner_dry_only_validates_metadata_and_never_launches(modules, monkeypatch, tmp_path, stage)


@pytest.mark.parametrize("defect", ["different_qa", "no_wait", "not_reaped", "signal", "changed_source"])
def test_v2_owner_rejects_unclosed_or_changed_producer(modules, monkeypatch, tmp_path, defect):
    prior.test_unclosed_or_changed_producer_cannot_launch_checker(modules, monkeypatch, tmp_path, defect)


@pytest.mark.parametrize("defect", ["partial", "child_failure", "source_changed_during_child"])
def test_v2_failure_does_not_retry_or_certify_partial_replay(modules, monkeypatch, tmp_path, defect):
    prior.test_failure_keeps_failed_owner_without_retry_or_partial_replay(modules, monkeypatch, tmp_path, defect)


def test_v2_qa_covers68_sources_and21_modules_without_mutating_v1_contract(modules, monkeypatch, tmp_path):
    v1 = importlib.import_module("execute_tracking_v5_directional_derivative")
    old_sources, old_tests = v1.QA_SOURCES, v1.QA_TESTS
    assert len(old_sources) == 62 and len(old_tests) == 20
    assert len(modules.owner.QA_SOURCES) == 68 and len(modules.owner.QA_TESTS) == 21
    new_sources = set(modules.owner.QA_SOURCES) - set(old_sources)
    new_tests = set(modules.owner.QA_TESTS) - set(old_tests)
    assert len(new_sources) == 6 and new_tests == {"tests/test_directional_derivative_v2.py"}
    qa = native_qa(modules, monkeypatch, tmp_path)
    for module in (v1, importlib.import_module("execute_tracking_v5_accepted_point_sqp")):
        monkeypatch.setattr(module, "REPO", tmp_path)
        monkeypatch.setattr(module, "NATIVE_PYTHON", tmp_path / "native_python")
    target = tmp_path / "qa/result.json"
    assert modules.owner.verify_native_qa(write_json(target, qa)) == qa
    for source in new_sources:
        changed = copy.deepcopy(qa)
        del changed["source_sha256"][source]
        with pytest.raises(ValueError):
            modules.owner.verify_native_qa(write_json(target, changed))
    changed = copy.deepcopy(qa)
    changed["command"].remove(next(iter(new_tests)))
    with pytest.raises(ValueError):
        modules.owner.verify_native_qa(write_json(target, changed))
    ref = write_json(target, qa)
    (tmp_path / "scripts/probe_tracking_v5_directional_derivative_v2.py").write_text("changed after v2 QA")
    with pytest.raises(ValueError):
        modules.owner.verify_native_qa(ref)
    assert v1.QA_SOURCES is old_sources and v1.QA_TESTS is old_tests


def test_v2_producer_dry_remains_metadata_only(modules, monkeypatch, tmp_path):
    prior.test_producer_dry_uses_only_prerequisite_metadata_and_never_constructs_dynamics(
        modules, monkeypatch, tmp_path
    )
