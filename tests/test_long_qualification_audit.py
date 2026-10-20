"""Clock, exact-prefix and real-available-reference qualification boundaries."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def audit(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("long_audit_test", root / "scripts/audit_wam_long_qualification.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_factory_and_physical_loop_ast_proof_reads_actual_committed_sources(audit):
    evidence = audit.Evidence()
    assert audit.factory_proof(evidence)["factory_reset_restore_ast_exact"]
    proof = audit.runtime_proof(evidence)
    assert proof["native_physical_try_body_ast_exact"]
    assert proof["oracle_wam_physical_try_body_ast_exact_except_readonly_clock"]
    evidence.recheck()


def test_small_numeric_agreement_never_becomes_exact600_prefix(audit):
    old = [dict(step=i, command=[0.0, 1.0]) for i in range(600)]
    current = [dict(step=i, command=[0.0, 1.0]) for i in range(1200)]
    assert audit.exact_prefix(current, old)["all600_shared_rows_exact"]
    current[590]["command"][0] = 1e-12
    result = audit.exact_prefix(current, old)
    assert not result["all600_shared_rows_exact"]
    assert result["within_existing_tolerance_1e10"]
    assert result["maximum_absolute_difference"] == 1e-12


def chunks(limit):
    result = []
    for step in range(0, limit, 8):
        count = min(40, limit - step)
        arrays = [[float(i)] for i in range(step, step + count)]
        result.append(
            dict(
                step=step,
                raw128=arrays,
                effective128=arrays,
                raw_targets_tcp=arrays,
                effective_targets_tcp=arrays,
                openings=[float(i) for i in range(step, step + count)],
                execution_endpoint_count=count,
                generation_s=limit,
                anchor=[step],
            )
        )
    return result


def store(path, limit, data):
    path.mkdir()
    (path / "chunks.json").write_text(json.dumps(data))
    (path / "steps.jsonl").write_text("".join(json.dumps(dict(step=i, x=float(i))) + "\n" for i in range(limit)))


def test_four_partial_old_queries_compare_only_existing_future_and_all_actual600(audit, tmp_path):
    new, old = tmp_path / "new", tmp_path / "old"
    current = chunks(1200)
    # Different data beyond old600 is allowed: it never existed in old query568.
    current[71]["raw128"] = [row[:] for row in current[71]["raw128"]]
    current[71]["raw128"][35] = [-123.0]
    store(new, 1200, current)
    store(old, 600, chunks(600))
    result = audit.oracle_prefix(new, old, audit.Evidence())
    assert result["all600_shared_rows_exact"] and result["all75_common_query_fields_and_available_endpoints_exact"]
    assert [r["old_available_endpoints"] for r in result["unavailable_old_future_suffixes_excluded"]] == [32, 24, 16, 8]
    current[71]["raw128"][31] = [-123.0]
    (new / "chunks.json").write_text(json.dumps(current))
    with pytest.raises(AssertionError):
        audit.oracle_prefix(new, old, audit.Evidence())


def test_clock_reconstructs1200_policy_steps_and_rejects_hidden_preroll(audit, tmp_path):
    observation = dict(
        effective_horizon=1300,
        ignore_done=False,
        internal_timestep=0,
        internal_done=False,
        control_freq=20,
        control_timestep_s=0.05,
        model_timestep_s=0.002,
        sim_time_s=0.25,
    )
    clock = dict(
        schema_version="s2a.long-episode-contract.v1",
        effective_horizon=1300,
        ignore_done=False,
        physical_steps=1200,
        observed_policy_steps=1200,
        initial_timestep=0,
        final_timestep=1200,
        scored_physical_step_budget=1200,
        initialization_policy_steps_added=0,
        initialization_excluded_from_scoring=True,
        initial_internal_done=False,
        final_internal_done=False,
        clock_consistent=True,
        control_freq=20,
        control_timestep_s=0.05,
        model_timestep_s=0.002,
        initial_sim_time_s=0.25,
        final_sim_time_s=60.25,
        initial_observation=observation,
        final_observation=dict(observation, internal_timestep=1200, sim_time_s=60.25),
    )
    path = tmp_path / "episode_contract.json"
    path.write_text(json.dumps(clock))
    rows = [dict(actual_dt=0.05) for _ in range(1200)]
    assert audit.episode_check(tmp_path, rows, audit.Evidence())["final_policy_timestep"] == 1200
    for changed in (dict(clock, initial_timestep=5), dict(clock, final_sim_time_s=61), dict(clock, ignore_done=True)):
        path.write_text(json.dumps(changed))
        with pytest.raises(AssertionError):
            audit.episode_check(tmp_path, rows, audit.Evidence())
