"""Verify final-only cache admission and preserved failure accounting."""

import copy
import json
import os

import pytest

from scripts.simulation.runtime import adaptive_long_final_endpoint_owner as owner
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from tests.research.native_runtime.test_adaptive_long_cached_checker_owner import cached_branch, partial_branch, write


def final_branch():
    branch = cached_branch()
    events = branch["runtime"]["adaptive_cached_endpoint_events"]
    branch["runtime"]["adaptive_cached_endpoint_events"] = [event for event in events if event["step"] == 113]
    return branch


def test_final_cache_is_bound_to_the_recorded_endpoint_only():
    result = owner.audit_final_cache(final_branch(), final_step=113)
    assert result["observed_steps"] == result["reused_steps"] == [113]
    assert result["event_count"] == 2 and result["final_endpoint_reused"]


@pytest.mark.parametrize("change", ["early", "extra", "state", "missing"])
def test_invalid_final_cache_sequence_is_rejected(change):
    branch = final_branch()
    events = branch["runtime"]["adaptive_cached_endpoint_events"]
    if change == "early":
        early = copy.deepcopy(events[0])
        early["step"] = 112
        events.insert(0, early)
    elif change == "extra":
        events.append(copy.deepcopy(events[-1]))
    elif change == "state":
        events[-1]["state_sha256"] = "0" * 64
    else:
        events.pop()
    with pytest.raises(ValueError):
        owner.audit_final_cache(branch, final_step=113)


def input_costs():
    config = json.loads(owner.admission.CONFIG.read_text())
    return dict(
        config=config,
        previous_physical_bounds=copy.deepcopy(config["previous_physical_bounds"]),
        previous_model_bounds=copy.deepcopy(config["previous_model_bounds"]),
    )


@pytest.mark.parametrize("mode", ["partial", "missing", "unreaped", "changed_source"])
def test_incomplete_replay_costs_keep_actual_or_registered_bounds(tmp_path, monkeypatch, mode):
    inputs, result, refs = input_costs(), {}, [{"path": "source", "sha256": "a" * 64}]
    monkeypatch.setattr(owner, "source_refs", lambda: [] if mode == "changed_source" else refs)
    directory = tmp_path / "checker"
    if mode != "missing":
        write(directory / "simulation/result.json", partial_branch())
    lifecycle = dict(workers=[dict(key="checker", actual_wait=True, reaped=mode != "unreaped")])
    owner.phase_costs(result, inputs=inputs, sources=refs, lifecycle=lifecycle, directory=directory)
    expected = dict(lower=2181, upper=2297) if mode == "partial" else dict(lower=2174, upper=2436)
    assert result["cumulative_physical_bounds"]["control_steps"] == expected
    assert result["cumulative_model_bounds"] == inputs["previous_model_bounds"]
    assert result["model_requests"] == result["denoise_calls"] == 0


@pytest.mark.parametrize("change", [None, "parallel", "skip", "unreaped", "target", "dry_source", "dry_physics"])
def test_final_replay_requires_complete_matching_cpu_evidence(tmp_path, monkeypatch, change):
    inputs = {"identity": "recorded"}
    refs, sources = [{"path": "source", "sha256": "a" * 64}], {"source": "a" * 64}
    before = write(tmp_path / "before.json", {"sources": sources})
    qa = dict(
        complete=True,
        passed=True,
        all_tests_passed=True,
        source_unchanged=True,
        actual_wait=True,
        reaped=True,
        signal_sent=False,
        configured_parallelism=64,
        execution_parallelism=64,
        skipped_count=0,
        xfail_count=0,
        xpass_count=0,
        aggregate_node_partition_exact=True,
        call_count=1,
        collection={"collection_errors": []},
        targets=list(owner.REQUIRED_QA_TARGETS),
        source_before_ref=before,
    )
    dry = dict(
        complete=True,
        passed=True,
        automatic_retry=False,
        inputs=inputs,
        source_refs=refs,
        native_models=0,
        physics_steps=0,
        model_initializations=0,
        model_requests=0,
        denoise_calls=0,
        fresh_cases=0,
    )
    monkeypatch.setattr(owner.admission, "QA", tmp_path / "qa.json")
    monkeypatch.setattr(owner.admission, "DRY", tmp_path / "dry.json")
    monkeypatch.setattr(owner.admission, "input_receipt", lambda value: value)
    monkeypatch.setattr(owner, "source_refs", lambda: refs)
    monkeypatch.setattr(owner, "snapshot", lambda _root: {"sources": sources})
    if change == "parallel":
        qa["execution_parallelism"] = 32
    elif change == "skip":
        qa["skipped_count"] = 1
    elif change == "unreaped":
        qa["reaped"] = False
    elif change == "target":
        qa["targets"].pop()
    elif change == "dry_source":
        dry["source_refs"] = []
    elif change == "dry_physics":
        dry["physics_steps"] = 1
    write(owner.admission.QA, qa)
    write(owner.admission.DRY, dry)
    if change is None:
        assert owner.check_cpu_admission(inputs) == sources
    else:
        with pytest.raises(ValueError):
            owner.check_cpu_admission(inputs)


@pytest.mark.parametrize("succeed", [False, True])
def test_owner_records_outcome_and_does_not_retry(tmp_path, monkeypatch, succeed):
    inputs, refs = input_costs(), [{"path": "source", "sha256": "a" * 64}]
    output = tmp_path / "trial"
    inputs.update(long_inputs={"config": {}}, producer={"counts": {"control_steps": 147}}, source_sha256={})
    inputs["producer"]["branch_ref"] = write(tmp_path / "producer.json", {"fixture": "producer"})
    inputs["producer"]["model_ref"] = write(tmp_path / "model.json", {"fixture": "model"})
    monkeypatch.setattr(owner.admission, "OUTPUT", output)
    monkeypatch.setattr(owner.admission, "load_inputs", lambda: inputs)
    monkeypatch.setattr(owner.admission, "input_receipt", lambda _inputs: {"identity": "recorded"})
    monkeypatch.setattr(owner, "check_cpu_admission", lambda _inputs: {})
    monkeypatch.setattr(owner, "source_refs", lambda: refs)
    monkeypatch.setattr(owner, "snapshot", lambda _root: {"sources": {}})
    monkeypatch.setattr(owner.historical, "load_recorded_responses", lambda _branch, _model: ["recorded"])
    counts = dict(control_steps=147, native_steps=3675, models=3, contexts=3)
    monkeypatch.setattr(owner.audit, "summarize_branch", lambda _branch, _phase, _config: counts.copy())
    monkeypatch.setattr(owner.audit, "compare_branches", lambda _producer, _branch: {"state": 0.0})
    monkeypatch.setattr(owner, "audit_final_cache", lambda _branch, final_step: {"final_endpoint_reused": True})
    monkeypatch.setattr(
        owner.previous, "compare_captures", lambda _producer, _branch: [107, 111, 119, 120, 128, 136, 144]
    )
    calls = []

    def run(specs, *, cwd, timeout_seconds):
        calls.append(specs)
        assert cwd == prefix.REPO and timeout_seconds == 3600
        assert os.environ["CUDA_VISIBLE_DEVICES"] == os.environ["LD_LIBRARY_PATH"] == ""
        assert specs[0]["argv"][-1] == "--worker"
        if succeed:
            runtime = dict(
                control_steps_attempted=147,
                control_steps_completed=147,
                counters=dict(
                    mj_step=dict(attempted=3675, completed=3675),
                    model_construct=dict(attempted=3, completed=3),
                    context=dict(attempted=3, completed=3),
                ),
            )
            branch = dict(complete=True, passed=True, runtime=runtime)
        else:
            branch = partial_branch()
        write(output / "checker/simulation/result.json", branch)
        specs[0]["log"].write_text("synthetic worker")
        return dict(
            launch_error=None,
            workers=[
                dict(
                    key="checker",
                    log=str(specs[0]["log"]),
                    returncode=0 if succeed else 1,
                    actual_wait=True,
                    reaped=True,
                    signal_sent=False,
                )
            ],
        )

    monkeypatch.setattr(owner, "run_owned_workers", run)
    if succeed:
        owner.execute()
    else:
        with pytest.raises(ValueError, match="exit and be reaped"):
            owner.execute()
    result = json.loads((output / "checker/result.json").read_text())
    assert len(calls) == 1 and result["complete"] is result["passed"] is succeed
    assert result["automatic_retry"] is False and result["model_requests"] == 0
    if succeed:
        assert result["cumulative_physical_bounds"]["control_steps"] == dict(lower=2321, upper=2436)
        assert result["cumulative_physical_bounds"]["native_steps"] == dict(lower=58025, upper=60900)
    else:
        assert result["cumulative_physical_bounds"]["control_steps"] == dict(lower=2181, upper=2297)
    with pytest.raises(FileExistsError):
        owner.execute()
    assert len(calls) == 1
