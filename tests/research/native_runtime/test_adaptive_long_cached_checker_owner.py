"""Check replay admission, endpoint evidence and conservative physical accounting."""

import copy
import json
import os

import pytest

from scripts.simulation.runtime import adaptive_long_cached_checker_owner as owner
from scripts.simulation.runtime import audit_adaptive_long_partial as partial
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from tests.research.native_runtime.test_adaptive_live_bowl_audit import rewrapped


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return prefix.reference(path)


def cached_branch():
    _config, _producer, branch = rewrapped()
    prior, adaptive = owner.historical_audit.periods(branch, "checker")
    rows = prior + adaptive
    events = []
    for step in range(107, len(rows) + 1):
        endpoint = rows[step - 1]["endpoint_receipt"]
        event = dict(step=step, time=endpoint["sim_time"], state_sha256=endpoint["sim_state_sha256"], kind="observed")
        events.append(event)
        if step in (110, len(rows)):
            events.append({**event, "kind": "reused"})
    branch["runtime"]["adaptive_cached_endpoint_events"] = events
    return branch


def test_events_bind_saved_endpoints_and_final_reuse():
    result = owner.audit_cache_events(cached_branch(), start_step=107)
    assert result["observed_steps"] == list(range(107, 114))
    assert result["reused_steps"] == [110, 113]
    assert result["event_count"] == 9 and result["final_endpoint_reused"]


@pytest.mark.parametrize("change", ["hash", "time", "order", "first_reuse", "missing", "final", "extra"])
def test_changed_cache_evidence_is_rejected(change):
    branch = cached_branch()
    events = branch["runtime"]["adaptive_cached_endpoint_events"]
    if change == "hash":
        events[1]["state_sha256"] = "0" * 64
    elif change == "time":
        events[1]["time"] += 1e-12
    elif change == "order":
        events[0], events[1] = events[1], events[0]
    elif change == "first_reuse":
        events[0]["kind"] = "reused"
    elif change == "missing":
        events.pop(1)
    elif change == "final":
        events.pop()
    else:
        events[-1]["unexpected"] = True
    with pytest.raises(ValueError):
        owner.audit_cache_events(branch, start_step=107)


def test_actual_saved_capture_bytes_are_checked_not_only_receipt_names():
    producer = prefix.read(prefix.read(partial.PRODUCER)["branch_ref"])
    checker = prefix.read(prefix.read(partial.CHECKER)["available_branch_ref"])
    assert owner.compare_captures(producer, checker) == [107, 111, 119, 120, 128, 136, 144]
    checker["runtime"]["adaptive_checker_captures"][0]["raw_cameras"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        owner.compare_captures(producer, checker)


def gate_context(tmp_path, monkeypatch):
    inputs = {"identity": "recorded"}
    refs = [{"path": "source", "sha256": "a" * 64}]
    sources = {"source": "a" * 64}
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
    return inputs, qa, dry


@pytest.mark.parametrize("change", [None, "parallel", "skip", "unreaped", "target", "source", "dry_physics"])
def test_cpu_admission_rejects_incomplete_or_stale_evidence(tmp_path, monkeypatch, change):
    inputs, qa, dry = gate_context(tmp_path, monkeypatch)
    if change == "parallel":
        qa["execution_parallelism"] = 32
    elif change == "skip":
        qa["skipped_count"] = 1
    elif change == "unreaped":
        qa["reaped"] = False
    elif change == "target":
        qa["targets"].pop()
    elif change == "source":
        dry["source_refs"] = []
    elif change == "dry_physics":
        dry["physics_steps"] = 1
    write(owner.admission.QA, qa)
    write(owner.admission.DRY, dry)
    if change is None:
        assert owner.check_cpu_admission(inputs) == {"source": "a" * 64}
    else:
        with pytest.raises(ValueError):
            owner.check_cpu_admission(inputs)


def cost_inputs():
    config = json.loads(owner.admission.CONFIG.read_text())
    return dict(
        config=config,
        previous_physical_bounds=copy.deepcopy(config["previous_physical_bounds"]),
        previous_model_bounds=copy.deepcopy(config["previous_model_bounds"]),
    )


def partial_branch():
    return dict(
        complete=False,
        runtime=dict(
            control_steps_attempted=8,
            control_steps_completed=7,
            counters=dict(
                mj_step=dict(attempted=198, completed=197),
                model_construct=dict(attempted=2, completed=2),
                context=dict(attempted=1, completed=1),
            ),
        ),
    )


@pytest.mark.parametrize("case", ["partial", "missing", "unreaped", "source_changed"])
def test_physical_costs_keep_partial_or_registered_unknown_bounds(tmp_path, monkeypatch, case):
    inputs, result, refs = cost_inputs(), {}, [{"path": "source", "sha256": "a" * 64}]
    monkeypatch.setattr(owner, "source_refs", lambda: [] if case == "source_changed" else refs)
    directory = tmp_path / "checker"
    if case != "missing":
        write(directory / "simulation/result.json", partial_branch())
    lifecycle = dict(workers=[dict(key="checker", actual_wait=True, reaped=case != "unreaped")])
    owner.phase_costs(result, inputs=inputs, sources=refs, lifecycle=lifecycle, directory=directory)
    if case == "partial":
        assert result["cumulative_physical_bounds"]["control_steps"] == dict(lower=2062, upper=2178)
        assert result["cumulative_physical_bounds"]["native_steps"] == dict(lower=51572, upper=54448)
    else:
        assert result["cumulative_physical_bounds"]["control_steps"] == dict(lower=2055, upper=2317)
        assert result["cumulative_physical_bounds"]["native_steps"] == dict(lower=51375, upper=57925)
    assert result["cumulative_model_bounds"] == inputs["previous_model_bounds"]
    assert result["model_requests"] == result["denoise_calls"] == 0


@pytest.mark.parametrize("succeed", [False, True])
def test_owner_publishes_failure_or_audited_completion_and_never_retries(tmp_path, monkeypatch, succeed):
    inputs, refs = cost_inputs(), [{"path": "source", "sha256": "a" * 64}]
    output = tmp_path / "trial"
    inputs.update(
        long_inputs={"config": {"prefix_steps": 107}},
        producer={"counts": {"control_steps": 147}},
        source_sha256={},
    )
    inputs["producer"]["branch_ref"] = write(tmp_path / "producer.json", {"fixture": "producer"})
    inputs["producer"]["model_ref"] = write(tmp_path / "model.json", {"fixture": "model"})
    monkeypatch.setattr(owner.admission, "OUTPUT", output)
    monkeypatch.setattr(owner.admission, "load_inputs", lambda: inputs)
    monkeypatch.setattr(owner.admission, "input_receipt", lambda _inputs: {"identity": "recorded"})
    monkeypatch.setattr(owner, "check_cpu_admission", lambda _inputs: {})
    monkeypatch.setattr(owner, "source_refs", lambda: refs)
    monkeypatch.setattr(owner, "snapshot", lambda _root: {"sources": {}})
    monkeypatch.setattr(owner.historical_worker, "load_recorded_responses", lambda _branch, _model: ["recorded"])
    counts = dict(control_steps=147, native_steps=3675, models=3, contexts=3)
    monkeypatch.setattr(owner.audit, "summarize_branch", lambda _branch, _phase, _config: counts.copy())
    monkeypatch.setattr(owner.audit, "compare_branches", lambda _producer, _branch: {"state": 0.0})
    monkeypatch.setattr(owner, "audit_cache_events", lambda _branch, start_step: {"final_endpoint_reused": True})
    monkeypatch.setattr(owner, "compare_captures", lambda _producer, _branch: [107, 111, 119, 120, 128, 136, 144])
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
        assert result["cumulative_physical_bounds"]["control_steps"] == dict(lower=2202, upper=2317)
        assert result["cumulative_physical_bounds"]["native_steps"] == dict(lower=55050, upper=57925)
        assert result["cache_audit"]["final_endpoint_reused"]
    else:
        assert result["cumulative_physical_bounds"]["control_steps"] == dict(lower=2062, upper=2178)
        assert result["error"]["type"] == "ValueError"
    with pytest.raises(FileExistsError):
        owner.execute()
    assert len(calls) == 1
