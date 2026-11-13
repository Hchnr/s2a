"""Verify earlier observations remain intact in the actual replay driver."""

import json

import pytest

from scripts.simulation.runtime import adaptive_long_final_endpoint_worker as worker
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from tests.research.native_runtime.test_adaptive_long_cached_checker_worker import observer
from tests.research.native_runtime.test_adaptive_streaming_replay import cases, replay
from tests.research.native_runtime.test_endpoint_cache import environment


def mutate_only_the_final_repeat(checker, calls):
    original = checker.observers.endpoint

    def endpoint(env, step):
        sample, receipt = original(env, step)
        calls.append(step)
        if step == 92 and calls.count(step) == 2:
            env.sim.data.qpos[0] += 1e-6
        return sample, receipt

    checker.observers.endpoint = endpoint
    checker.observers.render_anchor = None


def test_final_only_cache_preserves_earlier_repeated_reads_and_all_actions(tmp_path):
    original_dir = tmp_path / "original"
    original_dir.mkdir()
    _producer, original, stream, responses, checker_dir = cases(original_dir)
    original_calls = []
    mutate_only_the_final_repeat(original, original_calls)
    with pytest.raises(ValueError, match="Replay endpoint observation changed"):
        replay(original, stream, responses, checker_dir)
    assert original.receipt["adaptive_replay_partial"]["control_steps_completed"] == 12

    cached_dir = tmp_path / "cached"
    cached_dir.mkdir()
    _producer, checker, stream, responses, checker_dir = cases(cached_dir)
    calls = []
    mutate_only_the_final_repeat(checker, calls)
    checker.observers = worker.final_endpoint_observers(checker.observers, checker.receipt, final_step=92)
    result = replay(checker, stream, responses, checker_dir)
    assert result["complete"] and result["control_steps_completed"] == 12
    assert calls.count(83) == original_calls.count(83) == 2
    assert calls.count(87) == original_calls.count(87) == 2
    assert original_calls.count(92) == 2 and calls.count(92) == 1
    events = checker.receipt["adaptive_cached_endpoint_events"]
    assert [(row["step"], row["kind"]) for row in events] == [(92, "observed"), (92, "reused")]
    assert checker.receipt["newly_selected_commands"] == 0


def test_final_cache_rejects_state_change_or_a_later_step():
    env, calls, receipt = environment(), [], {}
    wrapped = worker.final_endpoint_observers(observer(env, calls), receipt, final_step=147)
    wrapped.endpoint(env, 146)
    wrapped.endpoint(env, 146)
    wrapped.endpoint(env, 147)
    env.sim.data.qvel[0] += 1e-15
    with pytest.raises(ValueError, match="exact same current state"):
        wrapped.endpoint(env, 147)
    with pytest.raises(ValueError, match="past its final endpoint"):
        wrapped.endpoint(env, 148)
    assert calls == [146, 146, 147]


def test_worker_uses_only_registered_final_endpoint_boundary(tmp_path, monkeypatch):
    from s2a.research.native_runtime import reset_bounded_runtime
    from scripts.simulation.runtime import adaptive_live_bowl as continuation

    output = tmp_path / "trial"
    directory = output / "checker/simulation"
    directory.mkdir(parents=True)
    producer_path, model_path = tmp_path / "producer.json", tmp_path / "model.json"
    producer_path.write_text(json.dumps({"fixture": "producer"}))
    model_path.write_text(json.dumps({"fixture": "model"}))
    case = dict(
        expected={}, expected_receipt={}, chunk={}, controller={}, tool={}, config={}, record={}, record_xml="", case={}
    )
    old_runtime = dict(cv2_admission={"cv2_probe_ref": {}, "cv2_runtime_source_refs": []}, asset_refs=[])
    base = {"base": {"case": case, "base": {"previous": {"base": old_runtime}}}}
    long_inputs = dict(
        prior={"base": {"source": {"previous": base}}}, config={"prefix_steps": 107}, expected_capture={}
    )
    inputs = dict(
        config={"seed": 42, "cache_step": 147, "max_control_steps": 147},
        producer={"branch_ref": prefix.reference(producer_path), "model_ref": prefix.reference(model_path)},
        long_inputs=long_inputs,
    )
    refs = [{"path": "source", "sha256": "a" * 64}]
    (output / "registration.json").write_text(json.dumps({"inputs": {"fixture": True}, "source_refs": refs}))
    monkeypatch.setattr(worker.admission, "OUTPUT", output)
    monkeypatch.setattr(worker.admission, "load_inputs", lambda: inputs)
    monkeypatch.setattr(worker.admission, "input_receipt", lambda _inputs: {"fixture": True})
    monkeypatch.setattr(worker.admission, "replay_registration", lambda _inputs, _directory: {"session": "fixture"})
    monkeypatch.setattr(worker.historical, "load_recorded_responses", lambda _producer, _model: ["recorded"])
    monkeypatch.setattr(worker.historical, "prefix_options", lambda _inputs, _phase: ("prefix-runner", {}))
    monkeypatch.setattr(worker.isolation, "checker_isolation", lambda: None)
    monkeypatch.setattr(worker.original, "require_no_simulator_import", lambda: None)
    monkeypatch.setattr(worker, "restore_cv2_environment", lambda _probe, _sources: {"fixture": True})
    monkeypatch.setattr(worker.cached, "candidate_modules_unloaded", lambda: True)
    calls = []

    def checker(env, _terminal, observers, receipt, **kwargs):
        assert kwargs["responses"] == ["recorded"] and kwargs["prefix_runner"] == "prefix-runner"
        for step in (146, 146, 147, 147):
            observers.endpoint(env, step)
        assert [(row["step"], row["kind"]) for row in receipt["adaptive_cached_endpoint_events"]] == [
            (147, "observed"),
            (147, "reused"),
        ]
        return {"fixture": "replayed"}

    class FakeRuntime:
        def __init__(self, **_kwargs):
            pass

        def run_feedback(self, *_args, continuation, registered_seed, max_control_steps):
            assert registered_seed == 42 and max_control_steps == 147
            env, receipt = environment(), {}
            continuation(env, None, observer(env, calls), receipt)
            return {"complete": True, "passed": True}

    monkeypatch.setattr(continuation, "run_checker", checker)
    monkeypatch.setattr(reset_bounded_runtime, "ResetBoundedRuntime", FakeRuntime)
    result = worker.run_worker(source_refs=lambda: refs)
    assert result["complete"] and result["passed"] and result["candidate_modules_unloaded"]
    assert calls == [146, 146, 147]
