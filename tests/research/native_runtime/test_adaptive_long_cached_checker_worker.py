"""Exercise cached endpoint reuse in the real adaptive replay driver."""

import json
from types import SimpleNamespace

import pytest

from s2a.research.wam.training.native.native_training import state_sha256
from scripts.simulation.runtime import adaptive_long_cached_checker_worker as worker
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from tests.research.native_runtime.test_adaptive_streaming_replay import cases, replay
from tests.research.native_runtime.test_endpoint_cache import environment


def observer(env, calls):
    def endpoint(actual, step):
        calls.append(step)
        state = actual.sim.get_state().flatten()
        return {"sim_states": state}, {
            "endpoint_step": step,
            "sim_time": actual.sim.data.time,
            "sim_state_sha256": state_sha256(state),
        }

    return SimpleNamespace(endpoint=endpoint, current=lambda _env: None, render_anchor=None, render_cameras=None)


def test_prefix_observations_remain_uncached_and_adaptive_repeats_are_reused():
    env, calls, receipt = environment(), [], {}
    wrapped = worker.cached_observers(observer(env, calls), receipt, start_step=107)
    wrapped.endpoint(env, 106)
    wrapped.endpoint(env, 106)
    wrapped.endpoint(env, 107)
    wrapped.endpoint(env, 107)
    assert calls == [106, 106, 107]
    assert [row["kind"] for row in receipt["adaptive_cached_endpoint_events"]] == ["observed", "reused"]


@pytest.mark.parametrize("change", ["position", "velocity", "time"])
def test_adaptive_cache_rejects_changed_current_state_without_another_read(change):
    env, calls, receipt = environment(), [], {}
    wrapped = worker.cached_observers(observer(env, calls), receipt, start_step=107)
    wrapped.endpoint(env, 107)
    if change == "position":
        env.sim.data.qpos[0] += 1e-15
    elif change == "velocity":
        env.sim.data.qvel[0] += 1e-15
    else:
        env.sim.data.time += 1e-15
    with pytest.raises(ValueError, match="exact same current state"):
        wrapped.endpoint(env, 107)
    assert calls == [107]


def mutating_duplicate(checker):
    original = checker.observers.endpoint
    seen = set()

    def endpoint(env, step):
        sample, receipt = original(env, step)
        if step in seen:
            env.sim.data.qpos[0] += 1e-6
        seen.add(step)
        return sample, receipt

    checker.observers.endpoint = endpoint
    checker.observers.render_anchor = None


def test_cache_prevents_duplicate_endpoint_mutation_during_independent_replay(tmp_path):
    uncached_dir = tmp_path / "uncached"
    uncached_dir.mkdir()
    _producer, uncached, stream, responses, checker_dir = cases(uncached_dir)
    mutating_duplicate(uncached)
    with pytest.raises(ValueError, match="Replay endpoint observation changed"):
        replay(uncached, stream, responses, checker_dir)
    assert uncached.receipt["adaptive_replay_partial"]["control_steps_completed"] == 3

    cached_dir = tmp_path / "cached"
    cached_dir.mkdir()
    _producer, checker, stream, responses, checker_dir = cases(cached_dir)
    mutating_duplicate(checker)
    checker.observers = worker.cached_observers(checker.observers, checker.receipt, start_step=80)
    result = replay(checker, stream, responses, checker_dir)
    assert result["complete"] and result["control_steps_completed"] == 12
    assert checker.receipt["newly_selected_commands"] == 0
    assert [row["step"] for row in checker.receipt["adaptive_cached_endpoint_events"] if row["kind"] == "reused"] == [
        83,
        87,
        92,
    ]


def test_worker_wires_registered_replay_to_adaptive_endpoint_cache(tmp_path, monkeypatch):
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
        config={"seed": 42, "max_control_steps": 147},
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
    monkeypatch.setattr(worker, "candidate_modules_unloaded", lambda: True)
    calls = []

    def checker(env, _terminal, observers, receipt, **kwargs):
        assert kwargs["responses"] == ["recorded"] and kwargs["prefix_runner"] == "prefix-runner"
        observers.endpoint(env, 107)
        observers.endpoint(env, 107)
        assert [row["kind"] for row in receipt["adaptive_cached_endpoint_events"]] == ["observed", "reused"]
        return {"fixture": "replayed"}

    class FakeRuntime:
        def __init__(self, **_kwargs):
            pass

        def run_feedback(self, *_args, continuation, registered_seed, max_control_steps):
            assert registered_seed == 42 and max_control_steps == 147
            env, receipt = environment(), {}
            continuation(env, None, observer(env, calls), receipt)
            return {"complete": True, "passed": True, "cache_events": receipt["adaptive_cached_endpoint_events"]}

    monkeypatch.setattr(continuation, "run_checker", checker)
    monkeypatch.setattr(reset_bounded_runtime, "ResetBoundedRuntime", FakeRuntime)
    result = worker.run_worker(source_refs=lambda: refs)
    assert result["complete"] and result["passed"] and result["candidate_modules_unloaded"]
    assert calls == [107]
    assert json.loads((directory / "result.json").read_text())["runtime"]["cache_events"][-1]["kind"] == "reused"
