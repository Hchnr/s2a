"""Exercise simulation-worker admission and recorded response binding on CPU."""

import copy
import json
import os
import subprocess
import sys
import threading

import pytest

from scripts.simulation.runtime import adaptive_live_bowl_admission as admission
from scripts.simulation.runtime import adaptive_live_bowl_worker as worker
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_bowl_next_reference as reference
from tests.research.native_runtime.test_adaptive_stream_reference import fake_service, run_driver, setup_adapter


def test_prefix_options_use_the_registered_historical_sources():
    inputs = admission.load_inputs()
    source, base = inputs["prior"]["base"]["source"], inputs["prior"]["base"]["source"]["previous"]
    producer, options = worker.prefix_options(inputs, "producer")
    assert producer.func is reference.run_producer
    assert producer.keywords["previous_88_producer"] is source["previous_88_producer"]
    assert options["baseline"] is base["base"]["baseline"]
    assert options["previous_56_producer"] is source["previous_producer"]
    assert options["load_second_reference"]() == (base["reference"], base["envelope"])
    checker, options = worker.prefix_options(inputs, "checker")
    assert checker.func is reference.run_checker
    assert checker.keywords["expected_step88_capture"] is source["capture88"]
    assert "baseline" not in options and "previous_56_producer" not in options


def saved_exchanges(tmp_path):
    case, adapter, channel, model_channel, _events, _owner = setup_adapter(tmp_path)
    errors = []
    thread = threading.Thread(target=fake_service, args=(case, model_channel, 3, errors))
    try:
        thread.start()
        stream = run_driver(case, adapter)
    finally:
        channel.close()
        thread.join(timeout=15)
    assert stream["complete"] and not thread.is_alive() and not errors
    rows = []
    for index, exchange in enumerate(case.receipt["adaptive_model_exchanges"]):
        response = prefix.read(exchange["response_ref"])
        raw = {
            "status": "success",
            "outputs": [
                {
                    "content": {
                        "action": response["action"],
                        "action128": {"normalized_action": response["normalized_action"]},
                    }
                }
            ],
        }
        path = tmp_path / f"raw{index}.json"
        path.write_text(json.dumps(raw))
        rows.append(dict(step=exchange["step"], reference_sent=True, actual_response_ref=prefix.reference(path)))
    producer = {"runtime": case.receipt}
    model = dict(
        responses=rows,
        responses_sent=3,
        model_requests_attempted=3,
        model_requests_completed=3,
        denoise_calls_attempted=105,
        denoise_calls_completed=105,
    )
    return producer, model


def test_replay_inputs_match_actual_saved_model_outputs(tmp_path):
    producer, model = saved_exchanges(tmp_path)
    responses = worker.load_recorded_responses(producer, model)
    assert [row["step"] for row in responses] == [80, 83, 87]


@pytest.mark.parametrize("change", ["count", "step", "send", "raw"])
def test_changed_model_evidence_is_rejected_before_replay(tmp_path, change):
    producer, model = saved_exchanges(tmp_path)
    model = copy.deepcopy(model)
    if change == "count":
        model["denoise_calls_completed"] -= 1
    elif change == "step":
        model["responses"][1]["step"] += 1
    elif change == "send":
        model["responses"][0]["reference_sent"] = False
    else:
        path = prefix.checked(model["responses"][0]["actual_response_ref"])
        value = json.loads(path.read_text())
        value["outputs"][0]["content"]["action"][0][0] = 0.2
        replacement = tmp_path / "changed_raw.json"
        replacement.write_text(json.dumps(value))
        model["responses"][0]["actual_response_ref"] = prefix.reference(replacement)
    with pytest.raises((ValueError, AssertionError)):
        worker.load_recorded_responses(producer, model)


def test_changed_phase_registration_is_published_without_simulator_import(tmp_path, monkeypatch):
    output = tmp_path / "trial"
    (output / "producer/simulation").mkdir(parents=True)
    (output / "producer/registration.json").write_text(
        json.dumps({"phase": "producer", "inputs": {"unexpected": True}, "source_refs": []})
    )
    monkeypatch.setattr(admission, "OUTPUT", output)
    monkeypatch.setattr(admission, "load_inputs", lambda: {})
    monkeypatch.setattr(admission, "input_receipt", lambda _inputs: {"expected": True})
    with pytest.raises(ValueError, match="exact registered"):
        worker.simulation_worker("producer", source_refs=lambda: [])
    result = json.loads((output / "producer/simulation/result.json").read_text())
    assert not result["complete"] and not result["passed"]
    assert result["error"]["type"] == "ValueError" and "runtime" not in result


def test_checker_worker_import_does_not_load_candidate_policy():
    code = """
import sys
from scripts.simulation.runtime.adaptive_live_bowl_worker import simulation_worker
for module in (
    's2a.research.joint_control.preview.causal_reference_replan',
    's2a.research.joint_control.preview.loaded_gripper_feedback',
    's2a.research.wam.tracking.runtime.adaptive_stream_reference',
):
    assert module not in sys.modules, module
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
    )
    assert result.returncode == 0, result.stdout + result.stderr
