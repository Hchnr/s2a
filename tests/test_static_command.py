"""Condition isolation, full128 admission, and actual prepare/predict handoff."""

import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_wam_static_command"), importlib.import_module("serve_wam_static_command")


def test_one_static_command_per_task_excludes_phase_plans(modules):
    runner, _ = modules
    cases = []
    for task in ("bowl", "wine", "cream"):
        for phase, command in (("approach", "first command"), ("release", "another command")):
            cases.append(
                dict(
                    task=task,
                    phase=phase,
                    published_prompts={
                        "40": json.dumps(
                            dict(
                                task_title="published title",
                                command=command,
                                task_plan=[phase],
                                current_subtask=phase,
                            )
                        )
                    },
                )
            )
    actual = {task: json.loads(value) for task, value in runner.static_prompts(cases).items()}
    assert actual == {task: dict(task_title=task, command="first command") for task in ("bowl", "wine", "cream")}
    for case in cases:
        case["phase"] = "arbitrary label"
    assert actual == {task: json.loads(value) for task, value in runner.static_prompts(cases).items()}


@pytest.mark.parametrize("normalized", [False, True])
def test_reproduction_checks_every_slot_of_raw_and_normalized(modules, normalized):
    runner, _ = modules
    raw = np.zeros((40, 128)).tolist()
    response = dict(outputs=[dict(content=dict(action=raw, action128=dict(normalized_action=copy.deepcopy(raw))))])
    runner.same_actions(response, copy.deepcopy(response))
    different = copy.deepcopy(response)
    values = different["outputs"][0]["content"]
    values = values["action128"]["normalized_action"] if normalized else values["action"]
    values[39][127] = 1e-15
    with pytest.raises(ValueError, match="reproduction differs"):
        runner.same_actions(different, response)


def test_actual_handoff_and_complete_reproduction_gate(modules, tmp_path):
    _, server = modules
    image = tmp_path / "source.png"
    image.write_bytes(b"unit input fingerprint only")
    directory = tmp_path / "ledger"
    directory.mkdir()
    inputs = [dict(name=f"request{i}", seed=42, vision_path=str(image)) for i in range(49)]
    protocol = dict(
        requests=[dict(request=item, mode="reproduction" if i < 48 else "expanded") for i, item in enumerate(inputs)]
    )

    class Service:
        def __init__(self):
            self.calls = 0

        def prepare(self, request, output_dir):
            return SimpleNamespace(seed=42, action_dim_mask=[True] * 10 + [False] * 118), dict(
                ai_caption=[json.dumps(dict(command="actual", task_plan=[]))]
            )

        def predict(self, prepared, batch):
            self.calls += 1
            return "predicted"

    service = Service()
    server.install_ledger(service, protocol, directory)

    def request(index):
        return SimpleNamespace(name=f"request{index}", model_dump=lambda **kw: inputs[index])

    for index in range(48):
        prepared, batch = service.prepare(request(index), directory)
        assert service.predict(prepared, batch) == "predicted"
    assert service.calls == 48
    with pytest.raises(FileNotFoundError):
        service.prepare(request(48), directory)
    assert service.calls == 48
    (tmp_path / "reproduction_admission.json").write_text(json.dumps(dict(passed=True, verified_queries=48)))
    prepared, batch = service.prepare(request(48), directory)
    batch["ai_caption"] = ["altered after prepare"]
    with pytest.raises(ValueError, match="handoff changed"):
        service.predict(prepared, batch)
    assert service.calls == 48
    assert not (directory / "request48.predict.json").exists()
    assert len(list(directory.glob("*.predict.json"))) == 48


def test_preparation_cannot_reset_an_inference_attempt(modules, tmp_path, monkeypatch):
    runner, _ = modules
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    configurations = {"reproduction": {}, "expanded": {}}
    samples = {mode: {"cases": [mode]} for mode in configurations}
    for mode in configurations:
        folder = tmp_path / mode / "samples"
        folder.mkdir(parents=True)
        (folder / "manifest.json").write_text(json.dumps(samples[mode]))
    assert runner.preparation_only_root(samples, configurations) == []
    (tmp_path / "orchestration").mkdir()
    with pytest.raises(AssertionError):
        runner.preparation_only_root(samples, configurations)


def test_empty_placeholder_preserves_all_other_caption_fields(modules):
    _, server = modules
    original = dict(task_title="goal", command="long command", fps=20.0)
    batch = dict(ai_caption=[json.dumps(original)])
    assert server.align_empty_plan(batch, expanded=True)
    assert json.loads(batch["ai_caption"][0]) == dict(original, task_plan=[])
    for extra in (dict(task_plan=["future demonstration"]), dict(current_subtask="grasp now")):
        batch = dict(ai_caption=[json.dumps(dict(original, **extra))])
        with pytest.raises(ValueError, match="forbids demonstration"):
            server.align_empty_plan(batch, expanded=True)
