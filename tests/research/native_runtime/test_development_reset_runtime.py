"""A reset admission verifies the actual endpoint and image before dispatch."""

import copy
from pathlib import Path

import numpy as np
import pytest

from s2a.research.native_runtime.development_prefix_replay import ReplayObservers
from s2a.research.native_runtime.development_reset_runtime import (
    DevelopmentResetRuntime,
    NoPriorCommands,
    observe_reset,
    validate_development_reset,
)
from tests.research.native_runtime.test_reset_endpoint import sample


def observers_for(actual, receipt, events, image):
    return ReplayObservers(
        current=lambda _env: events.append("current"),
        endpoint=lambda _env, index: (actual, receipt) if index == 0 else pytest.fail("future endpoint read"),
        render_anchor=lambda _env, index, *_args: {"canvas_rgb_sha256": image, "frame_index": index},
        render_cameras=lambda *_args: pytest.fail("unexpected second rendering"),
    )


def test_zero_prefix_is_not_a_saved_action_and_admission_is_explicit():
    empty = NoPriorCommands()
    assert len(empty) == 0 and empty[-1] is None
    with pytest.raises(IndexError):
        _ = empty[0]
    for task in (6, 8, 9):
        for seed in (42, 43):
            validate_development_reset({"task_id": task, "seed": seed}, seed)
    for config, seed in (
        ({"task_id": 6, "seed": 42}, 43),
        ({"task_id": 7, "seed": 42}, 42),
        ({"task_id": 6, "seed": 42}, True),
    ):
        with pytest.raises(ValueError, match="registered"):
            validate_development_reset(config, seed)


def test_current_reset_and_matching_image_are_required_before_terminal_callback():
    expected, expected_receipt, chunk, controller = sample()
    actual, events, captured = copy.deepcopy(expected), [], []
    observers = observers_for(actual, expected_receipt, events, chunk["image_sha256"])
    outcome = observe_reset(
        object(),
        NoPriorCommands(),
        {"sim_states": [expected["sim_states"]]},
        [expected_receipt],
        Path("."),
        observers,
        {},
        expected=expected,
        chunk=chunk,
        controller=controller,
        tool=np.eye(4),
        initialize=lambda _env: events.append("initialize"),
        on_terminal=lambda _sample, _receipt: captured.append("terminal"),
    )
    assert events == ["current", "initialize"] and captured == ["terminal"]
    assert outcome["checked_endpoints"] == outcome["exact_state_endpoints"] == 1
    assert outcome["control_steps_attempted"] == outcome["control_steps_completed"] == 0
    assert outcome["reset_image_match"] is True


def test_image_mismatch_withholds_terminal_and_keeps_diagnostic_receipt():
    expected, expected_receipt, chunk, controller = sample()
    receipt, captured = {}, []
    with pytest.raises(ValueError, match="image differs"):
        observe_reset(
            object(),
            NoPriorCommands(),
            {"sim_states": [expected["sim_states"]]},
            [expected_receipt],
            Path("."),
            observers_for(expected, expected_receipt, [], "different"),
            receipt,
            expected=expected,
            chunk=chunk,
            controller=controller,
            tool=np.eye(4),
            on_terminal=lambda *_args: captured.append("terminal"),
        )
    assert not captured and receipt["reset_image_match"] is False
    assert receipt["reset_render"]["canvas_rgb_sha256"] == "different"
    assert receipt["control_steps_attempted"] == receipt["control_steps_completed"] == 0


def test_reset_adapter_passes_zero_commands_and_never_claims_historical_replay():
    expected, expected_receipt, chunk, controller = sample()
    adapter = DevelopmentResetRuntime(
        expected=expected, expected_receipt=expected_receipt, chunk=chunk, controller=controller, tool=np.eye(4)
    )

    def fake_run(*args, **kwargs):
        assert len(args[3]) == 0 and args[3][-1] is None
        assert kwargs["registered_seed"] == 42 and "continuation" not in kwargs
        return {"behavioural_replay_verified": True, "previous_saved_command": None}

    adapter._run = fake_run
    result = adapter.run({"task_id": 6, "seed": 42}, {}, "", Path("."), {}, registered_seed=42)
    assert result["reset_observation_verified"] is True
    assert result["behavioural_replay_verified"] is False and result["previous_saved_command"] is None
