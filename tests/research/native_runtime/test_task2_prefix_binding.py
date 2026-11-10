"""The third-task binding preserves the frozen runtime and strict registration."""

import pytest

from s2a.research.native_runtime import development_prefix_runtime as original
from s2a.research.native_runtime.task2_prefix_binding import IsolatedTask2PrefixRuntime, validate_task2_runtime


def test_task2_binding_preserves_original_code_and_globals():
    old_validator = original.validate_registered_runtime
    runtime = IsolatedTask2PrefixRuntime()
    assert runtime._run.__code__ is original.run_prefix_runtime.__code__
    assert runtime._run.__globals__ is not original.run_prefix_runtime.__globals__
    assert runtime._run.__globals__["validate_registered_runtime"] is validate_task2_runtime
    assert original.validate_registered_runtime is old_validator
    original.validate_registered_runtime({"seed": 43, "task_id": 8}, 43)
    with pytest.raises(ValueError):
        original.validate_registered_runtime({"seed": 43, "task_id": 6}, 43)


def test_task2_requires_exact_task_and_seed():
    validate_task2_runtime({"seed": 43, "task_id": 6}, 43)
    for config, seed in (
        ({"seed": 42, "task_id": 6}, 42),
        ({"seed": 43, "task_id": 8}, 43),
        ({"seed": 43, "task_id": 9}, 43),
        ({"seed": 43.0, "task_id": 6}, 43),
        ({"seed": 43, "task_id": 6.0}, 43),
        ({"seed": 43, "task_id": 6}, True),
    ):
        with pytest.raises(ValueError):
            validate_task2_runtime(config, seed)


def test_binding_rejects_changed_original_validator(monkeypatch):
    monkeypatch.setattr(original, "validate_registered_runtime", lambda *_: None)
    with pytest.raises(ValueError, match="unchanged"):
        IsolatedTask2PrefixRuntime()
