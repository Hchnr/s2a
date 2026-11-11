"""Admit only the explicitly registered task-6/seed-42 replay."""

import pytest

from s2a.research.native_runtime import development_prefix_runtime as original
from s2a.research.native_runtime.task2_seed42_correct_prefix import (
    CorrectTask2Seed42PrefixRuntime,
    validate_task2_seed42,
)


def test_validator_is_narrow_and_does_not_change_original():
    before_run, before_validator = original.run_prefix_runtime, original.validate_registered_runtime
    validate_task2_seed42({"task_id": 6, "seed": 42}, 42)
    runtime = CorrectTask2Seed42PrefixRuntime()
    assert runtime._run.__code__ is before_run.__code__
    assert original.run_prefix_runtime is before_run and original.validate_registered_runtime is before_validator
    for config, seed in (
        ({"task_id": 6, "seed": 43}, 43),
        ({"task_id": 8, "seed": 42}, 42),
        ({"task_id": 6, "seed": True}, 42),
    ):
        with pytest.raises(ValueError, match="task-6 seed-42"):
            validate_task2_seed42(config, seed)
