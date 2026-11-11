"""The guarded release prefix admits only its registered task and seed."""

import pytest

from s2a.research.native_runtime import development_prefix_runtime as original
from s2a.research.native_runtime.guarded_task1_release_prefix import (
    GuardedTask1ReleasePrefixRuntime,
    validate_task1_release,
)


def test_runtime_is_isolated_and_seed_and_task_are_narrow():
    run, validator = original.run_prefix_runtime, original.validate_registered_runtime
    validate_task1_release({"task_id": 9, "seed": 43}, 43)
    runtime = GuardedTask1ReleasePrefixRuntime()
    assert runtime._run.__code__ is run.__code__
    assert original.run_prefix_runtime is run and original.validate_registered_runtime is validator
    for config, seed in (({"task_id": 9, "seed": 42}, 42), ({"task_id": 8, "seed": 43}, 43)):
        with pytest.raises(ValueError, match="task-9 seed-43"):
            validate_task1_release(config, seed)
    with pytest.raises(ValueError, match="full-prefix recording"):
        runtime.run(registered_seed=43, capture_final_prefix_inner=False)
