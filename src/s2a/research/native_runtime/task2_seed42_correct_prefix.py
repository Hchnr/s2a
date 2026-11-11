"""Narrow task-6/seed-42 admission for a saved correct-reference prefix."""

from types import FunctionType

from s2a.research.native_runtime import development_prefix_runtime as original

_RUN = original.run_prefix_runtime
_VALIDATE = original.validate_registered_runtime


def validate_task2_seed42(config, registered_seed):
    if (
        type(registered_seed) is not int
        or registered_seed != 42
        or type(config.get("seed")) is not int
        or config["seed"] != 42
        or type(config.get("task_id")) is not int
        or config["task_id"] != 6
    ):
        raise ValueError("Only the registered task-6 seed-42 correct reference is admitted")


class CorrectTask2Seed42PrefixRuntime:
    """Reuse the unchanged guarded runtime with an isolated task validator."""

    def __init__(self):
        if (
            original.run_prefix_runtime is not _RUN
            or original.validate_registered_runtime is not _VALIDATE
            or _RUN.__globals__.get("validate_registered_runtime") is not _VALIDATE
            or _RUN.__code__.co_names.count("validate_registered_runtime") != 1
        ):
            raise ValueError("The original prefix runtime and validator must remain unchanged")
        namespace = _RUN.__globals__.copy()
        namespace["validate_registered_runtime"] = validate_task2_seed42
        self._run = FunctionType(_RUN.__code__, namespace, _RUN.__name__, _RUN.__defaults__, _RUN.__closure__)
        self._run.__kwdefaults__ = dict(_RUN.__kwdefaults__ or {})

    def run(self, *args, **kwargs):
        if original.run_prefix_runtime is not _RUN or original.validate_registered_runtime is not _VALIDATE:
            raise ValueError("The original prefix runtime cannot change while in use")
        return self._run(*args, **kwargs)
