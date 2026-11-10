"""Private task-6 admission for the unchanged saved-prefix runtime."""

from types import FunctionType

from s2a.research.native_runtime import development_prefix_runtime as original

_ORIGINAL_RUN = original.run_prefix_runtime
_ORIGINAL_VALIDATE = original.validate_registered_runtime


def validate_task2_runtime(config, registered_seed):
    """Admit only the explicitly registered task-6, seed-43 replay."""
    if (
        type(registered_seed) is not int
        or registered_seed != 43
        or type(config.get("seed")) is not int
        or config["seed"] != registered_seed
        or type(config.get("task_id")) is not int
        or config["task_id"] != 6
    ):
        raise ValueError("Original task 6 and seed 43 require explicit registration")


class IsolatedTask2PrefixRuntime:
    """Reuse original runtime code with an isolated, narrower task validator.

    This does not alter the original module or its task-8/9 admission. All
    remaining runtime, source, asset and physical checks remain in that code.
    """

    def __init__(self):
        if (
            original.run_prefix_runtime is not _ORIGINAL_RUN
            or original.validate_registered_runtime is not _ORIGINAL_VALIDATE
            or _ORIGINAL_RUN.__globals__.get("validate_registered_runtime") is not _ORIGINAL_VALIDATE
            or _ORIGINAL_RUN.__code__.co_names.count("validate_registered_runtime") != 1
        ):
            raise ValueError("The original runtime and validator must remain unchanged")
        namespace = _ORIGINAL_RUN.__globals__.copy()
        namespace["validate_registered_runtime"] = validate_task2_runtime
        self._run = FunctionType(
            _ORIGINAL_RUN.__code__,
            namespace,
            _ORIGINAL_RUN.__name__,
            _ORIGINAL_RUN.__defaults__,
            _ORIGINAL_RUN.__closure__,
        )
        self._run.__kwdefaults__ = dict(_ORIGINAL_RUN.__kwdefaults__ or {})

    def run(self, *args, **kwargs):
        if (
            original.run_prefix_runtime is not _ORIGINAL_RUN
            or original.validate_registered_runtime is not _ORIGINAL_VALIDATE
        ):
            raise ValueError("The original runtime cannot change while the isolated runtime is in use")
        return self._run(*args, **kwargs)
