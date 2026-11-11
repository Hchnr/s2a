"""Guard the registered task-9/seed-43 release prefix without changing old runtimes."""

from types import FunctionType

from s2a.research.native_runtime import development_prefix_runtime as original
from s2a.research.native_runtime.guarded_correct_prefix import GuardedPrefixInner, guarded_replay

_RUN = original.run_prefix_runtime
_REPLAY = original.replay_saved_prefix
_FINAL = original.FinalPrefixInner
_VALIDATE = original.validate_registered_runtime


def validate_task1_release(config, registered_seed):
    if (
        type(registered_seed) is not int
        or registered_seed != 43
        or type(config.get("seed")) is not int
        or config["seed"] != 43
        or type(config.get("task_id")) is not int
        or config["task_id"] != 9
    ):
        raise ValueError("Only the registered task-9 seed-43 release case is admitted")
    _VALIDATE(config, registered_seed)


class GuardedTask1ReleasePrefixRuntime:
    """Replay one fixed 232-period prefix with per-period dispatch guards."""

    def __init__(self):
        if (
            original.run_prefix_runtime is not _RUN
            or original.validate_registered_runtime is not _VALIDATE
            or original.replay_saved_prefix is not _REPLAY
            or original.FinalPrefixInner is not _FINAL
        ):
            raise ValueError("The original runtime must remain unchanged")
        self.collector = None

        def make_collector(count):
            if count != 232 or self.collector is not None:
                raise ValueError("One registered 232-period prefix is required")
            self.collector = GuardedPrefixInner(count)
            return self.collector

        def replay(env, commands, saved, receipts, output, observers, receipt, **kwargs):
            if self.collector is None:
                raise ValueError("Full-prefix recording must be enabled")
            return guarded_replay(
                _REPLAY, self.collector, env, commands, saved, receipts, output, observers, receipt, **kwargs
            )

        namespace = _RUN.__globals__.copy()
        namespace.update(
            validate_registered_runtime=validate_task1_release,
            FinalPrefixInner=make_collector,
            replay_saved_prefix=replay,
        )
        self._run = FunctionType(_RUN.__code__, namespace, _RUN.__name__, _RUN.__defaults__, _RUN.__closure__)
        self._run.__kwdefaults__ = dict(_RUN.__kwdefaults__ or {})

    def run(self, *args, **kwargs):
        if (
            original.run_prefix_runtime is not _RUN
            or original.validate_registered_runtime is not _VALIDATE
            or original.replay_saved_prefix is not _REPLAY
            or original.FinalPrefixInner is not _FINAL
            or kwargs.get("capture_final_prefix_inner") is not True
            or kwargs.get("registered_seed") != 43
        ):
            raise ValueError("Unchanged runtime and explicit full-prefix recording are required")
        return self._run(*args, **kwargs)
