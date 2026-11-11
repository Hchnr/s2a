"""Guard the registered 80-period prefix before a 48-period stream."""

from types import FunctionType

from s2a.research.native_runtime import development_prefix_runtime_48 as extended
from s2a.research.native_runtime.guarded_correct_prefix import GuardedPrefixInner, guarded_replay
from s2a.research.native_runtime.task2_seed42_correct_prefix import validate_task2_seed42

_RUN = extended.run_prefix_runtime
_REPLAY = extended.replay_saved_prefix
_FINAL = extended.FinalPrefixInner
_VALIDATE = extended.validate_registered_runtime


def _unchanged():
    return (
        extended.run_prefix_runtime is _RUN
        and extended.replay_saved_prefix is _REPLAY
        and extended.FinalPrefixInner is _FINAL
        and extended.validate_registered_runtime is _VALIDATE
    )


class GuardedStreamPrefixRuntime:
    """Reuse the original inner-step guard without mutating either runtime."""

    def __init__(self):
        if not _unchanged():
            raise ValueError("The extended runtime must remain unchanged")
        self.collector = None

        def make_collector(count):
            if count != 80 or self.collector is not None:
                raise ValueError("One registered 80-period prefix is required")
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
            validate_registered_runtime=validate_task2_seed42,
            FinalPrefixInner=make_collector,
            replay_saved_prefix=replay,
        )
        self._run = FunctionType(_RUN.__code__, namespace, _RUN.__name__, _RUN.__defaults__, _RUN.__closure__)
        self._run.__kwdefaults__ = dict(_RUN.__kwdefaults__ or {})

    def run(self, *args, **kwargs):
        if (
            not _unchanged()
            or kwargs.get("capture_final_prefix_inner") is not True
            or kwargs.get("registered_seed") != 42
            or kwargs.get("continuation_budget") != 48
            or kwargs.get("continuation_limit") != 48
        ):
            raise ValueError("Unchanged runtime, seed 42, full-prefix recording and a 48-period budget are required")
        return self._run(*args, **kwargs)
