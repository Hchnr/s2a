"""Record and guard every inner step of one registered correct-reference prefix."""

from types import FunctionType

import numpy as np

from s2a.research.native_runtime import development_prefix_runtime as original
from s2a.research.native_runtime.final_prefix_inner import FinalPrefixInner
from s2a.research.native_runtime.task2_seed42_correct_prefix import validate_task2_seed42

_RUN = original.run_prefix_runtime
_REPLAY = original.replay_saved_prefix
_FINAL = original.FinalPrefixInner
_VALIDATE = original.validate_registered_runtime
_MISSING = object()


class PrefixBufferStop(RuntimeError):
    """The next command was withheld after an observed inner buffer violation."""


class GuardedPrefixInner(FinalPrefixInner):
    """Capture all periods while retaining the original final-period receipt."""

    def __init__(self, prefix_steps):
        super().__init__(prefix_steps)
        self.periods = []

    def observe(self, env, completed_steps):
        if type(completed_steps) is not int or not 0 <= completed_steps < self.prefix_steps:
            raise ValueError("An in-range completed prefix count is required")
        if completed_steps == len(self.periods):
            self.periods.append({"prefix_step": completed_steps, "samples": []})
        elif completed_steps != len(self.periods) - 1:
            raise ValueError("Prefix periods must be observed consecutively")
        period = self.periods[-1]
        if len(period["samples"]) >= 25:
            raise ValueError("A prefix period cannot exceed 25 inner samples")
        robot, sim = env.robots[0], env.sim
        q = np.asarray(sim.data.qpos)[robot._ref_joint_pos_indexes].copy()
        qvel = np.asarray(sim.data.qvel)[robot._ref_joint_vel_indexes].copy()
        ids = [sim.model.joint_name2id(f"robot0_joint{i}") for i in range(1, 8)]
        limits = np.asarray(sim.model.jnt_range)[ids].copy()
        time = float(sim.data.time)
        if (
            q.shape != (7,)
            or qvel.shape != (7,)
            or limits.shape != (7, 2)
            or not np.isfinite(q).all()
            or not np.isfinite(qvel).all()
            or not np.isfinite(limits).all()
            or not np.isfinite(time)
            or np.any(limits[:, 0] >= limits[:, 1])
        ):
            raise ValueError("Finite inner robot state and fixed joint limits are required")
        if self.periods[0].get("limits") is not None and not np.array_equal(self.periods[0]["limits"], limits):
            raise ValueError("Prefix joint limits must remain fixed")
        period["limits"] = limits
        period["samples"].append({"time": time, "q": q, "qvel": qvel})
        super().observe(env, completed_steps)
        if len(period["samples"]) == 25:
            np.testing.assert_allclose(np.diff([row["time"] for row in period["samples"]]), 0.002, atol=1e-12, rtol=0)
            positions = np.asarray([row["q"] for row in period["samples"]])
            margins = np.minimum(positions - limits[:, 0], limits[:, 1] - positions)
            period.update(
                minimum_margin_rad=float(margins.min()),
                below_buffer_samples=int(np.sum(np.any(margins < 0.05, axis=1))),
                outside_samples=int(np.sum(np.any(margins < 0, axis=1))),
            )

    def before_dispatch(self, completed_steps):
        if type(completed_steps) is not int or not 0 <= completed_steps < self.prefix_steps:
            raise ValueError("An in-range prefix dispatch is required")
        if len(self.periods) != completed_steps:
            raise ValueError("Every earlier prefix period must have been observed")
        if completed_steps:
            previous = self.periods[-1]
            if len(previous["samples"]) != 25 or "below_buffer_samples" not in previous:
                raise ValueError("The previous prefix period must be complete")
            if previous["below_buffer_samples"]:
                raise PrefixBufferStop("Prior prefix period violated the 0.05 rad buffer")


def guarded_replay(replay, collector, env, commands, saved, receipts, output, observers, receipt, **kwargs):
    """Withhold the next saved action if the completed prefix lost its buffer."""
    if len(commands) != collector.prefix_steps:
        raise ValueError("Collector and registered prefix lengths must agree")
    receipt["prefix_inner_periods"] = collector.periods
    own_step = vars(env).get("step", _MISSING)
    step = env.step

    def guarded_step(*args, **step_kwargs):
        completed = receipt["control_steps_completed"]
        try:
            collector.before_dispatch(completed)
        except PrefixBufferStop:
            receipt["prefix_inner_stop"] = {
                "reason": "observed_joint_buffer_loss",
                "after_step": completed - 1,
                "at_step": completed,
                "next_action_dispatched": False,
            }
            raise
        return step(*args, **step_kwargs)

    env.step = guarded_step
    try:
        return replay(env, commands, saved, receipts, output, observers, receipt, **kwargs)
    finally:
        if own_step is _MISSING:
            del env.step
        else:
            env.step = own_step


class GuardedCorrectTask2Seed42PrefixRuntime:
    """Isolate full-prefix recording without changing the established runtime."""

    def __init__(self):
        if (
            original.run_prefix_runtime is not _RUN
            or original.validate_registered_runtime is not _VALIDATE
            or original.replay_saved_prefix is not _REPLAY
            or original.FinalPrefixInner is not _FINAL
        ):
            raise ValueError("The original guarded runtime must remain unchanged")
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
            original.run_prefix_runtime is not _RUN
            or original.validate_registered_runtime is not _VALIDATE
            or original.replay_saved_prefix is not _REPLAY
            or original.FinalPrefixInner is not _FINAL
            or kwargs.get("capture_final_prefix_inner") is not True
            or kwargs.get("registered_seed") != 42
        ):
            raise ValueError("Unchanged runtime and explicit full-prefix recording are required")
        return self._run(*args, **kwargs)
