"""Run a reset-origin continuation with an explicit budget of at most 600 periods."""

import hashlib
from pathlib import Path
from types import FunctionType

import numpy as np

from s2a.research.native_runtime import development_prefix_runtime_bounded as bounded
from s2a.research.native_runtime import development_reset_runtime as reset
from s2a.research.native_runtime.development_prefix_replay import ReplayObservers
from s2a.research.native_runtime.reset_canvas_threads import render_with_two_torch_threads

_RUN = bounded.run_prefix_runtime
_REPLAY = bounded.replay_saved_prefix
_VALIDATE = bounded.validate_registered_runtime
_RUNTIME_SHA256 = "a11988768d7bfa906d860c3daf76fb7f2d3bbe10ec11bf202283e3fb95857bdf"
MAX_PERIODS = 600


def runtime_unchanged():
    """Require the exact compatibility runtime and its original callbacks."""
    return (
        bounded.run_prefix_runtime is _RUN
        and bounded.replay_saved_prefix is _REPLAY
        and bounded.validate_registered_runtime is _VALIDATE
        and _RUN.__globals__.get("replay_saved_prefix") is _REPLAY
        and _RUN.__globals__.get("validate_registered_runtime") is _VALIDATE
        and hashlib.sha256(Path(bounded.__file__).read_bytes()).hexdigest() == _RUNTIME_SHA256
    )


class ResetBoundedRuntime:
    """Start from an observed reset; require an explicit physical period budget."""

    def __init__(self, *, expected, expected_receipt, chunk, controller, tool):
        if not runtime_unchanged():
            raise ValueError("The pinned bounded native runtime must remain unchanged")
        self.expected, self.expected_receipt = expected, expected_receipt
        self.chunk, self.controller, self.tool = chunk, controller, tool

        def observe(env, commands, saved, receipts, output, observers, receipt, **kwargs):
            def render(*args, **render_kwargs):
                return render_with_two_torch_threads(observers.render_anchor, receipt, *args, **render_kwargs)

            wrapped = ReplayObservers(observers.endpoint, observers.current, render, observers.render_cameras)
            return reset.observe_reset(
                env,
                commands,
                saved,
                receipts,
                output,
                wrapped,
                receipt,
                expected=expected,
                chunk=chunk,
                controller=controller,
                tool=tool,
                **kwargs,
            )

        namespace = _RUN.__globals__.copy()
        namespace.update(validate_registered_runtime=reset.validate_development_reset, replay_saved_prefix=observe)
        self._run = FunctionType(_RUN.__code__, namespace, _RUN.__name__, _RUN.__defaults__, _RUN.__closure__)
        self._run.__kwdefaults__ = dict(_RUN.__kwdefaults__ or {})

    def run_feedback(
        self,
        config,
        record,
        record_xml,
        output,
        asset_refs,
        *,
        registered_seed,
        continuation,
        max_control_steps,
        progress=None,
    ):
        if (
            not runtime_unchanged()
            or not callable(continuation)
            or type(max_control_steps) is not int
            or not 1 <= max_control_steps <= MAX_PERIODS
        ):
            raise ValueError("The pinned runtime, continuation and explicit 1–600 period budget are required")
        saved = {"sim_states": np.asarray(self.expected["sim_states"])[None, :].copy()}
        result = self._run(
            config,
            record,
            record_xml,
            reset.NoPriorCommands(),
            saved,
            [self.expected_receipt],
            output,
            asset_refs,
            self.controller,
            registered_seed=registered_seed,
            continuation=continuation,
            continuation_budget=max_control_steps,
            continuation_limit=max_control_steps,
            progress=progress,
        )
        result["legacy_replay_flag_not_applicable"] = result.pop("behavioural_replay_verified")
        result["behavioural_replay_verified"] = False
        result["reset_observation_verified"] = True
        return result
