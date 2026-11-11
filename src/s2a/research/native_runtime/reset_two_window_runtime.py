"""Admit two eight-period windows from an exact development reset."""

import numpy as np

from s2a.research.native_runtime import development_reset_runtime as reset
from s2a.research.native_runtime.reset_canvas_threads import TwoThreadCanvasResetRuntime


class ResetTwoWindowRuntime(TwoThreadCanvasResetRuntime):
    """Reuse the pinned reset observer with an explicit sixteen-period limit."""

    def run_feedback(
        self, config, record, record_xml, output, asset_refs, *, registered_seed, continuation, progress=None
    ):
        if not callable(continuation):
            raise ValueError("An explicit two-window continuation is required")
        if (
            reset.original.run_prefix_runtime is not reset._RUN
            or reset.original.replay_saved_prefix is not reset._REPLAY
        ):
            raise ValueError("The original native runtime changed after reset admission")
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
            continuation_budget=16,
            continuation_limit=16,
            progress=progress,
        )
        result["legacy_replay_flag_not_applicable"] = result.pop("behavioural_replay_verified")
        result["behavioural_replay_verified"] = False
        result["reset_observation_verified"] = True
        return result
