"""Run an eight-period continuation only after an exact live reset and image."""

import numpy as np

from s2a.research.native_runtime import development_reset_runtime as reset
from s2a.research.native_runtime.reset_canvas_threads import TwoThreadCanvasResetRuntime


class ResetFeedbackRuntime(TwoThreadCanvasResetRuntime):
    """Bind the established reset admission to a bounded physical continuation."""

    def run_feedback(
        self, config, record, record_xml, output, asset_refs, *, registered_seed, continuation, progress=None
    ):
        if not callable(continuation):
            raise ValueError("An explicit eight-period continuation is required")
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
            continuation_budget=8,
            continuation_limit=8,
            progress=progress,
        )
        result["legacy_replay_flag_not_applicable"] = result.pop("behavioural_replay_verified")
        result["behavioural_replay_verified"] = False
        result["reset_observation_verified"] = True
        return result
