"""Run an actual development reset without replaying any historical command."""

import hashlib
from pathlib import Path
from types import FunctionType

import numpy as np

from s2a.research.native_runtime import development_prefix_runtime as original
from s2a.research.native_runtime.reset_endpoint import verify_reset_endpoint

_RUN = original.run_prefix_runtime
_REPLAY = original.replay_saved_prefix
_VALIDATE = original.validate_registered_runtime
_RUNTIME_SHA256 = "efac069505e92117766472fd474e0bbc19b929ae1a50a54b13ffcb3f7455886c"


class NoPriorCommands:
    """Empty prefix with an explicit missing-command value for the old receipt.

    The established runtime indexes the last saved command only to publish its
    receipt. Returning None for that one metadata lookup records that no command
    was executed; the injected reset observer never dispatches this object.
    """

    def __len__(self):
        return 0

    def __getitem__(self, index):
        if type(index) is int and index == -1:
            return None
        raise IndexError("There are no saved commands")


def validate_development_reset(config, registered_seed):
    """Require one of the fixed three task IDs and an explicitly bound seed."""
    if (
        type(registered_seed) is not int
        or registered_seed not in (42, 43)
        or type(config.get("seed")) is not int
        or config["seed"] != registered_seed
        or type(config.get("task_id")) is not int
        or config["task_id"] not in (6, 8, 9)
    ):
        raise ValueError("A registered development task and seed are required")


def observe_reset(
    env,
    commands,
    saved,
    receipts,
    output,
    observers,
    receipt,
    *,
    expected,
    chunk,
    controller,
    tool,
    initialize=None,
    on_terminal=None,
    progress=None,
):
    """Check the actual reset and live image before allowing any continuation."""
    if not isinstance(commands, NoPriorCommands) or len(commands) != 0 or len(receipts) != 1:
        raise ValueError("Exactly one reset endpoint and no previous commands are required")
    if len(saved["sim_states"]) != 1 or not np.array_equal(saved["sim_states"][0], expected["sim_states"]):
        raise ValueError("Only the original initial state may be supplied")
    receipt.update(
        checked_endpoints=0,
        exact_state_endpoints=0,
        control_steps_attempted=0,
        control_steps_completed=0,
        reset_image_match=False,
    )
    observers.current(env)
    if initialize is not None:
        initialize(env)
    actual, actual_receipt = observers.endpoint(env, 0)
    comparison = verify_reset_endpoint(expected, receipts[0], actual, actual_receipt, chunk, controller, tool)
    receipt["reset_endpoint_comparison"] = comparison
    receipt["checked_endpoints"] = 1
    receipt["exact_state_endpoints"] = int(comparison["state_bytes_equal"])
    capture = observers.render_anchor(env, 0, actual, actual_receipt, output)
    receipt["reset_render"] = capture
    receipt["reset_image_match"] = capture["canvas_rgb_sha256"] == chunk["image_sha256"]
    if not receipt["reset_image_match"]:
        raise ValueError("Live reset image differs from the original WAM request")
    if on_terminal is not None:
        on_terminal(actual, actual_receipt)
    if progress is not None:
        progress(receipt)
    return receipt


class DevelopmentResetRuntime:
    """Reuse the pinned native setup with a reset-only observation callback."""

    def __init__(self, *, expected, expected_receipt, chunk, controller, tool):
        if (
            original.run_prefix_runtime is not _RUN
            or original.replay_saved_prefix is not _REPLAY
            or original.validate_registered_runtime is not _VALIDATE
            or _RUN.__globals__.get("replay_saved_prefix") is not _REPLAY
            or _RUN.__globals__.get("validate_registered_runtime") is not _VALIDATE
            or hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest() != _RUNTIME_SHA256
        ):
            raise ValueError("The pinned historical runtime must remain unchanged")
        self.expected = expected
        self.expected_receipt = expected_receipt
        self.chunk = chunk
        self.controller = controller
        self.tool = tool

        def observe(*args, **kwargs):
            return observe_reset(*args, expected=expected, chunk=chunk, controller=controller, tool=tool, **kwargs)

        namespace = _RUN.__globals__.copy()
        namespace.update(validate_registered_runtime=validate_development_reset, replay_saved_prefix=observe)
        self._run = FunctionType(_RUN.__code__, namespace, _RUN.__name__, _RUN.__defaults__, _RUN.__closure__)
        self._run.__kwdefaults__ = dict(_RUN.__kwdefaults__ or {})

    def run(self, config, record, record_xml, output, asset_refs, *, registered_seed, progress=None):
        if original.run_prefix_runtime is not _RUN or original.replay_saved_prefix is not _REPLAY:
            raise ValueError("The historical runtime changed after reset adapter construction")
        saved = {"sim_states": np.asarray(self.expected["sim_states"])[None, :].copy()}
        result = self._run(
            config,
            record,
            record_xml,
            NoPriorCommands(),
            saved,
            [self.expected_receipt],
            output,
            asset_refs,
            self.controller,
            registered_seed=registered_seed,
            progress=progress,
        )
        # The original flag described replaying a positive prefix and is inapplicable here.
        result["legacy_replay_flag_not_applicable"] = result.pop("behavioural_replay_verified")
        result["behavioural_replay_verified"] = False
        result["reset_observation_verified"] = True
        return result
