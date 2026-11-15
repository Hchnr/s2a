"""Admit an oracle reference at reset without fabricating a model request."""

from types import FunctionType

import numpy as np

from s2a.geometry import validate_transform
from s2a.research.native_runtime import development_prefix_runtime_bounded as bounded
from s2a.research.native_runtime import development_reset_runtime as reset
from s2a.research.native_runtime.development_prefix_replay import ATOL, SAMPLE_FIELDS
from s2a.research.native_runtime.reset_bounded_runtime import MAX_PERIODS, runtime_unchanged
from s2a.research.wam.training.native.native_training import gripper_cache_receipt, state_sha256

_RUN = bounded.run_prefix_runtime


def verify_correct_reset(expected, expected_receipt, actual, actual_receipt, controller, tool, anchor, opening):
    """Verify the actual reset and known oracle anchor without querying a model."""
    tool, anchor = validate_transform(np.asarray(tool)), validate_transform(np.asarray(anchor))
    if tool.shape != (4, 4) or anchor.shape != (4, 4) or not np.isfinite(opening) or not 0 <= opening <= 1:
        raise ValueError("A finite fixed TCP, oracle anchor and measured opening are required")
    digests = []
    for sample, receipt in ((expected, expected_receipt), (actual, actual_receipt)):
        state = np.asarray(sample["sim_states"])
        digest = state_sha256(state)
        motor = gripper_cache_receipt(sample["gripper_current_action"])
        if (
            receipt["endpoint_step"] != 0
            or receipt["sim_state_sha256"] != digest
            or receipt["sim_state_dtype"] != "float64"
            or receipt["sim_state_shape"] != list(state.shape)
            or receipt["sim_time"] != float(sample["sim_time"])
            or any(receipt[key] != value for key, value in motor.items())
        ):
            raise ValueError("Each reset receipt must identify its own state, time and motor cache")
        digests.append(digest)
    if digests[0] != controller["actual_initial_physics_sha256"]:
        raise ValueError("The original controller must bind the recorded reset state")
    if not np.array_equal(expected["gripper_current_action"], controller["aligned_initial_motor_target"]):
        raise ValueError("The recorded initial motor target must match the controller contract")
    maximum = 0.0
    for field in (*SAMPLE_FIELDS, "gripper_current_action"):
        left, right = np.asarray(expected[field]), np.asarray(actual[field])
        if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError(f"Finite matching reset field required: {field}")
        difference = float(np.max(np.abs(left - right)))
        if difference > ATOL:
            raise ValueError(f"The current reset differs from the recorded field: {field}")
        maximum = max(maximum, difference)
    if abs(float(expected["opening"]) - opening) > ATOL:
        raise ValueError("The oracle reference opening must bind the recorded reset measurement")
    if not np.allclose(np.asarray(expected["source_pose"]) @ tool, anchor, atol=ATOL, rtol=0):
        raise ValueError("The oracle reference anchor must bind the recorded reset TCP")
    return dict(
        recorded_state_sha256=digests[0],
        actual_state_sha256=digests[1],
        state_bytes_equal=digests[0] == digests[1],
        max_field_difference=maximum,
        initial_time=float(actual["sim_time"]),
        measured_opening=float(actual["opening"]),
        motor_target=np.asarray(actual["gripper_current_action"]).copy(),
        reference_kind="recorded_correct_oracle",
        world_model_request=False,
    )


def observe_correct_reset(
    env,
    commands,
    saved,
    receipts,
    _output,
    observers,
    receipt,
    *,
    expected,
    controller,
    tool,
    anchor,
    opening,
    initialize=None,
    on_terminal=None,
    progress=None,
):
    """Observe only the reset endpoint; no recorded action is dispatched."""
    if not isinstance(commands, reset.NoPriorCommands) or len(commands) != 0 or len(receipts) != 1:
        raise ValueError("One reset endpoint and no historical command are required")
    if len(saved["sim_states"]) != 1 or not np.array_equal(saved["sim_states"][0], expected["sim_states"]):
        raise ValueError("Only the original reset state may be supplied")
    receipt.update(checked_endpoints=0, exact_state_endpoints=0, control_steps_attempted=0, control_steps_completed=0)
    observers.current(env)
    if initialize is not None:
        initialize(env)
    actual, actual_receipt = observers.endpoint(env, 0)
    checked = verify_correct_reset(expected, receipts[0], actual, actual_receipt, controller, tool, anchor, opening)
    receipt["reset_endpoint_comparison"] = checked
    receipt["checked_endpoints"] = 1
    receipt["exact_state_endpoints"] = int(checked["state_bytes_equal"])
    if on_terminal is not None:
        on_terminal(actual, actual_receipt)
    if progress is not None:
        progress(receipt)
    return receipt


class CorrectReferenceResetRuntime:
    """Use the pinned bounded simulator with a non-WAM reset observer."""

    def __init__(self, *, expected, expected_receipt, controller, tool, anchor, opening):
        if not runtime_unchanged():
            raise ValueError("The pinned bounded native runtime must remain unchanged")
        self.expected, self.expected_receipt = expected, expected_receipt
        self.controller, self.tool, self.anchor, self.opening = controller, tool, anchor, opening

        def observe(*args, **kwargs):
            return observe_correct_reset(
                *args, expected=expected, controller=controller, tool=tool, anchor=anchor, opening=opening, **kwargs
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
