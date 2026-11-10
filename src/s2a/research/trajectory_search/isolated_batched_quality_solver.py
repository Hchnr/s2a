"""Diagnostic quality solve with private rollout binding and frozen source bytes."""

from types import CodeType, FunctionType

import numpy as np

from s2a.research.joint_control.preview.batched_free_joint_rollout import BatchedFreeJointRollout
from s2a.research.joint_control.preview.interior_joint_preview import rollout_joint_preview
from s2a.research.trajectory_search.quality_capped_inner_joint_plan import solve_interior_plan


def _global_reference_count(code, name):
    """Count references in the solver and its nested callback code objects."""
    return int(name in code.co_names) + sum(
        _global_reference_count(item, name) for item in code.co_consts if isinstance(item, CodeType)
    )


class IsolatedBatchedQualitySolver:
    """Reuse the frozen solver code with a private, nonprojected rollout.

    The original module globals are never changed. This experimental binding
    is for isolated comparison only; every witness still needs the original
    independent audit, and no production controller selects it by default.
    """

    _MODEL_KEYS = ("q", "jacobian", "mass", "transition", "input", "affine", "anchor_response", "dt")

    def __init__(self, model):
        if (
            _global_reference_count(solve_interior_plan.__code__, "rollout_joint_preview") != 1
            or solve_interior_plan.__globals__.get("rollout_joint_preview") is not rollout_joint_preview
        ):
            raise ValueError("The original solver must bind the canonical sequential rollout")
        self._model = model
        self._model_values = {key: np.asarray(model[key]).copy() for key in self._MODEL_KEYS}
        self._candidate = BatchedFreeJointRollout(model)
        namespace = solve_interior_plan.__globals__.copy()
        namespace["rollout_joint_preview"] = self._rollout
        self._solver = FunctionType(
            solve_interior_plan.__code__,
            namespace,
            solve_interior_plan.__name__,
            solve_interior_plan.__defaults__,
            solve_interior_plan.__closure__,
        )
        self._solver.__kwdefaults__ = dict(solve_interior_plan.__kwdefaults__ or {})

    def _check_model(self, model):
        if model is not self._model or any(
            key not in model or not np.array_equal(np.asarray(model[key]), value)
            for key, value in self._model_values.items()
        ):
            raise ValueError("The prepared rollout requires the same unchanged frozen model")

    def _rollout(self, model, velocity, deltas, site, lower, upper, *, enforce_limits):
        if model is not self._model:
            raise ValueError("The rollout requires the bound model")
        return self._candidate.rollout(velocity, deltas, site, lower, upper, enforce_limits=enforce_limits)

    def solve(self, model, *args, **kwargs):
        self._check_model(model)
        result = self._solver(model, *args, **kwargs)
        self._check_model(model)
        return result
