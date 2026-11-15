"""Commit completed TCP prediction errors to a causal preview transaction."""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import finite
from s2a.research.joint_control.preview.restoration_horizon_feedback import RestorationHorizonSolverAudit
from s2a.research.response_models.causal.tcp_prediction_innovation import CausalTCPPredictionInnovation
from s2a.research.trajectory_search.causal_first_pose_constraint import CausalPoseConstraint
from s2a.research.trajectory_search.innovation_capped_quality_solver import validated_corrected_caps
from s2a.research.trajectory_search.staged_first_feasibility import solve_staged_first_feasibility


class CausalLookaheadSolverAudit(RestorationHorizonSolverAudit):
    """Use only completed feedback, and require independent audit before staging.

    The owner must call ``observe_completed`` only after the matching physical
    action and endpoint actually complete. History belongs to one trajectory;
    this class supplies no future-error or physical-safety guarantee.
    """

    def __init__(self, *, position_cap_m, rotation_cap_rad, solver=solve_staged_first_feasibility):
        self.caps = validated_corrected_caps(position_cap_m, rotation_cap_rad)
        self.history = CausalTCPPredictionInnovation()
        self._causal_solver = solver
        self._last_step = None
        self._awaiting_step = None
        self._staged_prediction = None
        self._translation = None
        self._rotation = None
        self._pose_index = None
        self._last_actual_safe = None
        self._active_constraint = None
        self.events = []
        super().__init__(solver=self._solve_causal)

    def observe_completed(self, step, predicted_error, actual_error, *, next_pose_index):
        """Update only from a matching completed prediction and measured error."""
        if type(next_pose_index) is not int or next_pose_index not in (0, 1):
            raise ValueError("The next corrected preview index must be zero or one")
        if self._pending is not None:
            raise ValueError("A proposed plan must be independently audited before completion")
        predicted, actual = finite(predicted_error, (6,)), finite(actual_error, (6,))
        if self._last_step is None:
            if self._awaiting_step is not None:
                raise ValueError("Initial history cannot have an in-flight decision")
            self.history.stage(step, predicted)
        else:
            if self._awaiting_step != step or step != self._last_step + 1:
                raise ValueError("Only the matching previously audited action may complete")
            np.testing.assert_allclose(predicted, self._staged_prediction, atol=1e-10, rtol=0)
        innovation = self.history.complete(step, actual)
        self._last_step, self._awaiting_step, self._staged_prediction = step, None, None
        self._translation = innovation["position_innovation_m"].copy()
        self._rotation = Rotation.from_rotvec(innovation["rotation_innovation_rad"]).as_matrix()
        self._pose_index = next_pose_index
        self._last_actual_safe = bool(np.linalg.norm(actual[:3]) <= 0.01 and np.linalg.norm(actual[3:]) <= 0.05)
        self.events.append(
            {
                "completed_step": step,
                "next_pose_index": next_pose_index,
                "actual_pose_caps_satisfied": self._last_actual_safe,
                "innovation": innovation,
            }
        )
        return innovation

    def _solve_causal(self, *args, **kwargs):
        if self._last_step is None or self._awaiting_step is not None or self._active_constraint is not None:
            raise ValueError("A fully observed causal history is required before the next solve")
        if not self._last_actual_safe:
            return {"status": "observed_pose_limit_violation", "response_evaluations": 0}
        constraint = CausalPoseConstraint(
            self._translation,
            self._rotation,
            pose_index=self._pose_index,
            position_cap_m=self.caps[0],
            rotation_cap_rad=self.caps[1],
        )
        solved = self._causal_solver(*args, first_pose_constraint=constraint, **kwargs)
        if solved.get("best_joint_safe") is not None:
            self._active_constraint = constraint
        return solved

    def audit(self, witness, *args, **kwargs):
        """Independently audit raw poses and the selected corrected pose."""
        constraint = self._active_constraint
        if constraint is None:
            raise ValueError("A causal candidate must be selected before audit")
        try:
            verified = super().audit(witness, *args, **kwargs)
            errors = finite(verified["tcp_errors"], (8, 6))
            index = constraint.pose_index
            if np.linalg.norm(errors[index, :3]) > 0.01 + 1e-10 or np.linalg.norm(errors[index, 3:]) > 0.05 + 1e-10:
                raise ValueError("The selected original preview pose exceeds its independently checked cap")
            corrected = np.r_[
                errors[index, :3] + self._translation,
                (Rotation.from_rotvec(errors[index, 3:]) * Rotation.from_matrix(self._rotation).inv()).as_rotvec(),
            ]
            np.testing.assert_allclose(corrected, witness["innovation_corrected_tcp_error"], atol=1e-10, rtol=0)
            if (
                np.linalg.norm(corrected[:3]) > self.caps[0] + 1e-10
                or np.linalg.norm(corrected[3:]) > self.caps[1] + 1e-10
            ):
                raise ValueError("The independently reconstructed causal pose exceeds its extra cap")
            next_step = self._last_step + 1
            self.history.stage(next_step, errors[0])
            self._awaiting_step, self._staged_prediction = next_step, errors[0].copy()
            verified.update(
                causal_corrected_pose_index=index,
                causal_corrected_tcp_error=corrected,
                hard_pose_prefix=index + 1,
                prefix_pose_caps_satisfied=True,
                prefix_pose_caps_enforced=True,
                future_innovation_guarantee=False,
            )
            self.events.append({"audited_step": next_step, "corrected_pose_index": index, "corrected_error": corrected})
            return verified
        finally:
            self._active_constraint = None
