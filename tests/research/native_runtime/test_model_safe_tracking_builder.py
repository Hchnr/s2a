"""Wire the scoped fallback to exactly the completed controller history."""

from types import SimpleNamespace

import numpy as np

from scripts.simulation.runtime import build_model_safe_tracking_feedback as builder


def test_builder_attaches_completed_history_and_preserves_reference_contract(monkeypatch):
    from s2a.research.joint_control.preview import completed_history, model_safe_tracking_fallback, refined_feedback
    from s2a.research.response_models.causal import causal_inner_error

    captured = {}
    completed = SimpleNamespace(previous_command=np.arange(7, dtype=float))
    tracking = SimpleNamespace(history=object())

    class Adapter:
        def __init__(self, receipt):
            captured["receipt"] = receipt

        def attach_history(self, history):
            captured["attached_history"] = history

        def solve(self, *_args, **_kwargs):
            pass

        def audit(self, *_args, **_kwargs):
            pass

    def make_refiner(history, receipt, **kwargs):
        captured.update(inner=history, refiner_options=kwargs)
        return object()

    def restore(*args, **kwargs):
        captured["completed"] = args[0]
        captured["restore_refiner"] = kwargs["refiner"]
        return object(), tracking, 0.0

    def make_policy(*args):
        captured["policy_args"] = args
        return SimpleNamespace(tracking=args[2])

    monkeypatch.setattr(builder, "current_joint_site_state", lambda _env: (np.zeros(7), None, np.zeros((6, 7))))
    monkeypatch.setattr(builder, "current_robot_dynamics", lambda _env, _jac: (None, np.zeros(7), np.eye(4)))
    monkeypatch.setattr(model_safe_tracking_fallback, "ModelSafeTrackingFallback", Adapter)
    monkeypatch.setattr(model_safe_tracking_fallback, "ScopedQualityFeedbackRefiner", make_refiner)
    monkeypatch.setattr(completed_history, "restore_recomputed_history", restore)
    monkeypatch.setattr(refined_feedback, "RefinedPreviewFeedback", make_policy)
    monkeypatch.setattr(causal_inner_error, "CausalInnerError", lambda: object())
    targets = np.tile(np.eye(4), (40, 1, 1))
    case = {
        "spec": {"prefix_steps": 16},
        "intervention_spec": {"cursor": 0},
        "previous_evidence": dict(
            site=np.eye(4), twist=np.zeros(6), coupled_prediction=np.zeros(12), tracking_prediction=np.zeros(12)
        ),
        "completed_history": completed,
        "controller": {"source_from_site": np.eye(4), "contract": {"fixed": True}},
        "tool": np.eye(4),
        "targets": targets,
        "anchor": np.eye(4),
    }
    receipt = {}
    policy, history = builder.build_refiner(None, case, receipt)
    assert policy.tracking is tracking and history is captured["inner"]
    assert captured["completed"] is completed and captured["attached_history"] is tracking.history
    assert captured["refiner_options"]["max_quality_response_evaluations"] == 320
    assert captured["refiner_options"]["start_step"] == 16 and captured["refiner_options"]["start_cursor"] == 0
    np.testing.assert_array_equal(captured["policy_args"][3], targets)
    np.testing.assert_array_equal(captured["policy_args"][7], completed.previous_command)
    assert receipt["initial_innovation_recompute_max_difference"] == 0
