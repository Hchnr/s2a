"""Build the bounded tracking fallback from completed causal controller history."""

from s2a.geometry import inverse
from s2a.research.native_runtime.development_continuation import current_joint_site_state, current_robot_dynamics


def build_refiner(env, case, receipt):
    """Expose only current controller state, completed history, and generated targets."""
    from s2a.research.joint_control.preview.completed_history import restore_recomputed_history
    from s2a.research.joint_control.preview.model_safe_tracking_fallback import (
        ModelSafeTrackingFallback,
        ScopedQualityFeedbackRefiner,
    )
    from s2a.research.joint_control.preview.refined_feedback import RefinedPreviewFeedback
    from s2a.research.response_models.causal.causal_inner_error import CausalInnerError

    _q, _limits, jacobian = current_joint_site_state(env)
    _mass, velocity, site = current_robot_dynamics(env, jacobian)
    adapter, history = ModelSafeTrackingFallback(receipt), CausalInnerError()
    refiner = ScopedQualityFeedbackRefiner(
        history,
        receipt,
        start_step=case["spec"]["prefix_steps"],
        start_cursor=case["intervention_spec"]["cursor"],
        solver=adapter.solve,
        auditor=adapter.audit,
        max_quality_response_evaluations=320,
    )
    evidence, completed = case["previous_evidence"], case["completed_history"]
    coupled, tracking, difference = restore_recomputed_history(
        completed,
        evidence["site"],
        evidence["twist"],
        evidence["coupled_prediction"],
        evidence["tracking_prediction"],
        site,
        jacobian @ velocity,
        refiner=refiner,
    )
    site_from_tcp = inverse(case["controller"]["source_from_site"]) @ case["tool"]
    policy = RefinedPreviewFeedback(
        env,
        coupled,
        tracking,
        case["targets"] @ inverse(site_from_tcp),
        case["anchor"] @ inverse(site_from_tcp),
        site_from_tcp,
        case["intervention_spec"]["cursor"],
        completed.previous_command,
        case["controller"]["contract"],
        receipt,
    )
    adapter.attach_history(policy.tracking.history)
    receipt["initial_innovation_recompute_max_difference"] = difference
    return policy, history
