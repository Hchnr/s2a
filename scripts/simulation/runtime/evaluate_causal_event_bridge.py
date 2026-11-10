"""Score actual completed endpoints against both inputs and their bridge."""

import numpy as np

from s2a.geometry import inverse, pose_error
from s2a.research.native_runtime.development_continuation import current_joint_site_state, current_robot_dynamics
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_event_feedback as base


def compare_scores(actual, expected, *, names):
    """Compare reference and measured values without accessing a policy."""
    np.testing.assert_allclose(actual["actual_tcp"], expected["actual_tcp"], atol=1e-10, rtol=0)
    for name in names:
        for key in ("reference_errors", "opening_targets", "opening_errors"):
            np.testing.assert_allclose(actual[key][name], expected[key][name], atol=1e-10, rtol=0)
    for key in ("opening", "qpos", "qvel", "motor_target"):
        np.testing.assert_allclose(actual["gripper"][key], expected["gripper"][key], atol=1e-10, rtol=0)
    prefix.require(
        actual["selected_absolute_endpoint"] == expected["selected_absolute_endpoint"], "Actual absolute endpoint"
    )


def score_completed(env, case, reference, offset, own_error, *, expected=None):
    """Measure an already completed action; no object label is a control input."""
    prefix.require(type(offset) is int and 0 <= offset < 16, "Registered completed continuation offset")
    _q, _limits, jacobian = current_joint_site_state(env)
    _mass, _velocity, site = current_robot_dynamics(env, jacobian)
    site_from_tcp = inverse(np.asarray(case["controller"]["source_from_site"])) @ case["tool"]
    actual_tcp = site @ site_from_tcp
    errors, opening_targets = {}, {}
    for name in ("latest", "remembered"):
        original = case["references"][name]
        index = original["cursor"] + offset
        errors[name] = pose_error(original["poses"][index], actual_tcp)
        opening_targets[name] = float(original["openings"][index])
    errors["bridge"] = pose_error(reference["poses"][offset], actual_tcp)
    opening_targets["bridge"] = float(reference["openings"][offset])
    np.testing.assert_allclose(errors["bridge"], own_error, atol=1e-10, rtol=0)
    gripper = base.measured_gripper(env)
    result = {
        "actual_tcp": actual_tcp,
        "reference_errors": errors,
        "opening_targets": opening_targets,
        "opening_errors": {name: gripper["opening"] - value for name, value in opening_targets.items()},
        "gripper": gripper,
        "selected_absolute_endpoint": case["spec"]["prefix_steps"] + offset + 1,
    }
    if expected is not None:
        compare_scores(result, expected, names=("latest", "remembered", "bridge"))
    return result
