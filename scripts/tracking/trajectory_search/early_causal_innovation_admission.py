"""Select the completed step121 state and causal history for step122."""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import finite
from s2a.research.wam.training.native.native_training import state_sha256
from scripts.simulation.runtime import prepare_quality_rate_repair_recorded_feedback as physical
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import cached_live_stop125_admission as earlier

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/joint_control/early_causal_innovation_quality_20261111.json"
PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/early_causal_innovation_quality_20261111.md"
OUTPUT = ROOT / "outputs/early_causal_innovation_quality_20261111"
QA = ROOT / "outputs/qa/early_causal_innovation_quality_20261111/result.json"
DRY = ROOT / "outputs/early_causal_innovation_quality_dry_20261111/result.json"
PRODUCER = {
    "path": str(ROOT / "outputs/quality_rate_repair_recorded_feedback_20261111/producer/simulation/result.json"),
    "sha256": "e936271d98ba46fc975c2eea3e2808930f6d7d895c0c4e18db254e08b1be154d",
}
CHECKER = {
    "path": str(ROOT / "outputs/quality_rate_repair_recorded_feedback_20261111/checker/simulation/result.json"),
    "sha256": "de27191cf2001e00b4b2c3cd2abd6e10488ebeb02f950f4435d535f5db3ea5f0",
}
FIXED = {
    "schema": "s2a.early-causal-innovation-quality.v1",
    "output": "outputs/early_causal_innovation_quality_20261111",
    "innovation_audit": "outputs/causal_tcp_prediction_audit_20261111/result.json",
    "innovation_audit_sha256": "0a70c6505f69ba9e666b13ccb000e8eac2f418b995c515d4db9fcec5efd02c2c",
    "physical_audit": "outputs/quality_rate_repair_recorded_feedback_audit_20261111/result.json",
    "physical_audit_sha256": "bd33ed32be8e4bb4a9ef0c3fb753a8d0883e7cb6bd3ef22fedd9729807b4c802",
    "case_index": 8,
    "generated_step": 120,
    "decision_step": 122,
    "last_completed_step": 121,
    "reference_cursor": 2,
    "branches": ["raw_quality", "innovation_quality"],
    "max_models": 1,
    "max_forward_queries": 1,
    "max_optimizer_calls": 2,
    "max_model_responses": 320,
    "max_independent_endpoint_queries": 400,
    "max_independent_transition_steps": 5200,
    "physics_steps": 0,
    "world_model_calls": 0,
    "fresh_cases": 0,
    "worker_timeout_seconds": 900,
    "automatic_retry": False,
}


def select_completed(producer, checker, innovation_audit):
    """Return current state and history without accessing step122 outcomes."""
    left = [row for row in physical.periods(producer) if row["step"] == 121]
    right = [row for row in physical.periods(checker) if row["step"] == 121]
    innovations = [row for row in innovation_audit["periods"] if row["step"] == 121]
    prefix.require(len(left) == len(right) == len(innovations) == 1, "One completed step121 and replay are required")
    actual, checked, innovation = left[0], right[0], innovations[0]
    physical.matched_prior_period(actual, checked)
    state, previous = finite(actual["actual_state"], (79,)).copy(), finite(actual["command"], (7,)).copy()
    prefix.require(state_sha256(state) == actual["endpoint_receipt"]["sim_state_sha256"], "Current endpoint identity")
    plan = finite(actual["decision"]["refined_plan"], (8, 6)).copy()
    np.testing.assert_array_equal(plan[0], previous[:6])
    seed = np.vstack((plan[1:], plan[-1]))
    prefix.require(
        np.max(np.abs(seed)) <= 1 + 1e-12
        and np.max(np.abs(np.diff(np.vstack((previous[:6], seed)), axis=0))) <= 0.25 + 1e-12,
        "The shifted completed plan must preserve box and rate",
    )
    history = actual["completed_feedback"]["past_inner_error"]
    profile = finite(history["inner_position_error_rad"], (25, 7)).copy()
    prefix.require(
        history["available"] is True
        and history["completed_periods"] == 2
        and history["total_completed_periods"] == 42
        and history["last_completed_step"] == 121
        and history["future_guarantee"] is False
        and np.all(profile >= 0)
        and np.all(np.diff(profile, axis=0) >= 0),
        "Only two completed current-segment periods may update the causal history",
    )
    predicted = finite(actual["decision"]["quality"]["first_tcp_error"], (6,))
    measured = finite(actual["tcp_error"], (6,))
    np.testing.assert_array_equal(predicted, innovation["raw"]["predicted_error"])
    np.testing.assert_array_equal(measured, innovation["actual_error"])
    prefix.require(
        innovation["completed_innovation"]["step"] == 121
        and innovation["completed_innovation"]["completed_history"] == 42,
        "The TCP innovation must be completed before step122",
    )
    translation = measured[:3] - predicted[:3]
    rotation = Rotation.from_rotvec(measured[3:]).inv() * Rotation.from_rotvec(predicted[3:])
    np.testing.assert_allclose(
        translation, innovation["completed_innovation"]["position_innovation_m"], atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        rotation.as_rotvec(), innovation["completed_innovation"]["rotation_innovation_rad"], atol=1e-12, rtol=0
    )
    return {
        "state": state,
        "previous": previous,
        "shifted_seed": seed,
        "past_inner_error_rad": profile,
        "translation_innovation": translation,
        "rotation_discrepancy": rotation.as_matrix(),
    }


def load_inputs():
    inputs = earlier.load_inputs()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    prefix.require(
        type(config) is dict
        and set(config) == set(FIXED)
        and all(type(config[key]) is type(value) and config[key] == value for key, value in FIXED.items()),
        "The exact early innovation-quality budget is required",
    )
    consumed = dict(inputs["source_sha256"])
    refs = {
        key: {"path": str(ROOT / config[key]), "sha256": config[key + "_sha256"]}
        for key in ("innovation_audit", "physical_audit")
    }
    innovation, physical_audit = (prefix.read(refs[key], consumed) for key in ("innovation_audit", "physical_audit"))
    prefix.require(
        innovation["complete"] is innovation["passed"] is physical_audit["complete"] is physical_audit["passed"] is True
        and physical_audit["checker_policy_isolated"] is True
        and physical_audit["continuation_periods"] == 46
        and physical_audit["control_qualified"] is False,
        "Audited innovation and independently replayed physics are required",
    )
    producer, checker = prefix.read(PRODUCER, consumed), prefix.read(CHECKER, consumed)
    prefix.require(
        producer["complete"] is producer["passed"] is checker["complete"] is checker["passed"] is True
        and checker["candidate_modules_unloaded"] is True,
        "Both source-pinned physical phases must be closed",
    )
    stopped = select_completed(producer, checker, innovation)
    prefix.checked(config_ref, consumed)
    prefix.reference(PLAN)
    inputs.update(
        config=config, early_config_ref=config_ref, early_refs=refs, stop_inputs=stopped, source_sha256=consumed
    )
    return inputs


def input_receipt(inputs):
    return {
        "inputs": earlier.input_receipt(inputs),
        "early_config_ref": inputs["early_config_ref"],
        "early_refs": inputs["early_refs"],
        "producer_ref": PRODUCER,
        "checker_ref": CHECKER,
    }
