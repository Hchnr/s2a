"""Bind a matched-state diagnostic to previously replayed correct-reference actions."""

import hashlib

import numpy as np

from s2a.infrastructure.publication import result_json
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import prepare_reset_correct_reference_cohort as cohort_inputs
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_correct_reference as original
from scripts.simulation.runtime import run_reset_correct_reference_cohort as cohort

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/joint_control/correct_reference_same_state_arm_20261116.json"
PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/correct_reference_same_state_arm_20261116.md"


def validate_config(config):
    fixed = dict(
        schema="s2a.correct-reference-same-state-arm.v1",
        output="outputs/correct_reference_same_state_arm_20261116",
        case_index=3,
        task_id=8,
        seed=42,
        target_object="akita_black_bowl_1",
        prefix_periods=40,
        intervention_periods=16,
        branches=["candidate_actions", "original_arm"],
        max_workers_per_phase=2,
        max_models_per_phase=6,
        max_contexts_per_phase=6,
        max_control_periods_per_phase=112,
        max_inner_steps_per_phase=2800,
        control_dt_s=0.05,
        inner_dt_s=0.002,
        arm_rate_limit=0.25,
        joint_buffer_rad=0.05,
        worker_timeout_seconds=1800,
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    if any(config.get(k) != value or type(config.get(k)) is not type(value) for k, value in fixed.items()):
        raise ValueError("The fixed same-state diagnostic and its original budgets are required")


def fixed_actions(candidate, baseline, *, start, count, rate_limit):
    """Keep gripper commands identical and reject box or transition-rate violations."""
    candidate, baseline = np.asarray(candidate, dtype=float), np.asarray(baseline, dtype=float)
    if (
        candidate.ndim != 2
        or baseline.ndim != 2
        or candidate.shape[1:] != (7,)
        or baseline.shape[1:] != (7,)
        or type(start) is not int
        or type(count) is not int
        or start < 1
        or count < 1
        or min(len(candidate), len(baseline)) < start + count
        or not np.isfinite(candidate).all()
        or not np.isfinite(baseline).all()
    ):
        raise ValueError("Finite recorded actions must cover the fixed diagnostic window")
    first = candidate[start : start + count].copy()
    alternate = baseline[start : start + count].copy()
    if not np.array_equal(first[:, 6], alternate[:, 6]):
        raise ValueError("Both recorded gripper command sequences must match exactly")
    alternate[:, 6] = first[:, 6]
    previous = candidate[start - 1].copy()
    rates = {}
    for name, sequence in (("candidate_actions", first), ("original_arm", alternate)):
        if np.max(np.abs(sequence)) > 1 + 1e-12:
            raise ValueError("Each diagnostic command must remain in the original box")
        rate = float(np.max(np.abs(np.diff(np.vstack((previous[:6], sequence[:, :6])), axis=0))))
        if rate > rate_limit + 1e-12:
            raise ValueError("Each diagnostic command must respect the original arm rate limit")
        rates[name] = rate
    return dict(candidate_actions=first, original_arm=alternate), previous, rates


def recorded_prefix(segments, steps):
    """Select only complete, uninterrupted recorded segments through the branchpoint."""
    selected, position = [], 0
    for row in segments:
        if position == steps:
            break
        outcome = row["continuation"]
        count = outcome["steps_completed"]
        if (
            row["start_step"] != position
            or row["boundary"] != "window_end"
            or count != 8
            or outcome["stop"] is not None
            or position + count > steps
        ):
            raise ValueError("The fixed branchpoint requires complete recorded prefix windows")
        selected.append(row)
        position += count
    if position != steps:
        raise ValueError("The recorded prefix does not reach the fixed branchpoint")
    return dict(segments=selected, complete=False, stop=None)


def load_inputs():
    prefix.environment()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    validate_config(config)
    refs = {
        key: cohort_inputs.configured_ref(config[key]) for key in ("cohort_audit", "cohort_dry", "divergence_audit")
    }
    audit, dry, divergence = (prefix.read(refs[key]) for key in ("cohort_audit", "cohort_dry", "divergence_audit"))
    index = config["case_index"]
    entry, identity, diagnostic = audit["cases"][index], dry["inputs"]["cases"][index], divergence["cases"][index]
    producer, checker = (prefix.read(entry[key]) for key in ("producer_ref", "checker_ref"))
    registration = prefix.read(prefix.read(audit["producer_ref"])["registration_ref"])
    prefix.require(
        audit["complete"] is audit["passed"] is divergence["complete"] is divergence["passed"] is True
        and audit["qualified_cases"] == 14
        and audit["registered_cases"] == 18
        and entry["case"] == identity["spec"] == diagnostic["case"]
        and entry["case"]["index"] == index
        and entry["qualified"] is False
        and entry["replayed"] is True
        and diagnostic["producer_ref"] == entry["producer_ref"]
        and registration["inputs"] == dry["inputs"]
        and registration["source_refs"] == cohort.source_refs(),
        "The fixed failed case, independent replay and original policy sources must match",
    )
    all_periods, checked_periods = original.periods(producer), original.periods(checker)
    prefix.require(
        len(all_periods) == len(checked_periods) == identity["spec"]["periods"], "Full recorded case required"
    )
    maxima = comparison.compare_periods(all_periods, checked_periods)
    prefix.require(all(value == 0.0 for value in maxima.values()), "The historical candidate replay must be exact")
    shared = cohort_inputs.shared_inputs()
    prefix.require(shared["receipt"] == dry["inputs"]["shared_receipt"], "The original simulator assets must match")
    case = cohort_inputs.case_inputs(identity, shared)
    rows = prefix.read(identity["refs"]["rows"])
    candidate_commands = [item["command"] for item in all_periods]
    baseline_commands = [row["command"] for row in rows]
    actions, previous, rates = fixed_actions(
        candidate_commands,
        baseline_commands,
        start=config["prefix_periods"],
        count=config["intervention_periods"],
        rate_limit=config["arm_rate_limit"],
    )
    recorded = recorded_prefix(producer["runtime"]["continuation"]["segments"], config["prefix_periods"])
    action_sha = {name: hashlib.sha256(result_json(value).encode()).hexdigest() for name, value in actions.items()}
    source_sha = dict(shared["receipt"]["source_sha256"])
    for ref in [config_ref, *refs.values(), *identity["refs"].values(), entry["producer_ref"], entry["checker_ref"]]:
        prefix.checked(ref, source_sha)
    return dict(
        config=config,
        config_ref=config_ref,
        refs=refs,
        identity=identity,
        shared_receipt=shared["receipt"],
        source_sha256=source_sha,
        candidate_producer_ref=entry["producer_ref"],
        candidate_checker_ref=entry["checker_ref"],
        case=case,
        shared=shared,
        recorded_prefix=recorded,
        candidate_periods=all_periods,
        actions=actions,
        previous_command=previous,
        action_sha256=action_sha,
        maximum_arm_rate=rates,
        object_state_columns=diagnostic["metrics"]["state_position_mapping"]["columns"],
    )


def input_receipt(inputs):
    return {
        key: inputs[key]
        for key in (
            "config_ref",
            "refs",
            "identity",
            "shared_receipt",
            "source_sha256",
            "candidate_producer_ref",
            "candidate_checker_ref",
            "action_sha256",
            "maximum_arm_rate",
            "object_state_columns",
        )
    }
