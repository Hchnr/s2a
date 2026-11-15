"""Bind the broader recovery pilot to its fixed training-only sources."""

from pathlib import Path

import numpy as np

from s2a.references.broader_recovery_timeline import broader_recovery_timeline
from scripts.simulation.runtime import prepare_reset_correct_reference_cohort as preparation
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_current_controller_recovery as previous

ROOT = prefix.REPO
CONFIG = ROOT / "configs/wam/training/recovery/broader_recovery_pilot_20261116.json"
PLAN = ROOT / "plans/v0.0.1/wam/training/recovery/broader_recovery_pilot_20261116.md"
OUTPUT = ROOT / "outputs/broader_recovery_pilot_20261116"
SOURCES = [
    dict(cohort_index=3, task_index=0, demo="demo_40", sign=1, native_periods=170),
    dict(cohort_index=4, task_index=0, demo="demo_48", sign=-1, native_periods=170),
    dict(cohort_index=5, task_index=1, demo="demo_39", sign=1, native_periods=244),
    dict(cohort_index=6, task_index=1, demo="demo_46", sign=-1, native_periods=334),
    dict(cohort_index=8, task_index=2, demo="demo_49", sign=1, native_periods=182),
    dict(cohort_index=13, task_index=2, demo="demo_5", sign=-1, native_periods=172),
]
VARIANTS = ("zero", "base_x", "base_y", "base_yaw")
TASK_IDS = (8, 9, 6)
REGISTERED_REFS = {
    "training_preparation": dict(
        path="outputs/wam_stability_20261021/clean_residual_training_20261023/recovery_20261023/preparation/prepared.json",
        sha256="e6fab51219fba6d108bd6b858f2467a9a0e54c5fc22890f8abd7c318742f8e54",
    ),
    "correct_reference_cohort": dict(
        path="configs/tracking/joint_control/reset_correct_reference_cohort_20261116.json",
        sha256="b4f9da8ff26a3e972b9cd996a36a0aff0285050a71e71eb4e16cf27d6a812312",
    ),
    "current_controller_audit": dict(
        path="outputs/cached_seed_preference_20261116/audit.json",
        sha256="ae8358fc25a9675d208f91cd8e7c52f4d20f120e9c314ee18755b97472744525",
    ),
    "coverage_audit": dict(
        path="outputs/wam_recovery_state_coverage_20261116/result.json",
        sha256="b1fe1a46929d4733677cf79e50cdb13b88e7d713ee24bdaf9aedc30c646f3846",
    ),
    "coverage_independent": dict(
        path="outputs/wam_recovery_state_coverage_20261116/independent.json",
        sha256="43a779cc0d68bdd76713c8c6df0eb402ebb9b473ebd399cb32072cf396b4def3",
    ),
    "previous_recovery_audit": dict(
        path="outputs/current_controller_recovery_repair_20261116/audit.json",
        sha256="a10dc09801d90efd92933e536d04a1e71ea1da75b724105dc384205c4d61c7f6",
    ),
}


def _same_typed(actual, expected):
    """Reject JSON booleans silently comparing equal to integer budget values."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(_same_typed(actual[k], value) for k, value in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_same_typed(a, b) for a, b in zip(actual, expected, strict=True))
    return actual == expected


def validate_config(config):
    fixed = dict(
        schema="s2a.broader-recovery-pilot.v1",
        output="outputs/broader_recovery_pilot_20261116",
        sources=SOURCES,
        variant_order=list(VARIANTS),
        translation_magnitude_m=0.06,
        base_yaw_magnitude_rad=0.5,
        perturb_steps=16,
        return_steps=40,
        anchors=[0, 16, 56],
        future_window_steps=40,
        seed=42,
        control_dt_s=0.05,
        inner_dt_s=0.002,
        position_cap_m=0.01,
        rotation_cap_rad=0.05,
        joint_buffer_rad=0.05,
        arm_rate_limit=0.25,
        max_baseline_responses_per_period=480,
        max_quality_responses_per_period=320,
        max_gripper_sequences_per_period=6561,
        registered_cases=24,
        probe_cases=[0],
        max_probe_control_periods=0,
        max_probe_inner_steps=0,
        max_probe_models=3,
        max_probe_contexts=3,
        max_workers_per_phase=24,
        max_periods_per_phase=6432,
        max_inner_steps_per_phase=160800,
        max_models_per_phase=72,
        max_contexts_per_phase=72,
        worker_timeout_seconds=3600,
        world_model_calls=0,
        gpu_calls=0,
        fits=0,
        fresh_cases=0,
        training_authorized=False,
        automatic_retry=False,
        **REGISTERED_REFS,
    )
    prefix.require(_same_typed(config, fixed), "The registered roster, identities, gates and budgets must match")
    for key in REGISTERED_REFS:
        preparation.configured_ref(config[key])


def validate_source(source, origin, manifest, trained, audit_case, native):
    """Require the same qualified source while rejecting held-out identities."""
    record, spec = manifest["record"], origin["spec"]
    sha = record["initial_state_sha256"]
    key = (source["task_index"], source["demo"])
    identity = dict(task_index=key[0], demo=key[1], initial_state_sha256=sha)
    candidates = [
        row for row in trained["training_anchors"] if row["anchor_frame"] == 0 and row["source_identity"] == identity
    ]
    excluded = trained["exclusion"]
    forbidden_keys = {tuple(item) for item in excluded["forbidden_demo_keys"] + excluded["validation_demo_keys"]}
    fresh = excluded["fresh_sources"]
    prefix.require(
        record["task_index"] == key[0]
        and record["demo"] == key[1]
        and record["task_id"] == TASK_IDS[key[0]] == spec["task_id"]
        and spec["index"] == source["cohort_index"]
        and spec["periods"] == source["native_periods"]
        and spec["seed"] == 42
        and key not in forbidden_keys
        and sha not in excluded["forbidden_initial_state_sha256"]
        and not any((item["task_index"], item["demo"]) == key or item["initial_state_sha256"] == sha for item in fresh)
        and len(candidates) == 1,
        "The reset must be a unique registered training source outside validation and fresh identities",
    )
    anchor = candidates[0]
    native_record = native["record"]
    prefix.require(
        native["complete"] is native["passed"] is native["native_quality_passed"] is True
        and native["physics_steps"] == source["native_periods"]
        and native["arrays"] == anchor["trajectory_ref"]
        and all(
            native_record[field] == record[field] for field in ("task_index", "task_id", "demo", "initial_state_sha256")
        )
        and audit_case["case"] == spec
        and audit_case["producer_status"] == audit_case["checker_status"] == "closed"
        and audit_case["qualified"] is audit_case["replayed"] is True,
        "The native trajectory and independently replayed current-controller qualification must match",
    )
    return anchor


def source_refs():
    """Bind the previous control/capture closure and this preparation code."""
    return previous.source_refs() + [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__).resolve(),
            ROOT / "src/s2a/references/broader_recovery_timeline.py",
            ROOT / "tests/unit/references/test_broader_recovery_timeline.py",
            ROOT / "tests/research/native_runtime/test_broader_recovery_preparation.py",
        )
    ]


def load_source(case, trajectory_ref):
    """Verify the original reset and all native targets before reusing them."""
    previous.make_timeline(case, trajectory_ref, 0.0)
    with np.load(prefix.checked(trajectory_ref), allow_pickle=False) as data:
        return np.asarray(data["source_pose"], dtype=float).copy(), np.asarray(data["opening"], dtype=float).copy()


def make_timeline(case, trajectory_ref, variant, sign):
    source, opening = load_source(case, trajectory_ref)
    return timeline_from_source(case, source, opening, variant, sign)


def timeline_from_source(case, source, opening, variant, sign):
    timeline, metadata = broader_recovery_timeline(source, opening, case["case"]["tool"], variant=variant, sign=sign)
    return timeline, metadata, previous.timeline_digest(timeline)


def load_inputs():
    """Construct 24 immutable case identities without simulating or reading fresh futures."""
    prefix.environment()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    validate_config(config)
    refs = {key: preparation.configured_ref(config[key]) for key in REGISTERED_REFS}
    trained = prefix.read(refs["training_preparation"])
    audit = prefix.read(refs["current_controller_audit"])
    original, shared = preparation.load_inputs(), preparation.shared_inputs()
    prefix.require(
        original["config_ref"] == refs["correct_reference_cohort"]
        and audit["complete"] is audit["passed"] is True
        and audit["registered_cases"] == len(audit["cases"]) == 18,
        "The fixed cohort and complete current-controller audit are required",
    )
    cases, seen = [], set()
    for source in SOURCES:
        origin = original["cases"][source["cohort_index"]]
        manifest = prefix.read(origin["refs"]["manifest"])
        identity = dict(
            task_index=source["task_index"],
            demo=source["demo"],
            initial_state_sha256=manifest["record"]["initial_state_sha256"],
        )
        candidates = [
            item
            for item in trained["training_anchors"]
            if item["anchor_frame"] == 0 and item["source_identity"] == identity
        ]
        prefix.require(len(candidates) == 1, "Each selected training source needs one initial anchor")
        native = prefix.read(candidates[0]["case_ref"])
        qualified = audit["cases"][source["cohort_index"]]
        anchor = validate_source(source, origin, manifest, trained, qualified, native)
        prefix.checked(qualified["producer_ref"])
        prefix.checked(qualified["checker_ref"])
        prefix.require(identity["initial_state_sha256"] not in seen, "Training initial identities must be distinct")
        seen.add(identity["initial_state_sha256"])
        case = preparation.case_inputs(origin, shared)
        poses, opening = load_source(case, anchor["trajectory_ref"])
        for variant in VARIANTS:
            timeline, metadata, digest = timeline_from_source(case, poses, opening, variant, source["sign"])
            prefix.require(timeline.periods == source["native_periods"] + 56, "The full native timeline is required")
            cases.append(
                dict(
                    index=len(cases),
                    source=source,
                    variant=variant,
                    periods=timeline.periods,
                    initial_state_sha256=identity["initial_state_sha256"],
                    trajectory_ref=anchor["trajectory_ref"],
                    training_case_ref=anchor["case_ref"],
                    original_identity=origin,
                    timeline_metadata=metadata,
                    timeline_sha256=digest,
                )
            )
    prefix.require(
        len(cases) == config["registered_cases"]
        and sum(case["periods"] for case in cases) == config["max_periods_per_phase"]
        and sum(case["periods"] for case in cases) * 25 == config["max_inner_steps_per_phase"],
        "All registered cases and full phase budgets must agree",
    )
    return config, dict(config_ref=config_ref, source_refs=refs, shared_receipt=shared["receipt"], cases=cases)
