"""Bind all original correct-reference cases to reset-only policy inputs."""

import hashlib
from pathlib import Path

import numpy as np

from s2a.infrastructure.publication import result_json
from s2a.references.oracle_timeline import OracleTimeline
from s2a.research.native_runtime.correct_reference_reset import verify_correct_reset
from s2a.research.native_runtime.development_prefix_replay import map_record_xml
from s2a.research.native_runtime.reset_endpoint import saved_reset_endpoint
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_correct_reference as previous
from scripts.verification.reference_ports.audit_reset_correct_reference import audit_targets
from scripts.wam.tracking.preview import bind_bounded_release_shadow as cohort

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/joint_control/reset_correct_reference_cohort_20261116.json"
PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/reset_correct_reference_cohort_20261116.md"
ROSTER = (
    (8, 164),
    (9, 254),
    (6, 183),
    (8, 170),
    (8, 170),
    (9, 244),
    (9, 334),
    (6, 215),
    (6, 182),
    (8, 167),
    (8, 168),
    (9, 281),
    (9, 228),
    (6, 172),
    (6, 175),
    (8, 600),
    (9, 600),
    (6, 600),
)
TARGETS = {"8": "akita_black_bowl_1", "9": "wine_bottle_1", "6": "cream_cheese_1"}


def validate_config(config):
    """Reject any roster, resource, or gate change before opening the cohort."""
    fixed = dict(
        schema="s2a.reset-correct-reference-cohort.v1",
        output="outputs/reset_correct_reference_cohort_20261116",
        reused_case=8,
        target_objects_by_task=TARGETS,
        max_workers_per_phase=17,
        max_models_per_phase=51,
        max_contexts_per_phase=51,
        max_new_periods_per_phase=4725,
        max_new_inner_steps_per_phase=118125,
        max_baseline_responses_per_period=480,
        max_quality_responses_per_period=320,
        max_gripper_sequences_per_period=6561,
        control_dt_s=0.05,
        inner_dt_s=0.002,
        position_cap_m=0.01,
        rotation_cap_rad=0.05,
        joint_buffer_rad=0.05,
        worker_timeout_seconds=3600,
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    cases = [dict(index=i, task_id=task, seed=42, periods=periods) for i, (task, periods) in enumerate(ROSTER)]
    if any(config.get(key) != value or type(config.get(key)) is not type(value) for key, value in fixed.items()):
        raise ValueError("The original cohort budgets, tasks, and gates must remain fixed")
    if config.get("cases") != cases:
        raise ValueError("The original ordered eighteen-case roster must remain fixed")
    if (
        sum(item["periods"] for item in cases) != 4907
        or sum(item["periods"] for item in cases if item["index"] != 8) != config["max_new_periods_per_phase"]
    ):
        raise ValueError("The original and newly executed period denominators must agree")


def configured_ref(value):
    if set(value) != {"path", "sha256"}:
        raise ValueError("Each configured input must have one path and SHA256")
    ref = dict(path=str(ROOT / value["path"]), sha256=value["sha256"])
    prefix.checked(ref)
    return ref


def shared_inputs():
    """Use the previously reviewed simulator assets and XML mapping."""
    source = prefix.load_inputs({2: 16, 6: 16})
    migrated = prefix.read(prefix.MIGRATED_MANIFEST)
    template = source["cases"][0]
    receipt = dict(
        spec_ref=source["spec_ref"],
        asset_refs=source["asset_refs"],
        source_sha256=source["source_sha256"],
        cv2_admission=source["cv2_admission"],
        migrated_ref=prefix.MIGRATED_MANIFEST,
    )
    return dict(
        receipt=receipt,
        asset_refs=source["asset_refs"],
        cv2_admission=source["cv2_admission"],
        migrated=migrated,
        template_record=template["record"],
        template_xml=template["record_xml"],
    )


def read_reset_endpoint(ref, count):
    """Read the first endpoint and count the remaining records without decoding them."""
    return prefix.read_recorded_prefix(prefix.checked(ref), 1, count + 1)[0]


def case_inputs(identity, shared):
    """Construct only a reset, fixed control contract and the oracle reference."""
    spec, refs = identity["spec"], identity["refs"]
    for ref in refs.values():
        prefix.checked(ref)
    baseline, checked = prefix.read(refs["result"]), prefix.read(refs["native_check"])
    manifest, control, tool_data, execution, parameters = (
        prefix.read(refs[key]) for key in ("manifest", "controller", "tool", "execution", "parameters")
    )
    if not (
        baseline["complete"] is baseline["passed"] is baseline["performance_qualified"] is True
        and checked["complete"] is checked["passed"] is checked["performance_qualified"] is True
        and baseline["case_index"] == spec["index"]
        and baseline["physics_steps"] == checked["physics_steps"] == spec["periods"]
        and checked["result_ref"] == refs["result"]
        and baseline["arrays_ref"] == refs["arrays"]
        and execution["response_parameters_ref"] == refs["parameters"]
        and manifest["runtime_config"]["task_id"] == spec["task_id"]
        and manifest["runtime_config"]["seed"] == spec["seed"]
    ):
        raise ValueError("Each case must retain its qualified original identity and reset")
    rows, chunks = prefix.read(refs["rows"]), prefix.read(refs["chunks"])
    timeline = OracleTimeline(chunks, periods=spec["periods"])
    target_parity = audit_targets(timeline, rows)
    with np.load(prefix.checked(refs["arrays"]), allow_pickle=False) as archive:
        initial = saved_reset_endpoint(archive)
    endpoint = read_reset_endpoint(refs["endpoints"], spec["periods"])
    if not baseline["images"] or endpoint != baseline["images"][0]["current_endpoint"]:
        raise ValueError("The reset endpoint must match the original qualified result")
    tool = np.asarray(tool_data["source_from_tcp"], dtype=float)
    anchor, opening = np.asarray(chunks[0]["anchor_tcp"], dtype=float), float(rows[0]["opening_before"])
    verified = verify_correct_reset(initial, endpoint, initial, endpoint, control, tool, anchor, opening)
    if verified["recorded_state_sha256"] != manifest["record"]["initial_state_sha256"]:
        raise ValueError("The reset must match its original manifest")
    reset = dict(expected=initial, expected_receipt=endpoint, anchor=anchor, opening=opening)
    reset_sha = hashlib.sha256(result_json(reset).encode()).hexdigest()
    if identity.get("reset_input_sha256") not in (None, reset_sha):
        raise ValueError("The registered reset inputs changed")
    config = dict(manifest["runtime_config"])
    for key in ("libero_root", "libero_config", "osmesa_library"):
        config[key] = shared["migrated"]["runtime_config"][key]
        if not Path(config[key]).is_relative_to(ROOT):
            raise ValueError("The simulator runtime must remain inside this repository")
    record_xml = map_record_xml(
        manifest["record"]["model_xml"].encode(),
        shared["template_record"]["model_xml"].encode(),
        shared["template_xml"].encode(),
    ).decode()
    poses, openings, _ = timeline.window(0)
    case = dict(
        chunk=dict(anchor_tcp=anchor, effective_targets_tcp=poses, openings=openings),
        controller=control,
        tool=tool,
        gripper_parameters=parameters,
    )
    return dict(
        spec=spec,
        case=case,
        reset=reset,
        reset_input_sha256=reset_sha,
        target_parity=target_parity,
        timeline=timeline,
        runtime_config=config,
        record=manifest["record"],
        record_xml=record_xml,
        target_object=TARGETS[str(spec["task_id"])],
    )


def load_inputs():
    """Validate the complete original roster and bind the reused case 08."""
    prefix.environment()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    validate_config(config)
    configured = {
        key: configured_ref(config[key])
        for key in ("prepared", "original_qualification", "reused_producer", "reused_checker", "reused_audit")
    }
    source = cohort.load_inputs()
    prefix.require(
        source["prepared_ref"] == configured["prepared"]
        and source["qualification_ref"] == configured["original_qualification"],
        "The original independently qualified 18-case roster is required",
    )
    profiles = prefix.read(configured["prepared"])["profiles"]
    shared = shared_inputs()
    registered, controls, parameters = [], [], []
    old = previous.load_inputs()
    prior_producer, prior_checker, prior_audit = (
        prefix.read(configured[key]) for key in ("reused_producer", "reused_checker", "reused_audit")
    )
    prior_registration = prefix.read(prior_producer["registration_ref"])
    prefix.require(
        prior_producer["complete"]
        is prior_producer["passed"]
        is prior_checker["complete"]
        is prior_checker["passed"]
        is True
        and prior_audit["complete"] is prior_audit["passed"] is True
        and prior_audit["producer_ref"] == configured["reused_producer"]
        and prior_audit["checker_ref"] == configured["reused_checker"]
        and prior_registration["source_refs"] == previous.source_refs(),
        "The reused case must retain its closed independent replay and exact policy sources",
    )
    for spec, item, entry in zip(config["cases"], source["cases"][:18], profiles, strict=True):
        index = spec["index"]
        baseline = prefix.read(item["refs"]["result"])
        execution = prefix.read(entry["execution_config_ref"])
        directory = cohort.ORACLE_ROOT / f"case{index:02d}_candidate"
        refs = dict(
            item["refs"],
            manifest=entry["manifest_ref"],
            controller=entry["controller_ref"],
            execution=entry["execution_config_ref"],
            parameters=execution["response_parameters_ref"],
            arrays=baseline["arrays_ref"],
            endpoints=prefix.reference(directory / "endpoints.jsonl"),
        )
        prefix.require(
            item["index"] == entry["case_index"] == index
            and item["steps"] == entry["profile"]["physics_steps"] == spec["periods"],
            "Original case ordering and period counts must match",
        )
        identity = dict(spec=spec, refs=refs)
        case = case_inputs(identity, shared)
        identity.update(reset_input_sha256=case["reset_input_sha256"], target_parity=case["target_parity"])
        registered.append(identity)
        control = case["case"]["controller"]
        controls.append(
            {
                key: control[key]
                for key in ("contract", "gripper_config", "gripper_velocity_envelope", "gripper_algorithm")
            }
        )
        parameters.append(refs["parameters"]["sha256"])
        if index == config["reused_case"]:
            prefix.require(
                identity["reset_input_sha256"] == old["identity"]["reset_input_sha256"]
                and refs["chunks"]["sha256"] == old["identity"]["reference_ref"]["sha256"]
                and case["case"]["controller"] == old["case"]["controller"]
                and case["case"]["gripper_parameters"] == old["case"]["gripper_parameters"]
                and case["target_object"] == previous.TARGET_OBJECT,
                "Reused case 08 must have the identical reset, targets, controller, and parameters",
            )
            np.testing.assert_array_equal(case["case"]["tool"], old["case"]["tool"])
    prefix.require(
        len(registered) == 18 and len(set(parameters)) == 1 and all(item == controls[0] for item in controls),
        "All cases must use one fixed controller contract and gripper model",
    )
    return dict(
        config=config,
        config_ref=config_ref,
        configured=configured,
        cases=registered,
        shared_receipt=shared["receipt"],
        reused_source_refs=prior_registration["source_refs"],
    )


def input_receipt(inputs):
    """Keep immutable source identities, never historical commands or future states."""
    return {key: inputs[key] for key in ("config_ref", "configured", "cases", "shared_receipt", "reused_source_refs")}
