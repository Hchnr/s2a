"""Fixed WAM case configuration with qualified preview control and no expert path."""

import copy
from pathlib import Path

from s2a.artifacts import REPO
from s2a.clean_training_runtime import GPU, read, require
from s2a.serialized_preview_wam_execution import PORT, ROOT
from s2a.terminal_development_control import BASE_REF, CHECKPOINT, FIXTURE_SHA, RELOCATION_REF, safe_template
from s2a.terminal_development_policy import case_order

CONTROL_FIELDS = (
    "arm_pose_policy",
    "arm_pose_objective",
    "source_provisional_inverse_weight",
    "arm_residual_policy",
    "arm_forward_parameter_ref",
    "arm_forward_qualification_ref",
    "pose_preview_chain_ref",
    "response_parameters_ref",
    "planner_preflight_ref",
    "gripper_velocity_envelope",
)


def make_case_profile(prepared, control_ref, *, case):
    """Non-executable fixed inputs for pre-registration inspection, with no circular refs."""
    require(
        Path(prepared["root"]) == ROOT
        and type(case.get("case_index")) is int
        and 0 <= case["case_index"] < 12
        and case == case_order()[case["case_index"]],
        "Fixed preview development preparation/case binding required",
    )
    base = read(BASE_REF)
    require(base["fixture_sha256"] == FIXTURE_SHA, "Fixed current-initial-state fixture required")
    require(prepared["tool_relocation_ref"] == RELOCATION_REF, "Registered tool contract relocation required")
    template = safe_template(prepared["tool_relocation_ref"])
    control = read(control_ref)
    value = copy.deepcopy(base)
    value.update({key: control[key] for key in CONTROL_FIELDS})
    value.update(
        output_dir=str((ROOT / "development/cases" / case["case_id"]).relative_to(REPO)),
        records=[case["record_name"]],
        sampling_seeds=[case["sampling_seed"]],
        simulation_python=str(REPO / base["simulation_python"]),
        fixture=str(REPO / base["fixture"]),
        policy_template=template["path"],
        cosmos_repo=prepared["model_config"]["cosmos_source_root"],
        online_max_steps=600,
        online_chunk_size=40,
        max_observation_interval_steps=8,
        stable_success_last_steps=40,
        gripper_preview_steps=0,
        inverse_weight=control["source_provisional_inverse_weight"],
        preview_control_closure_ref=prepared["preview_control_closure_ref"],
        notes=["Fixed preview S2A with current-image WAM references; inference latency excluded; no expert inputs"],
    )
    value["checkpoints"] = [dict(prepared["model_config"]["checkpoint"], name=CHECKPOINT, gpu_uuid=GPU, port=PORT)]
    return value


def make_case_config(prepared, registration, *, prepared_ref, registration_ref, case):
    require(
        registration["prepared_ref"] == prepared_ref
        and registration["preview_control_closure_ref"] == prepared["preview_control_closure_ref"],
        "Fixed preview development preparation/case binding required",
    )
    value = make_case_profile(prepared, registration["physical_control_admission"]["control_config_ref"], case=case)
    value["wam_development"] = dict(
        case=case,
        prepared_ref=prepared_ref,
        execution_registration_ref=registration_ref,
        service_source_ref=registration["service_source_ref"],
        readout_ref=prepared["candidate_params_ref"],
    )
    return value
