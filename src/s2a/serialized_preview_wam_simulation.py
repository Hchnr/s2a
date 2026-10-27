"""WAM development execution with the qualified preview control loop.

Requires a separate registered service and complete physical control closure.
"""

import importlib.metadata
import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import requests
from scipy.spatial.transform import Rotation

from s2a.action_reference import decode_action_targets, encode_oracle_targets
from s2a.artifacts import REPO, sha256, write_json
from s2a.cameras import render_current_cameras
from s2a.checkpoint_simulation import object_progress
from s2a.clean_training_runtime import checked
from s2a.coupled_pose_preview_runtime import solve_preview
from s2a.coupled_preview_control_gate import admit_preview_control
from s2a.coupled_preview_wam_protocol import validate_preview_wam_response
from s2a.coupled_response_history import CoupledResponseHistory
from s2a.current_coupled_response import build_response
from s2a.current_coupled_state import read_current_dynamics
from s2a.current_model_state_native_precision import current_model_state
from s2a.executed_command_residual import ExecutedCommandResidualOSC
from s2a.generated_reference_twist import reference_world_twist
from s2a.geometry import inverse, pose_error
from s2a.gripper_intent_hold import GripperIntentHold, GripperIntentHoldConfig
from s2a.gripper_motor_state import preserve_motor_target
from s2a.gripper_position_hold import GripperPositionHold, GripperPositionHoldConfig
from s2a.gripper_tracking import panda_opening_from_motor_target
from s2a.joint_wrench_condition_forward import nominal as arm_nominal
from s2a.libero_simulation import (
    aperture,
    controller_contract,
    current_observation,
    generate,
    make_environment,
    measured_twist,
    sha256_bytes,
    source_pose,
)
from s2a.loaded_velocity_control import LoadedVelocityControl
from s2a.osc import OSCInterface
from s2a.reference_governor import ReferenceGovernorConfig
from s2a.reference_governor_v2 import RecoveringReferenceGovernor
from s2a.reference_linear_velocity import reference_linear_velocity
from s2a.runtime_response_history import RuntimeResponseHistory
from s2a.stability_recovery_simulation import (
    execution_inputs as recovery_execution_inputs,
)
from s2a.stability_recovery_simulation import (
    sample_model_reference,
    simulator_inputs,
    verify_inputs,
)
from s2a.task_acceptance import task_outcome
from s2a.terminal_development_policy import case_order
from s2a.velocity_model_planner import VelocityModelPlanner
from s2a.wam_regularization import attenuate_wam_rotation


def position_variant(config, name, *, source):
    """Freeze the qualified v2/v3 grippers and explicit model interventions."""
    matches = [v for v in config["variants"] if v["name"] == name]
    if len(matches) != 1 or source not in ("wam", "oracle"):
        raise ValueError("requires one explicit variant and WAM/oracle source")
    variant = dict(matches[0])
    if variant["rotation"] != "response" or variant["gripper"] not in ("intent_hold", "position_hold"):
        raise ValueError("only response inverse and the explicit v2/v3 grippers are admitted")
    if variant["governor"] is not False and variant["governor"] != "recovering":
        raise ValueError("governor must be false or recovering")
    alpha = variant["model_rotation_alpha"]
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, (int, float)) or alpha not in (0, 0.5, 1):
        raise ValueError("model_rotation_alpha must explicitly be 1, .5 or 0")
    if source == "oracle" and alpha != 1:
        raise ValueError("oracle qualification must preserve its supplied reference")
    if asdict(GripperIntentHoldConfig(**config["gripper_intent_hold"])) != asdict(GripperIntentHoldConfig()):
        raise ValueError("this executor requires the physically qualified intent_hold defaults")
    if asdict(GripperPositionHoldConfig(**config["gripper_position_hold"])) != asdict(GripperPositionHoldConfig()):
        raise ValueError("position adapter parameters differ from the frozen qualification defaults")
    ReferenceGovernorConfig(**config["reference_governor"])
    return variant


def execution_inputs(config, config_path):
    """Bind this version and its actual imports, preserving live old sources."""
    signatures = recovery_execution_inputs(config, config_path)
    signatures.pop(str(REPO / "scripts/run_wam_recovery_matrix.py"))
    path = REPO / "scripts/run_wam_position_matrix.py"
    signatures[str(path)] = sha256(path)
    return signatures


def make_gripper(variant, initial_opening, config):
    if variant["gripper"] == "position_hold":
        return GripperPositionHold(initial_opening, config=GripperPositionHoldConfig(**config["gripper_position_hold"]))
    return GripperIntentHold(initial_opening, config=GripperIntentHoldConfig(**config["gripper_intent_hold"]))


def run_preview_wam_case(
    config,
    root,
    record,
    *,
    source,
    variant_name,
    sampling_seed,
    checkpoint=None,
    config_path,
    frozen_inputs=None,
    oracle_reference,
    endpoint_observer,
    response_parameters,
):
    """Execute current-image WAM references; observers record measured endpoints only."""
    if source != "wam" or oracle_reference is not None or not callable(endpoint_observer):
        raise ValueError("WAM requires a measured endpoint observer and no expert reference")
    admission = admit_preview_control(config["preview_control_closure_ref"])
    qualified_control = json.loads(Path(admission["control_config_ref"]["path"]).read_text())
    for key in (
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
    ):
        if config[key] != qualified_control[key]:
            raise ValueError("WAM control differs from the physical-qualified candidate")
    parameter_ref = qualified_control["response_parameters_ref"]
    checked(parameter_ref)
    if (
        response_parameters != json.loads(Path(parameter_ref["path"]).read_text())
        or config["gripper_preview_steps"] != 0
        or config["inverse_weight"] != qualified_control["source_provisional_inverse_weight"]
    ):
        raise ValueError("Qualified gripper parameters or SOURCE inverse weight differ")
    development = config["wam_development"]
    case = development["case"]
    if (
        type(case.get("case_index")) is not int
        or not 0 <= case["case_index"] < 12
        or case != case_order()[case["case_index"]]
        or record["name"] != case["record_name"]
        or sampling_seed != case["sampling_seed"]
        or config["online_max_steps"] != 600
        or config["online_chunk_size"] != 40
        or config["max_observation_interval_steps"] != 8
        or config["stable_success_last_steps"] != 40
    ):
        raise ValueError("Fixed12 WAM development case and600/75 budget required")
    for key in ("prepared_ref", "execution_registration_ref", "service_source_ref", "readout_ref"):
        checked(development[key])
    registration = json.loads(Path(development["execution_registration_ref"]["path"]).read_text())
    if (
        registration.get("schema_version") != "s2a.serialized-preview-wam-development-registration.v1"
        or any(registration.get(k) is not True for k in ("complete", "passed", "execution_authorized"))
        or registration.get("fresh_authorized") is not False
        or registration.get("automatic_retry") is not False
        or registration.get("prepared_ref") != development["prepared_ref"]
        or registration.get("preview_control_closure_ref") != config["preview_control_closure_ref"]
        or registration.get("case_order") != case_order()
        or registration.get("service_source_ref") != development["service_source_ref"]
    ):
        raise ValueError("Separate complete preview WAM execution registration required")
    registered_sources = registration["source_sha256"]
    for relative in (
        "src/s2a/serialized_preview_wam_simulation.py",
        "src/s2a/coupled_preview_control_gate.py",
        "src/s2a/coupled_preview_wam_protocol.py",
    ):
        path = str(REPO / relative)
        if registered_sources.get(path) != sha256(path):
            raise ValueError("WAM entry/gate/protocol not bound to execution registration")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("physics requires empty CUDA_VISIBLE_DEVICES and LD_LIBRARY_PATH")
    variant = position_variant(config, variant_name, source=source)
    if variant["governor"] or variant["model_rotation_alpha"] != 1:
        raise ValueError("Residual inverse requires ungoverned endpoint reference")
    VelocityModelPlanner(response_parameters, velocity_envelope=config["gripper_velocity_envelope"])
    arm_ref = config["arm_forward_parameter_ref"]
    qualification_ref = config["arm_forward_qualification_ref"]
    if sha256(arm_ref["path"]) != arm_ref["sha256"] or sha256(qualification_ref["path"]) != qualification_ref["sha256"]:
        raise ValueError("Qualified arm parameter source changed")
    qualification = json.loads(Path(qualification_ref["path"]).read_text())
    if not qualification["offline_forward_eligible"] or qualification["parameter_ref"] != arm_ref:
        raise ValueError("Arm model offline qualification missing")
    arm_parameters = json.loads(Path(arm_ref["path"]).read_text())
    proof_ref = config["pose_preview_chain_ref"]
    if sha256(proof_ref["path"]) != proof_ref["sha256"]:
        raise ValueError("Preview runtime proof changed")
    proof = json.loads(Path(proof_ref["path"]).read_text())
    if (
        not proof["complete"]
        or len(proof["results"]) != 18
        or sum(item["periods"] for item in proof["results"]) != 4907
        or proof["scope"]
        != "continuous reconstructed candidate caches on recorded raw endpoints; not counterfactual physics"
    ):
        raise ValueError("Preview runtime binding missing")
    if config["arm_pose_policy"] != "247 four-step pose preview;actual previous +/-0.25;explicit reset":
        raise ValueError("Fixed coupled inverse policy required")

    response_history = RuntimeResponseHistory()
    coupled_history = CoupledResponseHistory()
    previous_actual_command = np.zeros(7)
    signatures = execution_inputs(config, config_path)
    signatures.update({arm_ref["path"]: arm_ref["sha256"], qualification_ref["path"]: qualification_ref["sha256"]})
    signatures[proof_ref["path"]] = proof_ref["sha256"]
    for ref in (
        config["preview_control_closure_ref"],
        admission["control_config_ref"],
        *(
            development[key]
            for key in ("prepared_ref", "execution_registration_ref", "service_source_ref", "readout_ref")
        ),
    ):
        signatures[ref["path"]] = ref["sha256"]
    for relative in (
        "src/s2a/serialized_preview_wam_simulation.py",
        "src/s2a/coupled_preview_control_gate.py",
        "src/s2a/coupled_preview_wam_protocol.py",
        "src/s2a/terminal_development_policy.py",
        "src/s2a/coupled_pose_preview_runtime.py",
        "src/s2a/coupled_pose_preview_executable.py",
        "src/s2a/coupled_pose_preview.py",
        "src/s2a/full_coupled_pose_inverse.py",
        "src/s2a/full_coupled_pose_inverse_sqp.py",
        "src/s2a/generated_reference_twist.py",
        "src/s2a/current_coupled_response.py",
        "src/s2a/current_coupled_state.py",
        "src/s2a/coupled_response_history.py",
        "src/s2a/strict_convex_qp.py",
        "src/s2a/current_model_state.py",
        "src/s2a/current_model_state_native_precision.py",
        "src/s2a/runtime_response_history.py",
        "src/s2a/executed_command_residual.py",
        "src/s2a/joint_wrench_condition_forward.py",
    ):
        path = REPO / relative
        signatures[str(path)] = sha256(path)
    if frozen_inputs is not None:
        verify_inputs(frozen_inputs)
        if any(frozen_inputs.get(path) != digest for path, digest in signatures.items()):
            raise ValueError("worker execution inputs differ from coordinator snapshot")
        signatures = dict(frozen_inputs)
    template = json.loads(Path(config["policy_template"]).read_text())
    c = dict(
        template,
        seed=sampling_seed,
        task_id=record["task_id"],
        task_text=record["task"],
        camera_observations=False,
        record_video=False,
        return_video=False,
        refresh_controller_after_restore=True,
        action_chunk_size=config["online_chunk_size"],
        sampling_steps=config["sampling_steps"],
        guidance=config["guidance"],
        shift=config["shift"],
        control_fps=config["control_fps"],
    )
    root = Path(root).resolve()
    if not root.is_relative_to(REPO / "outputs"):
        raise ValueError("case outputs must be below repository outputs")
    dest = root / "simulation" / f"{checkpoint or source}_{variant_name}" / f"{record['name']}_s{sampling_seed}"
    dest.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        schema_version="s2a.position-execution.v1",
        implementation_origin="stability_recovery_simulation.py at d8421f9; explicit position/hold adapter version",
        execution_sha256=signatures,
        experiment=config,
        runtime_config=c,
        record=record,
        source=source,
        variant=variant,
        inference_latency_excluded=True,
        oracle_is_wam_success_evidence=False,
        raw_checkpoint_reference=case["variant"] == "raw",
        reference_intervention=case["variant"],
        development_case=case,
        physical_control_admission=admission,
        online_control_inputs="current measured TCP/twist, motor target, aperture/history and generated references",
        object_contact_and_task_predicate="metrics only; never adapter/governor inputs",
        oracle_reference_mode="fixed_world_chunks_selected_by_current_physical_step" if source == "oracle" else None,
        oracle_qualification_scope=(
            "privileged physical-step indexed chunks; differs from prior one-shot full600 reference qualification"
            if source == "oracle"
            else None
        ),
        environment={k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "MUJOCO_GL")},
        git_commit_metadata=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
    )
    expected_checkpoint = None
    if source == "wam":
        ck = next(ck for ck in config["checkpoints"] if ck["name"] == checkpoint)
        if ck["gpu_uuid"] != "GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f" or ck["port"] != 31365:
            raise ValueError("Registered237 UUID and HTTP port required")
        c["server_url"] = f"http://127.0.0.1:{ck['port']}"
        with requests.Session() as session:
            session.trust_env = False
            response = session.get(c["server_url"] + "/info", timeout=15)
            response.raise_for_status()
            info = response.json()
        expected_checkpoint = Path(ck["run"]) / "checkpoints" / ck["iteration"] / "model"
        if (
            Path(info["checkpoint_path"]).resolve() != expected_checkpoint.resolve()
            or info["ema_weights"]
            or not info["acceptance_intervention"]["active_slot_mask"]
        ):
            raise ValueError("wrong model or missing inactive-slot mask")
        identity = info["clean_training"]
        if (
            info.get("ready") is not True
            or info.get("current_case_id") != case["case_id"]
            or identity["gpu_uuid"] != ck["gpu_uuid"]
            or identity["variant"] != case["variant"]
            or any(
                identity[key] != development[key]
                for key in ("prepared_ref", "execution_registration_ref", "service_source_ref", "readout_ref")
            )
        ):
            raise ValueError("Registered WAM service/case/source identity differs")
        manifest["server_info"] = info
    else:
        if checkpoint is not None:
            raise ValueError("oracle cannot be labeled a WAM checkpoint")
        reference, reference_opening, oracle_signatures = oracle_reference
        signatures.update(oracle_signatures)
    h = np.asarray(json.loads(Path(c["tool_contract"]).read_text())["source_from_tcp"])
    h_inv, dt = inverse(h), 1 / c["control_fps"]
    if c["control_fps"] != 20:
        raise ValueError("Frozen residual inverse requires20Hz native cycles")
    write_json(dest / "manifest.json", manifest)
    env = None
    rows, chunks, failure = [], [], None
    start = time.monotonic()
    try:
        env, _ = make_environment(c, dest, record, 0)
        obs = current_observation(env)
        controller, contract = controller_contract(env)
        q = source_pose(obs)[:3, :3].T @ np.asarray(env.sim.data.get_site_xmat(controller.eef_name))
        if not np.isclose(env.env.control_timestep, dt):
            raise ValueError("control interval differs from declared model time")
        interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
        converter = ExecutedCommandResidualOSC(
            interface,
            contract["kp"],
            contract["kd"],
            dt,
            compensate=True,
            inverse_weight=config["inverse_weight"],
            max_command_rate=config["max_command_rate"],
        )
        for key in ("kp", "kd"):
            np.testing.assert_allclose(contract[key], arm_parameters[key], rtol=0, atol=1e-12)
        assert contract["control_freq"] == 20 and config["max_command_rate"] == 5
        initial_site = np.eye(4)
        initial_site[:3, :3] = np.asarray(env.sim.data.get_site_xmat(controller.eef_name))
        initial_site[:3, 3] = np.asarray(env.sim.data.get_site_xpos(controller.eef_name))
        source_from_site = inverse(source_pose(obs)) @ initial_site
        initial_opening = aperture(obs)
        loaded_controller = LoadedVelocityControl(
            response_parameters, initial_opening, velocity_envelope=config["gripper_velocity_envelope"]
        )
        native_initial_motor = np.asarray(env.robots[0].gripper.current_action).copy()
        env.robots[0].gripper.current_action = preserve_motor_target(native_initial_motor)
        gripper = make_gripper(variant, initial_opening, config)
        governor = RecoveringReferenceGovernor(config=ReferenceGovernorConfig(**config["reference_governor"]))
        initial_objects = object_progress(env)["objects"]
        simulator_signatures = simulator_inputs(env)
        verify_inputs(signatures)
        signatures.update(simulator_signatures)
        manifest["execution_sha256"] = signatures
        manifest["simulator_source_sha256"] = simulator_signatures
        model_xml = env.sim.model.get_xml()
        (dest / "model.xml").write_text(model_xml)
        manifest["initial_model_xml_sha256"] = sha256_bytes(model_xml.encode())
        manifest["package_versions"] = {}
        for package in ("numpy", "scipy", "pyarrow", "requests", "mujoco", "robosuite", "torch", "pillow"):
            try:
                manifest["package_versions"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                manifest["package_versions"][package] = None
        write_json(dest / "manifest.json", manifest)
        write_json(
            dest / "controller.json",
            dict(
                contract=contract,
                source_from_control_rotation=q,
                native_initial_motor_target=native_initial_motor,
                aligned_initial_motor_target=env.robots[0].gripper.current_action.copy(),
                motor_initialization="preserve current target; scalar reset broadcast only",
                gripper_config=asdict(gripper.config),
                gripper_algorithm="081 native-close loading and frozen079 tracking",
                arm_inverse_algorithm="247 four-step pose preview;explicit reset;native goal commit;173 SOURCE commit",
                source_from_site=source_from_site,
                gripper_velocity_envelope=config["gripper_velocity_envelope"],
                actual_initial_physics_sha256=sha256_bytes(np.asarray(env.sim.get_state().flatten()).tobytes()),
            ),
        )
        endpoint_observer(env, 0)
        previous_gripper_command = 0.0
        targets, openings, index, chunk_step, governed = None, None, 0, 0, None
        with (dest / "steps.jsonl").open("x") as stream:
            for step in range(config["online_max_steps"]):
                measured, opening = source_pose(obs), aperture(obs)
                measured_tcp = measured @ h
                need_query = targets is None
                if not need_query:
                    need_query = (
                        bool(governed["complete"]) if variant["governor"] else index >= len(targets)
                    ) or step - chunk_step >= config["max_observation_interval_steps"]
                if need_query:
                    before = np.asarray(env.sim.get_state().flatten()).copy()
                    before_time, query_start = float(env.sim.data.time), time.monotonic()
                    query_anchor, query_opening = measured_tcp.copy(), opening
                    identity = None
                    if source == "wam":
                        raw, image_sha = generate(
                            c, render_current_cameras(env, c["render_resolution"]), dest, len(chunks)
                        )
                        response_path = dest / f"response_{len(chunks):04d}.json"
                        response = json.loads(response_path.read_text())
                        request = json.loads((dest / f"request_{len(chunks):04d}.json").read_text())
                        authenticated, identity = validate_preview_wam_response(
                            response,
                            request,
                            case=case,
                            request_index=len(chunks),
                            checkpoint_path=expected_checkpoint,
                            readout_ref=development["readout_ref"],
                        )
                        if not np.array_equal(raw, authenticated):
                            raise ValueError("decoded generator action differs from saved response")
                        effective = attenuate_wam_rotation(raw, alpha=variant["model_rotation_alpha"])
                    else:
                        end = min(step + config["online_chunk_size"], len(reference))
                        fixed_targets = reference[step:end] @ h
                        fixed_openings = reference_opening[step:end]
                        raw = encode_oracle_targets(query_anchor, fixed_targets, fixed_openings)
                        effective, image_sha = raw.copy(), None
                    full_raw_targets, full_openings = decode_action_targets(query_anchor, raw, horizon=len(raw))
                    full_effective_targets, _ = decode_action_targets(query_anchor, effective, horizon=len(effective))
                    count = min(config["online_execution_horizon"], len(raw))
                    raw_targets, targets, openings = (
                        full_raw_targets[:count],
                        full_effective_targets[:count],
                        full_openings[:count],
                    )
                    after = np.asarray(env.sim.get_state().flatten()).copy()
                    after_time = float(env.sim.data.time)
                    if after_time != before_time or not np.array_equal(before, after):
                        raise RuntimeError("physics advanced while generating/transforming a reference")
                    summary = None
                    if variant["governor"]:
                        summary = governor.set_reference(
                            query_anchor, targets, openings, anchor_opening=opening, model_dt=dt
                        )
                    index, chunk_step = 0, step
                    chunks.append(
                        dict(
                            step=step,
                            anchor_tcp=query_anchor,
                            anchor_opening=query_opening,
                            raw128=raw,
                            effective128=effective,
                            raw_targets_tcp=full_raw_targets,
                            effective_targets_tcp=full_effective_targets,
                            openings=full_openings,
                            execution_endpoint_count=count,
                            model_rotation_alpha=variant["model_rotation_alpha"],
                            response_identity=identity,
                            request_sha256=sha256(dest / f"request_{len(chunks):04d}.json")
                            if source == "wam"
                            else None,
                            response_sha256=sha256(response_path) if source == "wam" else None,
                            image_sha256=image_sha,
                            physics_before_sha256=sha256_bytes(before.tobytes()),
                            physics_after_sha256=sha256_bytes(after.tobytes()),
                            physics_unchanged=True,
                            physics_before_time_s=before_time,
                            physics_after_time_s=after_time,
                            generation_s=time.monotonic() - query_start,
                            retiming=summary,
                        )
                    )
                    write_json(dest / "chunks.json", chunks)
                if variant["governor"]:
                    governed = governor.advance(measured_tcp, dt)
                    desired_tcp, desired_opening = governed["target_tcp"], governed["opening"]
                    model_time = governed["model_time_s"]
                else:
                    desired_tcp, desired_opening = targets[index], float(openings[index])
                    index += 1
                    model_time = index * dt
                raw_desired, raw_opening = sample_model_reference(
                    query_anchor,
                    raw_targets,
                    openings,
                    model_time_s=model_time,
                    dt=dt,
                    anchor_opening=query_opening,
                )
                effective_desired, effective_opening = sample_model_reference(
                    query_anchor,
                    targets,
                    openings,
                    model_time_s=model_time,
                    dt=dt,
                    anchor_opening=query_opening,
                )
                desired = desired_tcp @ h_inv
                twist = measured_twist(env)
                goal_before, goal_pos_before = (
                    np.asarray(controller.goal_ori).copy(),
                    np.asarray(controller.goal_pos).copy(),
                )
                velocity_reference = reference_linear_velocity(
                    full_effective_targets @ h_inv, index - 1, query_anchor @ h_inv, dt
                )
                converted = converter.convert(
                    measured, desired, twist, desired_linear_velocity=velocity_reference["velocity"]
                )
                motor_before = np.asarray(env.robots[0].gripper.current_action).copy()
                motor_opening = panda_opening_from_motor_target(motor_before)
                gripper_velocity = np.asarray(obs["robot0_gripper_qvel"], dtype=float).copy()
                direct_velocity = np.array([env.sim.data.qvel[i] for i in env.robots[0]._ref_gripper_joint_vel_indexes])
                np.testing.assert_allclose(gripper_velocity, direct_velocity, rtol=0, atol=1e-12)
                period_velocity = dt * (gripper_velocity[0] - gripper_velocity[1]) / 0.079
                grip_result = loaded_controller.update(
                    opening,
                    motor_before,
                    period_velocity,
                    full_effective_targets,
                    full_openings,
                    index - 1,
                    measured_tcp,
                    previous_command=previous_gripper_command,
                    dt=dt,
                )
                grip_command = grip_result.command
                provisional_command = np.r_[converted["command"], grip_command]
                model_input, current_wrench = current_model_state(
                    env,
                    provisional_command,
                    previous_actual_command,
                    previous_available=bool(step),
                    history=response_history,
                )
                current_site = model_input["x_site_pose"][0]
                np.testing.assert_allclose(current_site, measured @ source_from_site, rtol=0, atol=1e-8)
                desired_site = desired @ source_from_site
                site_velocity = reference_linear_velocity(
                    full_effective_targets @ h_inv @ source_from_site,
                    index - 1,
                    query_anchor @ h_inv @ source_from_site,
                    dt,
                )
                initial_dynamics = read_current_dynamics(env, model_input, provisional_command)
                site_twist = reference_world_twist(
                    full_effective_targets @ h_inv @ source_from_site,
                    index - 1,
                    query_anchor @ h_inv @ source_from_site,
                    dt,
                )
                np.testing.assert_allclose(site_twist["velocity"], site_velocity["velocity"], atol=1e-12, rtol=0)
                coupled_before = coupled_history.current(current_site, model_input["x_site_twist"][0])
                coupled_period_before = coupled_history.completed_periods
                solve_start = time.monotonic()
                solved = solve_preview(
                    initial_dynamics["mass"],
                    initial_dynamics["jacobian"],
                    current_site,
                    model_input["x_site_twist"][0],
                    full_effective_targets @ h_inv @ source_from_site,
                    query_anchor @ h_inv @ source_from_site,
                    index - 1,
                    previous_actual_command[:6],
                    contract,
                    innovation=coupled_before,
                )
                solve_seconds = time.monotonic() - solve_start
                command = provisional_command.copy()
                command[:6] = solved["command"]
                dynamics = read_current_dynamics(env, model_input, command)
                for key in ("mass", "jacobian", "robot_joint_position", "robot_joint_velocity"):
                    np.testing.assert_array_equal(dynamics[key], initial_dynamics[key])
                coupled_model = build_response(
                    dynamics["mass"],
                    dynamics["jacobian"],
                    dynamics["kp"],
                    dynamics["kd"],
                    dynamics["relative_rotation_goal"],
                )
                coupled_commit = coupled_history.commit_executed(
                    current_site, model_input["x_site_twist"][0], coupled_model, command[:3], contract
                )
                continuous_goal = (
                    model_input["x_previous_goal_ori"][0]
                    if solved["goal_mode"] == "latch"
                    else Rotation.from_rotvec(solved["physical_delta"][3:]).as_matrix() @ current_site[:3, :3]
                )
                np.testing.assert_allclose(dynamics["actual_rotation_goal"], continuous_goal, atol=1e-6, rtol=0)
                native_difference = np.abs(solved["nominal_response"] - coupled_commit["nominal_response"])
                goal_comparison = dict(
                    continuous_goal=continuous_goal,
                    native_goal_max_difference=float(
                        np.max(np.abs(dynamics["actual_rotation_goal"] - continuous_goal))
                    ),
                    native_response_block_max_difference=[
                        float(np.max(native_difference[part]))
                        for part in (slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12))
                    ],
                    response_block_units=["m", "rad", "m/s", "rad/s"],
                )
                physical_response_residual = converted["response_residual"]
                physical_response_commit = converter.commit_executed(measured, twist, command[:6])
                model_input["action_command"][0] = command
                model_input["action_goal_delta"][0] = np.r_[
                    solved["physical_delta"][:3], Rotation.from_matrix(dynamics["relative_rotation_goal"]).as_rotvec()
                ]
                converted["command"] = command[:6].copy()
                converted["response_residual"] = dict(
                    innovation=model_input["x_response_residual"][0].copy(),
                    completed_periods=response_history.completed_periods,
                )
                actual_nominal = arm_nominal(model_input, arm_parameters)[0]
                converted["nominal_predicted_motion"] = actual_nominal[:6].copy()
                converted["nominal_predicted_linear_velocity"] = actual_nominal[6:9] / dt
                converted["predicted_motion"] = np.r_[
                    solved["predicted_pose"][:3, 3] - current_site[:3, 3],
                    Rotation.from_matrix(solved["predicted_pose"][:3, :3] @ current_site[:3, :3].T).as_rotvec(),
                ]
                converted["predicted_linear_velocity"] = solved["predicted_twist"][:3].copy()
                converted["sent_delta"] = interface.decode(command[:6])
                converted["saturated"] = bool(
                    np.any(np.isclose(command[:6], contract["input_min"], atol=1e-10))
                    or np.any(np.isclose(command[:6], contract["input_max"], atol=1e-10))
                )
                converted["rate_limited"] = bool(
                    np.any(np.isclose(np.abs(command[:6] - previous_actual_command[:6]), 0.25, atol=1e-10))
                )
                gripper_qpos_before = np.asarray(obs["robot0_gripper_qpos"]).copy()
                before_time = float(env.sim.data.time)
                env.step(command)
                obs = current_observation(env)
                after_pose = source_pose(obs)
                after_site = np.eye(4)
                after_site[:3, :3] = np.asarray(env.sim.data.get_site_xmat(controller.eef_name))
                after_site[:3, 3] = np.asarray(env.sim.data.get_site_xpos(controller.eef_name))
                response_history.observe_completed(model_input, after_site, measured_twist(env), arm_parameters)
                coupled_after = coupled_history.observe_completed(after_site, measured_twist(env))
                previous_actual_command = command.copy()
                progress = object_progress(env)
                for name, obj in progress["objects"].items():
                    obj["height_gain_m"] = float(obj["position"][2] - initial_objects[name]["position"][2])
                row = dict(
                    qualified_forward_input=model_input,
                    coupled_dynamics=dynamics,
                    coupled_response_model=coupled_model,
                    pose_preview_inverse=solved,
                    pose_preview_solver_seconds=solve_seconds,
                    pose_preview_goal_comparison=goal_comparison,
                    provisional_source_command=provisional_command[:6].copy(),
                    coupled_response_history=dict(
                        innovation_before=coupled_before,
                        completed_periods_before=coupled_period_before,
                        commit=coupled_commit,
                        innovation_after=coupled_after,
                        completed_periods_after=coupled_history.completed_periods,
                    ),
                    nominal_model_kind="150 original uncoupled nominal;coupled nominal recorded separately",
                    prediction_kind="247 tangent preview;continuous first-command forecast;native nominal separately",
                    physical_response_residual=physical_response_residual,
                    physical_response_commit=physical_response_commit,
                    current_wrench=current_wrench,
                    desired_site=desired_site,
                    site_velocity_reference=site_velocity,
                    site_twist_reference=site_twist,
                    step=step,
                    chunk_index=len(chunks) - 1,
                    chunk_physical_step=step - chunk_step,
                    physical_time_before_s=before_time,
                    physical_time_after_s=float(env.sim.data.time),
                    measured_source=measured,
                    measured_twist=twist,
                    linear_velocity_reference=velocity_reference,
                    predicted_linear_velocity=converted["predicted_linear_velocity"],
                    nominal_predicted_linear_velocity=converted["nominal_predicted_linear_velocity"],
                    nominal_predicted_motion=converted["nominal_predicted_motion"],
                    response_residual=converted["response_residual"],
                    desired_source=desired,
                    desired_tcp=desired_tcp,
                    raw_desired_tcp=raw_desired,
                    effective_desired_tcp=effective_desired,
                    raw_opening_at_model_time=raw_opening,
                    effective_opening_at_model_time=effective_opening,
                    model_time_s=model_time,
                    retimed_clock_s=governed["retimed_clock_s"] if governed else model_time,
                    bridging=bool(governed["bridging"]) if governed else False,
                    bridge_recovery_advanced=bool(governed["bridge_recovery_advanced"]) if governed else False,
                    after_source=after_pose,
                    opening_target=desired_opening,
                    opening_before=opening,
                    opening_after=aperture(obs),
                    gripper_qpos_before=gripper_qpos_before,
                    gripper_qpos_after=np.asarray(obs["robot0_gripper_qpos"]).copy(),
                    motor_opening_before=motor_opening,
                    motor_opening_after=panda_opening_from_motor_target(env.robots[0].gripper.current_action),
                    motor_target_before=motor_before,
                    motor_target_after=env.robots[0].gripper.current_action.copy(),
                    measured_gripper_qvel=gripper_velocity,
                    measured_period_velocity=period_velocity,
                    actual_gripper_actuator_control=env.sim.data.ctrl[
                        env.robots[0]._ref_joint_gripper_actuator_indexes
                    ].copy(),
                    controller_goal_before=goal_before,
                    controller_goal_after=np.asarray(controller.goal_ori).copy(),
                    controller_goal_position_before=goal_pos_before,
                    controller_goal_position_after=np.asarray(controller.goal_pos).copy(),
                    controller_goal_source_rotation_before=goal_before @ q.T,
                    command=command,
                    actual_dt=float(env.sim.data.time) - before_time,
                    source_error=pose_error(desired, after_pose),
                    tcp_error=pose_error(desired_tcp, after_pose @ h),
                    raw_model_tcp_error=pose_error(raw_desired, after_pose @ h),
                    effective_model_tcp_error=pose_error(effective_desired, after_pose @ h),
                    reference_progress=governed,
                    gripper=asdict(grip_result) if grip_result else None,
                    saturated=converted["saturated"],
                    rate_limited=converted["rate_limited"],
                    **progress,
                )
                rows.append(row)
                endpoint_observer(env, step + 1)
                previous_gripper_command = float(grip_command)
                stream.write(json.dumps(row, default=lambda value: value.tolist(), allow_nan=False) + "\n")
                if step % 20 == 19:
                    stream.flush()
                    status = dict(steps=step + 1, chunks=len(chunks), ever_success=any(r["success"] for r in rows))
                    write_json(dest / "progress.json", status)
                    print(json.dumps(dict(record=record["name"], variant=variant_name, **status)), flush=True)
    except BaseException as exc:
        failure = repr(exc)
    finally:
        if env is not None:
            env.close()
    input_failure = None
    try:
        verify_inputs(signatures)
    except BaseException as exc:
        input_failure = repr(exc)
        failure = f"{failure + '; ' if failure else ''}{exc!r}"
    complete = len(rows) == config["online_max_steps"] and len(chunks) == 75 and failure is None
    outcome = (
        task_outcome(
            [r["success"] for r in rows],
            expected_steps=config["online_max_steps"],
            stable_steps=config["stable_success_last_steps"],
        )
        if complete
        else None
    )
    result = dict(
        record=record["name"],
        task=record["task"],
        source=source,
        checkpoint=checkpoint,
        variant=variant_name,
        sampling_seed=sampling_seed,
        model_rotation_alpha=variant["model_rotation_alpha"],
        raw_checkpoint_reference=manifest["raw_checkpoint_reference"],
        steps=len(rows),
        expected_steps=config["online_max_steps"],
        error=failure,
        complete=complete,
        chunks=len(chunks),
        elapsed_s=time.monotonic() - start,
        inference_latency_excluded=True,
        oracle_is_wam_success_evidence=False,
        stable_success=bool(outcome and outcome["stable"]),
        ever_success=any(r["success"] for r in rows),
        task_outcome=outcome,
        execution_inputs_unchanged=input_failure is None,
    )
    if rows:
        errors = np.asarray([r["tcp_error"] for r in rows])
        result.update(
            tcp_position_rmse_m=float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
            rotation_rmse_rad=float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
            opening_mae=float(np.mean([abs(r["opening_after"] - r["opening_target"]) for r in rows])),
            first_success_step=next((r["step"] for r in rows if r["success"]), None),
            final_success_steps=sum(r["success"] for r in rows[-config["stable_success_last_steps"] :]),
            physical_elapsed_s=sum(r["actual_dt"] for r in rows),
        )
    write_json(dest / "result.json", result)
    if failure:
        raise RuntimeError(failure)
    return result
