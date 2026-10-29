"""Compare normalized TCP error-vector Gauss-Newton under the original hard gates."""

import argparse
import json
import os
import time

import numpy as np
from probe_tracking_v5_interior_trajectory_search import initial_model, run_search
from probe_tracking_v5_stage_budget_search import CONFIG as BASELINE_CONFIG
from probe_tracking_v5_stage_budget_search import load_inputs as load_prior_inputs
from probe_tracking_v5_terminal_only_tail_repair import load_environment_guard

from s2a.mujoco_robot_dynamics import MujocoRobotDynamics
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.osc import OSCInterface
from s2a.preview_result_publication import publish_result
from s2a.restored_anchor_sqp_evidence import (
    CPU_ENVIRONMENT,
    REPO,
    checked,
    finish,
    model_registration,
    read,
    reference,
    remember,
    require,
)
from s2a.state_vector_gn_search import (
    POSE_CEILING_POLICY,
    POSE_DAMPING,
    POSE_OBJECTIVE_POLICY,
    initial_error_vector_jacobian,
    search,
)

ROOT = REPO / "outputs/tracking_v5_state_vector_gn_search_20261029"
PLAN = REPO / "plans/v0.0.1/state_vector_gn_search_20261029.md"
CONFIG = REPO / "configs/state_vector_gn_search_20261029.json"
FEEDBACK_POLICY = "safety_qualified_pose_ceiling_rejection_minimum_alpha"
CLOSED_PRIOR_ANALYSIS = dict(
    path=str(REPO / "outputs/stage_budget_search_analysis_20261029/result.json"),
    sha256="5e6ceea748c2d346d76d2265ddfe0c19f99ef65056be1d21e2abe16ab13a0767",
)
DIAGNOSTIC_REF = dict(
    path=str(REPO / "outputs/state_vector_prediction_diagnostic_v2_20261029/result.json"),
    sha256="65250c91d0046e0a127e108220c29e6552fe0390473fbcdcd3f20d435c43f3a5",
)
DIAGNOSTIC_OWNER_REF = dict(
    path=str(REPO / "outputs/state_vector_prediction_diagnostic_v2_20261029/owner.json"),
    sha256="f8f43d19e950a2eba0b0405d28cc62fc5fd8b66dad8fb28e3bdb5a8d2c6dc693",
)
SOURCES = (
    "src/s2a/state_vector_gn_search.py",
    "scripts/state_vector_gn_search_audit.py",
    "scripts/probe_tracking_v5_state_vector_gn_search.py",
    "scripts/check_tracking_v5_state_vector_gn_search.py",
    "tests/test_state_vector_gn_search.py",
    "tests/test_state_vector_gn_search_audit.py",
    "tests/test_state_vector_gn_search_execution.py",
    "configs/state_vector_gn_search_20261029.json",
    "plans/v0.0.1/state_vector_gn_search_20261029.md",
)


def load_inputs():
    bundle, prior_config, refs = load_prior_inputs()
    prior = read(remember(CLOSED_PRIOR_ANALYSIS, refs))
    require(
        prior["complete"] is prior["passed"] is prior["sources_unchanged"] is True
        and prior["costs"]["variant"]["response_calls_per_route"] == 90
        and prior["costs"]["variant"]["stage_count"] == 60
        and prior["variant"]["witness_indices"] == []
        and prior["variant"]["selected_response_index"] is None
        and prior["variant"]["termination_reason"] == "stage_budget_exhausted",
        "Closed 60-stage search without an adopted trajectory",
    )
    for ref in prior["source_refs_unchanged"]:
        remember(ref, refs)
    diagnostic = read(remember(DIAGNOSTIC_REF, refs))
    owner = read(remember(DIAGNOSTIC_OWNER_REF, refs))
    require(
        diagnostic["complete"] is diagnostic["passed"] is diagnostic["sources_unchanged"] is True
        and all(type(value) is int and value == 0 for value in diagnostic["new_activity"].values())
        and diagnostic["control_qualified"] is False
        and owner["complete"] is owner["passed"] is owner["actual_wait"] is owner["reaped"] is True
        and type(owner["returncode"]) is int
        and owner["returncode"] == 0
        and owner["signal_sent"] is owner["automatic_retry"] is False
        and owner["source_unchanged"] is owner["input_hashes_unchanged"] is True
        and owner["result_ref"] == DIAGNOSTIC_REF,
        "Closed zero-activity saved diagnostic and actually reaped owner",
    )
    # Bind motivation evidence only: no diagnostic fit or old response initializes the model.
    for ref in diagnostic["source_refs_unchanged"]:
        remember(ref, refs)
    config = read(remember(reference(CONFIG), refs))
    require(
        type(config["max_stages"]) is int
        and config["max_stages"] == 60
        and config == prior_config
        and config["max_responses"] == 122
        and CONFIG.read_bytes() == BASELINE_CONFIG.read_bytes(),
        "All numeric gates and both budgets exactly match the closed60-stage baseline",
    )
    for path in SOURCES:
        remember(reference(REPO / path), refs)
    finish(refs)
    return bundle, config, refs


def main(dry=False):
    strict_native_environment()
    require(
        all(os.environ.get(key) == value for key, value in CPU_ENVIRONMENT.items()), "Single-thread CPU environment"
    )
    bundle, config, refs = load_inputs()
    source, source_reg, control = (bundle[key] for key in ("source", "source_registration", "control"))
    interface = OSCInterface(*(control[key] for key in ("input_min", "input_max", "output_min", "output_max")))
    Js, Jq = initial_model(bundle)
    Je = initial_error_vector_jacobian(bundle)
    model_reg, proof = model_registration(source_reg, refs)
    guard = load_environment_guard(refs)
    if dry:
        print(
            json.dumps(dict(passed=True, dry=True, qp_solves=0, responses=0, queries=0, source_count=len(refs))),
            flush=True,
        )
        return
    require(not ROOT.exists(), "Exclusive variant output; no overwrite or automatic retry")
    registration = publish_result(
        ROOT / "registration.json",
        dict(
            experiment="state_vector_gn_search",
            source_refs=[reference(REPO / path) for path in SOURCES],
            input_refs=list(refs.values()),
            plan_ref=reference(PLAN),
            config_ref=reference(CONFIG),
            config=config,
            feedback_policy=FEEDBACK_POLICY,
            pose_ceiling_policy=POSE_CEILING_POLICY,
            pose_objective_policy=POSE_OBJECTIVE_POLICY,
            pose_damping=POSE_DAMPING,
            diagnostic_ref=DIAGNOSTIC_REF,
            diagnostic_owner_ref=DIAGNOSTIC_OWNER_REF,
            historical_response_index=3557,
            response_budget=config["max_responses"],
            qp_budget=config["max_stages"],
            qp_iteration_budget=100 * config["max_stages"],
            seed=42,
            automatic_retry=False,
            closed_prior_analysis_ref=CLOSED_PRIOR_ANALYSIS,
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        experiment="state_vector_gn_search",
        registration_ref=registration,
        asset_loading_proof=proof,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        actual_consumed_reference_count=0,
        control_qualified=False,
        automatic_retry=False,
        prediction_calls=0,
        response_evaluations=0,
        unique_command_count=0,
        numerical_inner_steps=0,
        query_count=0,
        qp_solves=0,
        model_construction_attempts=0,
        model_constructions_completed=0,
        witness_indices=[],
        qualified_witness_indices=[],
        witness_found=False,
        answer=None,
    )
    started = time.monotonic()
    try:
        result["environment_admission"] = restore_cv2_environment(
            guard["exact_probe_ref"], guard["runtime_source_refs"]
        )
        os.environ.pop("MUJOCO_GL", None)
        os.environ.pop("PYOPENGL_PLATFORM", None)
        result["model_construction_attempts"] += 1
        dynamics = MujocoRobotDynamics(
            checked(model_reg["xml_ref"]),
            control["controller_site"],
            [f"robot0_joint{k}" for k in range(1, 8)],
            source["fixed_gripper_state"],
        )
        result["model_constructions_completed"] += 1
        require(dynamics.model.opt.timestep == 0.002, "Original2ms numerical model")
        with np.load(checked(source_reg["states_ref"]), allow_pickle=False) as archive:
            state = archive["sim_states"][368].copy()
        np.testing.assert_array_equal(source["current_q"], state[1 + dynamics.qids])
        np.testing.assert_array_equal(source["current_velocity"], state[1 + dynamics.model.nq + dynamics.dofs])
        np.testing.assert_array_equal(source["joint_limits"], dynamics.model.jnt_range[dynamics.joints])

        def vector_search(evaluate, historical, safety_jacobian, squared_jacobian, previous, lo, hi, cfg):
            return search(evaluate, historical, safety_jacobian, squared_jacobian, Je.copy(), previous, lo, hi, cfg)

        run_search(source, bundle["base"], Js, Jq, config, interface, dynamics, result, search_fn=vector_search)
        finish(refs)
        strict_native_environment()
        result.update(complete=True, passed=True, source_refs_unchanged=list(refs.values()))
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        if result["answer"] is not None:
            result["qp_solves"] = len(result["answer"]["stages"])
        result["partial_response_queries"] = max(
            0, result["query_count"] - int(result["query_count"] > 0) - 312 * result["response_evaluations"]
        )
        publish_result(ROOT / "verified.json", result)
    print(
        json.dumps(
            dict(
                passed=True,
                responses=result["response_evaluations"],
                witness_found=result["witness_found"],
                feedback_updates=result["answer"]["rejected_update_count"],
                termination=result["answer"]["termination_reason"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true")
    main(parser.parse_args().dry)
