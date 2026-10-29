"""Run the isolated safe-rejection feedback variant under the original budget."""

import argparse
import json
import os
import time

import numpy as np
from probe_tracking_v5_interior_trajectory_search import CONFIG as BASELINE_CONFIG
from probe_tracking_v5_interior_trajectory_search import initial_model, run_search
from probe_tracking_v5_interior_trajectory_search import load_inputs as load_baseline_inputs
from probe_tracking_v5_terminal_only_tail_repair import load_environment_guard

from s2a.mujoco_robot_dynamics import MujocoRobotDynamics
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.osc import OSCInterface
from s2a.preview_result_publication import publish_result
from s2a.rejected_response_search import search
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

ROOT = REPO / "outputs/tracking_v5_rejected_response_search_20261029"
PLAN = REPO / "plans/v0.0.1/rejected_response_search_20261029.md"
CONFIG = REPO / "configs/rejected_response_search_20261029.json"
FEEDBACK_POLICY = "safety_qualified_pose_ceiling_rejection_minimum_alpha"
BASELINE_ANALYSIS = dict(
    path=str(REPO / "outputs/interior_trajectory_search_analysis_20261029/result.json"),
    sha256="e80df4508072b2052f7864832e51bd2fd5c9f32e13495a4147e3be1571279a50",
)
DIAGNOSTIC = dict(
    path=str(REPO / "outputs/interior_trajectory_rejected_ray_diagnostic_20261029/result.json"),
    sha256="84e33538ee8b1ae49371b9b3331127102299855512a8c9f5a42f2d04fcaf0049",
)
SOURCES = (
    "src/s2a/rejected_response_search.py",
    "scripts/rejected_response_search_audit.py",
    "scripts/probe_tracking_v5_rejected_response_search.py",
    "scripts/check_tracking_v5_rejected_response_search.py",
    "tests/test_rejected_response_search.py",
    "tests/test_rejected_response_search_audit.py",
    "tests/test_rejected_response_search_execution.py",
    "configs/rejected_response_search_20261029.json",
    "plans/v0.0.1/rejected_response_search_20261029.md",
)


def load_inputs():
    bundle, baseline_config, refs = load_baseline_inputs()
    baseline = read(remember(BASELINE_ANALYSIS, refs))
    diagnostic = read(remember(DIAGNOSTIC, refs))
    require(
        baseline["complete"] is baseline["passed"] is baseline["sources_unchanged"] is True
        and baseline["total_response_calls"] == 88
        and baseline["witness_indices"] == [],
        "Closed baseline without an adopted trajectory",
    )
    require(
        diagnostic["complete"] is diagnostic["passed"] is True
        and diagnostic["experiment"] == "posthoc_rejected_ray_residual_diagnostic"
        and diagnostic["independent_dynamics_evidence_added"] is False
        and all(value == 0 for value in diagnostic["new_activity"].values()),
        "The diagnostic is saved-data evidence, not newly trained dynamics",
    )
    for ref in baseline["input_refs"].values():
        remember(ref, refs)
    config = read(remember(reference(CONFIG), refs))
    require(
        config == baseline_config and CONFIG.read_bytes() == BASELINE_CONFIG.read_bytes(),
        "Only the feedback policy changes; all numeric gates and budgets match the baseline",
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
            experiment="rejected_response_search",
            source_refs=[reference(REPO / path) for path in SOURCES],
            input_refs=list(refs.values()),
            plan_ref=reference(PLAN),
            config_ref=reference(CONFIG),
            config=config,
            feedback_policy=FEEDBACK_POLICY,
            historical_response_index=3557,
            response_budget=config["max_responses"],
            qp_budget=config["max_stages"],
            seed=42,
            automatic_retry=False,
            baseline_analysis_ref=BASELINE_ANALYSIS,
            diagnostic_ref=DIAGNOSTIC,
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        experiment="rejected_response_search",
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
        run_search(source, bundle["base"], Js, Jq, config, interface, dynamics, result, search_fn=search)
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
