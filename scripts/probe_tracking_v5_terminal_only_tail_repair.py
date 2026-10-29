"""Repair the archived terminal-only ray3543 with one fixed fourteen-call grid."""

import argparse
import json
import os
import time

import numpy as np
from probe_tracking_v5_tail_translation_reserve import run_probe

from s2a.directional_derivative_probe import _canonical
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
from s2a.tail_translation_reserve import build_tail_translation_grid
from s2a.terminal_development_execution import CONTROL_REF
from s2a.terminal_only_tail_repair import probe_terminal_only_tail_repair

ROOT = REPO / "outputs/tracking_v5_terminal_only_tail_repair_v2_20261029"
INPUT = REPO / "outputs/terminal_only_tail_repair_inputs_20261029/bundle.json"
PLAN = REPO / "plans/v0.0.1/terminal_only_tail_repair_20261029.md"
BUNDLE_SHA = "637d172935c3dd852b03347f1d9b97f863089dd85a637ca5f539c6a66b19ee26"
CLOSURE = dict(
    path=str(REPO / "outputs/restored_anchor_sqp_final_closure_20261029.json"),
    sha256="e8d6c7a95c1ec4c4ac20f59c529b66e7b744e0cd59bbb928f915d624e4f62fb4",
)
ANALYSIS = dict(
    path=str(REPO / "outputs/restored_anchor_sqp_causal_failure_analysis_20261029/result.json"),
    sha256="c9dd1bf9fce6a92b32936c0a68c066b64f5b426ad33fa35aeff8fe0db9cf1a6b",
)
SOURCES = (
    "src/s2a/terminal_only_tail_repair.py",
    "scripts/probe_tracking_v5_terminal_only_tail_repair.py",
    "scripts/check_tracking_v5_terminal_only_tail_repair.py",
    "tests/test_terminal_only_tail_repair.py",
    "tests/test_terminal_only_tail_repair_check.py",
    "tests/test_terminal_only_tail_repair_execution.py",
    "plans/v0.0.1/terminal_only_tail_repair_20261029.md",
)


def prepare_inputs():
    """Extract one complete saved ray once; no model or numerical response."""
    require(not INPUT.exists(), "Exclusive input extraction; never overwrite")
    closure, analysis = read(CLOSURE), read(ANALYSIS)
    require(
        closure["complete"] is closure["passed"] is closure["sources_unchanged"] is True
        and closure["verified_responses"] == 3657
        and closure["boundary_classification_disagreements"] == []
        and analysis["complete"] is analysis["passed"] is analysis["sources_unchanged"] is True
        and analysis["closure_ref"] == CLOSURE,
        "Closed full search and causal analysis",
    )
    rays = [r for r in analysis["analysis"]["rays"] if r["terminal_only_with_prefix_and_descent"]]
    ranked = sorted(rays, key=lambda r: (r["observed_q9_objective"], r["response_index"]))
    require(len(ranked) == 40 and ranked[0]["response_index"] == 3543, "Fixed pre-response selection")
    refs = {ref["path"]: ref for ref in closure["source_refs_unchanged"]}
    source_path = str(REPO / "outputs/tracking_v5_prefix_preserving_trajectory_control_20261028/verified.json")
    source_ref = refs[source_path]
    source = read(source_ref)
    source.pop("answer")
    source_reg = read(source["registration_ref"])
    # Memory is bounded by one archived producer; subsequent calls use the small bundle.
    producer = read(closure["producer_result_ref"])
    require(producer["complete"] is producer["passed"] is True, "Completed original producer")
    ray = producer["answer"]["phases"][50]["ray_attempts"][3]
    require(ray["response_index"] == 3543, "Exact complete historical ray locator")
    historical = ray["entry"]
    require(
        historical["prefix_length"] == 9
        and np.min(historical["safety_slack"][:-1]) >= 1e-8
        and historical["safety_slack"][-1] < 0,
        "Prefix9 with only terminal failing the original buffer",
    )
    control = read(source_reg["control_ref"])["contract"]
    interface = OSCInterface(*(control[k] for k in ("input_min", "input_max", "output_min", "output_max")))
    grid = build_tail_translation_grid(
        historical["commands"], source["previous_command"][:6], interface.input_min, interface.input_max
    )
    require(grid["admitted"], "All twelve fixed candidates admitted without new queries")
    bundle = dict(
        source=source,
        source_ref=source_ref,
        source_registration=source_reg,
        historical_anchor=historical,
        historical_response_index=3543,
        historical_locator="answer.phases[50].ray_attempts[3].entry",
        producer_ref=closure["producer_result_ref"],
        closure_ref=CLOSURE,
        analysis_ref=ANALYSIS,
        selection=ranked[0],
        control=control,
        grid=grid,
        input_refs=list(refs.values()),
        new_responses=0,
        new_queries=0,
    )
    finish(refs)
    print(json.dumps(publish_result(INPUT, bundle)), flush=True)


def load_inputs():
    """Bind the extracted input and the existing complete provenance once per stage."""
    require(isinstance(BUNDLE_SHA, str), "Input SHA must be fixed before execution")
    bundle = read(dict(path=str(INPUT), sha256=BUNDLE_SHA))
    require(bundle["historical_response_index"] == 3543, "Fixed historical candidate")
    refs = {ref["path"]: ref for ref in bundle["input_refs"]}
    for ref in (CLOSURE, ANALYSIS, dict(path=str(INPUT), sha256=BUNDLE_SHA)):
        remember(ref, refs)
    for path in SOURCES:
        remember(reference(REPO / path), refs)
    finish(refs)
    return bundle, refs


def load_environment_guard(refs):
    """The environment admission is separate from the controller parameter record."""
    guard = read(remember(CONTROL_REF, refs))["cv2_guard_registration"]
    require(isinstance(guard["runtime_source_refs"], list) and guard["runtime_source_refs"], "Runtime guard sources")
    for ref in (guard["exact_probe_ref"], *guard["runtime_source_refs"]):
        remember(ref, refs)
    return guard


def main(dry=False):
    strict_native_environment()
    require(all(os.environ.get(k) == v for k, v in CPU_ENVIRONMENT.items()), "Fixed single-thread CPU")
    bundle, refs = load_inputs()
    source, source_reg, control = (bundle[k] for k in ("source", "source_registration", "control"))
    interface = OSCInterface(*(control[k] for k in ("input_min", "input_max", "output_min", "output_max")))
    grid = build_tail_translation_grid(
        bundle["historical_anchor"]["commands"],
        source["previous_command"][:6],
        interface.input_min,
        interface.input_max,
    )
    require(_canonical(grid) == _canonical(bundle["grid"]) and grid["admitted"], "Exact preflight grid")
    model_reg, proof = model_registration(source_reg, refs)
    guard = load_environment_guard(refs)
    if dry:
        print(json.dumps(dict(passed=True, dry=True, responses=0, queries=0, source_count=len(refs))), flush=True)
        return
    require(not ROOT.exists(), "Exclusive output; no automatic retry")
    registration = publish_result(
        ROOT / "registration.json",
        dict(
            experiment="terminal_only_tail_repair",
            input_ref=dict(path=str(INPUT), sha256=BUNDLE_SHA),
            plan_ref=reference(PLAN),
            source_refs=[reference(REPO / p) for p in SOURCES],
            input_refs=list(refs.values()),
            grid=grid,
            historical_response_index=3543,
            response_budget=14,
            seed=42,
            automatic_retry=False,
        ),
    )
    result = dict(
        complete=False,
        passed=False,
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
        numerical_inner_steps=0,
        query_count=0,
        model_construction_attempts=0,
        model_constructions_completed=0,
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
        require(dynamics.model.opt.timestep == 0.002, "Original 2ms model")
        with np.load(checked(source_reg["states_ref"]), allow_pickle=False) as archive:
            state = archive["sim_states"][368].copy()
        np.testing.assert_array_equal(source["current_q"], state[1 + dynamics.qids])
        np.testing.assert_array_equal(source["current_velocity"], state[1 + dynamics.model.nq + dynamics.dofs])
        np.testing.assert_array_equal(source["joint_limits"], dynamics.model.jnt_range[dynamics.joints])
        run_probe(
            source,
            grid,
            bundle["historical_anchor"],
            None,
            interface,
            dynamics,
            probe_fn=probe_terminal_only_tail_repair,
            progress=result,
        )
        finish(refs)
        strict_native_environment()
        result.update(complete=True, passed=True, source_refs_unchanged=list(refs.values()))
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        completed = result.get("response_evaluations", 0)
        queries = result.get("query_count", 0)
        result["partial_response_queries"] = max(0, queries - int(queries > 0) - 312 * completed)
        result["numerical_inner_steps_scope"] = "Completed full responses; partial query cost recorded separately"
        publish_result(ROOT / "verified.json", result)
    print(
        json.dumps(
            dict(
                passed=True, witness_found=result["witness_found"], selected=result["answer"]["selected_response_index"]
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--dry", action="store_true")
    arguments = parser.parse_args()
    if arguments.prepare:
        prepare_inputs()
    else:
        main(arguments.dry)
