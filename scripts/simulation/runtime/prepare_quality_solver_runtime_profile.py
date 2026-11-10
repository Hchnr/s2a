"""Bind offline solver timing to saved causal states and generated references."""

import hashlib
from pathlib import Path

import numpy as np

from s2a.controllers.osc import OSCInterface
from s2a.geometry import inverse, validate_transform
from s2a.infrastructure.publication import result_json
from s2a.references.generated_twist import reference_world_twist
from scripts.simulation.runtime import bind_causal_event_bridge_factorial as binding
from scripts.simulation.runtime import compare_multicase_emergency_retreat as saved_chunk
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/quality_solver_runtime_profile_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/quality_solver_runtime_profile_20261111.md"
RECOVERY_RESULTS = {
    "producer": {
        "path": str(REPO / "outputs/causal_event_bridge_factorial_recovery_20261111/producer/result.json"),
        "sha256": "1693239e64068685806efdb46132322c051d2063ece86c9aaed12991c75cbd69",
    },
    "checker": {
        "path": str(REPO / "outputs/causal_event_bridge_factorial_recovery_20261111/checker/result.json"),
        "sha256": "ca0cbc4345a8800695fc98c17eac78c170fd44063788f6638b27997686a28406",
    },
}


def source_refs():
    return binding.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            CONFIG,
            PLAN,
            REPO / "src/s2a/research/trajectory_search/quality_capped_inner_joint_plan.py",
            REPO / "src/s2a/research/joint_control/preview/interior_joint_preview.py",
            REPO / "src/s2a/research/joint_control/tracking/early_quality_diagnostic.py",
            REPO / "src/s2a/research/response_models/forward/joint_limit_response.py",
            REPO / "tests/research/native_runtime/test_quality_solver_runtime_inputs.py",
        )
    ]


def read_config():
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    prefix.require(
        config
        == {
            "schema": "s2a.quality-solver-runtime-profile.v1",
            "case_index": 5,
            "case_id": "task1_init0_s42_terminal058",
            "seed": 42,
            "steps": [352, 355, 359],
            "reference_step": 352,
            "plan_horizon": 8,
            "control_dt_s": 0.05,
            "inner_dt_s": 0.002,
            "null_kp": 10.0,
            "past_inner_error": "zero_diagnostic_only",
            "solver_calls_per_sample": 2,
            "max_solver_calls": 6,
            "max_response_evaluations_per_call": 160,
            "max_response_evaluations": 960,
            "max_independent_audits": 3,
            "max_workers": 3,
            "worker_timeout_seconds": 600,
            "physical_steps": 0,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
        "Exact registered causal timing samples and budget",
    )
    return ref, config


def prepare_sample(case, row, previous_row, chunk, runtime, step, config):
    """Use only the saved state, last completed action and generated targets."""
    horizon, offset = config["plan_horizon"], step - config["reference_step"]
    prefix.require(
        step in config["steps"]
        and row["step"] == step
        and previous_row["step"] == step - 1
        and row["chunk_index"] == step // 8
        and row["chunk_physical_step"] == step % 8
        and chunk["step"] == config["reference_step"]
        and 0 <= offset <= 40 - horizon,
        "Saved current state and already generated reference time",
    )
    target = validate_transform(chunk["effective_targets_tcp"])
    anchor = validate_transform(chunk["anchor_tcp"])
    prefix.require(target.shape == (40, 4, 4) and anchor.shape == (4, 4), "Complete generated TCP reference")
    dynamics, saved = row["coupled_dynamics"], row["qualified_forward_input"]
    q, velocity = np.asarray(dynamics["robot_joint_position"]), np.asarray(dynamics["robot_joint_velocity"])
    np.testing.assert_array_equal(q, saved["x_joint_position"][0])
    np.testing.assert_array_equal(velocity, saved["x_joint_velocity"][0])
    previous = np.asarray(saved["x_previous_command"][0], dtype=float)
    np.testing.assert_array_equal(previous, previous_row["command"])
    seed = np.asarray(row["tracking_refinement"]["planned_commands"], dtype=float)
    contract = case["controller"]["contract"]
    interface = OSCInterface(*(contract[key] for key in ("input_min", "input_max", "output_min", "output_max")))
    prefix.require(
        seed.shape == (horizon, 6)
        and previous.shape == (7,)
        and np.isfinite(seed).all()
        and np.max(np.abs(np.diff(np.vstack((previous[:6], seed)), axis=0))) <= 0.25 + 1e-12
        and np.all(seed >= interface.input_min - 1e-12)
        and np.all(seed <= interface.input_max + 1e-12)
        and all(np.any(interface.decode(command)[3:] != 0) for command in seed),
        "Original bounded seed and actual previous command",
    )
    site = validate_transform(saved["x_site_pose"][0])
    tool = inverse(np.asarray(case["controller"]["source_from_site"])) @ np.asarray(case["tool"])
    limits = np.asarray(runtime["final_prefix_inner"]["limits"], dtype=float)
    initial_joint = np.asarray(runtime["runtime_initial_joint"], dtype=float)
    prefix.require(
        limits.shape == (7, 2)
        and initial_joint.shape == (7,)
        and np.isfinite(limits).all()
        and np.isfinite(initial_joint).all(),
        "Fixed robot bounds and replay-runtime initial joint",
    )
    return {
        "step": step,
        "reference_step": chunk["step"],
        "reference_offset": offset,
        "mass": np.asarray(dynamics["mass"], dtype=float),
        "jacobian": np.asarray(dynamics["jacobian"], dtype=float),
        "kp": np.asarray(dynamics["kp"], dtype=float),
        "kd": np.asarray(dynamics["kd"], dtype=float),
        "q": q,
        "velocity": velocity,
        "initial_joint": initial_joint,
        "seed": seed,
        "previous": previous[:6].copy(),
        "site": site,
        "site_from_tcp": tool,
        "targets": target[offset : offset + horizon].copy(),
        "twists": np.asarray(
            [reference_world_twist(target, index, anchor, 0.05)["twist"] for index in range(offset, offset + horizon)]
        ),
        "lower": limits[:, 0].copy(),
        "upper": limits[:, 1].copy(),
        "contract": {key: contract[key] for key in ("input_min", "input_max", "output_min", "output_max")},
        "past_inner_error_rad": np.zeros((25, 7)),
    }


def sample_fingerprint(sample):
    return hashlib.sha256(result_json(sample).encode()).hexdigest()


def load_inputs():
    config_ref, config = read_config()
    original = binding.load_inputs()
    case = original["case"]
    prefix.require(
        case["spec"]["case_index"] == config["case_index"]
        and case["spec"]["case_id"] == config["case_id"]
        and case["config"]["seed"] == config["seed"],
        "Exact saved development case and seed",
    )
    refs = dict(original["source_sha256"])
    refs[config_ref["path"]] = config_ref["sha256"]
    runtimes = []
    for phase, ref in RECOVERY_RESULTS.items():
        result = prefix.read(ref, refs)
        prefix.require(
            result["complete"] is result["passed"] is result["sources_unchanged"] is True and result["phase"] == phase,
            "Both factorial phases and physical prefix replay closed",
        )
        for mode in binding.MODES:
            branch = prefix.read(result["branch_refs"][mode], refs)
            prefix.require(
                branch["complete"] is branch["passed"] is True and branch["runtime"]["previous_verified_steps"] == 8,
                "Reproduced runtime bound to the registered case",
            )
            runtimes.append(branch["runtime"])
    for runtime in runtimes[1:]:
        np.testing.assert_array_equal(runtime["runtime_initial_joint"], runtimes[0]["runtime_initial_joint"])
        np.testing.assert_array_equal(
            runtime["final_prefix_inner"]["limits"], runtimes[0]["final_prefix_inner"]["limits"]
        )
    chunk_ref = case["gripper_inputs"]["chunks_ref"]
    chunk = saved_chunk.read_array_item(prefix.checked(chunk_ref, refs), config["reference_step"] // 8)
    samples = {
        str(step): prepare_sample(case, case["rows"][step], case["rows"][step - 1], chunk, runtimes[0], step, config)
        for step in config["steps"]
    }
    return {
        "config_ref": config_ref,
        "config": config,
        "case": case["spec"],
        "source_sha256": refs,
        "asset_refs": original["asset_refs"],
        "samples": samples,
    }


def input_receipt(inputs):
    return {
        "config_ref": inputs["config_ref"],
        "case": inputs["case"],
        "source_sha256": inputs["source_sha256"],
        "asset_refs": inputs["asset_refs"],
        "sample_fingerprints": {step: sample_fingerprint(sample) for step, sample in inputs["samples"].items()},
    }
