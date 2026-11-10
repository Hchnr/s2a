"""Bind a free-rollout benchmark to previously profiled causal states."""

from pathlib import Path

from scripts.simulation.runtime import prepare_quality_solver_runtime_profile as saved
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/batched_joint_rollout_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/batched_joint_rollout_20261111.md"
PROFILE_RESULT = {
    "path": str(REPO / "outputs/quality_solver_runtime_profile_20261111/result.json"),
    "sha256": "12271b66e24e546702626bc884e0be294571d3411415312e4151c2e2b7952d0f",
}


def source_refs():
    return saved.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            CONFIG,
            PLAN,
            REPO / "src/s2a/research/joint_control/preview/batched_free_joint_rollout.py",
            REPO / "tests/research/joint_control/preview/test_batched_free_joint_rollout.py",
            REPO / "tests/research/native_runtime/test_batched_joint_rollout_inputs.py",
        )
    ]


def read_config():
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    prefix.require(
        config
        == {
            "schema": "s2a.batched-joint-rollout.v1",
            "case_index": 5,
            "case_id": "task1_init0_s42_terminal058",
            "seed": 42,
            "steps": [352, 355, 359],
            "plan_horizon": 8,
            "inner_steps": 25,
            "inner_dt_s": 0.002,
            "null_kp": 10.0,
            "enforce_limits": False,
            "comparison_atol": 1e-10,
            "comparison_rtol": 0,
            "warmup_calls_per_implementation": 1,
            "timed_calls_per_implementation": 50,
            "max_rollout_calls": 306,
            "max_workers": 3,
            "worker_timeout_seconds": 600,
            "solver_calls": 0,
            "physical_steps": 0,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
        "Exact registered unprojected rollout comparison",
    )
    return ref, config


def load_inputs():
    config_ref, config = read_config()
    inputs = saved.load_inputs()
    prefix.require(
        inputs["case"]["case_index"] == config["case_index"]
        and inputs["case"]["case_id"] == config["case_id"]
        and inputs["config"]["seed"] == config["seed"]
        and inputs["config"]["steps"] == config["steps"],
        "Same previously profiled states and seed",
    )
    refs = dict(inputs["source_sha256"])
    refs[config_ref["path"]] = config_ref["sha256"]
    profile = prefix.read(PROFILE_RESULT, refs)
    prefix.require(
        profile["complete"] is profile["passed"] is profile["sources_unchanged"] is True
        and profile["counts"]["solver_calls_completed"] == 6
        and profile["physical_steps"] == profile["world_model_calls"] == profile["fresh_cases"] == 0,
        "Prior causal quality profile completed without physical execution",
    )
    for step in config["steps"]:
        sample = prefix.read(profile["sample_refs"][str(step)], refs)
        prefix.require(
            sample["complete"] is sample["passed"] is True
            and sample["data"]["sample_sha256"] == saved.sample_fingerprint(inputs["samples"][str(step)]),
            "Each benchmark sample is identical to the profiled causal input",
        )
    return {
        "config_ref": config_ref,
        "config": config,
        "case": inputs["case"],
        "source_sha256": refs,
        "asset_refs": inputs["asset_refs"],
        "samples": inputs["samples"],
    }


def input_receipt(inputs):
    return {
        "config_ref": inputs["config_ref"],
        "case": inputs["case"],
        "source_sha256": inputs["source_sha256"],
        "asset_refs": inputs["asset_refs"],
        "sample_fingerprints": {step: saved.sample_fingerprint(sample) for step, sample in inputs["samples"].items()},
    }
