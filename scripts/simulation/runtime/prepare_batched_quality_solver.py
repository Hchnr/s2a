"""Bind the solver comparison to independently checked free-rollout benchmarks."""

from pathlib import Path

from scripts.simulation.runtime import prepare_batched_joint_rollout as saved
from scripts.simulation.runtime import prepare_quality_solver_runtime_profile as saved_samples
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/batched_quality_solver_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/batched_quality_solver_20261111.md"
BENCHMARK_RESULT = {
    "path": str(REPO / "outputs/batched_joint_rollout_20261111/result.json"),
    "sha256": "456d35ea3312bcc94336a6af0cae98572256a11c763fa4c261a30c7c5a6eb981",
}


def source_refs():
    return saved.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            CONFIG,
            PLAN,
            REPO / "src/s2a/research/trajectory_search/quality_capped_inner_joint_plan.py",
            REPO / "src/s2a/research/trajectory_search/isolated_batched_quality_solver.py",
            REPO / "tests/research/trajectory_search/test_quality_capped_inner_joint_plan.py",
            REPO / "tests/research/trajectory_search/test_isolated_batched_quality_solver.py",
            REPO / "tests/research/native_runtime/test_batched_quality_solver_inputs.py",
        )
    ]


def read_config():
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    prefix.require(
        config
        == {
            "schema": "s2a.batched-quality-solver.v1",
            "case_index": 5,
            "case_id": "task1_init0_s42_terminal058",
            "seed": 42,
            "steps": [352, 355, 359],
            "repetitions_per_implementation": 2,
            "max_solver_calls": 12,
            "max_response_evaluations_per_call": 160,
            "max_response_evaluations": 1920,
            "max_independent_audits": 12,
            "comparison_atol": 1e-10,
            "comparison_rtol": 0,
            "past_inner_error": "zero_diagnostic_only",
            "max_workers": 3,
            "worker_timeout_seconds": 600,
            "physical_steps": 0,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
        "Exact registered quality-solver comparison",
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
        "Previously compared fixed states and seed",
    )
    refs = dict(inputs["source_sha256"])
    refs[config_ref["path"]] = config_ref["sha256"]
    benchmark = prefix.read(BENCHMARK_RESULT, refs)
    prefix.require(
        benchmark["complete"] is benchmark["passed"] is benchmark["sources_unchanged"] is True
        and all(value == 153 for value in benchmark["counts"].values())
        and benchmark["solver_calls"]
        == benchmark["physical_steps"]
        == benchmark["world_model_calls"]
        == benchmark["fresh_cases"]
        == 0,
        "Prior full rollout comparison and lifecycle closed",
    )
    for step in config["steps"]:
        sample = prefix.read(benchmark["sample_refs"][str(step)], refs)
        prefix.require(
            sample["complete"] is sample["passed"] is True
            and sample["data"]["sample_sha256"] == saved_samples.sample_fingerprint(inputs["samples"][str(step)])
            and sample["data"]["compared_pairs"] == 51
            and max(sample["data"]["max_abs_difference_by_field"].values()) <= config["comparison_atol"],
            "Each fixed sample passed the original full-output rollout comparison",
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
        "sample_fingerprints": {
            step: saved_samples.sample_fingerprint(sample) for step, sample in inputs["samples"].items()
        },
    }
