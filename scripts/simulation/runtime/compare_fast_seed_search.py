"""Compare seed-first and full cached search on fixed saved causal states."""

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

from s2a.controllers.osc import OSCInterface
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result, result_json
from s2a.research.joint_control.preview.cached_seed_preference import solve_prefer_cached_seed
from s2a.research.joint_control.preview.fast_seed_preference import solve_fast_preferred_seed
from s2a.research.joint_control.preview.interior_joint_preview import build_joint_preview
from s2a.research.joint_control.tracking.two_step_quality_audit import audit_two_step_witness
from s2a.research.trajectory_search.margin_two_step_search import margin_prefix_satisfied
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import prepare_broader_recovery_pilot as strict
from scripts.simulation.runtime import prepare_quality_solver_runtime_profile as binding
from scripts.simulation.runtime import prepare_reset_correct_reference_cohort as preparation
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/joint_control/fast_seed_search_20261116.json"
PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/fast_seed_search_20261116.md"
REPAIR_PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/fast_seed_search_dry_repair_20261116.md"
INCIDENT = ROOT / "reports/tracking/joint_control/fast_seed_search_dry_incident_20261116.md"
FAILURE = ROOT / "outputs/fast_seed_search_dry_failure_20261116.json"
OUTPUT = ROOT / "outputs/fast_seed_search_20261116"
DRY = ROOT / "outputs/fast_seed_search_dry_v2_20261116.json"
QA = ROOT / "outputs/qa/fast_seed_search_final_v2_20261116/result.json"
MODULE = "scripts.simulation.runtime.compare_fast_seed_search"


def validate_config(config):
    expected = dict(
        schema="s2a.fast-seed-search.v1",
        output="outputs/fast_seed_search_20261116",
        seed=42,
        source_profile=dict(
            path="outputs/quality_solver_runtime_profile_20261111/result.json",
            sha256="12271b66e24e546702626bc884e0be294571d3411415312e4151c2e2b7952d0f",
        ),
        steps=[352, 355, 359],
        repetitions_per_solver=2,
        solver_calls=12,
        max_responses_per_call=320,
        max_responses=3840,
        max_independent_audits=12,
        past_inner_error="zero_diagnostic_only",
        world_model_calls=0,
        gpu_calls=0,
        physical_steps=0,
        fits=0,
        fresh_cases=0,
        physical_qualification=False,
        automatic_retry=False,
    )
    prefix.require(
        strict._same_typed(config, expected), "The fixed three-state and bounded compute contract is required"
    )


def source_refs():
    return binding.source_refs() + [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            REPAIR_PLAN,
            INCIDENT,
            Path(__file__).resolve(),
            ROOT / "src/s2a/research/joint_control/preview/fast_seed_preference.py",
            ROOT / "src/s2a/research/joint_control/preview/cached_seed_preference.py",
            ROOT / "src/s2a/research/joint_control/preview/shared_shift_fallback.py",
            ROOT / "src/s2a/research/trajectory_search/margin_guarded_fallback.py",
            ROOT / "src/s2a/research/trajectory_search/margin_two_step_search.py",
            ROOT / "tests/research/joint_control/preview/test_fast_seed_preference.py",
            ROOT / "tests/research/native_runtime/test_fast_seed_search.py",
        )
    ]


def load_inputs():
    prefix.environment()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    validate_config(config)
    old_ref = preparation.configured_ref(config["source_profile"])
    old = prefix.read(old_ref)
    failure_ref = prefix.reference(FAILURE)
    failure = prefix.read(failure_ref)
    inputs = binding.load_inputs()
    prefix.require(
        old["complete"] is old["passed"] is old["sources_unchanged"] is True
        and old["physical_steps"] == old["world_model_calls"] == old["fresh_cases"] == 0
        and sorted(map(int, inputs["samples"])) == config["steps"]
        and inputs["config"]["seed"] == config["seed"]
        and inputs["config"]["past_inner_error"] == config["past_inner_error"]
        and failure_ref["sha256"] == "c94867190501c00eada103570ef0572a0950823600d0e3b9c4c863dbedb5d29b"
        and failure["passed"] is failure["complete"] is failure["automatic_retry"] is False
        and failure["solver_calls"]
        == failure["model_responses"]
        == failure["physical_steps"]
        == failure["gpu_calls"]
        == 0,
        "The saved-state profile, zero-error limitation and failed dry evidence must match",
    )
    return (
        config,
        dict(
            comparison_config_ref=config_ref,
            profile_ref=old_ref,
            failed_dry_ref=failure_ref,
            profile_inputs=binding.input_receipt(inputs),
        ),
        inputs,
    )


def run_sample(
    sample, ledger, *, baseline=solve_prefer_cached_seed, fast=solve_fast_preferred_seed, auditor=audit_two_step_witness
):
    """Compare two repetitions each without executing a robot control period."""
    before_sample = binding.sample_fingerprint(sample)
    model = build_joint_preview(
        sample["mass"],
        sample["jacobian"],
        sample["kp"],
        sample["kd"],
        sample["q"],
        sample["initial_joint"],
        null_kp=10.0,
        dt=0.002,
    )
    before_model = hashlib.sha256(result_json(model).encode()).hexdigest()
    contract = sample["contract"]
    interface = OSCInterface(*(contract[key] for key in ("input_min", "input_max", "output_min", "output_max")))
    args = (
        model,
        sample["velocity"],
        sample["seed"],
        sample["previous"],
        interface,
        sample["site"],
        sample["site_from_tcp"],
        sample["targets"],
        sample["twists"],
        sample["lower"],
        sample["upper"],
    )
    audit_args = (
        sample["mass"],
        sample["jacobian"],
        sample["kp"],
        sample["kd"],
        sample["q"],
        sample["velocity"],
        sample["initial_joint"],
        interface,
        sample["previous"],
        sample["site"],
        sample["site_from_tcp"],
        sample["targets"],
        sample["lower"],
        sample["upper"],
    )
    pairs, partial = [], ledger["calls"]
    for repeat in range(2):
        values = {}
        order = ("baseline", "fast") if repeat == 0 else ("fast", "baseline")
        for name in order:
            solver = baseline if name == "baseline" else fast
            counts = dict(attempted=0, completed=0)
            record = dict(repetition=repeat, solver=name, complete=False, response_events=counts)
            partial.append(record)
            ledger["solver_attempted"] += 1

            def response(event):
                prefix.require(event in counts, "Only actual attempted/completed responses may be recorded")
                counts[event] += 1
                ledger[f"responses_{event}"] += 1
                prefix.require(
                    counts["attempted"] <= 320 and counts["completed"] <= counts["attempted"],
                    "The original per-call response budget applies",
                )

            wall, cpu = time.perf_counter(), time.process_time()
            result = solver(*args, past_inner_error_rad=sample["past_inner_error_rad"], on_response=response)
            record.update(wall_seconds=time.perf_counter() - wall, process_cpu_seconds=time.process_time() - cpu)
            prefix.require(
                counts["attempted"] == counts["completed"] == result["response_evaluations"] <= 320,
                "All actual response events and reported counts must match",
            )
            prefix.require(
                before_sample == binding.sample_fingerprint(sample)
                and before_model == hashlib.sha256(result_json(model).encode()).hexdigest(),
                "Each solver must leave the shared saved input and model unchanged",
            )
            ledger["solver_completed"] += 1
            candidate = result.get("best_joint_safe")
            check = None
            if candidate is not None:
                ledger["audits_attempted"] += 1
                check = auditor(candidate, *audit_args, past_inner_error_rad=sample["past_inner_error_rad"])
                prefix.require(
                    check["passed"]
                    is check["prefix_pose_caps_satisfied"]
                    is check["braking_constraint_satisfied"]
                    is True
                    and (
                        candidate["selection_scope"] != "margin_two_step"
                        or margin_prefix_satisfied(check["tcp_errors"])
                    ),
                    "The unchanged independent dual-pose and joint witness must pass",
                )
                ledger["audits_completed"] += 1
            record.update(
                complete=True,
                status=result["status"],
                candidate_scope=None if candidate is None else candidate["selection_scope"],
                planned_commands=None if candidate is None else candidate["planned_commands"],
                tcp_errors=None if candidate is None else candidate["tcp_errors"],
                audit_passed=None if check is None else check["passed"],
                fast_seed=result.get("fast_seed"),
                seed_preference=result.get("seed_preference"),
            )
            values[name] = result
        left, right = values["baseline"], values["fast"]
        old, new = left.get("best_joint_safe"), right.get("best_joint_safe")
        prefix.require(
            left["status"] == right["status"] and (old is None) == (new is None),
            "Both paths must retain the same outcome",
        )
        difference = None
        if old is not None:
            prefix.require(
                old["selection_scope"] == new["selection_scope"], "Selected audit scope must remain unchanged"
            )
            fields = ("planned_commands", "terminal_normals", "tcp_errors")
            differences = [
                float(np.max(np.abs(np.asarray(old[key]) - np.asarray(new[key])))) if np.asarray(old[key]).size else 0.0
                for key in fields
            ]
            prefix.require(set(old["rollout"]) == set(new["rollout"]), "The selected rollout fields must match")
            for key in old["rollout"]:
                a, b = np.asarray(old["rollout"][key]), np.asarray(new["rollout"][key])
                prefix.require(a.shape == b.shape, "Selected rollout shapes must match")
                differences.append(float(np.max(np.abs(a - b))) if a.size else 0.0)
            difference = max(differences)
            prefix.require(difference <= 1e-10, "The actual selected plan and full model rollout must match")
        early = right["fast_seed"]["early_selected"]
        prefix.require(
            early is left["seed_preference"]["seed_in_eligible_cache"]
            and (
                not early
                or (
                    new is not None
                    and np.array_equal(new["planned_commands"], sample["seed"])
                    and right["response_evaluations"] == 1
                )
            )
            and (early or right["response_evaluations"] == left["response_evaluations"]),
            "Early eligibility and fallback response accounting must agree with the full cached search",
        )
        pairs.append(
            dict(repetition=repeat, order=list(order), early_selected=early, max_selected_rollout_difference=difference)
        )
    prefix.require(
        before_sample == binding.sample_fingerprint(sample)
        and before_model == hashlib.sha256(result_json(model).encode()).hexdigest(),
        "The saved current input and frozen model must remain unchanged",
    )
    return dict(step=sample["step"], sample_sha256=before_sample, pairs=pairs)


def worker(step):
    prefix.environment()
    directory = OUTPUT / f"step_{step}"
    registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
    config, identity, inputs = load_inputs()
    prefix.require(
        step in config["steps"] and identity == registration["inputs"] and source_refs() == registration["source_refs"],
        "Registered fixed-state worker and sources must match",
    )
    ledger = dict(
        solver_attempted=0,
        solver_completed=0,
        responses_attempted=0,
        responses_completed=0,
        audits_attempted=0,
        audits_completed=0,
        calls=[],
    )
    result = dict(
        complete=False, passed=False, step=step, ledger=ledger, physical_qualification=False, automatic_retry=False
    )
    try:
        result["comparison"] = run_sample(inputs["samples"][str(step)], ledger)
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        print(publish_result(directory / "result.json", result), flush=True)


def preflight():
    prefix.environment()
    qa_ref, dry_ref = prefix.reference(QA), prefix.reference(DRY)
    qa, dry = prefix.read(qa_ref), prefix.read(dry_ref)
    before, sources = snapshot(ROOT)["sources"], source_refs()
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["configured_parallelism"] == 64
        and qa["skipped_count"] == 0
        and not qa["collection"]["collection_errors"]
        and before == prefix.read(qa["source_before_ref"])["sources"]
        and dry["complete"] is dry["passed"] is True
        and dry["solver_calls"] == dry["physical_steps"] == dry["gpu_calls"] == dry["world_model_calls"] == 0,
        "A source-matched 64-worker QA and zero-solver dry are required",
    )
    config, identity, inputs = load_inputs()
    prefix.require(
        dry["inputs"] == identity and dry["source_refs"] == sources, "Dry source and sample identities must match"
    )
    return config, identity, inputs, before, sources, qa_ref, dry_ref


def execute():
    config, identity, _inputs, before, sources, qa_ref, dry_ref = preflight()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    registration_ref = publish_result(
        OUTPUT / "registration.json",
        dict(inputs=identity, source_refs=sources, qa_ref=qa_ref, dry_ref=dry_ref, automatic_retry=False),
    )
    specs = [
        dict(
            key=str(step),
            argv=[sys.executable, "-m", MODULE, "--worker", str(step)],
            log=OUTPUT / f"step_{step}/worker.log",
        )
        for step in config["steps"]
    ]
    lifecycle = run_owned_workers(specs, cwd=ROOT, timeout_seconds=600)
    for child in lifecycle["workers"]:
        child["log_ref"] = prefix.reference(child.pop("log"))
    lifecycle_ref = publish_result(OUTPUT / "workers.json", lifecycle)
    result = dict(
        complete=False,
        passed=False,
        physical_qualification=False,
        registration_ref=registration_ref,
        lifecycle_ref=lifecycle_ref,
        cases={},
        physical_steps=0,
        world_model_calls=0,
        gpu_calls=0,
        fits=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        for step in config["steps"]:
            path = OUTPUT / f"step_{step}/result.json"
            row = result["cases"][str(step)] = dict(status="failed")
            if path.is_file():
                row["available_ref"] = prefix.reference(path)
                data = prefix.read(row["available_ref"])
                row["ledger"] = data["ledger"]
                if data["complete"] is data["passed"] is True:
                    row.update(status="closed", result_ref=row["available_ref"])
        prefix.require(
            lifecycle["launch_error"] is None
            and len(lifecycle["workers"]) == 3
            and {item["key"] for item in lifecycle["workers"]} == {str(step) for step in config["steps"]}
            and all(
                item["returncode"] == 0
                and item["actual_wait"] is item["reaped"] is True
                and item["signal_sent"] is False
                for item in lifecycle["workers"]
            )
            and all(row["status"] == "closed" for row in result["cases"].values()),
            "All three registered comparison workers must close and be reaped",
        )
        ledgers = [row["ledger"] for row in result["cases"].values()]
        totals = {
            key: sum(row[key] for row in ledgers)
            for key in (
                "solver_attempted",
                "solver_completed",
                "responses_attempted",
                "responses_completed",
                "audits_attempted",
                "audits_completed",
            )
        }
        prefix.require(
            totals["solver_attempted"] == totals["solver_completed"] == config["solver_calls"]
            and totals["responses_attempted"] == totals["responses_completed"] <= config["max_responses"]
            and totals["audits_attempted"] == totals["audits_completed"] <= config["max_independent_audits"],
            "The finite solver, response and independent-audit budgets must close",
        )
        _config, after, _inputs = load_inputs()
        prefix.require(
            after == identity and source_refs() == sources and snapshot(ROOT)["sources"] == before,
            "Saved inputs and sources must remain unchanged",
        )
        result.update(complete=True, passed=True, sources_unchanged=True, totals=totals)
    except Exception as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
    print(publish_result(OUTPUT / "result.json", result), flush=True)
    if not result["passed"]:
        raise SystemExit("The fixed-state comparison did not close; no automatic retry")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--worker", type=int)
    args = parser.parse_args()
    prefix.environment()
    if args.dry:
        _config, identity, _inputs = load_inputs()
        print(
            publish_result(
                DRY,
                dict(
                    complete=True,
                    passed=True,
                    inputs=identity,
                    source_refs=source_refs(),
                    solver_calls=0,
                    physical_steps=0,
                    gpu_calls=0,
                    world_model_calls=0,
                    physical_qualification=False,
                ),
            ),
            flush=True,
        )
    elif args.execute:
        execute()
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
