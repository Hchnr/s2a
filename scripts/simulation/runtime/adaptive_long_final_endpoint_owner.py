"""Supervise one CPU replay with caching only at its last endpoint."""

import sys
from pathlib import Path

from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.native_runtime.physical_cost_bounds import physical_cost_bounds
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import adaptive_live_bowl_recovery_audit as audit
from scripts.simulation.runtime import adaptive_live_bowl_recovery_owner as accounting
from scripts.simulation.runtime import adaptive_live_bowl_supervisor as supervisor
from scripts.simulation.runtime import adaptive_live_bowl_worker as historical
from scripts.simulation.runtime import adaptive_long_cached_checker_owner as previous
from scripts.simulation.runtime import adaptive_long_final_endpoint_admission as admission
from scripts.simulation.runtime import adaptive_long_final_endpoint_worker as worker
from scripts.simulation.runtime import audit_cached_checker_incident_20261113 as incident
from scripts.simulation.runtime import registered_feedback_phase as lifecycle_check
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical

MODULE = "scripts.simulation.runtime.run_adaptive_long_final_endpoint"
REQUIRED_QA_TARGETS = tuple(
    dict.fromkeys(
        (
            *previous.REQUIRED_QA_TARGETS,
            "tests/research/native_runtime/test_audit_cached_checker_incident_20261113.py",
            "tests/research/native_runtime/test_adaptive_long_final_endpoint_admission.py",
            "tests/research/native_runtime/test_adaptive_long_final_endpoint_worker.py",
            "tests/research/native_runtime/test_adaptive_long_final_endpoint_owner.py",
        )
    )
)


def source_refs():
    """Bind both historical failures and all sources of this final-only replay."""
    paths = (
        admission.CONFIG,
        admission.PLAN,
        Path(incident.__file__).resolve(),
        Path(admission.__file__).resolve(),
        Path(worker.__file__).resolve(),
        Path(__file__).resolve(),
        admission.ROOT / "scripts/simulation/runtime/run_adaptive_long_final_endpoint.py",
        *(admission.ROOT / path for path in REQUIRED_QA_TARGETS),
    )
    refs = previous.source_refs()
    known = {ref["path"]: ref["sha256"] for ref in refs}
    for path in paths:
        ref = prefix.reference(path)
        if ref["path"] in known:
            prefix.require(known[ref["path"]] == ref["sha256"], "Conflicting source identities")
        else:
            refs.append(ref)
            known[ref["path"]] = ref["sha256"]
    return refs


def dry_run():
    """Validate saved inputs without starting a model or simulator."""
    prefix.environment()
    inputs = admission.load_inputs()
    return publish(
        admission.DRY,
        dict(
            complete=True,
            passed=True,
            inputs=admission.input_receipt(inputs),
            source_refs=source_refs(),
            native_models=0,
            physics_steps=0,
            model_initializations=0,
            model_requests=0,
            denoise_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
    )


def check_cpu_admission(inputs):
    """Require source-matched 64-worker regression and zero-work dry run."""
    qa, dry = prefix.read(prefix.reference(admission.QA)), prefix.read(prefix.reference(admission.DRY))
    before = snapshot(prefix.REPO)["sources"]
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["configured_parallelism"] == qa["execution_parallelism"] == 64
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and qa["aggregate_node_partition_exact"] is True
        and qa["call_count"] > 0
        and not qa["collection"]["collection_errors"]
        and set(REQUIRED_QA_TARGETS).issubset(set(qa["targets"]))
        and prefix.read(qa["source_before_ref"])["sources"] == before
        and dry["complete"] is dry["passed"] is True
        and dry["automatic_retry"] is False
        and dry["inputs"] == admission.input_receipt(inputs)
        and dry["source_refs"] == source_refs()
        and dry["native_models"]
        == dry["physics_steps"]
        == dry["model_initializations"]
        == dry["model_requests"]
        == dry["denoise_calls"]
        == dry["fresh_cases"]
        == 0,
        "Matching 64-worker CPU regression and zero-work dry run are required",
    )
    return before


def audit_final_cache(branch, *, final_step):
    """Require precisely one measured and one reused final endpoint."""
    result = previous.audit_cache_events(branch, start_step=final_step)
    prefix.require(
        result["observed_steps"] == [final_step]
        and result["reused_steps"] == [final_step]
        and result["event_count"] == 2
        and result["final_endpoint_step"] == final_step,
        "Only the repeated final endpoint may use the cache",
    )
    return result


def phase_costs(result, *, inputs, sources, lifecycle, directory):
    """Retain final receipts or the full registered bounds on failure."""
    matched = False
    try:
        matched = source_refs() == sources
    except Exception as error:
        result["partial_source_check_error"] = {"type": type(error).__name__, "message": str(error)}
    result["partial_source_match"] = matched
    branch = None
    path = directory / "simulation/result.json"
    if path.is_file():
        try:
            result["available_branch_ref"] = prefix.reference(path)
            branch = prefix.read(result["available_branch_ref"])
        except Exception as error:
            result["branch_receipt_error"] = {"type": type(error).__name__, "message": str(error)}
    config = inputs["config"]
    limits = dict(
        control_steps=config["max_control_steps"],
        native_steps=config["max_native_steps"],
        models=config["max_models"],
        contexts=config["max_contexts"],
    )
    final = matched and accounting._closed_worker(lifecycle, "checker") and type(branch) is dict
    try:
        bounds = physical_cost_bounds(branch.get("runtime") if final else None, limits=limits, receipt_final=final)
    except (TypeError, ValueError, KeyError) as error:
        result["physical_receipt_error"] = {"type": type(error).__name__, "message": str(error)}
        bounds = physical_cost_bounds(None, limits=limits, receipt_final=False)
    result.update(
        phase_physical_bounds=bounds,
        cumulative_physical_bounds=audit.add_cost_bounds(inputs["previous_physical_bounds"], bounds),
        cumulative_model_bounds=inputs["previous_model_bounds"],
        model_initializations=0,
        model_requests=0,
        denoise_calls=0,
    )


def execute():
    """Run the single registered replay and preserve success or failure."""
    prefix.environment()
    inputs = admission.load_inputs()
    before, sources = check_cpu_admission(inputs), source_refs()
    admission.OUTPUT.mkdir(parents=True, exist_ok=False)
    directory = admission.OUTPUT / "checker"
    simulation = directory / "simulation"
    simulation.mkdir(parents=True)
    result = dict(complete=False, passed=False, phase="checker", control_qualified=False, automatic_retry=False)
    lifecycle = None
    try:
        result["registration_ref"] = publish(
            admission.OUTPUT / "registration.json", dict(inputs=admission.input_receipt(inputs), source_refs=sources)
        )
        environment = physical._cpu_environment(simulation)
        spec = dict(key="checker", argv=[sys.executable, "-m", MODULE, "--worker"], log=directory / "checker.log")
        with physical._temporary_environment(environment):
            lifecycle = run_owned_workers(
                [spec], cwd=prefix.REPO, timeout_seconds=inputs["config"]["worker_timeout_seconds"]
            )
        result["workers_ref"] = publish(directory / "workers.json", supervisor._record_worker_logs(lifecycle))
        lifecycle_check.require_closed_workers(lifecycle, ["checker"])
        branch_ref = prefix.reference(simulation / "result.json")
        branch = prefix.read(branch_ref)
        producer = prefix.read(inputs["producer"]["branch_ref"])
        model = prefix.read(inputs["producer"]["model_ref"])
        historical.load_recorded_responses(producer, model)
        counts = audit.summarize_branch(branch, "checker", inputs["long_inputs"]["config"])
        counts["independent_max_absolute_difference"] = audit.compare_branches(producer, branch)
        prefix.require(
            counts["control_steps"] == inputs["producer"]["counts"]["control_steps"] == 147
            and counts["native_steps"] == 3675,
            "The replay must complete the recorded 147 control periods",
        )
        result["cache_audit"] = audit_final_cache(branch, final_step=inputs["config"]["cache_step"])
        result["capture_steps"] = previous.compare_captures(producer, branch)
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(source_refs() == sources and snapshot(prefix.REPO)["sources"] == before, "Sources changed")
        result.update(complete=True, passed=True, sources_unchanged=True, branch_ref=branch_ref, counts=counts)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        try:
            phase_costs(result, inputs=inputs, sources=sources, lifecycle=lifecycle, directory=directory)
            if result["complete"]:
                for key in ("control_steps", "native_steps", "models", "contexts"):
                    cost = result["phase_physical_bounds"][key]
                    prefix.require(
                        cost["lower"] == cost["upper"] == result["counts"][key], "Complete costs must be exact"
                    )
        except Exception as error:
            result.update(
                complete=False,
                passed=False,
                cost_accounting_error={"type": type(error).__name__, "message": str(error)},
            )
        result_ref = publish(directory / "result.json", result)
        print(result_ref, flush=True)
        if "cost_accounting_error" in result:
            raise ValueError("Final-endpoint checker cost accounting failed")
    return result_ref
