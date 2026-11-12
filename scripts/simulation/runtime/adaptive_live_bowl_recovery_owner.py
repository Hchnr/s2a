"""Own the separately registered adaptive recovery trial and its evidence."""

import os
import sys
from pathlib import Path

from s2a.infrastructure.lifecycle.owned_socket_pair import OwnedSocketPairInterrupted, run_owned_socket_pair
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.native_runtime.physical_cost_bounds import physical_cost_bounds
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import adaptive_live_bowl_recovery_admission as admission
from scripts.simulation.runtime import adaptive_live_bowl_recovery_audit as audit
from scripts.simulation.runtime import adaptive_live_bowl_supervisor as historical
from scripts.simulation.runtime import adaptive_live_bowl_worker as simulation
from scripts.simulation.runtime import registered_feedback_phase as lifecycle_check
from scripts.simulation.runtime import registered_model_bundle as registered
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro
from scripts.wam.tracking.preview import live_cv2_admission
from scripts.wam.tracking.preview import ordered_feedback_wam_inference_20261111 as engine
from scripts.wam.tracking.preview import run_causal_stream_live_wam as model_owner

ROOT, OUTPUT, QA, DRY = admission.ROOT, admission.OUTPUT, admission.QA, admission.DRY
MODULE = "scripts.simulation.runtime.run_adaptive_live_bowl_recovery"
REQUIRED_QA_TARGETS = (
    "tests/research/native_runtime/test_adaptive_live_bowl_recovery_admission.py",
    "tests/research/native_runtime/test_adaptive_live_bowl_recovery_audit.py",
    "tests/research/native_runtime/test_adaptive_live_bowl_recovery_owner.py",
    "tests/research/native_runtime/test_adaptive_recovery_inputs.py",
    "tests/research/native_runtime/test_step107_model_inputs.py",
    "tests/research/native_runtime/test_registered_model_bundle.py",
    "tests/research/native_runtime/test_adaptive_live_bowl_source_manifest.py",
    "tests/research/native_runtime/test_managed_adaptive_simulation.py",
    "tests/research/native_runtime/test_managed_physical_worker.py",
    "tests/research/native_runtime/test_physical_cost_bounds.py",
    "tests/unit/infrastructure/lifecycle/test_interruptible_worker.py",
    "tests/unit/infrastructure/lifecycle/test_owned_socket_pair.py",
    "tests/research/native_runtime/test_adaptive_live_bowl.py",
    "tests/research/native_runtime/test_adaptive_streaming_replay.py",
    "tests/research/native_runtime/test_adaptive_stream_reference.py",
    "tests/research/native_runtime/test_adaptive_streaming_continuation.py",
    "tests/research/wam/test_adaptive_model_worker.py",
    "tests/research/wam/test_adaptive_live_requests.py",
    "tests/research/wam/test_adaptive_live_pipeline.py",
    "tests/unit/infrastructure/test_adaptive_stream_protocol.py",
    "tests/research/joint_control/preview/test_causal_reference_replan.py",
)


def source_refs():
    """Freeze the same sources in fresh model, simulation and owner processes."""
    paths = (
        admission.CONFIG,
        admission.PLAN,
        Path(__file__).resolve(),
        ROOT / "scripts/simulation/runtime/run_adaptive_live_bowl_recovery.py",
        ROOT / "scripts/simulation/runtime/adaptive_live_bowl_recovery_admission.py",
        ROOT / "scripts/simulation/runtime/adaptive_live_bowl_recovery_audit.py",
        ROOT / "scripts/simulation/runtime/adaptive_recovery_inputs.py",
        ROOT / "scripts/simulation/runtime/step107_model_inputs.py",
        ROOT / "scripts/simulation/runtime/adaptive_live_bowl_source_manifest.py",
        ROOT / "scripts/simulation/runtime/registered_model_bundle.py",
        ROOT / "scripts/simulation/runtime/managed_physical_worker.py",
        ROOT / "scripts/simulation/runtime/managed_adaptive_simulation.py",
        ROOT / "src/s2a/infrastructure/lifecycle/interruptible_worker.py",
        ROOT / "src/s2a/research/native_runtime/physical_cost_bounds.py",
        *(ROOT / path for path in REQUIRED_QA_TARGETS),
    )
    return registered.prior_adaptive_source_refs() + [prefix.reference(path) for path in paths]


def model_bundle(inputs):
    return dict(
        input_receipt=admission.input_receipt(inputs),
        prepared=inputs["prepared"],
        bridge=inputs["bridge"],
        template=inputs["template"],
        weight_hashes=inputs["weight_hashes"],
        model_registration=admission.model_registration(inputs, OUTPUT / "producer/simulation"),
    )


def dry_run():
    inputs = admission.load_inputs()
    result = dict(
        complete=True,
        passed=True,
        inputs=admission.input_receipt(inputs),
        source_refs=source_refs(),
        model_bundle_sha256=registered._digest(model_bundle(inputs)),
        native_models=0,
        physics_steps=0,
        model_initializations=0,
        model_requests=0,
        denoise_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    print(publish(DRY, result), flush=True)


def check_cpu_admission(inputs):
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    before = snapshot(ROOT)["sources"]
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["configured_parallelism"] == 64
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and not qa["collection"]["collection_errors"]
        and set(REQUIRED_QA_TARGETS).issubset(set(qa["targets"]))
        and prefix.read(qa["source_before_ref"])["sources"] == before
        and dry["complete"] is dry["passed"] is True
        and dry["inputs"] == admission.input_receipt(inputs)
        and dry["source_refs"] == source_refs()
        and dry["model_bundle_sha256"] == registered._digest(model_bundle(inputs))
        and dry["native_models"]
        == dry["physics_steps"]
        == dry["model_initializations"]
        == dry["model_requests"]
        == dry["denoise_calls"]
        == dry["fresh_cases"]
        == 0,
        "Matching 64-worker CPU regression and zero-physics recovery dry run are required",
    )
    return before


def preflight():
    prefix.environment()
    inputs = admission.load_inputs()
    before, sources = check_cpu_admission(inputs), source_refs()
    prefix.require(engine.GPU == inputs["config"]["gpu_uuid"], "GPU observation must target the registered UUID")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    started = dict(complete=False, config=prefix.reference(admission.CONFIG), pid=os.getpid())
    publish(OUTPUT / "preflight_started.json", started)
    try:
        gpu = engine.gpu_observation()
        prefix.require(gpu["memory_mib"] == 0 and gpu["compute_apps"] == [], "The registered GPU must be idle")
        manifests = model_owner.verify_model_manifests(inputs)
        probe_dir = OUTPUT / "cv2_probe"
        probe_dir.mkdir()
        probe = live_cv2_admission.run_cpu_probe(inputs["prepared"]["model_config"]["model_python"], probe_dir)
        probe_ref = publish(probe_dir / "result.json", probe)
        prefix.require(snapshot(ROOT)["sources"] == before and source_refs() == sources, "Preflight sources changed")
        result = dict(
            complete=True,
            passed=True,
            inputs=admission.input_receipt(inputs),
            source_refs=sources,
            qa_ref=prefix.reference(QA),
            dry_ref=prefix.reference(DRY),
            model_bundle_sha256=registered._digest(model_bundle(inputs)),
            gpu_uuid=inputs["config"]["gpu_uuid"],
            gpu_before=gpu,
            cv2_probe_ref=probe_ref,
            model_python_sha256=repro.sha256(inputs["prepared"]["model_config"]["model_python"]),
            model_initializations=0,
            model_requests=0,
            physics_steps=0,
            automatic_retry=False,
            **manifests,
        )
        print(publish(OUTPUT / "preflight.json", result), flush=True)
    except BaseException as error:
        publish(
            OUTPUT / "preflight_failed.json",
            {**started, "error": {"type": type(error).__name__, "message": str(error)}},
        )
        raise


def check_preflight(inputs):
    value = prefix.read(prefix.reference(OUTPUT / "preflight.json"))
    prefix.require(
        value["complete"] is value["passed"] is True
        and value["inputs"] == admission.input_receipt(inputs)
        and value["source_refs"] == source_refs()
        and value["qa_ref"] == prefix.reference(QA)
        and value["dry_ref"] == prefix.reference(DRY)
        and value["model_bundle_sha256"] == registered._digest(model_bundle(inputs))
        and value["gpu_uuid"] == inputs["config"]["gpu_uuid"]
        and value["source_files_verified"] == len(inputs["model_source_hashes"])
        and value["weight_files_verified"] == len(inputs["weight_hashes"]) == 163
        and value["model_python_sha256"] == repro.sha256(inputs["prepared"]["model_config"]["model_python"])
        and value["weight_stats"] == model_owner.file_stats(inputs["weight_hashes"]),
        "The source-matched recovery preflight is required",
    )
    probe = prefix.read(value["cv2_probe_ref"])
    prefix.require(probe["actual_wait"] is probe["reaped"] is True and probe["returncode"] == 0, "CPU OpenCV probe")
    return value


def model_worker_inputs():
    """Use the binding-free source manifest inside the fresh model interpreter."""
    return registered.load_registered_model_bundle(
        output=OUTPUT, qa_path=QA, dry_path=DRY, gpu_uuid=engine.GPU, source_refs=source_refs
    )


def require_completed_model(model, config):
    """Validate actual model/device counts before accepting physical evidence."""
    prefix.require(
        model["complete"] is model["passed"] is True
        and model["gpu_uuid"] == config["gpu_uuid"]
        and model["model_initialization_attempts"] == model["model_initializations"] == 1
        and 0 <= model["responses_sent"] <= config["max_model_requests"]
        and model["model_requests_attempted"] == model["model_requests_completed"] == model["responses_sent"]
        and model["denoise_calls_attempted"] == model["denoise_calls_completed"] == 35 * model["responses_sent"],
        "The completed model receipt must match the registered device and bounded calls",
    )


def model_cost_bounds(receipt, *, config, receipt_final):
    """Keep completed and attempted model work bounded when receipts are partial."""
    prefix.require(
        (receipt is None or type(receipt) is dict) and type(receipt_final) is bool, "Valid model receipt required"
    )
    receipt = {} if receipt is None else receipt
    mapping = (
        ("initializations", "model_initialization_attempts", "model_initializations", "max_model_initializations"),
        ("requests", "model_requests_attempted", "model_requests_completed", "max_model_requests"),
        ("denoise_calls", "denoise_calls_attempted", "denoise_calls_completed", "max_denoise_calls"),
    )
    result = {}
    for name, attempted_key, completed_key, limit_key in mapping:
        limit = config[limit_key]
        attempted, completed = receipt.get(attempted_key), receipt.get(completed_key)
        prefix.require(type(limit) is int and limit >= 0, "Explicit model limits required")
        if attempted is completed is None:
            result[name + "_attempted"] = dict(lower=0, upper=limit, evidence="not_recorded")
            result[name + "_completed"] = dict(lower=0, upper=limit, evidence="not_recorded")
            continue
        prefix.require(
            type(attempted) is type(completed) is int and 0 <= completed <= attempted <= limit,
            "Model counters must remain within the registered budget",
        )
        evidence = "final_receipt" if receipt_final else "partial_snapshot"
        result[name + "_attempted"] = dict(
            lower=attempted, upper=attempted if receipt_final else limit, evidence=evidence
        )
        result[name + "_completed"] = dict(
            lower=completed, upper=attempted if receipt_final else limit, evidence=evidence
        )
    return result


def known_model_bounds(config):
    """Represent historically completed/attempted model counts as exact values."""
    mapping = {
        "initializations_completed": "previous_model_initializations",
        "requests_attempted": "previous_requests_attempted",
        "requests_completed": "previous_requests_completed",
        "denoise_calls_completed": "previous_denoises",
    }
    prefix.require(
        all(type(config[key]) is int and config[key] >= 0 for key in mapping.values()),
        "Valid historically known model costs are required",
    )
    return {name: dict(lower=config[key], upper=config[key]) for name, key in mapping.items()}


def add_model_cost_bounds(prior, bounds):
    """Add new model ranges without losing costs from an earlier phase."""
    names = ("initializations_completed", "requests_attempted", "requests_completed", "denoise_calls_completed")
    prefix.require(type(prior) is dict and set(prior) == set(names), "Complete previous model ranges required")
    result = {}
    for name in names:
        previous, value = prior[name], bounds[name]
        prefix.require(
            type(previous) is type(value) is dict
            and type(previous.get("lower")) is type(previous.get("upper")) is int
            and 0 <= previous["lower"] <= previous["upper"]
            and type(value.get("lower")) is type(value.get("upper")) is int
            and 0 <= value["lower"] <= value["upper"],
            "Valid prior and new model cost bounds are required",
        )
        result[name] = dict(lower=previous["lower"] + value["lower"], upper=previous["upper"] + value["upper"])
    return result


def _closed_worker(lifecycle, name):
    if type(lifecycle) is not dict:
        return False
    rows = [row for row in lifecycle.get("workers", []) if row.get("name", row.get("key")) == name]
    return len(rows) == 1 and rows[0].get("actual_wait") is rows[0].get("reaped") is True


def _phase_costs(result, *, phase, directory, inputs, sources, lifecycle, prior_bounds, prior_model_bounds):
    """Account for available final receipts and conservatively retain unknowns."""
    source_match = False
    try:
        source_match = source_refs() == sources
    except Exception as error:
        result["partial_source_check_error"] = {"type": type(error).__name__, "message": str(error)}
    result["partial_source_match"] = source_match
    branch = model = None
    branch_path = directory / "simulation/result.json"
    model_path = directory / "model/result.json"
    if branch_path.is_file():
        try:
            result["available_branch_ref"] = prefix.reference(branch_path)
            branch = prefix.read(result["available_branch_ref"])
        except Exception as error:
            result["branch_receipt_error"] = {"type": type(error).__name__, "message": str(error)}
    if phase == "producer" and model_path.is_file():
        try:
            result["available_model_ref"] = prefix.reference(model_path)
            model = prefix.read(result["available_model_ref"])
        except Exception as error:
            result["model_receipt_error"] = {"type": type(error).__name__, "message": str(error)}
    limits = dict(
        control_steps=inputs["config"]["max_control_steps_per_phase"],
        native_steps=inputs["config"]["max_native_steps_per_phase"],
        models=inputs["config"]["max_models_per_phase"],
        contexts=inputs["config"]["max_contexts_per_phase"],
    )
    worker_name = "simulation" if phase == "producer" else "checker"
    branch_final = source_match and _closed_worker(lifecycle, worker_name) and type(branch) is dict
    runtime = branch.get("runtime") if branch_final else None
    try:
        physical_bounds = physical_cost_bounds(runtime, limits=limits, receipt_final=branch_final)
    except (TypeError, ValueError, KeyError) as error:
        result["physical_receipt_error"] = {"type": type(error).__name__, "message": str(error)}
        physical_bounds = physical_cost_bounds(None, limits=limits, receipt_final=False)
    if phase == "checker":
        model = dict(
            model_initialization_attempts=0,
            model_initializations=0,
            model_requests_attempted=0,
            model_requests_completed=0,
            denoise_calls_attempted=0,
            denoise_calls_completed=0,
        )
        model_final = True
    else:
        model_final = source_match and _closed_worker(lifecycle, "model") and type(model) is dict
        if not model_final:
            model = None
    try:
        model_bounds = model_cost_bounds(model, config=inputs["config"], receipt_final=model_final)
    except (TypeError, ValueError, KeyError) as error:
        result["model_counter_error"] = {"type": type(error).__name__, "message": str(error)}
        model_bounds = model_cost_bounds(None, config=inputs["config"], receipt_final=False)
    result.update(
        phase_physical_bounds=physical_bounds,
        cumulative_physical_bounds=audit.add_cost_bounds(prior_bounds, physical_bounds),
        phase_model_bounds=model_bounds,
        cumulative_model_bounds=add_model_cost_bounds(prior_model_bounds, model_bounds),
    )


def require_complete_phase_costs(result):
    """A successful phase cannot hide missing, changed or in-flight counters."""
    errors = (
        "partial_source_check_error",
        "branch_receipt_error",
        "model_receipt_error",
        "physical_receipt_error",
        "model_counter_error",
    )
    physical = result["phase_physical_bounds"]
    model = result["phase_model_bounds"]
    counts = result["counts"]
    prefix.require(
        result["partial_source_match"] is True
        and not any(key in result for key in errors)
        and set(physical) == {"control_steps", "native_steps", "models", "contexts"}
        and set(model)
        == {
            "initializations_attempted",
            "initializations_completed",
            "requests_attempted",
            "requests_completed",
            "denoise_calls_attempted",
            "denoise_calls_completed",
        }
        and all(
            physical[key]["evidence"] == "final_receipt"
            and physical[key]["lower"] == physical[key]["upper"] == counts[key]
            for key in ("control_steps", "native_steps", "models", "contexts")
        )
        and all(value["evidence"] == "final_receipt" and value["lower"] == value["upper"] for value in model.values())
        and (result["phase"] != "checker" or all(value["upper"] == 0 for value in model.values())),
        "A complete phase requires exact final physical and model cost receipts",
    )


def verified_producer_bounds(inputs, result, sources):
    """Recompute producer costs before checker dispatch, preserving prior uncertainty."""
    prefix.require(
        result["complete"] is result["passed"] is result["sources_unchanged"] is result["gpu_released"] is True
        and result["phase"] == "producer"
        and result["branch_ref"] == result["available_branch_ref"]
        and result["model_ref"] == result["available_model_ref"],
        "A completed producer with released GPU and final receipts is required",
    )
    registration = prefix.read(result["registration_ref"])
    bundle = prefix.read(registration["model_bundle_ref"])
    lifecycle = prefix.read(result["workers_ref"])
    prefix.require(
        registration["phase"] == "producer"
        and registration["inputs"] == admission.input_receipt(inputs)
        and registration["source_refs"] == sources
        and bundle == model_bundle(inputs)
        and lifecycle["passed"] is True
        and lifecycle["launch_error"] is None
        and lifecycle["timed_out"] is lifecycle["interrupted"] is False
        and len(lifecycle["workers"]) == 2
        and {row["name"] for row in lifecycle["workers"]} == {"simulation", "model"}
        and all(
            row["actual_wait"] is row["reaped"] is True and row["returncode"] == 0 and row["signal_sent"] is False
            for row in lifecycle["workers"]
        ),
        "Producer inputs, sources and reaped worker lifecycle must remain exact",
    )
    for row in lifecycle["workers"]:
        prefix.checked(row["log_ref"])
    producer = prefix.read(result["branch_ref"])
    model = prefix.read(result["model_ref"])
    require_completed_model(model, inputs["config"])
    simulation.load_recorded_responses(producer, model)
    counts = audit.summarize_branch(producer, "producer", inputs["config"])
    prefix.require(
        counts == result["counts"]
        and counts["adaptive_steps"] == producer["runtime"]["continuation"]["adaptive"]["control_steps_completed"]
        and model["responses_sent"] == len(producer["runtime"]["adaptive_model_exchanges"]),
        "Producer receipts and completed model exchanges must agree",
    )
    physical = audit.add_cost_bounds(
        inputs["failed_attempt"]["cumulative_physical_bounds"], audit.exact_cost_bounds(counts)
    )
    model_bounds = add_model_cost_bounds(
        known_model_bounds(inputs["config"]), model_cost_bounds(model, config=inputs["config"], receipt_final=True)
    )
    require_complete_phase_costs(result)
    prefix.require(
        result["cumulative_physical_bounds"] == physical and result["cumulative_model_bounds"] == model_bounds,
        "Producer cumulative ranges must preserve the prior failed attempt",
    )
    return physical, model_bounds


def execute(phase):
    """Run the separately registered producer or its independent replay once."""
    prefix.environment()
    prefix.require(phase in ("producer", "checker"), "A registered physical phase is required")
    inputs = admission.load_inputs()
    before, sources = check_cpu_admission(inputs), source_refs()
    check_preflight(inputs)
    previous_result = None
    prior_bounds = inputs["failed_attempt"]["cumulative_physical_bounds"]
    prior_model_bounds = known_model_bounds(inputs["config"])
    if phase == "checker":
        previous_result = prefix.read(prefix.reference(OUTPUT / "producer/result.json"))
        prior_bounds, prior_model_bounds = verified_producer_bounds(inputs, previous_result, sources)
    directory = OUTPUT / phase
    directory.mkdir(exist_ok=False)
    result = dict(
        complete=False,
        passed=False,
        phase=phase,
        control_qualified=False,
        gpu_uuid=inputs["config"]["gpu_uuid"],
        automatic_retry=False,
    )
    lifecycle = None
    try:
        simulation_dir = directory / "simulation"
        simulation_dir.mkdir()
        model_ref = None
        if phase == "producer":
            model_dir = directory / "model"
            model_dir.mkdir()
            model_ref = publish(model_dir / "registered_inputs.json", model_bundle(inputs))
        result["registration_ref"] = publish(
            directory / "registration.json",
            dict(
                phase=phase,
                inputs=admission.input_receipt(inputs),
                source_refs=sources,
                model_bundle_ref=model_ref,
                producer_branch_ref=None if previous_result is None else previous_result["branch_ref"],
                producer_model_ref=None if previous_result is None else previous_result["model_ref"],
            ),
        )
        simulation_env = physical._cpu_environment(simulation_dir)
        command = [sys.executable, "-m", MODULE]
        if phase == "producer":
            result["gpu_before"] = engine.gpu_observation()
            prefix.require(
                result["gpu_before"]["memory_mib"] == 0 and result["gpu_before"]["compute_apps"] == [],
                "The registered GPU must be idle before producer startup",
            )
            model_env = model_owner._model_environment(model_dir)
            model_env["CUDA_VISIBLE_DEVICES"] = inputs["config"]["gpu_uuid"]
            specs = {
                "simulation": dict(
                    argv=command + ["--worker", "producer"],
                    env=simulation_env,
                    log=str((directory / "simulation.log").relative_to(ROOT)),
                ),
                "model": dict(
                    argv=[inputs["prepared"]["model_config"]["model_python"], "-m", MODULE, "--worker", "model"],
                    env=model_env,
                    log=str((directory / "model.log").relative_to(ROOT)),
                ),
            }
            try:
                lifecycle = run_owned_socket_pair(specs, timeout_seconds=inputs["config"]["worker_timeout_seconds"])
            except OwnedSocketPairInterrupted as error:
                lifecycle = error.result
                result["workers_ref"] = publish(directory / "workers.json", historical._record_worker_logs(lifecycle))
                result["interrupted"] = True
                raise
            result["workers_ref"] = publish(directory / "workers.json", historical._record_worker_logs(lifecycle))
            prefix.require(lifecycle["passed"] is True, "Both model and simulation workers must exit and be reaped")
            result["model_ref"] = prefix.reference(model_dir / "result.json")
            actual_model = prefix.read(result["model_ref"])
            require_completed_model(actual_model, inputs["config"])
            result["gpu_after"] = engine.gpu_observation()
            result["gpu_released"] = (
                result["gpu_after"]["memory_mib"] == 0 and result["gpu_after"]["compute_apps"] == []
            )
            prefix.require(result["gpu_released"] is True, "The registered GPU must be released")
        else:
            spec = {"key": "checker", "argv": command + ["--worker", "checker"], "log": directory / "checker.log"}
            with physical._temporary_environment(simulation_env):
                lifecycle = run_owned_workers(
                    [spec], cwd=ROOT, timeout_seconds=inputs["config"]["worker_timeout_seconds"]
                )
            result["workers_ref"] = publish(directory / "workers.json", historical._record_worker_logs(lifecycle))
            lifecycle_check.require_closed_workers(lifecycle, ["checker"])
            result["gpu_released"] = True
        branch_ref = prefix.reference(simulation_dir / "result.json")
        branch = prefix.read(branch_ref)
        counts = audit.summarize_branch(branch, phase, inputs["config"])
        if phase == "producer":
            model = prefix.read(result["model_ref"])
            simulation.load_recorded_responses(branch, model)
            prefix.require(
                counts["adaptive_steps"] == branch["runtime"]["continuation"]["adaptive"]["control_steps_completed"]
                and model["responses_sent"] == len(branch["runtime"]["adaptive_model_exchanges"]),
                "Producer physics and accepted model exchanges must agree",
            )
        else:
            producer = prefix.read(previous_result["branch_ref"])
            counts["independent_max_absolute_difference"] = audit.compare_branches(producer, branch)
            prefix.require(
                counts["control_steps"] == previous_result["counts"]["control_steps"], "Replay count differs"
            )
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(source_refs() == sources and snapshot(ROOT)["sources"] == before, "Trial sources changed")
        result.update(complete=True, passed=True, sources_unchanged=True, branch_ref=branch_ref, counts=counts)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        try:
            _phase_costs(
                result,
                phase=phase,
                directory=directory,
                inputs=inputs,
                sources=sources,
                lifecycle=lifecycle,
                prior_bounds=prior_bounds,
                prior_model_bounds=prior_model_bounds,
            )
            if result["complete"] is True:
                require_complete_phase_costs(result)
        except Exception as error:
            result.update(
                complete=False,
                passed=False,
                cost_accounting_error={"type": type(error).__name__, "message": str(error)},
            )
        if phase == "producer" and "gpu_after" not in result:
            try:
                result["gpu_after"] = engine.gpu_observation()
                result["gpu_released"] = (
                    result["gpu_after"]["memory_mib"] == 0 and result["gpu_after"]["compute_apps"] == []
                )
            except Exception as error:
                result["gpu_after_error"] = {"type": type(error).__name__, "message": str(error)}
        print(publish(directory / "result.json", result), flush=True)
        if "cost_accounting_error" in result:
            raise ValueError("Recovery cost accounting failed")
