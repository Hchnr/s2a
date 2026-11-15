"""Run the registered training-only recovery controls with independent replay."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result
from s2a.research.native_runtime.correct_reference_continuation import run_correct_reference_continuation
from s2a.research.native_runtime.correct_reference_reset import CorrectReferenceResetRuntime
from s2a.research.wam.training.native_observations import render_anchor, save_arrays
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import managed_physical_worker as managed
from scripts.simulation.runtime import prepare_broader_recovery_pilot as pilot
from scripts.simulation.runtime import prepare_reset_correct_reference_cohort as preparation
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_current_controller_recovery as recovery
from scripts.simulation.runtime import run_reset_correct_reference as previous
from scripts.simulation.runtime import run_reset_correct_reference_cohort as cohort
from scripts.simulation.runtime import run_reset_start_feedback as original

ROOT = prefix.REPO
OUTPUT = pilot.OUTPUT
QA = ROOT / "outputs/qa/broader_recovery_pilot_final_20261116/result.json"
MODULE = "scripts.simulation.runtime.run_broader_recovery_pilot"


def source_refs():
    return pilot.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__).resolve(),
            ROOT / "tests/research/native_runtime/test_broader_recovery_pilot.py",
        )
    ]


def validate_registration(phase, index, registration, expected_sources):
    """Admit only one registered case with the checker bound to its producer."""
    prefix.require(
        phase in ("producer", "checker", "probe")
        and type(index) is int
        and 0 <= index < 24
        and (phase != "probe" or index == 0),
        "One registered case and phase are required",
    )
    cases = registration["inputs"]["cases"]
    prefix.require(
        registration["phase"] == phase
        and index in registration["eligible_cases"]
        and registration["source_refs"] == expected_sources
        and len(cases) == 24
        and cases[index]["index"] == index
        and registration["automatic_retry"] is False,
        "The registered case, retry policy and source identities must match",
    )
    producers = registration["producer_branch_refs"]
    prefix.require(
        (phase == "checker" and str(index) in producers) or (phase != "checker" and producers == {}),
        "Only an independently replayed producer may be admitted to a checker",
    )
    return cases[index]


def bind_case(identity, shared):
    """Rebuild the exact source timeline used by the registration."""
    inputs = preparation.case_inputs(identity["original_identity"], shared)
    timeline, metadata, digest = pilot.make_timeline(
        inputs, identity["trajectory_ref"], identity["variant"], identity["source"]["sign"]
    )
    prefix.require(
        timeline.periods == identity["periods"]
        and metadata == identity["timeline_metadata"]
        and digest == identity["timeline_sha256"],
        "The registered training timeline must match",
    )
    return inputs, timeline


def worker(phase, index):
    """Run one managed worker and publish its actual endpoints and commands."""
    prefix.require(
        phase in ("producer", "checker", "probe")
        and type(index) is int
        and 0 <= index < 24
        and (phase != "probe" or index == 0),
        "One registered case and phase are required",
    )
    directory = OUTPUT / phase / f"case{index:02d}" / "worker"
    directory.mkdir(parents=True, exist_ok=False)
    registration = prefix.read(prefix.reference(OUTPUT / phase / "registration.json"))

    def execute():
        identity = validate_registration(phase, index, registration, source_refs())
        shared = preparation.shared_inputs()
        prefix.require(shared["receipt"] == registration["inputs"]["shared_receipt"], "Shared assets must match")
        inputs, timeline = bind_case(identity, shared)
        recorded = None
        if phase in ("checker", "probe"):
            recovery.checker_isolation()
        if phase == "checker":
            produced = prefix.read(registration["producer_branch_refs"][str(index)])
            prefix.require(produced["case"] == identity, "The replayed producer must have the same case identity")
            cohort.validate_branch(produced, "producer", identity["periods"])
            recorded = produced["runtime"]["continuation"]
        cv2 = shared["cv2_admission"]
        cv2_receipt = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        original.require_no_simulator_import()
        reset, case = inputs["reset"], inputs["case"]
        first_poses, first_openings, _ = timeline.window(0)
        case["chunk"] = dict(anchor_tcp=reset["anchor"], effective_targets_tcp=first_poses, openings=first_openings)
        adapter = CorrectReferenceResetRuntime(
            expected=reset["expected"],
            expected_receipt=reset["expected_receipt"],
            controller=case["controller"],
            tool=case["tool"],
            anchor=reset["anchor"],
            opening=reset["opening"],
        )
        runtime_directory = directory / "runtime"
        capture_dir = runtime_directory / "capture"
        captures = []
        factory = None
        if phase == "producer":
            from s2a.research.joint_control.preview.cached_seed_preference import CachedSeedPreferredAudit
            from s2a.research.joint_control.preview.reset_feedback_policy import policy_inputs
            from s2a.research.joint_control.preview.segmented_reset_policy import build_segmented_reset_policy

            def factory(env, terminal, receipt):
                return build_segmented_reset_policy(
                    env, policy_inputs(case, terminal), receipt, adapter_factory=CachedSeedPreferredAudit
                )

        def continuation(env, terminal, observers, receipt, *, progress=None):
            capture = recovery.capture_reset(runtime_directory, case["tool"], env, terminal, render_anchor)
            captures.append(capture)
            if phase == "probe":
                return dict(segments=[], stop=None, probe_no_actions=True)
            wrapped = SimpleNamespace(
                endpoint=capture.wrap_endpoint(observers.endpoint),
                current=observers.current,
                render_cameras=observers.render_cameras,
            )
            return run_correct_reference_continuation(
                env,
                terminal,
                wrapped,
                receipt,
                timeline=timeline,
                tool=case["tool"],
                target_object=inputs["target_object"],
                phase=phase,
                score_gripper=original.measured_gripper,
                compare_score=original.compare_score,
                policy_factory=factory,
                recorded=recorded,
                progress=progress,
            )

        def progress(receipt):
            print(json.dumps(dict(case_index=index, **previous.progress_event(phase, receipt))), flush=True)

        runtime = adapter.run_feedback(
            inputs["runtime_config"],
            inputs["record"],
            inputs["record_xml"],
            runtime_directory,
            shared["asset_refs"],
            registered_seed=42,
            continuation=continuation,
            max_control_steps=1 if phase == "probe" else identity["periods"],
            progress=progress,
        )
        prefix.require(len(captures) == 1, "One actual endpoint collector is required")
        capture = captures[0]
        prefix.require(
            len(capture.samples) == runtime["control_steps_completed"] + 1, "Every executed endpoint must be captured"
        )
        arrays = capture.arrays()
        periods = previous.periods(dict(runtime=runtime))
        arrays["sent_commands"] = np.asarray([row["command"] for row in periods], dtype=np.float64).reshape(-1, 7)
        arrays_ref = save_arrays(capture_dir / "actual_trajectory.npz", arrays)
        if phase in ("checker", "probe"):
            recovery.checker_isolation()
        return dict(
            complete=True,
            passed=True,
            case=identity,
            runtime=runtime,
            cv2_admission=cv2_receipt,
            candidate_modules_unloaded=phase in ("checker", "probe"),
            requested_phase=phase,
            arrays_ref=arrays_ref,
            endpoint_receipts=capture.receipts,
            images=capture.images,
            anchor_windows=capture.anchor_windows(),
            physical_labels_from_reference=False,
            label_semantics=dict(
                action128_opening="next_measured_opening_not_controller_command",
                sent_commands="actual_seven_dimensional_controller_command",
            ),
            fresh_cases=0,
        )

    managed.run_managed_physical_worker(
        phase="checker" if phase == "probe" else phase, result_path=directory / "result.json", execute=execute
    )


def dry_run():
    """Publish the frozen zero-physics roster, leaving the simulator unloaded."""
    original.require_no_simulator_import()
    config, identity = pilot.load_inputs()
    recovery.checker_isolation()
    return publish_result(
        OUTPUT / "dry.json",
        dict(
            complete=True,
            passed=True,
            inputs=identity,
            source_refs=source_refs(),
            physics_steps=0,
            world_model_calls=config["world_model_calls"],
            gpu_calls=0,
            fresh_cases=0,
            training_authorized=False,
        ),
    )


def preflight():
    """Require an unchanged 64-worker regression and matching no-physics dry run."""
    prefix.environment()
    qa_ref, dry_ref = prefix.reference(QA), prefix.reference(OUTPUT / "dry.json")
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
        and dry["physics_steps"] == dry["world_model_calls"] == dry["gpu_calls"] == dry["fresh_cases"] == 0
        and dry["training_authorized"] is False,
        "A complete source-matched CPU regression and zero-physics preflight are required",
    )
    config, identity = pilot.load_inputs()
    prefix.require(
        dry["inputs"] == identity and dry["source_refs"] == sources, "Dry identity and source hashes must match"
    )
    return config, identity, before, sources, qa_ref, dry_ref


def verify_probe_admission(probe, identity, sources, qa_ref, dry_ref):
    """Recheck actual reset, image, source identity and child lifecycle."""
    prefix.require(
        probe["complete"] is probe["passed"] is probe["owner_closed"] is probe["sources_unchanged"] is True
        and probe["inputs"] == identity
        and probe["source_refs"] == sources
        and probe["qa_ref"] == qa_ref
        and probe["dry_ref"] == dry_ref,
        "A complete source-matched no-action probe is required",
    )
    lifecycle = prefix.read(probe["lifecycle_ref"])
    prefix.require(
        lifecycle["launch_error"] is None
        and len(lifecycle["workers"]) == 1
        and lifecycle["workers"][0]["key"] == "0"
        and lifecycle["workers"][0]["returncode"] == 0
        and lifecycle["workers"][0]["actual_wait"] is lifecycle["workers"][0]["reaped"] is True
        and lifecycle["workers"][0]["signal_sent"] is False,
        "The no-action probe worker must be actually reaped",
    )
    prefix.checked(lifecycle["workers"][0]["log_ref"])
    registration = prefix.read(probe["registration_ref"])
    prefix.require(
        registration["phase"] == "probe"
        and registration["eligible_cases"] == [0]
        and registration["inputs"] == identity
        and registration["source_refs"] == sources
        and registration["qa_ref"] == qa_ref
        and registration["dry_ref"] == dry_ref
        and registration["producer_branch_refs"] == {}
        and registration["automatic_retry"] is False,
        "The probe must use the registered training reset and source identities",
    )
    branch = prefix.read(probe["branch_ref"])
    prefix.require(branch["case"] == identity["cases"][0], "The probe must use the original case 0")
    shared = preparation.shared_inputs()
    case, timeline = bind_case(identity["cases"][0], shared)
    checked = recovery.verify_probe_branch(branch, case, timeline)
    prefix.require(checked == probe["capture_check"], "The independent probe image check must remain reproducible")


def probe_run():
    """Execute one reset and current-image capture with zero control steps."""
    config, identity, before, sources, qa_ref, dry_ref = preflight()
    directory = OUTPUT / "probe"
    directory.mkdir(parents=True, exist_ok=False)
    registration_ref = publish_result(
        directory / "registration.json",
        dict(
            phase="probe",
            inputs=identity,
            source_refs=sources,
            qa_ref=qa_ref,
            dry_ref=dry_ref,
            eligible_cases=[0],
            producer_branch_refs={},
            automatic_retry=False,
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        owner_closed=False,
        sources_unchanged=False,
        inputs=identity,
        source_refs=sources,
        qa_ref=qa_ref,
        dry_ref=dry_ref,
        registration_ref=registration_ref,
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        lifecycle = run_owned_workers(
            [
                dict(
                    key="0",
                    argv=[sys.executable, "-m", MODULE, "--worker", "probe", "0"],
                    log=directory / "case00/worker.log",
                )
            ],
            cwd=ROOT,
            timeout_seconds=config["worker_timeout_seconds"],
        )
        for item in lifecycle["workers"]:
            item["log_ref"] = prefix.reference(item.pop("log"))
        result["lifecycle_ref"] = publish_result(directory / "workers.json", lifecycle)
        child = lifecycle["workers"][0] if len(lifecycle["workers"]) == 1 else {}
        result["owner_closed"] = child.get("actual_wait") is child.get("reaped") is True
        prefix.require(
            lifecycle["launch_error"] is None
            and result["owner_closed"]
            and child["returncode"] == 0
            and child["signal_sent"] is False,
            "The no-action probe worker must close successfully",
        )
        result["branch_ref"] = prefix.reference(directory / "case00/worker/result.json")
        branch = prefix.read(result["branch_ref"])
        prefix.require(branch["case"] == identity["cases"][0], "Probe case identity must match")
        shared = preparation.shared_inputs()
        case, timeline = bind_case(identity["cases"][0], shared)
        result["capture_check"] = recovery.verify_probe_branch(branch, case, timeline)
        _config, after = pilot.load_inputs()
        result["sources_unchanged"] = (
            snapshot(ROOT)["sources"] == before and source_refs() == sources and after == identity
        )
        result["complete"] = result["passed"] = result["owner_closed"] and result["sources_unchanged"]
        verify_probe_admission(result, identity, sources, qa_ref, dry_ref)
    except Exception as error:
        result.update(complete=False, passed=False, error=dict(type=type(error).__name__, message=str(error)))
    print(publish_result(directory / "result.json", result), flush=True)
    if not result["passed"]:
        raise SystemExit("The registered no-action probe did not pass")
