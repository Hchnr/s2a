"""Run the registered training-only recovery controls with independent replay."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.research.native_runtime.correct_reference_continuation import run_correct_reference_continuation
from s2a.research.native_runtime.correct_reference_reset import CorrectReferenceResetRuntime
from s2a.research.wam.training.native_observations import render_anchor, save_arrays
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
