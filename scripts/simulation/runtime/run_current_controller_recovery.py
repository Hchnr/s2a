"""Qualify fixed training-only recovery trajectories with the current controller."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from s2a.geometry import pose_error
from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result
from s2a.references.training_recovery_timeline import training_recovery_timeline
from s2a.research.native_runtime.correct_reference_continuation import run_correct_reference_continuation
from s2a.research.native_runtime.correct_reference_reset import CorrectReferenceResetRuntime
from s2a.research.native_runtime.training_recovery_capture import TrainingRecoveryCapture
from s2a.research.wam.training.native.native_training import state_sha256
from s2a.research.wam.training.native_observations import render_anchor, save_arrays
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import managed_physical_worker as managed
from scripts.simulation.runtime import prepare_reset_correct_reference_cohort as preparation
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_cached_seed_preference as base
from scripts.simulation.runtime import run_reset_correct_reference as previous
from scripts.simulation.runtime import run_reset_correct_reference_cohort as cohort
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.verification.response_models.forward.verify_training_recovery_labels import check_labels
from scripts.verification.wam.training.verify_settled_native_training import verify_canvas

ROOT = prefix.REPO
CONFIG = ROOT / "configs/wam/training/recovery/current_controller_recovery_20261116.json"
PLAN = ROOT / "plans/v0.0.1/wam/training/recovery/current_controller_recovery_20261116.md"
OUTPUT = ROOT / "outputs/current_controller_recovery_20261116"
QA = ROOT / "outputs/qa/current_controller_recovery_final_20261116/result.json"
MODULE = "scripts.simulation.runtime.run_current_controller_recovery"
OFFSETS = (0.0, 0.02, -0.02)
PERIODS = (220, 310, 239)
TASKS = (8, 9, 6)
ANCHORS = (0, 16, 56)
TRAINING_NAMES = ("task0_native_demo_8", "task1_native_demo_1", "task2_native_demo_2")
REGISTERED_REFS = {
    "prior_recovery_config": dict(
        path="configs/wam/training/recovery/training_state_recovery_qualification_20261025.json",
        sha256="518e2705fad29a025105c93988c3712df357f4265628079d5c71a56666724ab5",
    ),
    "training_preparation": dict(
        path="outputs/wam_stability_20261021/clean_residual_training_20261023/recovery_20261023/preparation/prepared.json",
        sha256="e6fab51219fba6d108bd6b858f2467a9a0e54c5fc22890f8abd7c318742f8e54",
    ),
    "correct_reference_cohort": dict(
        path="configs/tracking/joint_control/reset_correct_reference_cohort_20261116.json",
        sha256="b4f9da8ff26a3e972b9cd996a36a0aff0285050a71e71eb4e16cf27d6a812312",
    ),
    "current_controller_audit": dict(
        path="outputs/cached_seed_preference_20261116/audit.json",
        sha256="ae8358fc25a9675d208f91cd8e7c52f4d20f120e9c314ee18755b97472744525",
    ),
}


def source_refs():
    """Bind current code plus the already validated fixed controller closure."""
    return base.source_refs() + [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__).resolve(),
            ROOT / "src/s2a/references/training_recovery_timeline.py",
            ROOT / "src/s2a/research/native_runtime/training_recovery_capture.py",
            ROOT / "src/s2a/research/wam/training/native/training_state_recovery.py",
            ROOT / "src/s2a/research/wam/training/native_observations.py",
            ROOT / "scripts/verification/response_models/forward/verify_training_recovery_labels.py",
            ROOT / "scripts/verification/wam/training/verify_settled_native_training.py",
            ROOT / "tests/unit/references/test_training_recovery_timeline.py",
            ROOT / "tests/research/native_runtime/test_training_recovery_capture.py",
            ROOT / "tests/research/native_runtime/test_current_controller_recovery.py",
        )
    ]


def validate_config(config):
    fixed = dict(
        schema="s2a.current-controller-training-recovery.v1",
        output="outputs/current_controller_recovery_20261116",
        case_indices=[0, 1, 2],
        offsets_m=list(OFFSETS),
        perturb_steps=16,
        return_steps=40,
        anchors=list(ANCHORS),
        future_window_steps=40,
        expected_periods_per_task=list(PERIODS),
        registered_cases=9,
        max_periods_per_phase=2307,
        max_inner_steps_per_phase=57675,
        max_workers_per_phase=9,
        max_models_per_phase=27,
        max_contexts_per_phase=27,
        max_baseline_responses_per_period=480,
        max_quality_responses_per_period=320,
        max_gripper_sequences_per_period=6561,
        control_dt_s=0.05,
        inner_dt_s=0.002,
        position_cap_m=0.01,
        rotation_cap_rad=0.05,
        joint_buffer_rad=0.05,
        worker_timeout_seconds=3600,
        world_model_calls=0,
        fits=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    refs = tuple(REGISTERED_REFS)
    prefix.require(
        isinstance(config, dict)
        and set(config) == set(fixed) | set(refs)
        and all(type(config[key]) is type(value) and config[key] == value for key, value in fixed.items()),
        "The fixed training-only roster, gates and resources are required",
    )
    for key in refs:
        prefix.require(config[key] == REGISTERED_REFS[key], "Each registered source identity must remain fixed")
        preparation.configured_ref(config[key])


def timeline_digest(timeline):
    digest = hashlib.sha256()
    for step in range(timeline.periods):
        poses, openings, _metadata = timeline.window(step)
        digest.update(np.ascontiguousarray(poses[0], dtype=np.float64).tobytes())
        digest.update(np.asarray(openings[0], dtype=np.float64).tobytes())
    return digest.hexdigest()


def make_timeline(case, trajectory_ref, offset):
    """Bind the oracle to native measured endpoints and the fixed tool."""
    with np.load(prefix.checked(trajectory_ref), allow_pickle=False) as data:
        source = np.asarray(data["source_pose"], dtype=float).copy()
        openings = np.asarray(data["opening"], dtype=float).copy()
    tool = np.asarray(case["case"]["tool"], dtype=float)
    np.testing.assert_allclose(source[0], case["reset"]["expected"]["source_pose"], atol=1e-10, rtol=0)
    prefix.require(len(source) == case["spec"]["periods"] + 1, "The full native endpoint roster is required")
    for step in range(case["spec"]["periods"]):
        pose, opening, _metadata = case["timeline"].window(step)
        np.testing.assert_allclose(pose[0], source[step + 1] @ tool, atol=1e-10, rtol=0)
        prefix.require(opening[0] == openings[step + 1], "The original and current opening references must match")
    timeline, metadata = training_recovery_timeline(source, openings, tool, offset_m=offset)
    return timeline, metadata, timeline_digest(timeline)


def load_inputs():
    prefix.environment()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    validate_config(config)
    refs = {
        key: preparation.configured_ref(config[key])
        for key in (
            "prior_recovery_config",
            "training_preparation",
            "correct_reference_cohort",
            "current_controller_audit",
        )
    }
    prior, trained, audit = (
        prefix.read(refs[key]) for key in ("prior_recovery_config", "training_preparation", "current_controller_audit")
    )
    original_cohort, shared = preparation.load_inputs(), preparation.shared_inputs()
    prefix.require(
        original_cohort["config_ref"] == refs["correct_reference_cohort"]
        and prior["base_x_offsets_m"] == list(OFFSETS)
        and prior["perturb_steps"] == 16
        and prior["return_steps"] == 40
        and prior["physics_steps"] == 2307
        and audit["complete"] is audit["passed"] is True
        and len(audit["cases"]) == 18
        and all(audit["cases"][i]["qualified"] is audit["cases"][i]["replayed"] is True for i in range(3)),
        "The exact original recovery scope and current three-case positive control are required",
    )
    forbidden = set(trained["exclusion"]["forbidden_initial_state_sha256"])
    training_ids = {
        (
            row["source_identity"]["task_index"],
            row["source_identity"]["demo"],
            row["source_identity"]["initial_state_sha256"],
        )
        for row in trained["training_anchors"]
        if row["anchor_frame"] == 0
    }
    cases = []
    for task_index in range(3):
        origin = original_cohort["cases"][task_index]
        record = prior["records"][task_index]["record"]
        source_ref = prior["records"][task_index]["trajectory_ref"]
        prefix.checked(source_ref)
        prefix.checked(prior["records"][task_index]["case_ref"])
        manifest = prefix.read(origin["refs"]["manifest"])
        initial_sha = record["initial_state_sha256"]
        prefix.require(
            record["name"] == TRAINING_NAMES[task_index]
            and record["task_id"] == origin["spec"]["task_id"] == TASKS[task_index]
            and initial_sha == manifest["record"]["initial_state_sha256"]
            and (task_index, record["demo"], initial_sha) in training_ids
            and initial_sha not in forbidden,
            "Every reset must be the same registered, non-held-out training source",
        )
        case = preparation.case_inputs(origin, shared)
        for offset_index, offset in enumerate(OFFSETS):
            timeline, metadata, digest = make_timeline(case, source_ref, offset)
            prefix.require(timeline.periods == PERIODS[task_index], "The original 063 physical length must match")
            cases.append(
                dict(
                    index=3 * task_index + offset_index,
                    task_index=task_index,
                    task_id=TASKS[task_index],
                    seed=42,
                    offset_m=offset,
                    periods=timeline.periods,
                    source_name=record["name"],
                    initial_sha256=initial_sha,
                    trajectory_ref=source_ref,
                    original_identity=origin,
                    timeline_sha256=digest,
                    timeline_metadata=metadata,
                )
            )
    prefix.require(len(cases) == 9 and sum(row["periods"] for row in cases) == 2307, "Full nine-case budget required")
    identity = dict(config_ref=config_ref, source_refs=refs, shared_receipt=shared["receipt"], cases=cases)
    return config, identity


def checker_isolation():
    base.checker_isolation()


def dry_run():
    original.require_no_simulator_import()
    config, identity = load_inputs()
    checker_isolation()
    return publish_result(
        OUTPUT / "dry.json",
        dict(
            complete=True,
            passed=True,
            inputs=identity,
            source_refs=source_refs(),
            physics_steps=0,
            world_model_calls=config["world_model_calls"],
            fresh_cases=0,
        ),
    )


def worker(phase, index):
    prefix.require(
        phase in ("producer", "checker") and type(index) is int and 0 <= index < 9, "One case and phase required"
    )
    directory = OUTPUT / phase / f"case{index:02d}" / "worker"
    directory.mkdir(parents=True, exist_ok=False)
    registration = prefix.read(prefix.reference(OUTPUT / phase / "registration.json"))

    def execute():
        prefix.require(
            registration["phase"] == phase
            and index in registration["eligible_cases"]
            and registration["source_refs"] == source_refs(),
            "The registered case and source identities must match",
        )
        identity = registration["inputs"]["cases"][index]
        shared = preparation.shared_inputs()
        prefix.require(shared["receipt"] == registration["inputs"]["shared_receipt"], "Shared assets must match")
        inputs = preparation.case_inputs(identity["original_identity"], shared)
        timeline, metadata, digest = make_timeline(inputs, identity["trajectory_ref"], identity["offset_m"])
        prefix.require(
            timeline.periods == identity["periods"]
            and metadata == identity["timeline_metadata"]
            and digest == identity["timeline_sha256"],
            "The registered training timeline must match",
        )
        recorded = None
        if phase == "checker":
            checker_isolation()
            produced = prefix.read(registration["producer_branch_refs"][str(index)])
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
        capture_dir = directory / "capture"
        capture_dir.mkdir()
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
            capture = TrainingRecoveryCapture(case["tool"], capture_dir, render_anchor)
            capture.append(env, terminal["sample"], terminal["endpoint_receipt"])
            captures.append(capture)
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
            directory / "runtime",
            shared["asset_refs"],
            registered_seed=42,
            continuation=continuation,
            max_control_steps=identity["periods"],
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
        if phase == "checker":
            checker_isolation()
        return dict(
            complete=True,
            passed=True,
            case=identity,
            runtime=runtime,
            cv2_admission=cv2_receipt,
            candidate_modules_unloaded=phase == "checker",
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

    managed.run_managed_physical_worker(phase=phase, result_path=directory / "result.json", execute=execute)


def phase_run(phase):
    prefix.environment()
    prefix.require(phase in ("producer", "checker"), "One registered physical phase is required")
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
        and dry["physics_steps"] == 0,
        "A complete source-matched CPU regression and zero-physics preflight are required",
    )
    config, identity = load_inputs()
    prefix.require(
        dry["inputs"] == identity and dry["source_refs"] == sources, "Dry identity and source hashes must match"
    )
    producer_refs = {}
    eligible = [item["index"] for item in identity["cases"]]
    if phase == "checker":
        producer = prefix.read(prefix.reference(OUTPUT / "producer/result.json"))
        prefix.require(
            producer["owner_closed"] is producer["sources_unchanged"] is True, "The producer owner must close"
        )
        eligible = [index for index in eligible if producer["cases"][str(index)]["status"] == "closed"]
        producer_refs = {str(index): producer["cases"][str(index)]["branch_ref"] for index in eligible}
        for ref in producer_refs.values():
            prefix.checked(ref)
    directory = OUTPUT / phase
    directory.mkdir(parents=True, exist_ok=False)
    registration_ref = publish_result(
        directory / "registration.json",
        dict(
            phase=phase,
            inputs=identity,
            source_refs=sources,
            qa_ref=qa_ref,
            dry_ref=dry_ref,
            eligible_cases=eligible,
            producer_branch_refs=producer_refs,
            automatic_retry=False,
        ),
    )
    specs = [
        dict(
            key=str(index),
            argv=[sys.executable, "-m", MODULE, "--worker", phase, str(index)],
            log=directory / f"case{index:02d}/worker.log",
        )
        for index in eligible
    ]
    lifecycle = (
        run_owned_workers(specs, cwd=ROOT, timeout_seconds=config["worker_timeout_seconds"])
        if specs
        else dict(workers=[], launch_error=None)
    )
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    lifecycle_ref = publish_result(directory / "workers.json", lifecycle)
    by_index = {int(item["key"]): item for item in lifecycle["workers"]}
    cases = {}
    result = dict(
        complete=False,
        passed=False,
        owner_closed=False,
        sources_unchanged=False,
        phase=phase,
        registration_ref=registration_ref,
        lifecycle_ref=lifecycle_ref,
        cases=cases,
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        for spec in identity["cases"]:
            index = spec["index"]
            entry = cases[str(index)] = dict(status="not_run", expected_periods=spec["periods"])
            if index not in eligible:
                entry["reason"] = "producer_not_closed"
                continue
            info = by_index.get(index)
            path = directory / f"case{index:02d}/worker/result.json"
            if path.is_file():
                entry["available_branch_ref"] = prefix.reference(path)
            try:
                prefix.require(
                    info is not None
                    and info["returncode"] == 0
                    and info["actual_wait"] is info["reaped"] is True
                    and info["signal_sent"] is False
                    and path.is_file(),
                    "Owned worker must close",
                )
                ref = prefix.reference(path)
                branch = prefix.read(ref)
                prefix.require(branch["case"] == spec, "The case identity must remain fixed")
                counts = cohort.validate_branch(branch, phase, spec["periods"])
                if phase == "checker":
                    produced = prefix.read(producer_refs[str(index)])
                    counts["replay_max_difference"] = comparison.compare_periods(
                        previous.periods(produced), previous.periods(branch)
                    )
                    prefix.require(
                        produced["runtime"]["continuation"]["stop"] == branch["runtime"]["continuation"]["stop"],
                        "The reported producer stop must be retained",
                    )
                entry.update(status="closed", branch_ref=ref, counts=counts)
            except Exception as error:
                entry.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
        count = sum(item.get("counts", {}).get("control_steps", 0) for item in cases.values())
        prefix.require(count <= config["max_periods_per_phase"], "The physical budget cannot be exceeded")
        owner_closed = len(lifecycle["workers"]) == len(eligible) and all(
            item["actual_wait"] is item["reaped"] is True for item in lifecycle["workers"]
        )
        _config, after = load_inputs()
        unchanged = snapshot(ROOT)["sources"] == before and source_refs() == sources and after == identity
        result.update(
            owner_closed=owner_closed,
            sources_unchanged=unchanged,
            complete=owner_closed and unchanged and lifecycle["launch_error"] is None,
            passed=owner_closed
            and unchanged
            and lifecycle["launch_error"] is None
            and len(cases) == 9
            and all(item["status"] == "closed" for item in cases.values()),
        )
    except Exception as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
    print(publish_result(directory / "result.json", result), flush=True)
    if not result["passed"]:
        raise SystemExit("The physical phase did not close all registered cases")


def verify_capture(branch, expected_periods, tool, timeline):
    """Independently reconstruct labels and check actual endpoint/image binding."""
    rows = previous.periods(branch)
    receipts, images = branch["endpoint_receipts"], branch["images"]
    prefix.require(len(receipts) == len(rows) + 1 <= expected_periods + 1, "Actual endpoint count must match")
    with np.load(prefix.checked(branch["arrays_ref"]), allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]).copy() for key in archive.files}
    prefix.require(len(data["sim_states"]) == len(receipts), "Every actual endpoint must be saved")
    prefix.require(data["sent_commands"].shape == (len(rows), 7), "Actual controller commands must be retained")
    offsets = data["gripper_current_action_offsets"]
    flat = data["gripper_current_action_flat"]
    prefix.require(
        offsets.shape == (len(receipts) + 1,)
        and offsets[0] == 0
        and offsets[-1] == len(flat)
        and np.all(np.diff(offsets) >= 0),
        "Packed actual motor caches must be complete",
    )
    for index, receipt in enumerate(receipts):
        motor = np.ascontiguousarray(flat[offsets[index] : offsets[index + 1]])
        prefix.require(
            receipt["endpoint_step"] == index
            and state_sha256(data["sim_states"][index]) == receipt["sim_state_sha256"]
            and float(data["sim_time"][index]) == receipt["sim_time"]
            and bool(data["native_success"][index]) == receipt["native_success"]
            and list(motor.shape) == receipt["gripper_current_action_shape"]
            and str(motor.dtype) == receipt["gripper_current_action_dtype"]
            and hashlib.sha256(motor.tobytes()).hexdigest() == receipt["gripper_current_action_sha256"],
            "Each endpoint state, clock and native outcome must match its receipt",
        )
        if index:
            row = rows[index - 1]
            np.testing.assert_array_equal(data["sim_states"][index], row["actual_state"])
            np.testing.assert_array_equal(data["sent_commands"][index - 1], row["command"])
            prefix.require(
                receipt == row["endpoint_receipt"] and bool(data["native_success"][index]) == row["task_success"],
                "Captured and physical endpoint identities and outcomes must match",
            )
            target, reference_opening, _metadata = timeline.window(index - 1)
            np.testing.assert_allclose(
                pose_error(target[0], data["source_pose"][index] @ tool), row["tcp_error"], atol=1e-10, rtol=0
            )
            score = row["completed_feedback"]["score"]
            np.testing.assert_allclose(
                [data["opening"][index], reference_opening[0]],
                [score["opening"], score["opening_reference"]],
                atol=1e-10,
                rtol=0,
            )
    if rows:
        check_labels(data, tool)
    else:
        prefix.require("action128" not in data, "A reset-only prefix cannot fabricate action labels")
    expected_anchors = [step for step in ANCHORS if step <= len(rows)]
    prefix.require(
        [item["frame_index"] for item in images] == expected_anchors, "Reached image anchors must be retained"
    )
    checked_images = [verify_canvas(image, data, receipts[image["frame_index"]]) for image in images]
    windows = [
        dict(
            step=step,
            image_reached=step in expected_anchors,
            full_h40=step in expected_anchors and len(rows) >= step + 40,
        )
        for step in ANCHORS
    ]
    prefix.require(branch["anchor_windows"] == windows, "Future-window availability must follow actual completed steps")
    offset_at_16 = None if len(rows) < 16 else (data["source_pose"][16, :3, 3] - data["source_pose"][0, :3, 3]).tolist()
    return_at_56 = (
        None
        if len(rows) < 56
        else float(np.linalg.norm(data["source_pose"][56, :3, 3] - data["source_pose"][0, :3, 3]))
    )
    return dict(
        endpoint_count=len(receipts),
        labels_reconstructed=bool(rows),
        images=checked_images,
        anchor_windows=windows,
        observed_offset_m=offset_at_16,
        return_position_error_m=return_at_56,
    ), data


def audit():
    prefix.environment()
    before, sources = snapshot(ROOT)["sources"], source_refs()
    config, identity = load_inputs()
    producer_ref, checker_ref = (prefix.reference(OUTPUT / phase / "result.json") for phase in ("producer", "checker"))
    producer, checker = prefix.read(producer_ref), prefix.read(checker_ref)
    prefix.require(
        producer["owner_closed"]
        is checker["owner_closed"]
        is producer["sources_unchanged"]
        is checker["sources_unchanged"]
        is True,
        "Both physical owners must close",
    )
    registrations = [prefix.read(row["registration_ref"]) for row in (producer, checker)]
    prefix.require(
        all(row["inputs"] == identity and row["source_refs"] == source_refs() for row in registrations),
        "Both physical phases must retain their inputs and source hashes",
    )
    shared, cases = preparation.shared_inputs(), []
    for spec in identity["cases"]:
        index = spec["index"]
        p, c = producer["cases"][str(index)], checker["cases"][str(index)]
        entry = dict(
            case=spec,
            producer_status=p["status"],
            checker_status=c["status"],
            replayed=False,
            evidence_complete=False,
            qualified=False,
        )
        if p["status"] == c["status"] == "closed":
            try:
                left, right = prefix.read(p["branch_ref"]), prefix.read(c["branch_ref"])
                cohort.validate_branch(left, "producer", spec["periods"])
                cohort.validate_branch(right, "checker", spec["periods"])
                differences = comparison.compare_periods(previous.periods(left), previous.periods(right))
                prefix.require(differences == c["counts"]["replay_max_difference"], "Replay difference must match")
                case = preparation.case_inputs(spec["original_identity"], shared)
                timeline, metadata, digest = make_timeline(case, spec["trajectory_ref"], spec["offset_m"])
                prefix.require(
                    metadata == spec["timeline_metadata"] and digest == spec["timeline_sha256"], "Timeline changed"
                )
                left_check, left_data = verify_capture(left, spec["periods"], case["case"]["tool"], timeline)
                right_check, right_data = verify_capture(right, spec["periods"], case["case"]["tool"], timeline)
                prefix.require(set(left_data) == set(right_data), "Both captured array schemas must match")
                for key in left_data:
                    np.testing.assert_allclose(left_data[key], right_data[key], atol=1e-10, rtol=0)
                prefix.require(
                    left["endpoint_receipts"] == right["endpoint_receipts"]
                    and [x["canvas_rgb_sha256"] for x in left["images"]]
                    == [x["canvas_rgb_sha256"] for x in right["images"]],
                    "Independent endpoint and image identities must agree",
                )
                summary = left["runtime"]["continuation"]["summary"]
                valid = cohort.qualifies(summary, True, config)
                entry.update(
                    replayed=True,
                    evidence_complete=True,
                    qualified=valid,
                    summary=summary,
                    replay_max_difference=differences,
                    producer_ref=p["branch_ref"],
                    checker_ref=c["branch_ref"],
                    capture=left_check,
                    checker_capture=right_check,
                    observed_h40_anchors=[x["step"] for x in left_check["anchor_windows"] if x["full_h40"]],
                    qualified_teacher_h40_anchors=[
                        x["step"] for x in left_check["anchor_windows"] if x["full_h40"] and valid
                    ],
                )
            except Exception as error:
                entry["error"] = dict(type=type(error).__name__, message=str(error))
        cases.append(entry)
    _config, after_identity = load_inputs()
    unchanged = snapshot(ROOT)["sources"] == before and source_refs() == sources and after_identity == identity
    complete = unchanged and len(cases) == 9 and all(item["evidence_complete"] for item in cases)
    zero = [item for item in cases if item["case"]["offset_m"] == 0]
    nonzero = [item for item in cases if item["case"]["offset_m"] != 0]
    result = dict(
        complete=True,
        passed=complete,
        sources_unchanged=unchanged,
        zero_offset_qualified=sum(item["qualified"] for item in zero),
        nonzero_qualified=sum(item["qualified"] for item in nonzero),
        registered_cases=9,
        cases=cases,
        training_authorized=False,
        producer_ref=producer_ref,
        checker_ref=checker_ref,
        physics_steps=0,
        world_model_calls=0,
        fits=0,
        fresh_cases=0,
    )
    print(publish_result(OUTPUT / "audit.json", result), flush=True)
    if not complete:
        raise SystemExit("One or more recovery cases failed independent evidence checks")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry", action="store_true")
    group.add_argument("--phase", choices=("producer", "checker"))
    group.add_argument("--worker", nargs=2, metavar=("PHASE", "CASE"))
    group.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    prefix.environment()
    if args.dry:
        print(dry_run(), flush=True)
    elif args.phase:
        phase_run(args.phase)
    elif args.worker:
        worker(args.worker[0], int(args.worker[1]))
    else:
        audit()


if __name__ == "__main__":
    main()
