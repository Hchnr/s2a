"""Replay registered recovery commands to capture current event images on CPU."""

import argparse
import json
import sys
from pathlib import Path
from types import FunctionType

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result
from s2a.research.native_runtime import development_prefix_runtime_bounded as bounded
from s2a.research.native_runtime import development_reset_runtime as reset
from s2a.research.native_runtime.correct_reference_reset import CorrectReferenceResetRuntime
from s2a.research.native_runtime.development_prefix_replay import saved_prefix
from s2a.research.native_runtime.recovery_event_image_replay import event_image_replay
from s2a.research.native_runtime.reset_bounded_runtime import runtime_unchanged
from s2a.research.wam.training.native_observations import render_anchor
from s2a.research.wam.training.recovery_h40_contract import audit_window
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import managed_physical_worker as managed
from scripts.simulation.runtime import prepare_reset_correct_reference_cohort as preparation
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_current_controller_recovery as recovery
from scripts.simulation.runtime import run_reset_correct_reference as previous
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.verification.wam.training.verify_settled_native_training import verify_canvas

ROOT = prefix.REPO
CONFIG = ROOT / "configs/wam/training/recovery/recovery_event_images_20261116.json"
PLAN = ROOT / "plans/v0.0.1/wam/training/recovery/recovery_event_images_20261116.md"
OUTPUT = ROOT / "outputs/recovery_event_images_20261116"
QA = ROOT / "outputs/qa/recovery_event_images_final_20261116/result.json"
MODULE = "scripts.simulation.runtime.run_recovery_event_images"
ANCHORS = ((89, 120), (89, 120), (89, 120), (212,), (212,), (212,), (147,), (147,), (147,))
REFERENCES = {
    "recovery_audit": (
        "outputs/current_controller_recovery_repair_20261116/audit.json",
        "a10dc09801d90efd92933e536d04a1e71ea1da75b724105dc384205c4d61c7f6",
    ),
    "h40_audit": (
        "outputs/recovery_h40_contract_20261116/result.json",
        "0753c4c30f432e6b09e103152238cb4f6f47dffa30ce76e60636e59f2329dcec",
    ),
    "event_audit": (
        "outputs/recovery_event_coverage_20261116/result.json",
        "0618788ddbde31d9128f0b8de5b1daeb0dc838a88d7fdab81e61d0b3561aff29",
    ),
}


def validate_config(config):
    expected = dict(
        schema="s2a.recovery-event-images.v1",
        output="outputs/recovery_event_images_20261116",
        case_indices=list(range(9)),
        anchors_by_case=[list(item) for item in ANCHORS],
        first_success_lead_periods=8,
        horizon=40,
        seed=42,
        probe_cases=[0],
        max_probe_control_periods=0,
        max_probe_inner_steps=0,
        max_probe_models=3,
        max_probe_contexts=3,
        max_workers=9,
        max_replay_control_periods=1437,
        max_replay_inner_steps=35925,
        max_replay_models=27,
        max_replay_contexts=27,
        max_saved_event_images=12,
        max_unsaved_camera_renders=183,
        worker_timeout_seconds=3600,
        world_model_calls=0,
        gpu_calls=0,
        fits=0,
        fresh_cases=0,
        training_authorized=False,
        automatic_retry=False,
    )
    prefix.require(
        isinstance(config, dict)
        and set(config) == set(expected) | set(REFERENCES)
        and all(type(config[key]) is type(value) and config[key] == value for key, value in expected.items()),
        "The exact nine-case event capture roster and bounded resources are required",
    )
    for key, (path, sha) in REFERENCES.items():
        prefix.require(config[key] == dict(path=path, sha256=sha), "The registered audit source must match")


def source_refs():
    return recovery.source_refs() + [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__).resolve(),
            ROOT / "src/s2a/research/native_runtime/recovery_event_image_replay.py",
            ROOT / "tests/research/native_runtime/test_recovery_event_image_replay.py",
            ROOT / "tests/research/native_runtime/test_recovery_event_image_runtime.py",
        )
    ]


def load_inputs():
    prefix.environment()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    validate_config(config)
    refs = {key: preparation.configured_ref(config[key]) for key in REFERENCES}
    prior, h40, events = (prefix.read(refs[key]) for key in REFERENCES)
    _old_config, old = recovery.load_inputs()
    prefix.require(
        prior["complete"]
        is prior["passed"]
        is h40["complete"]
        is h40["passed"]
        is events["complete"]
        is events["passed"]
        is True
        and prior["registered_cases"] == len(prior["cases"]) == len(events["cases"]) == 9
        and h40["window_count"] == 27
        and events["validation_identity"]["identity_sets_disjoint"] is True
        and h40["training_authorized"] is events["training_authorized"] is False,
        "All original control, label and validation identity audits are required",
    )
    cases = []
    for index in range(9):
        row, event, source = prior["cases"][index], events["cases"][index], old["cases"][index]
        prefix.require(
            row["case"] == source
            and event["case_index"] == source["index"] == index
            and event["source_initial_sha256"] == source["initial_sha256"]
            and row["qualified"] is row["replayed"] is row["evidence_complete"] is True,
            "Each registered case must retain its independently replayed training identity",
        )
        producer = prefix.read(row["producer_ref"])
        prefix.checked(row["checker_ref"])
        prefix.require(
            producer["case"] == source
            and producer["arrays_ref"] == event["trajectory_ref"]
            and producer["complete"] is producer["passed"] is True,
            "The physical producer and its trajectory must match the event audit",
        )
        success = event["first_success_action_index"]
        prefix.require(type(success) is int and 0 <= success < source["periods"], "A real success endpoint is required")
        success_anchor = min(max(0, success - 7), source["periods"] - 40)
        chosen = [success_anchor]
        if event["selected_anchor"] is not None:
            chosen.append(event["selected_anchor"]["anchor_step"])
        prefix.require(
            sorted(chosen) == list(ANCHORS[index]), "The recorded offline selection must match the fixed anchors"
        )
        for step in ANCHORS[index]:
            prefix.require(
                step + 40 <= source["periods"] and producer["endpoint_receipts"][step]["endpoint_step"] == step,
                "Each image must have a real original endpoint and a complete existing H40",
            )
        with np.load(prefix.checked(producer["arrays_ref"]), allow_pickle=False) as archive:
            steps = max(ANCHORS[index])
            commands = np.asarray(archive["sent_commands"][:steps], dtype=np.float64)
            recorded = np.asarray([item["command"] for item in previous.periods(producer)[:steps]], dtype=np.float64)
            prefix.require(
                commands.shape == (steps, 7) and np.array_equal(commands, recorded),
                "Only original actually dispatched commands may be replayed",
            )
        cases.append(
            dict(
                index=index,
                source=source,
                producer_ref=row["producer_ref"],
                checker_ref=row["checker_ref"],
                arrays_ref=producer["arrays_ref"],
                anchors=list(ANCHORS[index]),
                replay_periods=max(ANCHORS[index]),
                first_success_action_index=success,
                success_anchor=success_anchor,
            )
        )
    prefix.require(
        sum(case["replay_periods"] for case in cases) == config["max_replay_control_periods"]
        and sum(len(case["anchors"]) for case in cases) == config["max_saved_event_images"]
        and sum((case["replay_periods"] + 7) // 8 for case in cases) == config["max_unsaved_camera_renders"],
        "The full replay and image budgets must be exact",
    )
    return config, dict(config_ref=config_ref, refs=refs, shared_receipt=old["shared_receipt"], cases=cases)


def dry_run():
    original.require_no_simulator_import()
    config, identity = load_inputs()
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
            fresh_cases=0,
        ),
    )


def event_runtime(anchors):
    prefix.require(runtime_unchanged(), "The pinned native runtime must remain unchanged")
    source = bounded.run_prefix_runtime
    namespace = source.__globals__.copy()
    namespace.update(
        validate_registered_runtime=reset.validate_development_reset,
        replay_saved_prefix=event_image_replay(bounded.replay_saved_prefix, anchors),
    )
    runner = FunctionType(source.__code__, namespace, source.__name__, source.__defaults__, source.__closure__)
    runner.__kwdefaults__ = dict(source.__kwdefaults__ or {})
    return runner


def worker(phase, index):
    prefix.require(
        phase in ("probe", "replay") and type(index) is int and 0 <= index < 9 and (phase != "probe" or index == 0),
        "One registered probe or replay case is required",
    )
    directory = OUTPUT / phase / f"case{index:02d}" / "worker"
    directory.mkdir(parents=True, exist_ok=False)
    registration = prefix.read(prefix.reference(OUTPUT / phase / "registration.json"))

    def execute():
        prefix.require(
            registration["phase"] == phase
            and index in registration["eligible_cases"]
            and registration["source_refs"] == source_refs(),
            "The registered worker and source identities must match",
        )
        recovery.checker_isolation()
        identity = registration["inputs"]["cases"][index]
        shared = preparation.shared_inputs()
        prefix.require(shared["receipt"] == registration["inputs"]["shared_receipt"], "Shared assets must match")
        inputs = preparation.case_inputs(identity["source"]["original_identity"], shared)
        producer = prefix.read(identity["producer_ref"])
        prefix.require(producer["arrays_ref"] == identity["arrays_ref"], "Original command source must match")
        cv2 = shared["cv2_admission"]
        cv2_receipt = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        original.require_no_simulator_import()
        runtime_path = directory / "runtime"

        def progress(receipt):
            print(json.dumps(dict(case_index=index, **previous.progress_event(phase, receipt))), flush=True)

        if phase == "probe":
            state, case = inputs["reset"], inputs["case"]
            adapter = CorrectReferenceResetRuntime(
                expected=state["expected"],
                expected_receipt=state["expected_receipt"],
                controller=case["controller"],
                tool=case["tool"],
                anchor=state["anchor"],
                opening=state["opening"],
            )
            captures = []

            def continuation(env, terminal, _observers, _receipt, *, progress=None):
                captures.append(recovery.capture_reset(runtime_path, case["tool"], env, terminal, render_anchor))
                return dict(segments=[], stop=None, probe_no_actions=True)

            runtime = adapter.run_feedback(
                inputs["runtime_config"],
                inputs["record"],
                inputs["record_xml"],
                runtime_path,
                shared["asset_refs"],
                registered_seed=42,
                continuation=continuation,
                max_control_steps=1,
                progress=progress,
            )
            prefix.require(len(captures) == 1, "One reset image capture is required")
            images = captures[0].images
        else:
            steps = identity["replay_periods"]
            with np.load(prefix.checked(identity["arrays_ref"]), allow_pickle=False) as archive:
                commands = np.asarray(archive["sent_commands"][:steps], dtype=np.float64).copy()
                saved = saved_prefix(archive, steps)
            receipts = producer["endpoint_receipts"][: steps + 1]
            runtime = event_runtime(identity["anchors"])(
                inputs["runtime_config"],
                inputs["record"],
                inputs["record_xml"],
                commands,
                saved,
                receipts,
                runtime_path,
                shared["asset_refs"],
                inputs["case"]["controller"],
                registered_seed=42,
                progress=progress,
            )
            images = runtime["event_images"]
        recovery.checker_isolation()
        return dict(
            complete=True,
            passed=True,
            requested_phase=phase,
            case=identity,
            runtime=runtime,
            images=images,
            cv2_admission=cv2_receipt,
            candidate_modules_unloaded=True,
            fresh_cases=0,
        )

    managed.run_managed_physical_worker(phase="checker", result_path=directory / "result.json", execute=execute)


def validate_worker(branch, phase, identity):
    """Require actual control, native-step, renderer and cleanup receipts."""
    runtime, counters = branch["runtime"], branch["runtime"]["counters"]
    steps = 0 if phase == "probe" else identity["replay_periods"]
    frames = [0] if phase == "probe" else identity["anchors"]
    prefix.require(
        branch["complete"] is branch["passed"] is runtime["complete"] is runtime["passed"] is True
        and branch["requested_phase"] == phase
        and branch["candidate_modules_unloaded"] is True
        and branch["case"] == identity
        and branch["fresh_cases"] == runtime["world_model_calls"] == runtime["newly_selected_commands"] == 0
        and runtime["control_steps_attempted"] == runtime["control_steps_completed"] == steps
        and counters["mj_step"]["attempted"] == counters["mj_step"]["completed"] == 25 * steps
        and counters["mj_step2"]["attempted"] == counters["mj_step2"]["completed"] == 0
        and counters["model_construct"]["attempted"] == counters["model_construct"]["completed"] == 3
        and counters["context"]["attempted"] == counters["context"]["completed"] == 3
        and runtime["cleanup_complete"] is True
        and [item["frame_index"] for item in branch["images"]] == frames,
        "The actual worker must close with its exact physical and image counts",
    )
    if phase == "probe":
        prefix.require(
            runtime["reset_observation_verified"] is True
            and runtime["continuation"] == dict(segments=[], stop=None, probe_no_actions=True),
            "The real probe must stop immediately after reset and current image",
        )
    else:
        prefix.require(
            runtime["behavioural_replay_verified"] is True
            and runtime["prefix_control_steps"] == steps
            and runtime["checked_endpoints"] == runtime["exact_state_endpoints"] == steps + 1
            and runtime["max_state_difference"] == 0.0
            and runtime["max_cached_field_difference"] <= 1e-10
            and runtime["event_images"] == branch["images"]
            and runtime["unsaved_current_camera_renders"] == (steps + 7) // 8,
            "Every replay endpoint and original controller cache must match",
        )
    return dict(control_steps=steps, inner_steps=25 * steps, images=len(frames))


def verify_images(branch, identity):
    producer = prefix.read(identity["producer_ref"])
    with np.load(prefix.checked(identity["arrays_ref"]), allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]).copy() for key in archive.files}
    checks = []
    for image in branch["images"]:
        frame = image["frame_index"]
        receipt = producer["endpoint_receipts"][frame]
        prefix.require(
            image["current_endpoint"] == receipt and frame < len(data["sim_states"]),
            "Each current image must bind the original independently replayed endpoint",
        )
        checks.append(verify_canvas(image, data, receipt))
    return checks


def preflight():
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
        and dry["physics_steps"] == 0,
        "A source-matched 64-worker CPU regression and zero-physics dry check are required",
    )
    config, identity = load_inputs()
    prefix.require(dry["inputs"] == identity and dry["source_refs"] == sources, "Dry inputs and sources must match")
    return config, identity, before, sources, qa_ref, dry_ref


def verify_probe(probe, identity, sources, qa_ref, dry_ref):
    prefix.require(
        probe["complete"] is probe["passed"] is probe["owner_closed"] is probe["sources_unchanged"] is True
        and probe["inputs"] == identity
        and probe["source_refs"] == sources
        and probe["qa_ref"] == qa_ref
        and probe["dry_ref"] == dry_ref,
        "A source-matched real zero-action image probe is required",
    )
    lifecycle = prefix.read(probe["lifecycle_ref"])
    prefix.require(
        lifecycle["launch_error"] is None
        and len(lifecycle["workers"]) == 1
        and lifecycle["workers"][0]["key"] == "0"
        and lifecycle["workers"][0]["returncode"] == 0
        and lifecycle["workers"][0]["actual_wait"] is lifecycle["workers"][0]["reaped"] is True
        and lifecycle["workers"][0]["signal_sent"] is False,
        "The real probe worker must be successfully reaped",
    )
    prefix.checked(lifecycle["workers"][0]["log_ref"])
    branch = prefix.read(probe["cases"]["0"]["branch_ref"])
    validate_worker(branch, "probe", identity["cases"][0])
    prefix.require(verify_images(branch, identity["cases"][0]) == probe["cases"]["0"]["images"], "Probe image changed")


def phase_run(phase):
    prefix.require(phase in ("probe", "replay"), "A registered probe or replay phase is required")
    config, identity, before, sources, qa_ref, dry_ref = preflight()
    probe_ref = None
    if phase == "replay":
        probe_ref = prefix.reference(OUTPUT / "probe/result.json")
        verify_probe(prefix.read(probe_ref), identity, sources, qa_ref, dry_ref)
    directory = OUTPUT / phase
    directory.mkdir(parents=True, exist_ok=False)
    eligible = [0] if phase == "probe" else list(range(9))
    registration_ref = publish_result(
        directory / "registration.json",
        dict(
            phase=phase,
            inputs=identity,
            source_refs=sources,
            qa_ref=qa_ref,
            dry_ref=dry_ref,
            probe_ref=probe_ref,
            eligible_cases=eligible,
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
    lifecycle = run_owned_workers(specs, cwd=ROOT, timeout_seconds=config["worker_timeout_seconds"])
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
        inputs=identity,
        source_refs=sources,
        qa_ref=qa_ref,
        dry_ref=dry_ref,
        registration_ref=registration_ref,
        lifecycle_ref=lifecycle_ref,
        cases=cases,
        automatic_retry=False,
    )
    try:
        for index in eligible:
            item = cases[str(index)] = dict(status="failed")
            path = directory / f"case{index:02d}/worker/result.json"
            if path.is_file():
                item["available_branch_ref"] = prefix.reference(path)
            child = by_index.get(index)
            try:
                prefix.require(
                    child is not None
                    and child["returncode"] == 0
                    and child["actual_wait"] is child["reaped"] is True
                    and child["signal_sent"] is False
                    and path.is_file(),
                    "Each registered owned worker must close successfully",
                )
                branch_ref = prefix.reference(path)
                branch = prefix.read(branch_ref)
                counts = validate_worker(branch, phase, identity["cases"][index])
                checks = verify_images(branch, identity["cases"][index])
                item.update(status="closed", branch_ref=branch_ref, counts=counts, images=checks)
            except Exception as error:
                item["error"] = dict(type=type(error).__name__, message=str(error))
        result["owner_closed"] = len(lifecycle["workers"]) == len(eligible) and all(
            item["actual_wait"] is item["reaped"] is True for item in lifecycle["workers"]
        )
        _config, after = load_inputs()
        result["sources_unchanged"] = (
            snapshot(ROOT)["sources"] == before and source_refs() == sources and after == identity
        )
        result["complete"] = (
            result["owner_closed"] and result["sources_unchanged"] and lifecycle["launch_error"] is None
        )
        result["passed"] = result["complete"] and all(item["status"] == "closed" for item in cases.values())
    except Exception as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
    print(publish_result(directory / "result.json", result), flush=True)
    if not result["passed"]:
        raise SystemExit("The registered image-capture phase did not close")


def audit():
    config, identity, before, sources, qa_ref, dry_ref = preflight()
    probe_ref, replay_ref = (prefix.reference(OUTPUT / phase / "result.json") for phase in ("probe", "replay"))
    probe, replay = prefix.read(probe_ref), prefix.read(replay_ref)
    verify_probe(probe, identity, sources, qa_ref, dry_ref)
    prefix.require(
        replay["complete"] is replay["passed"] is replay["owner_closed"] is replay["sources_unchanged"] is True
        and replay["inputs"] == identity
        and replay["source_refs"] == sources
        and replay["qa_ref"] == qa_ref
        and replay["dry_ref"] == dry_ref
        and len(replay["cases"]) == 9,
        "All registered replay workers and sources must close",
    )
    registration = prefix.read(replay["registration_ref"])
    prefix.require(
        registration["probe_ref"] == probe_ref
        and registration["inputs"] == identity
        and registration["source_refs"] == sources,
        "Replay must be admitted by the exact no-action probe",
    )
    h40 = prefix.read(identity["refs"]["h40_audit"])
    normalization = prefix.read(h40["normalization_ref"])
    shared, cases, unique_old, unique_new = preparation.shared_inputs(), [], set(), set()
    prefix.require(shared["receipt"] == identity["shared_receipt"], "Shared simulator assets changed")
    for item in identity["cases"]:
        index = item["index"]
        entry = replay["cases"][str(index)]
        prefix.require(entry["status"] == "closed", "Every case must have closed physical evidence")
        branch = prefix.read(entry["branch_ref"])
        counts = validate_worker(branch, "replay", item)
        checks = verify_images(branch, item)
        prefix.require(checks == entry["images"], "Independent image verification must match the owner")
        with np.load(prefix.checked(item["arrays_ref"]), allow_pickle=False) as archive:
            data = {key: np.asarray(archive[key]).copy() for key in archive.files}
        source = preparation.case_inputs(item["source"]["original_identity"], shared)
        image_rows = []
        for image, check in zip(branch["images"], checks, strict=True):
            frame = image["frame_index"]
            label = audit_window(data, frame, source["case"]["tool"], normalization)
            label.pop("observed_openings")
            label.pop("actual_gripper_commands")
            unique_new.update((index, step) for step in range(frame, frame + 40))
            image_rows.append(
                dict(
                    anchor_step=frame,
                    image_ref=image["canvas_png"],
                    raw_camera_ref=image["raw_cameras"],
                    endpoint=image["current_endpoint"],
                    canvas_check=check,
                    label_check=label,
                    uses_future_training_labels_for_selection=True,
                )
            )
        unique_old.update((index, step) for start in (0, 16, 56) for step in range(start, start + 40))
        cases.append(dict(case_index=index, counts=counts, branch_ref=entry["branch_ref"], images=image_rows))
    _config, after = load_inputs()
    unchanged = snapshot(ROOT)["sources"] == before and source_refs() == sources and after == identity
    prefix.require(
        unchanged
        and sum(case["counts"]["control_steps"] for case in cases) == config["max_replay_control_periods"]
        and sum(case["counts"]["inner_steps"] for case in cases) == config["max_replay_inner_steps"]
        and sum(len(case["images"]) for case in cases) == config["max_saved_event_images"],
        "The full recorded physical, image and source budgets must close",
    )
    return publish_result(
        OUTPUT / "audit.json",
        dict(
            complete=True,
            passed=True,
            sources_unchanged=True,
            probe_ref=probe_ref,
            replay_ref=replay_ref,
            cases=cases,
            images=sum(len(case["images"]) for case in cases),
            new_window_label_rows=40 * sum(len(case["images"]) for case in cases),
            unique_new_case_transition_rows=len(unique_new),
            unique_prior_case_transition_rows=len(unique_old),
            unique_combined_case_transition_rows=len(unique_old | unique_new),
            added_case_transition_rows=len(unique_new - unique_old),
            control_periods=config["max_replay_control_periods"],
            inner_steps=config["max_replay_inner_steps"],
            world_model_calls=0,
            gpu_calls=0,
            fits=0,
            fresh_cases=0,
            training_authorized=False,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry", action="store_true")
    group.add_argument("--probe", action="store_true")
    group.add_argument("--replay", action="store_true")
    group.add_argument("--worker", nargs=2, metavar=("PHASE", "CASE"))
    group.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    prefix.environment()
    if args.dry:
        print(dry_run(), flush=True)
    elif args.probe:
        phase_run("probe")
    elif args.replay:
        phase_run("replay")
    elif args.worker:
        worker(args.worker[0], int(args.worker[1]))
    else:
        print(audit(), flush=True)


if __name__ == "__main__":
    main()
