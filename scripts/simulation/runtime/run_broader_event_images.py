"""Capture preregistered current event images by replaying recorded commands on CPU."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result
from s2a.research.native_runtime.correct_reference_reset import CorrectReferenceResetRuntime
from s2a.research.native_runtime.development_prefix_replay import saved_prefix
from s2a.research.wam.training.native_observations import render_anchor
from s2a.research.wam.training.recovery_h40_contract import audit_window
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import managed_physical_worker as managed
from scripts.simulation.runtime import prepare_broader_event_images as stage
from scripts.simulation.runtime import prepare_reset_correct_reference_cohort as preparation
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_broader_recovery_pilot as broader
from scripts.simulation.runtime import run_current_controller_recovery as recovery
from scripts.simulation.runtime import run_recovery_event_images as legacy
from scripts.simulation.runtime import run_reset_correct_reference as previous
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.verification.wam.training.audit_broader_recovery_h40 import command_counts

ROOT = prefix.REPO
OUTPUT = stage.OUTPUT
QA = ROOT / "outputs/qa/broader_event_images_final_20261116/result.json"
MODULE = "scripts.simulation.runtime.run_broader_event_images"


def source_refs():
    return stage.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__).resolve(),
            ROOT / "tests/research/native_runtime/test_broader_event_images.py",
        )
    ]


def validate_registration(phase, index, registration, expected_sources):
    """Admit only the exact registered reset or one of the 24 recorded prefixes."""
    prefix.require(
        phase in ("probe", "replay") and type(index) is int and 0 <= index < 24 and (phase != "probe" or index == 0),
        "One registered probe or replay case is required",
    )
    cases = registration["inputs"]["cases"]
    prefix.require(
        registration["phase"] == phase
        and registration["eligible_cases"] == ([0] if phase == "probe" else list(range(24)))
        and registration["source_refs"] == expected_sources
        and registration["automatic_retry"] is False
        and len(cases) == 24
        and all(item["index"] == i for i, item in enumerate(cases)),
        "The exact worker roster and source identities are required",
    )
    return cases[index]


def worker(phase, index):
    prefix.require(
        phase in ("probe", "replay") and type(index) is int and 0 <= index < 24 and (phase != "probe" or index == 0),
        "One registered probe or replay case is required",
    )
    directory = OUTPUT / phase / f"case{index:02d}" / "worker"
    directory.mkdir(parents=True, exist_ok=False)
    registration = prefix.read(prefix.reference(OUTPUT / phase / "registration.json"))

    def execute():
        identity = validate_registration(phase, index, registration, source_refs())
        dry = prefix.read(registration["dry_ref"])
        prefix.checked(registration["qa_ref"])
        prefix.require(
            dry["complete"] is dry["passed"] is True
            and dry["inputs"] == registration["inputs"]
            and dry["source_refs"] == registration["source_refs"],
            "The worker identity must match the registered zero-physics dry run",
        )
        recovery.checker_isolation()
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
            runtime = legacy.event_runtime(identity["anchors"])(
                inputs["runtime_config"],
                inputs["record"],
                inputs["record_xml"],
                commands,
                saved,
                producer["endpoint_receipts"][: steps + 1],
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


def dry_run():
    original.require_no_simulator_import()
    config, identity = stage.load_inputs()
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
        "A complete source-matched 64-worker regression and zero-physics preflight are required",
    )
    config, identity = stage.load_inputs()
    prefix.require(dry["inputs"] == identity and dry["source_refs"] == sources, "Dry identity and sources must match")
    return config, identity, before, sources, qa_ref, dry_ref


def verify_probe(probe, identity, sources, qa_ref, dry_ref):
    """Recheck the real reset, image, original registration and reaped child."""
    prefix.require(
        probe["complete"] is probe["passed"] is probe["owner_closed"] is probe["sources_unchanged"] is True
        and probe["resource_counts_complete"] is True
        and probe["inputs"] == identity
        and probe["source_refs"] == sources
        and probe["qa_ref"] == qa_ref
        and probe["dry_ref"] == dry_ref,
        "A source-matched real zero-action image probe is required",
    )
    registration = prefix.read(probe["registration_ref"])
    prefix.require(
        registration["phase"] == "probe"
        and registration["eligible_cases"] == [0]
        and registration["inputs"] == identity
        and registration["source_refs"] == sources
        and registration["qa_ref"] == qa_ref
        and registration["dry_ref"] == dry_ref
        and registration["probe_ref"] is None
        and registration["automatic_retry"] is False,
        "The exact registered probe identity is required",
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
    entry = probe["cases"]["0"]
    prefix.require(
        entry["status"] == "closed" and entry["branch_ref"] == entry["available_branch_ref"], "Probe must close"
    )
    branch = prefix.read(entry["branch_ref"])
    counts = legacy.validate_worker(branch, "probe", identity["cases"][0])
    checks = legacy.verify_images(branch, identity["cases"][0])
    prefix.require(counts == entry["counts"] and checks == entry["images"], "Probe image and counts must match")


def phase_run(phase):
    """Own the fixed worker set and retain unknown or partial physical costs."""
    prefix.require(phase in ("probe", "replay"), "A registered probe or replay phase is required")
    config, identity, before, sources, qa_ref, dry_ref = preflight()
    probe_ref = None
    if phase == "replay":
        probe_ref = prefix.reference(OUTPUT / "probe/result.json")
        verify_probe(prefix.read(probe_ref), identity, sources, qa_ref, dry_ref)
    directory = OUTPUT / phase
    directory.mkdir(parents=True, exist_ok=False)
    eligible = [0] if phase == "probe" else list(range(24))
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
        resource_counts_complete=False,
        phase=phase,
        inputs=identity,
        source_refs=sources,
        qa_ref=qa_ref,
        dry_ref=dry_ref,
        registration_ref=registration_ref,
        lifecycle_ref=lifecycle_ref,
        cases=cases,
        world_model_calls=0,
        gpu_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        for index in eligible:
            spec = identity["cases"][index]
            entry = cases[str(index)] = dict(
                status="failed",
                expected_periods=0 if phase == "probe" else spec["replay_periods"],
                observed_resources=None,
            )
            child = by_index.get(index)
            path = directory / f"case{index:02d}/worker/result.json"
            try:
                branch = None
                if path.is_file():
                    entry["available_branch_ref"] = prefix.reference(path)
                    branch = prefix.read(entry["available_branch_ref"])
                    entry["observed_resources"] = broader.observed_resources(branch)
                prefix.require(
                    child is not None
                    and child["returncode"] == 0
                    and child["actual_wait"] is child["reaped"] is True
                    and child["signal_sent"] is False
                    and branch is not None,
                    "Each registered owned worker must close successfully",
                )
                counts = legacy.validate_worker(branch, phase, spec)
                checks = legacy.verify_images(branch, spec)
                entry.update(status="closed", branch_ref=entry["available_branch_ref"], counts=counts, images=checks)
            except Exception as error:
                entry["error"] = dict(type=type(error).__name__, message=str(error))
        known = [cases[str(index)]["observed_resources"] for index in eligible]
        known = [item for item in known if item is not None]
        totals = dict(
            control_steps_attempted=sum(item["control_steps_attempted"] for item in known),
            control_steps_completed=sum(item["control_steps_completed"] for item in known),
            inner_steps_attempted=sum(item["counters"]["mj_step"]["attempted"] for item in known),
            inner_steps_completed=sum(item["counters"]["mj_step"]["completed"] for item in known),
            models_attempted=sum(item["counters"]["model_construct"]["attempted"] for item in known),
            contexts_attempted=sum(item["counters"]["context"]["attempted"] for item in known),
        )
        result["observed_resource_totals"] = totals
        result["resource_counts_complete"] = len(known) == len(eligible)
        budget = "probe" if phase == "probe" else "replay"
        prefix.require(
            totals["control_steps_attempted"] <= config[f"max_{budget}_control_periods"]
            and totals["inner_steps_attempted"] <= config[f"max_{budget}_inner_steps"]
            and totals["models_attempted"] <= config[f"max_{budget}_models"]
            and totals["contexts_attempted"] <= config[f"max_{budget}_contexts"],
            "The observed physical resource budget cannot be exceeded",
        )
        result["owner_closed"] = (
            len(lifecycle["workers"]) == len(eligible)
            and {item["key"] for item in lifecycle["workers"]} == {str(index) for index in eligible}
            and all(item["actual_wait"] is item["reaped"] is True for item in lifecycle["workers"])
        )
        _config, after = stage.load_inputs()
        result["sources_unchanged"] = (
            snapshot(ROOT)["sources"] == before and source_refs() == sources and after == identity
        )
        result["complete"] = (
            result["owner_closed"] and result["sources_unchanged"] and lifecycle["launch_error"] is None
        )
        result["passed"] = (
            result["complete"]
            and result["resource_counts_complete"]
            and all(item["status"] == "closed" for item in cases.values())
        )
    except Exception as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
    print(publish_result(directory / "result.json", result), flush=True)
    if not result["passed"]:
        raise SystemExit("The registered image-capture phase did not close; no automatic retry")


def verify_replay_owner(owner, identity, sources, qa_ref, dry_ref, probe_ref):
    """Independently recheck all 24 registered child processes and receipts."""
    registration = prefix.read(owner["registration_ref"])
    lifecycle = prefix.read(owner["lifecycle_ref"])
    expected = {str(index) for index in range(24)}
    workers = {row["key"]: row for row in lifecycle["workers"]}
    prefix.require(
        owner["complete"] is owner["passed"] is owner["owner_closed"] is owner["sources_unchanged"] is True
        and owner["resource_counts_complete"] is True
        and owner["phase"] == registration["phase"] == "replay"
        and owner["automatic_retry"] is registration["automatic_retry"] is False
        and owner["inputs"] == registration["inputs"] == identity
        and owner["source_refs"] == registration["source_refs"] == sources
        and owner["qa_ref"] == registration["qa_ref"] == qa_ref
        and owner["dry_ref"] == registration["dry_ref"] == dry_ref
        and registration["probe_ref"] == probe_ref
        and registration["eligible_cases"] == list(range(24))
        and set(owner["cases"]) == expected
        and lifecycle["launch_error"] is None
        and len(lifecycle["workers"]) == 24
        and set(workers) == expected,
        "The complete replay roster, sources and registered probe must match",
    )
    for key, child in workers.items():
        entry = owner["cases"][key]
        prefix.require(
            child["returncode"] == 0
            and child["actual_wait"] is child["reaped"] is True
            and child["signal_sent"] is False
            and entry["status"] == "closed"
            and entry["branch_ref"] == entry["available_branch_ref"],
            "Each replay must bind its successfully reaped child",
        )
        prefix.checked(child["log_ref"])
        prefix.checked(entry["branch_ref"])


def verify_label(candidate, data, tool, normalization):
    """Match a new current image to its previously executed, independently checked H40."""
    label = audit_window(data, candidate["anchor_step"], tool, normalization)
    openings, commands = label.pop("observed_openings"), label.pop("actual_gripper_commands")
    prefix.require(
        all(candidate[key] == value for key, value in label.items())
        and candidate["opening_range"] == [float(openings.min()), float(openings.max())]
        and candidate["opening_below_count"] == int(np.count_nonzero(openings < 0.35))
        and candidate["command_counts"] == command_counts(commands)
        and candidate["selection_uses_future_training_labels"] is candidate["full_h40"] is True,
        "The original H40 labels, clipping and offline selection must match the candidate receipt",
    )
    return label


def audit():
    config, identity, before, sources, qa_ref, dry_ref = preflight()
    probe_ref, replay_ref = (prefix.reference(OUTPUT / phase / "result.json") for phase in ("probe", "replay"))
    probe, replay = prefix.read(probe_ref), prefix.read(replay_ref)
    verify_probe(probe, identity, sources, qa_ref, dry_ref)
    verify_replay_owner(replay, identity, sources, qa_ref, dry_ref, probe_ref)
    events = prefix.read(identity["refs"]["event_audit"])
    union = prefix.read(identity["refs"]["event_union"])
    normalization = prefix.read(identity["refs"]["normalization"])
    shared, cases, unique_old, unique_new = preparation.shared_inputs(), [], set(), set()
    prefix.require(shared["receipt"] == identity["shared_receipt"], "Shared simulator assets changed")
    for item in identity["cases"]:
        index = item["index"]
        entry = replay["cases"][str(index)]
        branch = prefix.read(entry["branch_ref"])
        counts = legacy.validate_worker(branch, "replay", item)
        checks = legacy.verify_images(branch, item)
        prefix.require(
            counts == entry["counts"] and checks == entry["images"], "Independent replay verification must match"
        )
        with np.load(prefix.checked(item["arrays_ref"]), allow_pickle=False) as archive:
            data = {key: np.asarray(archive[key]).copy() for key in archive.files}
        source = preparation.case_inputs(item["source"]["original_identity"], shared)
        candidates = {candidate["anchor_step"]: candidate for candidate in item["candidates"]}
        image_rows = []
        for image, check in zip(branch["images"], checks, strict=True):
            frame = image["frame_index"]
            label = verify_label(candidates[frame], data, source["case"]["tool"], normalization)
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
        old = {step for start in (0, 16, 56) for step in range(start, start + 40)}
        new = {step for frame in item["anchors"] for step in range(frame, frame + 40)}
        unique_old.update((index, step) for step in old)
        groups = dict(existing=old, candidates=new, union=old | new, added=new - old)
        for name, steps in groups.items():
            expected = union["cases"][index]["groups"][name]
            prefix.require(
                len(steps) == expected["unique_case_transition_rows"]
                and sum(bool(data["opening"][step + 1] < 0.35) for step in steps) == expected["low_opening_endpoints"]
                and sum(bool(data["native_success"][step + 1]) for step in steps)
                == expected["native_success_endpoints"],
                "The previously audited actual event union must match each case",
            )
        cases.append(dict(case_index=index, counts=counts, branch_ref=entry["branch_ref"], images=image_rows))
    _config, after = stage.load_inputs()
    unchanged = snapshot(ROOT)["sources"] == before and source_refs() == sources and after == identity
    images = sum(len(case["images"]) for case in cases)
    clipped = sum(row["label_check"]["clipped_count"] for case in cases for row in case["images"])
    clipped_slots = sum(
        (
            np.asarray(row["label_check"]["clipped_per_active_slot"], dtype=np.int64)
            for case in cases
            for row in case["images"]
        ),
        start=np.zeros(10, dtype=np.int64),
    ).tolist()
    prefix.require(
        unchanged
        and sum(case["counts"]["control_steps"] for case in cases) == config["max_replay_control_periods"]
        and sum(case["counts"]["inner_steps"] for case in cases) == config["max_replay_inner_steps"]
        and images == config["max_saved_event_images"] == events["candidate_count"]
        and images * 40 == events["candidate_window_label_rows"]
        and len(unique_new) == events["candidate_unique_case_transition_rows"]
        and len(unique_old | unique_new) == events["existing_and_candidate_unique_case_transition_rows"]
        and len(unique_old) == union["totals"]["existing"]["unique_case_transition_rows"]
        and len(unique_new - unique_old) == union["totals"]["added"]["unique_case_transition_rows"]
        and clipped == events["candidate_clipping_total"]
        and clipped_slots == events["candidate_clipping_per_active_slot"],
        "The actual images, original H40, source and physical budgets must all close",
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
            images=images,
            new_window_label_rows=40 * images,
            unique_new_case_transition_rows=len(unique_new),
            unique_prior_case_transition_rows=len(unique_old),
            unique_combined_case_transition_rows=len(unique_old | unique_new),
            added_case_transition_rows=len(unique_new - unique_old),
            candidate_clipped_count=clipped,
            candidate_clipped_per_active_slot=clipped_slots,
            union_event_counts=union["totals"]["union"],
            control_periods=sum(case["counts"]["control_steps"] for case in cases),
            inner_steps=sum(case["counts"]["inner_steps"] for case in cases),
            candidate_selection_uses_future_training_labels=True,
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
    group.add_argument("--phase", choices=("probe", "replay"))
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
        print(audit(), flush=True)


if __name__ == "__main__":
    main()
