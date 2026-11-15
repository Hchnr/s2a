"""Compare the opt-in refresh preference with one independently replayed baseline."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import bounded_reference_choice as runtime
from scripts.simulation.runtime import managed_physical_worker as managed
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_bounded_reference_choice as baseline
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.simulation.runtime import run_segmented_reset_equivalence as isolation

ROOT = prefix.REPO
OUTPUT = ROOT / "outputs/refresh_margin_physical_20261115"
PLAN = ROOT / "plans/v0.0.1/tracking/references/refresh_margin_physical_20261115.md"
QA = ROOT / "outputs/qa/refresh_margin_physical_final_20261115/result.json"
DRY = OUTPUT / "dry.json"
MODULE = "scripts.simulation.runtime.run_refresh_margin_physical"
OLD_COMMIT = "9110f91133a54a96deea87fb2348137f7b542afd"
OLD_QA = {
    "path": str(ROOT / "outputs/qa/bounded_reference_choice_recovery_20261115/result.json"),
    "sha256": "65c4daaea88500780ab614d28d9a5bfe70d446f7695c9233c61518b5488e44c1",
}
OLD_GROUPS = {
    "producer": {
        "path": str(ROOT / "outputs/bounded_reference_choice_recovery_20261115/producer/result.json"),
        "sha256": "961767e5061b79ceed7166bbf1c06070d4a4857590ed3a33d5bb27b08af2e55a",
    },
    "checker": {
        "path": str(ROOT / "outputs/bounded_reference_choice_recovery_20261115/checker/result.json"),
        "sha256": "89c5d71b70c9c3666d7489014036b06f89f23b2662b464447c04ecdab6e9b327",
    },
}
OLD_BRANCH_SHAS = {
    "producer": "4988d608a14b167382fb9a5839f7485746a4a33dcade2e1f9ad3333b0e5b206a",
    "checker": "d6ad66ff90fed70c8b9a3a0c2d20067b25225251e2605ca4b35a8aaf4f558e0e",
}
CHANGED_PATHS = (
    "src/s2a/research/trajectory_search/margin_two_step_search.py",
    "src/s2a/research/trajectory_search/margin_guarded_fallback.py",
    "src/s2a/research/joint_control/preview/reset_feedback_policy.py",
    "src/s2a/research/joint_control/preview/segmented_reset_policy.py",
    "scripts/simulation/runtime/bounded_reference_choice.py",
)
FORBIDDEN_CANDIDATE = "s2a.research.joint_control.preview.refresh_margin_preference"


def source_refs():
    paths = (
        Path(__file__).resolve(),
        PLAN,
        ROOT / "plans/v0.0.1/tracking/references/refresh_margin_preference_20261115.md",
        ROOT / "plans/v0.0.1/tracking/references/refresh_margin_preference_addendum_20261115.md",
        ROOT / "src/s2a/research/joint_control/preview/refresh_margin_preference.py",
        ROOT / "tests/research/joint_control/preview/test_refresh_margin_preference.py",
        ROOT / "tests/research/trajectory_search/test_margin_two_step_search.py",
        ROOT / "tests/research/trajectory_search/test_margin_guarded_fallback.py",
        *(ROOT / path for path in CHANGED_PATHS),
    )
    return baseline.source_refs() + [prefix.reference(path) for path in paths]


def historical_sources():
    qa = prefix.read(OLD_QA)
    source_map = prefix.read(qa["source_before_ref"])["sources"]
    rows = []
    for path in CHANGED_PATHS:
        blob = subprocess.run(["git", "show", f"{OLD_COMMIT}:{path}"], cwd=ROOT, capture_output=True, check=True).stdout
        digest = hashlib.sha256(blob).hexdigest()
        prefix.require(source_map[path] == digest, f"The original QA snapshot must bind the historical blob: {path}")
        current = prefix.reference(ROOT / path)
        prefix.require(current["sha256"] != digest, f"The documented source change must exist: {path}")
        rows.append(dict(path=path, historical_commit=OLD_COMMIT, historical_sha256=digest, current=current))
    return rows


def prepare():
    """Bind the old replay and expose only the original completed prefix."""
    data = baseline.prepare()
    groups = {phase: prefix.read(ref) for phase, ref in OLD_GROUPS.items()}
    branches = {}
    for phase, group in groups.items():
        prefix.require(
            group["complete"] is group["passed"] is group["sources_unchanged"] is True
            and group["branch_refs"]["new"]["sha256"] == OLD_BRANCH_SHAS[phase]
            and all(
                row["actual_wait"] is row["reaped"] is True and row["returncode"] == 0
                for row in group["lifecycle"]["workers"]
            ),
            "The original baseline group and replay must be closed",
        )
        branches[phase] = prefix.read(group["branch_refs"]["new"])
    registration = prefix.read(groups["producer"]["registration_ref"])
    prefix.require(registration["identity"] == data["identity"], "The current prefix must match the original baseline")
    old_producer, old_checker = branches["producer"], branches["checker"]
    baseline.validate_branch(old_producer, "producer")
    baseline.validate_branch(old_checker, "checker")
    old = old_producer["runtime"]["continuation"]
    old_history = old["first_window"]["periods"] + old["second_window"]["periods"]
    comparison.compare_periods(data["history"], old_history)
    comparison.compare_periods(baseline.tail_periods(old_producer), baseline.tail_periods(old_checker))
    prefix.require(
        len(baseline.tail_periods(old_producer)) == 1
        and old["tail"]["stop"]["reason"] == "quality_refresh_required"
        and old["tail"]["stop"]["at_step"] == 17,
        "The recorded baseline must retain its single completed new-reference period",
    )
    identity = dict(
        original=data["identity"],
        baseline_groups=OLD_GROUPS,
        baseline_branch_refs={phase: group["branch_refs"]["new"] for phase, group in groups.items()},
        historical_sources=historical_sources(),
    )
    data.update(identity=identity, baseline_producer=old_producer)
    return data


def dry_run():
    prefix.environment()
    data = prepare()
    print(
        publish_result(
            DRY,
            dict(
                complete=True,
                passed=True,
                identity=data["identity"],
                source_refs=source_refs(),
                native_models=0,
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
                automatic_retry=False,
            ),
        ),
        flush=True,
    )


def worker(phase):
    prefix.require(phase in ("producer", "checker"), "One registered physical phase is required")
    print(json.dumps(dict(event="worker_started", pid=os.getpid(), phase=phase)), flush=True)
    directory = OUTPUT / phase / "worker"
    directory.mkdir(parents=True, exist_ok=False)
    registration = prefix.read(prefix.reference(OUTPUT / phase / "registration.json"))

    def execute():
        data = prepare()
        prefix.require(
            registration["phase"] == phase
            and registration["identity"] == data["identity"]
            and registration["source_refs"] == source_refs(),
            "The registered physical inputs must remain fixed",
        )
        saved = None
        if phase == "checker":
            isolation.checker_isolation()
            prefix.require(FORBIDDEN_CANDIDATE not in sys.modules, "The checker cannot load the experimental candidate")
            produced = prefix.read(registration["producer_branch_ref"])
            prefix.require(produced["complete"] is produced["passed"] is True, "A closed producer is required")
            saved = produced["runtime"]["continuation"]
        cv2 = data["inputs"]["base"]["cv2_admission"]
        cv2_receipt = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        original.require_no_simulator_import()
        from s2a.research.native_runtime.reset_bounded_runtime import ResetBoundedRuntime

        case, inputs = data["inputs"]["case"], data["inputs"]
        adapter = ResetBoundedRuntime(
            expected=case["expected"],
            expected_receipt=case["expected_receipt"],
            chunk=case["chunk"],
            controller=case["controller"],
            tool=case["tool"],
        )
        factory = activation = None
        if phase == "producer":
            from s2a.research.joint_control.preview.refresh_margin_preference import RefreshMarginPreference
            from s2a.research.joint_control.preview.reset_feedback_policy import policy_inputs
            from s2a.research.joint_control.preview.segmented_reset_policy import build_segmented_reset_policy

            preference = RefreshMarginPreference()

            def factory(env, case, terminal, receipt):
                return build_segmented_reset_policy(
                    env, policy_inputs(case, terminal), receipt, adapter_factory=preference.adapter_factory
                )

            activation = preference.activate

        def continuation(env, terminal, observers, receipt, *, progress=None):
            return runtime.run_comparison(
                env,
                terminal,
                observers,
                receipt,
                phase=phase,
                mode="new",
                case=case,
                history=data["history"],
                original_captures=data["captures"],
                exchanges=data["exchanges"],
                responses=data["responses"],
                runtime_output=directory / "runtime",
                producer=saved,
                progress=progress,
                policy_factory=factory,
                after_history=activation,
            )

        def progress(receipt):
            print(json.dumps(baseline.progress_event(phase, "new", receipt)), flush=True)

        actual = adapter.run_feedback(
            case["config"],
            case["record"],
            case["record_xml"],
            directory / "runtime",
            inputs["base"]["asset_refs"],
            registered_seed=43,
            continuation=continuation,
            max_control_steps=48,
            progress=progress,
        )
        if phase == "checker":
            isolation.checker_isolation()
            prefix.require(FORBIDDEN_CANDIDATE not in sys.modules, "The checker must not load the candidate")
        return dict(
            complete=True, passed=True, runtime=actual, case=case["case"], cv2_admission=cv2_receipt, fresh_cases=0
        )

    managed.run_managed_physical_worker(phase=phase, result_path=directory / "result.json", execute=execute)


def validate_candidate(branch, data):
    counts = baseline.validate_branch(branch, "producer")
    actual = branch["runtime"]
    prefix.require(
        actual.get("refresh_preference_activation") == dict(step=16, completed_periods=16, future_observations=0),
        "The candidate must activate only after the completed original history",
    )
    traces = actual["segment_receipts"][0]["shared_shift_traces"]
    prefix.require(len(traces) >= 17, "The candidate must actually evaluate the first continuation decision")
    preference = traces[16]["shared_search"].get("refresh_preference")
    prefix.require(
        isinstance(preference, dict) and preference["enabled"] is True, "The first decision must record preference"
    )
    tail = baseline.tail_periods(branch)
    prefix.require(len(tail) >= 1, "The independently audited first candidate must execute")
    original_first = baseline.tail_periods(data["baseline_producer"])[0]
    if preference["eligible_candidates"] == 0:
        prefix.require(preference["selected_changed"] is False, "No eligible candidate must preserve the original plan")
        np.testing.assert_array_equal(tail[0]["command"], original_first["command"])
    counts.update(
        first_preference=preference,
        first_command_max_difference=float(
            np.max(np.abs(np.asarray(tail[0]["command"]) - np.asarray(original_first["command"])))
        ),
    )
    return counts


def phase_run(phase):
    prefix.environment()
    prefix.require(phase in ("producer", "checker"), "One physical phase is required")
    qa_ref, dry_ref = prefix.reference(QA), prefix.reference(DRY)
    qa, dry = prefix.read(qa_ref), prefix.read(dry_ref)
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True
        and dry["complete"] is dry["passed"] is True
        and dry["physics_steps"] == 0,
        "The full CPU QA and zero-physics preflight must pass",
    )
    before = snapshot(ROOT)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "The CPU QA sources must match")
    data, sources = prepare(), source_refs()
    prefix.require(dry["identity"] == data["identity"] and dry["source_refs"] == sources, "Dry identities must match")
    directory = OUTPUT / phase
    prefix.require(not directory.exists(), "A fresh physical phase directory is required")
    producer_ref = None
    if phase == "checker":
        previous = prefix.read(prefix.reference(OUTPUT / "producer/result.json"))
        prefix.require(previous["complete"] is previous["passed"] is True, "A completed producer phase is required")
        producer_ref = previous["branch_ref"]
        prefix.checked(producer_ref)
    registration_ref = publish_result(
        directory / "registration.json",
        dict(
            phase=phase,
            owner_pid=os.getpid(),
            identity=data["identity"],
            source_refs=sources,
            qa_ref=qa_ref,
            dry_ref=dry_ref,
            producer_branch_ref=producer_ref,
            max_control_steps=48,
            max_native_steps=1200,
            automatic_retry=False,
        ),
    )
    specs = [
        dict(key="candidate", argv=[sys.executable, "-m", MODULE, "--worker", phase], log=directory / "worker.log")
    ]
    lifecycle = run_owned_workers(specs, cwd=ROOT, timeout_seconds=3600)
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    result = dict(
        complete=False,
        passed=False,
        phase=phase,
        registration_ref=registration_ref,
        lifecycle=lifecycle,
        automatic_retry=False,
    )
    try:
        prefix.require(
            lifecycle["launch_error"] is None
            and len(lifecycle["workers"]) == 1
            and lifecycle["workers"][0]["returncode"] == 0
            and lifecycle["workers"][0]["actual_wait"] is lifecycle["workers"][0]["reaped"] is True,
            "The owned physical worker must exit and be reaped",
        )
        branch_ref = prefix.reference(directory / "worker/result.json")
        branch = prefix.read(branch_ref)
        counts = validate_candidate(branch, data) if phase == "producer" else baseline.validate_branch(branch, phase)
        if phase == "checker":
            produced = prefix.read(producer_ref)
            expected, actual = produced["runtime"]["continuation"], branch["runtime"]["continuation"]

            def periods(value):
                return (
                    value["first_window"]["periods"]
                    + value["second_window"]["periods"]
                    + [row for segment in value["tail"]["segments"] for row in segment["continuation"]["periods"]]
                )

            counts["replay_max_difference"] = comparison.compare_periods(periods(expected), periods(actual))
            prefix.require(
                expected["tail"]["stop"] == actual["tail"]["stop"], "Recorded producer stop must be retained"
            )
        prefix.require(
            snapshot(ROOT)["sources"] == before and prepare()["identity"] == data["identity"],
            "Sources and inputs changed",
        )
        result.update(
            complete=True,
            passed=True,
            sources_unchanged=True,
            branch_ref=branch_ref,
            counts=counts,
            world_model_calls=0,
            fresh_cases=0,
        )
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish_result(directory / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--phase", choices=("producer", "checker"))
    parser.add_argument("--worker", choices=("producer", "checker"))
    args = parser.parse_args()
    if sum((args.dry, args.phase is not None, args.worker is not None)) != 1:
        parser.error("Choose exactly one dry run, phase or worker")
    if args.dry:
        dry_run()
    elif args.phase:
        phase_run(args.phase)
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
