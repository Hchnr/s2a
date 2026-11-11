"""Audit original, segmented-producer, and independent-checker reset results."""

from pathlib import Path

from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.simulation.runtime import run_segmented_reset_equivalence as study
from scripts.verification.response_models.audit_reset_start_feedback import compare_replay

ROOT = prefix.REPO
PRODUCER = {
    "path": str(ROOT / "outputs/segmented_reset_equivalence_20261111/producer/result.json"),
    "sha256": "182f91a7603c94b58a9c0bd832773cbe8346a909517c4bda178b8ccf6c68d150",
}
CHECKER = {
    "path": str(ROOT / "outputs/segmented_reset_equivalence_20261111/checker/result.json"),
    "sha256": "459656c0dff3cf43bfd16480316dcfbec0869da98443bf69a551ebf3c7a0b207",
}
QA = ROOT / "outputs/qa/segmented_reset_equivalence_audit_20261111/result.json"
OUTPUT = ROOT / "outputs/segmented_reset_equivalence_audit_20261111/result.json"


def validate_workers(lifecycle):
    """Do not count a started, signalled, or unreaped worker as a result."""
    workers = lifecycle.get("workers", [])
    prefix.require(
        lifecycle.get("launch_error") is None
        and len(workers) == 12
        and all(
            item.get("key") == original.branch_key(i, "feedback")
            and item.get("returncode") == 0
            and item.get("actual_wait") is item.get("reaped") is True
            and item.get("signal_sent") is False
            for i, item in enumerate(workers)
        ),
        "All twelve registered workers must exit, be waited, and be reaped",
    )


def execute():
    prefix.environment()
    before = snapshot(ROOT)["sources"]
    sources = [
        prefix.reference(path)
        for path in (
            Path(__file__).resolve(),
            ROOT / "tests/research/native_runtime/test_segmented_reset_equivalence_audit.py",
            ROOT / "scripts/verification/response_models/audit_reset_start_feedback.py",
            study.PLAN,
        )
    ]
    qa = prefix.read(prefix.reference(QA))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and not qa["collection"]["collection_errors"]
        and prefix.read(qa["source_before_ref"])["sources"] == before,
        "A source-matched CPU regression is required",
    )
    inputs = study.load_inputs(verify_assets=True)
    baseline = prefix.read(study.BASELINE)
    owners = [prefix.read(PRODUCER), prefix.read(CHECKER)]
    for owner, phase in zip(owners, ("producer", "checker"), strict=True):
        totals = owner["counts"]["totals"]
        prefix.require(
            owner["complete"] is owner["passed"] is owner["sources_unchanged"] is True
            and owner["control_qualified"] is owner["automatic_retry"] is False
            and owner["phase"] == phase
            and owner["world_model_calls"] == owner["fresh_cases"] == 0
            and len(owner["branch_refs"]) == 12
            and totals["models"] == totals["contexts"] == 36
            and totals["control_steps"] == 96
            and totals["native_steps"] == 2400
            and totals["short_window_passes"] == 12
            and totals["newly_selected_commands"] == (96 if phase == "producer" else 0),
            "Both complete fixed-budget physical phases are required",
        )
        lifecycle = prefix.read(owner["workers_ref"])
        validate_workers(lifecycle)
        for item in lifecycle["workers"]:
            prefix.checked(item["log_ref"])
        registration = prefix.read(owner["registration_ref"])
        prefix.require(
            registration["phase"] == phase
            and registration["inputs"] == study.input_receipt(inputs)
            and registration["source_refs"] == study.source_refs(),
            "The registered input and source identities must match",
        )
        for ref in registration["source_refs"]:
            prefix.checked(ref)
        if phase == "checker":
            expected = {original.branch_key(i, "feedback"): ref for i, ref in enumerate(owners[0]["branch_refs"])}
            prefix.require(registration["producer_branch_refs"] == expected, "Checker must bind this producer")
    rows, old_max, checker_max = [], {}, {}
    for i, case in enumerate(inputs["cases"]):
        old, producer, checker = [prefix.read(owner["branch_refs"][i]) for owner in (baseline, *owners)]
        prefix.require(
            old["case"] == producer["case"] == checker["case"] == case["case"]
            and old["complete"] is old["passed"] is producer["complete"] is producer["passed"] is True
            and checker["complete"] is checker["passed"] is checker["candidate_modules_unloaded"] is True,
            "The original, segmented, and isolated checker cases must align",
        )
        runtimes = [value["runtime"] for value in (old, producer, checker)]
        prefix.require(
            all(
                runtime["reset_observation_verified"] is runtime["reset_image_match"] is True
                and runtime["previous_saved_command"] is None
                and runtime["continuation"]["stop"] is None
                for runtime in runtimes
            )
            and runtimes[0]["reset_endpoint_comparison"]
            == runtimes[1]["reset_endpoint_comparison"]
            == runtimes[2]["reset_endpoint_comparison"]
            and runtimes[0]["reset_render"]["canvas_rgb_sha256"]
            == runtimes[1]["reset_render"]["canvas_rgb_sha256"]
            == runtimes[2]["reset_render"]["canvas_rgb_sha256"]
            and len(runtimes[1]["segment_receipts"]) == 1
            and runtimes[1]["segment_trace_starts"] == [dict(shared=0, causal=0)],
            "All phases must use the same exact reset and one segmented window",
        )
        periods = [runtime["continuation"]["periods"] for runtime in runtimes]
        comparisons = (compare_replay(periods[0], periods[1]), compare_replay(periods[1], periods[2]))
        for destination, comparison in zip((old_max, checker_max), comparisons, strict=True):
            for key, value in comparison.items():
                destination[key] = max(destination.get(key, 0.0), value)
        rows.append(
            dict(
                case=case["case"],
                original_ref=baseline["branch_refs"][i],
                producer_ref=owners[0]["branch_refs"][i],
                checker_ref=owners[1]["branch_refs"][i],
                original_vs_segmented=comparisons[0],
                producer_vs_checker=comparisons[1],
            )
        )
    prefix.require(
        snapshot(ROOT)["sources"] == before and sources == [prefix.reference(ref["path"]) for ref in sources],
        "Audit sources must remain unchanged",
    )
    print(
        publish(
            OUTPUT,
            dict(
                complete=True,
                passed=True,
                control_qualified=False,
                baseline_ref=study.BASELINE,
                baseline_audit_ref=study.BASELINE_AUDIT,
                producer_ref=PRODUCER,
                checker_ref=CHECKER,
                qa_ref=prefix.reference(QA),
                source_refs=sources,
                sources_unchanged=True,
                cases=rows,
                original_vs_segmented_max=old_max,
                producer_vs_checker_max=checker_max,
                original_commands_compared=96,
                checker_commands_compared=96,
                inner_samples_compared_per_pair=2400,
                checker_candidate_modules_unloaded=12,
                native_models=0,
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
            ),
        ),
        flush=True,
    )


if __name__ == "__main__":
    execute()
