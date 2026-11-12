"""Bind the long development registration to the reusable adaptive owner."""

from pathlib import Path

from scripts.simulation.runtime import adaptive_live_bowl_audit as historical_audit
from scripts.simulation.runtime import adaptive_live_bowl_long_admission as admission
from scripts.simulation.runtime import adaptive_live_bowl_recovery_owner as previous_owner
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as period_comparison
from scripts.simulation.runtime import registered_adaptive_execution as execution
from scripts.simulation.runtime import registered_adaptive_trial as registered
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_live_two_window_replay as capture_comparison

MODULE = "scripts.simulation.runtime.run_adaptive_live_bowl_long"
GPU = "GPU-c0b598ba-0a6e-73af-bbcc-97bc89379bdf"
REQUIRED_QA_TARGETS = tuple(
    dict.fromkeys(
        (
            *previous_owner.REQUIRED_QA_TARGETS,
            "tests/research/native_runtime/test_adaptive_live_bowl_long_admission.py",
            "tests/research/native_runtime/test_registered_adaptive_trial.py",
            "tests/research/native_runtime/test_registered_adaptive_execution.py",
            "tests/research/native_runtime/test_long_adaptive_stream.py",
            "tests/research/native_runtime/test_adaptive_live_bowl_long_owner.py",
        )
    )
)


def source_refs():
    """Provide one stable source identity in owner, simulator and model processes."""
    paths = (
        admission.CONFIG,
        admission.PLAN,
        Path(admission.__file__).resolve(),
        Path(registered.__file__).resolve(),
        Path(execution.__file__).resolve(),
        Path(__file__).resolve(),
        admission.ROOT / "scripts/simulation/runtime/run_adaptive_live_bowl_long.py",
        *(admission.ROOT / path for path in REQUIRED_QA_TARGETS),
    )
    refs = previous_owner.source_refs()
    known = {ref["path"]: ref["sha256"] for ref in refs}
    for path in paths:
        ref = prefix.reference(path)
        if ref["path"] in known:
            prefix.require(known[ref["path"]] == ref["sha256"], "Conflicting source identities")
        else:
            refs.append(ref)
            known[ref["path"]] = ref["sha256"]
    return refs


def compare_previous(inputs, branch):
    """Compare recorded past periods after execution, without affecting policy."""
    prior = prefix.read(inputs["prior_recovery_producer"]["branch_ref"])
    old_prefix, old_adaptive = historical_audit.periods(prior, "producer")
    new_prefix, new_adaptive = historical_audit.periods(branch, "producer")
    old_rows, new_rows = old_prefix + old_adaptive, new_prefix + new_adaptive
    prefix.require(len(old_rows) == 115, "The previously audited 115-period producer is required")
    count = min(len(old_rows), len(new_rows))
    result = dict(
        prior_periods=len(old_rows),
        new_periods=len(new_rows),
        compared_periods=count,
        all_previous_periods_compared=count == len(old_rows),
    )
    try:
        maxima = period_comparison.compare_periods(old_rows[:count], new_rows[:count])
        result.update(
            periods_within_tolerance=True, periods_exact=all(value == 0 for value in maxima.values()), maxima=maxima
        )
    except ValueError as error:
        result.update(periods_within_tolerance=False, periods_exact=False, period_error=str(error))

    old_captures = {row["frame_index"]: row for row in prior["runtime"]["adaptive_captures"]}
    new_captures = {
        row["frame_index"]: row for row in branch["runtime"]["adaptive_captures"] if row["frame_index"] < len(old_rows)
    }
    common = sorted(old_captures.keys() & new_captures.keys())
    result.update(
        prior_capture_steps=sorted(old_captures),
        new_capture_steps=sorted(new_captures),
        compared_capture_steps=common,
        capture_schedule_matches=old_captures.keys() == new_captures.keys(),
    )
    try:
        for step in common:
            capture_comparison.compare_capture(old_captures[step], new_captures[step])
        result["common_captures_match"] = True if common else None
    except (ValueError, AssertionError) as error:
        result.update(common_captures_match=False, capture_error=str(error))
    return result


def make_trial():
    return registered.RegisteredAdaptiveTrial(
        output=admission.OUTPUT,
        qa=admission.QA,
        dry=admission.DRY,
        module=MODULE,
        gpu_uuid=GPU,
        required_qa_targets=REQUIRED_QA_TARGETS,
        load_inputs=admission.load_inputs,
        input_receipt=admission.input_receipt,
        source_refs=source_refs,
        model_registration=admission.model_registration,
        compare_previous=compare_previous,
    )
