"""Measure actual event coverage of registered candidate H40 window unions."""

from pathlib import Path

import numpy as np

from s2a.infrastructure.publication import publish_result
from s2a.research.wam.training.broader_recovery_h40 import clipping_flags
from scripts.simulation.runtime import prepare_reset_correct_reference_cohort as preparation
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_start_feedback as feedback
from scripts.verification.wam.training import audit_broader_recovery_events as primary
from scripts.verification.wam.training.audit_recovery_h40_contract import Evidence, load_data
from scripts.verification.wam.training.verify_broader_recovery_h40 import scalar_normalization

ROOT = prefix.REPO
PRIMARY = ROOT / "outputs/broader_recovery_events_20261116/result.json"
INDEPENDENT = ROOT / "outputs/broader_recovery_events_20261116/independent.json"
OUTPUT = ROOT / "outputs/broader_recovery_events_20261116/union.json"


def summarize_union(data, candidate_steps, normalization):
    """Count measured events and clipping on unique old/new case-time indices."""
    periods = len(data["opening"]) - 1
    prefix.require(
        periods >= 96
        and data["native_success"].shape == (periods + 1,)
        and data["action128"].shape == (periods + 1, 128)
        and len(candidate_steps) == len(set(candidate_steps))
        and all(type(step) is int and 0 <= step <= periods - 40 for step in candidate_steps),
        "Unique full-horizon actual candidates and endpoints are required",
    )
    existing = set(range(0, 40)) | set(range(16, 56)) | set(range(56, 96))
    candidates = {index for step in candidate_steps for index in range(step, step + 40)}
    groups = dict(existing=existing, candidates=candidates, union=existing | candidates, added=candidates - existing)
    result = {}
    for name, indices in groups.items():
        ordered = sorted(indices)
        low = [index for index in ordered if float(data["opening"][index + 1]) < 0.35]
        successful = [index for index in ordered if bool(data["native_success"][index + 1])]
        flags = clipping_flags(data["action128"][ordered], normalization)
        _, independent_flags = scalar_normalization(data["action128"][ordered], normalization)
        np.testing.assert_array_equal(flags, independent_flags)
        prefix.require(
            len(low) == int(np.count_nonzero(data["opening"][np.asarray(ordered, dtype=int) + 1] < 0.35))
            and len(successful) == int(np.count_nonzero(data["native_success"][np.asarray(ordered, dtype=int) + 1])),
            "Scalar and vectorized event counts must agree",
        )
        result[name] = dict(
            unique_case_transition_rows=len(ordered),
            low_opening_endpoints=len(low),
            native_success_endpoints=len(successful),
            clipped_per_active_slot=flags.sum(axis=0).tolist(),
            clipped_total=int(flags.sum()),
        )
    return result


def verify():
    prefix.environment()
    feedback.require_no_simulator_import()
    evidence = Evidence()
    source_ref, checked_ref = prefix.reference(PRIMARY), prefix.reference(INDEPENDENT)
    source, checked = evidence.json(source_ref), evidence.json(checked_ref)
    config = evidence.json(prefix.reference(primary.CONFIG))
    primary.validate_config(config)
    refs = {key: preparation.configured_ref(config[key]) for key in primary.REFERENCES}
    normalization, h40 = (evidence.json(refs[key]) for key in ("normalization", "h40_audit"))
    for path in (Path(__file__).resolve(), ROOT / "tests/research/wam/training/test_broader_recovery_event_union.py"):
        evidence.path(prefix.reference(path))
    for ref in source["input_refs"] + checked["input_refs"]:
        evidence.path(ref)
    prefix.require(
        source["complete"] is source["passed"] is checked["complete"] is checked["passed"] is True
        and checked["primary_ref"] == source_ref
        and source["registered_cases"] == len(source["cases"]) == checked["cases_checked"] == 24
        and source["training_authorized"] is checked["training_authorized"] is False,
        "The complete independently checked event-candidate roster is required",
    )
    cases = []
    for index, case in enumerate(source["cases"]):
        prefix.require(case["case_index"] == index, "Original case order is required")
        data = load_data(evidence, case["trajectory_ref"])
        groups = summarize_union(data, [row["anchor_step"] for row in case["candidates"]], normalization)
        prefix.require(
            groups["existing"]["unique_case_transition_rows"] == h40["cases"][index]["unique_transition_rows"]
            and groups["existing"]["clipped_per_active_slot"] == h40["cases"][index]["unique_clipping_per_active_slot"]
            and groups["candidates"]["unique_case_transition_rows"] == case["candidate_unique_transition_rows"]
            and groups["union"]["unique_case_transition_rows"] == case["union_unique_transition_rows"]
            and groups["added"]["unique_case_transition_rows"] == case["additional_unique_transition_rows"]
            and groups["existing"]["low_opening_endpoints"] == case["low_opening"]["covered"]
            and groups["existing"]["native_success_endpoints"] == case["native_success"]["covered"],
            "Candidate union and previous measured H40 counts must agree",
        )
        cases.append(
            dict(case_index=index, source_identity=case["source_identity"], variant=case["variant"], groups=groups)
        )
    totals = {}
    for name in ("existing", "candidates", "union", "added"):
        selected = [case["groups"][name] for case in cases]
        totals[name] = dict(
            unique_case_transition_rows=sum(row["unique_case_transition_rows"] for row in selected),
            low_opening_endpoints=sum(row["low_opening_endpoints"] for row in selected),
            native_success_endpoints=sum(row["native_success_endpoints"] for row in selected),
            clipped_per_active_slot=np.sum([row["clipped_per_active_slot"] for row in selected], axis=0).tolist(),
            clipped_total=sum(row["clipped_total"] for row in selected),
        )
    prefix.require(
        totals["existing"]["unique_case_transition_rows"] == h40["unique_case_transition_rows"]
        and totals["existing"]["clipped_per_active_slot"] == h40["unique_clipping_per_active_slot"]
        and totals["candidates"]["unique_case_transition_rows"] == source["candidate_unique_case_transition_rows"]
        and totals["union"]["unique_case_transition_rows"]
        == source["existing_and_candidate_unique_case_transition_rows"],
        "The source H40 and candidate-union denominators must remain unchanged",
    )
    evidence.verify()
    feedback.require_no_simulator_import()
    return dict(
        complete=True,
        passed=True,
        primary_ref=source_ref,
        independent_ref=checked_ref,
        input_refs=list(evidence.refs.values()),
        registered_cases=len(cases),
        cases=cases,
        totals=totals,
        candidate_images_are_hypothetical=True,
        training_authorized=False,
        world_model_calls=0,
        gpu_calls=0,
        physical_steps=0,
        new_current_images=0,
        fits=0,
        fresh_cases=0,
    )


def main():
    if OUTPUT.exists():
        raise SystemExit("Candidate-union output already exists; no automatic retry or overwrite")
    try:
        result = verify()
    except Exception as error:
        publish_result(
            OUTPUT, dict(complete=False, passed=False, error=dict(type=type(error).__name__, message=str(error)))
        )
        raise
    print(publish_result(OUTPUT, result), flush=True)


if __name__ == "__main__":
    main()
