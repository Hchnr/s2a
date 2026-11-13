"""Admit the already-qualified producer into the shared fixed-H40 comparison."""

import argparse
from pathlib import Path

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_adaptive_audit as audit
from scripts.simulation.runtime import reset_multitask_adaptive_admission as admission
from scripts.simulation.runtime import run_multitask_fixed_h40 as base


def qualified_case(index):
    """Bind case 8 to its original successful producer and independent checker."""
    prefix.environment()
    prefix.require(index == 8, "Only the originally qualified case uses this admission")
    config = prefix.read({"path": str(base.CONFIG), "sha256": base.CONFIG_SHA})
    prefix.checked({"path": str(base.PLAN), "sha256": base.PLAN_SHA})
    qa = prefix.read(prefix.reference(base.QA))
    prefix.require(
        config["case_indices"] == list(base.INDICES)
        and config["sampling_seeds"] == [42, 43, 42, 43, 42, 43]
        and config["max_control_steps_per_phase"] == 48
        and config["max_native_steps_per_phase"] == 1200
        and config["new_model_requests"] == config["fresh_cases"] == 0
        and config["automatic_retry"] is False
        and qa["passed"] is qa["source_unchanged"] is True,
        "The pinned six-case comparison and its fixed-reference tests are required",
    )
    directory = base.ORIGINAL / "case_08"
    owner_ref = prefix.reference(directory / "producer/result.json")
    checker_ref = prefix.reference(directory / "checker/result.json")
    group_ref = prefix.reference(base.ORIGINAL / "orchestration/checker/result.json")
    owner, checker, group = (prefix.read(ref) for ref in (owner_ref, checker_ref, group_ref))
    group_row = next(row for row in group["cases"] if row["case_index"] == index)
    prefix.require(
        owner["complete"] is owner["passed"] is owner["sources_unchanged"] is owner["gpu_released"] is True
        and checker["complete"] is checker["passed"] is checker["sources_unchanged"] is True
        and checker["phase"] == "checker"
        and group_row["started"] is group_row["case_complete"] is group_row["case_passed"] is True
        and group_row["result_ref"] == checker_ref
        and group_row["worker"]["actual_wait"] is group_row["worker"]["reaped"] is True
        and group_row["worker"]["returncode"] == 0
        and group_row["worker"]["signal_sent"] is False
        and all(value == 0 for value in checker["counts"]["independent_max_absolute_difference"].values()),
        "The original producer and independently reaped checker must both close",
    )
    registration = prefix.read(owner["registration_ref"])
    for ref in registration["source_refs"]:
        prefix.checked(ref)
    inputs = admission.load_inputs(index, verify_assets=False)
    trial = admission.make_trial(index)
    producer, model, bundle = (
        prefix.read(owner["branch_ref"]),
        prefix.read(owner["model_ref"]),
        prefix.read(registration["model_bundle_ref"]),
    )
    checker_branch = prefix.read(checker["branch_ref"])
    prefix.require(
        registration["inputs"] == admission.input_receipt(inputs)
        and bundle == trial.model_bundle(inputs)
        and producer["case"]["case_index"] == checker_branch["case"]["case_index"] == index
        and checker_branch["candidate_modules_unloaded"] is True,
        "The reset, seed and independently isolated checker must match",
    )
    counts = audit.summarize_branch(producer, "producer", inputs["config"])
    exchange = audit.verify_model_exchange(producer, model, bundle)
    prefix.require(
        counts == owner["counts"]
        and exchange == owner["model_exchange"]
        and counts["control_steps"] == checker["counts"]["control_steps"]
        and counts["model_requests"] == exchange["requests"] >= 1,
        "The actual model responses and replayed physical work must agree",
    )
    runtime = producer["runtime"]
    capture, channel, row = (
        runtime["adaptive_captures"][0],
        runtime["adaptive_model_exchanges"][0],
        model["responses"][0],
    )
    prefix.require(capture["frame_index"] == channel["step"] == row["step"] == 8, "The first reference is step8")
    response, envelope = prefix.read(row["actual_response_ref"]), prefix.read(channel["response_ref"])
    refs = dict(
        config=prefix.reference(base.CONFIG),
        plan=prefix.reference(base.PLAN),
        code=prefix.reference(Path(base.__file__).resolve()),
        qualification_code=prefix.reference(Path(__file__).resolve()),
        qa=prefix.reference(base.QA),
        original_owner=owner_ref,
        original_producer=owner["branch_ref"],
        original_model=owner["model_ref"],
        original_replay=checker_ref,
        original_checker_group=group_ref,
        response=row["actual_response_ref"],
        envelope=channel["response_ref"],
        image=capture["canvas_png"],
    )
    for ref in refs.values():
        prefix.checked(ref)
    return config, inputs, producer, capture, response, envelope, registration, refs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-index", type=int, choices=(8, 10), required=True)
    parser.add_argument("--phase", choices=("producer", "checker"))
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    old_load_case = base.load_case
    base.load_case = lambda index: qualified_case(index) if index == 8 else old_load_case(index)
    if args.preflight:
        config, inputs, *_ = base.load_case(args.case_index)
        print(
            {
                "case_index": args.case_index,
                "seed": inputs["config"]["seed"],
                "max_steps": config["max_control_steps_per_phase"],
            }
        )
    elif args.phase:
        base.run_case(args.case_index, args.phase)
    else:
        parser.error("--phase or --preflight is required")


if __name__ == "__main__":
    main()
