"""Close the original port's complete matrix while retaining failed attempts."""

import json
from pathlib import Path

from s2a.artifacts import REPO
from s2a.clean_training_runtime import publish, read, reference, require


def stage_ref(path):
    ref = reference(REPO / "outputs" / path / "result.json")
    stage = read(ref)
    require(stage["complete"] is stage["passed"] is True, "Successful owned stage")
    exit = read(stage["exit_ref"])
    release = read(stage["release_ref"])
    require(
        exit["actual_wait"] is release["reaped"] is True and exit["returncode"] == release["returncode"] == 0,
        "Actual wait0/reap",
    )
    return ref


def main():
    prepared_ref = reference(REPO / "outputs/tracking_correct_reference_20261027/prepared.json")
    prepared = read(prepared_ref)
    cases = []
    for i in range(18):
        if i == 0:
            candidate = "tracking_correct_reference_candidate00_20261027/candidate"
            replay = "tracking_correct_reference_candidate00_replay_v3_20261027/native_replay"
            check = "tracking_correct_reference_check_v3_20261027"
        elif i == 1:
            candidate = "tracking_correct_reference_remaining_20261027/case01/candidate"
            replay = "tracking_correct_reference_remaining_20261027/case01/native_replay"
            check = "tracking_correct_reference_check_v3_20261027"
        elif i == 2:
            candidate = "tracking_correct_reference_remaining_20261027/case02/candidate"
            replay = "tracking_correct_reference_candidate02_replay_v4_20261027/native_replay"
            check = "tracking_correct_reference_check_v4_20261027"
        else:
            candidate = f"tracking_correct_reference_remaining_v4_20261027/case{i:02d}/candidate"
            replay = f"tracking_correct_reference_remaining_v4_20261027/case{i:02d}/native_replay"
            check = "tracking_correct_reference_check_v4_20261027"
        verified_ref = reference(REPO / f"outputs/{check}/case{i:02d}/verified.json")
        verified = read(verified_ref)
        require(verified["complete"] is verified["passed"] is True, "Complete independent replay")
        require(
            verified["physics_steps"] == prepared["profiles"][i]["profile"]["physics_steps"],
            "Original per-case replay budget",
        )
        result = read(verified["result_ref"])
        require(
            result["physics_steps"] == verified["physics_steps"]
            and result["performance_qualified"] == verified["performance_qualified"],
            "Matching original candidate result",
        )
        cases.append(
            dict(
                case_index=i,
                candidate_stage_ref=stage_ref(candidate),
                replay_stage_ref=stage_ref(replay),
                verified_ref=verified_ref,
                physics_steps=verified["physics_steps"],
                performance_qualified=verified["performance_qualified"],
            )
        )
    require(sum(c["physics_steps"] for c in cases) == 4907, "Full18 budget")
    failures = [c["case_index"] for c in cases if not c["performance_qualified"]]
    require(failures == [6], "Observed failed case retained")
    owners = []
    for folder, expected in [
        ("tracking_correct_reference_candidate00_20261027", False),
        ("tracking_correct_reference_candidate00_replay_v2_20261027", False),
        ("tracking_correct_reference_candidate00_replay_v3_20261027", True),
        ("tracking_correct_reference_remaining_20261027", False),
        ("tracking_correct_reference_candidate02_replay_v4_20261027", True),
        ("tracking_correct_reference_remaining_v4_20261027", True),
    ]:
        ref = reference(REPO / f"outputs/{folder}/result.json")
        require(read(ref)["passed"] is expected, "Owner outcome retained")
        owners.append(ref)
    failed_ref = reference(REPO / "outputs/tracking_correct_reference_check_v3_20261027/case02/verified.json")
    require(
        read(failed_ref)["passed"] is False and read(failed_ref)["physics_steps"] == 52, "Failed replay budget retained"
    )
    ref = publish(
        REPO / "outputs/tracking_original_matrix_closed_20261027/verified.json",
        dict(
            complete=True,
            passed=True,
            source_ref=reference(Path(__file__)),
            prepared_ref=prepared_ref,
            cases=cases,
            owner_result_refs=owners,
            candidate_physics_steps=4907,
            successful_replay_physics_steps=4907,
            failed_replay_ref=failed_ref,
            failed_replay_physics_steps=52,
            qualified_cases=17,
            performance_qualified=False,
            failed_cases=failures,
            world_model_calls=0,
            fresh_cases=0,
            scope="Complete original-port execution/performance matrix; failed initialization/observation-order attempts retained; input/image audits not complete for all18. Not ordered-version qualification.",
        ),
    )
    print(json.dumps(ref), flush=True)


if __name__ == "__main__":
    main()
