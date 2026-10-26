"""Close084 offline oracle ablation with actual stage/owner lifecycle evidence."""

import json
from pathlib import Path

from audit_wam_position import Evidence

from s2a.artifacts import sha256
from s2a.owned_receipt_contract import normal_receipts

ROOT = Path("outputs/native_gripper_ablation_20261025")


def absent(child):
    path = Path(f"/proc/{child['pid']}/stat")
    assert not path.exists() or int(path.read_text().rsplit(")", 1)[1].split()[19]) != child["start_ticks"]


def main():
    evidence = Evidence()
    evidence.read(__file__)
    config = evidence.json("configs/native_gripper_ablation_20261025.json")
    closure_ref = config["prior_closure_ref"]
    evidence.check(closure_ref["path"], closure_ref["sha256"])
    prior = evidence.json(closure_ref["path"])
    assert prior["complete"] and prior["verified_cases"] == 18 and prior["owner_normal_wait_verified"]
    summary_ref = prior["summary_ref"]
    evidence.check(summary_ref["path"], summary_ref["sha256"])
    original = evidence.json(summary_ref["path"])
    execution = evidence.json(ROOT / "execution_result.json")
    assert execution["complete"] and execution["passed"] and len(execution["stages"]) == 6
    assert execution["physics_steps"] == config["physics_steps"] == 566
    assert [p["prior_case_index"] for p in config["profiles"]] == [3, 10, 12]
    cases = []
    for i, entry in enumerate(config["profiles"]):
        for parent in ("orchestration", "verification"):
            folder = ROOT / parent / f"case{i:02d}"
            launch = normal_receipts(
                evidence, folder / "launch.json", folder / "exit.json", folder / "release.json", format="durable"
            )
            absent(launch["child"])
            assert (
                launch["selected_environment"]["CUDA_VISIBLE_DEVICES"]
                == launch["selected_environment"]["LD_LIBRARY_PATH"]
                == ""
            )
        folder = ROOT / f"case{i:02d}"
        verified = evidence.json(folder / "independent_verified.json")
        assert verified["complete"] and verified["mechanical_labels_images_passed"]
        assert verified["oracle_ablation_only"] and verified["deployable_s2a_qualification"] is False
        assert verified["state_derived_gripper_policy"] is False
        assert verified["oracle_gripper_command_steps"] == entry["profile"]["physics_steps"]
        ref = entry["profile"]["trajectory_ref"]
        evidence.check(ref["path"], ref["sha256"])
        baseline = original["cases"][entry["prior_case_index"]]
        assert not baseline["performance_qualified"]
        control = verified["control"]
        cases.append(
            dict(
                case=i,
                prior_case_index=entry["prior_case_index"],
                record=entry["profile"]["record"]["name"],
                physics_steps=entry["profile"]["physics_steps"],
                outcome=control["outcome"],
                tracking=control["tracking"]["admitted"],
                oracle_ablation_tracking_and_task_passed=verified["performance_qualified"],
                original083_outcome=baseline["outcome"],
            )
        )
    folder = ROOT / "owner_wait"
    launch = normal_receipts(
        evidence, folder / "launch.json", folder / "exit.json", folder / "release.json", format="durable"
    )
    absent(launch["child"])
    supervision = evidence.json(ROOT / "supervision_result.json")
    assert supervision["owner_actual_wait_recorded"]
    evidence.check(supervision["owner_stage_ref"]["path"], supervision["owner_stage_ref"]["sha256"])
    for path, signature in evidence.json(ROOT / "source_sha256.json").items():
        evidence.check(path, signature)
    assert sum(c["physics_steps"] for c in cases) == 566
    evidence.recheck()
    result = dict(
        complete=True,
        mechanical_qualification=True,
        owner_normal_wait_verified=True,
        oracle_ablation_only=True,
        deployable_s2a_qualification=False,
        source_roster_unchanged=True,
        physics_steps=566,
        candidate_trials=3,
        oracle_ablation_stable=sum(c["outcome"]["stable"] for c in cases),
        cases=cases,
        model_calls=0,
        new_fits=0,
        fresh_cases=0,
    )
    with (ROOT / "ablation_summary.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    evidence.read(ROOT / "ablation_summary.json")
    evidence.recheck()
    closure = dict(
        complete=True,
        verified_cases=3,
        owner_normal_wait_verified=True,
        oracle_ablation_only=True,
        deployable_s2a_qualification=False,
        summary_ref=dict(
            path=str((ROOT / "ablation_summary.json").resolve()), sha256=sha256(ROOT / "ablation_summary.json")
        ),
        source_sha256=evidence.hashes,
        physics_steps=566,
        model_calls=0,
        fresh_cases=0,
    )
    with (ROOT / "closure_verified.json").open("x") as stream:
        json.dump(closure, stream, indent=2)
        stream.write("\n")
    print("084 closed: oracle ablation stable", result["oracle_ablation_stable"], "/3; not deployable qualification")


if __name__ == "__main__":
    main()
