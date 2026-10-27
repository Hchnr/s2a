"""Verify078 complete collection, every stage's wait/reap, and the execution owner wait."""

import json
from pathlib import Path

from audit_wam_position import Evidence

from s2a.owned_receipt_contract import normal_receipts

ROOT = Path("outputs/native_gripper_response_interventions_20261025_v3")


def main():
    e = Evidence()
    e.read(__file__)
    execution = e.json(ROOT / "execution_result.json")
    assert execution["complete"] and execution["passed"] and len(execution["stages"]) == 24
    assert [(d["case"], d["kind"]) for d in execution["stages"][:6]] == [
        (0, "collect"),
        (0, "verify"),
        (4, "collect"),
        (4, "verify"),
        (8, "collect"),
        (8, "verify"),
    ]
    children = []
    datasets = []
    for i in range(12):
        for directory in (ROOT / "orchestration" / f"case{i:02d}", ROOT / "verification" / f"case{i:02d}"):
            launch = normal_receipts(
                e, directory / "launch.json", directory / "exit.json", directory / "release.json", format="durable"
            )
            child = launch["child"]
            p = Path(f"/proc/{child['pid']}/stat")
            assert not p.exists() or int(p.read_text().rsplit(")", 1)[1].split()[19]) != child["start_ticks"]
            assert (
                launch["selected_environment"]["CUDA_VISIBLE_DEVICES"]
                == launch["selected_environment"]["LD_LIBRARY_PATH"]
                == ""
            )
            children.append(dict(pid=child["pid"], start_ticks=child["start_ticks"], wait0_reaped_no_signal=True))
        folder = ROOT / f"case{i:02d}"
        verified = e.json(folder / "independent_verified.json")
        assert verified["complete"] and verified["mechanical_verified"]
        if i in (0, 4, 8):
            assert verified["baseline_native_match"]
        result = e.json(folder / "collection_result.json")
        assert result["complete"]
        for path, sha in result["source_sha256"].items():
            e.check(path, sha)
        e.check(result["arrays_ref"]["path"], result["arrays_ref"]["sha256"])
        datasets.append(
            dict(
                case=i,
                branch=result["profile"]["branch"],
                arrays_ref=result["arrays_ref"],
                physics_steps=result["physics_steps"],
            )
        )
    owner = normal_receipts(
        e,
        ROOT / "owner_wait/launch.json",
        ROOT / "owner_wait/exit.json",
        ROOT / "owner_wait/release.json",
        format="durable",
    )["child"]
    p = Path(f"/proc/{owner['pid']}/stat")
    assert not p.exists() or int(p.read_text().rsplit(")", 1)[1].split()[19]) != owner["start_ticks"]
    supervision = e.json(ROOT / "supervision_result.json")
    e.check(supervision["owner_stage_ref"]["path"], supervision["owner_stage_ref"]["sha256"])
    assert supervision["owner_actual_wait_recorded"]
    for p, sha in e.json(ROOT / "source_sha256.json").items():
        e.check(p, sha)
    assert sum(d["physics_steps"] for d in datasets) == 1568
    e.recheck()
    output = dict(
        complete=True,
        collection_qualified=True,
        owner_normal_wait_verified=True,
        verified_cases=12,
        baseline_native_matches=3,
        physics_steps=1568,
        children=children,
        datasets=datasets,
        source_sha256=e.hashes,
        new_fits=0,
        model_calls=0,
        fresh_cases=0,
        task_success_qualification=False,
    )
    with (ROOT / "closure_verified.json").open("x") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
    print(
        "12cases/1568steps qualified;3 native baselines;24 stages and execution owner actualwait0/reap verified;no task claim"
    )


if __name__ == "__main__":
    main()
