"""087 final data qualification with producer/verifier and owner actual wait receipts."""

import json
from pathlib import Path

from audit_wam_position import Evidence

from s2a.artifacts import sha256
from s2a.owned_receipt_contract import normal_receipts

ROOT = Path("outputs/native_arm_response_data_20261025")


def main():
    e = Evidence()
    e.read(__file__)
    for directory in ("orchestration/reconstruct", "verification/data", "owner_wait"):
        folder = ROOT / directory
        child = normal_receipts(
            e, folder / "launch.json", folder / "exit.json", folder / "release.json", format="durable"
        )["child"]
        state = Path(f"/proc/{child['pid']}/stat")
        assert not state.exists() or int(state.read_text().rsplit(")", 1)[1].split()[19]) != child["start_ticks"]
    execution = e.json(ROOT / "execution_result.json")
    assert execution["complete"] and execution["passed"] and len(execution["stages"]) == 2
    supervision = e.json(ROOT / "supervision_result.json")
    assert supervision["owner_actual_wait_recorded"]
    ref = supervision["owner_stage_ref"]
    e.check(ref["path"], ref["sha256"])
    produced = e.json(ROOT / "reconstruction_result.json")
    verified = e.json(ROOT / "independent_verified.json")
    assert produced["complete"] and verified["complete"] and verified["mechanical_data_verified"]
    assert len(produced["trajectories"]) == len(verified["trajectories"]) == 30
    assert produced["rows_by_split"] == verified["rows_by_split"]
    for row in produced["trajectories"]:
        ref = row["cache_ref"]
        e.check(ref["path"], ref["sha256"])
    for path, signature in e.json(ROOT / "source_sha256.json").items():
        e.check(path, signature)
    for path, signature in verified["source_sha256"].items():
        e.check(path, signature)
    e.recheck()
    closure = dict(
        complete=True,
        data_qualified=True,
        verified_trajectories=30,
        fit_trajectories=24,
        holdout_trajectories=6,
        rows_by_split=verified["rows_by_split"],
        owner_normal_wait_verified=True,
        source_roster_unchanged=True,
        source_sha256=e.hashes,
        summary_ref=dict(
            path=str((ROOT / "independent_verified.json").resolve()), sha256=sha256(ROOT / "independent_verified.json")
        ),
        physics_steps=0,
        new_fits=0,
        model_calls=0,
        fresh_cases=0,
        task_success_qualification=False,
    )
    with (ROOT / "closure_verified.json").open("x") as stream:
        json.dump(closure, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("087 closed30trajectories", closure["rows_by_split"], "data only; producer/verifier/owner actualwait0")


if __name__ == "__main__":
    main()
