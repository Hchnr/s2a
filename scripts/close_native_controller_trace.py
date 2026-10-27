"""093 full replay/trace/actual-wait closure; no task qualification."""

import json
from pathlib import Path

from audit_wam_position import Evidence

from s2a.owned_receipt_contract import normal_receipts

ROOT = Path("outputs/native_controller_trace_20261025")


def main():
    e = Evidence()
    e.read(__file__)
    execution = e.json(ROOT / "execution_result.json")
    assert execution["complete"] and execution["passed"] and len(execution["stages"]) == 12
    supervision = e.json(ROOT / "supervision_result.json")
    assert supervision["owner_actual_wait_recorded"]
    ref = supervision["owner_stage_ref"]
    e.check(ref["path"], ref["sha256"])
    rows = []
    for group in range(6):
        for kind in ("orchestration", "verification"):
            folder = ROOT / kind / f"group{group:02d}"
            child = normal_receipts(
                e, folder / "launch.json", folder / "exit.json", folder / "release.json", format="durable"
            )["child"]
            path = Path(f"/proc/{child['pid']}/stat")
            assert not path.exists() or int(path.read_text().rsplit(")", 1)[1].split()[19]) != child["start_ticks"]
        v = e.json(ROOT / f"group{group:02d}/independent_verified.json")
        assert v["complete"] and v["prefix_matches"] and v["inner_calls"] == 25
        for path, signature in v["source_sha256"].items():
            e.check(path, signature)
        rows.append(
            dict(
                group=group,
                physics_steps=v["physics_steps"],
                actual_uncoupling=v["actual_uncoupling"],
                saturated_inner_calls=v["saturated_inner_calls"],
            )
        )
    assert sum(r["physics_steps"] for r in rows) == 477
    folder = ROOT / "owner_wait"
    normal_receipts(e, folder / "launch.json", folder / "exit.json", folder / "release.json", format="durable")
    for path, signature in e.json(ROOT / "source_sha256.json").items():
        e.check(path, signature)
    e.recheck()
    result = dict(
        complete=True,
        owner_normal_wait_verified=True,
        source_roster_unchanged=True,
        groups=rows,
        physics_steps=477,
        inner_calls=150,
        environment_initializations=6,
        initialization_internal_steps_not_counted=True,
        model_calls=0,
        new_fits=0,
        fresh_cases=0,
        control_qualification=False,
        source_sha256=e.hashes,
    )
    with (ROOT / "closure_verified.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(rows)


if __name__ == "__main__":
    main()
