"""096 audit fixed gates and actual wait/reap receipts without upgrading scope."""

import json
from pathlib import Path

from audit_wam_position import Evidence

from s2a.owned_receipt_contract import normal_receipts

ROOT = Path("outputs/causal_local_response_20261025/owned")


def main():
    e = Evidence()
    e.read(__file__)
    for sub in ("stages/holdout", "stages/anchors", "owner_wait"):
        folder = ROOT / sub
        child = normal_receipts(
            e, folder / "launch.json", folder / "exit.json", folder / "release.json", format="durable"
        )["child"]
        stat = Path(f"/proc/{child['pid']}/stat")
        assert not stat.exists() or int(stat.read_text().rsplit(")", 1)[1].split()[19]) != child["start_ticks"]
    execution = e.json(ROOT / "execution_result.json")
    supervision = e.json(ROOT / "supervision_result.json")
    assert execution["complete"] and execution["passed"] and len(execution["stages"]) == 2
    assert supervision["owner_actual_wait_recorded"]
    ref = supervision["owner_stage_ref"]
    e.check(ref["path"], ref["sha256"])
    held = e.json(ROOT / "results/holdout_prequential.json")
    local = e.json(ROOT / "results/local_response.json")
    assert held["complete"] and local["complete"]
    assert len(held["holdout"]) == len(local["local_response"]) == 6
    assert sum(r["rows"] for r in held["holdout"]) == 1315
    for row in held["holdout"]:
        assert row["gates"] == {
            k: row["metrics"]["adapted"][k] <= row["metrics"]["nominal"][k]
            for k in ("position", "rotation", "linear_velocity", "angular_velocity")
        }
    for row in local["local_response"]:
        assert row["probe_endpoint_updates"] == 0
        assert row["gates"] == {
            k: v["fitted_frobenius_error"] <= v["nominal_frobenius_error"] for k, v in row["metrics"].items()
        }
        assert row["used_for_contrast_fit"] == (row["anchor_group"] in (0, 3, 5))
    for result in (held, local):
        for path, signature in result["source_sha256"].items():
            e.check(path, signature)
    for path, signature in e.json(ROOT / "source_sha256.json").items():
        e.check(path, signature)
    h = sum(sum(r["gates"].values()) for r in held["holdout"])
    local_passed = sum(sum(r["gates"].values()) for r in local["local_response"])
    e.recheck()
    result = dict(
        complete=True,
        owner_normal_wait_verified=True,
        source_roster_unchanged=True,
        holdout_passed_gates=h,
        local_response_passed_gates=local_passed,
        offline_forward_eligible=h == local_passed == 24,
        control_qualification=False,
        online_updates_this_execution=held["online_updates"] + local["online_updates"],
        independent_numerical_solves_this_execution=held["independent_numerical_solves"]
        + local["independent_numerical_solves"],
        new_global_fits=0,
        physics_steps=0,
        model_calls=0,
        fresh_cases=0,
        source_sha256=e.hashes,
    )
    with (ROOT / "closure_verified.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print({k: v for k, v in result.items() if k != "source_sha256"})


if __name__ == "__main__":
    main()
