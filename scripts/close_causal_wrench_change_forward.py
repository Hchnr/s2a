"""112 full numerical/performance/lifecycle audit with explicit held anchors."""

import json
from pathlib import Path

from audit_wam_position import Evidence

from s2a.artifacts import sha256
from s2a.owned_receipt_contract import normal_receipts

ROOT = Path("outputs/causal_wrench_change_forward_20261025")


def main():
    e = Evidence()
    e.read(__file__)
    for directory in ("orchestration/fit", "verification/fit", "verification/validation", "owner_wait"):
        folder = ROOT / directory
        child = normal_receipts(
            e, folder / "launch.json", folder / "exit.json", folder / "release.json", format="durable"
        )["child"]
        path = Path(f"/proc/{child['pid']}/stat")
        assert not path.exists() or int(path.read_text().rsplit(")", 1)[1].split()[19]) != child["start_ticks"]
    execution = e.json(ROOT / "execution_result.json")
    assert execution["complete"] and execution["passed"] and len(execution["stages"]) == 3
    supervision = e.json(ROOT / "supervision_result.json")
    assert supervision["owner_actual_wait_recorded"]
    ref = supervision["owner_stage_ref"]
    e.check(ref["path"], ref["sha256"])
    fit = e.json(ROOT / "fit_result.json")
    numeric = e.json(ROOT / "independent_fit_verified.json")
    validation = e.json(ROOT / "validation_result.json")
    assert fit["complete"] and fit["new_fits"] == 1 and fit["rows"] == 5000 and fit["fit_trajectories"] == 24
    assert numeric["complete"] and numeric["numerically_verified"] and numeric["rows"] == 5000
    assert validation["complete"] and validation["control_qualification"] is False
    assert fit["native_rows"] == 4865 and fit["contrast_equations"] == 135
    assert validation["training_anchor_groups"] == [0, 3, 5] and validation["validation_anchor_groups"] == [1, 2, 4]
    assert len(validation["holdout"]) == len(validation["local_response"]) == 6
    assert sum(p["rows"] for p in validation["holdout"]) == 1315
    for row in validation["holdout"]:
        gates = {
            k: row["metrics"]["fitted"][k] <= row["metrics"]["nominal"][k]
            for k in ("position", "rotation", "linear_velocity", "angular_velocity")
        }
        assert gates == row["gates"]
        ref = row["predictions_ref"]
        e.check(ref["path"], ref["sha256"])
    for row in validation["local_response"]:
        gates = {k: v["fitted_frobenius_error"] <= v["nominal_frobenius_error"] for k, v in row["metrics"].items()}
        assert gates == row["gates"] and len(gates) == 4
    for row in validation["local_response"]:
        assert row["used_for_contrast_fit"] == (row["anchor_group"] in [0, 3, 5])
    assert sum(row["used_for_contrast_fit"] for row in validation["local_response"]) == 3
    held = all(all(x["gates"].values()) for x in validation["holdout"])
    local = all(all(x["gates"].values()) for x in validation["local_response"])
    assert validation["holdout_gate_passed"] == held and validation["local_response_gate_passed"] == local
    assert validation["offline_forward_eligible"] == (held and local)
    for ref in (fit["parameter_ref"], fit["cache_ref"]):
        e.check(ref["path"], ref["sha256"])
    for path, signature in e.json(ROOT / "source_sha256.json").items():
        e.check(path, signature)
    for path, signature in validation["source_sha256"].items():
        e.check(path, signature)
    e.recheck()
    result = dict(
        complete=True,
        numerically_verified=True,
        owner_normal_wait_verified=True,
        source_roster_unchanged=True,
        fit_rows=5000,
        history_dimension=12,
        state_dimension=89,
        design_dimension=1152,
        current_measured_wrench_condition=True,
        native_fit_rows=4865,
        contrast_equations=135,
        holdout_rows=1315,
        local_response_groups=6,
        new_fits=1,
        holdout_passed_gates=sum(sum(x["gates"].values()) for x in validation["holdout"]),
        local_response_passed_gates=sum(sum(x["gates"].values()) for x in validation["local_response"]),
        untrained_local_response_passed_gates=sum(
            sum(x["gates"].values()) for x in validation["local_response"] if not x["used_for_contrast_fit"]
        ),
        training_response_passed_gates=sum(
            sum(x["gates"].values()) for x in validation["local_response"] if x["used_for_contrast_fit"]
        ),
        offline_forward_eligible=held and local,
        control_qualification=False,
        source_sha256=e.hashes,
        validation_ref=dict(
            path=str((ROOT / "validation_result.json").resolve()), sha256=sha256(ROOT / "validation_result.json")
        ),
        physics_steps=0,
        model_calls=0,
        fresh_cases=0,
    )
    with (ROOT / "closure_verified.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        "112 closed;holdout",
        result["holdout_passed_gates"],
        "/24;local",
        result["local_response_passed_gates"],
        "/24;eligible",
        result["offline_forward_eligible"],
    )


if __name__ == "__main__":
    main()
