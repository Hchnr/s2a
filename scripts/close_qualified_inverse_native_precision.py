"""160 complete physical qualification and actual owner/worker wait audit."""

import json
from pathlib import Path

import numpy as np
from audit_wam_position import Evidence

from s2a.artifacts import sha256
from s2a.owned_receipt_contract import normal_receipts

ROOT = Path("outputs/qualified_inverse_native_precision_20261025")


def absent(child):
    path = Path(f"/proc/{child['pid']}/stat")
    assert not path.exists() or int(path.read_text().rsplit(")", 1)[1].split()[19]) != child["start_ticks"]


def main():
    e = Evidence()
    e.read(__file__)
    execution = e.json(ROOT / "execution_result.json")
    assert execution["complete"] and execution["passed"] and len(execution["stages"]) == 36
    config = e.json("configs/qualified_inverse_native_precision_20261025.json")
    prior_path = Path("outputs/pose_velocity_loaded_control_20261025/regression_summary.json")
    prior = e.json(prior_path)
    assert prior["complete"] and prior["candidate_trials"] == 18
    prior_config = e.json("configs/pose_velocity_loaded_control_20261025.json")
    assert config["profiles"] == prior_config["profiles"]
    cases = []
    for i, profile in enumerate(config["profiles"]):
        folder = ROOT / f"case{i:02d}"
        for parent in ("orchestration", "verification"):
            directory = ROOT / parent / f"case{i:02d}"
            launch = normal_receipts(
                e, directory / "launch.json", directory / "exit.json", directory / "release.json", format="durable"
            )
            absent(launch["child"])
            assert (
                launch["selected_environment"]["CUDA_VISIBLE_DEVICES"]
                == launch["selected_environment"]["LD_LIBRARY_PATH"]
                == ""
            )
        verified = e.json(folder / "independent_verified.json")
        assert verified["complete"] and verified["mechanical_labels_images_passed"]
        collection = e.json(folder / "collection_result.json")
        assert collection["complete"]
        ref = collection["arrays_ref"]
        e.check(ref["path"], ref["sha256"])
        e.check(profile["trajectory_ref"]["path"], profile["trajectory_ref"]["sha256"])
        with (
            np.load(ref["path"], allow_pickle=False) as actual,
            np.load(profile["trajectory_ref"]["path"], allow_pickle=False) as original,
        ):
            desired = original["opening"][1:] if profile["format"] == "native_endpoints" else original["openings"]
            assert len(actual["opening"]) == len(desired) + 1 == profile["physics_steps"] + 1
            opening_rmse = float(np.sqrt(np.mean((actual["opening"][1:] - desired) ** 2)))
        routing = verified["routing_summary"]
        assert routing["native_load_cycles"] + routing["tracking_cycles"] == profile["physics_steps"]
        ctrl = verified["control"]
        qualified = (
            ctrl["outcome"]["stable"]
            and ctrl["tracking"]["admitted"]["position_rmse_m"] <= 0.01
            and ctrl["tracking"]["admitted"]["so3_rmse_rad"] <= 0.05
        )
        assert qualified == verified["performance_qualified"]
        cases.append(
            dict(
                case=i,
                record=profile["record"]["name"],
                physics_steps=profile["physics_steps"],
                outcome=ctrl["outcome"],
                tracking=ctrl["tracking"]["admitted"],
                opening_rmse=opening_rmse,
                opening_rmse_prior083=prior["cases"][i]["opening_rmse"],
                opening_rmse_change_vs083=opening_rmse - prior["cases"][i]["opening_rmse"],
                prior083_stable=prior["cases"][i]["outcome"]["stable"],
                routing_summary=verified["routing_summary"],
                performance_qualified=bool(qualified),
            )
        )
    owner = normal_receipts(
        e,
        ROOT / "owner_wait/launch.json",
        ROOT / "owner_wait/exit.json",
        ROOT / "owner_wait/release.json",
        format="durable",
    )["child"]
    absent(owner)
    supervision = e.json(ROOT / "supervision_result.json")
    assert supervision["owner_actual_wait_recorded"]
    e.check(supervision["owner_stage_ref"]["path"], supervision["owner_stage_ref"]["sha256"])
    for p, s in e.json(ROOT / "source_sha256.json").items():
        e.check(p, s)
    assert len(cases) == 18 and sum(d["physics_steps"] for d in cases) == 4907
    e.recheck()
    summary = dict(
        complete=True,
        execution_and_independent_checks_passed=True,
        performance_qualified=all(c["performance_qualified"] for c in cases),
        candidate_stable=sum(c["outcome"]["stable"] for c in cases),
        candidate_trials=18,
        prior083_stable=prior["candidate_stable"],
        opening_rmse_regressed_cases=sum(c["opening_rmse_change_vs083"] > 1e-10 for c in cases),
        executed_predictions_outside_velocity_envelope=sum(
            c["routing_summary"]["executed_predictions_outside_velocity_envelope"] for c in cases
        ),
        physics_steps=4907,
        cases=cases,
        owner_normal_wait_verified=True,
        source_roster_unchanged=True,
        model_calls=0,
        new_fits=0,
        fresh_cases=0,
    )
    payload = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    with (ROOT / "regression_summary.json").open("x") as stream:
        stream.write(payload)
    e.read(ROOT / "regression_summary.json")
    e.recheck()
    closure = dict(
        complete=True,
        verified_cases=18,
        owner_normal_wait_verified=True,
        mechanical_qualification=True,
        performance_qualified=summary["performance_qualified"],
        summary_ref=dict(
            path=str((ROOT / "regression_summary.json").resolve()), sha256=sha256(ROOT / "regression_summary.json")
        ),
        source_sha256=e.hashes,
        physics_steps=4907,
        model_calls=0,
        new_fits=0,
        fresh_cases=0,
    )
    with (ROOT / "closure_verified.json").open("x") as stream:
        json.dump(closure, stream, indent=2)
        stream.write("\n")
    print(
        "160 complete",
        summary["candidate_stable"],
        "/18;36stages+owner normalwait0 verified;performance",
        summary["performance_qualified"],
    )


if __name__ == "__main__":
    main()
