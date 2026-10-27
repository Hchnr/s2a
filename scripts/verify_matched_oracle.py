"""Independent full062 mechanical and native-outcome replay; no simulation."""

import json

from audit_wam_position import Evidence
from wam_position_acceptance_checker import source_proof, verify_case

from s2a.artifacts import REPO
from s2a.owned_receipt_contract import normal_receipts

ROOT = REPO / "outputs/matched_oracle_requalification_20261025_v2"


def main():
    evidence = Evidence()
    evidence.read(__file__)
    proof = source_proof(evidence)
    config = evidence.json(REPO / "configs/matched_oracle_requalification_20261025.json")
    execution = evidence.json(ROOT / "execution_result.json")
    if not execution["complete"] or not execution["passed"] or len(execution["stages"]) != 3:
        raise ValueError("Requires three normally completed CPU stages")
    cases = []
    for record in config["records"]:
        directory = ROOT / "orchestration" / record
        normal_receipts(
            evidence,
            directory / "launch.json",
            directory / "exit.json",
            directory / "release.json",
            format="durable",
        )
        case = ROOT / "simulation/oracle_position_hold" / f"{record}_s42"
        manifest = evidence.json(case / "manifest.json")
        if list(case.glob("request_*.json")) or list(case.glob("response_*.json")):
            raise ValueError("Oracle must have zero WAM requests")
        result = verify_case(
            config,
            manifest["record"],
            config["variants"][0],
            case,
            evidence,
            source="oracle",
            checkpoint=None,
            full=True,
            sampling_seed=42,
            first_query_validator=None,
        )
        cases.append(result)
    qualified = all(
        c["outcome"]["stable"]
        and c["chunks"] == 75
        and c["tracking"]["admitted"]["position_rmse_m"] <= 0.01
        and c["tracking"]["admitted"]["so3_rmse_rad"] <= 0.05
        for c in cases
    )
    evidence.recheck()
    result = dict(
        complete=True,
        qualified=qualified,
        cases=cases,
        source_proof=proof,
        input_sha256=evidence.hashes,
        new_model_calls=0,
        new_physics_steps=0,
        fresh_cases=0,
        scope="Correct-reference control qualification only; not WAM success",
    )
    with (ROOT / "independent_verified.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(complete=True, qualified=qualified, cases=len(cases))))


if __name__ == "__main__":
    main()
