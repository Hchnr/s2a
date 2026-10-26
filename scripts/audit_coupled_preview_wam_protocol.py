"""Replay all061 request/response JSON through the new preview transport check.

This reads historical model outputs only; it neither starts a service nor
qualifies the preview controller for WAM execution.
"""

import json
from pathlib import Path

from audit_wam_position import Evidence

from s2a.coupled_preview_wam_protocol import validate_preview_wam_response
from s2a.terminal_development_policy import case_order


def main():
    e = Evidence()
    for source in (
        __file__,
        "src/s2a/coupled_preview_wam_protocol.py",
        "src/s2a/stability_recovery_simulation.py",
        "src/s2a/terminal_development_policy.py",
    ):
        e.read(source)
    root = Path("outputs/terminal_development_20261025/development/cases")
    candidate = e.json(
        root / "task0_init0_s42_terminal058/simulation/8400_masked_position_hold/task0_init0_s42/response_0000.json"
    )["clean_training"]["readout_ref"]
    e.check(candidate["path"], candidate["sha256"])
    cases = []
    for case in case_order():
        directory = (
            root
            / case["case_id"]
            / "simulation/8400_masked_position_hold"
            / (f"{case['record_name']}_s{case['sampling_seed']}")
        )
        assert len(list(directory.glob("response_*.json"))) == 75
        assert len(list(directory.glob("request_*.json"))) == 75
        checkpoint = None
        for i in range(75):
            response = e.json(directory / f"response_{i:04d}.json")
            request = e.json(directory / f"request_{i:04d}.json")
            actual = response["outputs"][0]["content"]["action128"]["checkpoint_path"]
            if checkpoint is None:
                checkpoint = actual
            actions, _ = validate_preview_wam_response(
                response, request, case=case, request_index=i, checkpoint_path=checkpoint, readout_ref=candidate
            )
            assert actions.shape == (40, 128)
        cases.append(dict(case_id=case["case_id"], requests=75, reference_endpoints=3000))
    e.recheck()
    output = Path("outputs/coupled_preview_wam_protocol_20261026.json")
    with output.open("x") as stream:
        json.dump(
            dict(
                complete=True,
                scope="archived061 full JSON transport replay; not new model execution or physical control qualification",
                cases=cases,
                requests=900,
                reference_endpoints=36000,
                candidate_parameter_ref=candidate,
                input_sha256=e.hashes,
                new_model_calls=0,
                new_physics_steps=0,
                fresh_cases=0,
                execution_authorized=False,
                checkpoint_scope="historical first-response identity, consistent within each case; live service admission remains separate",
            ),
            stream,
            indent=2,
        )
        stream.write("\n")
    print("12 cases / 900 complete archived responses verified; no new execution")


if __name__ == "__main__":
    main()
