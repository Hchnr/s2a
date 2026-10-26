"""Independent image/request/normalization checks on12 archived first queries.

Only the new chunk provenance field is populated from the archived response;
this does not make those old trajectories preview-controller physical evidence.
"""

import copy
import json
from pathlib import Path

from audit_wam_position import Evidence
from coupled_preview_wam_acceptance_checker import verify_query


def main():
    e = Evidence()
    e.read(__file__)
    e.read("scripts/coupled_preview_wam_acceptance_checker.py")
    root = Path("outputs/terminal_development_20261025/development")
    head = e.json(
        root
        / "cases/task0_init0_s42_terminal058/simulation/8400_masked_position_hold/task0_init0_s42/response_0000.json"
    )["clean_training"]["readout_ref"]
    cases = [(task, seed, variant) for task in range(3) for seed in (42, 43) for variant in ("raw", "terminal058")]
    results = []
    for i, (task, seed, variant) in enumerate(cases):
        case_id = f"task{task}_init0_s{seed}_{variant}"
        directory = root / "cases" / case_id / "simulation/8400_masked_position_hold" / f"task{task}_init0_s{seed}"
        config = copy.deepcopy(e.json(root / "configs" / f"{case_id}.json"))
        config["checkpoints"][0].update(gpu_uuid="GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f", port=31365)
        config["wam_development"] = dict(
            case=dict(
                case_index=i,
                case_id=case_id,
                task_index=task,
                record_name=f"task{task}_init0",
                sampling_seed=seed,
                variant=variant,
                expected_requests=75,
            ),
            readout_ref=head,
        )
        response = e.json(directory / "response_0000.json")
        chunk = copy.deepcopy(e.json(directory / "chunks.json")[0])
        chunk["response_identity"]["development_response"] = response["clean_training"]
        verify_query(config, directory, 0, chunk, e, "8400_masked", sampling_seed=seed)
        results.append(dict(case_id=case_id, query=0, passed=True))
    e.recheck()
    output = Path("outputs/coupled_preview_wam_queries_20261026.json")
    with output.open("x") as stream:
        json.dump(
            dict(
                complete=True,
                queries=12,
                results=results,
                scope="archived061 current PNG, request/response and normalization; added chunk provenance only; not new physics",
                input_sha256=e.hashes,
                new_model_calls=0,
                new_physics_steps=0,
                fresh_cases=0,
            ),
            stream,
            indent=2,
        )
        stream.write("\n")
    print("12 historical first-query image/JSON/normalization checks passed")


if __name__ == "__main__":
    main()
