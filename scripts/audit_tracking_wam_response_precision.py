"""Check tighter independent response integration against all closed v4 matrices."""

import json
from pathlib import Path

import numpy as np
from current_coupled_precision_independent import response

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.preview_result_publication import publish_result as publish
from s2a.terminal_development_policy import case_order


def main():
    root = REPO / "outputs/tracking_wam_response_precision_20261028"
    require(not root.exists(), "No overwrite or automatic retry")
    production = REPO / "outputs/tracking_preview_wam_development_v4_20261027"
    owner_ref = reference(production / "development/orchestration/result.json")
    require(read(owner_ref)["complete"] is read(owner_ref)["passed"] is True, "Closed original12 producer")
    sources = [reference(Path(__file__)), reference(REPO / "scripts/current_coupled_precision_independent.py")]
    registration = publish(
        root / "registration.json",
        dict(
            owner_ref=owner_ref,
            source_refs=sources,
            case_indices=list(range(12)),
            original_matrix_atol=1e-8,
            rtol=0,
            physics_budget=0,
            world_model_calls=0,
            fresh_cases=0,
            scope="Improve independent numerical derivative/integration accuracy; original acceptance tolerance unchanged",
        ),
    )
    results = []
    for case in case_order():
        worker_ref = reference(production / "development/cases" / case["case_id"] / "worker_result.json")
        worker = read(worker_ref)
        rows_ref = reference(Path(worker["physical_case_dir"]) / "steps.jsonl")
        rows = [json.loads(x) for x in checked(rows_ref).read_text().splitlines()]
        require(len(rows) == 600, "Original full600")
        maxima = {}
        for i, row in enumerate(rows):
            d = row["coupled_dynamics"]
            computed = response(d["mass"], d["jacobian"], d["kp"], d["kd"], d["relative_rotation_goal"])
            for key, value in computed.items():
                original = np.asarray(row["coupled_response_model"][key])
                difference = float(np.max(abs(original - value)))
                if difference > maxima.get(key, {}).get("max_abs", -1):
                    maxima[key] = dict(step=i, max_abs=difference)
                np.testing.assert_allclose(
                    original, value, atol=1e-8, rtol=0, err_msg=f"{case['case_id']} step{i} {key}"
                )
        results.append(dict(case=case, rows_ref=rows_ref, periods=600, maxima=maxima))
        publish(root / f"case{case['case_index']:02d}.json", results[-1])
        print(
            json.dumps(dict(case=case["case_id"], passed=True, max_abs=max(v["max_abs"] for v in maxima.values()))),
            flush=True,
        )
    print(
        publish(
            root / "verified.json",
            dict(
                complete=True,
                passed=True,
                registration_ref=registration,
                cases=results,
                periods=7200,
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
                scope="All7200 saved-input response matrices only; original full development audit remains separate",
            ),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
