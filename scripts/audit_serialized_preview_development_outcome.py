"""Read-only outcome bound for the stopped development; no mechanical qualification."""

import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import publish, read, reference, require


def main():
    root = REPO / "outputs/serialized_preview_wam_development_20261026"
    prepared_ref = reference(root / "prepared.json")
    prepared = read(prepared_ref)
    records = []
    completed_terminal = stable_terminal = 0
    for case in prepared["case_order"]:
        folder = root / "development/cases" / case["case_id"]
        worker_path = folder / "worker_result.json"
        if not worker_path.exists():
            records.append(dict(case=case, status="not_collected"))
            continue
        worker_ref = reference(worker_path)
        worker = read(worker_ref)
        result_paths = list(folder.glob("simulation/*/*/result.json"))
        require(len(result_paths) == 1, "Exactly one physical result required")
        result_ref = reference(result_paths[0])
        result = read(result_ref)
        rows_ref = reference(result_paths[0].parent / "steps.jsonl")
        rows = [json.loads(line) for line in Path(rows_ref["path"]).read_text().splitlines()]
        require(len(rows) == result["steps"], "Physical result/log count mismatch")
        require(
            all(row["step"] == i and type(row["success"]) is bool for i, row in enumerate(rows)), "Step ledger differs"
        )
        complete = worker["complete"] is True and result["complete"] is True
        stable = len(rows) == 600 and all(row["success"] for row in rows[-40:])
        require(result["ever_success"] == any(row["success"] for row in rows), "Ever success differs")
        require(result["stable_success"] == stable, "Stable predicate differs")
        errors = np.asarray([row["tcp_error"] for row in rows])
        require(errors.shape == (len(rows), 6) and np.isfinite(errors).all(), "Invalid tracking errors")
        position = float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1))))
        rotation = float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1))))
        np.testing.assert_allclose(position, result["tcp_position_rmse_m"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(rotation, result["rotation_rmse_rad"], rtol=1e-12, atol=1e-12)
        if complete:
            require(len(rows) == 600 and result["chunks"] == 75, "Complete case budget differs")
            if case["variant"] == "terminal058":
                completed_terminal += 1
                stable_terminal += int(stable)
        records.append(
            dict(
                case=case,
                complete=complete,
                steps=len(rows),
                chunks=result["chunks"],
                stable=stable,
                ever=result["ever_success"],
                position_rmse_m=position,
                rotation_rmse_rad=rotation,
                error=result["error"],
                worker_ref=worker_ref,
                result_ref=result_ref,
                rows_ref=rows_ref,
            )
        )
    maximum = stable_terminal + 6 - completed_terminal
    value = dict(
        complete=True,
        passed=True,
        scope="saved outcome ledger and tracking arithmetic only",
        prepared_ref=prepared_ref,
        source_ref=reference(Path(__file__)),
        cases=records,
        completed_terminal=completed_terminal,
        stable_terminal=stable_terminal,
        maximum_possible_terminal_stable=maximum,
        required_terminal_stable=5,
        development_gate_unreachable=maximum < 5,
        mechanical_independent_verification_complete=False,
        development_qualified=False,
        fresh_authorized=False,
        new_model_calls=0,
        new_physics_steps=0,
    )
    print(json.dumps(publish(REPO / "outputs/serialized_preview_outcome_audit_20261027/verified.json", value)))


if __name__ == "__main__":
    main()
