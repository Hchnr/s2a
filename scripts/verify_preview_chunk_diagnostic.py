"""Independent paired-prefix, budget, outcome and actual-exit audit for fixed branches."""

import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require


def main():
    owner_ref = reference(REPO / "outputs/preview_chunk_diagnostic_continuation_20261027/result.json")
    owner = read(owner_ref)
    require(
        owner["complete"] is owner["passed"] is True and len(owner["stages"]) == 14, "All14 completed stages required"
    )
    prepared_ref = reference(REPO / "outputs/preview_chunk_diagnostic_preparation_20261027/prepared.json")
    prepared = read(prepared_ref)
    entries, inputs, sources = {}, [], {}
    for item in owner["stages"]:
        stage, result = read(item["stage_result_ref"]), read(item["result_ref"])
        require(
            stage["complete"] is stage["passed"] is result["complete"] is result["passed"] is True, "Incomplete stage"
        )
        exit_receipt, release = read(stage["exit_ref"]), read(stage["release_ref"])
        require(exit_receipt["actual_wait"] is True and exit_receipt["returncode"] == 0, "Actual wait0 required")
        require(release["reaped"] is True and release["returncode"] == 0, "Actual reap required")
        launch = read(stage["launch_ref"])
        child = launch["child"]
        stat = Path(f"/proc/{child['pid']}/stat")
        if stat.exists():
            fields = stat.read_text().rsplit(")", 1)[1].split()
            require(int(fields[19]) != child["start_ticks"] or fields[0] in ("Z", "X"), "Branch still live")
        sources.update(stage["source_sha256"])
        index, horizon = item["case_index"], item["horizon"]
        require((index, horizon) not in entries and horizon in (8, 40), "Duplicate/unknown branch")
        require(
            result["prefix_steps"] == prepared["cases"][index]["replay_periods"] and result["branch_steps"] == horizon,
            "Budget differs",
        )
        steps_ref = reference(Path(item["result_ref"]["path"]).parent / "steps.json")
        rows = read(steps_ref)
        require(len(rows) == horizon and result["new_model_calls"] == result["fresh_cases"] == 0, "Scope differs")
        entries[index, horizon] = (result, rows)
        inputs += [
            item["stage_result_ref"],
            item["result_ref"],
            steps_ref,
            stage["exit_ref"],
            stage["release_ref"],
            stage["launch_ref"],
        ]
    for path, digest in sources.items():
        checked(dict(path=path, sha256=digest))
    comparisons = []
    for i, case in enumerate(prepared["cases"]):
        short, a = entries[i, 8]
        long, b = entries[i, 40]
        require(
            short["branch_initial_state_sha256"] == long["branch_initial_state_sha256"], "Branch initial state differs"
        )
        np.testing.assert_array_equal(short["branch_initial_motor"], long["branch_initial_motor"])
        for left, right in zip(a, b[:8], strict=True):
            require(left == right, "Common8 rows differ")
        manifest = read(case["manifest_ref"])
        target = {8: "akita_black_bowl_1", 9: "wine_bottle_1", 6: "cream_cheese_1"}[manifest["record"]["task_id"]]
        stats = []
        for rows in (a, b):
            stats.append(
                dict(
                    steps=len(rows),
                    final_opening=rows[-1]["opening"],
                    target_grasp_steps=sum(r["objects"][target]["grasp_contact"] for r in rows),
                    target_z_change_m=rows[-1]["objects"][target]["position"][2]
                    - rows[0]["objects"][target]["position"][2],
                    successful_steps=sum(r["success"] for r in rows),
                )
            )
        comparisons.append(dict(case=case["case"], common8_exact=True, short=stats[0], long=stats[1]))
    budget = sum(result["prefix_steps"] + result["branch_steps"] for result, _ in entries.values())
    require(budget == 1216, "Original total physics budget differs")
    value = dict(
        complete=True,
        passed=True,
        owner_ref=owner_ref,
        prepared_ref=prepared_ref,
        input_refs=inputs,
        source_ref=reference(Path(__file__)),
        cases=comparisons,
        physical_steps=budget,
        scope="paired saved-chunk experiment; not600-step task acceptance or full mechanical replay",
        fresh_authorized=False,
        new_model_calls=0,
    )
    print(json.dumps(publish(REPO / "outputs/preview_chunk_diagnostic_verified_20261027/verified.json", value)))


if __name__ == "__main__":
    main()
