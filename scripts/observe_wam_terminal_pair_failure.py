"""Describe only the completed first 061 pair, without model or physics calls."""

import json
from collections import Counter
from pathlib import Path

from analyze_wam_release_horizon import analyze, reference

ROOT = Path("outputs/terminal_development_20261025")


def main():
    cases = []
    consumed = []
    for variant in ("raw", "terminal058"):
        directory = ROOT / "development/cases" / f"task0_init0_s42_{variant}"
        worker_path = directory / "worker_result.json"
        worker = json.loads(worker_path.read_text())
        if not worker["complete"] or worker["physics_steps"] != 600 or worker["actual_requests"] != 75:
            raise ValueError("Only sealed complete first-pair cases may be described")
        (chunks_path,) = directory.glob("simulation/*/*/chunks.json")
        steps_path = chunks_path.with_name("steps.jsonl")
        before = [reference(p) for p in (worker_path, chunks_path, steps_path)]
        rows = [json.loads(line) for line in steps_path.read_text().splitlines()]
        if len(rows) != 600 or any(type(row["success"]) is not bool for row in rows):
            raise ValueError("Requires all 600 native boolean outcomes")
        horizon = analyze(chunks_path)
        if len(horizon["records"]) != 75:
            raise ValueError("Requires all 75 generated chunks")
        bowl = "akita_black_bowl_1"
        plate = "plate_1"
        relative = [
            [a - b for a, b in zip(row["objects"][bowl]["position"], row["objects"][plate]["position"], strict=True)]
            for row in rows
        ]
        cases.append(
            {
                "variant": variant,
                "inputs": before,
                "success_steps": sum(row["success"] for row in rows),
                "last40_success_steps": sum(row["success"] for row in rows[-40:]),
                "bowl_max_height_gain_m": max(row["objects"][bowl]["height_gain_m"] for row in rows),
                "bowl_contact_steps": sum(row["objects"][bowl]["grasp_contact"] for row in rows),
                "bowl_minus_plate_initial_logged_m": relative[0],
                "bowl_minus_plate_terminal_m": relative[-1],
                "last40_reference_opening_range": [
                    min(row["opening_target"] for row in rows[-40:]),
                    max(row["opening_target"] for row in rows[-40:]),
                ],
                "last40_measured_opening_range": [
                    min(row["opening_after"] for row in rows[-40:]),
                    max(row["opening_after"] for row in rows[-40:]),
                ],
                "last40_gripper_reasons": dict(Counter(row["gripper"]["reason"] for row in rows[-40:])),
                "saturated_steps": sum(row["saturated"] for row in rows),
                "release_horizon": horizon,
            }
        )
        if before != [reference(p) for p in (worker_path, chunks_path, steps_path)]:
            raise RuntimeError("Consumed completed inputs changed")
        consumed.extend(before)
    result = {
        "scope": "Partial development description, not causal attribution or full qualification; object truth is offline only",
        "new_model_calls": 0,
        "new_physics_steps": 0,
        "fresh_cases": 0,
        "candidate_stability_qualified": False,
        "script_ref": reference(Path(__file__)),
        "horizon_analyzer_ref": reference(Path("scripts/analyze_wam_release_horizon.py")),
        "consumed_inputs": consumed,
        "cases": cases,
    }
    with (ROOT / "first_pair_failure_description.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    for case in cases:
        print(json.dumps({k: v for k, v in case.items() if k not in ("inputs", "release_horizon")}))


if __name__ == "__main__":
    main()
