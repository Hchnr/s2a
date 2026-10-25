"""First completed061 raw case observation; not full development qualification."""

import json
import os
from pathlib import Path

from verify_wam_clean_training_capture import Evidence, require
from verify_wam_terminal_development import normal_owned, outcome

from s2a.clean_training_runtime import publish, reference


def main():
    require(all(os.environ.get(k) == "" for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")), "CPU required")
    root = Path("outputs/terminal_development_20261025").resolve()
    require(not (root / "first_raw_case_observation_v2.json").exists(), "Immutable observation required")
    e = Evidence()
    worker_ref = reference(root / "development/cases/task0_init0_s42_raw/worker_result.json")
    worker = e.json(worker_ref)
    require(
        worker["passed"] is worker["complete"] is True
        and worker["case"]["variant"] == "raw"
        and worker["physics_steps"] == 600
        and worker["actual_requests"] == 75,
        "Complete raw600/75 required",
    )
    folder = root / "development/orchestration/task0_init0_s42_raw"
    receipts = {
        k: reference(folder / path)
        for k, path in (("launch", "launch.json"), ("exit", "exit.json"), ("release", "ownership/release.json"))
    }
    normal_owned(e, receipts, "")
    physical = e.json(worker["physical_result_ref"])
    directory = Path(worker["physical_result_ref"]["path"]).parent
    steps_ref = reference(directory / "steps.jsonl")
    rows = [json.loads(line) for line in e.path(steps_ref).read_text().splitlines()]
    result = outcome([r["success"] for r in rows])
    require(
        physical["complete"] is True
        and physical["error"] is None
        and physical["steps"] == 600
        and physical["chunks"] == 75
        and physical["stable_success"] is result["stable"]
        and physical["ever_success"] is result["ever"],
        "Native600/tail40 outcomes differ",
    )
    e.finish()
    publish(
        root / "first_raw_case_observation_v2.json",
        dict(
            passed=True,
            complete=True,
            partial_development=True,
            worker_ref=worker_ref,
            physical_result_ref=worker["physical_result_ref"],
            steps_ref=steps_ref,
            owned_command=receipts,
            outcome=result,
            actual_requests=75,
            physics_steps=600,
            input_sha256=e.inputs,
            source_refs=[reference(Path(__file__)), reference(Path("scripts/verify_wam_terminal_development.py"))],
            new_model_calls=0,
            new_physics_steps=0,
            fresh_cases=0,
            candidate_stability_qualified=False,
            boundary="One raw baseline only; all12/900,31500 numerical nodes,7200 physical endpoints and final independent lifecycle audit still required.",
        ),
    )


if __name__ == "__main__":
    main()
