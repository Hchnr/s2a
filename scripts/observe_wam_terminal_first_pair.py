"""Record complete first raw/058 pair, without scoring an incomplete cohort."""

import json
import os
from pathlib import Path

from verify_wam_clean_training_capture import Evidence, require
from verify_wam_terminal_development import normal_owned, outcome

from s2a.clean_training_runtime import publish, reference


def main():
    require(all(os.environ.get(k) == "" for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")), "CPU required")
    root = Path("outputs/terminal_development_20261025").resolve()
    require(not (root / "first_pair_observation.json").exists(), "Immutable observation required")
    e = Evidence()
    cases = []
    for variant in ("raw", "terminal058"):
        name = f"task0_init0_s42_{variant}"
        worker_ref = reference(root / "development/cases" / name / "worker_result.json")
        worker = e.json(worker_ref)
        require(
            worker["passed"] is worker["complete"] is True
            and worker["actual_requests"] == 75
            and worker["physics_steps"] == 600,
            "Complete600/75 case required",
        )
        directory = root / "development/orchestration" / name
        receipts = {
            key: reference(directory / path)
            for key, path in (("launch", "launch.json"), ("exit", "exit.json"), ("release", "ownership/release.json"))
        }
        normal_owned(e, receipts, "")
        physical = e.json(worker["physical_result_ref"])
        steps_ref = reference(Path(worker["physical_result_ref"]["path"]).parent / "steps.jsonl")
        rows = [json.loads(line) for line in e.path(steps_ref).read_text().splitlines()]
        result = outcome([row["success"] for row in rows])
        require(
            physical["complete"] is True
            and physical["error"] is None
            and physical["steps"] == 600
            and physical["chunks"] == 75
            and physical["stable_success"] is result["stable"]
            and physical["ever_success"] is result["ever"],
            "Native task outcome differs",
        )
        cases.append(
            dict(
                variant=variant,
                worker_ref=worker_ref,
                physical_result_ref=worker["physical_result_ref"],
                steps_ref=steps_ref,
                owned_command=receipts,
                outcome=result,
                own_generated_reference_tracking={
                    k: physical[k] for k in ("tcp_position_rmse_m", "rotation_rmse_rad", "opening_mae")
                },
            )
        )
    e.finish()
    publish(
        root / "first_pair_observation.json",
        dict(
            passed=True,
            complete=True,
            partial_development=True,
            cases=cases,
            completed_cases=2,
            fixed_cases=12,
            actual_requests=150,
            physics_steps=1200,
            input_sha256=e.inputs,
            source_ref=reference(Path(__file__)),
            new_model_calls=0,
            new_physics_steps=0,
            fresh_cases=0,
            candidate_stability_qualified=False,
            boundary="Native observed first pair only; reference trajectories differ, tracking errors are not errors to one shared task target. Full fixed cohort and independent audit remain required.",
        ),
    )


if __name__ == "__main__":
    main()
