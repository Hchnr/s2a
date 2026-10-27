"""Read-only first-query baseline regression against completed051, not task scoring."""

import os
from pathlib import Path

import numpy as np
from verify_wam_clean_training_capture import Evidence, require

from s2a.clean_training_runtime import publish, reference


def main():
    require(all(os.environ.get(k) == "" for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")), "CPU required")
    root = Path("outputs/terminal_development_20261025").resolve()
    require(not (root / "raw_first_query_regression.json").exists(), "Immutable output required")
    e = Evidence()
    observation = e.json(
        reference(
            Path("outputs/clean_training_development_dependency_repair_20261024/qa/first_raw_case_observation.json")
        )
    )
    old_worker = e.json(observation["worker_ref"])
    require(old_worker["passed"] is old_worker["complete"] is True, "Completed051 baseline required")
    old = Path(observation["physical_result_ref"]["path"]).parent
    current = root / "development/cases/task0_init0_s42_raw/simulation/8400_masked_position_hold/task0_init0_s42"
    images = [reference(p / "input_0000.png") for p in (old, current)]
    for ref in images:
        e.path(ref)
    responses = [e.json(reference(p / "response_0000.json"))["outputs"][0]["content"] for p in (old, current)]
    checks = dict(
        current_png_exact=images[0]["sha256"] == images[1]["sha256"],
        action128_exact=bool(np.array_equal(responses[0]["action"], responses[1]["action"])),
        normalized_action128_exact=bool(
            np.array_equal(
                responses[0]["action128"]["normalized_action"], responses[1]["action128"]["normalized_action"]
            )
        ),
    )
    e.finish()
    publish(
        root / "raw_first_query_regression.json",
        dict(
            passed=all(checks.values()),
            complete=True,
            partial=True,
            checks=checks,
            input_sha256=e.inputs,
            source_ref=reference(Path(__file__)),
            new_model_calls=0,
            physics_steps=0,
            fresh_cases=0,
            boundary="One current image and generated first reference only; no full episode/mechanical or candidate stability qualification.",
        ),
    )


if __name__ == "__main__":
    main()
