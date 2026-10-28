"""Own saved label and image audits for the ordered-reference case06."""

import json
from pathlib import Path

from s2a.artifacts import REPO
from s2a.clean_training_runtime import publish, read, reference, require
from s2a.durable_stage_supervisor import cpu_environment, owned_stage


def main():
    root = REPO / "outputs/tracking_ordered_labels_images06_execution_20261027"
    require(not root.exists(), "No overwrite or automatic retry")
    prepared_ref = reference(REPO / "outputs/tracking_correct_reference_20261027/prepared.json")
    prior_ref = reference(REPO / "outputs/tracking_correct_reference_ordered_execution_20261027/registration.json")
    prior = read(prior_ref)
    sources = dict(prior["source_sha256"])
    for path in (
        Path(__file__),
        REPO / "scripts/run_tracking_correct_reference_ordered.py",
        REPO / "scripts/audit_tracking_ordered_labels_images.py",
    ):
        ref = reference(path)
        sources[ref["path"]] = ref["sha256"]
    registration = publish(
        root / "registration.json",
        dict(
            source_sha256=sources,
            prepared_ref=prepared_ref,
            original_registration_ref=prior_ref,
            case_indices=[6],
            physics_budget=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
    )
    result = dict(complete=False, passed=False, registration_ref=registration, stages=[])
    try:
        for index in (6,):
            receipt = owned_stage(
                [
                    str(REPO / "outputs/libero_env311/bin/python"),
                    "scripts/audit_tracking_ordered_labels_images.py",
                    "--case-index",
                    str(index),
                ],
                environment=cpu_environment(),
                directory=root / f"case{index:02d}",
                timeout_s=600,
                source_sha256=sources,
            )
            result["stages"].append(dict(case_index=index, stage_ref=receipt))
            print(json.dumps(dict(completed_case=index)), flush=True)
        result.update(
            complete=True,
            passed=True,
            verification_ref=reference(REPO / "outputs/tracking_ordered_labels_images_20261027/case06/verified.json"),
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish(root / "result.json", result)


if __name__ == "__main__":
    main()
