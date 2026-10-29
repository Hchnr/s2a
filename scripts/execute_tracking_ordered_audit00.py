"""Own input and label/image audits of closed ordered-reference case00."""

import json
from pathlib import Path

from s2a.artifacts import REPO
from s2a.clean_training_runtime import publish, read, reference, require
from s2a.durable_stage_supervisor import cpu_environment, owned_stage


def main():
    root = REPO / "outputs/tracking_ordered_audit00_execution_20261027"
    require(not root.exists(), "No overwrite or automatic retry")
    closed_ref = reference(REPO / "outputs/tracking_correct_reference_ordered_execution_20261027/case00/closed.json")
    require(read(closed_ref)["performance_qualified"], "Closed original performance gate")
    prior_ref = reference(REPO / "outputs/tracking_correct_reference_ordered_execution_20261027/registration.json")
    sources = dict(read(prior_ref)["source_sha256"])
    scripts = ["audit_tracking_ordered_robot_inputs.py", "audit_tracking_ordered_labels_images.py"]
    for path in [Path(__file__), *[REPO / "scripts" / s for s in scripts]]:
        ref = reference(path)
        sources[ref["path"]] = ref["sha256"]
    reg = publish(
        root / "registration.json",
        dict(
            source_sha256=sources,
            producer_registration_ref=prior_ref,
            closed_case_ref=closed_ref,
            case_indices=[0],
            physics_budget=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
    )
    result = dict(complete=False, passed=False, registration_ref=reg, stages=[])
    try:
        for label, script in zip(["robot_inputs", "labels_images"], scripts, strict=True):
            ref = owned_stage(
                [str(REPO / "outputs/libero_env311/bin/python"), f"scripts/{script}", "--case-index", "0"],
                environment=cpu_environment(),
                directory=root / label,
                timeout_s=600,
                source_sha256=sources,
            )
            verified = reference(REPO / f"outputs/tracking_ordered_{label}_20261027/case00/verified.json")
            require(read(verified)["complete"] is read(verified)["passed"] is True, "Complete audit")
            result["stages"].append(dict(label=label, stage_ref=ref, verified_ref=verified))
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish(root / "result.json", result)
    print(json.dumps(dict(complete=True, passed=True)), flush=True)


if __name__ == "__main__":
    main()
