"""Own candidate case00 and an independent native action replay."""

import json
from pathlib import Path

from s2a.artifacts import REPO
from s2a.clean_training_runtime import publish, read, reference, require
from s2a.durable_stage_supervisor import cpu_environment, owned_stage


def main():
    root = REPO / "outputs/tracking_correct_reference_candidate00_replay_v2_20261027"
    require(not root.exists(), "No overwrite or automatic retry")
    prepared_ref = reference(REPO / "outputs/tracking_correct_reference_20261027/prepared.json")
    prior_ref = reference(REPO / "outputs/tracking_correct_reference_candidate00_20261027/registration.json")
    prior = read(prior_ref)
    sources = dict(prior["source_sha256"])
    for path in (
        Path(__file__),
        REPO / "scripts/run_tracking_correct_reference.py",
        REPO / "scripts/check_tracking_correct_reference_v2.py",
    ):
        ref = reference(path)
        sources[ref["path"]] = ref["sha256"]
    registration = publish(
        root / "registration.json",
        dict(
            source_sha256=sources,
            prepared_ref=prepared_ref,
            original_registration_ref=prior_ref,
            candidate_reused_ref=reference(
                REPO / "outputs/tracking_correct_reference_20261027/case00_candidate/result.json"
            ),
            physics_budget=164,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
    )
    result = dict(complete=False, passed=False, registration_ref=registration, stages=[])
    try:
        for name, script, extra in (("native_replay", "check_tracking_correct_reference_v2.py", []),):
            receipt = owned_stage(
                [str(REPO / "outputs/libero_env311/bin/python"), f"scripts/{script}", "--case-index", "0", *extra],
                environment=cpu_environment(),
                directory=root / name,
                timeout_s=1800,
                source_sha256=sources,
            )
            result["stages"].append(dict(name=name, stage_ref=receipt))
            print(json.dumps(dict(completed_stage=name)), flush=True)
        result.update(
            complete=True,
            passed=True,
            verification_ref=reference(
                REPO / "outputs/tracking_correct_reference_check_v2_20261027/case00/verified.json"
            ),
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish(root / "result.json", result)


if __name__ == "__main__":
    main()
