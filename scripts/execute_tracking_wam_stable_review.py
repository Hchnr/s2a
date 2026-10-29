"""Own a separately registered full review, preserving original failed acceptance."""

import json
from pathlib import Path

from run_tracking_preview_wam_development_v4 import cpu_environment
from run_wam_readout import owned_command

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require


def main():
    root = REPO / "outputs/tracking_wam_stable_review_execution_20261028"
    require(not root.exists(), "No overwrite or automatic retry")
    production = REPO / "outputs/tracking_preview_wam_development_v4_20261027"
    prepared_ref = reference(production / "prepared.json")
    registration_ref = reference(production / "registration.json")
    development_ref = reference(production / "development/orchestration/result.json")
    prior_failed_ref = reference(production / "review/DEVELOPMENT_verified.json")
    require(read(prior_failed_ref)["passed"] is False, "Original independent failure retained")
    numerical_ref = reference(REPO / "outputs/tracking_wam_response_stable_20261028/verified.json")
    require(
        read(numerical_ref)["complete"] is read(numerical_ref)["passed"] is True,
        "All7200 matrix checks at original tolerances",
    )
    prepared = read(prepared_ref)
    sources = dict(prepared["source_sha256"])
    for path in [
        Path(__file__),
        *[
            REPO / "scripts" / name
            for name in (
                "current_coupled_stable_independent.py",
                "coupled_pose_preview_stable_independent.py",
                "tracking_preview_arm_log_replay_stable.py",
                "tracking_preview_wam_acceptance_checker_stable_review.py",
                "verify_tracking_preview_wam_development_stable_review.py",
            )
        ],
        REPO / "tests/fixtures/coupled_near_singular_response.json",
    ]:
        ref = reference(path)
        sources[ref["path"]] = ref["sha256"]
    for path, digest in sources.items():
        checked(dict(path=path, sha256=digest))
    reg = publish(
        root / "registration.json",
        dict(
            source_sha256=sources,
            prepared_ref=prepared_ref,
            execution_registration_ref=registration_ref,
            development_ref=development_ref,
            prior_failed_ref=prior_failed_ref,
            numerical_ref=numerical_ref,
            physics_budget=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
            original_tolerances_unchanged=True,
            case_count=12,
            requests=900,
            head_points=31500,
            scope="Independent numerical-method revision; no producer/source edits, changed task gates or new qualification implied",
        ),
    )
    result = dict(complete=False, passed=False, registration_ref=reg)
    try:
        output = root / "verified.json"
        command = [
            str(REPO / "outputs/libero_env311/bin/python"),
            "scripts/verify_tracking_preview_wam_development_stable_review.py",
        ]
        for label, ref in [
            ("prepared", prepared_ref),
            ("execution-registration", registration_ref),
            ("development", development_ref),
        ]:
            command.extend(["--" + label, ref["path"], "--" + label + "-sha256", ref["sha256"]])
        command.extend(["--output", str(output), "--seed", "42"])
        owned = owned_command(
            command, cpu_environment(prepared), root / "independent", timeout=prepared["resources"]["audit_timeout_s"]
        )
        verified_ref = reference(output)
        verified = read(verified_ref)
        require(verified["complete"] is verified["passed"] is True, "Complete independently revised review")
        for path, digest in sources.items():
            checked(dict(path=path, sha256=digest))
        result.update(
            complete=True, passed=True, owned_command=owned, verified_ref=verified_ref, score=verified["score"]
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish(root / "result.json", result)
    print(json.dumps(dict(complete=True, passed=True, score=result["score"])), flush=True)


if __name__ == "__main__":
    main()
