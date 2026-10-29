"""Own a separately registered full review, preserving original failed acceptance."""

import json
from pathlib import Path

from run_tracking_preview_wam_development_v4 import cpu_environment
from run_wam_readout import owned_command

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require


def main():
    root = REPO / "outputs/tracking_wam_prefetch_review_execution_20261028"
    require(not root.exists(), "No overwrite or automatic retry")
    prior_root = REPO / "outputs/tracking_wam_fast_review_execution_20261028"
    prior_result_ref = reference(prior_root / "result.json")
    prior_release_ref = reference(prior_root / "independent/ownership/release.json")
    prior_result, prior_release = read(prior_result_ref), read(prior_release_ref)
    require(
        prior_result["passed"] is False and prior_release["reaped"] is True,
        "Only after the previous full review actually failed and its worker was reaped",
    )
    kernel = Path("/proc") / str(prior_release["pid"]) / "stat"
    if kernel.exists():
        ticks = int(kernel.read_text().rsplit(")", 1)[1].split()[19])
        require(ticks != prior_release["start_ticks"], "Previous review worker still exists")
    prefetch_ref = reference(REPO / "outputs/terminal_points_prefetch_check_v2_20261028/verified.json")
    prefetch = read(prefetch_ref)
    require(
        prefetch["complete"] is prefetch["passed"] is True
        and len(prefetch["records"]) == 2
        and all(
            r["full_points"] == 35
            and r["consumed_files"] == 70
            and r["corrupted_sha_rejected"]
            and r["corrupted_stream_rejected"]
            for r in prefetch["records"]
        ),
        "Real raw/terminal ordered audit parity required",
    )
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
                "loaded_velocity_independent_fast.py",
                "verify_wam_terminal_points_prefetch.py",
                "check_terminal_points_prefetch_v2.py",
                "velocity_plan_compiled_audit.py",
                "velocity_scalar_audit.cpp",
                "current_coupled_stable_independent.py",
                "coupled_pose_preview_stable_independent.py",
                "tracking_preview_arm_log_replay_stable.py",
                "tracking_preview_wam_acceptance_checker_fast_review.py",
                "verify_tracking_preview_wam_development_prefetch_review.py",
            )
        ],
        REPO / "tests/fixtures/coupled_near_singular_response.json",
    ]:
        ref = reference(path)
        sources[ref["path"]] = ref["sha256"]
    for artifact in ("build.json", "libvelocity_scalar_audit.so"):
        ref = reference(REPO / "outputs/velocity_scalar_audit_native_20261028" / artifact)
        sources[ref["path"]] = ref["sha256"]
    for path, digest in sources.items():
        checked(dict(path=path, sha256=digest))
    for ref in prefetch["source_refs"]:
        checked(ref)
        sources[ref["path"]] = ref["sha256"]
    reg = publish(
        root / "registration.json",
        dict(
            source_sha256=sources,
            prepared_ref=prepared_ref,
            execution_registration_ref=registration_ref,
            development_ref=development_ref,
            prior_failed_ref=prior_failed_ref,
            prior_fast_result_ref=prior_result_ref,
            prior_fast_release_ref=prior_release_ref,
            prefetch_parity_ref=prefetch_ref,
            prefetch_workers=8,
            maximum_prefetched_requests=1,
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
            "scripts/verify_tracking_preview_wam_development_prefetch_review.py",
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
