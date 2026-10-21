"""Read-only012 guidance3 state audit with the pre-registered six-metric gate."""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from audit_wam_earlier_screen import audit_metrics
from run_wam_short_replan import cpu_environment, process_record, write_new
from summarize_wam_earlier_screen import verify_earlier
from summarize_wam_stability_screen import Evidence, aggregate, verify_cell

from s2a.artifacts import REPO, sha256

ROOT = REPO / "outputs/wam_stability_20261021"
LABEL = "8400_g30"
LAUNCH = ROOT / "guidance3_coordinator/launch.json"
LAUNCH_SHA = "7e97169765ce2e910fdacfdf07acbe945a6c33203e322f9b1ee0c8e725c2ab2b"
PREPARED = ROOT / "guidance3_preflight/prepared_cpu_review.json"
PREPARED_SHA = "2fab8ce9df690bbb6c95c0e1725caf4d3c59d5005523e5558ee9b7115f44fb5f"
WITNESS = ROOT / "guidance3_preflight/owned_process_identity_snapshot.json"
WITNESS_SHA = "522cdf05dff2d0ec484116d046ee03c414691b6ff6e98299e285176b35b62e96"
HELPERS = {
    "scripts/summarize_wam_stability_screen.py": "24dc6178dbd16c67a746b12ce86d2de6978b4b8f6ecc76226968affc31785ec8",
    "scripts/summarize_wam_earlier_screen.py": "7faa848e20aa8f20e3a8a2b6630a4f6ca48eee6bfb1463620fb44d16b91c6c25",
    "scripts/audit_wam_earlier_screen.py": "2c3727f74c54b03a3b089cbbe501fca7dce77dd899a9c23fc3e8f608e6c251c7",
    "scripts/run_wam_short_replan.py": "b78d21e9f9826c99dc2d65288e7975d8cefb0c7d31e3f3abbeab2b2ef8b65524",
}


def fingerprint_stats(evidence):
    path = REPO / "outputs/wam_independent_20261020/checkpoint_fingerprints.json"
    entry = next(x for x in evidence.json(path)["checkpoints"] if x["name"] == "8400_masked")
    assert len(entry["files"]) == 163
    for name, frozen in entry["files"].items():
        stat = Path(name).stat()
        assert (stat.st_size, stat.st_mtime_ns) == (frozen["bytes"], frozen["mtime_ns"]), name
    return entry


def resource_snapshot(launch, evidence):
    evidence.check(WITNESS, WITNESS_SHA)
    witness = evidence.json(WITNESS)
    for path, digest in witness["source_sha256"].items():
        evidence.check(path, digest)
    owned = witness["processes"]
    assert owned["coordinator"]["pid"] == launch["coordinator_pid"]
    assert owned["coordinator"]["start_ticks"] == int(launch["coordinator_start_ticks"])
    processes = {
        key: process_record(value["pid"], "7" if key == "model" else "", start_ticks=value["start_ticks"])
        for key, value in owned.items()
    }
    directory = ROOT / "screening/8400_g30/orchestration"
    error = evidence.json(directory / "error.json") if (directory / "error.json").exists() else None
    path = directory / "release.json"
    release = evidence.json(path) if path.exists() else None
    released = False
    if release is not None:
        assert release["model_pid"] == owned["model"]["pid"]
        assert release["job_pid"] == owned["state_worker"]["pid"]
        assert release["job_returncode"] == 0 and release["model_returncode"] in (0, -15)
        released = not any(p["present"] for p in processes.values()) and error is None
    return dict(
        processes=processes,
        recorded_execution_error=error,
        release=release,
        owned_service_released_normally=released,
        identity_witness_sha256=WITNESS_SHA,
    )


def admission(candidate, baseline):
    checks = []
    for horizon in (8, 16):
        for metric in ("position_rmse_m", "rotation_rmse_rad", "opening_mae"):
            actual = candidate["all"][str(horizon)]["wam"][metric]
            reference = baseline["all"][str(horizon)]["wam"][metric]
            checks.append(
                dict(
                    horizon=horizon,
                    metric=metric,
                    candidate=actual,
                    cached_guidance1=reference,
                    absolute_tolerance=1e-10,
                    nonregression=actual <= reference + 1e-10,
                )
            )
    improvements = {
        str(h): 1
        - candidate["all"][str(h)]["wam"]["position_rmse_m"] / baseline["all"][str(h)]["wam"]["position_rmse_m"]
        for h in (8, 16)
    }
    all_six = all(x["nonregression"] for x in checks)
    ten_percent = any(
        candidate["all"][str(h)]["wam"]["position_rmse_m"] <= 0.9 * baseline["all"][str(h)]["wam"]["position_rmse_m"]
        for h in (8, 16)
    )
    return dict(
        checks=checks,
        all_six_nonregression=all_six,
        position_fractional_improvements=improvements,
        at_least_one_position_improves_ten_percent=ten_percent,
        passed=all_six and ten_percent,
        physical_trial_authorized=False,
        requires_separate_future_multitask_registration=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cpu_environment()
    assert not args.output.exists() and args.output.resolve().is_relative_to(REPO / "outputs")
    evidence = Evidence()
    evidence.read(__file__)
    for path, digest in HELPERS.items():
        evidence.check(path, digest)
    evidence.check(LAUNCH, LAUNCH_SHA)
    launch = evidence.json(LAUNCH)
    assert launch["budget_git_commit"] == "17844f2"
    assert launch["registered_model_requests"] == 48 and launch["new_physics_steps"] == 0
    assert (launch["physical_gpu"], launch["port"]) == (7, 9577)
    assert not launch["actual_noise_bitexact_pairing_claimed"]
    subprocess.run(["git", "merge-base", "--is-ancestor", "17844f2", launch["launch_git_commit"]], check=True)
    evidence.check(PREPARED, PREPARED_SHA)
    assert launch["preflight_sha256"] == PREPARED_SHA
    prepared = evidence.json(PREPARED)
    for path, digest in prepared["source_before_after_sha256"].items():
        evidence.check(path, digest)
    weights = fingerprint_stats(evidence)
    config, selected, cases, progress = verify_cell(LABEL, ROOT, evidence)
    assert (config["guidance"], config["sampling_steps"], config["shift"]) == (3, 35, 5)
    assert (
        Path(weights["checkpoint"]).resolve()
        == (
            Path(config["checkpoints"][0]["run"]) / "checkpoints" / config["checkpoints"][0]["iteration"] / "model"
        ).resolve()
    )
    evidence.check(REPO / "configs/wam_stability_screen_8400_g30_20261021.json", launch["config_sha256"])
    evidence.check(ROOT / "screening/8400_g30/samples/manifest.json", launch["samples_sha256"])
    resources = resource_snapshot(launch, evidence)
    complete = progress["complete"] and resources["owned_service_released_normally"]
    metrics = independent = comparisons = decision = per_episode = None
    if complete:
        assert len(cases) == 48
        metrics = aggregate(list(cases.values()))
        independent = audit_metrics(
            config, evidence.json(ROOT / "screening/8400_g30/samples/manifest.json"), dict(metrics=metrics)
        )
        old_config, old_selected, old_cases, old_progress = verify_earlier("8400_g10", ROOT, evidence)
        assert (
            old_progress["complete"]
            and len(old_cases) == 48
            and old_selected == selected
            and old_config["guidance"] == 1
        )
        baseline = aggregate(list(old_cases.values()))
        independent_old = audit_metrics(
            old_config, evidence.json(Path(old_config["output_dir"]) / "samples/manifest.json"), dict(metrics=baseline)
        )
        old_audit_path = ROOT / "earlier_screening/independent_final_audit.json"
        evidence.check(old_audit_path, "dfbd07b7376053a56fbfc79b8eecc5b1feb52e6ca1953bbe880c9601d1da24e8")
        old_audit = evidence.json(old_audit_path)
        summary_path = ROOT / "earlier_screening/summary_final.json"
        evidence.check(summary_path, old_audit["summary_sha256"])
        old_summary = evidence.json(summary_path)
        assert baseline == old_summary["groups"]["8400_g10"]["metrics"]
        comparisons = dict(
            cached_guidance1=baseline, new_guidance3=metrics, independent_cached_guidance1=independent_old
        )
        per_episode = {
            ep: {
                label: aggregate([c for c in cell.values() if c["sample"]["episode"] == ep])["all"]
                for label, cell in (("guidance1", old_cases), ("guidance3", cases))
            }
            for ep in sorted({s["episode"] for s in selected})
        }
        decision = admission(metrics, baseline)
        assert [metrics["all"][str(h)]["wam"]["valid_endpoints"] for h in (8, 16, 40)] == [384, 768, 1778]
    evidence.recheck()
    fingerprint_stats(evidence)
    result = dict(
        schema_version="s2a.guidance-state-readonly-audit.v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        complete=complete,
        state_matrix_complete=progress["complete"],
        verified_queries=len(cases),
        registered_queries=48,
        new_model_requests=0,
        new_physics_steps=0,
        episodes=6,
        distinct_anchors=22,
        progress=progress,
        resources=resources,
        independent_gram_schmidt_reconstruction=independent,
        comparison=comparisons,
        per_episode=per_episode,
        frozen_admission=decision,
        fingerprint_files_stat_rechecked=163,
        full_weight_bytes_rehashed_now=False,
        actual_noise_witness_available=False,
        actual_noise_bitexact_pairing_claimed=False,
        input_sha256=evidence.hashes,
        warnings=[
            "Only complete48 requests plus normal owned release support the frozen six-metric admission decision.",
            "Original24 development phases/6 episodes/22 anchors; seeds and repeated phases are not independent episodes.",
            "Same request seeds do not certify actual initial-noise pairing;011 is a separate experiment.",
            "State admission does not establish task success or authorize physics; no fresh or pretraining-unseen claim.",
        ],
    )
    write_new(args.output, result)
    print(
        json.dumps(
            dict(
                path=str(args.output),
                sha256=sha256(args.output),
                complete=complete,
                verified_queries=len(cases),
                admission=decision["passed"] if decision is not None else None,
            )
        )
    )


if __name__ == "__main__":
    main()
