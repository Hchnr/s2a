"""Read-only009 latest15600 state audit; existing four-cell reports stay immutable."""

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from audit_wam_earlier_screen import audit_metrics
from summarize_wam_stability_screen import Evidence, aggregate, verify_cell

from s2a.artifacts import REPO, sha256

ROOT = REPO / "outputs/wam_stability_20261021"
LABEL = "15600_g10"
REFERENCES = ("8400_g10", "8600_g10", "8800_g10", "15400_g10")


def fingerprint_stats(evidence):
    path = ROOT / "checkpoint_15600_complete_fingerprints.json"
    record = evidence.json(path)
    assert record["checkpoint_name"] == "15600_masked" and record["configured_iteration"] == "iter_000015600"
    assert record["all_stat_checks_passed"] and record["total_files"] == 99 and not record["ema_weights"]
    assert record["budget_git_commit"] == "0a248f2"
    for name, expected in record["files"].items():
        stat = Path(name).stat()
        actual = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, inode=stat.st_ino, device=stat.st_dev)
        assert actual == {key: expected[key] for key in actual}, name
    return record


def released(output, evidence):
    path = output / "orchestration/release.json"
    if not path.exists():
        return None
    value = evidence.json(path)
    assert value["job_returncode"] == 0 and value["model_returncode"] in (0, -15)
    assert not (output / "orchestration/error.json").exists()
    for key in ("model_pid", "job_pid"):
        assert not Path(f"/proc/{value[key]}").exists(), "inspect owned PID identity before final release claim"
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    if Path.cwd().resolve() != REPO or any(
        os.environ.get(k) != "" for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")
    ):
        raise ValueError("read-only audit requires repository cwd and empty CUDA/LD paths")
    evidence = Evidence()
    launch = evidence.json(ROOT / "latest15600_coordinator/launch.json")
    assert launch["budget_git_commit"] == "0a248f2" and launch["max_queries"] == 48
    assert launch["new_physics_steps"] == 0 and launch["physical_gpu"] == 6 and launch["port"] == 9166
    subprocess.run(["git", "merge-base", "--is-ancestor", "0a248f2", launch["launch_git_commit"]], check=True)
    fingerprints = ROOT / "checkpoint_15600_complete_fingerprints.json"
    evidence.check(fingerprints, launch["fingerprints_sha256"])
    fingerprint = fingerprint_stats(evidence)
    evidence.check(ROOT / "model_sampling_review/candidate15600_metadata.json", launch["metadata_sha256"])
    config_path = REPO / "configs/wam_stability_screen_15600_g10_20261021.json"
    evidence.check(config_path, launch["config_sha256"])
    config, selected, cases, progress = verify_cell(LABEL, ROOT, evidence)
    assert config["sampling_steps"] == 35 and config["guidance"] == 1 and config["shift"] == 5
    assert config["checkpoints"][0]["gpu"] == 6 and config["checkpoints"][0]["port"] == 9166
    output = Path(config["output_dir"])
    sample_path = output / "samples/manifest.json"
    evidence.check(sample_path, launch["samples_sha256"])
    release = released(output, evidence)
    complete = progress["complete"] and release is not None
    if not complete and not args.partial:
        raise RuntimeError("the fixed48-query state matrix and owned service release must finish before final audit")
    comparison, independently_recomputed = {}, None
    if complete:
        assert len(cases) == 48 and progress["complete"]
        metrics = aggregate(list(cases.values()))
        independently_recomputed = audit_metrics(config, evidence.json(sample_path), dict(metrics=metrics))
        old_summary_path = ROOT / "earlier_screening/summary_final.json"
        old_audit = evidence.json(ROOT / "earlier_screening/independent_final_audit.json")
        evidence.check(old_summary_path, old_audit["summary_sha256"])
        old = evidence.json(old_summary_path)
        assert old["complete"] and old_audit["complete"] and set(old["groups"]) == set(REFERENCES)
        for path, digest in old["input_sha256"].items():
            assert not path.endswith(".distcp"), "never rehash old weight payloads"
            evidence.check(path, digest)
        for label in REFERENCES:
            previous_config = evidence.json(REPO / f"configs/wam_stability_screen_{label}_20261021.json")
            previous_samples = evidence.json(Path(previous_config["output_dir"]) / "samples/manifest.json")
            assert previous_samples["cases"] == selected
            assert previous_config["guidance"] == 1 and previous_config["sampling_steps"] == 35
            assert previous_config["sampling_seeds"] == [42, 43]
            comparison[label] = dict(
                checkpoint=old["groups"][label]["checkpoint"],
                metrics=old["groups"][label]["metrics"],
                independent_source="existing completed four-cell independent audit with rechecked642 inputs",
            )
        comparison[LABEL] = dict(checkpoint=config["checkpoints"][0], metrics=metrics)
        assert metrics["all"]["8"]["wam"]["valid_endpoints"] == 384
        assert metrics["all"]["16"]["wam"]["valid_endpoints"] == 768
        assert metrics["all"]["40"]["wam"]["valid_endpoints"] == 1778
    evidence.recheck()
    fingerprint_stats(evidence)
    destination = ROOT / "latest15600_audit"
    destination.mkdir(exist_ok=True)
    name = (
        "complete48_verified.json"
        if complete
        else f"partial{len(cases)}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.json"
    )
    path = destination / name
    result = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        seed=42,
        complete=complete,
        state_matrix_complete=progress["complete"],
        owned_service_released=release is not None,
        progress=progress,
        verified_queries=len(cases),
        registered_queries=48,
        episodes=6,
        distinct_anchors=len({(s["episode"], s["anchor_frame"]) for s in selected}),
        new_model_requests=0,
        new_physics_steps=0,
        budget_git_commit="0a248f2",
        fingerprints_sha256=sha256(fingerprints),
        full_weight_bytes_not_rehashed=True,
        fingerprint_files_stat_rechecked=fingerprint["total_files"],
        independent_gram_schmidt_reconstruction=independently_recomputed,
        comparison=comparison,
        service_release=release,
        input_sha256=evidence.hashes,
        audit_script_sha256=sha256(__file__),
        limitations=[
            "Partial queries are not a failure denominator or a checkpoint ranking.",
            "Six known development episodes; phase repeats and sampling seeds are not independent episodes.",
            "No S2A execution, task success, fresh holdout or pretraining-unseen claim.",
            "Latest complete checkpoint is a candidate, not a best-model declaration.",
        ],
    )
    with path.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(dict(path=str(path), sha256=sha256(path), complete=complete, verified_queries=len(cases))))


if __name__ == "__main__":
    main()
