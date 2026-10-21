"""Independently verify complete development-state screens without new model calls."""

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.artifacts import sha256, write_json

LABELS = ("8400_g10", "8600_g10", "8800_g10", "15400_g10")


def decode_rotation(values):
    """Independent explicit Gram-Schmidt for concatenated rotation columns."""
    first = np.asarray(values[:3], dtype=float)
    second = np.asarray(values[3:], dtype=float)
    norm = np.linalg.norm(first)
    if norm < 1e-8:
        raise ValueError("degenerate first rotation column")
    first = first / norm
    second = second - np.dot(first, second) * first
    norm = np.linalg.norm(second)
    if norm < 1e-8:
        raise ValueError("degenerate second rotation column")
    second = second / norm
    return np.column_stack((first, second, np.cross(first, second)))


def audit_metrics(config, samples, expected_group):
    """Recompute from raw state/action arrays without S2A geometry/metric imports."""
    checkpoint = config["checkpoints"][0]
    directory = Path(config["output_dir"]) / "state_queries" / checkpoint["name"]
    pooled = {str(prefix): dict(position=[], rotation=[], opening=[]) for prefix in (8, 16, 40)}
    maximum = dict(position=0.0, rotation=0.0, opening=0.0)
    for sample in samples["cases"]:
        with np.load(sample["arrays_path"]) as data:
            states = np.asarray(data["states"], dtype=float)
        frame = sample["anchor_frame"]
        count = min(40, sample["valid_future_steps"], len(states) - frame - 1)
        for seed in (42, 43):
            name = f"{sample['name']}_h40_task_only_s{seed}"
            case = directory / name
            content = json.loads((case / "response.json").read_text())["outputs"][0]["content"]
            actions = np.asarray(content["action"], dtype=float)
            result = json.loads((case / "result.json").read_text())
            position = states[frame, :3].copy()
            rotation = decode_rotation(states[frame, 3:9])
            errors = dict(position=[], rotation=[], opening=[])
            for step, action in enumerate(actions[:count], start=1):
                position = position + rotation @ action[:3]
                rotation = rotation @ decode_rotation(action[3:9])
                truth = states[frame + step]
                errors["position"].append(float(np.linalg.norm(truth[:3] - position)))
                relative = decode_rotation(truth[3:9]) @ rotation.T
                errors["rotation"].append(float(Rotation.from_matrix(relative).magnitude()))
                errors["opening"].append(float(np.clip(action[9], 0, 1) - truth[9]))
            for key, saved in (
                ("position", "position_errors_m"),
                ("rotation", "rotation_errors_rad"),
                ("opening", "opening_errors"),
            ):
                actual = np.asarray(errors[key])
                np.testing.assert_allclose(actual, result["metrics"][saved], atol=1e-12, rtol=0)
                maximum[key] = max(maximum[key], float(np.max(np.abs(actual - result["metrics"][saved]))))
                for prefix in (8, 16, 40):
                    pooled[str(prefix)][key].extend(errors[key][:prefix])
    metrics = {}
    for prefix, errors in pooled.items():
        values = dict(
            valid_endpoints=len(errors["position"]),
            position_rmse_mm=float(np.sqrt(np.mean(np.square(errors["position"]))) * 1000),
            rotation_rmse_rad=float(np.sqrt(np.mean(np.square(errors["rotation"])))),
            opening_mae=float(np.mean(np.abs(errors["opening"]))),
        )
        expected = expected_group["metrics"]["all"][prefix]["wam"]
        for key, actual in values.items():
            np.testing.assert_allclose(actual, expected[key], atol=1e-10, rtol=0, err_msg=f"{prefix}:{key}")
        metrics[prefix] = values
    return dict(queries=48, independent_metrics=metrics, maximum_per_endpoint_difference=maximum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("independent audit requires empty CUDA_VISIBLE_DEVICES and LD_LIBRARY_PATH")
    if args.output.exists() or not args.output.resolve().is_relative_to(Path("outputs").resolve()):
        raise ValueError("audit output must be a new artifact in repository outputs")
    summary_sha = sha256(args.summary)
    summary = json.loads(args.summary.read_text())
    if not summary["complete"] or set(summary["groups"]) != set(LABELS):
        raise ValueError("all four48-query cells must be complete before independent final audit")
    for filename, digest in summary["input_sha256"].items():
        if sha256(filename) != digest:
            raise ValueError(f"summary input SHA changed: {filename}")
    root = Path("outputs/wam_stability_20261021/earlier_screening")
    preflight = json.loads((root / "preflight.json").read_text())
    launch = json.loads((root / "coordinator_launch/launch.json").read_text())
    if sha256(root / "preflight.json") != launch["preflight_sha256"]:
        raise ValueError("launch preflight SHA differs")
    budget_commit, launch_head = "5b41f06", launch["budget_git_commit"]
    subprocess.run(["git", "merge-base", "--is-ancestor", budget_commit, launch_head], check=True)
    releases = {}
    for prepared in preflight["prepared"]:
        stage = prepared["checkpoint"].split("_")[0]
        directory = root / f"{stage}_g10"
        if sha256(prepared["config_path"]) != prepared["config_sha256"]:
            raise ValueError("prepared configuration SHA changed")
        if sha256(prepared["samples_path"]) != prepared["samples_sha256"]:
            raise ValueError("prepared sample manifest SHA changed")
        for filename, signature in prepared["existing_byte_fingerprints"].items():
            stat = Path(filename).stat()
            if stat.st_size != signature["bytes"] or stat.st_mtime_ns != signature["mtime_ns"]:
                raise ValueError("weight file stat differs from the reused full byte SHA")
        release_path = directory / "orchestration/release.json"
        release = json.loads(release_path.read_text())
        if release["job_returncode"] != 0 or release["model_returncode"] not in (0, -15):
            raise ValueError("owned screen/service did not release normally")
        if (directory / "orchestration/error.json").exists():
            raise ValueError("owned screen recorded an execution error")
        for key in ("model_pid", "job_pid"):
            if Path(f"/proc/{release[key]}").exists():
                raise ValueError("an owned model or state worker PID remains present; inspect before final audit")
        releases[stage] = dict(path=str(release_path), sha256=sha256(release_path), **release)
    groups = {}
    for label in LABELS:
        config = json.loads(Path(f"configs/wam_stability_screen_{label}_20261021.json").read_text())
        samples = json.loads((Path(config["output_dir"]) / "samples/manifest.json").read_text())
        groups[label] = audit_metrics(config, samples, summary["groups"][label])
    # Ensure no evidence changed during the independent reconstruction.
    if sha256(args.summary) != summary_sha:
        raise ValueError("summary changed during audit")
    for filename, digest in summary["input_sha256"].items():
        if sha256(filename) != digest:
            raise ValueError(f"evidence changed during independent audit: {filename}")
    output = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        complete=True,
        summary_path=str(args.summary),
        summary_sha256=summary_sha,
        independently_verified_input_files=len(summary["input_sha256"]),
        budget_introducing_commit=budget_commit,
        launch_snapshot_head=launch_head,
        budget_commit_is_ancestor_of_launch_head=True,
        groups=groups,
        earlier_service_releases=releases,
        weight_verification="Reuse previously completed full byte SHA; current path/size/mtime checked, no new full-weight rehash",
        model_calls=0,
        physics_steps=0,
        interpretation="Development-state comparison only; no task-success winner, new heldout or pretraining-unseen claim.",
        audit_script_sha256=sha256(__file__),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, output)
    print(json.dumps(dict(output=str(args.output), complete=True, verified_queries=192, inference_performed=False)))


if __name__ == "__main__":
    main()
