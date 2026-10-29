"""Independently recompute returned long-plan costs and interface constraints."""

import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.generated_reference_twist import reference_world_twist
from s2a.geometry import pose_error


def main():
    producer_ref = reference(REPO / "outputs/long_joint_preview_probe_20261027/verified.json")
    producer = read(producer_ref)
    require(producer["complete"] is producer["passed"] is True, "Complete producer required")
    registration = read(producer["registration_ref"])
    for path, digest in registration["source_sha256"].items():
        checked(dict(path=path, sha256=digest))
    prepared = read(registration["prepared_ref"])
    summaries = []
    for index, result in zip(registration["case_indices"], producer["cases"], strict=True):
        case = prepared["cases"][index]
        require(result["case"] == case["case"], "Case identity")
        refs = {Path(r["path"]).name: r for r in case["original_refs"]}
        chunk = read(refs["chunks.json"])[case["event"]["query_index"]]
        targets, anchor = np.asarray(chunk["effective_targets_tcp"]), np.asarray(chunk["anchor_tcp"])
        control = read(reference(Path(refs["chunks.json"]["path"]).parent / "controller.json"))
        tool = np.linalg.inv(control["source_from_site"]) @ np.asarray(read(case["tool_ref"])["source_from_tcp"])
        before = [json.loads(line) for line in checked(refs["steps.jsonl"]).read_text().splitlines()][
            case["replay_periods"] - 1
        ]
        previous = np.asarray(before["command"])[:6]
        rows = []
        for item in result["results"]:
            horizon = item["horizon"]
            command = np.asarray(item["planned_commands"])
            require(command.shape == (horizon, 6), "Command count")
            require(np.all(command >= np.asarray(control["contract"]["input_min"]) - 1e-12), "Lower box")
            require(np.all(command <= np.asarray(control["contract"]["input_max"]) + 1e-12), "Upper box")
            require(np.max(abs(np.diff(np.vstack((previous, command)), axis=0))) <= 0.25 + 1e-12, "Rate")
            metrics = {}
            for key, score_key in (("rollout", "objective"), ("seed_rollout", "seed_objective")):
                rollout = item[key]
                residuals, errors = [], []
                for k, (site, twist) in enumerate(zip(rollout["pose"], rollout["twist"], strict=True)):
                    site, twist = np.asarray(site), np.asarray(twist)
                    error = pose_error(targets[k], site @ tool)
                    offset = site[:3, :3] @ tool[:3, 3]
                    tcp_twist = np.r_[twist[:3] + np.cross(twist[3:], offset), twist[3:]]
                    desired = reference_world_twist(targets, k, anchor, 0.05)["twist"]
                    residuals.extend(
                        np.r_[error, desired - tcp_twist] * np.array([100] * 3 + [20] * 3 + [5] * 3 + [1] * 3)
                    )
                    errors.append(error)
                residuals, errors = np.asarray(residuals), np.asarray(errors)
                score = 0.5 * float(residuals @ residuals)
                np.testing.assert_allclose(score, item[score_key], atol=1e-9, rtol=1e-10)
                metrics[key] = dict(
                    score=score,
                    position_rmse_m=float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
                    rotation_rmse_rad=float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
                )
            require(item["objective"] <= item["seed_objective"], "Model objective improvement")
            rows.append(
                dict(horizon=horizon, metrics=metrics, status=item["optimizer_message"], seconds=item["seconds"])
            )
        summaries.append(dict(case=index, rows=rows))
    root = REPO / "outputs/long_joint_preview_verified_20261027"
    root.mkdir(exist_ok=False)
    print(
        json.dumps(
            publish(
                root / "verified.json",
                dict(
                    complete=True,
                    passed=True,
                    producer_ref=producer_ref,
                    verifier_ref=reference(Path(__file__)),
                    cases=summaries,
                    physics_steps=0,
                    scope="independent cost/command checks of stored model predictions; not native trajectory validation",
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
