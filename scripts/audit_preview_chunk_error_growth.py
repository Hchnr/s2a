"""Describe reference motion, saved tracking errors and command saturation jointly."""

import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.geometry import pose_error


def main():
    tracking_ref = reference(REPO / "outputs/preview_chunk_tracking_restored_20261027/verified.json")
    tracking = read(tracking_ref)
    require(tracking["passed"] is True, "Endpoint audit required")
    endpoints = read(tracking["endpoints_ref"])
    paired = read(tracking["verified_ref"])
    prepared = read(paired["prepared_ref"])
    profiles, summaries = [], []
    for i, case in enumerate(prepared["cases"]):
        refs = {Path(r["path"]).name: r for r in case["original_refs"]}
        chunk = read(refs["chunks.json"])[case["event"]["query_index"]]
        poses, anchor = np.asarray(chunk["effective_targets_tcp"]), np.asarray(chunk["anchor_tcp"])
        original = [json.loads(line) for line in checked(refs["steps.jsonl"]).read_text().splitlines()]
        previous = np.asarray(original[case["replay_periods"] - 1]["command"])[:6]
        controller_ref = reference(Path(refs["chunks.json"]["path"]).parent / "controller.json")
        contract = read(controller_ref)["contract"]
        lo, hi = np.asarray(contract["input_min"]), np.asarray(contract["input_max"])
        rows_ref = reference(REPO / f"outputs/preview_chunk_branches_20261027/case{i:02d}_h40/steps.json")
        rows = read(rows_ref)
        restored = endpoints[2 * i + 1]
        require(restored["case_index"] == i and restored["horizon"] == 40, "Endpoint ordering")
        profile = []
        for j, (row, endpoint) in enumerate(zip(rows, restored["rows"], strict=True)):
            command = np.asarray(row["command"])[:6]
            error = np.asarray(endpoint["error"])
            delta = pose_error(poses[j], poses[j - 1] if j else anchor)
            profile.append(
                dict(
                    index=j,
                    position_error_m=float(np.linalg.norm(error[:3])),
                    rotation_error_rad=float(np.linalg.norm(error[3:])),
                    reference_linear_speed_m_s=float(np.linalg.norm(delta[:3]) / 0.05),
                    reference_angular_speed_rad_s=float(np.linalg.norm(delta[3:]) / 0.05),
                    at_lower=(abs(command - lo) < 1e-9).tolist(),
                    at_upper=(abs(command - hi) < 1e-9).tolist(),
                    at_rate=(abs(abs(command - previous) - 0.25) < 1e-9).tolist(),
                    command=command.tolist(),
                    opening_target=row["opening_target"],
                )
            )
            previous = command
        lower = np.asarray([r["at_lower"] for r in profile])
        upper = np.asarray([r["at_upper"] for r in profile])
        summaries.append(
            dict(
                case=case["case"],
                rows_ref=rows_ref,
                controller_ref=controller_ref,
                first_position_error_above_1cm=next(
                    (r["index"] for r in profile if r["position_error_m"] > 0.01), None
                ),
                lower_bound_steps_by_axis=lower.sum(axis=0).tolist(),
                upper_bound_steps_by_axis=upper.sum(axis=0).tolist(),
                max_reference_linear_speed_m_s=max(r["reference_linear_speed_m_s"] for r in profile),
                max_position_error_m=max(r["position_error_m"] for r in profile),
            )
        )
        profiles.append(dict(case_index=i, rows=profile))
    root = REPO / "outputs/preview_chunk_error_growth_20261027"
    profile_ref = publish(root / "profiles.json", profiles)
    print(
        json.dumps(
            publish(
                root / "verified.json",
                dict(
                    complete=True,
                    passed=True,
                    tracking_ref=tracking_ref,
                    profile_ref=profile_ref,
                    source_ref=reference(Path(__file__)),
                    cases=summaries,
                    physics_steps=0,
                    model_calls=0,
                    fresh_cases=0,
                    scope="saved-reference and executed-command descriptors; saturation is not proof of causal infeasibility",
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
