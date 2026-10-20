"""Describe complete010 native events, execution constraints and reference geometry.

All quantities are post-step diagnostics; object/task information is never fed
to a controller. This reads an already complete12-case independent audit.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from run_wam_short_replan import cpu_environment, write_new
from scipy.spatial.transform import Rotation
from summarize_wam_stability_screen import Evidence

from s2a.artifacts import REPO, sha256

TARGETS = {8: "akita_black_bowl_1", 9: "wine_bottle_1", 6: "cream_cheese_1"}


def distribution(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return dict(count=0, minimum=None, median=None, p95=None, maximum=None)
    return dict(
        count=len(values),
        minimum=float(np.min(values)),
        median=float(np.median(values)),
        p95=float(np.quantile(values, 0.95)),
        maximum=float(np.max(values)),
    )


def describe(case, evidence):
    directory = Path(case["case"])
    manifest = evidence.json(directory / "manifest.json")
    config = manifest["experiment"]
    template = evidence.json(config["policy_template"])
    tool = np.asarray(evidence.json(template["tool_contract"])["source_from_tcp"], dtype=float)
    rows = [json.loads(line) for line in evidence.read(directory / "steps.jsonl").splitlines()]
    assert len(rows) == 600
    actual_tcp = np.asarray([r["after_source"] for r in rows]) @ tool
    reference = np.asarray([r["desired_tcp"] for r in rows])
    position = np.linalg.norm(reference[:, :3, 3] - actual_tcp[:, :3, 3], axis=1)
    rotation = Rotation.from_matrix(reference[:, :3, :3] @ np.transpose(actual_tcp[:, :3, :3], (0, 2, 1))).magnitude()
    cached = np.asarray([r["tcp_error"] for r in rows])
    np.testing.assert_allclose(position, np.linalg.norm(cached[:, :3], axis=1), atol=1e-12, rtol=0)
    np.testing.assert_allclose(rotation, np.linalg.norm(cached[:, 3:], axis=1), atol=1e-12, rtol=0)
    opening = np.abs(np.asarray([r["opening_target"] - r["opening_after"] for r in rows]))
    pos_rmse, rot_rmse = float(np.sqrt(np.mean(position**2))), float(np.sqrt(np.mean(rotation**2)))
    np.testing.assert_allclose(
        [pos_rmse, rot_rmse],
        [case["tracking"]["admitted"]["position_rmse_m"], case["tracking"]["admitted"]["so3_rmse_rad"]],
        atol=1e-12,
        rtol=0,
    )
    flags = [r["success"] for r in rows]
    assert all(type(x) is bool for x in flags)
    trailing = next((i for i, x in enumerate(reversed(flags)) if not x), len(flags))
    outcome = dict(
        ever=any(flags),
        terminal=flags[-1],
        stable=all(flags[-40:]),
        successful_steps=sum(flags),
        trailing_success_steps=trailing,
    )
    assert outcome == case["outcome"]
    intervals = []
    for step, flag in enumerate(flags):
        if flag and (step == 0 or not flags[step - 1]):
            intervals.append(dict(start_step=step, end_step_exclusive=step + 1))
        elif flag:
            intervals[-1]["end_step_exclusive"] = step + 1
    origin = rows[0]["physical_time_before_s"]
    for interval in intervals:
        interval["first_endpoint_elapsed_s"] = rows[interval["start_step"]]["physical_time_after_s"] - origin
        interval["last_endpoint_elapsed_s"] = rows[interval["end_step_exclusive"] - 1]["physical_time_after_s"] - origin
    target = TARGETS[manifest["record"]["task_id"]]
    target_position = np.asarray([r["objects"][target]["position"] for r in rows])
    desired_xy = np.linalg.norm(reference[:, :2, 3] - target_position[:, :2], axis=1)
    actual_xy = np.linalg.norm(actual_tcp[:, :2, 3] - target_position[:, :2], axis=1)
    assert config["gripper_position_hold"]["close_threshold"] == 0.6
    low_reference = np.asarray([r["opening_target"] <= 0.6 for r in rows])
    closing = np.asarray([r["gripper"]["closing_episode"] for r in rows])
    contacts = {name: sum(r["objects"][name]["grasp_contact"] for r in rows) for name in rows[0]["objects"]}
    assert contacts[target] == case["target_grasp_contact_steps"]
    subsets = {}
    for label, mask in [
        ("all", np.ones(len(rows), dtype=bool)),
        ("reference_opening_at_most_qualified_close_threshold", low_reference),
        ("adapter_closing_episode", closing),
    ]:
        subsets[label] = dict(
            desired_tcp_to_target_center_xy_m=distribution(desired_xy[mask]),
            actual_tcp_to_target_center_xy_m=distribution(actual_xy[mask]),
            tcp_tracking_position_m=distribution(position[mask]),
        )
    return dict(
        case=case["case"],
        profile=case["profile"],
        record=case["record"],
        steps=600,
        queries=case["chunks"],
        raw_checkpoint_reference=case["raw_checkpoint_evidence"],
        component_ablation=case["component_ablation"],
        outcome=outcome,
        success_intervals=intervals,
        last40=[
            dict(step=r["step"], success=r["success"], physical_endpoint_elapsed_s=r["physical_time_after_s"] - origin)
            for r in rows[-40:]
        ],
        admitted_tracking=dict(
            position_rmse_m=pos_rmse,
            so3_rmse_rad=rot_rmse,
            opening_mae=float(np.mean(opening)),
            position_errors_m=distribution(position),
            rotation_errors_rad=distribution(rotation),
            opening_errors=distribution(opening),
        ),
        saturated_steps=sum(r["saturated"] for r in rows),
        rate_limited_steps=sum(r["rate_limited"] for r in rows),
        all_recorded_dual_grasp_contact_steps=contacts,
        target_object=target,
        target_grasp_contact_steps=contacts[target],
        descriptive_target_center_geometry=subsets,
        reconstruction="Actual TCP=after_source@source_from_tcp; per-step norm and RMSE independently match audited tracking.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cpu_environment()
    assert not args.output.exists() and args.output.resolve().is_relative_to(REPO / "outputs")
    evidence = Evidence()
    evidence.check(args.audit, args.audit_sha256)
    audit = evidence.json(args.audit)
    assert audit["schema_version"] == "s2a.short-replan-readonly-audit.v1"
    assert audit["complete"] and audit["all12_cases_independently_verified"]
    assert len(audit["cases"]) == 12 and audit["verified_physics_steps"] == 7200
    for path, digest in audit["source_before_after_hashes"].items():
        evidence.check(path, digest)
    evidence.read(__file__)
    cases = [describe(case, evidence) for case in audit["cases"]]
    evidence.recheck()
    result = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        complete=True,
        source_audit_sha256=args.audit_sha256,
        cases=cases,
        verified_steps=7200,
        new_model_requests=0,
        new_physics_steps=0,
        input_sha256=evidence.hashes,
        limitations=[
            "Task flags/contact/object positions are recorded native post-step metrics; no new physical predicate evaluation.",
            "XY distance is to recorded object center, not a grasp pose, task reward or an acceptance threshold.",
            "Opening<=.6 reuses the already qualified adapter close threshold only for descriptive stratification.",
            "Execution constraints and reference offsets can coexist; this report cannot uniquely attribute failure to controller or weights.",
            "All12 cases are included; no causal intervention, new model query, physics, fresh holdout or unseen-training claim.",
        ],
    )
    write_new(args.output, result)
    print(json.dumps(dict(path=str(args.output), sha256=sha256(args.output), complete=True, cases=12)))


if __name__ == "__main__":
    main()
