"""Search fixed-pose symmetric-closing box witnesses on two saved endpoints."""

import hashlib
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.infrastructure.publication import publish_result
from s2a.research.native_runtime.bilateral_box_feasibility import pair_constraints, refine_euclidean, solve_infinity
from s2a.research.native_runtime.box_separation import box_separation
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import audit_reset_bowl_fixed_h40 as physical
from scripts.simulation.runtime import diagnose_reset_bowl_saved_geometry_v2 as geometry_recovery
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/gripper/bowl_static_close_feasibility_20261112.json"
PLAN = ROOT / "plans/v0.0.1/tracking/gripper/bowl_static_close_feasibility_20261112.md"
OUTPUT = ROOT / "outputs/bowl_static_close_feasibility_20261112/result.json"
QA = ROOT / "outputs/qa/bowl_static_close_feasibility_20261112/result.json"
GEOMETRY = {
    "path": str(ROOT / "outputs/bowl_pad_alignment_recovery_audit_20261112/result.json"),
    "sha256": "6fb6ad52ce9a81f0e241215578f151b368235813db480e4a892819c8a9175447",
}
FIXED_PHYSICAL = {
    "path": str(ROOT / "outputs/reset_bowl_fixed_h40_audit_20261112/result.json"),
    "sha256": "df5e3d2a7ff15fb432f22aad566e4b35c5ea092e1568b3552c05f52484442e8b",
}
LEAD_PHYSICAL = {
    "path": str(ROOT / "outputs/bowl_h40_opening_lead_audit_20261112/result.json"),
    "sha256": "005cb8a5e91c8625f2bb7befee71cd9978d0d1fb1276b75629769ec4d67b8577",
}
FIXED = dict(
    schema="s2a.bowl-static-close-feasibility.v1",
    output=str(OUTPUT.relative_to(ROOT)),
    seed=42,
    branches=["fixed", "lead"],
    endpoint_step=48,
    position_gate_m=0.01,
    strict_overlap_margin_m=1e-5,
    feasibility_tolerance_m=1e-8,
    max_lp_per_branch=1600,
    max_refinements_per_branch=1600,
    geometry_audit_sha256=GEOMETRY["sha256"],
    fixed_physical_audit_sha256=FIXED_PHYSICAL["sha256"],
    lead_physical_audit_sha256=LEAD_PHYSICAL["sha256"],
    new_models=0,
    new_forwards=0,
    new_physics_steps=0,
    new_contexts=0,
    new_model_requests=0,
    fresh_cases=0,
    automatic_retry=False,
)


def verify_joint_contract(xml):
    """Validate the two slide limits and their direction in the site parent frame."""
    root = ElementTree.fromstring(xml)
    hand = root.find(".//body[@name='gripper0_right_gripper']")
    prefix.require(hand is not None, "The registered gripper parent must exist")
    eef = hand.find("./body[@name='gripper0_eef']")
    prefix.require(
        eef is not None and eef.find("./site[@name='gripper0_grip_site']") is not None and eef.get("quat") is None,
        "The controller site must share the gripper parent rotation",
    )
    for body_name, joint_name, limits in (
        ("gripper0_leftfinger", "gripper0_finger_joint1", [0, 0.04]),
        ("gripper0_rightfinger", "gripper0_finger_joint2", [-0.04, 0]),
    ):
        body = hand.find(f"./body[@name='{body_name}']")
        prefix.require(body is not None, "Each registered finger body must exist")
        joint = body.find(f"./joint[@name='{joint_name}']")
        prefix.require(joint is not None and joint.get("type") == "slide", "Each registered finger must slide")
        np.testing.assert_allclose(np.fromstring(joint.get("range", ""), sep=" "), limits, atol=1e-12, rtol=0)
        axis = np.fromstring(joint.get("axis", ""), sep=" ")
        quat = np.fromstring(body.get("quat", ""), sep=" ")
        prefix.require(axis.shape == (3,) and quat.shape == (4,), "Finger slide axes and orientations are required")
        axis_in_site = Rotation.from_quat(quat[[1, 2, 3, 0]]).apply(axis)
        np.testing.assert_allclose(axis_in_site, [-1, 0, 0], atol=1e-10, rtol=0)


def solve_branch(point, period, config, *, progress=None):
    """Enumerate all target-box combinations and verify each potential witness."""
    a = point["pad_alignment"]
    boxes = a["box_projections"]
    prefix.require(
        len(boxes) == 42 and point["state_sha256"] == period["endpoint_receipt"]["sim_state_sha256"],
        "The saved geometry must match the physical endpoint",
    )
    site_axis = np.asarray(a["site_rotation_world"]).T @ np.asarray(a["jaw_axis_world"])
    np.testing.assert_allclose(site_axis, [1, 0, 0], atol=1e-9, rtol=0)
    q = np.asarray(period["completed_feedback"]["score"]["qpos"], dtype=float)
    e = np.asarray(period["tcp_error"][:3], dtype=float)
    prefix.require(
        q.shape == (2,) and np.isfinite(q).all() and 0 <= q[0] <= 0.04 and -0.04 <= q[1] <= 0,
        "The measured finger joints must be inside their native ranges",
    )
    maximum_close = float(min(q[0], -q[1]))
    axis, margin = np.asarray(a["jaw_axis_world"]), config["strict_overlap_margin_m"]
    left = [pair_constraints(boxes[0], target, axis, 1, margin) for target in boxes[2:]]
    right = [pair_constraints(boxes[1], target, axis, -1, margin) for target in boxes[2:]]
    records, witnesses = [], []
    refinements = 0
    if progress is not None:
        progress.update(records=records, witnesses=witnesses, lp_attempts=0, refinements=0)
    for i, lc in enumerate(left):
        for j, rc in enumerate(right):
            prefix.require(len(records) < config["max_lp_per_branch"], "The registered LP budget is exhausted")
            if progress is not None:
                progress["lp_attempts"] += 1
            lp = solve_infinity(lc, rc, e, maximum_close)
            row = dict(left_target=boxes[i + 2]["name"], right_target=boxes[j + 2]["name"], lp=lp, refinement=None)
            records.append(row)
            if (
                lp["success"]
                and lp["infinity_error_m"] <= config["position_gate_m"] + config["feasibility_tolerance_m"]
            ):
                prefix.require(refinements < config["max_refinements_per_branch"], "The refinement budget is exhausted")
                refinements += 1
                if progress is not None:
                    progress["refinements"] = refinements
                refined = refine_euclidean(lc, rc, e, maximum_close, lp)
                gaps = []
                for pad, target, sign in ((boxes[0], boxes[i + 2], 1), (boxes[1], boxes[j + 2], -1)):
                    center = (
                        np.asarray(pad["center_world_m"])
                        + refined["translation_world_m"]
                        + sign * refined["closing_m"] * axis
                    )
                    gap = box_separation(
                        center,
                        pad["rotation_world"],
                        pad["half_size_m"],
                        target["center_world_m"],
                        target["rotation_world"],
                        target["half_size_m"],
                    )
                    gaps.append(gap["maximum_gap_m"])
                refined["independent_sat_maximum_gaps_m"] = gaps
                refined["verified_witness"] = bool(
                    refined["euclidean_error_m"] <= config["position_gate_m"] + config["feasibility_tolerance_m"]
                    and refined["maximum_constraint_violation_m"] <= config["feasibility_tolerance_m"]
                    and all(gap <= -margin + config["feasibility_tolerance_m"] for gap in gaps)
                    and 0 <= refined["closing_m"] <= maximum_close
                )
                row["refinement"] = refined
                if refined["verified_witness"]:
                    witnesses.append(dict(left_target=row["left_target"], right_target=row["right_target"], **refined))
    statuses = {
        str(code): sum(row["lp"]["status"] == code for row in records)
        for code in sorted({r["lp"]["status"] for r in records})
    }
    feasible = [row["lp"]["infinity_error_m"] for row in records if row["lp"]["success"]]
    return dict(
        joint_qpos=q,
        maximum_symmetric_closing_m=maximum_close,
        original_reference_error_m=e,
        lp_attempts=len(records),
        refinements=refinements,
        lp_statuses=statuses,
        min_infinity_error_m=min(feasible) if feasible else None,
        verified_witnesses=witnesses,
        records=records,
    )


def main():
    prefix.environment()
    config = prefix.read(prefix.reference(CONFIG))
    prefix.require(config == FIXED, "The exact pre-registered fixed-pose search is required")
    qa = prefix.read(prefix.reference(QA))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and not qa["collection"]["collection_errors"],
        "A complete 32-worker regression is required",
    )
    before = snapshot(ROOT)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "The QA source snapshot differs")
    ga, fa, la = (prefix.read(ref) for ref in (GEOMETRY, FIXED_PHYSICAL, LEAD_PHYSICAL))
    prefix.require(
        all(x["complete"] is x["passed"] is True and x["control_qualified"] is False for x in (ga, fa, la))
        and ga["audited_points"] == 97
        and ga["audited_box_projections"] == 4074,
        "All three previously audited sources must be complete",
    )
    worker = prefix.read(ga["worker_ref"])
    branches = {name: prefix.read(audit["producer_branch_ref"]) for name, audit in (("fixed", fa), ("lead", la))}
    inputs = geometry_recovery.bind_recovery().load_inputs()
    verify_joint_contract(inputs["case"]["record_xml"])
    output = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        geometry_audit_ref=GEOMETRY,
        fixed_physical_ref=FIXED_PHYSICAL,
        lead_physical_ref=LEAD_PHYSICAL,
        config_ref=prefix.reference(CONFIG),
        plan_ref=prefix.reference(PLAN),
        qa_ref=prefix.reference(QA),
        xml_sha256=hashlib.sha256(inputs["case"]["record_xml"].encode()).hexdigest(),
        source_refs=[
            prefix.reference(path)
            for path in (
                Path(__file__).resolve(),
                ROOT / "src/s2a/research/native_runtime/bilateral_box_feasibility.py",
                ROOT / "tests/research/native_runtime/test_bilateral_box_feasibility.py",
                ROOT / "tests/research/native_runtime/test_analyze_bowl_static_close_feasibility.py",
            )
        ],
        branches={},
        new_models=0,
        new_forwards=0,
        new_physics_steps=0,
        new_contexts=0,
        new_model_requests=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        for name in config["branches"]:
            first, rest = physical.flatten_fixed(branches[name])
            periods = first + rest
            prefix.require(len(periods) == 48 and periods[-1]["step"] == 47, "The full saved trajectory is required")
            progress = {}
            output["branches"][name] = progress
            output["branches"][name] = solve_branch(
                worker["geometry"]["points"][name][-1], periods[-1], config, progress=progress
            )
        prefix.require(snapshot(ROOT)["sources"] == before, "Sources changed during the static analysis")
        for ref in (GEOMETRY, FIXED_PHYSICAL, LEAD_PHYSICAL):
            prefix.checked(ref)
        output.update(complete=True, passed=True)
    except BaseException as error:
        output["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        print(publish_result(OUTPUT, output), flush=True)


if __name__ == "__main__":
    main()
