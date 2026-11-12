"""Independently verify recovered closing witnesses against saved box geometry."""

import numpy as np

from s2a.infrastructure.publication import publish_result
from s2a.research.native_runtime.box_separation import box_separation
from scripts.simulation.runtime import audit_bowl_static_site_offset_lp as independent_lp
from scripts.simulation.runtime import audit_reset_bowl_fixed_h40 as physical
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

ROOT = prefix.REPO
SOURCE = {
    "path": str(ROOT / "outputs/bowl_static_close_feasibility_recovery_20261112/result.json"),
    "sha256": "f90afbd74260b43dafe9851ed80ff596f1e977423ddfacac3c01fca7e58aaabb",
}
OUTPUT = ROOT / "outputs/bowl_static_close_feasibility_recovery_audit_20261112/result.json"


def opening_after_close(q, closing):
    q = np.asarray(q, dtype=float)
    if (
        q.shape != (2,)
        or not np.isfinite(q).all()
        or not np.isfinite(closing)
        or not 0 <= q[0] <= 0.04
        or not -0.04 <= q[1] <= 0
        or not 0 <= closing <= min(q[0], -q[1])
    ):
        raise ValueError("Valid measured joints and symmetric closing within both native ranges are required")
    new_q = np.array([q[0] - closing, q[1] + closing])
    return dict(qpos=new_q, opening=float(np.clip((new_q[0] - new_q[1] - 0.001) / 0.079, 0, 1)))


def verify_geometry(pad, target, axis, direction, solution, margin, tolerance):
    translated = dict(
        pad,
        center_world_m=np.asarray(pad["center_world_m"])
        + np.asarray(solution["translation_world_m"])
        + direction * solution["closing_m"] * axis,
    )
    gap = box_separation(
        translated["center_world_m"],
        pad["rotation_world"],
        pad["half_size_m"],
        target["center_world_m"],
        target["rotation_world"],
        target["half_size_m"],
    )["maximum_gap_m"]
    prefix.require(gap <= -margin + tolerance, "The independent SAT must verify the specified overlap margin")
    return translated, float(gap)


def main():
    prefix.environment()
    result = prefix.read(SOURCE)
    geometry, fixed, lead = (
        prefix.read(ref)
        for ref in (result["geometry_audit_ref"], result["fixed_physical_ref"], result["lead_physical_ref"])
    )
    worker = prefix.read(geometry["worker_ref"])
    prefix.require(
        result["complete"] is result["passed"] is geometry["complete"] is geometry["passed"] is True
        and result["control_qualified"] is False
        and result["prior_lp_attempts"] == 13
        and result["cumulative_lp_attempts"] == 3213
        and result["solver_primal_tolerance"] == result["solver_dual_tolerance"] == 1e-9
        and result["new_models"] == result["new_forwards"] == result["new_physics_steps"] == 0,
        "The complete recovered search, previous cost and explicit precision are required",
    )
    summaries, sat_checks, independent_lp_checks = {}, 0, 0
    for name, physical_audit in (("fixed", fixed), ("lead", lead)):
        branch = prefix.read(physical_audit["producer_branch_ref"])
        first, rest = physical.flatten_fixed(branch)
        period = (first + rest)[-1]
        point = worker["geometry"]["points"][name][-1]
        prefix.require(
            point["state_sha256"] == period["endpoint_receipt"]["sim_state_sha256"],
            "Each analyzed point must match the physical trajectory",
        )
        boxes = point["pad_alignment"]["box_projections"]
        by_name = {box["name"]: box for box in boxes[2:]}
        axis = np.asarray(point["pad_alignment"]["jaw_axis_world"], dtype=float)
        q = np.asarray(period["completed_feedback"]["score"]["qpos"], dtype=float)
        error = np.asarray(period["tcp_error"][:3], dtype=float)
        reference_opening = float(period["completed_feedback"]["score"]["opening_reference"])
        search = result["branches"][name]
        prefix.require(
            search["lp_attempts"] == len(search["records"]) == 1600
            and sum(search["lp_statuses"].values()) == 1600
            and search["refinements"] == sum(row["refinement"] is not None for row in search["records"]),
            "Every registered pair and refinement must be accounted for",
        )
        found = []
        for row in search["records"]:
            lp = row["lp"]
            if lp["success"]:
                prefix.require(
                    lp["status"] == 0
                    and lp["maximum_constraint_violation_m"] <= 1e-8
                    and np.linalg.norm(np.asarray(lp["translation_world_m"]) - error, ord=np.inf)
                    <= lp["infinity_error_m"] + 1e-8,
                    "Every returned LP solution must obey its original error bound",
                )
                opening_after_close(q, lp["closing_m"])
                for pad, target, sign in (
                    (boxes[0], by_name[row["left_target"]], 1),
                    (boxes[1], by_name[row["right_target"]], -1),
                ):
                    verify_geometry(pad, target, axis, sign, lp, 1e-5, 1e-8)
                    sat_checks += 1
            refined = row["refinement"]
            if refined is None or not refined["verified_witness"]:
                continue
            prefix.require(
                refined["euclidean_error_m"] <= 0.01 + 1e-8 and refined["maximum_constraint_violation_m"] <= 1e-8,
                "Each reported witness must satisfy the unmodified position gate",
            )
            np.testing.assert_allclose(
                refined["euclidean_error_m"],
                np.linalg.norm(np.asarray(refined["translation_world_m"]) - error),
                atol=1e-12,
                rtol=0,
            )
            opening = opening_after_close(q, refined["closing_m"])
            slack = []
            for pad, target, sign in (
                (boxes[0], by_name[row["left_target"]], 1),
                (boxes[1], by_name[row["right_target"]], -1),
            ):
                moved, _ = verify_geometry(pad, target, axis, sign, refined, 1e-5, 1e-8)
                clearance = independent_lp.box_slack(moved, target)
                prefix.require(clearance > 1e-10, "The independent LP must find a strict interior witness")
                slack.append(clearance)
                independent_lp_checks += 1
            found.append(
                dict(
                    left_target=row["left_target"],
                    right_target=row["right_target"],
                    translation_world_m=refined["translation_world_m"],
                    closing_m=refined["closing_m"],
                    euclidean_error_m=refined["euclidean_error_m"],
                    qpos=opening["qpos"],
                    opening=opening["opening"],
                    original_opening_reference=reference_opening,
                    opening_error=opening["opening"] - reference_opening,
                    independent_box_slack_m=slack,
                )
            )
        prefix.require(len(found) == len(search["verified_witnesses"]), "All stored witnesses must be checked")
        summaries[name] = dict(
            lp_statuses=search["lp_statuses"],
            refinements=search["refinements"],
            min_infinity_error_m=search["min_infinity_error_m"],
            witness_count=len(found),
            witness_opening_min=min((row["opening"] for row in found), default=None),
            witness_opening_max=max((row["opening"] for row in found), default=None),
            original_opening_reference=reference_opening,
            witnesses=found,
        )
    value = dict(
        complete=True,
        passed=True,
        control_qualified=False,
        source_ref=SOURCE,
        summaries=summaries,
        lp_solutions_sat_checks=sat_checks,
        witness_box_lp_checks=independent_lp_checks,
        audit_models=0,
        audit_forwards=0,
        audit_physics_steps=0,
        audit_model_requests=0,
        automatic_retry=False,
    )
    print(publish_result(OUTPUT, value), flush=True)


if __name__ == "__main__":
    main()
