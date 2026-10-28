"""Own two new interior-feedback branches and bind independently checked baselines."""

import json
from pathlib import Path

from s2a.artifacts import REPO
from s2a.clean_training_runtime import publish, read, reference, require
from s2a.durable_stage_supervisor import cpu_environment, owned_stage


def main():
    root = REPO / "outputs/terminal_margin_feedback_v2_execution_20261027"
    require(not root.exists(), "No overwrite or automatic retry")
    baseline_ref = reference(REPO / "outputs/early_preventive_verified_20261027/verified.json")
    baseline = read(baseline_ref)
    require(baseline["complete"] is baseline["passed"] is True, "Checked baseline required")
    model_verified_ref = reference(REPO / "outputs/terminal_joint_plan_verified_20261027/verified.json")
    require(read(model_verified_ref)["passed"] is True, "Independent terminal-model check required")
    old_ref = reference(REPO / "outputs/serialized_preview_wam_development_20261026/prepared.json")
    sources = dict(read(old_ref)["source_sha256"])
    for path in [
        Path(__file__),
        REPO / "scripts/run_terminal_margin_feedback_v2.py",
        REPO / "plans/v0.0.1/terminal_margin_feedback_20261027.md",
        REPO / "plans/v0.0.1/terminal_margin_feedback_v2_20261027.md",
        *[
            REPO / f"src/s2a/{n}.py"
            for n in (
                "terminal_margin_native_v2",
                "terminal_margin_hybrid_v2",
                "strict_terminal_joint_plan",
                "strict_adjustable_interior_joint_plan",
                "pose_cap_check",
                "adaptive_margin_hybrid",
                "adaptive_plan_history",
                "adaptive_joint_margin",
                "adjustable_interior_joint_plan",
                "interior_joint_hybrid",
                "interior_joint_plan",
                "interior_joint_preview",
                "preventive_joint_inverse",
                "fast_long_joint_preview",
                "single_face_projection",
                "joint_limit_preview",
                "joint_limit_preview_objective",
                "joint_limit_native",
                "joint_limit_inverse",
                "joint_limit_response",
                "nullspace_convex_qp",
            )
        ],
    ]:
        ref = reference(path)
        sources[ref["path"]] = ref["sha256"]
    registration_ref = publish(
        root / "registration.json",
        dict(
            source_sha256=sources,
            baseline_verified_ref=baseline_ref,
            terminal_model_verified_ref=model_verified_ref,
            internal_pose_check_atol=1e-11,
            unchanged_outer_pose_check_atol=1e-10,
            superseded_failed_owner_ref=reference(
                REPO / "outputs/terminal_margin_feedback_execution_20261027/result.json"
            ),
            terminal_rule="non-outward terminal velocity only on free-forecast near-boundary normals in full fixed-margin stage",
            physics_budget=256,
            world_model_calls=0,
            fresh_cases=0,
            stages=[[0, "terminal"], [1, "terminal"]],
            reference_schedule=[dict(query=9, steps=8), dict(query=10, steps=8), dict(query=11, steps=40)],
            first_caps_witness="max(free seed first error, .01m/.05rad)",
            first_caps_adaptive="max(free seed first error, .01m/.05rad)",
            first_caps_fallback="max(ideal-limit seed first error, .01m/.05rad)",
            joint_margin_rad=0.05,
            hard_objective="tracking only",
            fallback_margin_weight=1000.0,
            hard_max_evaluations=160,
            adaptive_max_evaluations=160,
            tracking_refinement_max_evaluations=160,
            total_response_budget=640,
            adaptive_minimum_retained_margin_rad=1e-6,
            refinement_margin_buffer_rad=1e-9,
            soft_max_evaluations=160,
            automatic_retry=False,
        ),
    )
    summary = dict(complete=False, passed=False, registration_ref=registration_ref, stages=[])
    try:
        for case in (0, 1):
            stage = owned_stage(
                [
                    str(REPO / "outputs/libero_env311/bin/python"),
                    "scripts/run_terminal_margin_feedback_v2.py",
                    "--case-index",
                    str(case),
                    "--horizon",
                    "56",
                    "--strategy",
                    "terminal",
                ],
                environment=cpu_environment(),
                directory=root / f"case{case:02d}",
                timeout_s=1800,
                source_sha256=sources,
            )
            result_ref = reference(
                REPO / f"outputs/terminal_margin_feedback_v2_branches_20261027/case{case:02d}_terminal/result.json"
            )
            result = read(result_ref)
            require(result["complete"] is result["passed"] is result["native_goals_verified"] is True, "Branch")
            summary["stages"].append(
                dict(case_index=case, strategy="terminal", horizon=56, stage_ref=stage, result_ref=result_ref)
            )
            print(json.dumps(dict(completed_case=case)), flush=True)
        summary.update(complete=True, passed=True)
    except BaseException as error:
        summary["error"] = repr(error)
        raise
    finally:
        publish(root / "result.json", summary)


if __name__ == "__main__":
    main()
