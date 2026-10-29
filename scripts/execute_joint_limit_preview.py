"""Own fixed protocol and physical comparison stages with source and exit receipts."""

import json
from pathlib import Path

from s2a.artifacts import REPO
from s2a.clean_training_runtime import publish, read, reference, require
from s2a.durable_stage_supervisor import cpu_environment, owned_stage


def main():
    root = REPO / "outputs/joint_limit_preview_execution_20261027"
    require(not root.exists(), "No overwrite or automatic retry")
    old_ref = reference(REPO / "outputs/serialized_preview_wam_development_20261026/prepared.json")
    sources = dict(read(old_ref)["source_sha256"])
    for path in [
        Path(__file__),
        REPO / "scripts/run_joint_limit_preview_branch.py",
        REPO / "scripts/verify_joint_limit_preview.py",
        REPO / "plans/v0.0.1/joint_limit_preview_physical_20261027.md",
        *[
            REPO / f"src/s2a/{name}.py"
            for name in (
                "joint_limit_native",
                "joint_limit_inverse",
                "joint_limit_response",
                "nullspace_convex_qp",
                "joint_limit_preview",
                "joint_limit_preview_objective",
                "joint_limit_preview_feasible_inverse",
                "joint_limit_preview_native",
            )
        ],
    ]:
        ref = reference(path)
        sources[ref["path"]] = ref["sha256"]
    registration_ref = publish(
        root / "registration.json",
        dict(
            source_sha256=sources,
            original_prepared_ref=old_ref,
            execution_authorized=True,
            automatic_retry=False,
            fresh_authorized=False,
            physics_budget=306,
            cases=[0, 1],
            stages=[[0, 1], [1, 1], [0, 40], [1, 40]],
            model_calls=0,
        ),
    )
    summary = dict(complete=False, passed=False, registration_ref=registration_ref, stages=[])
    try:
        for case, horizon in ((0, 1), (1, 1), (0, 40), (1, 40)):
            stage = owned_stage(
                [
                    str(REPO / "outputs/libero_env311/bin/python"),
                    "scripts/run_joint_limit_preview_branch.py",
                    "--case-index",
                    str(case),
                    "--horizon",
                    str(horizon),
                ],
                environment=cpu_environment(),
                directory=root / f"case{case:02d}_h{horizon}",
                timeout_s=1800,
                source_sha256=sources,
            )
            result_ref = reference(
                REPO / f"outputs/joint_limit_preview_branches_20261027/case{case:02d}_h{horizon}/result.json"
            )
            result = read(result_ref)
            require(result["passed"] is result["complete"] is result["native_goals_verified"] is True, "Stage failed")
            summary["stages"].append(dict(case_index=case, horizon=horizon, stage_ref=stage, result_ref=result_ref))
            print(json.dumps(dict(completed=len(summary["stages"]), case=case, horizon=horizon)), flush=True)
        summary.update(complete=True, passed=True)
    except BaseException as error:
        summary["error"] = repr(error)
        raise
    finally:
        publish(root / "result.json", summary)


if __name__ == "__main__":
    main()
