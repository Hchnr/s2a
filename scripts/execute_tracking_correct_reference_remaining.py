"""Own the remaining original seventeen correct-reference candidate evaluations."""

import json
from pathlib import Path

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.durable_stage_supervisor import cpu_environment, owned_stage


def main():
    root = REPO / "outputs/tracking_correct_reference_remaining_20261027"
    require(not root.exists(), "No overwrite or automatic retry")
    prepared_ref = reference(REPO / "outputs/tracking_correct_reference_20261027/prepared.json")
    prepared = read(prepared_ref)
    prior_ref = reference(REPO / "outputs/tracking_correct_reference_candidate00_replay_v3_20261027/result.json")
    prior = read(prior_ref)
    require(prior["complete"] is prior["passed"] is True, "Prior replay owner complete")
    first_ref = prior["verification_ref"]
    require(read(first_ref)["performance_qualified"], "First case original performance gate")
    sources = dict(read(prior["registration_ref"])["source_sha256"])
    for path in (
        Path(__file__),
        REPO / "scripts/run_tracking_correct_reference.py",
        REPO / "scripts/check_tracking_correct_reference_v3.py",
    ):
        ref = reference(path)
        sources[ref["path"]] = ref["sha256"]
    for path, digest in sources.items():
        checked(dict(path=path, sha256=digest))
    budget = sum(p["profile"]["physics_steps"] for p in prepared["profiles"][1:])
    require(budget == 4743, "Unchanged remaining cycles")
    registration_ref = publish(
        root / "registration.json",
        dict(
            source_sha256=sources,
            prepared_ref=prepared_ref,
            reused_case00_ref=first_ref,
            cases=list(range(1, 18)),
            candidate_physics_budget=budget,
            replay_physics_budget=budget,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
            thresholds=dict(stable_steps=40, position_rmse_m=0.01, so3_rmse_rad=0.05),
            required_qualified_cases=18,
        ),
    )
    summary = dict(
        complete=False, passed=False, registration_ref=registration_ref, cases=[], reused_case00_ref=first_ref
    )
    try:
        for index in range(1, 18):
            stages = []
            for label, script, extra in [
                ("candidate", "run_tracking_correct_reference.py", ["--policy", "candidate"]),
                ("native_replay", "check_tracking_correct_reference_v3.py", []),
            ]:
                stage = owned_stage(
                    [
                        str(REPO / "outputs/libero_env311/bin/python"),
                        f"scripts/{script}",
                        "--case-index",
                        str(index),
                        *extra,
                    ],
                    environment=cpu_environment(),
                    directory=root / f"case{index:02d}" / label,
                    timeout_s=3600,
                    source_sha256=sources,
                )
                stages.append(dict(label=label, stage_ref=stage))
            verified_ref = reference(
                REPO / f"outputs/tracking_correct_reference_check_v3_20261027/case{index:02d}/verified.json"
            )
            verified = read(verified_ref)
            require(verified["complete"] is verified["passed"] is True, "Independent execution check")
            item = dict(
                case_index=index,
                stages=stages,
                verified_ref=verified_ref,
                performance_qualified=verified["performance_qualified"],
            )
            summary["cases"].append(item)
            publish(root / f"case{index:02d}" / "closed.json", item)
            print(
                json.dumps(dict(completed_case=index, performance_qualified=verified["performance_qualified"])),
                flush=True,
            )
        count = int(read(first_ref)["performance_qualified"]) + sum(
            c["performance_qualified"] for c in summary["cases"]
        )
        summary.update(
            complete=True,
            passed=True,
            performance_qualified=count == 18,
            qualified_cases=count,
            candidate_total_physics_steps=4907,
            replay_total_physics_steps=4907,
            qualification_scope="native action replay and original performance gates; full causality/model/image audit pending",
        )
    except BaseException as error:
        summary["error"] = repr(error)
        raise
    finally:
        publish(root / "result.json", summary)


if __name__ == "__main__":
    main()
