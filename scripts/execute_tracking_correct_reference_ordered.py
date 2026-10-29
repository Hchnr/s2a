"""Own all eighteen ordered-reference candidate runs and native action replays."""

import json
from pathlib import Path

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.durable_stage_supervisor import cpu_environment, owned_stage


def main():
    root = REPO / "outputs/tracking_correct_reference_ordered_execution_20261027"
    require(not root.exists(), "No overwrite or automatic retry")
    prepared_ref = reference(REPO / "outputs/tracking_correct_reference_20261027/prepared.json")
    prepared = read(prepared_ref)
    port_owner_ref = reference(REPO / "outputs/tracking_correct_reference_ordered_port06_v2_20261027/result.json")
    port = read(port_owner_ref)
    require(port["complete"] is port["passed"] is True, "Complete port parity owner")
    require(
        read(port["parity_ref"])["complete"] is read(port["parity_ref"])["passed"] is True,
        "Complete case06 port parity",
    )
    sources = dict(read(port["registration_ref"])["source_sha256"])
    for path in (
        Path(__file__),
        REPO / "scripts/run_tracking_correct_reference_ordered.py",
        REPO / "scripts/check_tracking_correct_reference_ordered.py",
    ):
        ref = reference(path)
        sources[ref["path"]] = ref["sha256"]
    for path, digest in sources.items():
        checked(dict(path=path, sha256=digest))
    order = [6] + [i for i in range(18) if i != 6]
    budget = sum(p["profile"]["physics_steps"] for p in prepared["profiles"])
    require(budget == 4907, "Unchanged original18 budget")
    registration_ref = publish(
        root / "registration.json",
        dict(
            source_sha256=sources,
            prepared_ref=prepared_ref,
            port_owner_ref=port_owner_ref,
            cases=order,
            candidate_physics_budget=budget,
            replay_physics_budget=budget,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
            required_qualified_cases=18,
            thresholds=dict(stable_steps=40, position_rmse_m=0.01, so3_rmse_rad=0.05),
            scope="Original18 execution/performance checks; full input/image audits required separately",
        ),
    )
    summary = dict(complete=False, passed=False, registration_ref=registration_ref, cases=[])
    try:
        for index in order:
            stages = []
            for label, script, extra in [
                ("candidate", "run_tracking_correct_reference_ordered.py", ["--policy", "candidate"]),
                ("native_replay", "check_tracking_correct_reference_ordered.py", []),
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
                REPO / f"outputs/tracking_correct_reference_ordered_check_20261027/case{index:02d}/verified.json"
            )
            verified = read(verified_ref)
            require(verified["complete"] is verified["passed"] is True, "Independent native action replay")
            item = dict(
                case_index=index,
                stages=stages,
                verified_ref=verified_ref,
                performance_qualified=verified["performance_qualified"],
            )
            summary["cases"].append(item)
            publish(root / f"case{index:02d}/closed.json", item)
            print(
                json.dumps(dict(completed_case=index, performance_qualified=verified["performance_qualified"])),
                flush=True,
            )
        count = sum(c["performance_qualified"] for c in summary["cases"])
        summary.update(
            complete=True,
            passed=True,
            qualified_cases=count,
            performance_qualified=count == 18,
            candidate_physics_steps=budget,
            replay_physics_steps=budget,
            full_input_image_audit_pending=True,
        )
    except BaseException as error:
        summary["error"] = repr(error)
        raise
    finally:
        publish(root / "result.json", summary)


if __name__ == "__main__":
    main()
