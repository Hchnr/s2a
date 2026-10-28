"""Close all18 execution, performance, robot-input and image/label evidence."""

import argparse
import json
from pathlib import Path

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require


def check_stage(ref):
    stage = read(ref)
    require(stage["complete"] is stage["passed"] is True, "Successful owned stage")
    exit, release = read(stage["exit_ref"]), read(stage["release_ref"])
    require(
        exit["actual_wait"] is release["reaped"] is True and exit["returncode"] == release["returncode"] == 0,
        "Actual wait0/reap",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit-owner", action="append", required=True)
    args = p.parse_args()
    root = REPO / "outputs/dependency_safe_matrix_closed_20261027"
    require(not root.exists(), "No overwrite")
    owner_ref = reference(REPO / "outputs/dependency_safe_correct_reference_execution_20261027/result.json")
    owner = read(owner_ref)
    require(
        owner["complete"] is owner["passed"] is owner["performance_qualified"] is True,
        "Complete original performance gate",
    )
    require(
        len(owner["cases"]) == owner["qualified_cases"] == 18
        and owner["candidate_physics_steps"] == owner["replay_physics_steps"] == 4907,
        "Original18 budgets",
    )
    prepared_ref = read(owner["registration_ref"])["prepared_ref"]
    prepared = read(prepared_ref)
    sources = {}

    def merge(reg_ref):
        reg = read(reg_ref)
        for path, digest in reg["source_sha256"].items():
            require(path not in sources or sources[path] == digest, "Consistent immutable source")
            sources[path] = digest

    merge(owner["registration_ref"])
    audits, audit_owners = {}, []
    for value in args.audit_owner:
        ref = reference(Path(value))
        audited = read(ref)
        require(audited["complete"] is audited["passed"] is True, "Complete audit owner")
        merge(audited["registration_ref"])
        audit_owners.append(ref)
        for stage in audited["stages"]:
            require(stage.get("reused", False) is False, "Supply original audit owners, not reused summaries")
            check_stage(stage["stage_ref"])
            receipt = read(stage["stage_ref"])
            command = receipt["command"]
            allowed = {f"scripts/audit_dependency_safe_{kind}.py": kind for kind in ("robot_inputs", "labels_images")}
            require(
                len(command) == 4 and command[1] in allowed and command[2] == "--case-index", "Exact executed audit CLI"
            )
            index, kind = int(command[3]), allowed[command[1]]
            require(0 <= index < 18, "Executed original case index")
            if "case_index" in stage:
                require(stage["case_index"] == index, "Executed and declared case identity")
            if "kind" in stage:
                require(stage["kind"] == kind, "Executed and declared audit kind")
            key = (index, kind)
            require(key not in audits, "No duplicate audit coverage")
            output_ref = reference(REPO / f"outputs/dependency_safe_{kind}_20261027/case{index:02d}/verified.json")
            if "verified_ref" in stage:
                require(stage["verified_ref"] == output_ref, "Explicit audit output identity")
            audits[key] = dict(verified_ref=output_ref, stage_ref=stage["stage_ref"])
    require(
        set(audits) == {(i, k) for i in range(18) for k in ("robot_inputs", "labels_images")},
        "All36 independent audits",
    )
    cases = []
    require({c["case_index"] for c in owner["cases"]} == set(range(18)), "Exact original case roster")
    for item in sorted(owner["cases"], key=lambda c: c["case_index"]):
        index = item["case_index"]
        require(len(item["stages"]) == 2, "Candidate and replay")
        for stage in item["stages"]:
            check_stage(stage["stage_ref"])
        v = read(item["verified_ref"])
        result = read(v["result_ref"])
        require(
            v["complete"] is v["passed"] is v["performance_qualified"] is True, "Independent original performance gate"
        )
        require(
            result["physics_steps"] == v["physics_steps"] == prepared["profiles"][index]["profile"]["physics_steps"],
            "Original per-case budget",
        )
        require(
            v["outcome"]["stable"] and v["position_rmse_m"] <= 0.01 and v["so3_rmse_rad"] <= 0.05,
            "Original numerical and task thresholds",
        )
        refs = {}
        for kind in ("robot_inputs", "labels_images"):
            ref = audits[index, kind]["verified_ref"]
            a = read(ref)
            require(
                a["complete"] is a["passed"] is True and a["result_ref"] == v["result_ref"],
                "Audit binds actual candidate",
            )
            if kind == "labels_images":
                require(
                    a["replay_pixels_compared"] is True and a["replay_ref"] == item["verified_ref"],
                    "Actual independent pixel replay",
                )
            refs[kind] = ref
        cases.append(
            dict(case_index=index, verified_ref=item["verified_ref"], audit_refs=refs, physics_steps=v["physics_steps"])
        )
    for path, digest in sources.items():
        checked(dict(path=path, sha256=digest))
    own = reference(Path(__file__))
    sources[own["path"]] = own["sha256"]
    print(
        json.dumps(
            publish(
                root / "verified.json",
                dict(
                    complete=True,
                    passed=True,
                    performance_qualified=True,
                    owner_ref=owner_ref,
                    audit_owner_refs=audit_owners,
                    prepared_ref=prepared_ref,
                    source_ref=own,
                    source_sha256=sources,
                    cases=cases,
                    qualified_cases=18,
                    candidate_physics_steps=4907,
                    replay_physics_steps=4907,
                    world_model_calls=0,
                    fresh_cases=0,
                    scope="Complete original18 native execution/performance and recorded input/history/image/label checks; not optimizer optimality, WAM task qualification, real-time feasibility or fresh-initial-state success",
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
