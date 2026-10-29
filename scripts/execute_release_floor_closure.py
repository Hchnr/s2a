"""Own final qualification aggregation after all18 runs and all36 audits close."""

import json
from pathlib import Path

from s2a.artifacts import REPO
from s2a.clean_training_runtime import publish, read, reference, require
from s2a.durable_stage_supervisor import cpu_environment, owned_stage

AUDIT_OWNERS = ["release_floor_audits_all18_20261028"]


def main():
    root = REPO / "outputs/release_floor_closure_execution_20261028"
    require(not root.exists(), "No overwrite or automatic retry")
    owner_ref = reference(REPO / "outputs/release_floor_correct_reference_execution_20261028/result.json")
    owner = read(owner_ref)
    require(owner["complete"] is owner["passed"] is True, "Complete original18 owner")
    audits = [reference(REPO / f"outputs/{folder}/result.json") for folder in AUDIT_OWNERS]
    sources = dict(read(owner["registration_ref"])["source_sha256"])
    for ref in [
        owner_ref,
        *audits,
        reference(Path(__file__)),
        reference(REPO / "scripts/close_release_floor_matrix.py"),
    ]:
        sources[ref["path"]] = ref["sha256"]
    reg = publish(
        root / "registration.json",
        dict(
            owner_ref=owner_ref,
            audit_owner_refs=audits,
            source_sha256=sources,
            physics_budget=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
    )
    result = dict(complete=False, passed=False, registration_ref=reg)
    try:
        flags = [arg for ref in audits for arg in ["--audit-owner", ref["path"]]]
        stage = owned_stage(
            [str(REPO / ".venv/bin/python"), "scripts/close_release_floor_matrix.py", *flags],
            environment=cpu_environment(),
            directory=root / "aggregate",
            timeout_s=600,
            source_sha256=sources,
        )
        verified_ref = reference(REPO / "outputs/release_floor_matrix_closed_20261028/verified.json")
        require(read(verified_ref)["complete"] is read(verified_ref)["passed"] is True, "Final aggregation")
        result.update(complete=True, passed=True, stage_ref=stage, verified_ref=verified_ref)
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish(root / "result.json", result)
    print(json.dumps(dict(complete=True, passed=True)), flush=True)


if __name__ == "__main__":
    main()
