"""Own all14 fixed CPU branches with actual waits and frozen source checks."""

import json
from pathlib import Path

from s2a.artifacts import REPO
from s2a.clean_training_runtime import publish, read, reference, require
from s2a.durable_stage_supervisor import cpu_environment, owned_stage


def main():
    root = REPO / "outputs/preview_chunk_diagnostic_execution_20261027"
    require(not root.exists(), "No overwrite or automatic retry")
    prep_ref = reference(REPO / "outputs/preview_chunk_diagnostic_preparation_20261027/prepared.json")
    prepared = read(prep_ref)
    old_ref = reference(REPO / "outputs/serialized_preview_wam_development_20261026/prepared.json")
    sources = dict(read(old_ref)["source_sha256"])
    for path in [
        Path(__file__),
        REPO / "scripts/run_preview_chunk_branch.py",
        REPO / "scripts/prepare_preview_chunk_diagnostic.py",
        Path(prepared["plan_ref"]["path"]),
    ]:
        ref = reference(path)
        sources[ref["path"]] = ref["sha256"]
    registration = dict(
        prepared_ref=prep_ref,
        original_prepared_ref=old_ref,
        source_sha256=sources,
        execution_authorized=True,
        automatic_retry=False,
        fresh_authorized=False,
        branches=14,
        physics_budget=1216,
        new_model_calls=0,
    )
    registration_ref = publish(root / "registration.json", registration)
    results = []
    summary = dict(complete=False, passed=False, registration_ref=registration_ref, stages=results)
    try:
        for i in range(7):
            for horizon in (8, 40):
                command = [
                    str(REPO / "outputs/libero_env311/bin/python"),
                    "scripts/run_preview_chunk_branch.py",
                    "--case-index",
                    str(i),
                    "--horizon",
                    str(horizon),
                ]
                stage = owned_stage(
                    command,
                    environment=cpu_environment(),
                    directory=root / f"case{i:02d}_h{horizon}",
                    timeout_s=600,
                    source_sha256=sources,
                )
                result_ref = reference(
                    REPO / f"outputs/preview_chunk_branches_20261027/case{i:02d}_h{horizon}/result.json"
                )
                result = read(result_ref)
                require(result["complete"] is result["passed"] is True, "Branch failed")
                results.append(
                    dict(
                        case_index=i,
                        horizon=horizon,
                        result_ref=result_ref,
                        stage_result_ref=reference(Path(stage["result_ref_path"])),
                    )
                )
                print(json.dumps(dict(completed=len(results), case_index=i, horizon=horizon)), flush=True)
        summary.update(complete=True, passed=True)
    except BaseException as error:
        summary["error"] = repr(error)
        raise
    finally:
        publish(root / "result.json", summary)


if __name__ == "__main__":
    main()
