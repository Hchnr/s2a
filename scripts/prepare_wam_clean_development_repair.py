"""Freeze051 only after real failed049/qualified048 and complete CPU preparation."""

import copy
import json
import shutil

from s2a.development_repair_admission import BUDGET, FAILURE_REF, GPU, OLD_REF, PREFIX, ROOT, SOURCES, STATE_AUDIT_REF
from s2a.durable_stage_supervisor import REPO, check_sources, publish, read, reference, require


def main():
    require(not (ROOT / "execution/proposal.json").exists(), "051 already bound")
    old = read(OLD_REF)
    check_sources(old["source_sha256"])
    check_sources(old["snapshot_sha256"])
    config = {
        k: old[k] for k in ("prepared_ref", "execution_registration_ref", "state_owner_ref", "state_verification_ref")
    }
    generation_ref = reference(ROOT / "source_generation_runtime_py311.json")
    generation = read(generation_ref)
    config.update(
        stage_scope="DEVELOPMENT_only",
        root=str(ROOT),
        budget=BUDGET,
        gpu_uuid=GPU,
        automatic_retry=False,
        old049_ref=OLD_REF,
        state_audit_ref=STATE_AUDIT_REF,
        failure_audit_ref=FAILURE_REF,
        source_generation_ref=generation_ref,
        relocation_ref=reference(ROOT / "path_relocation.json"),
        generated_to_original={r["generated_ref"]["path"]: r["original_ref"] for r in generation["generated"]},
        input_preflight_ref=reference(ROOT / "preflight_runtime_py311/all_case_inputs.json"),
        qa_ref=reference(ROOT / "qa/preparation.json"),
        owner_timeout_s=172980,
    )
    qa = read(config["qa_ref"])
    require(qa["complete"] is qa["passed"] is True, "051 QA missing")
    check_sources(qa["tested_source_sha256"])
    config_path = REPO / "configs/wam_clean_development_dependency_repair_20261024.json"
    with config_path.open("x") as stream:
        json.dump(config, stream, indent=2, allow_nan=False)
    sources = copy.deepcopy(old["source_sha256"])
    snapshots = copy.deepcopy(old["snapshot_sha256"])
    extras = [
        *(REPO / r for r in SOURCES),
        *(r["generated_ref"]["path"] for r in generation["generated"]),
        config_path,
        REPO / "plans/v0.0.1/051_clean_development_dependency_repair_20261024.md",
    ]
    for source in extras:
        from pathlib import Path

        source = Path(source)
        ref = reference(source)
        require(ref["path"] not in sources, "051 extra source already frozen")
        sources[ref["path"]] = ref["sha256"]
        snapshot = ROOT / "source_snapshot" / source.relative_to(REPO)
        require(not snapshot.exists(), "051 snapshot exists")
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, snapshot)
        snapshots[str(snapshot)] = reference(snapshot)["sha256"]
    proposal = dict(
        config,
        schema_version=PREFIX + "proposal.v1",
        complete=True,
        passed=True,
        execution_authorized=False,
        source_sha256=sources,
        snapshot_sha256=snapshots,
        config_ref=reference(config_path),
    )
    print(json.dumps(publish(ROOT / "execution/proposal.json", proposal)))


if __name__ == "__main__":
    main()
