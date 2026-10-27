"""Bind049 only to an actual completed and jointly qualified048 STATE run."""

import copy
import json
import os
import shutil

from s2a.clean_training_development_handoff import (
    BUDGET,
    OWNER_PATH,
    PREFIX,
    ROOT,
    SOURCES,
    STATE_ROOT,
    command,
    qualified_state,
)
from s2a.durable_stage_supervisor import REPO, check_cpu_environment, check_sources, publish, read, reference, require


def main():
    check_cpu_environment(os.environ)
    require(not (ROOT / "execution/proposal.json").exists(), "049 already bound")
    state_ref = reference(STATE_ROOT / "execution/admitted.json")
    terminal = qualified_state(state_ref)
    old = read(state_ref)
    policy_path = REPO / "configs/wam_clean_training_development_continuation_policy_20261024.json"
    plan_path = REPO / "plans/v0.0.1/049_clean_training_development_continuation_20261024.md"
    policy = read(reference(policy_path))
    require(
        policy["implementation_ready"] is True and policy["execution_authorized"] is False, "049 preparation incomplete"
    )
    config = {k: old[k] for k in ("prepared_ref", "execution_registration_ref")}
    prepared = read(config["prepared_ref"])
    config["case_order"] = read(config["execution_registration_ref"])["development"]["case_order"]
    config.update(
        protocol="049",
        root=str(ROOT),
        stage_scope="DEVELOPMENT_only",
        gpu_uuid=policy["gpu_uuid"],
        private_port=31339,
        budget=BUDGET,
        automatic_retry=False,
        state_registration_ref=state_ref,
        state_terminal=terminal,
        state_owner_ref=terminal["state_owner_ref"],
        state_verification_ref=terminal["verification_ref"],
        owner_ref=reference(OWNER_PATH),
        owner_timeout_s=prepared["resources"]["development_timeout_s"] + 180,
        qa_ref=reference(ROOT / "qa/preparation.json"),
    )
    qa = read(config["qa_ref"])
    require(qa["complete"] is qa["passed"] is True, "049 preparation QA missing")
    check_sources(qa["tested_source_sha256"])
    config["command"] = command(config)
    config_path = REPO / "configs/wam_clean_training_development_continuation_20261024.json"
    check_sources(old["source_sha256"])
    check_sources(old["snapshot_sha256"])
    with config_path.open("x") as stream:
        json.dump(config, stream, indent=2, allow_nan=False)
    sources = copy.deepcopy(old["source_sha256"])
    snapshots = copy.deepcopy(old["snapshot_sha256"])
    for source in [*(REPO / r for r in SOURCES), policy_path, plan_path, config_path]:
        ref = reference(source)
        require(ref["path"] not in sources, "049 source already frozen")
        sources[ref["path"]] = ref["sha256"]
        dest = ROOT / "source_snapshot" / source.relative_to(REPO)
        require(not dest.exists(), "049 snapshot exists")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        snapshots[str(dest)] = reference(dest)["sha256"]
    proposal = copy.deepcopy(config)
    proposal.update(
        schema_version=PREFIX + "proposal.v1",
        complete=True,
        passed=True,
        execution_authorized=False,
        source_sha256=sources,
        snapshot_sha256=snapshots,
        config_ref=reference(config_path),
        policy_ref=reference(policy_path),
        plan_ref=reference(plan_path),
    )
    print(json.dumps(publish(ROOT / "execution/proposal.json", proposal)))


if __name__ == "__main__":
    main()
