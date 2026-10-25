"""Prepare048 source bridge or bind real completed047 refs; never launch models."""

import argparse
import copy
import json
import shutil

from s2a.clean_training_state_handoff import BRIDGE_DIRECTORY, BRIDGE_PATH, BUDGET, BUNDLE, GPU, PREFIX, ROOT, SOURCES
from s2a.clean_training_w2_admission import D1_REGISTRATION_REF
from s2a.clean_training_w2_admission import ROOT as W2_ROOT
from s2a.clean_training_w2_handoff import qualified_w2
from s2a.durable_stage_supervisor import REPO, check_cpu_environment, check_sources, publish, read, reference, require


def bridge_source():
    d1 = read(D1_REGISTRATION_REF)
    old_path = BUNDLE / "run_wam_clean_training.py"
    old_ref = dict(path=str(old_path), sha256=d1["source_sha256"][str(old_path)])
    require(reference(old_path) == old_ref, "original041 owner source changed")
    return f'''"""048 explicit metadata import bridge; numerical functions remain original."""
from s2a.clean_training_import_bridge import load_frozen_delegate
from s2a.clean_training_state_handoff import audit_environment, metadata_prior, model_environment

_original = load_frozen_delegate({old_ref!r}, name="_s2a048_original_training_owner")
MASTER_PORT = _original.MASTER_PORT
_flags = _original._flags
observe_release = _original.observe_release

def prior(prepared, prepared_ref, registration_ref, stage):
    if stage == "W2":
        return metadata_prior(prepared, prepared_ref, registration_ref)
    return _original.prior(prepared, prepared_ref, registration_ref, stage)

def environment(prepared, *, gpu):
    base = _original.environment(prepared, gpu=gpu)
    return model_environment(base) if gpu else audit_environment(base)
'''


def write_bridge():
    code = bridge_source()
    BRIDGE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with BRIDGE_PATH.open("x") as stream:
        stream.write(code)
    return reference(BRIDGE_PATH)


def bind():
    require(
        (W2_ROOT / "supervisor/result.json").is_file(),
        "complete047 unavailable; no048 config/proposal may be published",
    )
    old = read(reference(W2_ROOT / "execution/admitted.json"))
    config = {k: old[k] for k in ("prepared_ref", "execution_registration_ref")}
    config.update(
        protocol="048",
        root=str(ROOT),
        stage_scope="STATE_only",
        gpu_uuid=GPU,
        budget=BUDGET,
        private_port=31338,
        owner_timeout_s=45500,
        automatic_retry=False,
        w2_registration_ref=reference(W2_ROOT / "execution/admitted.json"),
        w2_owner_ref=reference(W2_ROOT / "supervisor/result.json"),
        w2_fit_ref=reference(W2_ROOT / "fit/W2/result.json"),
        w2_verification_ref=reference(W2_ROOT / "review/W2_verified.json"),
        w2_params_ref=reference(W2_ROOT / "fit/W2/parameters.json"),
    )
    qualified_w2(config)
    require(BRIDGE_PATH.is_file(), "048 bridge source must be prepared and tested first")
    config.update(
        original_owner_ref=reference(BUNDLE / "run_wam_clean_training_state.py"),
        original_sampler_ref=reference(BUNDLE / "sample_wam_clean_training_state.py"),
        original_verifier_ref=reference(BUNDLE / "verify_wam_clean_training_state.py"),
        original_training_owner_ref=reference(BUNDLE / "run_wam_clean_training.py"),
        bridge_ref=reference(BRIDGE_PATH),
        module_refs={
            "run_wam_clean_training": reference(BRIDGE_PATH),
            "continuation_runtime": reference(BUNDLE / "continuation_runtime.py"),
            "collect_wam_clean_training": reference(BUNDLE / "collect_wam_clean_training.py"),
        },
    )
    qa_ref = reference(ROOT / "qa/qa_receipts_v2.json")
    qa = read(qa_ref)
    require(qa["complete"] is qa["passed"] is True, "048 actual preparation QA missing")
    check_sources(qa["tested_source_sha256"])
    check_sources(old["source_sha256"])
    check_sources(old["snapshot_sha256"])
    config["qa_ref"] = qa_ref
    policy = REPO / "configs/wam_clean_training_state_handoff_policy_20261024.json"
    plan = REPO / "plans/v0.0.1/048_clean_training_state_handoff_20261024.md"
    path = REPO / "configs/wam_clean_training_state_handoff_20261024.json"
    with path.open("x") as stream:
        json.dump(config, stream, indent=2, ensure_ascii=False, allow_nan=False)
    sources = copy.deepcopy(old["source_sha256"])
    snapshots = copy.deepcopy(old["snapshot_sha256"])
    for source in [*(REPO / p for p in SOURCES), BRIDGE_PATH, policy, plan, path]:
        ref = reference(source)
        require(ref["path"] not in sources, "048 source already frozen upstream")
        sources[ref["path"]] = ref["sha256"]
        dest = ROOT / "source_snapshot" / source.relative_to(REPO)
        dest.parent.mkdir(parents=True, exist_ok=True)
        require(not dest.exists(), "048 snapshot exists")
        shutil.copyfile(source, dest)
        snapshots[str(dest)] = reference(dest)["sha256"]
    proposal = copy.deepcopy(config)
    proposal.update(
        schema_version=PREFIX + "proposal.v1",
        complete=True,
        passed=True,
        execution_authorized=False,
        config_ref=reference(path),
        policy_ref=reference(policy),
        plan_ref=reference(plan),
        source_sha256=sources,
        snapshot_sha256=snapshots,
    )
    return publish(ROOT / "execution/proposal.json", proposal)


def main():
    import os
    from pathlib import Path

    require(Path.cwd().resolve() == REPO, "048 preparation must run from S2A")
    check_cpu_environment(os.environ)
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write-bridge-only", action="store_true")
    mode.add_argument("--bind", action="store_true")
    args = parser.parse_args()
    print(json.dumps(write_bridge() if args.write_bridge_only else bind()), flush=True)


if __name__ == "__main__":
    main()
