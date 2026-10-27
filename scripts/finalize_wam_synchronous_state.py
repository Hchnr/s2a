"""Freeze054 final source closure, then register and actually dry its model CLI."""

import argparse
import copy
import json
import subprocess
from pathlib import Path

from prepare_wam_synchronous_validation import source_closure
from run_wam_clean_training import environment
from run_wam_synchronous_state import PORT, worker_command

from s2a.artifacts import REPO, sha256
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.synchronous_state_sampling import BUDGET, GPU, candidate_head, paired_entries

ROOT = REPO / "outputs/synchronous_state_probe052_20261024"


def prepare():
    previous_ref = reference(ROOT / "prepared_v2.json")
    require(
        previous_ref["sha256"] == "06a0a61c3e9d2d42e11252dfeb266164945b83c78065d205b35ac9afb1454d84",
        "Historical054 preparation changed",
    )
    value = copy.deepcopy(read(previous_ref))
    require(not (ROOT / "state_sampling").exists() and not (ROOT / "orchestration").exists(), "054 has already run")
    old = read(
        reference(
            REPO
            / "outputs/wam_stability_20261021/clean_residual_training_20261023/recovery_20261023/preparation/prepared.json"
        )
    )
    candidate_head(value["candidate_fit_ref"], value["candidate_verification_ref"])
    qualification = read(value["prerequisite_refs"]["bridge_qualification"])
    value["sampling_entries"] = paired_entries(
        value["current_inputs"], value["request_template"], value["zero_cases"], qualification
    )
    config = {
        "schema_version": "s2a.synchronous-state-execution-policy.v1",
        "gpu_uuid": GPU,
        "master_port": PORT,
        "budget": BUDGET,
        "model_timeout_s": 43200,
        "numeric_timeout_s": 14400,
        "score_timeout_s": 3600,
        "candidate_params_ref": value["candidate_params_ref"],
        "current_inputs_ref": value["current_inputs_ref"],
        "automatic_retry": False,
        "fresh_authorized": False,
    }
    config_path = REPO / "configs/wam_synchronous_state_execution_20261024.json"
    publish(ROOT / "execution_policy.json", config)
    with config_path.open("x") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")
    entries = [
        "scripts/finalize_wam_synchronous_state.py",
        "scripts/run_wam_synchronous_state.py",
        "scripts/sample_wam_synchronous_state.py",
        "scripts/verify_wam_synchronous_state.py",
        "scripts/score_wam_synchronous_state.py",
        "scripts/verify_wam_synchronous_state_metadata.py",
        "src/s2a/synchronous_state_sampling.py",
        "src/s2a/synchronous_state_metrics.py",
        "tests/test_synchronous_state_sampling.py",
        "tests/test_synchronous_state_verification.py",
        "tests/test_clean_training_state_verification.py",
        "tests/test_durable_stage_supervisor.py",
        "configs/wam_synchronous_state_execution_20261024.json",
        "plans/v0.0.1/054_synchronous_state_probe052_20261024.md",
    ]
    sources = dict(old["qualified_model_source_sha256"])
    local = source_closure(entries)
    require(
        not any(k in sources and sources[k] != v for k, v in local.items()),
        "054 final closure conflicts with qualified model code",
    )
    sources.update(local)
    for path, digest in sources.items():
        checked({"path": path, "sha256": digest})
    folder = ROOT / "source_final"
    folder.mkdir(exist_ok=False)
    snapshots = {}
    for index, (path, digest) in enumerate(sorted(sources.items())):
        target = folder / f"{index:04d}_{Path(path).name}"
        with target.open("xb") as stream:
            stream.write(Path(path).read_bytes())
        require(sha256(target) == digest, "054 final source changed during snapshot")
        snapshots[str(target)] = digest
    value.update(
        source_sha256=sources,
        snapshot_sha256=snapshots,
        preparation_revision={
            "previous_ref": previous_ref,
            "reason": "final numerical audit, lifecycle owner, scoring and explicit zero response-origin binding; no prior model execution",
        },
        execution_policy_ref=reference(config_path),
    )
    return publish(ROOT / "prepared_final.json", value)


def register():
    pref = reference(ROOT / "prepared_final.json")
    prepared = read(pref)
    meta_ref = reference(ROOT / "metadata_final_verified.json")
    metadata = read(meta_ref)
    require(
        metadata["prepared_ref"] == pref
        and metadata["passed"] is True
        and metadata["worker_body_proof"]["passed"] is metadata["numerical_body_proof"]["passed"] is True,
        "054 final independent metadata and complete body proofs required",
    )
    for mapping in [prepared["source_sha256"], prepared["snapshot_sha256"], prepared["input_sha256"]]:
        for path, digest in mapping.items():
            checked({"path": path, "sha256": digest})
    registered = {
        "schema_version": "s2a.synchronous-state-registration.v1",
        "complete": True,
        "passed": True,
        "execution_authorized": True,
        "prepared_ref": pref,
        "metadata_verification_ref": meta_ref,
        "budget": BUDGET,
        "numerical_verifier_ref": reference(REPO / "scripts/verify_wam_synchronous_state.py"),
        "scorer_ref": reference(REPO / "scripts/score_wam_synchronous_state.py"),
        "owner_ref": reference(REPO / "scripts/run_wam_synchronous_state.py"),
        "master_port": PORT,
        "gpu_uuid": GPU,
        "automatic_retry": False,
        "fresh_authorized": False,
    }
    regref = publish(ROOT / "registration.json", registered)
    cmd = worker_command(prepared, pref, regref) + ["--dry"]
    env = environment(prepared, gpu=False)
    env["MASTER_PORT"] = str(PORT)
    result = subprocess.run(cmd, cwd=REPO, env=env, text=True, capture_output=True, check=False)
    receipt = {
        "passed": False,
        "prepared_ref": pref,
        "registration_ref": regref,
        "command": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "model_initializations": 0,
        "requests": 0,
    }
    if result.returncode == 0:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        receipt["passed"] = (
            payload.get("passed") is payload.get("dry") is True
            and payload["model_initializations"] == payload["requests"] == 0
        )
    target = ROOT / ("successful_cli_dry.json" if receipt["passed"] else "failed_cli_dry.json")
    publish(target, receipt)
    require(receipt["passed"], "054 actual admitted model CLI dry failed; no model launch")
    return regref


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    print(json.dumps(register() if args.register else prepare()), flush=True)


if __name__ == "__main__":
    main()
