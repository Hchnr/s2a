"""Prepare055, or register its independently checked metadata and dry its model CLI."""

import argparse
import copy
import json
import subprocess
from pathlib import Path

from prepare_wam_synchronous_validation import source_closure
from run_wam_clean_training import environment
from run_wam_synchronous_state_continuation import PORT, worker_command

from s2a.artifacts import REPO, sha256
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.synchronous_state_continuation import BUDGET, ROOT

OLD = REPO / "outputs/synchronous_state_probe052_20261024"


def prepare():
    require(not ROOT.exists(), "055 namespace already exists; no automatic retry")
    prefix_ref = reference(OLD / "prefix_verified.json")
    require(
        prefix_ref["sha256"] == "e0fda7e66b45e25d7cb446c13973b10fa4a990dd63a5af5706a2ac852ccbed58",
        "054 prefix proof changed",
    )
    prefix = read(prefix_ref)
    old = read(prefix["prepared_ref"])
    failed = read(prefix["failed_sampling_ref"])
    value = copy.deepcopy(old)
    value.update(
        schema_version="s2a.synchronous-state-continuation-preparation.v1",
        root=str(ROOT),
        budget=BUDGET,
        prefix_verified_ref=prefix_ref,
        prefix_parameter_identity=failed["parameters_before"],
        remaining_indices=list(range(4, 99)),
        execution_authorized=False,
        fresh_authorized=False,
    )
    raw = failed["generations"][3]
    initial = read(raw["initial_ref"])
    value["input_sha256"].update(
        {
            ref["path"]: ref["sha256"]
            for ref in [prefix_ref, prefix["failed_sampling_ref"], raw["initial_ref"], initial["arrays"]]
        }
    )
    for mapping in [old["source_sha256"], old["snapshot_sha256"], value["input_sha256"]]:
        for path, digest in mapping.items():
            checked({"path": path, "sha256": digest})
    config = {
        "schema_version": "s2a.synchronous-state-continuation-policy.v1",
        "prefix_verified_ref": prefix_ref,
        "original_indices": list(range(4, 99)),
        "budget": BUDGET,
        "master_port": PORT,
        "gpu_uuid": "GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f",
        "automatic_retry": False,
        "fresh_authorized": False,
    }
    config_path = REPO / "configs/wam_synchronous_state_continuation_20261024.json"
    with config_path.open("x") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")
    entries = [
        "scripts/prepare_wam_synchronous_state_continuation.py",
        "scripts/verify_wam_synchronous_state_continuation_metadata.py",
        "scripts/sample_wam_synchronous_state_continuation.py",
        "scripts/verify_wam_synchronous_state_continuation.py",
        "scripts/run_wam_synchronous_state_continuation.py",
        "src/s2a/synchronous_state_continuation.py",
        "src/s2a/synchronous_state_dispatch.py",
        "tests/test_synchronous_state_dispatch.py",
        "scripts/audit_wam_synchronous_state_prefix.py",
        "tests/test_synchronous_state_continuation.py",
        "configs/wam_synchronous_state_continuation_20261024.json",
        "plans/v0.0.1/055_synchronous_state_dispatch_continuation_20261024.md",
    ]
    local = source_closure(entries)
    require(
        not any(k in old["source_sha256"] and old["source_sha256"][k] != v for k, v in local.items()),
        "055 changed a frozen054 source",
    )
    sources = dict(old["source_sha256"])
    sources.update(local)
    ROOT.mkdir()
    (ROOT / "source").mkdir()
    snapshots = {}
    for index, (path, digest) in enumerate(sorted(sources.items())):
        checked({"path": path, "sha256": digest})
        target = ROOT / "source" / f"{index:04d}_{Path(path).name}"
        with target.open("xb") as stream:
            stream.write(Path(path).read_bytes())
        require(sha256(target) == digest, "055 source changed during snapshot")
        snapshots[str(target)] = digest
    value.update(source_sha256=sources, snapshot_sha256=snapshots, continuation_policy_ref=reference(config_path))
    return publish(ROOT / "prepared.json", value)


def register():
    pref = reference(ROOT / "prepared.json")
    p = read(pref)
    mref = reference(ROOT / "metadata_verified.json")
    meta = read(mref)
    require(
        meta["prepared_ref"] == pref and meta["passed"] is True and meta["suffix_dispatch"]["requests"] == 95,
        "055 independent metadata required",
    )
    for mapping in [p["source_sha256"], p["snapshot_sha256"], p["input_sha256"]]:
        for path, digest in mapping.items():
            checked({"path": path, "sha256": digest})
    spec = {
        "schema_version": "s2a.synchronous-state-continuation-registration.v1",
        "complete": True,
        "passed": True,
        "execution_authorized": True,
        "prepared_ref": pref,
        "metadata_verification_ref": mref,
        "budget": BUDGET,
        "owner_ref": reference(REPO / "scripts/run_wam_synchronous_state_continuation.py"),
        "numerical_verifier_ref": reference(REPO / "scripts/verify_wam_synchronous_state_continuation.py"),
        "scorer_ref": reference(REPO / "scripts/score_wam_synchronous_state.py"),
        "automatic_retry": False,
        "fresh_authorized": False,
    }
    regref = publish(ROOT / "registration.json", spec)
    cmd = worker_command(p, pref, regref) + ["--dry"]
    env = environment(p, gpu=False)
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
        output = json.loads(result.stdout.strip().splitlines()[-1])
        receipt["passed"] = (
            output.get("passed") is output.get("dry") is True
            and output["model_initializations"] == output["requests"] == 0
        )
    publish(ROOT / ("successful_cli_dry.json" if receipt["passed"] else "failed_cli_dry.json"), receipt)
    require(receipt["passed"], "055 actual model CLI dry failed; no model launch")
    return regref


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    print(json.dumps(register() if args.register else prepare()), flush=True)


if __name__ == "__main__":
    main()
