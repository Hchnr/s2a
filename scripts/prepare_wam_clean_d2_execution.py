"""Freeze056 execution sources, or register independent metadata and dry CLI."""

import argparse
import json
import subprocess
from pathlib import Path

from prepare_wam_synchronous_validation import source_closure
from run_wam_clean_d2 import PORT, worker_command
from run_wam_clean_training import environment

from s2a.artifacts import REPO
from s2a.clean_training_d2 import BUDGET
from s2a.clean_training_d2_execution import ROOT, admit
from s2a.clean_training_runtime import GPU, checked, publish, read, reference, require


def prepare():
    require(not (ROOT / "prepared.json").exists() and not (ROOT / "source").exists(), "056 execution already prepared")
    manifest_ref = reference(ROOT / "current_manifest.json")
    proposal = read(reference(ROOT / "proposal_verified_by_preparer.json"))
    require(
        proposal["passed"] is True
        and proposal["execution_authorized"] is False
        and proposal["manifest_ref"] == manifest_ref,
        "056 checked proposal required",
    )
    old = read(reference(REPO / "outputs/synchronous_state_probe052_continuation_20261024/prepared.json"))
    d1 = read(reference(REPO / "configs/wam_clean_training_d1_handoff_20261024.json"))
    entries = [
        "scripts/prepare_wam_clean_d2_execution.py",
        "scripts/verify_wam_clean_d2_metadata.py",
        "scripts/collect_wam_clean_d2.py",
        "scripts/verify_wam_clean_d2.py",
        "scripts/run_wam_clean_d2.py",
        "src/s2a/clean_training_d2_execution.py",
        "tests/test_clean_training_d2.py",
        "tests/test_clean_training_d2_worker.py",
        "configs/wam_clean_on_policy_aggregation_20261024.json",
        "plans/v0.0.1/056_clean_on_policy_aggregation_20261024.md",
    ]
    local = source_closure(entries)
    sources = dict(old["source_sha256"])
    require(not any(k in sources and sources[k] != v for k, v in local.items()), "056 changed a frozen prerequisite")
    sources.update(local)
    inputs = dict(proposal["inputs_sha256"])
    for ref in (
        manifest_ref,
        reference(ROOT / "proposal_verified_by_preparer.json"),
        d1["d0_owner_ref"],
        d1["d0_capture_audit_ref"],
    ):
        inputs[ref["path"]] = ref["sha256"]
    inputs.update(old["input_sha256"])
    for mapping in (sources, inputs):
        for path, digest in mapping.items():
            checked(dict(path=path, sha256=digest))
    for path, expected in old["weight_files"].items():
        stat = Path(path).stat()
        require(
            stat.st_size == expected["bytes"] and stat.st_mtime_ns == expected["mtime_ns"], "056 checkpoint changed"
        )
    directory = ROOT / "source"
    directory.mkdir()
    snapshots = {}
    for index, (path, digest) in enumerate(sorted(sources.items())):
        snapshot = directory / f"{index:04d}_{Path(path).name}"
        with snapshot.open("xb") as stream:
            stream.write(checked(dict(path=path, sha256=digest)).read_bytes())
        checked(dict(path=str(snapshot), sha256=digest))
        snapshots[str(snapshot)] = digest
    prepared = dict(
        schema_version="s2a.clean-d2-execution-preparation.v1",
        root=str(ROOT),
        budget=BUDGET,
        gpu_uuid=GPU,
        master_port=PORT,
        current_manifest_ref=manifest_ref,
        d0_owner_ref=d1["d0_owner_ref"],
        d0_capture_audit_ref=d1["d0_capture_audit_ref"],
        model_config=old["model_config"],
        prerequisite_refs=old["prerequisite_refs"],
        weight_files=old["weight_files"],
        source_sha256=sources,
        snapshot_sha256=snapshots,
        input_sha256=inputs,
        execution_authorized=False,
        automatic_retry=False,
        fresh_authorized=False,
    )
    return publish(ROOT / "prepared.json", prepared)


def register():
    pref = reference(ROOT / "prepared.json")
    prepared = read(pref)
    metadata_ref = reference(ROOT / "metadata_verified.json")
    metadata = read(metadata_ref)
    require(
        metadata["passed"] is True
        and metadata["prepared_ref"] == pref
        and metadata["requests_verified"] == 300
        and metadata["execution_authorized"] is False,
        "056 independent metadata required",
    )
    registered = dict(
        schema_version="s2a.clean-d2-execution-registration.v1",
        complete=True,
        passed=True,
        execution_authorized=True,
        prepared_ref=pref,
        metadata_verification_ref=metadata_ref,
        budget=BUDGET,
        automatic_retry=False,
        fresh_authorized=False,
        worker_ref=reference(REPO / "scripts/collect_wam_clean_d2.py"),
        owner_ref=reference(REPO / "scripts/run_wam_clean_d2.py"),
        numerical_verifier_ref=reference(REPO / "scripts/verify_wam_clean_d2.py"),
    )
    regref = publish(ROOT / "registration.json", registered)
    worker_inputs, *_ = admit(pref, regref, required_sources=[Path(__file__)])
    command = worker_command(worker_inputs, pref, regref) + ["--dry"]
    completed = subprocess.run(
        command,
        cwd=REPO,
        env=environment(prepared, gpu=False),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    record = dict(
        passed=completed.returncode == 0,
        prepared_ref=pref,
        registration_ref=regref,
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        model_initializations=0,
    )
    if completed.returncode == 0:
        dry = json.loads(completed.stdout)
        require(
            dry["passed"] is dry["complete"] is True
            and dry["model_initializations"] == dry["future_label_reads"] == 0
            and dry["requests"] == 300,
            "056 actual dry response differs",
        )
    publish(ROOT / ("successful_cli_dry.json" if record["passed"] else "failed_cli_dry.json"), record)
    require(record["passed"], "056 actual admitted model CLI dry failed")
    return regref


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    print(json.dumps(register() if args.register else prepare()), flush=True)
