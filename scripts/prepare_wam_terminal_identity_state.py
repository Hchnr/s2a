"""Freeze terminal058 validation only after full aggregate fit/QR/owner success."""

import argparse
import copy
import json
import subprocess
from pathlib import Path

from run_wam_clean_training import environment
from run_wam_terminal_identity_state import PORT, worker_command

from s2a.artifacts import REPO
from s2a.clean_training_runtime import GPU, checked, publish, read, reference, require
from s2a.terminal_candidate import candidate_head
from s2a.terminal_identity_state_execution import BUDGET, ROOT, admit, admit_sampling
from s2a.terminal_identity_state_sampling import paired_entries


def prepare():
    require(not ROOT.exists(), "Terminal058 validation directory exists; no overwrite/retry")
    owner_ref = reference(REPO / "outputs/terminal_solver_convergence_20261025/fit_verified.json")
    _, params_ref = candidate_head(owner_ref)
    from prepare_wam_synchronous_validation import source_closure

    bref = reference(REPO / "outputs/synchronous_state_probe052_continuation_20261024/prepared.json")
    baseline = read(bref)
    current = read(baseline["current_inputs_ref"])
    require(
        current["passed"] is current["complete"] is True and current["current_inputs"] == baseline["current_inputs"],
        "Terminal058 original synchronized current inputs required",
    )
    qualification = read(baseline["prerequisite_refs"]["bridge_qualification"])
    entries = paired_entries(
        current["current_inputs"], baseline["request_template"], baseline["zero_cases"], qualification
    )
    fields = (
        "model_config",
        "prerequisite_refs",
        "weight_files",
        "current_inputs_ref",
        "current_inputs",
        "request_template",
        "zero_cases",
        "training_seed_plan_ref",
        "noise_roster",
    )
    value = {field: copy.deepcopy(baseline[field]) for field in fields}
    value.update(
        schema_version="s2a.terminal-identity-state-preparation.v1",
        root=str(ROOT),
        baseline_prepared_ref=bref,
        candidate_proof_ref=owner_ref,
        candidate_params_ref=params_ref,
        budget=BUDGET,
        sampling_entries=entries,
        execution_authorized=False,
        fresh_authorized=False,
        automatic_retry=False,
    )
    previous_root = REPO / "outputs/terminal_state_validation_20261025"
    failure_refs = [
        reference(previous_root / name)
        for name in (
            "state_sampling/result.json",
            "orchestration/result.json",
            "orchestration/model/exit.json",
            "orchestration/model/ownership/release.json",
            "orchestration/gpu_after.json",
        )
    ]
    failed, owner, exit_record, release, gpu = [read(ref) for ref in failure_refs]
    require(
        failed["passed"] is owner["passed"] is False
        and failed["actual_requests"] == 4
        and failed["actual_denoise_calls"] == 140
        and failed["error"]["type"] == "AttributeError",
        "Actual059 interface failure required",
    )
    require(
        exit_record["returncode"] == 1
        and exit_record["actual_wait"] is True
        and release["reaped"] is True
        and release["signal_sent"] is False
        and gpu["released"] is True
        and gpu["memory_mib"] == 0
        and gpu["compute_apps"] == [],
        "Actual059 exit and GPU release required",
    )
    value["previous_failure_refs"] = failure_refs
    config_ref = reference(REPO / "configs/wam_terminal_identity_state_20261025.json")
    config = read(config_ref)
    require(
        config["budget"] == BUDGET
        and config["gpu_uuid"] == GPU
        and config["master_port"] == PORT
        and config["seed"] == 42
        and config["automatic_retry"] is config["fresh_authorized"] is False,
        "Terminal058 fixed validation config differs",
    )
    local = source_closure(
        [
            "scripts/prepare_wam_terminal_identity_state.py",
            "scripts/verify_wam_terminal_identity_state_metadata.py",
            "scripts/sample_wam_terminal_identity_state.py",
            "scripts/verify_wam_terminal_identity_state.py",
            "scripts/run_wam_terminal_identity_state.py",
            "scripts/score_wam_terminal_identity_state.py",
            "tests/test_terminal_identity_state_worker.py",
            "tests/test_terminal_identity_state_verification.py",
            "tests/test_terminal_identity_state_admission.py",
            "tests/test_terminal_identity_state_metadata.py",
            "configs/wam_terminal_identity_state_20261025.json",
            "scripts/verify_wam_terminal_points.py",
            "src/s2a/terminal_capture.py",
            "src/s2a/terminal_capture_identity.py",
            "tests/test_terminal_capture_identity.py",
            "src/s2a/terminal_readout_policy.py",
            "src/s2a/terminal_candidate.py",
            "tests/test_terminal_points.py",
            "tests/test_terminal_capture.py",
            "tests/test_terminal_readout_policy.py",
            "plans/v0.0.1/060_terminal_identity_handoff_20261025.md",
        ]
    )
    sources = dict(baseline["source_sha256"])
    require(
        not any(k in sources and sources[k] != v for k, v in local.items()),
        "Terminal058 changed a frozen baseline dependency",
    )
    sources.update(local)
    refs = [
        bref,
        owner_ref,
        params_ref,
        config_ref,
        baseline["current_inputs_ref"],
        current["source_audit_ref"],
        baseline["training_seed_plan_ref"],
        *baseline["prerequisite_refs"].values(),
    ]
    refs += failure_refs
    refs += [a["image_ref"] for a in current["current_inputs"]]
    for case in baseline["zero_cases"]:
        refs.extend([case["image_ref"], case["expected_response_ref"]])
    inputs = {ref["path"]: ref["sha256"] for ref in refs}
    require(
        all(Path(path).suffix in (".json", ".png") for path in inputs),
        "Terminal058 input map cannot contain future binary arrays",
    )
    for mapping in (sources, inputs):
        for path, digest in mapping.items():
            checked(dict(path=path, sha256=digest))
    ROOT.mkdir()
    (ROOT / "source").mkdir()
    snapshots = {}
    for index, (path, digest) in enumerate(sorted(sources.items())):
        target = ROOT / "source" / f"{index:04d}_{Path(path).name}"
        with target.open("xb") as stream:
            stream.write(checked(dict(path=path, sha256=digest)).read_bytes())
        checked(dict(path=str(target), sha256=digest))
        snapshots[str(target)] = digest
    value.update(source_sha256=sources, snapshot_sha256=snapshots, input_sha256=inputs, config_ref=config_ref)
    return publish(ROOT / "prepared.json", value)


def register():
    pref = reference(ROOT / "prepared.json")
    prepared = read(pref)
    mref = reference(ROOT / "metadata_verified.json")
    meta = read(mref)
    require(
        meta["passed"] is meta["complete"] is meta["current_only"] is True
        and meta["prepared_ref"] == pref
        and meta["requests_verified"] == 99
        and meta["execution_authorized"] is False,
        "Terminal058 independent metadata required",
    )
    value = dict(
        schema_version="s2a.terminal-identity-state-registration.v1",
        complete=True,
        passed=True,
        execution_authorized=True,
        prepared_ref=pref,
        metadata_verification_ref=mref,
        budget=BUDGET,
        automatic_retry=False,
        fresh_authorized=False,
        worker_ref=reference(REPO / "scripts/sample_wam_terminal_identity_state.py"),
        owner_ref=reference(REPO / "scripts/run_wam_terminal_identity_state.py"),
        numerical_verifier_ref=reference(REPO / "scripts/verify_wam_terminal_identity_state.py"),
        scorer_ref=reference(REPO / "scripts/score_wam_terminal_identity_state.py"),
    )
    regref = publish(ROOT / "registration.json", value)
    p, _, _ = admit(pref, regref, required_sources=[Path(__file__)])
    admit_sampling(p, pref, regref, p["candidate_proof_ref"])
    command = worker_command(p, pref, regref) + ["--dry"]
    result = subprocess.run(
        command,
        cwd=REPO,
        env=environment(prepared, gpu=False),
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )
    record = dict(
        passed=result.returncode == 0,
        prepared_ref=pref,
        registration_ref=regref,
        command=command,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        new_model_initializations=0,
    )
    if record["passed"]:
        dry = json.loads(result.stdout)
        require(
            dry["passed"] is dry["dry"] is True and dry["model_initializations"] == dry["requests"] == 0,
            "Terminal058 actual model CLI dry response differs",
        )
    publish(ROOT / ("successful_cli_dry.json" if record["passed"] else "failed_cli_dry.json"), record)
    require(record["passed"], "Terminal058 actual model CLI dry failed")
    return regref


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    print(json.dumps(register() if args.register else prepare()), flush=True)
