"""Freeze preview WAM code/current metadata and separately register verified execution."""

import argparse
import copy
import json
import subprocess
from pathlib import Path

from run_wam_clean_training import environment

from s2a.artifacts import REPO
from s2a.clean_training_runtime import GPU, checked, publish, read, reference, require
from s2a.coupled_preview_control_gate import admit_preview_control
from s2a.serialized_preview_recovery import recovery_evidence
from s2a.serialized_preview_wam_control import BASE_REF, RELOCATION_REF, make_case_config
from s2a.serialized_preview_wam_execution import BUDGET, CONTROL_REF, MASTER_PORT, PORT, ROLES, ROOT, admit_execution
from s2a.terminal_development_gate import ROOT as STATE_ROOT
from s2a.terminal_development_gate import admit_state
from s2a.terminal_development_policy import case_order

TESTS = sorted(
    {
        "tests/test_coupled_preview_control_gate.py",
        "tests/test_preview_result_publication.py",
        "tests/test_serialized_preview_derivation.py",
    }
    | {
        str(p.relative_to(REPO))
        for pattern in (
            "test_serialized_preview_wam_*.py",
            "test_coupled_preview_wam_protocol.py",
            "test_coupled_pose_preview*.py",
        )
        for p in (REPO / "tests").glob(pattern)
    }
)


def qa():
    folder = REPO / "outputs/serialized_preview_wam_preparation_20261026"
    command = [str(REPO / ".venv/bin/python"), "-m", "pytest", "-q", *TESTS]
    env = dict(
        __import__("os").environ,
        CUDA_VISIBLE_DEVICES="",
        LD_LIBRARY_PATH="",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH="src:scripts",
        OPENBLAS_NUM_THREADS="1",
    )
    result = subprocess.run(command, cwd=REPO, env=env, text=True, capture_output=True, check=False)
    refs = [reference(REPO / p) for p in TESTS]
    refs += [reference(REPO / "scripts" / name) for name in ROLES.values()]
    record = dict(
        passed=result.returncode == 0,
        complete=True,
        command=command,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        source_refs=refs,
        new_model_calls=0,
        physics_steps=0,
    )
    publish(folder / "component_qa.json", record)
    require(record["passed"], "061 component QA failed")


def prepare():
    require(not ROOT.exists(), "061 output exists; no overwrite/retry")
    _, a = admit_state(reference(STATE_ROOT / "orchestration/result.json"))
    state = read(a["state_prepared_ref"])
    meta_ref = reference(REPO / "outputs/terminal_development_preparation_20261025/policy_metadata.json")
    meta = read(meta_ref)
    preflight_ref = reference(
        REPO / "outputs/serialized_preview_wam_preparation_20261026/input_preflight_py311/result.json"
    )
    preflight = read(preflight_ref)
    require(
        preflight["passed"]
        and preflight["case_count"] == 12
        and preflight["actual_model_calls"] == preflight["actual_physics_steps"] == 0,
        "Actual12 input preflight required",
    )
    old_registration_ref = reference(
        REPO / "outputs/clean_training_development_continuation_20261024/execution/admitted.json"
    )
    old = read(read(old_registration_ref)["prepared_ref"])
    qref = reference(REPO / "outputs/serialized_preview_wam_preparation_20261026/component_qa.json")
    q = read(qref)
    require(q["passed"] and q["returncode"] == 0, "Actual component QA required")
    value = dict(
        copy.deepcopy(meta["policy_manifest"]),
        schema_version="s2a.serialized-preview-wam-development-preparation.v1",
        budget=BUDGET,
        state_owner_ref=a["state_owner_ref"],
        candidate_proof_ref=a["candidate_proof_ref"],
        candidate_params_ref=a["candidate_params_ref"],
        baseline_params_ref=reference(REPO / "outputs/clean_regularization_probe_20261024/parameters.json"),
        tool_relocation_ref=RELOCATION_REF,
        config=dict(gpu_uuid=GPU),
        resources=old["resources"],
        case_order=case_order(),
        policy_metadata_ref=meta_ref,
        input_preflight_ref=preflight_ref,
        qa_ref=qref,
        legacy_resource_registration_ref=old_registration_ref,
        execution_authorized=False,
        fresh_authorized=False,
        automatic_retry=False,
    )
    for key in ("model_config", "prerequisite_refs", "weight_files", "noise_roster"):
        value[key] = copy.deepcopy(state[key])
    closure_ref = reference(REPO / "outputs/coupled_preview_control_20261026/closure_verified.json")
    physical = admit_preview_control(closure_ref)
    require(preflight["physical_control_admission"] == physical, "Actual preflight physical qualification differs")
    value.update(root=str(ROOT), preview_control_closure_ref=closure_ref, physical_control_admission=physical)
    from prepare_wam_synchronous_validation import source_closure

    entries = (
        TESTS
        + [
            "scripts/prepare_serialized_preview_wam_development.py",
            "scripts/verify_serialized_preview_wam_metadata.py",
            "scripts/serialized_preview_wam_audit_admission.py",
            "scripts/verify_wam_terminal_development_points.py",
            "plans/v0.0.1/237_coupled_wam_development_handoff_20261026.md",
            "scripts/check_serialized_preview_wam_inputs.py",
            "tests/test_terminal_development_service.py",
            "tests/fixtures/coupled_preview_wam_responses.json",
            "tests/fixtures/coupled_preview_case07_step88.json",
        ]
        + ["scripts/" + n for n in ROLES.values()]
    )
    entries += ["tests/fixtures/serialized_preview_derivation.json"]
    local = source_closure(entries)
    sources = dict(state["source_sha256"])
    require(all(k not in sources or sources[k] == v for k, v in local.items()), "061 modified a frozen prerequisite")
    sources.update(local)
    refs = [
        a["state_owner_ref"],
        a["state_verification_ref"],
        a["state_score_ref"],
        a["state_prepared_ref"],
        a["candidate_proof_ref"],
        a["candidate_params_ref"],
        value["baseline_params_ref"],
        RELOCATION_REF,
        BASE_REF,
        CONTROL_REF,
        closure_ref,
        meta_ref,
        preflight_ref,
        qref,
        old_registration_ref,
        *value["prerequisite_refs"].values(),
    ]
    relocation = read(RELOCATION_REF)
    refs += [relocation[k] for k in ("old_template_ref", "new_template_ref", "safe_copy_ref")]
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    refs += [guard["helper_ref"], guard["exact_probe_ref"], *guard["runtime_source_refs"], *q["source_refs"]]
    value["recovery"] = recovery_evidence()
    refs += list(value["recovery"]["refs"].values())
    inputs = {ref["path"]: ref["sha256"] for ref in refs}
    for row in preflight["records"]:
        inputs.update(row["input_sha256"])
        inputs[row["config_ref"]["path"]] = row["config_ref"]["sha256"]
    for mapping in (sources, inputs):
        for path, digest in mapping.items():
            checked(dict(path=path, sha256=digest))
    ROOT.mkdir()
    (ROOT / "source").mkdir()
    snapshots = {}
    for i, (path, digest) in enumerate(sorted(sources.items())):
        target = ROOT / "source" / f"{i:04d}_{Path(path).name}"
        with target.open("xb") as stream:
            stream.write(Path(path).read_bytes())
        checked(dict(path=str(target), sha256=digest))
        snapshots[str(target)] = digest
    value.update(source_sha256=sources, snapshot_sha256=snapshots, input_sha256=inputs)
    return publish(ROOT / "prepared.json", value)


def register():
    pref = reference(ROOT / "prepared.json")
    p = read(pref)
    mref = reference(ROOT / "metadata_verified.json")
    m = read(mref)
    require(
        m["passed"] is m["complete"] is True and m["prepared_ref"] == pref and m["cases_verified"] == 12,
        "061 metadata must pass before registration",
    )
    _, a = admit_state(p["state_owner_ref"])
    reg = dict(
        schema_version="s2a.serialized-preview-wam-development-registration.v1",
        passed=True,
        complete=True,
        execution_authorized=True,
        automatic_retry=False,
        fresh_authorized=False,
        prepared_ref=pref,
        metadata_verification_ref=mref,
        budget=BUDGET,
        development=dict(port=PORT, master_port=MASTER_PORT, case_order=case_order()),
        roles={role: reference(REPO / "scripts" / name) for role, name in ROLES.items()},
        state_admission=a,
        preview_control_closure_ref=p["preview_control_closure_ref"],
        physical_control_admission=admit_preview_control(p["preview_control_closure_ref"]),
        source_sha256=p["source_sha256"],
        case_order=case_order(),
        service_source_ref=reference(REPO / "scripts" / ROLES["service"]),
        development_control_ref=CONTROL_REF,
    )
    ref = publish(ROOT / "registration.json", reg)
    admit_execution(pref, ref, required_sources=[__file__])
    common = []
    for name, r in (
        ("prepared", pref),
        ("execution-registration", ref),
        ("state-owner", a["state_owner_ref"]),
        ("state-verification", a["state_verification_ref"]),
    ):
        common += ["--" + name, r["path"], "--" + name + "-sha256", r["sha256"]]
    commands = [
        (
            "owner",
            [
                str(REPO / ".venv/bin/python"),
                "scripts/run_serialized_preview_wam_development.py",
                *common,
                "--seed",
                "42",
                "--dry-run",
            ],
        ),
        (
            "service",
            [
                p["model_config"]["model_python"],
                "scripts/serve_serialized_preview_wam_development.py",
                *common,
                "--host",
                "127.0.0.1",
                "--port",
                str(PORT),
                "--gpu-uuid",
                GPU,
                "--seed",
                "42",
                "--dry",
            ],
        ),
    ]
    for row in read(p["input_preflight_ref"])["records"]:
        config = make_case_config(p, reg, prepared_ref=pref, registration_ref=ref, case=row["case"])
        require(
            {k: v for k, v in config.items() if k != "wam_development"} == read(row["config_ref"]),
            "Executable config changed the independently checked input profile",
        )
        r = publish(ROOT / "cli_configs" / f"{row['case']['case_id']}.json", config)
        commands.append(
            (
                row["case"]["case_id"],
                [
                    str(REPO / "outputs/libero_env311/bin/python"),
                    "scripts/collect_serialized_preview_wam_development.py",
                    *common,
                    "--case-config",
                    r["path"],
                    "--case-config-sha256",
                    r["sha256"],
                    "--case-index",
                    str(row["case"]["case_index"]),
                    "--seed",
                    "42",
                    "--dry",
                ],
            )
        )
    records = []
    for role, cmd in commands:
        result = subprocess.run(
            cmd, cwd=REPO, env=environment(p, gpu=False), text=True, capture_output=True, timeout=600, check=False
        )
        records.append(
            dict(role=role, command=cmd, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        )
        if result.returncode != 0:
            publish(ROOT / "failed_cli_dry.json", dict(passed=False, records=records))
            raise ValueError("061 actual CLI dry failed: " + role)
    publish(
        ROOT / "successful_cli_dry.json",
        dict(
            passed=True,
            complete=True,
            records=records,
            prepared_ref=pref,
            registration_ref=ref,
            new_model_initializations=0,
            physics_steps=0,
        ),
    )
    return ref


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--qa", action="store_true")
    group.add_argument("--register", action="store_true")
    args = parser.parse_args()
    print(json.dumps(qa() if args.qa else register() if args.register else prepare()), flush=True)
