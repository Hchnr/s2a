"""051 immutable metadata admission and per-process source qualification."""

import copy
import os
import sys
from pathlib import Path

from s2a.clean_training_d1_admission import canonical
from s2a.clean_training_d1_lifecycle_addendum import Evidence
from s2a.clean_training_state_handoff import GPU
from s2a.durable_stage_supervisor import REPO, identity, publish, read, reference, require

ROOT = REPO / "outputs/clean_training_development_dependency_repair_20261024"
BUNDLE = ROOT / "source_runtime_py311"
PREFIX = "s2a.clean-development-dependency-repair-"
ENV_PATH = "S2A_DEVELOPMENT_REPAIR_PATH"
ENV_SHA = "S2A_DEVELOPMENT_REPAIR_SHA256"
OLD_REF = dict(
    path=str(REPO / "outputs/clean_training_development_continuation_20261024/execution/admitted.json"),
    sha256="66b7a2e48fbeee0ac4283d30cd079f15b8e730c5f8baa8fe6e3c90da7ec721b6",
)
STATE_AUDIT_REF = dict(
    path=str(REPO / "outputs/clean_training_state_continuation_20261024/qa/terminal_audit.json"),
    sha256="0701495d0d1512f51de8bbf4ac0216eba74df3535953e88235a31d32affc40dd",
)
FAILURE_REF = dict(
    path=str(REPO / "outputs/clean_training_development_continuation_20261024/qa/retained_failure_audit.json"),
    sha256="39f517b2b1969603acca12ff7dc98d510c042fff9faab3fc13df0b9a35579819",
)
BUDGET = dict(
    model_initializations=1,
    cumulative_development_model_initializations=2,
    cases=12,
    requests=900,
    denoise_calls=31500,
    physics_steps=7200,
    fit_calls=0,
    qr_calls=0,
    fresh_cases=0,
    automatic_retry=False,
)


SOURCES = (
    "src/s2a/development_source_relocation.py",
    "src/s2a/development_runtime_bridge.py",
    "src/s2a/development_runtime_sources.py",
    "src/s2a/development_repair_admission.py",
    "scripts/prepare_wam_clean_development_runtime.py",
    "scripts/prepare_wam_clean_development_repair.py",
    "scripts/verify_wam_clean_development_repair.py",
    "scripts/run_wam_clean_development_repair.py",
    "scripts/check_wam_clean_development_runtime_inputs.py",
    "tests/test_development_source_relocation.py",
    "tests/test_development_runtime_bridge.py",
    "tests/test_development_runtime_sources.py",
    "tests/test_development_repair_admission.py",
)


def validate(proposal_ref):
    e = Evidence()
    p = e.json(proposal_ref)
    require(
        p["schema_version"] == PREFIX + "proposal.v1"
        and p["complete"] is p["passed"] is True
        and p["execution_authorized"] is False
        and p["root"] == str(ROOT)
        and p["stage_scope"] == "DEVELOPMENT_only"
        and p["budget"] == BUDGET
        and p["gpu_uuid"] == GPU
        and p["automatic_retry"] is False,
        "051 finite development proposal required",
    )
    require(
        p["old049_ref"] == OLD_REF
        and p["state_audit_ref"] == STATE_AUDIT_REF
        and p["failure_audit_ref"] == FAILURE_REF,
        "051 exact qualified/failed basis required",
    )
    config = e.json(p["config_ref"])
    require(all(p.get(key) == value for key, value in config.items()), "051 final config differs")
    old = e.json(OLD_REF)
    state = e.json(STATE_AUDIT_REF)
    failure = e.json(FAILURE_REF)
    require(
        state["complete"] is state["passed"] is state["numerical_contract_passed"] is True
        and state["state_gate_passed"] is False
        and state["validation_initial_sync_certified"] is False,
        "STATE limitations or qualification changed",
    )
    require(
        failure["complete"] is failure["passed"] is failure["original_false_preserved"] is True
        and failure["model_initializations"] == 1
        and failure["actual_requests"] == failure["physics_steps"] == failure["completed_cases"] == 0
        and failure["service_actual_wait1_reap_no_signal"] is True,
        "049 retained failure budget differs",
    )
    e.json(failure["original_supervisor_ref"])
    e.json(failure["original_service_ref"])
    for key in ("service_exit_ref", "service_release_ref", "gpu_after_ref"):
        e.json(failure[key])
    require(len(old["source_sha256"]) == 562, "049 source baseline incomplete")
    for key in ("source_sha256", "snapshot_sha256"):
        require(
            all(p[key].get(path) == digest for path, digest in old[key].items()), "051 modified original562 closure"
        )
        e.mapping(p[key])
    require(set(p["source_sha256"].values()) <= set(p["snapshot_sha256"].values()), "051 snapshots incomplete")
    for key in ("prepared_ref", "execution_registration_ref", "state_owner_ref", "state_verification_ref"):
        require(p[key] == old[key], "051 original041/STATE identity differs")
    relocation = e.json(p["relocation_ref"])
    before = e.json(relocation["old_template_ref"])
    after = e.json(relocation["new_template_ref"])
    e.file(relocation["safe_copy_ref"])
    e.file(relocation["historical_proof_ref"])
    require(
        relocation["safe_copy_ref"]["sha256"]
        == relocation["historical_sha256"]
        == "6fcaed3fe499aacb15d03170d002b1c897091481a0022db5f2922c06320a3a0c"
        and before["tool_contract"] == relocation["historical_declaration"]
        and after == dict(before, tool_contract=relocation["safe_copy_ref"]["path"]),
        "051 tool contract changed rather than relocated",
    )
    generated = e.json(p["source_generation_ref"])
    expected = {item["generated_ref"]["path"]: item["original_ref"] for item in generated["generated"]}
    require(p["generated_to_original"] == expected and len(expected) == 5, "051 finite entrypoint basis differs")
    for item in generated["generated"]:
        e.file(item["generated_ref"])
        e.file(item["original_ref"])
        require(
            p["source_sha256"].get(item["generated_ref"]["path"]) == item["generated_ref"]["sha256"],
            "051 generated source not frozen",
        )
    from s2a.clean_training_state_handoff import BUNDLE as OLD_BUNDLE
    from s2a.development_runtime_sources import verify_generated

    verify_generated(
        generated, old_bundle=OLD_BUNDLE, new_bundle=BUNDLE, template=relocation["new_template_ref"]["path"], repo=REPO
    )
    preflight = e.json(p["input_preflight_ref"])
    require(
        preflight["complete"] is preflight["passed"] is True
        and preflight["case_count"] == 12
        and preflight["expected_requests"] == 900,
        "051 full-case preflight missing",
    )
    for row in preflight["records"]:
        e.mapping(row["input_sha256"])
    qa = e.json(p["qa_ref"])
    require(
        qa["complete"] is qa["passed"] is qa["ruff_passed"] is qa["format_passed"] is True
        and qa["pytest_returncode"] == 0
        and qa["actual_model_calls"] == qa["actual_physics_steps"] == 0,
        "051 preparation QA incomplete",
    )
    e.file(qa["pytest_log_ref"])
    require(
        set(str(REPO / r) for r in SOURCES) <= set(qa["tested_source_sha256"]), "051 actual QA source coverage missing"
    )
    for path, digest in qa["tested_source_sha256"].items():
        require(p["source_sha256"].get(path) == digest, "051 tested source drift")
    require(
        p["source_sha256"].get(str(Path(__file__).resolve())) == reference(__file__)["sha256"],
        "051 actual admission source not frozen",
    )
    e.finish()
    return p


def admit(ref):
    require(ref["path"] == str(ROOT / "execution/admitted.json"), "051 actual admitted path required")
    a = read(ref)
    p = validate(a["proposal_ref"])
    v = read(a["independent_verification_ref"])
    require(
        v["schema_version"] == PREFIX + "verification.v1"
        and v["complete"] is v["passed"] is v["eligible_for_execution"] is True
        and v["proposal_ref"] == a["proposal_ref"]
        and v["verifier_ref"] == reference(REPO / "scripts/verify_wam_clean_development_repair.py"),
        "051 independent proof differs",
    )
    require(a["source_sha256"].get(v["verifier_ref"]["path"]) == v["verifier_ref"]["sha256"], "051 verifier not frozen")
    expected = copy.deepcopy(p)
    expected.update(
        schema_version=PREFIX + "registration.v1",
        execution_authorized=True,
        proposal_ref=a["proposal_ref"],
        independent_verification_ref=a["independent_verification_ref"],
    )
    require(canonical(expected) == canonical(a), "051 admitted payload differs")
    return a


def current_qualification():
    ref = dict(path=os.environ.get(ENV_PATH, ""), sha256=os.environ.get(ENV_SHA, ""))
    a = admit(ref)
    filename = Path(sys.argv[0]).resolve()
    roles = {
        "run_wam_clean_training_development.py": "owner",
        "serve_wam_clean_training.py": "model",
        "execute_wam_clean_training_case.py": "client",
        "verify_wam_clean_training_development.py": "audit",
    }
    require(filename.parent == BUNDLE and filename.name in roles, "051 actual role entrypoint differs")
    role = roles[filename.name]
    require(
        os.environ.get("LD_LIBRARY_PATH") == ""
        and os.environ.get("CUDA_VISIBLE_DEVICES") == (GPU if role == "model" else ""),
        "051 actual role CUDA/LD differs",
    )
    if role == "client":
        require("--case-index" in sys.argv, "051 case identity missing")
        index = int(sys.argv[sys.argv.index("--case-index") + 1])
        require(0 <= index < 12, "051 case index differs")
        role = f"client_{index:02d}"
    if "--dry-run" not in sys.argv:
        path = ROOT / "role_receipts" / (role + ".json")
        who = identity(os.getpid())
        receipt = dict(
            complete=True,
            passed=True,
            registration_ref=ref,
            role=role,
            identity=who,
            entrypoint_ref=reference(filename),
        )
        if path.exists():
            prior = read(reference(path))
            require(
                prior["registration_ref"] == ref
                and all(prior["identity"][k] == who[k] for k in ("pid", "start_ticks", "boot_id")),
                "051 role receipt belongs to another process",
            )
        else:
            publish(path, receipt)
    return a


def current_sources(original):
    from s2a.development_runtime_bridge import extend_sources

    return extend_sources(original, current_qualification()["source_sha256"])
