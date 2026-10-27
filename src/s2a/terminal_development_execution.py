"""Explicit061 execution registration, distinct from preparation-only metadata."""

from pathlib import Path

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, require
from s2a.terminal_development_gate import admit_state as admit_state_gate
from s2a.terminal_development_policy import case_order

ROOT = REPO / "outputs/terminal_development_20261025"
PORT = 31363
MASTER_PORT = 31364
BUDGET = dict(
    model_initializations=1,
    requests=900,
    denoise_calls=31500,
    head_calls=31500,
    physics_steps=7200,
    fits=0,
    fresh_cases=0,
)
ROLES = dict(
    service="serve_wam_terminal_development.py",
    owner="run_wam_terminal_development.py",
    client="run_wam_terminal_development_case.py",
    verifier="verify_wam_terminal_development.py",
)


def admit_execution(prepared_ref, registration_ref, *, required_sources):
    prepared, registration = read(prepared_ref), read(registration_ref)
    require(
        prepared["schema_version"] == "s2a.terminal-development-preparation.v1"
        and Path(prepared["root"]) == ROOT
        and prepared["budget"] == BUDGET,
        "061 fixed preparation differs",
    )
    require(
        registration["schema_version"] == "s2a.terminal-development-registration.v1"
        and registration["passed"] is registration["complete"] is registration["execution_authorized"] is True
        and registration["automatic_retry"] is registration["fresh_authorized"] is False
        and registration["prepared_ref"] == prepared_ref
        and registration["budget"] == BUDGET
        and registration["development"] == dict(port=PORT, master_port=MASTER_PORT, case_order=case_order()),
        "061 complete execution registration required",
    )
    metadata = read(registration["metadata_verification_ref"])
    require(
        metadata["schema_version"] == "s2a.terminal-development-metadata.v1"
        and metadata["passed"] is metadata["complete"] is True
        and metadata["prepared_ref"] == prepared_ref
        and metadata["cases_verified"] == 12
        and metadata["execution_authorized"] is False,
        "061 independent complete metadata required",
    )
    sources = prepared["source_sha256"]
    verifier = metadata["verifier_ref"]
    require(
        Path(verifier["path"]) == REPO / "scripts/verify_wam_terminal_development_metadata.py"
        and sources.get(verifier["path"]) == verifier["sha256"],
        "061 metadata verifier source differs",
    )
    checked(verifier)
    require(set(registration["roles"]) == set(ROLES), "061 exact role roster required")
    for mapping in (sources, prepared["snapshot_sha256"], prepared["input_sha256"]):
        require(bool(mapping), "061 nonempty complete source/input closure required")
        for path, digest in mapping.items():
            checked(dict(path=path, sha256=digest))
    require(all(str(Path(path).resolve()) in sources for path in required_sources), "061 caller source not frozen")
    for role, filename in ROLES.items():
        ref = registration["roles"][role]
        require(
            Path(ref["path"]) == REPO / "scripts" / filename and sources.get(ref["path"]) == ref["sha256"],
            "061 role source differs",
        )
        checked(ref)
    require(len(prepared["weight_files"]) == 163, "061 full checkpoint roster required")
    for path, expected in prepared["weight_files"].items():
        stat = Path(path).stat()
        require(
            stat.st_size == expected["bytes"] and stat.st_mtime_ns == expected["mtime_ns"], "061 checkpoint changed"
        )
    _, admission = admit_state_gate(prepared["state_owner_ref"])
    require(
        registration["state_admission"] == admission
        and prepared["candidate_params_ref"] == admission["candidate_params_ref"],
        "061 actual060 qualification differs",
    )
    return prepared, registration, sources


def admit_service_state(prepared, prepared_ref, registration_ref, owner_ref, verification_ref):
    policy, admission = admit_state_gate(owner_ref)
    require(
        owner_ref == prepared["state_owner_ref"]
        and verification_ref == admission["state_verification_ref"]
        and prepared["candidate_params_ref"] == admission["candidate_params_ref"],
        "061 explicit state/candidate arguments differ",
    )
    return admission, policy, admission["candidate_params_ref"]


CONTROL_REF = dict(
    path=str(REPO / "outputs/clean_training_development_preflight_20261023/dependency_addendum.json"),
    sha256="106f45543f8bc0aa0e336c0ad9b5c01e0bce252a5e5dad0ec3ebbba349e5fda0",
)


def admit_development(prepared_ref, registration_ref, state_owner_ref, state_verification_ref, *, required_sources):
    prepared, registration, sources = admit_execution(prepared_ref, registration_ref, required_sources=required_sources)
    admission, _, _ = admit_service_state(
        prepared, prepared_ref, registration_ref, state_owner_ref, state_verification_ref
    )
    require(registration["development_control_ref"] == CONTROL_REF, "061 qualified control reference differs")
    control = read(CONTROL_REF)
    require(
        control["passed"] is control["complete"] is control["control_contract_reuse_eligible"] is True
        and control["new_physics_control_qualification"] is control["execution_ready"] is False,
        "061 original conditional control qualification required",
    )
    guard = control["cv2_guard_registration"]
    for ref in (guard["helper_ref"], guard["exact_probe_ref"], *guard["runtime_source_refs"]):
        checked(ref)
        require(
            prepared["input_sha256"].get(ref["path"], sources.get(ref["path"])) == ref["sha256"],
            "061 exact native preload/probe binding missing",
        )
    return prepared, registration, sources, admission, control
