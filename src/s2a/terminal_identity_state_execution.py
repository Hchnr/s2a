"""Separate056 aggregate validation admission; no fit or future-label decoding."""

from pathlib import Path

from s2a.artifacts import REPO
from s2a.clean_training_runtime import GPU as GPU
from s2a.clean_training_runtime import checked, read, require
from s2a.terminal_candidate import candidate_head
from s2a.terminal_identity_state_sampling import paired_entries

ROOT = REPO / "outputs/terminal_identity_state_validation_20261025"
BUDGET = dict(
    requests=99,
    denoise_calls=3465,
    head_boundary_calls=3465,
    model_initializations=1,
    fits=0,
    physics_steps=0,
    fresh_cases=0,
)


def select_readout(variant, *, zero, candidate):
    if variant == "raw":
        return None
    if variant == "zero":
        return zero
    if variant == "terminal058":
        return candidate
    raise ValueError("056 unregistered aggregate validation variant")


def admit(prepared_ref, registration_ref, *, required_sources):
    p, registration = read(prepared_ref), read(registration_ref)
    require(
        p["schema_version"] == "s2a.terminal-identity-state-preparation.v1"
        and Path(p["root"]) == ROOT
        and p["budget"] == BUDGET,
        "056 aggregate validation root/budget differs",
    )
    require(
        registration["schema_version"] == "s2a.terminal-identity-state-registration.v1"
        and registration["prepared_ref"] == prepared_ref
        and registration["passed"] is registration["complete"] is registration["execution_authorized"] is True
        and registration["automatic_retry"] is registration["fresh_authorized"] is False
        and registration["budget"] == BUDGET,
        "056 independent aggregate validation registration required",
    )
    meta = read(registration["metadata_verification_ref"])
    require(
        meta["schema_version"] == "s2a.terminal-identity-state-metadata.v1"
        and meta["passed"] is meta["complete"] is meta["current_only"] is True
        and meta["prepared_ref"] == prepared_ref
        and meta["requests_verified"] == 99
        and meta["execution_authorized"] is False,
        "056 independent aggregate validation metadata required",
    )
    sources = p["source_sha256"]
    for field, name in (
        ("worker_ref", "sample_wam_terminal_identity_state.py"),
        ("owner_ref", "run_wam_terminal_identity_state.py"),
        ("numerical_verifier_ref", "verify_wam_terminal_identity_state.py"),
        ("scorer_ref", "score_wam_terminal_identity_state.py"),
    ):
        ref = registration[field]
        require(
            Path(ref["path"]) == REPO / "scripts" / name and sources.get(ref["path"]) == ref["sha256"],
            "056 aggregate execution role not frozen",
        )
        checked(ref)
    verifier = meta["verifier_ref"]
    require(
        Path(verifier["path"]) == REPO / "scripts/verify_wam_terminal_identity_state_metadata.py"
        and sources.get(verifier["path"]) == verifier["sha256"],
        "056 aggregate metadata verifier not frozen",
    )
    for mapping in (sources, p["snapshot_sha256"], p["input_sha256"]):
        require(bool(mapping), "056 empty aggregate validation closure")
        for path, digest in mapping.items():
            checked(dict(path=path, sha256=digest))
    require(all(str(Path(path).resolve()) in sources for path in required_sources), "056 aggregate caller not frozen")
    require(len(p["weight_files"]) == 163, "056 aggregate full checkpoint roster required")
    for path, expected in p["weight_files"].items():
        stat = Path(path).stat()
        require(
            stat.st_size == expected["bytes"] and stat.st_mtime_ns == expected["mtime_ns"],
            "056 aggregate checkpoint changed",
        )
    return p, registration, sources


def admit_sampling(prepared, prepared_ref, registration_ref, candidate_proof_ref):
    require(candidate_proof_ref == prepared["candidate_proof_ref"], "056 aggregate candidate argument differs")
    head, params_ref = candidate_head(candidate_proof_ref)
    require(params_ref == prepared["candidate_params_ref"], "056 aggregate candidate parameter identity differs")
    qualification_ref = prepared["prerequisite_refs"]["bridge_qualification"]
    qualification = read(qualification_ref)
    require(
        qualification["passed"] is qualification["complete"] is qualification["bridge_qualified"] is True,
        "056 original bridge qualification required",
    )
    current = read(prepared["current_inputs_ref"])
    require(
        current["passed"] is current["complete"] is True and current["current_inputs"] == prepared["current_inputs"],
        "056 synchronized current inputs differ",
    )
    expected = paired_entries(
        current["current_inputs"], prepared["request_template"], prepared["zero_cases"], qualification
    )
    require(prepared["sampling_entries"] == expected, "056 fixed aggregate sampling matrix differs")
    return head, params_ref, qualification_ref, qualification, expected
