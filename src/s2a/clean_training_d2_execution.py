"""056 execution admission, separate from the non-executable current proposal."""

from pathlib import Path

from s2a.artifacts import REPO
from s2a.clean_training_d2 import BUDGET, validate_roster
from s2a.clean_training_runtime import GPU, checked, read, require
from s2a.synchronous_state_sampling import candidate_head

ROOT = REPO / "outputs/clean_on_policy_aggregation_20261024"


def admit(prepared_ref, registration_ref, *, required_sources):
    prepared, registration = read(prepared_ref), read(registration_ref)
    require(prepared["schema_version"] == "s2a.clean-d2-execution-preparation.v1", "056 preparation schema differs")
    require(Path(prepared["root"]) == ROOT and prepared["budget"] == BUDGET, "056 root/budget differs")
    require(
        registration["schema_version"] == "s2a.clean-d2-execution-registration.v1"
        and registration["prepared_ref"] == prepared_ref
        and registration["complete"] is registration["passed"] is registration["execution_authorized"] is True
        and registration["automatic_retry"] is registration["fresh_authorized"] is False
        and registration["budget"] == BUDGET,
        "Independent056 execution registration required",
    )
    metadata = read(registration["metadata_verification_ref"])
    require(
        metadata["schema_version"] == "s2a.clean-d2-execution-metadata.v1"
        and metadata["complete"] is metadata["passed"] is True
        and metadata["execution_authorized"] is False
        and metadata["prepared_ref"] == prepared_ref
        and metadata["requests_verified"] == 300
        and metadata["current_only"] is True,
        "056 independent metadata not qualified",
    )
    sources = prepared["source_sha256"]
    roles = {
        "worker_ref": "scripts/collect_wam_clean_d2.py",
        "owner_ref": "scripts/run_wam_clean_d2.py",
        "numerical_verifier_ref": "scripts/verify_wam_clean_d2.py",
    }
    for field, relative in roles.items():
        ref = registration[field]
        require(
            Path(ref["path"]) == REPO / relative and sources.get(ref["path"]) == ref["sha256"],
            "056 execution role not frozen",
        )
        checked(ref)
    verifier = metadata["verifier_ref"]
    require(
        Path(verifier["path"]) == REPO / "scripts/verify_wam_clean_d2_metadata.py"
        and sources.get(verifier["path"]) == verifier["sha256"],
        "056 metadata verifier not frozen",
    )
    for mapping in (sources, prepared["snapshot_sha256"], prepared["input_sha256"]):
        require(bool(mapping), "056 empty evidence closure")
        for path, digest in mapping.items():
            checked(dict(path=path, sha256=digest))
    require(all(str(Path(path).resolve()) in sources for path in required_sources), "056 caller outside source closure")
    require(len(prepared["weight_files"]) == 163, "056 complete checkpoint file roster required")
    for path, expected in prepared["weight_files"].items():
        stat = Path(path).stat()
        require(
            stat.st_size == expected["bytes"] and stat.st_mtime_ns == expected["mtime_ns"],
            "056 checkpoint stat changed",
        )
    require(prepared["gpu_uuid"] == GPU, "056 GPU UUID differs")
    manifest = read(prepared["current_manifest_ref"])
    validate_roster(manifest)
    head, params_ref = candidate_head(manifest["candidate_fit_ref"], manifest["candidate_verification_ref"])
    require(params_ref == manifest["candidate_params_ref"], "056 collection head differs")
    d0 = read(manifest["d0_collection_ref"])
    audit = read(prepared["d0_capture_audit_ref"])
    owner = read(prepared["d0_owner_ref"])
    require(
        d0["passed"]
        is d0["complete"]
        is audit["passed"]
        is audit["complete"]
        is owner["passed"]
        is owner["complete"]
        is True
        and audit["collection_ref"] == manifest["d0_collection_ref"]
        and audit["round_name"] == "D0"
        and audit["actual_requests"] == d0["actual_requests"] == 300
        and audit["actual_denoise_calls"] == d0["actual_denoise_calls"] == 10500
        and owner["stage"] == "D0"
        and owner["producer_ref"] == manifest["d0_collection_ref"]
        and owner["verification_ref"] == prepared["d0_capture_audit_ref"],
        "056 qualified normally completed original D0 required",
    )
    for actual, entry in zip(d0["requests"], manifest["entries"], strict=True):
        require(
            actual["complete"] is True
            and actual["initial_ref"] == entry["d0_initial_ref"]
            and actual["request_ref"] == entry["d0_request_ref"],
            "056 D0 pair references differ",
        )
        require(read(entry["d0_request_ref"]) == entry["request"], "056 actual D0 request differs")
        require(
            all(actual[k] == entry[k] for k in ("index", "name", "anchor_name", "sampling_seed")),
            "056 D0 pair identity differs",
        )
    worker_inputs = dict(
        prepared,
        training_anchors=manifest["anchors"],
        training_requests=manifest["entries"],
        noise_roster=manifest["noise_roster"],
        training_seed_plan_ref=manifest["training_seed_plan_ref"],
    )
    return worker_inputs, registration, sources, head, params_ref, d0
