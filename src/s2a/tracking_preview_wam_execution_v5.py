"""Separate preview WAM registration; never reuse old061 execution authorization."""

from pathlib import Path

from s2a.artifacts import REPO
from s2a.clean_training_runtime import GPU, checked, read, require
from s2a.coupled_preview_control_gate import admit_preview_control
from s2a.release_floor_wam_admission import admit_tracking_control
from s2a.terminal_development_execution import BUDGET, CONTROL_REF, admit_service_state
from s2a.terminal_development_gate import admit_state
from s2a.terminal_development_policy import case_order

ROOT = REPO / "outputs/tracking_preview_wam_development_v5_20261028"
PORT = 31375
MASTER_PORT = 31376
ROLES = dict(
    service="serve_tracking_preview_wam_development_v5.py",
    owner="run_tracking_preview_wam_development_v5.py",
    client="collect_tracking_preview_wam_development_v5.py",
    verifier="verify_tracking_preview_wam_development_v5.py",
)


def admit_execution(prepared_ref, registration_ref, *, required_sources):
    prepared, registration = read(prepared_ref), read(registration_ref)
    require(
        prepared["schema_version"] == "s2a.tracking-preview-wam-v5-development-preparation.v1"
        and Path(prepared["root"]) == ROOT
        and prepared["budget"] == BUDGET
        and prepared["config"]["gpu_uuid"] == GPU
        and prepared["case_order"] == case_order(),
        "Fixed preview WAM preparation required",
    )
    require(
        registration["schema_version"] == "s2a.tracking-preview-wam-v5-development-registration.v1"
        and all(registration[k] is True for k in ("complete", "passed", "execution_authorized"))
        and registration["fresh_authorized"] is registration["automatic_retry"] is False
        and registration["prepared_ref"] == prepared_ref
        and registration["budget"] == BUDGET
        and registration["case_order"] == case_order()
        and registration["development"] == dict(port=PORT, master_port=MASTER_PORT, case_order=case_order()),
        "Separate complete preview WAM execution registration required",
    )
    admit_tracking_control(prepared["tracking_admission"])
    require(
        registration["tracking_admission"] == prepared["tracking_admission"], "Tracking prerequisite binding differs"
    )
    control = admit_preview_control(prepared["preview_control_closure_ref"])
    require(
        registration["preview_control_closure_ref"] == prepared["preview_control_closure_ref"]
        and registration["physical_control_admission"] == control,
        "Actual physical qualification differs",
    )
    _, state = admit_state(prepared["state_owner_ref"])
    require(
        registration["state_admission"] == state and prepared["candidate_params_ref"] == state["candidate_params_ref"],
        "Actual060 state qualification differs",
    )
    metadata = read(registration["metadata_verification_ref"])
    require(
        metadata["schema_version"] == "s2a.tracking-preview-wam-v5-development-metadata.v1"
        and metadata["complete"] is metadata["passed"] is True
        and metadata["prepared_ref"] == prepared_ref
        and metadata["cases_verified"] == 12
        and metadata["execution_authorized"] is False,
        "Independent complete preview metadata required",
    )
    sources = prepared["source_sha256"]
    require(registration["source_sha256"] == sources, "Execution source roster differs from preparation")
    for mapping in (sources, prepared["snapshot_sha256"], prepared["input_sha256"]):
        require(bool(mapping), "Nonempty source/input closure required")
        for path, digest in mapping.items():
            checked(dict(path=path, sha256=digest))
    require(all(str(Path(p).resolve()) in sources for p in required_sources), "Caller source not frozen")
    require(set(registration["roles"]) == set(ROLES), "Complete preview role roster required")
    for role, name in ROLES.items():
        ref = registration["roles"][role]
        require(
            Path(ref["path"]) == REPO / "scripts" / name and sources.get(ref["path"]) == ref["sha256"],
            "Preview role/source differs",
        )
        checked(ref)
    require(registration["service_source_ref"] == registration["roles"]["service"], "Service binding differs")
    verifier = metadata["verifier_ref"]
    require(
        Path(verifier["path"]) == REPO / "scripts/verify_tracking_preview_wam_metadata_revision5.py"
        and sources.get(verifier["path"]) == verifier["sha256"],
        "Metadata verifier source differs",
    )
    checked(verifier)
    require(len(prepared["weight_files"]) == 163, "Complete checkpoint roster required")
    for path, expected in prepared["weight_files"].items():
        stat = Path(path).stat()
        require(stat.st_size == expected["bytes"] and stat.st_mtime_ns == expected["mtime_ns"], "Checkpoint changed")
    return prepared, registration, sources


__all__ = ["BUDGET", "MASTER_PORT", "PORT", "ROLES", "ROOT", "admit_execution", "admit_service_state"]


def admit_development(prepared_ref, registration_ref, state_owner_ref, state_verification_ref, *, required_sources):
    prepared, registration, sources = admit_execution(prepared_ref, registration_ref, required_sources=required_sources)
    admission, _, _ = admit_service_state(
        prepared, prepared_ref, registration_ref, state_owner_ref, state_verification_ref
    )
    require(registration["development_control_ref"] == CONTROL_REF, "Qualified native preload reference differs")
    control = read(CONTROL_REF)
    require(control["passed"] is control["complete"] is True, "Native dependency qualification required")
    guard = control["cv2_guard_registration"]
    for ref in (guard["helper_ref"], guard["exact_probe_ref"], *guard["runtime_source_refs"]):
        checked(ref)
        require(
            prepared["input_sha256"].get(ref["path"], sources.get(ref["path"])) == ref["sha256"],
            "Native preload/probe not frozen",
        )
    return prepared, registration, sources, admission, control
