"""055 admission for exactly the unexecuted suffix of the failed054 matrix."""

from pathlib import Path

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, require
from s2a.synchronous_state_dispatch import select_readout
from s2a.synchronous_state_sampling import GPU as GPU
from s2a.synchronous_state_sampling import candidate_head, paired_entries

ROOT = REPO / "outputs/synchronous_state_probe052_continuation_20261024"
BUDGET = {
    "requests": 95,
    "denoise_calls": 3325,
    "head_boundary_calls": 3325,
    "model_initializations": 1,
    "cumulative_model_initializations": 2,
    "fits": 0,
    "physics_steps": 0,
    "fresh_cases": 0,
}


def prefix_evidence(prepared):
    prefix = read(prepared["prefix_verified_ref"])
    require(
        prefix["passed"] is prefix["complete"] is prefix["inputs_unchanged_after"] is True
        and prefix["verified_nodes"] == 140
        and prefix["zero_fullpoint_nodes_verified"] == 105
        and prefix["remaining_original_indices"] == list(range(4, 99))
        and prefix["remaining_requests"] == 95
        and prefix["remaining_denoise_calls"] == 3325,
        "Exact independently verified054 prefix required",
    )
    failed = read(prefix["failed_sampling_ref"])
    require(
        failed["passed"] is failed["complete"] is False
        and failed["actual_requests"] == 4
        and failed["error"] == {"type": "KeyError", "message": "'probe052'"}
        and failed["actual_denoise_calls"] == failed["actual_head_calls"] == 140
        and failed["readout_ref"] == prepared["candidate_params_ref"],
        "054 failure/parameter boundary changed",
    )
    require(failed["parameters_before"] == prepared["prefix_parameter_identity"], "Prefix base parameters differ")
    return prefix, failed


def prefix_raw_records(prepared):
    _, failed = prefix_evidence(prepared)
    raw = failed["generations"][3]
    require(
        raw["index"] == 3 and raw["variant"] == "raw" and raw["complete"] is True,
        "055 first pair requires exact completed054 raw",
    )
    return {raw["pair_id"]: raw}


def dispatch_matrix(entries, *, zero, probe):
    """Exercise the same selection used by the actual model request loop."""
    return [select_readout(e["variant"], zero=zero, probe=probe) for e in entries]


def validate_parameter_scope(before, after, prefix):
    """Object/version identities compare within one process, never across addresses."""
    require(before == after and before["count"] > 0, "055 process-local parameter identity changed")
    require(before["count"] == prefix["count"], "055 loaded parameter count differs from prefix")


def checkpoint_stats(prepared):
    require(len(prepared["weight_files"]) == 163, "055 all163 checkpoint files required")
    for name, expected in prepared["weight_files"].items():
        stat = Path(name).stat()
        require(
            stat.st_size == expected["bytes"] and stat.st_mtime_ns == expected["mtime_ns"],
            "055 frozen read-only checkpoint file state changed",
        )


def admit(prepared_ref, registration_ref, *, required_sources):
    prepared, registered = read(prepared_ref), read(registration_ref)
    require(
        prepared["schema_version"] == "s2a.synchronous-state-continuation-preparation.v1"
        and registered["schema_version"] == "s2a.synchronous-state-continuation-registration.v1"
        and registered["prepared_ref"] == prepared_ref
        and registered["passed"] is registered["execution_authorized"] is True
        and registered["automatic_retry"] is False
        and registered["budget"] == BUDGET,
        "Independent055 registration required",
    )
    require(Path(prepared["root"]) == ROOT and prepared["budget"] == BUDGET, "055 fixed namespace/budget differs")
    meta = read(registered["metadata_verification_ref"])
    require(
        meta["passed"] is True and meta["execution_authorized"] is False and meta["prepared_ref"] == prepared_ref,
        "055 independent metadata differs",
    )
    sources = prepared["source_sha256"]
    require(
        sources.get(meta["verifier_ref"]["path"]) == meta["verifier_ref"]["sha256"], "055 metadata verifier not frozen"
    )
    for field, filename in [
        ("owner_ref", "run_wam_synchronous_state_continuation.py"),
        ("numerical_verifier_ref", "verify_wam_synchronous_state_continuation.py"),
        ("scorer_ref", "score_wam_synchronous_state.py"),
    ]:
        ref = registered[field]
        require(
            Path(ref["path"]) == REPO / "scripts" / filename and sources.get(ref["path"]) == ref["sha256"],
            "055 execution role not bound",
        )
        checked(ref)
    for mapping in [sources, prepared["snapshot_sha256"], prepared["input_sha256"]]:
        for path, digest in mapping.items():
            checked({"path": path, "sha256": digest})
    require(all(str(Path(p).resolve()) in sources for p in required_sources), "055 worker outside source closure")
    checkpoint_stats(prepared)
    prefix, failed = prefix_evidence(prepared)
    current = read(prepared["current_inputs_ref"])
    require(current["current_inputs"] == prepared["current_inputs"], "055 current-only inputs differ")
    full = paired_entries(
        prepared["current_inputs"],
        prepared["request_template"],
        prepared["zero_cases"],
        read(prepared["prerequisite_refs"]["bridge_qualification"]),
    )
    require(
        full == prepared["sampling_entries"] and prepared["remaining_indices"] == list(range(4, 99)),
        "055 original matrix changed",
    )
    marker = object()
    selected = dispatch_matrix(full[4:], zero=object(), probe=marker)
    require(
        len(selected) == 95 and selected[0] is marker and sum(x is marker for x in selected) == 48,
        "055 actual suffix dispatch differs",
    )
    candidate_head(prepared["candidate_fit_ref"], prepared["candidate_verification_ref"])
    return prepared, registered, sources


def admit_sampling(prepared, prepared_ref, registration_ref, candidate_fit_ref, candidate_verification_ref):
    require(
        candidate_fit_ref == prepared["candidate_fit_ref"]
        and candidate_verification_ref == prepared["candidate_verification_ref"],
        "055 candidate differs",
    )
    head, params = candidate_head(candidate_fit_ref, candidate_verification_ref)
    qref = prepared["prerequisite_refs"]["bridge_qualification"]
    return head, params, qref, read(qref), prepared["sampling_entries"][4:]
