"""Require full control regression and actual recovered dispatch before WAM use."""

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.tracking_control_gate import admit_tracking_control


def admit_dependency_safe_control(evidence):
    require(
        set(evidence) == {"tracking_evidence", "recovery_owner_ref"},
        "Exact guarded control admission references",
    )
    admission = admit_tracking_control(evidence["tracking_evidence"])
    sources = dict(admission["source_sha256"])
    recovery_owner = read(evidence["recovery_owner_ref"])
    require(recovery_owner["complete"] is recovery_owner["passed"] is True, "Completed recovery owner")
    owned = recovery_owner["owned_command"]
    exit_record, release = read(owned["exit"]), read(owned["release"])
    require(
        exit_record["actual_wait"] is release["reaped"] is True
        and exit_record["returncode"] == release["returncode"] == 0,
        "Actual recovered dispatch wait0/reap",
    )
    launch = read(owned["launch"])
    # Bind the actual owned command, rather than accepting a seed-only diagnostic.
    require(
        launch["command"][-1] == "scripts/probe_dependency_safe_tracking_dispatch.py",
        "Actual complete tracking transaction probe",
    )
    recovery = read(recovery_owner["result_ref"])
    require(
        recovery["complete"] is recovery["passed"] is recovery["all_prefix_endpoint_sha_equal"] is True,
        "Exact native prefix and successful dispatch",
    )
    require(
        recovery["prefix_physics_steps"] == 210
        and recovery["new_physics_steps"] == 1
        and recovery["physics_steps"] == recovery["completed_history_periods"] == 211
        and recovery["world_model_calls"] == recovery["fresh_cases"] == 0,
        "Unchanged isolated recovery evidence scope",
    )
    require(
        recovery["qp_recovery"]["route"] == "nullspace_after_dependent_blocker",
        "Recovery route actually exercised",
    )
    proof = recovery["independent_proof"]
    require(
        proof["horizon"] == 4
        and proof["max_independent_stationarity"] <= 1e-8
        and proof["max_complementarity"] <= 1e-8
        and proof["maximum_prediction_difference"] <= 1e-8,
        "Original independent preview tolerances",
    )
    checked(recovery["dispatch_ref"])
    for path, digest in read(recovery_owner["registration_ref"])["source_sha256"].items():
        require(path not in sources or sources[path] == digest, "Same qualified and recovered source versions")
        sources[path] = digest
    qualified_sources = admission["source_sha256"]
    for name in (
        "dependency_safe_qp",
        "dependency_safe_pose_preview",
        "dependency_safe_preview_executable",
        "dependency_safe_preview_runtime",
    ):
        ref = reference(REPO / f"src/s2a/{name}.py")
        require(qualified_sources.get(ref["path"]) == ref["sha256"], "Guarded implementation actually qualified")
    for path, digest in sources.items():
        checked(dict(path=path, sha256=digest))
    return dict(
        evidence=evidence,
        source_sha256=sources,
        scope="Full original control gate plus one recovered transaction; no WAM/fresh task or real-time success implied",
    )
