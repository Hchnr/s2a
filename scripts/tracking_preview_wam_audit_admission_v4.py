"""Independent preview WAM completed-run lineage and lifecycle checks, JSON only."""

from pathlib import Path

from coupled_preview_wam_physical_admission import audit_physical_control

from s2a.artifacts import REPO
from s2a.dependency_safe_wam_admission import admit_tracking_control

ROOT = REPO / "outputs/tracking_preview_wam_development_v4_20261027"
CHECK_NAMES = {
    "short_joint_improvement",
    "strictly_beats_hold_and_constant_velocity",
    "all_components_protected",
    "at_least_five_episodes",
    "no_episode_short_outlier",
    "both_seeds_protected",
}
COMPONENT_NAMES = {
    f"h{h}_{metric}" for h in (8, 16, 40) for metric in ("position_rmse_m", "rotation_rmse_rad", "opening_mae")
}


def admission(ev, prepared_ref, registration_ref, run_ref, *, require, normal_owned, gpu, order):
    p, r, run = (ev.json(ref) for ref in (prepared_ref, registration_ref, run_ref))
    require(
        p["schema_version"] == "s2a.tracking-preview-wam-v4-development-preparation.v1"
        and Path(p["root"]) == ROOT
        and r["schema_version"] == "s2a.tracking-preview-wam-v4-development-registration.v1"
        and r["passed"] is r["complete"] is r["execution_authorized"] is True
        and r["prepared_ref"] == prepared_ref
        and r["automatic_retry"] is r["fresh_authorized"] is False
        and r["development"] == dict(port=31375, master_port=31376, case_order=order()),
        "061 independent execution identity differs",
    )
    admit_tracking_control(p["tracking_admission"])
    require(r["tracking_admission"] == p["tracking_admission"], "Tracking admission binding differs")
    physical = audit_physical_control(ev, p["preview_control_closure_ref"])
    require(
        r["preview_control_closure_ref"] == p["preview_control_closure_ref"]
        and r["physical_control_admission"] == physical
        and run["preview_control_closure_ref"] == p["preview_control_closure_ref"],
        "Full preview physical prerequisite lineage differs",
    )
    require(
        r["source_sha256"] == p["source_sha256"]
        and r["service_source_ref"] == r["roles"]["service"]
        and r["case_order"] == order(),
        "Preview source/service/case binding differs",
    )
    metadata = ev.json(r["metadata_verification_ref"])
    require(
        metadata["schema_version"] == "s2a.tracking-preview-wam-v4-development-metadata.v1"
        and metadata["passed"] is metadata["complete"] is True
        and metadata["cases_verified"] == 12
        and metadata["prepared_ref"] == prepared_ref
        and metadata["execution_authorized"] is False,
        "061 independent metadata qualification required",
    )
    for mapping in (p["source_sha256"], p["snapshot_sha256"], p["input_sha256"]):
        require(bool(mapping), "061 empty source closure")
        ev.mapping(mapping)
    for role, filename in (
        ("service", "serve_tracking_preview_wam_development_v4.py"),
        ("owner", "run_tracking_preview_wam_development_v4.py"),
        ("client", "collect_tracking_preview_wam_development_v4.py"),
        ("verifier", "verify_tracking_preview_wam_development_v4.py"),
    ):
        ref = r["roles"][role]
        require(
            Path(ref["path"]) == REPO / "scripts" / filename and p["source_sha256"].get(ref["path"]) == ref["sha256"],
            "061 role source differs",
        )
        ev.path(ref)
    require(
        Path(metadata["verifier_ref"]["path"]) == REPO / "scripts/verify_tracking_preview_wam_metadata_revision4.py",
        "061 exact independent metadata verifier required",
    )
    ev.path(metadata["verifier_ref"])
    require(
        p["source_sha256"].get(metadata["verifier_ref"]["path"]) == metadata["verifier_ref"]["sha256"],
        "061 metadata verifier not frozen",
    )
    require(
        run["schema_version"] == "s2a.tracking-preview-wam-v4-development-owner.v1"
        and run["passed"] is run["complete"] is True
        and run["prepared_ref"] == prepared_ref
        and run["execution_registration_ref"] == registration_ref
        and run["automatic_retry"] is False,
        "061 complete production required",
    )
    state_owner, state = (ev.json(run[k]) for k in ("state_owner_ref", "state_verification_ref"))
    score = ev.json(state_owner["state_score_ref"])
    require(
        run["state_owner_ref"] == p["state_owner_ref"]
        and Path(run["state_owner_ref"]["path"])
        == REPO / "outputs/terminal_identity_state_validation_20261025/orchestration/result.json",
        "061 exact060 owner lineage differs",
    )
    require(
        score["prepared_ref"] == state["prepared_ref"] == state_owner["prepared_ref"], "061 state preparation differs"
    )
    require(
        state_owner["schema_version"] == "s2a.terminal-identity-state-owner.v1"
        and state["schema_version"] == "s2a.terminal-identity-state-audit.v1"
        and all(v["passed"] is v["complete"] is True for v in (state_owner, state, score))
        and state_owner["numerical_ref"] == run["state_verification_ref"]
        and score["numerical_ref"] == run["state_verification_ref"]
        and score["state_gate_passed"] is score["state_gate"]["passed"] is score["eligible_for_development"] is True
        and set(score["state_gate"]["checks"]) == CHECK_NAMES
        and all(v is True for v in score["state_gate"]["checks"].values())
        and set(score["state_gate"]["component_checks"]) == COMPONENT_NAMES
        and all(v is True for v in score["state_gate"]["component_checks"].values())
        and state["actual_requests"] == 99
        and state["actual_nodes"] == 3465
        and state["paired_initial_conditions_verified"] == 48
        and state["zero_nodes_verified"] == 105
        and state["fresh_cases"] == 0
        and sum(v is True for v in score["state_gate"]["episode_passes"].values()) >= 5
        and state["numerical_contract_passed"] is state["lifecycle_passed"] is True,
        "061 actual060 state gate must pass",
    )
    require(len(state_owner["owned_commands"]) == 3, "060 three normal roles required")
    for i, command in enumerate(state_owner["owned_commands"]):
        normal_owned(ev, command, gpu if i == 0 else "")
    require(
        run["readout_ref"] == score["candidate_params_ref"] == p["candidate_params_ref"]
        and run["candidate_proof_ref"] == score["candidate_proof_ref"] == p["candidate_proof_ref"]
        and run["state_score_ref"] == state_owner["state_score_ref"],
        "061 candidate lineage differs",
    )
    require(
        p["baseline_params_ref"]["sha256"] == "ca01b9d8a49f2f1585c917f492e52a0a89cdb0b447f6bd2beee050ed0d2fccc0",
        "Original052 baseline required",
    )
    control = ev.json(r["development_control_ref"])
    require(
        r["development_control_ref"]["sha256"] == "106f45543f8bc0aa0e336c0ad9b5c01e0bce252a5e5dad0ec3ebbba349e5fda0"
        and control["passed"] is control["complete"] is control["control_contract_reuse_eligible"] is True
        and control["new_physics_control_qualification"] is False,
        "061 original conditional control qualification required",
    )
    relocation = ev.json(p["tool_relocation_ref"])
    old, new = ev.json(relocation["old_template_ref"]), ev.json(relocation["new_template_ref"])
    expected = dict(old, tool_contract=relocation["safe_copy_ref"]["path"])
    require(
        new == expected and relocation["safe_copy_ref"]["sha256"] == relocation["historical_sha256"],
        "061 template changed beyond contract path",
    )
    ev.path(relocation["safe_copy_ref"])
    require(
        len(run["cases"]) == len(run["owned_commands"]) == 12 and [c["case"] for c in run["cases"]] == order(),
        "061 full12 cases required",
    )
    normal_owned(ev, run["service_owned_command"], gpu)
    for ref in (state_owner["gpu_after_ref"], run["gpu_after_ref"]):
        released = ev.json(ref)
        require(
            released["released"] is True
            and released["uuid"] == gpu
            and released["memory_mib"] == 0
            and released["compute_apps"] == [],
            "061 actual GPU release required",
        )
    require(len(p["weight_files"]) == 163, "061 complete checkpoint roster required")
    for path, expected_stat in p["weight_files"].items():
        stat = Path(path).stat()
        require(
            stat.st_size == expected_stat["bytes"] and stat.st_mtime_ns == expected_stat["mtime_ns"],
            "061 checkpoint changed",
        )
    return p, r, run, score, control, p["source_sha256"]
