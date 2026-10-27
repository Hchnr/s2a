"""Require complete060 state performance and normal lifecycle before061 prep."""

from pathlib import Path

from s2a.artifacts import REPO
from s2a.clean_training_aggregation import normal_exit
from s2a.clean_training_runtime import GPU, read, require
from s2a.terminal_candidate import candidate_head

ROOT = REPO / "outputs/terminal_identity_state_validation_20261025"
CHECKS = frozenset(
    (
        "short_joint_improvement",
        "strictly_beats_hold_and_constant_velocity",
        "all_components_protected",
        "at_least_five_episodes",
        "no_episode_short_outlier",
        "both_seeds_protected",
    )
)
COMPONENTS = frozenset(
    f"h{h}_{m}" for h in (8, 16, 40) for m in ("position_rmse_m", "rotation_rmse_rad", "opening_mae")
)


def admit_state(owner_ref, *, read_json=read):
    require(Path(owner_ref["path"]) == ROOT / "orchestration/result.json", "Fixed060 owner required")
    owner = read_json(owner_ref)
    require(
        owner["schema_version"] == "s2a.terminal-identity-state-owner.v1"
        and owner["passed"]
        is owner["complete"]
        is owner["state_gate_passed"]
        is owner["eligible_for_development"]
        is True
        and owner["automatic_retry"] is owner["fresh_authorized"] is False
        and len(owner["owned_commands"]) == 3,
        "Complete060 performance-qualified owner required",
    )
    for role, entry in zip(("model", "independent", "scoring"), owner["owned_commands"], strict=True):
        require(
            Path(entry["launch"]["path"]) == ROOT / "orchestration" / role / "launch.json",
            "060 actual owned role identity differs",
        )
        normal_exit(entry, read_json)
    gpu = read_json(owner["gpu_after_ref"])
    require(
        gpu["released"] is True
        and gpu["uuid"] == GPU
        and gpu["memory_mib"] == 0
        and gpu["compute_apps"] == []
        and gpu["device_minor"] == 6
        and gpu["visible_gpu"] == 2,
        "Actual060 allowed GPU release required",
    )
    numerical = read_json(owner["numerical_ref"])
    score = read_json(owner["state_score_ref"])
    require(
        numerical["schema_version"] == "s2a.terminal-identity-state-audit.v1"
        and all(numerical[k] is True for k in ("passed", "complete", "numerical_contract_passed", "lifecycle_passed"))
        and numerical["actual_requests"] == 99
        and numerical["actual_nodes"] == 3465
        and numerical["zero_nodes_verified"] == 105
        and numerical["paired_initial_conditions_verified"] == 48
        and numerical["fresh_cases"] == 0
        and numerical["validation_future_archive_reads"] == 0,
        "Full060 numerical/no-future qualification required",
    )
    model = owner["owned_commands"][0]
    require(
        all(numerical[f"worker_{k}_ref"] == model[k] for k in ("launch", "exit", "release"))
        and numerical["gpu_after_ref"] == owner["gpu_after_ref"],
        "060 model receipt identities differ",
    )
    require(
        score["schema_version"] == "s2a.terminal-identity-state-score.v1"
        and score["passed"]
        is score["complete"]
        is score["state_gate_passed"]
        is score["eligible_for_development"]
        is True
        and score["numerical_ref"] == owner["numerical_ref"]
        and score["prepared_ref"] == numerical["prepared_ref"] == owner["prepared_ref"],
        "Actual060 score/numerical/owner lineage differs",
    )
    gate = score["state_gate"]
    require(
        gate["passed"] is True
        and set(gate["checks"]) == CHECKS
        and all(v is True for v in gate["checks"].values())
        and set(gate["component_checks"]) == COMPONENTS
        and all(v is True for v in gate["component_checks"].values())
        and len(gate["episode_passes"]) == 6
        and sum(v is True for v in gate["episode_passes"].values()) >= 5,
        "Original complete060 state gate must pass; no legacy exception",
    )
    prepared = read_json(owner["prepared_ref"])
    require(
        Path(prepared["root"]) == ROOT
        and score["candidate_proof_ref"] == numerical["candidate_proof_ref"] == prepared["candidate_proof_ref"]
        and score["candidate_params_ref"] == numerical["candidate_params_ref"] == prepared["candidate_params_ref"],
        "060 fixed candidate lineage differs",
    )
    policy, params_ref = candidate_head(prepared["candidate_proof_ref"], read_json=read_json)
    require(params_ref == prepared["candidate_params_ref"], "058 candidate parameter identity differs")
    return policy, dict(
        state_owner_ref=owner_ref,
        state_verification_ref=owner["numerical_ref"],
        state_score_ref=owner["state_score_ref"],
        state_prepared_ref=owner["prepared_ref"],
        candidate_proof_ref=prepared["candidate_proof_ref"],
        candidate_params_ref=params_ref,
        development_preconditions_passed=True,
        execution_authorized=False,
        fresh_authorized=False,
    )
