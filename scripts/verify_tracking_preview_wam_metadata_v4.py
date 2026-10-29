"""Independent preview WAM metadata/profile/source checks; no arrays, model or physics."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from coupled_preview_wam_physical_admission import audit_physical_control
from verify_tracking_preview_wam_development import normal_owned
from verify_wam_clean_training_capture import Evidence, require
from verify_wam_terminal_development_points import head_parameters

from s2a.artifacts import REPO
from s2a.clean_training_runtime import publish, reference
from s2a.dependency_safe_wam_admission import admit_tracking_control
from s2a.serialized_preview_recovery import recovery_evidence

ROOT = REPO / "outputs/tracking_preview_wam_development_20261027"
GPU = "GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f"
BUDGET = dict(
    model_initializations=1,
    requests=900,
    denoise_calls=31500,
    head_calls=31500,
    physics_steps=7200,
    fits=0,
    fresh_cases=0,
)
ROLES = (
    "serve_tracking_preview_wam_development.py",
    "run_tracking_preview_wam_development.py",
    "collect_tracking_preview_wam_development.py",
    "verify_tracking_preview_wam_development.py",
)


def order():
    return [
        dict(
            case_index=i,
            case_id=f"task{t}_init0_s{s}_{v}",
            task_index=t,
            record_name=f"task{t}_init0",
            sampling_seed=s,
            variant=v,
            expected_requests=75,
        )
        for i, (t, s, v) in enumerate((t, s, v) for t in range(3) for s in (42, 43) for v in ("raw", "terminal058"))
    ]


def check_profile(base, config, case, p, template_path, qualified_control):
    expected = copy.deepcopy(base)
    root = ROOT / "development/cases" / case["case_id"]
    expected.update(
        output_dir=str(root.relative_to(REPO)),
        records=[case["record_name"]],
        sampling_seeds=[case["sampling_seed"]],
        simulation_python=str(REPO / base["simulation_python"]),
        fixture=str(REPO / base["fixture"]),
        policy_template=template_path,
        cosmos_repo=p["model_config"]["cosmos_source_root"],
    )
    expected["checkpoints"] = [dict(p["model_config"]["checkpoint"], name="8400_masked", gpu_uuid=GPU, port=31375)]
    for key in (
        "arm_pose_policy",
        "arm_pose_objective",
        "source_provisional_inverse_weight",
        "arm_residual_policy",
        "arm_forward_parameter_ref",
        "arm_forward_qualification_ref",
        "pose_preview_chain_ref",
        "response_parameters_ref",
        "planner_preflight_ref",
        "gripper_velocity_envelope",
    ):
        expected[key] = qualified_control[key]
    expected.update(
        gripper_preview_steps=0,
        inverse_weight=qualified_control["source_provisional_inverse_weight"],
        preview_control_closure_ref=p["preview_control_closure_ref"],
        tracking_admission=p["tracking_admission"],
        online_max_steps=600,
        online_chunk_size=40,
        max_observation_interval_steps=8,
        stable_success_last_steps=40,
    )
    require(
        {k: v for k, v in config.items() if k != "notes"} == {k: v for k, v in expected.items() if k != "notes"},
        "061 whole fixed control profile differs",
    )


def verify(pref):
    e = Evidence()
    p = e.json(pref)
    admit_tracking_control(p["tracking_admission"])
    require(
        p["schema_version"] == "s2a.tracking-preview-wam-development-preparation.v1"
        and Path(p["root"]) == ROOT
        and p["budget"] == BUDGET
        and p["execution_authorized"] is p["fresh_authorized"] is p["automatic_retry"] is False
        and p["case_order"] == order(),
        "061 fixed preparation differs",
    )
    require(p["recovery"] == recovery_evidence(), "Serialization recovery provenance differs")
    for ref in p["recovery"]["refs"].values():
        e.path(ref)
        require(p["input_sha256"].get(ref["path"]) == ref["sha256"], "Recovery input omitted")
    physical = audit_physical_control(e, p["preview_control_closure_ref"])
    require(p["physical_control_admission"] == physical, "Actual18/4907 physical admission differs")
    qualified_control = e.json(physical["control_config_ref"])
    for key in ("source_sha256", "snapshot_sha256", "input_sha256"):
        require(bool(p[key]), "061 empty closure")
        e.mapping(p[key])
    for relative in [
        *("scripts/" + name for name in ROLES),
        "scripts/verify_tracking_preview_wam_metadata.py",
        "scripts/verify_wam_terminal_development_points.py",
        "scripts/tracking_preview_wam_audit_admission.py",
    ]:
        ref = reference(REPO / relative)
        require(p["source_sha256"].get(ref["path"]) == ref["sha256"], "061 audit/role source not frozen")
    require(
        p["state_owner_ref"]
        == dict(
            path=str(REPO / "outputs/terminal_identity_state_validation_20261025/orchestration/result.json"),
            sha256="af5d937045eed71de29c5863249c7aed6168215546e956b0e3c20c39ebed578b",
        ),
        "Actual060 completed owner required",
    )
    owner = e.json(p["state_owner_ref"])
    num = e.json(owner["numerical_ref"])
    score = e.json(owner["state_score_ref"])
    state = e.json(owner["prepared_ref"])
    require(
        all(d["passed"] is d["complete"] is True for d in (owner, num, score))
        and score["state_gate_passed"] is score["eligible_for_development"] is True
        and num["actual_requests"] == 99
        and num["actual_nodes"] == 3465
        and num["numerical_contract_passed"] is num["lifecycle_passed"] is True,
        "061 complete actual060 qualification required",
    )
    require(len(owner["owned_commands"]) == 3, "060 normal role count")
    for i, command in enumerate(owner["owned_commands"]):
        normal_owned(e, command, GPU if i == 0 else "")
    for key in ("model_config", "prerequisite_refs", "weight_files", "noise_roster"):
        require(p[key] == state[key], "061 changed qualified model field: " + key)
    require(
        p["candidate_params_ref"] == score["candidate_params_ref"]
        and p["candidate_proof_ref"] == score["candidate_proof_ref"],
        "061 candidate lineage differs",
    )
    proof = e.json(p["candidate_proof_ref"])
    require(
        proof["passed"] is proof["complete"] is True
        and proof["params_ref"] == p["candidate_params_ref"]
        and proof["position_relative_gradient_residual"] <= 1e-7
        and -1e-10 <= proof["opening_gap"] <= 2e-4,
        "058 independent numerical proof required",
    )
    require(
        p["baseline_params_ref"]["sha256"] == "ca01b9d8a49f2f1585c917f492e52a0a89cdb0b447f6bd2beee050ed0d2fccc0",
        "Original052 baseline required",
    )
    old, new = (head_parameters(e.json(ref)) for ref in (p["baseline_params_ref"], p["candidate_params_ref"]))
    require(
        np.array_equal(old[0][:, 3:9], new[0][:, 3:9]) and np.array_equal(old[1], new[1]) and old[2] == new[2],
        "061 frozen rotation/feature geometry differs",
    )
    relocation = e.json(p["tool_relocation_ref"])
    old_template = e.json(relocation["old_template_ref"])
    new_template = e.json(relocation["new_template_ref"])
    require(
        new_template == dict(old_template, tool_contract=relocation["safe_copy_ref"]["path"])
        and relocation["safe_copy_ref"]["sha256"] == relocation["historical_sha256"],
        "061 changed control template beyond registered relocation",
    )
    e.path(relocation["safe_copy_ref"])
    base = e.json(
        dict(
            path=str(REPO / "configs/wam_short_replan_8_alpha1_20261021.json"),
            sha256="51f64aa81a53f89cd91f8cf9925efe3e3c9a3a8de13a1856b054a37693dc979b",
        )
    )
    preflight = e.json(p["input_preflight_ref"])
    require(
        preflight["passed"] is preflight["complete"] is True
        and preflight["case_count"] == 12
        and [row["case"] for row in preflight["records"]] == order()
        and preflight["actual_model_calls"] == preflight["actual_physics_steps"] == preflight["array_decodes"] == 0,
        "Actual all12 sim input preflight required",
    )
    for row, case in zip(preflight["records"], order(), strict=True):
        config = e.json(row["config_ref"])
        check_profile(base, config, case, p, relocation["new_template_ref"]["path"], qualified_control)
        require(
            all(p["input_sha256"].get(k) == v for k, v in row["input_sha256"].items()), "061 preflight input omitted"
        )
        e.mapping(row["input_sha256"])
    require(len(p["weight_files"]) == 163, "061 complete checkpoint roster required")
    for path, expected in p["weight_files"].items():
        stat = Path(path).stat()
        require(
            stat.st_size == expected["bytes"] and stat.st_mtime_ns == expected["mtime_ns"], "061 checkpoint changed"
        )
    old_registration = e.json(p["legacy_resource_registration_ref"])
    old_prepared = e.json(old_registration["prepared_ref"])
    require(p["resources"] == old_prepared["resources"], "061 original explicit resource budget changed")
    qa = e.json(p["qa_ref"])
    tests = sorted(
        {
            "tests/test_coupled_preview_control_gate.py",
            "tests/test_preview_result_publication.py",
            "tests/test_serialized_preview_derivation.py",
            "tests/test_tracking_wam_transport.py",
            "tests/test_tracking_preview_transaction.py",
        }
        | {
            str(f.relative_to(REPO))
            for pattern in (
                "test_serialized_preview_wam_*.py",
                "test_coupled_preview_wam_protocol.py",
                "test_coupled_pose_preview*.py",
            )
            for f in (REPO / "tests").glob(pattern)
        }
    )
    require(
        qa["passed"] is qa["complete"] is True
        and qa["returncode"] == 0
        and qa["command"] == [str(REPO / ".venv/bin/python"), "-m", "pytest", "-q", *tests],
        "061 actual complete QA command differs",
    )
    for ref in qa["source_refs"]:
        e.path(ref)
        require(p["source_sha256"].get(ref["path"]) == ref["sha256"], "061 QA source changed")
    e.finish()
    return dict(
        schema_version="s2a.tracking-preview-wam-development-metadata-review.v2",
        passed=True,
        complete=True,
        prepared_ref=pref,
        verifier_ref=reference(Path(__file__)),
        base_verifier_ref=reference(REPO / "scripts/verify_tracking_preview_wam_metadata.py"),
        registration_compatible=False,
        cases_verified=12,
        current_only=True,
        physical_control_admission=physical,
        profile_scope="fixed input profiles before execution-reference binding; not executable configs",
        execution_authorized=False,
        fresh_authorized=False,
        input_sha256=e.inputs,
        new_model_initializations=0,
        physics_steps=0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            publish(args.prepared.resolve().parent / "metadata_review_v2.json", verify(reference(args.prepared)))
        ),
        flush=True,
    )
