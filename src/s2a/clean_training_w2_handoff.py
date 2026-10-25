"""Read-only completed047 W2 provenance for CPU and model metadata consumers.

This checker decodes no arrays and cannot authorize a new model launch. A
separate048 execution envelope must bind its caller and execution budget.
"""

import copy
import math

from s2a.clean_training_d1_admission import Evidence, canonical
from s2a.clean_training_w2_admission import BUDGET, PREFIX, ROOT, normal_receipts
from s2a.durable_stage_supervisor import REPO, reference, require


def complete(value, schema):
    require(
        value.get("schema_version") == schema and value.get("complete") is value.get("passed") is True,
        "048 predecessor is incomplete or wrong schema",
    )


def parameters(value):
    require(
        set(value) == {"schema_version", "coefficients", "center", "scale", "ridge"}
        and value["schema_version"] == "s2a.clean-residual-readout.v1",
        "048 W2 parameter schema differs",
    )
    require(
        isinstance(value["center"], list)
        and len(value["center"]) == 1290
        and isinstance(value["coefficients"], list)
        and len(value["coefficients"]) == 1291
        and all(isinstance(row, list) and len(row) == 10 for row in value["coefficients"]),
        "048 W2 parameter geometry differs",
    )
    scalars = [*value["center"], *(v for row in value["coefficients"] for v in row), value["scale"], value["ridge"]]
    require(
        all(type(v) in (int, float) and math.isfinite(v) for v in scalars)
        and value["scale"] > 0
        and value["ridge"] == 0.1,
        "048 W2 parameters are nonfinite or use another ridge",
    )


def qualified_w2(spec, e=None):
    """Read real047 math, independent results and actual normal child receipts."""
    e = Evidence() if e is None else e
    require((ROOT / "supervisor/result.json").is_file(), "complete047 W2 is unavailable; no STATE handoff")
    for key, path in (
        ("w2_registration_ref", ROOT / "execution/admitted.json"),
        ("w2_owner_ref", ROOT / "supervisor/result.json"),
        ("w2_fit_ref", ROOT / "fit/W2/result.json"),
        ("w2_verification_ref", ROOT / "review/W2_verified.json"),
        ("w2_params_ref", ROOT / "fit/W2/parameters.json"),
    ):
        require(path.is_file() and spec[key] == reference(path), "048 exact W2 artifact differs: " + key)
    registered = e.json(spec["w2_registration_ref"])
    complete(registered, PREFIX + "registration.v1")
    require(
        registered["execution_authorized"] is True
        and registered["stage_scope"] == "W2_only"
        and registered["budget"] == BUDGET
        and registered["automatic_retry"] is False,
        "047 authorization or budget differs",
    )
    proposal = e.json(registered["proposal_ref"])
    complete(proposal, PREFIX + "proposal.v1")
    verified_metadata = e.json(registered["independent_verification_ref"])
    complete(verified_metadata, PREFIX + "verification.v1")
    require(
        proposal["execution_authorized"] is False
        and verified_metadata["proposal_ref"] == registered["proposal_ref"]
        and verified_metadata["eligible_for_execution"] is True,
        "047 independent metadata envelope differs",
    )
    expected = copy.deepcopy(proposal)
    expected.update(
        schema_version=PREFIX + "registration.v1",
        execution_authorized=True,
        proposal_ref=registered["proposal_ref"],
        independent_verification_ref=registered["independent_verification_ref"],
    )
    require(canonical(expected) == canonical(registered), "047 exact proposal envelope differs")
    e.mapping(registered["source_sha256"])
    e.mapping(registered["snapshot_sha256"])
    verifier = verified_metadata["verifier_ref"]
    require(
        verifier == reference(REPO / "scripts/verify_wam_clean_training_w2_fit.py")
        and registered["source_sha256"].get(verifier["path"]) == verifier["sha256"],
        "047 actual metadata verifier source differs",
    )
    for key in ("prepared_ref", "execution_registration_ref"):
        require(registered[key] == spec[key], "048 original041 identity differs")
    owner = e.json(spec["w2_owner_ref"])
    complete(owner, PREFIX + "owner.v1")
    require(
        owner["stage"] == "W2"
        and owner["repair_ref"] == spec["w2_registration_ref"]
        and owner["prepared_ref"] == spec["prepared_ref"]
        and owner["execution_registration_ref"] == spec["execution_registration_ref"]
        and owner["producer_ref"] == spec["w2_fit_ref"]
        and owner["verification_ref"] == spec["w2_verification_ref"]
        and owner["automatic_retry"] is owner["fresh_performance_authorized"] is owner["d1_authorized"] is False,
        "047 complete owner provenance differs",
    )
    entries = owner["owned_commands"]
    require([v["role"] for v in entries] == ["producer", "independent"], "047 role order differs")
    require(entries[0]["owner_ref"] != entries[1]["owner_ref"], "047 roles reused one owner")
    original = e.json(spec["execution_registration_ref"])
    original_check = e.json(original["execution_verification_ref"])
    original_verifier = original_check["verifier_ref"]
    for entry in entries:
        role = entry["role"]
        owned_ref = entry["owner_ref"]
        require(owned_ref == reference(ROOT / "supervisor" / role / "result.json"), "047 role owner path differs")
        owned = e.json(owned_ref)
        complete(owned, "s2a.durable-stage-owner.v1")
        require(owned["timed_out"] is False and owned["automatic_retry"] is False, "047 role timed out or retried")
        launch = normal_receipts(e, owned["launch_ref"], owned["exit_ref"], owned["release_ref"])
        require(
            launch["command"][:2]
            == [registered["runtime"]["executable"], "scripts/execute_wam_clean_training_w2_fit.py"],
            "047 real role interpreter/entrypoint differs",
        )
        require(
            owned["stage_result_ref"] == reference(ROOT / "workers" / role / "result.json"),
            "047 role wrapper result path differs",
        )
        worker = e.json(owned["stage_result_ref"])
        complete(worker, PREFIX + "worker.v1")
        delegated = spec["w2_fit_ref"] if role == "producer" else spec["w2_verification_ref"]
        require(
            worker["repair_ref"] == spec["w2_registration_ref"]
            and worker["role"] == role
            and worker["delegate_result_ref"] == delegated
            and worker["delegate_calls_attempted"] == 1
            and worker["producer_fits"] == int(role == "producer")
            and worker["independent_reconstructions"] == int(role == "independent")
            and worker["source_before_after_exact"] is True
            and worker["runtime"] == registered["runtime"]
            and worker["d1_owner_ref"] == registered["d1_owner_ref"]
            and worker["d1_handoff_owner_ref"] == registered["d1_handoff_owner_ref"],
            "047 math worker provenance differs",
        )
        proof = worker["source_proof"]
        require(
            proof["registration_ref"] == spec["execution_registration_ref"]
            and proof["source_verifier_ref"] == original_verifier
            and proof["callback_corrected_count"] == 1
            and proof["source_guard_completed"] is True,
            "047 source-proof was not uniquely applied/restored",
        )
        require(
            all(
                worker[k] == 0
                for k in ("new_model_initializations", "new_denoise_calls", "new_physics_steps", "new_fresh_cases")
            ),
            "047 role exceeded CPU-only scope",
        )
    fit = e.json(spec["w2_fit_ref"])
    verified = e.json(spec["w2_verification_ref"])
    complete(fit, "s2a.clean-training-fit.v1")
    complete(verified, "s2a.clean-training-fit-verification.v1")
    for value in (fit, verified):
        require(
            value["stage"] == "W2"
            and value["prepared_ref"] == spec["prepared_ref"]
            and value["execution_registration_ref"] == spec["execution_registration_ref"]
            and value["actual_rows"] == 840000
            and value["params_ref"] == spec["w2_params_ref"],
            "048 W2 math identity or row denominator differs",
        )
    require(
        fit["fit_attempts"] == fit["producer_fits"] == verified["independent_reconstructions"] == 1
        and fit["rounds"] == ["D0", "D1"]
        and fit["collection_refs"] == [registered["d0_collection_ref"], registered["d1_collection_ref"]]
        and fit["capture_audit_refs"] == [registered["d0_capture_audit_ref"], registered["d1_capture_audit_ref"]]
        and verified["fit_ref"] == spec["w2_fit_ref"]
        and verified["full_training_row_reconstruction"] is True
        and verified["rtol"] == 1e-9
        and verified["atol"] == 1e-11,
        "048 full double-round W2 qualification differs",
    )
    require(
        fit["label_anchor_count"]
        == len(fit["label_reconstruction_receipts"])
        == len(verified["label_reconstruction_receipts"])
        == 150,
        "048 W2 full label roster differs",
    )
    primary, qr = fit["fit_metadata"], verified["qr_metadata"]
    require(
        primary["rows"] == qr["rows"] == 840000
        and primary["feature_columns"] == qr["feature_columns"]
        and primary["feature_columns"]
        == [*(f"hidden_{i:04d}" for i in range(1280)), *(f"c0_{i:02d}" for i in range(10))]
        and primary["ridge"] == qr["ridge"] == 0.1
        and primary["replay_verified"] is qr["replay_verified"] is True
        and primary["replay_sha256"]
        == dict(
            features=qr["replay_sha256"]["features"],
            target_clean_residual=qr["replay_sha256"]["targets"],
            weights=qr["replay_sha256"]["weights"],
        ),
        "048 independent full-row identity differs",
    )
    parameters(e.json(spec["w2_params_ref"]))
    e.finish()
    return owner
