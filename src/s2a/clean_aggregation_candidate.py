"""Admit056 aggregate parameters by their own complete fit/QR/owner identity.

Numerical admission is not validation performance or permission to use fresh.
"""

from s2a.clean_training_aggregation import RIDGE, ROWS, normal_exit
from s2a.clean_training_runtime import read, require
from s2a.wam_clean_readout import CleanResidualReadout


def candidate_head(owner_ref, *, read_json=read):
    owner = read_json(owner_ref)
    require(
        owner["schema_version"] == "s2a.clean-aggregation-owner.v1"
        and owner["passed"] is owner["complete"] is True
        and owner["candidate_numerically_qualified"] is True,
        "056 full aggregate owner required",
    )
    require(
        owner["fresh_authorized"] is owner["validation_authorized"] is owner["automatic_retry"] is False
        and len(owner["owned_commands"]) == 2,
        "056 numerical owner cannot grant evaluation permission",
    )
    for entry in owner["owned_commands"]:
        normal_exit(entry, read_json)
    produced, verified = read_json(owner["producer_ref"]), read_json(owner["verification_ref"])
    require(
        produced["schema_version"] == "s2a.clean-aggregation-fit.v1"
        and verified["schema_version"] == "s2a.clean-aggregation-qr.v1"
        and all(v["complete"] is v["passed"] is True for v in (produced, verified)),
        "056 actual aggregate producer/QR required",
    )
    require(
        produced["registration_ref"] == owner["registration_ref"]
        and verified["producer_ref"] == owner["producer_ref"]
        and verified["params_ref"] == produced["params_ref"] == owner["params_ref"]
        and produced["producer_fits"] == verified["independent_reconstructions"] == 1,
        "056 aggregate parameter or attempt identity differs",
    )
    require(
        produced["collection_refs"] == verified["collection_refs"]
        and len(produced["collection_refs"]) == 3
        and produced["d2_owner_ref"] == verified["d2_owner_ref"],
        "056 aggregate data identities differ",
    )
    require(
        produced["fit_metadata"]["rows"] == verified["qr_metadata"]["rows"] == ROWS
        and verified["rtol"] == 1e-9
        and verified["atol"] == 1e-11
        and set(verified["comparisons"]) == {"coefficients", "center", "scale"}
        and all(v["passed"] is True for v in verified["comparisons"].values()),
        "056 full aggregate QR qualification differs",
    )
    a, b = produced["fit_metadata"]["replay_sha256"], verified["qr_metadata"]["replay_sha256"]
    require(
        (a["features"], a["target_clean_residual"], a["weights"]) == (b["features"], b["targets"], b["weights"]),
        "056 aggregate full row streams differ",
    )
    params = read_json(owner["params_ref"])
    require(params["ridge"] == RIDGE and len(params["center"]) == 1290, "056 aggregate layout/ridge differs")
    return CleanResidualReadout.from_parameters(params), owner["params_ref"]
