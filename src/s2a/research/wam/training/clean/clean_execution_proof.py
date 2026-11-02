"""Metadata-only proof for proposal -> independent audit -> execution envelope.

Separating these three immutable files avoids a circular manifest SHA. This
module contains no model, label, normalization or scoring implementation.
"""

import copy


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_envelope(registration, proposal, verification, *, prepared_ref):
    require(
        proposal.get("schema_version") == "s2a.clean-training-execution-proposal.v1"
        and proposal.get("complete") is True
        and proposal.get("passed") is True
        and proposal.get("prepared_ref") == prepared_ref
        and proposal.get("readiness", {}).get("ready_for_execution") is False
        and proposal.get("execution_authorized") is False
        and proposal.get("automatic_retry") is False
        and proposal.get("fresh_performance_authorized") is False,
        "execution proposal is incomplete, unrelated or already claims execution admission",
    )
    require(
        verification.get("schema_version") == "s2a.clean-training-execution-verification.v1"
        and verification.get("complete") is True
        and verification.get("passed") is True
        and verification.get("eligible_for_execution") is True
        and verification.get("prepared_ref") == prepared_ref
        and verification.get("proposal_ref") == registration.get("proposal_ref"),
        "independent execution audit does not admit this exact proposal",
    )
    expected = copy.deepcopy(proposal)
    expected.update(
        schema_version="s2a.clean-training-execution-registration.v1",
        execution_authorized=True,
        proposal_ref=registration.get("proposal_ref"),
        execution_verification_ref=registration.get("execution_verification_ref"),
    )
    expected["readiness"]["ready_for_execution"] = True
    require(registration == expected, "execution envelope changed the independently audited proposal payload")
    verifier = verification["verifier_ref"]
    require(
        registration["source_sha256"].get(verifier["path"]) == verifier["sha256"],
        "independent execution verifier is outside the frozen source closure",
    )
    return verifier


def check_execution_proof(registration, *, prepared_ref, read_json, check_file):
    """Use the caller's evidence tracker for every proof file and verifier SHA."""
    proposal = read_json(registration["proposal_ref"])
    verification = read_json(registration["execution_verification_ref"])
    verifier = validate_envelope(registration, proposal, verification, prepared_ref=prepared_ref)
    check_file(verifier)
    return verification
