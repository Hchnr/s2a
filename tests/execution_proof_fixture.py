"""Synthetic immutable proof files for CPU-only admission tests."""

import copy
import hashlib
import json
from pathlib import Path

from s2a.clean_training_seed_plan import apply_seed_plan, build_seed_plan


def reference(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return reference(path)


def training_seed_metadata(*, anchors=None, request_template=None):
    """Synthetic original63 metadata, distinct from an actual prepared dataset."""
    anchors = copy.deepcopy(anchors) if anchors is not None else [dict(name=f"anchor{i}") for i in range(150)]
    entries = []
    for index in range(300):
        seed = (1 << 40) + 1000 + index
        name = f"train_{index}"
        request = copy.deepcopy(request_template) if request_template is not None else {}
        request.update(name=name, seed=seed)
        entries.append(
            dict(
                index=index,
                name=name,
                anchor_name=anchors[index // 2]["name"],
                root_seed=(101, 102)[index % 2],
                sampling_seed=seed,
                request=request,
            )
        )
    return dict(
        training_anchors=anchors,
        training_requests=entries,
        noise_roster=dict(training_seeds=[e["sampling_seed"] for e in entries], evaluation_seeds=[42, 43]),
        noise_seed_algorithm="synthetic original63 metadata; no actual dataset identity claim",
    )


def effective_training_metadata(directory, prepared):
    """Persist the original fixture and apply its separately registered plan."""
    prepared_ref = write(directory / "original_seed_metadata.json", prepared)
    plan = build_seed_plan(prepared_ref, prepared)
    plan_ref = write(directory / "original_seed_plan.json", plan)
    return apply_seed_plan(prepared, prepared_ref, plan, plan_ref)


def seal_execution(directory, payload):
    """Model no real qualification: create typed fixtures for proof validation."""
    proposal = copy.deepcopy(payload)
    proposal.update(
        schema_version="s2a.clean-training-execution-proposal.v1",
        execution_authorized=False,
        automatic_retry=False,
        fresh_performance_authorized=False,
    )
    proposal.setdefault("readiness", {})["ready_for_execution"] = False
    prepared_path = Path(payload["prepared_ref"]["path"])
    # The separate proof-only tests deliberately use the /prepared sentinel:
    # they assert envelope equality, never training or execution admission.
    if prepared_path.is_file():
        assert reference(prepared_path) == payload["prepared_ref"]
        prepared = json.loads(prepared_path.read_text())
        plan = build_seed_plan(payload["prepared_ref"], prepared)
        proposal["training_seed_plan_ref"] = write(directory / "training_seed_plan.json", plan)
    verifier = directory / "synthetic_execution_verifier.py"
    verifier.write_text("# Fixture only; no actual execution qualification.\n")
    verifier_ref = reference(verifier)
    proposal["source_sha256"][str(verifier)] = verifier_ref["sha256"]
    proposal_ref = write(directory / "execution_proposal.json", proposal)
    verification = dict(
        schema_version="s2a.clean-training-execution-verification.v1",
        complete=True,
        passed=True,
        eligible_for_execution=True,
        prepared_ref=payload["prepared_ref"],
        proposal_ref=proposal_ref,
        verifier_ref=verifier_ref,
    )
    verification_ref = write(directory / "execution_verified.json", verification)
    admitted = copy.deepcopy(proposal)
    admitted.update(
        schema_version="s2a.clean-training-execution-registration.v1",
        execution_authorized=True,
        proposal_ref=proposal_ref,
        execution_verification_ref=verification_ref,
    )
    admitted["readiness"]["ready_for_execution"] = True
    return admitted
