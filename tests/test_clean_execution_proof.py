"""An independent audit must bind every unchanged execution payload field."""

import copy
import json
from pathlib import Path

import pytest
from execution_proof_fixture import reference, seal_execution

from s2a.clean_execution_proof import check_execution_proof


def read(ref):
    assert reference(Path(ref["path"])) == ref
    return json.loads(Path(ref["path"]).read_text())


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "missing_proof",
        "other_proposal",
        "failed_audit",
        "head_changed",
        "budget_changed",
        "verifier_unfrozen",
        "proposal_ready",
        "proposal_executed",
    ],
)
def test_exact_independent_proof_and_payload_required(tmp_path, fault):
    payload = dict(
        complete=True,
        passed=True,
        prepared_ref=dict(path="/prepared", sha256="a" * 64),
        source_sha256={},
        budget=dict(requests=1599),
        frozen_candidate="W2",
    )
    reg = seal_execution(tmp_path, payload)
    before = copy.deepcopy(reg)
    if fault == "missing_proof":
        reg.pop("execution_verification_ref")
    elif fault == "other_proposal":
        reg["proposal_ref"] = reg["execution_verification_ref"]
    elif fault in ("failed_audit", "verifier_unfrozen"):
        path = Path(reg["execution_verification_ref"]["path"])
        value = json.loads(path.read_text())
        value["eligible_for_execution"] = False if fault == "failed_audit" else True
        if fault == "verifier_unfrozen":
            value["verifier_ref"]["sha256"] = "0" * 64
        path.write_text(json.dumps(value))
        reg["execution_verification_ref"] = reference(path)
    elif fault == "head_changed":
        reg["frozen_candidate"] = "W1"
    elif fault == "budget_changed":
        reg["budget"]["requests"] = 1600
    elif fault in ("proposal_ready", "proposal_executed"):
        path = Path(reg["proposal_ref"]["path"])
        value = json.loads(path.read_text())
        if fault == "proposal_ready":
            value["readiness"]["ready_for_execution"] = True
        else:
            value["execution_authorized"] = True
        path.write_text(json.dumps(value))
        reg["proposal_ref"] = reference(path)
    consumed = []

    def checked(ref):
        consumed.append(ref)
        return ref

    if fault is None:
        result = check_execution_proof(reg, prepared_ref=payload["prepared_ref"], read_json=read, check_file=checked)
        assert result["eligible_for_execution"] is True
        assert reg == before and consumed == [result["verifier_ref"]]
    else:
        with pytest.raises((ValueError, KeyError)):
            check_execution_proof(reg, prepared_ref=payload["prepared_ref"], read_json=read, check_file=checked)
