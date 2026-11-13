"""Independently verify the persisted real seed-43 run and reject corruption."""

import copy

import pytest

from scripts.verification.wam.model_contracts import audit_adaptive_multiseed_real43_20261113 as audit


@pytest.fixture(scope="module")
def records():
    producer, preflight, qa = audit.read(audit.PRODUCER), audit.read(audit.PREFLIGHT), audit.read(audit.QA)
    registration = audit.read(producer["registration_ref"])
    workers, model, client = (audit.read(producer[key]) for key in ("workers_ref", "model_ref", "client_ref"))
    row = model["responses"][0]
    request, response = audit.read(row["model_request_ref"]), audit.read(row["actual_response_ref"])
    refs = registration["inputs"]["refs"]
    old_request, old_response, repeated, case = (
        audit.read(refs[key])
        for key in ("historical_request", "historical_response", "historical_repeat_response", "historical_case")
    )
    return [
        producer,
        preflight,
        qa,
        registration,
        workers,
        model,
        client,
        request,
        response,
        old_request,
        old_response,
        repeated,
        case,
    ]


def test_independent_audit_rechecks_real_saved_artifacts():
    result = audit.audit()
    assert result["complete"] is result["passed"] is result["same_image_and_task"] is True
    assert not result["control_qualified"] and result["registered_source_files_checked"] == 32
    assert result["seed"] == 43 and result["new_model_requests"] == result["new_physics_steps"] == 0
    for row in result["comparisons"].values():
        assert row["different_elements"] == row["different_active_elements"] == 400


@pytest.mark.parametrize(
    "change", ["seed", "response_seed", "unreaped", "changed_array", "physics", "qa_skip", "old_replay"]
)
def test_independent_audit_rejects_corrupt_records(records, change):
    data = copy.deepcopy(records)
    (
        producer,
        _preflight,
        qa,
        _registration,
        workers,
        _model,
        client,
        request,
        response,
        _old_request,
        _old_response,
        repeated,
        _case,
    ) = data
    if change == "seed":
        request["seed"] = 42
    elif change == "response_seed":
        response["outputs"][0]["content"]["action128"]["seed"] = 42
    elif change == "unreaped":
        workers["workers"][0]["reaped"] = False
    elif change == "changed_array":
        client["exchange"]["response"]["action"][0][0] += 1.0
    elif change == "physics":
        producer["physics_steps"] = 1
    elif change == "qa_skip":
        qa["skipped_count"] = 1
    else:
        repeated["outputs"][0]["content"]["action"][0][0] += 1.0
    with pytest.raises(ValueError):
        audit.verify(*data)
