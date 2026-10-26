"""Actual archived raw/terminal responses and cross-case/stale-input rejection."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from s2a.coupled_preview_wam_protocol import validate_preview_wam_response
from s2a.terminal_development_policy import case_order

FIXTURE = Path(__file__).parent / "fixtures/coupled_preview_wam_responses.json"


def inputs(index):
    entries = json.loads(FIXTURE.read_text())["cases"]
    value = entries[index]
    response = value["response"]
    return (
        response,
        value["request"],
        dict(
            case=case_order()[index],
            request_index=0,
            checkpoint_path=response["outputs"][0]["content"]["action128"]["checkpoint_path"],
            readout_ref=entries[1]["response"]["clean_training"]["readout_ref"],
        ),
    )


@pytest.mark.parametrize("index", [0, 1])
def test_archived_branches_preserve_generated_commands(index):
    response, request, kwargs = inputs(index)
    original = copy.deepcopy(response)
    actions, identity = validate_preview_wam_response(response, request, **kwargs)
    np.testing.assert_array_equal(actions, response["outputs"][0]["content"]["action"])
    assert identity["development_response"]["variant"] == kwargs["case"]["variant"]
    assert response == original


@pytest.mark.parametrize(
    "field,value",
    [
        ("variant", "raw"),
        ("case_index", 0),
        ("request_index", 1),
        ("global_request_index", 0),
        ("readout_ref", None),
        ("sampling_seed", 43),
    ],
)
def test_reject_cross_case_or_stale_response(field, value):
    response, request, kwargs = inputs(1)
    response["clean_training"][field] = value
    with pytest.raises(ValueError, match="identity differs"):
        validate_preview_wam_response(response, request, **kwargs)


@pytest.mark.parametrize("field", ["vision_path", "prompt", "name"])
def test_reject_different_current_input(field):
    response, request, kwargs = inputs(0)
    request[field] += "_different"
    with pytest.raises(ValueError, match="actual current request"):
        validate_preview_wam_response(response, request, **kwargs)


def test_reject_boolean_index_even_when_json_equality_accepts_it():
    response, request, kwargs = inputs(0)
    response["clean_training"]["request_index"] = False
    with pytest.raises(ValueError, match="identity differs"):
        validate_preview_wam_response(response, request, **kwargs)


def test_reject_nonzero_inactive_action_before_control():
    response, request, kwargs = inputs(1)
    response["outputs"][0]["content"]["action"][0][127] = 1
    with pytest.raises(ValueError, match="inactive slots"):
        validate_preview_wam_response(response, request, **kwargs)
