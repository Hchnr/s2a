"""D2 cannot silently alter training coverage or expose future anchor labels."""

import copy

import pytest

from s2a.clean_training_d2 import BUDGET, PARAMETER_SHA, SCHEMA, validate_roster


def manifest():
    anchors = [{"name": f"a{i}", "task": "task", "image_ref": {}} for i in range(150)]
    entries = [
        dict(
            index=i,
            name=f"r{i}",
            anchor_name=f"a{i // 2}",
            sampling_seed=1000 + i,
            request={},
            d0_initial_ref={},
            d0_request_ref={},
        )
        for i in range(300)
    ]
    return dict(
        schema_version=SCHEMA,
        anchors=anchors,
        entries=entries,
        noise_roster=dict(training_seeds=[e["sampling_seed"] for e in entries], evaluation_seeds=[42, 43]),
        request_templates={"task": {}},
        candidate_params_ref={"sha256": PARAMETER_SHA},
        budget=copy.deepcopy(BUDGET),
        execution_authorized=False,
    )


def check(value):
    # Request bytes are covered by the real preparation and existing current
    # request tests. These cases isolate roster/label-exposure guards.
    return validate_roster(value, validate_request=lambda *a, **k: None)


def test_complete_training_roster_and_no_execution_grant():
    assert check(manifest()) == {
        "anchors": 150,
        "requests": 300,
        "distinct_training_seeds": 300,
        "future_labels_read": 0,
    }


@pytest.mark.parametrize("field", ["trajectory_ref", "native_commands", "future_state128"])
def test_current_anchor_refuses_future_or_command_payload(field):
    value = manifest()
    value["anchors"][0][field] = "must never enter worker inputs"
    with pytest.raises(ValueError, match="current-only"):
        check(value)


@pytest.mark.parametrize("seed", [42, 43, 2**31, True])
def test_evaluation_noise_or_incompatible_rng_seed_rejected(seed):
    value = manifest()
    value["entries"][0]["sampling_seed"] = seed
    value["noise_roster"]["training_seeds"][0] = seed
    with pytest.raises(ValueError):
        check(value)


def test_two_noise_requests_cannot_move_to_another_anchor():
    value = manifest()
    value["entries"][1]["anchor_name"] = "a1"
    with pytest.raises(ValueError, match="two-noise anchor order"):
        check(value)


def test_another_head_cannot_impersonate_fixed_collection_policy():
    value = manifest()
    value["candidate_params_ref"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="collection policy"):
        check(value)


def test_proposal_cannot_be_toggled_into_execution_registration():
    value = manifest()
    value["execution_authorized"] = True
    with pytest.raises(ValueError, match="cannot launch"):
        check(value)
