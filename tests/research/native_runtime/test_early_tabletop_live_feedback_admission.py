"""An actual live response cannot be reused for a different branch state."""

import copy

import pytest

from scripts.simulation.runtime import prepare_early_tabletop_live_feedback as admission


def test_distinct_live_responses_are_bound_to_their_own_images():
    item = admission.load_inputs()["early_tabletop_live_feedback"]
    assert list(item["branches"]) == ["original", "pose_bounded"]
    left, right = item["branches"].values()
    assert left["state_sha256"] != right["state_sha256"]
    assert left["image_ref"]["sha256"] != right["image_ref"]["sha256"]
    assert left["response_ref"]["sha256"] != right["response_ref"]["sha256"]
    assert left["actions"].shape == right["actions"].shape == (40, 128)


def test_swapping_branch_state_or_model_output_is_rejected():
    spec = {"mode": "original", "sha256": "r", "image_sha256": "i", "state_sha256": "s"}
    image = {"mode": "original", "step": 120, "sha256": "i", "state_sha256": "s"}
    output = {
        "mode": "original",
        "step": 120,
        "response_sha256": "r",
        "guard": {"mode": "original", "image_sha256": "i", "state_sha256": "s"},
    }
    capture = {"current_endpoint": {"sim_state_sha256": "s"}}
    admission.verify_branch_binding(spec, image, output, capture)
    changed = copy.deepcopy(output)
    changed["guard"]["state_sha256"] = "other"
    with pytest.raises(ValueError):
        admission.verify_branch_binding(spec, image, changed, capture)
