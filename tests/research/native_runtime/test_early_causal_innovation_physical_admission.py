"""One-period feedback must bind the checked early static result."""

import copy

import pytest

from scripts.simulation.runtime import prepare_early_causal_innovation_physical as admission
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix


def test_one_period_scope_binds_six_references_and_42_completed_periods():
    inputs = admission.load_inputs()
    assert inputs["feedback_config"]["max_control_steps_per_phase"] == 123
    assert len(inputs["records"]) == 6
    assert [row["step"] for row in inputs["prior_periods"][:42]] == list(range(80, 122))


def test_unverified_static_candidate_is_not_admitted():
    config = admission.FIXED
    saved = prefix.read({"path": str(admission.ROOT / config["static_early"]), "sha256": config["static_early_sha256"]})
    changed = copy.deepcopy(saved)
    changed["analysis"]["candidates"]["innovation_quality"]["certified_model_witness"] = False
    with pytest.raises(ValueError, match="independently checked"):
        admission.validate_static(changed)
