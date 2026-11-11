"""Preserve the saved-state budget and count per-sample simulated force."""

import json

import pytest

from scripts.tracking.response_models.early_box_face_bowl_effect import CONFIG, normal_force_summary, validate_config


def test_registered_analysis_budget_cannot_start_physics():
    config = json.loads(CONFIG.read_text())
    validate_config(config)
    config["physics_steps"] = 1
    with pytest.raises(ValueError):
        validate_config(config)


def test_normal_force_summary_distinguishes_samples_and_contact_points():
    contact = dict(active_constraint=True, geom1="finger", geom2="bowl", wrench_contact_frame=[2, 0, 0, 0, 0, 0])
    other = {**contact, "wrench_contact_frame": [1, 0, 0, 0, 0, 0]}
    samples = [{"contacts": [contact, other]}, {"contacts": [other]}, {"contacts": []}]
    result = normal_force_summary(samples, {"finger", "bowl"})
    assert result["active_inner_samples"] == 2 and result["active_contact_records"] == 3
    assert result["maximum_single_contact_normal_n"] == 2
    assert result["maximum_sum_normal_n_per_sample"] == 3
    assert result["mean_sum_normal_n_per_active_sample"] == 2
    with pytest.raises(ValueError):
        normal_force_summary(
            [{"contacts": [{**contact, "wrench_contact_frame": [-1, 0, 0, 0, 0, 0]}]}], {"finger", "bowl"}
        )
