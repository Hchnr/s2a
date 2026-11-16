"""Saved-contact comparisons must preserve missing future and label mismatches."""

import copy
import json

import pytest

from scripts.simulation.runtime import reconstruct_seed_policy_contacts as run


def endpoint(step, *, target=(0, 0, 0), tcp=(0, 0, 0), contact=False, spatial=False, success=False, grasp=False):
    return dict(
        step=step,
        target_world_m=list(target),
        tcp_world_m=list(tcp),
        opening=0.5,
        native_success=success,
        native_grasp=grasp,
        left_pad_contact=False,
        right_pad_contact=False,
        bilateral=False,
        conditions=dict(support_contact=contact, spatial_satisfied=spatial, reconstructed_success=contact and spatial),
        tcp_reconstruction_difference_m=0.0,
        object_mapping_difference_m=0.0,
        state_normalization_difference=0.0,
    )


def test_pair_comparison_never_fills_unexecuted_future():
    fast = [endpoint(0), endpoint(1, target=(0.002, 0, 0), contact=True)]
    bounded = [endpoint(0), endpoint(1), endpoint(2, success=True)]
    result = run.pair_metrics(fast, bounded)
    assert result["common_endpoints"] == 2
    assert result["object_first_above_1mm"] == 1
    assert result["first_support_contact_difference"] == 1
    assert result["first_success_difference"] is None
    assert run.pair_metrics([], bounded)["object_max_m"] is None


def test_reconstructed_labels_do_not_overwrite_native_observations():
    rows = [endpoint(0, contact=True, spatial=True, success=False), endpoint(1, grasp=True)]
    result = run.summary(rows)
    assert result["goal_label_mismatch_steps"] == [0]
    assert result["grasp_label_mismatch_steps"] == [1]
    assert result["interpretation_eligible"] is False
    assert result["native_success_count"] == 0


def test_fixed_source_and_no_physics_budget():
    config = json.loads(run.CONFIG.read_text())
    run.validate_config(config)
    for key, value in (("physics_steps", 1), ("gpu_calls", 1), ("max_forwards", 9459)):
        with pytest.raises(ValueError):
            run.validate_config(dict(config, **{key: value}))
    changed = copy.deepcopy(config)
    changed["bounded_cohort_audit"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        run.validate_config(changed)


def test_all_registered_source_rosters_and_counts_match():
    config, _identity, audits, _first, mapping = run.inputs()
    assert config["candidate_cases"] == len(audits[0]["cases"]) == len(mapping["cases"]) == 18


def test_failed_or_unreaped_pilot_prevents_the_remaining_workers():
    workers = [
        dict(key=key, returncode=0, actual_wait=True, reaped=True, signal_sent=False) for key in ("3", "7", "12")
    ]
    lifecycle = dict(workers=workers, launch_error=None)
    assert run.initial_wave_ready(lifecycle) is True
    for key, value in (("returncode", 1), ("reaped", False), ("signal_sent", True), ("key", "1")):
        changed = copy.deepcopy(lifecycle)
        changed["workers"][0][key] = value
        assert run.initial_wave_ready(changed) is False
