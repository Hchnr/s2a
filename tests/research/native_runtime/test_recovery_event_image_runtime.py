"""The event-image owner must preserve the registered replay and probe boundaries."""

import copy
import json

import pytest

from scripts.simulation.runtime import run_recovery_event_images as run


def test_fixed_roster_and_physical_budgets_bind_original_cases():
    config = json.loads(run.CONFIG.read_text())
    run.validate_config(config)
    for change in (
        dict(max_replay_control_periods=1438),
        dict(training_authorized=True),
        dict(max_saved_event_images=13),
    ):
        with pytest.raises(ValueError, match="bounded resources"):
            run.validate_config(dict(config, **change))
    changed = copy.deepcopy(config)
    changed["event_audit"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="audit source"):
        run.validate_config(changed)
    _config, identity = run.load_inputs()
    assert [case["anchors"] for case in identity["cases"]] == [list(row) for row in run.ANCHORS]
    assert sum(case["replay_periods"] for case in identity["cases"]) == 1437


def test_worker_rejects_unreplayed_endpoints_or_any_probe_action():
    identity = dict(index=0, anchors=[89, 120], replay_periods=120)
    counters = {
        name: dict(attempted=count, completed=count)
        for name, count in (("mj_step", 3000), ("mj_step2", 0), ("model_construct", 3), ("context", 3))
    }
    images = [dict(frame_index=89), dict(frame_index=120)]
    runtime = dict(
        complete=True,
        passed=True,
        world_model_calls=0,
        newly_selected_commands=0,
        control_steps_attempted=120,
        control_steps_completed=120,
        counters=counters,
        cleanup_complete=True,
        behavioural_replay_verified=True,
        prefix_control_steps=120,
        checked_endpoints=121,
        exact_state_endpoints=121,
        max_state_difference=0.0,
        max_cached_field_difference=0.0,
        event_images=images,
        unsaved_current_camera_renders=15,
    )
    branch = dict(
        complete=True,
        passed=True,
        requested_phase="replay",
        candidate_modules_unloaded=True,
        fresh_cases=0,
        case=identity,
        runtime=runtime,
        images=images,
    )
    assert run.validate_worker(branch, "replay", identity)["control_steps"] == 120
    bad = copy.deepcopy(branch)
    bad["runtime"]["exact_state_endpoints"] = 120
    with pytest.raises(ValueError, match="controller cache"):
        run.validate_worker(bad, "replay", identity)
    probe = copy.deepcopy(branch)
    probe.update(requested_phase="probe", images=[dict(frame_index=0)])
    probe["runtime"].update(
        control_steps_attempted=0,
        control_steps_completed=0,
        reset_observation_verified=True,
        continuation=dict(segments=[], stop=None, probe_no_actions=True),
    )
    probe["runtime"]["counters"]["mj_step"] = dict(attempted=0, completed=0)
    assert run.validate_worker(probe, "probe", identity)["control_steps"] == 0
    probe["runtime"]["counters"]["mj_step"]["completed"] = 1
    with pytest.raises(ValueError, match="physical and image counts"):
        run.validate_worker(probe, "probe", identity)
