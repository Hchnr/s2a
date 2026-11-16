"""The event-image roster must replay only its own recorded training prefixes."""

import copy
import json

import pytest

from scripts.simulation.runtime import prepare_broader_event_images as run
from scripts.simulation.runtime import prepare_reset_correct_reference_cohort as preparation


def test_frozen_config_rejects_changed_sources_roster_and_physical_budget():
    config = json.loads(run.CONFIG.read_text())
    run.validate_config(config)
    for change in (
        dict(max_replay_control_periods=3553),
        dict(max_saved_event_images=35),
        dict(max_probe_control_periods=1),
        dict(training_authorized=True),
        dict(fresh_cases=1),
    ):
        with pytest.raises(ValueError, match="fixed event-image roster"):
            run.validate_config(dict(config, **change))
    changed = copy.deepcopy(config)
    changed["anchors_by_case"][0][0] = 90
    with pytest.raises(ValueError, match="fixed event-image roster"):
        run.validate_config(changed)
    changed = copy.deepcopy(config)
    changed["event_audit"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fixed event-image roster"):
        run.validate_config(changed)


def test_actual_candidate_binding_rejects_changed_endpoint_or_fabricated_image():
    config = json.loads(run.CONFIG.read_text())
    recovery = json.loads(run.prefix.checked(preparation.configured_ref(config["recovery_audit"])).read_text())
    events = json.loads(run.prefix.checked(preparation.configured_ref(config["event_audit"])).read_text())
    row, event = recovery["cases"][0], events["cases"][0]
    producer = json.loads(run.prefix.checked(row["producer_ref"]).read_text())
    source = row["case"]
    bound = run.bind_case(0, row, event, source, producer)
    assert bound["anchors"] == [89, 118] and bound["replay_periods"] == 118
    altered = copy.deepcopy(event)
    altered["candidates"][0]["current_endpoint"]["sim_state_sha256"] = "changed"
    with pytest.raises(ValueError, match="actual endpoint"):
        run.bind_case(0, row, altered, source, producer)
    altered = copy.deepcopy(event)
    altered["candidates"][0]["current_image_available"] = True
    with pytest.raises(ValueError, match="missing-image identity"):
        run.bind_case(0, row, altered, source, producer)


def test_real_sources_bind_all_24_command_prefixes_without_physics():
    config, identity = run.load_inputs()
    assert len(identity["cases"]) == 24
    assert [case["anchors"] for case in identity["cases"]] == [list(row) for row in run.ANCHORS]
    assert sum(case["replay_periods"] for case in identity["cases"]) == config["max_replay_control_periods"] == 3552
    assert sum(len(case["anchors"]) for case in identity["cases"]) == 36
    assert identity["cases"][13]["replay_periods"] == 167 < 295
