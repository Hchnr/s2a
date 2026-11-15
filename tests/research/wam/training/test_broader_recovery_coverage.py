"""Bind the broader comparison to real replayed sources and fixed limits."""

import copy
import json

import numpy as np
import pytest

from scripts.simulation.runtime import prepare_reset_correct_reference_cohort as preparation
from scripts.verification.wam.training import audit_broader_recovery_coverage as run
from scripts.verification.wam.training.audit_recovery_h40_contract import Evidence


def test_registered_roster_rejects_mutated_limits_and_authority():
    config = json.loads(run.CONFIG.read_text())
    run.validate_config(config)
    for change in (
        dict(request_count=214),
        dict(training_authorized=True),
        dict(fresh_cases=1),
        dict(joint_rotation_limit_rad=0.7),
        dict(sensitivity_excluded_cases=[]),
        dict(new_anchor_count=True),
    ):
        with pytest.raises(ValueError, match="fixed coverage roster"):
            run.validate_config(dict(config, **change))
    altered = copy.deepcopy(config)
    altered["broader_recovery_audit"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fixed coverage roster"):
        run.validate_config(altered)


def test_actual_replayed_pair_rejects_changed_camera_step_and_case():
    config = json.loads(run.CONFIG.read_text())
    broad_ref = preparation.configured_ref(config["broader_recovery_audit"])
    broad = json.loads(run.prefix.checked(broad_ref).read_text())
    row = broad["cases"][0]
    producer = json.loads(run.prefix.checked(row["producer_ref"]).read_text())
    checker = json.loads(run.prefix.checked(row["checker_ref"]).read_text())
    tool_ref = preparation.configured_ref(config["tool_contract"])
    assert run.verify_pair(0, row, producer, checker, tool_ref) == run.pilot.SOURCES[0]
    bad = copy.deepcopy(checker)
    bad["images"][1]["frame_index"] = 17
    with pytest.raises(ValueError, match="qualified and replayed"):
        run.verify_pair(0, row, producer, bad, tool_ref)
    bad = copy.deepcopy(row)
    bad["case"]["variant"] = "base_x"
    with pytest.raises(ValueError, match="qualified and replayed"):
        run.verify_pair(0, bad, producer, checker, tool_ref)


def test_all_actual_image_anchors_bind_to_replayed_current_state():
    config = json.loads(run.CONFIG.read_text())
    tool_ref = preparation.configured_ref(config["tool_contract"])
    tool = np.asarray(json.loads(run.prefix.checked(tool_ref).read_text())["source_from_tcp"], dtype=float)
    broad_ref = preparation.configured_ref(config["broader_recovery_audit"])
    broad = json.loads(run.prefix.checked(broad_ref).read_text())
    evidence = Evidence()
    anchors = run.broader_anchors(evidence, broad, tool_ref, tool)
    assert len(anchors) == 72 and len({row["anchor_id"] for row in anchors}) == 72
    assert len([row for row in anchors if row["pointwise_case_eligible"]]) == 69
    assert {row["case_index"] for row in anchors if not row["pointwise_case_eligible"]} == {13}
    assert all(row["anchor_step"] in (0, 16, 56) for row in anchors)
    evidence.verify()
