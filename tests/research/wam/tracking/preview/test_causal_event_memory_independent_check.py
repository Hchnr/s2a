"""A separate reconstruction must reject altered shadow receipts and inputs."""

import copy
import json

import pytest

from s2a.infrastructure.publication import result_json
from scripts.verification.reference_ports import check_causal_event_memory as check
from scripts.wam.tracking.preview import replay_causal_event_memory as producer
from tests.research.wam.tracking.preview.test_causal_event_memory_shadow import fixture


def sample():
    chunks, rows, tool, config = fixture()
    case = {"case_index": 0}
    produced = json.loads(result_json(producer.shadow_case(case, chunks, rows, tool, config)))
    return case, produced, chunks, rows, tool, config


def test_independently_reconstructs_multi_replan_event_and_baseline_boundary():
    case, produced, chunks, rows, tool, config = sample()
    result = check.verify_case(case, produced, chunks, rows, tool, config)
    assert result["summary"] == produced["summary"]
    assert result["first_intervention_step"] == 16
    assert result["summary"]["closure_interventions"] == 1
    assert result["max_numerical_difference"] <= 1e-10


@pytest.mark.parametrize(
    "corruption",
    ["reason", "source", "index", "endpoint", "pose", "opening", "event", "memory", "guard", "label", "summary"],
)
def test_rejects_altered_decision_or_summary(corruption):
    case, produced, chunks, rows, tool, config = sample()
    changed = copy.deepcopy(produced)
    item = changed["records"][2]
    if corruption == "reason":
        item["reason"] = "event_retained"
    elif corruption == "source":
        item["source_step"] = 8
    elif corruption == "index":
        item["source_indices"][0] += 1
    elif corruption == "endpoint":
        item["absolute_endpoints"][0] += 1
    elif corruption == "pose":
        item["poses"][0][0][3] += 0.001
    elif corruption == "opening":
        item["openings"][0] = 0.2
    elif corruption == "event":
        item["event"]["absolute_endpoint"] += 1
    elif corruption == "memory":
        item["memory_after"]["last_completed_replan_step"] = 8
    elif corruption == "guard":
        item["guards"]["event_position_m"] += 0.001
    elif corruption == "label":
        item["baseline_observation_label"] = "baseline_observation_before_intervention"
    else:
        changed["summary"]["interventions"] = 2
    with pytest.raises((ValueError, AssertionError)):
        check.verify_case(case, changed, chunks, rows, tool, config)


def test_rejects_changed_original_time_and_measured_anchor():
    case, produced, chunks, rows, tool, config = sample()
    rows[8]["opening_before"] = 0.2
    with pytest.raises((ValueError, AssertionError)):
        check.verify_case(case, produced, chunks, rows, tool, config)


def test_real_receipt_binds_all_twelve_original_development_cases():
    inputs = check.load_inputs()
    assert [item["case"]["case_id"] for item in inputs["cases"]] == [check.expected_case(i) for i in range(12)]
    assert all("/development/cases/" in item["refs"]["chunks"]["path"] for item in inputs["cases"])
