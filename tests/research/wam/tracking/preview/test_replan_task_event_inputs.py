"""The task-phase audit preserves each actual 20 Hz reference timeline."""

import copy

import numpy as np
import pytest

from scripts.wam.tracking.preview import audit_replan_task_events as audit


def dataset():
    identity = np.eye(4)
    chunks, rows = [], []
    for query in range(75):
        opening = np.ones(40)
        if query == 0:
            opening[20:] = 0.1
        if query == 1:
            opening[22:] = 0.1
        chunks.append(
            {
                "step": 8 * query,
                "effective_targets_tcp": np.tile(identity, (40, 1, 1)),
                "openings": opening,
                "anchor_tcp": identity,
                "anchor_opening": 0.9,
            }
        )
        for offset in range(8):
            rows.append(
                {
                    "step": 8 * query + offset,
                    "chunk_index": query,
                    "chunk_physical_step": offset,
                    "opening_target": float(opening[offset]),
                    "opening_before": 0.9,
                    "measured_source": identity,
                    "success": False,
                    "objects": {"target": {"grasp_contact": False}},
                }
            )
    return chunks, rows, identity


def test_full_case_aligns_executed_openings_and_keeps_absolute_timing():
    chunks, rows, tool = dataset()
    result = audit.describe_case({"case_index": 0}, "target", chunks, rows, tool)
    assert result["periods"] == 600 and result["queries"] == 75 and len(result["pairs"]) == 74
    event = result["pairs"][0]["closure"]
    assert event["old_absolute_endpoint"] == 21 and event["new_first_absolute_endpoint"] == 31
    assert event["new_first_shift_steps"] == 10 and event["first_event_revision"] == "later"
    assert event["opposite_opening_at_old_event"] and event["currently_opposite_phase"]


@pytest.mark.parametrize("corruption", ["step", "chunk", "opening", "anchor", "label"])
def test_full_case_rejects_misaligned_references_and_labels(corruption):
    chunks, rows, tool = dataset()
    if corruption == "step":
        rows[15]["step"] += 1
    elif corruption == "chunk":
        rows[8]["chunk_index"] = 0
    elif corruption == "opening":
        rows[1]["opening_target"] = 0.9
    elif corruption == "anchor":
        chunks[1]["anchor_tcp"] = np.eye(4)
        chunks[1]["anchor_tcp"][0, 3] = 0.01
    else:
        rows[5]["success"] = 0
    with pytest.raises((ValueError, AssertionError)):
        audit.describe_case({"case_index": 0}, "target", chunks, rows, tool)


def test_summary_keeps_all_cases_pairs_and_aligned_endpoint_denominators():
    chunks, rows, tool = dataset()
    sample = audit.describe_case({"case_index": 0}, "target", chunks, rows, tool)
    cases = [copy.deepcopy(sample) for _ in range(12)]
    result = audit.summarize(cases)
    assert result["cases"] == 12 and result["queries"] == 900 and result["periods"] == 7200
    assert result["replan_pairs"] == 888 and result["matched_endpoints"] == 28416
    assert result["closure"]["deferred"] >= 12 and result["closure"]["revisions"]["later"] >= 12


def test_development_inputs_are_the_reviewed_twelve_and_never_final_cases():
    registered = audit.load_inputs()
    assert [item["case"]["case_id"] for item in registered["cases"]] == [audit.expected_case(i) for i in range(12)]
    assert len({item["refs"]["chunks"]["path"] for item in registered["cases"]}) == 12
    assert all("/development/cases/" in item["refs"]["chunks"]["path"] for item in registered["cases"])
    with pytest.raises(ValueError):
        audit.expected_case(12)
