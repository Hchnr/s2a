"""Shadow receipts must preserve causal indices and the baseline-only boundary."""

import copy

import numpy as np
import pytest

from s2a.infrastructure.publication import result_json
from scripts.wam.tracking.preview import replay_causal_event_memory as shadow


def fixture():
    config = {
        "horizon": 40,
        "query_interval": 8,
        "queries_per_case": 75,
        "steps_per_case": 600,
        "closing_below": 0.35,
        "opening_above": 0.8,
        "reference_position_scale_m": 0.02,
        "reference_rotation_scale_rad": 0.1,
        "current_position_scale_m": 0.03,
        "current_rotation_scale_rad": 0.15,
    }
    identity = np.eye(4)
    chunks, rows = [], []
    for query in range(75):
        opening = np.ones(40)
        if query == 0:
            opening[20:] = 0.1
        chunks.append(
            {
                "step": query * 8,
                "effective_targets_tcp": np.tile(identity, (40, 1, 1)),
                "openings": opening,
                "anchor_tcp": identity.copy(),
                "anchor_opening": 0.9,
            }
        )
        for offset in range(8):
            rows.append(
                {
                    "step": query * 8 + offset,
                    "chunk_index": query,
                    "chunk_physical_step": offset,
                    "opening_target": float(opening[offset]),
                    "opening_before": 0.9,
                    "measured_source": identity.copy(),
                }
            )
    return chunks, rows, identity, config


def test_retained_event_uses_original_absolute_endpoints_and_labels_later_observations():
    chunks, rows, tool, config = fixture()
    result = shadow.shadow_case({"case_index": 0}, chunks, rows, tool, config)
    records = result["records"]
    assert result["first_intervention_step"] == 16
    assert records[0]["armed_new_event"] and records[1]["reason"] == "event_retained"
    assert records[2]["source"] == "remembered" and records[2]["source_step"] == 0
    assert records[2]["source_indices"] == list(range(16, 24))
    assert records[2]["absolute_endpoints"] == list(range(17, 25))
    assert records[2]["event"]["absolute_endpoint"] == 21
    np.testing.assert_array_equal(records[2]["openings"], chunks[0]["openings"][16:24])
    assert records[1]["baseline_observation_label"] == "baseline_observation_before_intervention"
    assert records[2]["baseline_observation_label"] == "first_intervention_on_baseline"
    assert records[3]["baseline_observation_label"] == "baseline_observation_after_intervention"
    assert result["summary"]["closure_interventions"] == 1
    assert result["summary"]["changed_opening_endpoints"] == 4
    assert result["summary"]["after_first_intervention_queries"] == 72


def test_later_chunks_and_observations_cannot_change_an_earlier_decision():
    chunks, rows, tool, config = fixture()
    original = shadow.shadow_case({"case_index": 0}, chunks, rows, tool, config)
    changed_chunks, changed_rows = copy.deepcopy(chunks), copy.deepcopy(rows)
    changed_chunks[3]["openings"][:24] = 0.1
    for offset in range(8):
        changed_rows[24 + offset]["opening_target"] = 0.1
    changed_rows[24]["opening_before"] = changed_chunks[3]["anchor_opening"] = 0.2
    changed = shadow.shadow_case({"case_index": 0}, changed_chunks, changed_rows, tool, config)
    assert result_json(original["records"][:3]) == result_json(changed["records"][:3])
    assert (
        original["records"][3]["reason"] != changed["records"][3]["reason"]
        or original["records"][3]["armed_new_event"] != changed["records"][3]["armed_new_event"]
    )


def test_no_generated_event_never_selects_a_remembered_reference():
    chunks, rows, tool, config = fixture()
    chunks[0]["openings"][:] = 1.0
    result = shadow.shadow_case({"case_index": 0}, chunks, rows, tool, config)
    assert result["first_intervention_step"] is None
    assert result["summary"]["armed_events"] == result["summary"]["interventions"] == 0
    assert all(
        record["baseline_observation_label"] == "baseline_observation_before_intervention"
        for record in result["records"]
    )


@pytest.mark.parametrize("corruption", ["step", "chunk_index", "opening_target", "anchor", "measured_opening"])
def test_rejects_a_misaligned_original_timeline(corruption):
    chunks, rows, tool, config = fixture()
    if corruption == "step":
        chunks[1]["step"] = 7
    elif corruption == "chunk_index":
        rows[8]["chunk_index"] = 0
    elif corruption == "opening_target":
        rows[9]["opening_target"] = 0.5
    elif corruption == "anchor":
        chunks[1]["anchor_tcp"][0, 3] = 0.01
    else:
        rows[8]["opening_before"] = 0.3
    with pytest.raises((ValueError, AssertionError)):
        shadow.shadow_case({"case_index": 0}, chunks, rows, tool, config)


def test_registration_uses_only_the_reviewed_twelve_development_cases():
    inputs = shadow.load_inputs()
    assert [item["case"]["case_id"] for item in inputs["cases"]] == [shadow.prior.expected_case(i) for i in range(12)]
    assert all("/development/cases/" in item["refs"]["chunks"]["path"] for item in inputs["cases"])
