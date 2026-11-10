"""Independent scalar replay rejects wrong clock, geometry and phase labels."""

import copy
import subprocess
import sys

import pytest

from scripts.verification.reference_ports import check_replan_task_events as checker
from scripts.wam.tracking.preview import audit_replan_task_events as producer
from tests.research.wam.tracking.preview.test_replan_task_event_inputs import dataset


def case():
    chunks, rows, tool = dataset()
    chunks[0]["openings"][12:] = 0.1
    item = {"case": {"case_index": 0}, "target": "target"}
    result = producer.describe_case(item["case"], item["target"], chunks, rows, tool)
    return item, result, chunks, rows, tool


def test_independent_replay_checks_every_endpoint_and_opportunity():
    item, result, chunks, rows, tool = case()
    checked = checker.verify_case(item, result, chunks, rows, tool)
    assert checked["pairs"] == 74 and checked["matched_endpoints"] == 2368
    assert checked["events"]["closure"]["opportunities_next_eight"] == 1
    assert checked["events"]["closure"]["opportunity_all"] == 1
    assert max(checked["max_differences"].values()) < 1e-10


@pytest.mark.parametrize("corruption", ["time", "pose", "opening", "phase", "summary"])
def test_independent_replay_rejects_corrupted_producer_evidence(corruption):
    item, result, chunks, rows, tool = case()
    damaged = copy.deepcopy(result)
    pair = damaged["pairs"][0]
    if corruption == "time":
        pair["closure"]["old_absolute_endpoint"] += 1
    elif corruption == "pose":
        pair["rotation_difference_rad"][20] += 0.01
    elif corruption == "opening":
        pair["opening_difference"][3] += 0.1
    elif corruption == "phase":
        pair["closure"]["first_event_revision"] = "same"
    else:
        damaged["task_success_steps"] = 1
    with pytest.raises((ValueError, AssertionError)):
        checker.verify_case(item, damaged, chunks, rows, tool)


def test_checker_import_does_not_import_producer_or_its_phase_algorithm():
    code = (
        "import sys; import scripts.verification.reference_ports.check_replan_task_events; "
        "assert 'scripts.wam.tracking.preview.audit_replan_task_events' not in sys.modules; "
        "assert 's2a.research.wam.tracking.preview.replan_task_events' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_checker_binds_all_twelve_producer_results_to_original_inputs():
    inputs = checker.load_inputs()
    assert len(inputs["cases"]) == 12
    assert [item["case"]["case_index"] for item in inputs["cases"]] == list(range(12))
    assert inputs["producer_ref"] == checker.PRODUCER
