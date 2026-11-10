"""Compare the two references available at one exact replan endpoint."""

import json
from pathlib import Path

import numpy as np

from s2a.research.joint_control.tracking.replan_reference_scoring import score_reference_pair
from scripts.simulation.runtime import compare_multicase_emergency_retreat as previous
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_combined_warning_feedback as base

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/replan_reference_switch_20261110.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/replan_reference_switch_20261110.md"
QA = REPO / "outputs/qa/replan_reference_switch_20261110/result.json"
DRY = REPO / "outputs/replan_reference_switch_dry_20261110/result.json"
ROOT = REPO / "outputs/replan_reference_switch_20261110"
PRIOR_PRODUCER = base.PRIOR_PRODUCER
PRIOR_LONG = {
    "path": str(REPO / "outputs/combined_warning_long_feedback_20261110/checker/result.json"),
    "sha256": "2166f517817c637db1a2e40bf7095678add714222b441e4604ea889e54846216",
}
BRANCHES = ((2, "new"), (2, "old"), (6, "new"), (6, "old"))
TARGETS = {2: "akita_black_bowl_1", 6: "wine_bottle_1"}
CURSORS = {"old": 8, "new": 0}


def source_refs():
    paths = (
        Path(__file__),
        CONFIG,
        PLAN,
        REPO / "src/s2a/research/joint_control/tracking/replan_reference_scoring.py",
        REPO / "tests/research/joint_control/tracking/test_replan_reference_scoring.py",
        REPO / "tests/research/native_runtime/test_replan_reference_switch_inputs.py",
    )
    return base.source_refs(longer=True) + [prefix.reference(path) for path in paths]


def load_inputs():
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    prefix.require(
        config
        == {
            "schema": "s2a.replan-reference-switch.v1",
            "cases": [2, 6],
            "sampling_seed": 43,
            "prefix_steps": 16,
            "old_chunk_index": 1,
            "old_cursor": 8,
            "new_chunk_index": 2,
            "new_cursor": 0,
            "branches": ["new", "old"],
            "continuation_steps": 8,
            "control_dt_s": 0.05,
            "joint_buffer_rad": 0.05,
            "position_cap_m": 0.01,
            "rotation_cap_rad": 0.05,
            "arm_rate_limit": 0.25,
            "max_control_steps_per_phase": 96,
            "max_native_steps_per_phase": 2400,
            "max_models_per_phase": 12,
            "max_contexts_per_phase": 12,
            "max_refiner_calls_per_phase": 32,
            "max_refiner_response_evaluations_per_phase": 15360,
            "worker_timeout_seconds": 1800,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
        "Exact registered reference-switch comparison",
    )
    for ref in (PRIOR_PRODUCER, PRIOR_LONG):
        earlier = prefix.read(ref)
        prefix.require(
            earlier["complete"] is earlier["passed"] is earlier["sources_unchanged"] is True,
            "Earlier feedback evidence closed",
        )
    inputs = prefix.load_inputs({2: 16, 6: 16})
    refs = inputs["source_sha256"]
    refs.update(
        {
            str(CONFIG): config_ref["sha256"],
            PRIOR_PRODUCER["path"]: PRIOR_PRODUCER["sha256"],
            PRIOR_LONG["path"]: PRIOR_LONG["sha256"],
        }
    )
    review = prefix.read(prefix.REVIEW)
    source_map = review["source_and_artifact_sha256"]
    for case in inputs["cases"]:
        index, n = case["spec"]["case_index"], case["spec"]["prefix_steps"]
        prefix.require(index in TARGETS and n == 16, "Exact two original cases and completed prefixes")
        folder = Path(review["cases"][index]["mechanical"]["case"])
        prefix.require(folder.is_relative_to(REPO / "outputs"), "Reviewed local development directory")
        chunk_path, tool_path, rows_path = (
            folder / "chunks.json",
            Path(case["config"]["tool_contract"]),
            folder / "steps.jsonl",
        )
        for path in (chunk_path, tool_path, rows_path):
            ref = prefix.reference(path)
            prefix.require(source_map.get(ref["path"]) == ref["sha256"], "Reviewed chunk, tool, and prefix identities")
            refs[ref["path"]] = ref["sha256"]
        rows = prefix.read_recorded_prefix(rows_path, n, 600)
        old, new = previous.read_array_item(chunk_path, 1), previous.read_array_item(chunk_path, 2)
        prefix.require(old["step"] == 8 and new["step"] == 16, "Both chunks existed by the current endpoint")
        chunks = {"old": np.asarray(old["effective_targets_tcp"]), "new": np.asarray(new["effective_targets_tcp"])}
        anchors = {"old": np.asarray(old["anchor_tcp"]), "new": np.asarray(new["anchor_tcp"])}
        tool = np.asarray(json.loads(tool_path.read_text())["source_from_tcp"])
        prefix.require(
            all(value.shape == (40, 4, 4) and np.isfinite(value).all() for value in chunks.values())
            and all(value.shape == (4, 4) and np.isfinite(value).all() for value in anchors.values())
            and tool.shape == (4, 4)
            and np.isfinite(tool).all(),
            "Finite full reference chunks and transforms",
        )
        measured_tcp = np.asarray(case["saved"]["source_pose"][-1]) @ tool
        np.testing.assert_array_equal(anchors["new"], measured_tcp)
        prefix.require(rows[-1]["chunk_index"] == 1 and rows[-1]["chunk_physical_step"] == 7, "Completed old chunk")
        np.testing.assert_allclose(rows[-1]["desired_tcp"], chunks["old"][7], atol=1e-10, rtol=0)
        old_row = rows[-1]
        case.update(
            physical_folder=folder,
            reference_chunks=chunks,
            reference_anchors=anchors,
            scoring_targets={"old": chunks["old"][8:16].copy(), "new": chunks["new"][:8].copy()},
            tool=tool,
            completed_history=base.completed_from_prefix(rows, case["commands"]),
            previous_evidence={
                "site": old_row["qualified_forward_input"]["x_site_pose"][0],
                "twist": old_row["qualified_forward_input"]["x_site_twist"][0],
                "coupled_prediction": old_row["coupled_response_history"]["commit"]["nominal_response"],
                "tracking_prediction": old_row["tracking_refinement"]["nominal_response"],
            },
        )
    return inputs


def selected_case(case, mode):
    """Pass exactly one already-generated reference to the feedback policy."""
    prefix.require(mode in CURSORS, "Registered reference choice")
    selected = {
        key: value
        for key, value in case.items()
        if key not in ("reference_chunks", "reference_anchors", "scoring_targets")
    }
    cursor = CURSORS[mode]
    selected.update(
        targets=case["reference_chunks"][mode],
        anchor=case["reference_anchors"][mode],
        fixed_targets=case["scoring_targets"][mode],
        intervention_spec={"cursor": cursor, "target": TARGETS[case["spec"]["case_index"]]},
    )
    return selected


def completed_reference_scores(actual_tcp, scoring_targets, mode, offset, own_error, *, expected=None):
    """Score the completed actual endpoint; references never choose an action here."""
    prefix.require(mode in CURSORS and set(scoring_targets) == set(CURSORS), "Both aligned scoring references required")
    scores = score_reference_pair(actual_tcp, scoring_targets["old"], scoring_targets["new"], offset)
    np.testing.assert_allclose(scores[mode], own_error, atol=1e-10, rtol=0)
    result = {"actual_tcp_pose": np.asarray(actual_tcp).copy(), "reference_errors": scores}
    if expected is not None:
        np.testing.assert_allclose(result["actual_tcp_pose"], expected["actual_tcp_pose"], atol=1e-10, rtol=0)
        for name in CURSORS:
            np.testing.assert_allclose(scores[name], expected["reference_errors"][name], atol=1e-10, rtol=0)
    return result
