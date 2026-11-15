"""Independently recompute recorded WAM-to-training current-state distances."""

import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.infrastructure.publication import publish_result
from scripts.simulation.runtime import prepare_reset_correct_reference_cohort as preparation
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_start_feedback as feedback
from scripts.verification.wam.training.audit_recovery_h40_contract import Evidence

ROOT = prefix.REPO
OUTPUT = ROOT / "outputs/wam_recovery_state_coverage_20261116/independent.json"
PRIMARY = {
    "path": str(ROOT / "outputs/wam_recovery_state_coverage_20261116/result.json"),
    "sha256": "b1fe1a46929d4733677cf79e50cdb13b88e7d713ee24bdaf9aedc30c646f3846",
}
TOOL = {
    "path": "outputs/clean_training_development_dependency_repair_20261024/inputs/eef_tool_contract.json",
    "sha256": "6fcaed3fe499aacb15d03170d002b1c897091481a0022db5f2922c06320a3a0c",
}


def scalar_nearest(query, anchors):
    """Use a scalar search and SciPy SO(3), independently of the producer helper."""
    eligible = [anchor for anchor in anchors if anchor["task_index"] == query["task_index"]]
    if not eligible:
        raise ValueError("At least one same-task anchor is required")
    point = np.asarray(query["tcp_pose"], dtype=np.float64)
    best, distance2 = None, None
    for anchor in eligible:
        candidate = np.asarray(anchor["tcp_pose"], dtype=np.float64)
        delta = [float(candidate[i, 3] - point[i, 3]) for i in range(3)]
        score = sum(value * value for value in delta)
        if distance2 is None or score < distance2:
            best, distance2 = anchor, score
    relative = point[:3, :3].T @ np.asarray(best["tcp_pose"], dtype=np.float64)[:3, :3]
    return dict(
        anchor_id=best["anchor_id"],
        source_kind=best["source_kind"],
        position_m=float(np.sqrt(distance2)),
        rotation_rad=float(Rotation.from_matrix(relative).magnitude()),
        opening_abs=float(abs(float(best["opening"]) - float(query["opening"]))),
    )


def compare_row(query, anchors, claimed):
    rebuilt = scalar_nearest(query, anchors)
    if any(rebuilt[key] != claimed[key] for key in ("anchor_id", "source_kind")):
        raise ValueError("The selected same-task position anchor differs")
    errors = {key: abs(rebuilt[key] - claimed[key]) for key in ("position_m", "rotation_rad", "opening_abs")}
    if max(errors.values()) > 1e-10:
        raise ValueError("The independently reconstructed nearest-anchor metrics differ")
    return errors


def current_archive(evidence, ref, cache):
    path = evidence.path(ref)
    if str(path) not in cache:
        with np.load(path, allow_pickle=False) as data:
            cache[str(path)] = {
                key: np.asarray(data[key]).copy() for key in ("source_pose", "opening", "sim_states", "sim_time")
            }
    return cache[str(path)]


def verify_anchor(evidence, anchor, tool, cache):
    data = current_archive(evidence, anchor["trajectory_ref"], cache)
    evidence.path(anchor["image_ref"])
    frame, receipt = anchor["anchor_step"], anchor["current_endpoint"]
    state = np.ascontiguousarray(data["sim_states"][frame])
    prefix.require(
        receipt["endpoint_step"] == frame
        and receipt["sim_state_sha256"] == hashlib.sha256(state.tobytes()).hexdigest()
        and receipt["sim_time"] == float(data["sim_time"][frame])
        and anchor["task_index"] == anchor["source_identity"]["task_index"]
        and float(anchor["opening"]) == float(data["opening"][frame]),
        "The training image and current endpoint must match source bytes",
    )
    np.testing.assert_allclose(anchor["tcp_pose"], data["source_pose"][frame] @ tool, atol=1e-12, rtol=0)


def verify_query(evidence, query, tool, cache):
    ref = query["producer_ref"]
    if ref["path"] not in cache:
        cache[ref["path"]] = evidence.json(ref)
    runtime = cache[ref["path"]]["runtime"]
    captures, exchanges = runtime["adaptive_captures"], runtime["adaptive_model_exchanges"]
    ordinal = query["ordinal"]
    capture, exchange = captures[ordinal], exchanges[ordinal]
    request = evidence.json(query["request_ref"])
    evidence.path(query["image_ref"])
    periods = list(runtime["continuation"]["first_window"]["periods"])
    for segment in runtime["continuation"]["adaptive"]["segments"]:
        periods.extend(segment["continuation"]["periods"])
    frame = query["step"]
    actual = np.ascontiguousarray(np.asarray(periods[frame - 1]["actual_state"], dtype=np.float64))
    prefix.require(
        capture["frame_index"] == exchange["step"] == request["step"] == frame
        and exchange["request_ref"] == query["request_ref"]
        and capture["current_endpoint"] == query["current_endpoint"]
        and capture["canvas_png"] == query["image_ref"]
        and request["image"]["sha256"] == query["image_ref"]["sha256"]
        and request["state_sha256"] == hashlib.sha256(actual.tobytes()).hexdigest()
        and request["state_sha256"] == query["current_endpoint"]["sim_state_sha256"]
        and request["sim_time_s"] == query["current_endpoint"]["sim_time"]
        and request["seed"] == query["seed"]
        and float(capture["current_opening"]) == float(query["opening"]),
        "The actual WAM request image and current endpoint must match source bytes",
    )
    np.testing.assert_allclose(query["tcp_pose"], np.asarray(capture["current_source_pose"]) @ tool, atol=1e-12, rtol=0)


def verify():
    prefix.environment()
    feedback.require_no_simulator_import()
    evidence = Evidence()
    primary = evidence.json(PRIMARY)
    tool_ref = preparation.configured_ref(TOOL)
    tool = np.asarray(evidence.json(tool_ref)["source_from_tcp"], dtype=np.float64)
    for path in (Path(__file__).resolve(), ROOT / "tests/research/wam/training/test_state_anchor_coverage_verify.py"):
        evidence.path(prefix.reference(path))
    for ref in primary["input_refs"]:
        evidence.path(ref)
    anchors, rows = primary["anchors"], primary["requests"]
    baseline = [row for row in anchors if row["source_kind"] == "baseline"]
    recovery = [row for row in anchors if row["source_kind"] == "recovery"]
    prefix.require(
        primary["complete"] is primary["passed"] is True
        and primary["support_qualification"] is primary["training_authorized"] is False
        and len(baseline) == 150
        and len(recovery) == 39
        and len(rows) == 213
        and len({row["anchor_id"] for row in anchors}) == 189,
        "The fixed historical request and image-anchor roster is required",
    )
    archive_cache, producer_cache = {}, {}
    for anchor in anchors:
        verify_anchor(evidence, anchor, tool, archive_cache)
    max_errors = dict(position_m=0.0, rotation_rad=0.0, opening_abs=0.0)
    recovered = closer = 0
    for row in rows:
        query = row["query"]
        verify_query(evidence, query, tool, producer_cache)
        for group, claimed in ((baseline, row["baseline"]), (baseline + recovery, row["augmented"])):
            errors = compare_row(query, group, claimed)
            for key, value in errors.items():
                max_errors[key] = max(max_errors[key], value)
        gain = row["baseline"]["position_m"] - row["augmented"]["position_m"]
        prefix.require(abs(gain - row["improvement_m"]) <= 1e-12 and gain >= -1e-12, "Coverage gain differs")
        closer += int(gain > 1e-12)
        recovered += int(row["augmented"]["source_kind"] == "recovery")
    prefix.require(
        closer == primary["strictly_closer"] and recovered == primary["recovery_selected"],
        "The independently counted recovery neighbors must agree",
    )
    evidence.verify()
    feedback.require_no_simulator_import()
    return dict(
        complete=True,
        passed=True,
        primary_ref=PRIMARY,
        input_refs=list(evidence.refs.values()),
        anchors_checked=len(anchors),
        requests_checked=len(rows),
        nearest_selections_checked=2 * len(rows),
        maximum_absolute_difference=max_errors,
        recovery_selected=recovered,
        strictly_closer=closer,
        independent_algorithm="scalar position search; scipy Rotation magnitude",
        support_qualification=False,
        training_authorized=False,
        world_model_calls=0,
        gpu_calls=0,
        physical_steps=0,
        fits=0,
        fresh_cases=0,
    )


def main():
    if OUTPUT.exists():
        raise SystemExit("Independent audit output already exists; no automatic retry or overwrite")
    try:
        result = verify()
    except Exception as error:
        publish_result(
            OUTPUT, dict(complete=False, passed=False, error=dict(type=type(error).__name__, message=str(error)))
        )
        raise
    print(publish_result(OUTPUT, result), flush=True)


if __name__ == "__main__":
    main()
