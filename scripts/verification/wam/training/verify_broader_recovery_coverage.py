"""Independently recompute broader WAM current-state coverage from source bytes."""

import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.infrastructure.publication import publish_result
from scripts.simulation.runtime import prepare_reset_correct_reference_cohort as preparation
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_start_feedback as feedback
from scripts.verification.wam.training import audit_broader_recovery_coverage as primary
from scripts.verification.wam.training import verify_wam_recovery_state_coverage as previous
from scripts.verification.wam.training.audit_recovery_h40_contract import Evidence

ROOT = prefix.REPO
PRIMARY = ROOT / "outputs/broader_recovery_coverage_20261116/result.json"
OUTPUT = ROOT / "outputs/broader_recovery_coverage_20261116/independent.json"


def scalar_joint(query, anchors, *, max_position_m, max_rotation_rad, max_opening_abs):
    """Search in registration order with scalar distances and SciPy rotations."""
    query_pose = np.asarray(query["tcp_pose"], dtype=float)
    for anchor in anchors:
        if anchor["task_index"] != query["task_index"]:
            continue
        pose = np.asarray(anchor["tcp_pose"], dtype=float)
        position = math.sqrt(sum((float(pose[i, 3]) - float(query_pose[i, 3])) ** 2 for i in range(3)))
        angle = float(Rotation.from_matrix(query_pose[:3, :3].T @ pose[:3, :3]).magnitude())
        opening = abs(float(anchor["opening"]) - float(query["opening"]))
        if position <= max_position_m and angle <= max_rotation_rad and opening <= max_opening_abs:
            return dict(
                anchor_id=anchor["anchor_id"],
                source_kind=anchor["source_kind"],
                position_m=position,
                rotation_rad=angle,
                opening_abs=opening,
            )
    return None


def compare_match(expected, claimed):
    """Check identity and all components, including an absent joint match."""
    if expected is None or claimed is None:
        prefix.require(expected is claimed is None, "Independent joint match existence differs")
        return dict(position_m=0.0, rotation_rad=0.0, opening_abs=0.0)
    prefix.require(
        all(expected[key] == claimed[key] for key in ("anchor_id", "source_kind")),
        "The independently selected anchor identity differs",
    )
    errors = {key: abs(expected[key] - claimed[key]) for key in ("position_m", "rotation_rad", "opening_abs")}
    prefix.require(max(errors.values()) <= 1e-10, "Independently computed state differences disagree")
    return errors


def scalar_quantile(values, fraction):
    ordered = sorted(float(value) for value in values)
    prefix.require(bool(ordered) and all(math.isfinite(value) for value in ordered), "Finite values are required")
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def compare_summary(rows, claimed, group):
    nearest = [row[group]["nearest"] for row in rows]
    distances = [item["position_m"] for item in nearest]
    position = claimed["position"]
    prefix.require(
        position["count"] == len(rows)
        and abs(position["median_m"] - scalar_quantile(distances, 0.5)) <= 1e-10
        and abs(position["p90_m"] - scalar_quantile(distances, 0.9)) <= 1e-10
        and abs(position["max_m"] - max(distances)) <= 1e-10
        and position["at_most_m"]
        == {str(limit): sum(value <= limit for value in distances) for limit in (0.01, 0.02, 0.05, 0.1)}
        and claimed["joint_counts"]
        == {str(limit): sum(row[group]["joint"][str(limit)] is not None for row in rows) for limit in (0.05, 0.1)}
        and claimed["broader_selected"] == sum(item["source_kind"] == "broader_recovery" for item in nearest)
        and claimed["strictly_closer"]
        == sum(row[group]["nearest"]["position_m"] < row["previous"]["nearest"]["position_m"] - 1e-12 for row in rows),
        "The independent pooled or per-case position and joint summary differs",
    )
    for metric in ("rotation_rad", "opening_abs"):
        values = [row[metric] for row in nearest]
        expected = dict(median=scalar_quantile(values, 0.5), p90=scalar_quantile(values, 0.9), maximum=max(values))
        prefix.require(
            all(abs(claimed[metric][key] - value) <= 1e-10 for key, value in expected.items()),
            "The independent associated-state summary differs",
        )


def verify_broader_source(evidence, anchor, broad, cache):
    """Rebind the new image to its producer and separately replayed endpoint."""
    index, step = anchor["case_index"], anchor["anchor_step"]
    case = broad["cases"][index]
    if index not in cache:
        cache[index] = tuple(evidence.json(case[key]) for key in ("producer_ref", "checker_ref"))
    producer, checker = cache[index]
    image = next(row for row in producer["images"] if row["frame_index"] == step)
    replay = next(row for row in checker["images"] if row["frame_index"] == step)
    source = case["case"]["source"]
    prefix.require(
        case["qualified"] is case["replayed"] is case["evidence_complete"] is True
        and anchor["source_kind"] == "broader_recovery"
        and anchor["variant"] == case["case"]["variant"]
        and anchor["pointwise_case_eligible"] is (index != 13)
        and anchor["source_identity"]
        == dict(
            task_index=source["task_index"],
            demo=source["demo"],
            initial_state_sha256=case["case"]["initial_state_sha256"],
        )
        and anchor["trajectory_ref"] == producer["arrays_ref"]
        and anchor["image_ref"] == image["canvas_png"]
        and anchor["current_endpoint"] == image["current_endpoint"] == replay["current_endpoint"]
        and image["canvas_rgb_sha256"] == replay["canvas_rgb_sha256"]
        and image["render_before"] == image["render_after"]
        and replay["render_before"] == replay["render_after"],
        "The broader anchor must match both the actual and replayed image endpoint",
    )
    evidence.path(replay["canvas_png"])
    evidence.path(replay["raw_cameras"])


def verify():
    prefix.environment()
    feedback.require_no_simulator_import()
    evidence = Evidence()
    primary_ref = prefix.reference(PRIMARY)
    result = evidence.json(primary_ref)
    config_ref = prefix.reference(primary.CONFIG)
    config = evidence.json(config_ref)
    primary.validate_config(config)
    refs = {key: preparation.configured_ref(config[key]) for key in primary.REFERENCES}
    prior = evidence.json(refs["previous_coverage"])
    broad = evidence.json(refs["broader_recovery_audit"])
    tool = np.asarray(evidence.json(refs["tool_contract"])["source_from_tcp"], dtype=float)
    for path in (
        Path(__file__).resolve(),
        ROOT / "tests/research/wam/training/test_broader_recovery_coverage_verify.py",
    ):
        evidence.path(prefix.reference(path))
    for ref in result["input_refs"]:
        evidence.path(ref)
    anchors, rows = result["anchors"], result["requests"]
    old, new = anchors[:189], anchors[189:]
    strict = [item for item in new if item["pointwise_case_eligible"]]
    prefix.require(
        result["complete"] is result["passed"] is True
        and result["config_ref"] == config_ref
        and result["support_qualification"] is result["training_authorized"] is False
        and len(old) == 189
        and len(new) == 72
        and len(strict) == 69
        and len(rows) == len(prior["requests"]) == 213
        and old == prior["anchors"]
        and len({anchor["anchor_id"] for anchor in anchors}) == 261,
        "The exact previously audited requests and all new anchors are required",
    )
    archive_cache, producer_cache, broader_cache = {}, {}, {}
    for anchor in anchors:
        previous.verify_anchor(evidence, anchor, tool, archive_cache)
        if anchor["source_kind"] == "broader_recovery":
            verify_broader_source(evidence, anchor, broad, broader_cache)
    groups = dict(previous=old, all=old + new, sensitivity=old + strict)
    rebuilt_rows, max_errors = [], dict(position_m=0.0, rotation_rad=0.0, opening_abs=0.0)
    for row, prior_row in zip(rows, prior["requests"], strict=True):
        query = row["query"]
        prefix.require(query == prior_row["query"], "The historical request order and identity must remain fixed")
        previous.verify_query(evidence, query, tool, producer_cache)
        compared = {}
        for name, group in groups.items():
            nearest = previous.scalar_nearest(query, group)
            differences = compare_match(nearest, row[name]["nearest"])
            for key in max_errors:
                max_errors[key] = max(max_errors[key], differences[key])
            joint = {}
            for limit in config["joint_position_limits_m"]:
                key = str(limit)
                joint[key] = scalar_joint(
                    query,
                    group,
                    max_position_m=limit,
                    max_rotation_rad=config["joint_rotation_limit_rad"],
                    max_opening_abs=config["joint_opening_limit"],
                )
                differences = compare_match(joint[key], row[name]["joint"][key])
                for metric in max_errors:
                    max_errors[metric] = max(max_errors[metric], differences[metric])
            compared[name] = dict(nearest=nearest, joint=joint)
        rebuilt_rows.append(dict(query=query, **compared))
    for name in primary.GROUPS:
        compare_summary(rebuilt_rows, result["groups"][name], name)
    prefix.require(len(result["per_case"]) == 6, "Each development case must have its own summary")
    for expected_index, expected_count, case in zip(
        primary.CASE_INDICES, primary.REQUESTS, result["per_case"], strict=True
    ):
        subset = [row for row in rebuilt_rows if row["query"]["case_index"] == expected_index]
        prefix.require(
            case["case_index"] == expected_index and len(subset) == expected_count, "The case roster differs"
        )
        for name in primary.GROUPS:
            compare_summary(subset, case["groups"][name], name)
    evidence.verify()
    feedback.require_no_simulator_import()
    return dict(
        complete=True,
        passed=True,
        primary_ref=primary_ref,
        input_refs=list(evidence.refs.values()),
        anchors_checked=len(anchors),
        new_anchors_checked=len(new),
        requests_checked=len(rows),
        nearest_selections_checked=len(rows) * len(groups),
        joint_selections_checked=len(rows) * len(groups) * len(config["joint_position_limits_m"]),
        maximum_absolute_difference=max_errors,
        independent_algorithm="scalar Euclidean search; SciPy Rotation magnitude; scalar linear quantiles",
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
