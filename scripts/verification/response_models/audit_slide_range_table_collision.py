"""Independently verify positive slide-range collision witnesses."""

from pathlib import Path

import numpy as np

from s2a.infrastructure.publication import publish_result as publish
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.response_models import slide_range_table_collision as study

ROOT = prefix.REPO
RESULT = ROOT / "outputs/slide_range_table_collision_20261111/result.json"
RESULT_SHA = "1d9719ebaa95e85fbb5312bb05d902e0ae4bc7bf11b733cd98e81f2ca3701951"
OUTPUT = ROOT / "outputs/slide_range_table_collision_audit_20261111/result.json"


def spherical_cap_axis_interval(cosine, angle):
    """Bound a rotated unit axis's component using angular geometry."""
    if (
        not np.isfinite(cosine)
        or not -1 - 1e-12 <= cosine <= 1 + 1e-12
        or not np.isfinite(angle)
        or not 0 <= angle <= np.pi
    ):
        raise ValueError("A valid unit-axis component and rotation angle are required")
    alpha = float(np.arccos(np.clip(cosine, -1, 1)))
    return float(np.cos(min(np.pi, alpha + angle))), float(np.cos(max(0.0, alpha - angle)))


def independent_witness_slack(decision, axis_tcp, q0, target_range, table_pose, table_size, vertex_index, config):
    """Use exact spherical-cap axis bounds to recheck one saved vertex."""
    desired, table = np.asarray(decision["original_target_tcp"]), np.asarray(table_pose)
    vertex = np.asarray(decision["decision_vertices_tcp"])[vertex_index]
    axis, size = np.asarray(axis_tcp), np.asarray(table_size)
    point = (desired[:3, :3] @ vertex + desired[:3, 3] - table[:3, 3]) @ table[:3, :3]
    nominal_axis = (desired[:3, :3] @ axis) @ table[:3, :3]
    theta = config["rotation_tolerance_rad"]
    error = (
        config["position_tolerance_m"]
        + 2 * np.linalg.norm(vertex) * np.sin(theta / 2)
        + config["additional_shape_uncertainty_m"]
    )
    delta = (target_range[0] - q0, target_range[1] - q0)
    lower, upper = [], []
    for coordinate in range(3):
        a, b = spherical_cap_axis_interval(float(nominal_axis[coordinate]), theta)
        values = [a * delta[0], a * delta[1], b * delta[0], b * delta[1]]
        lower.append(point[coordinate] + min(values) - error)
        upper.append(point[coordinate] + max(values) + error)
    lower, upper = np.asarray(lower), np.asarray(upper)
    return float(np.min(np.minimum(lower + size, size - upper)))


def execute():
    inputs = study.load_inputs()
    result_ref = {"path": str(RESULT), "sha256": RESULT_SHA}
    result = prefix.read(result_ref)
    prefix.require(
        result["complete"] is result["passed"] is result["sources_unchanged"] is True
        and result["control_qualified"] is False
        and result["inputs"] == study.input_receipt(inputs)
        and result["endpoints"] == 30
        and result["native_models"] == result["forward_queries"] == result["physics_steps"] == 0,
        "The complete source-pinned saved-target study is required",
    )
    config, analysis = inputs["config"], result["analysis"]
    table_pose, table_size = analysis["table_pose"], analysis["table_half_size_m"]
    fresh_lookup = {(row["mode"], row["step"]): row for row in inputs["directional"]["analysis"]["rows"]}
    records = []
    for mode, branch in inputs["fresh"]["base"]["branches"].items():
        for period in branch["producer"]["runtime"]["continuation"]["probe"]["periods"]:
            old = fresh_lookup[(mode, period["step"])]
            records.append(
                (
                    "fresh_reference_rows",
                    mode,
                    period["step"],
                    period["decision"],
                    old["slide_axis_tcp"],
                    old["slide_position_m"],
                    old["state_sha256"],
                )
            )
    for mode, producer in inputs["early_producers"].items():
        for period in producer["runtime"]["continuation"]["probe"]["periods"]:
            decision = period["decision"]
            records.append(
                (
                    "early_reference_rows",
                    mode,
                    period["step"],
                    decision,
                    decision["decision_slide_axis_tcp"],
                    decision["decision_joint_position_m"],
                    None,
                )
            )
    checks, counts = [], {"fixed_current": 0, "full_joint_range": 0}
    for group, mode, step, decision, axis, q0, state_sha in records:
        matches = [row for row in analysis[group] if row["mode"] == mode and row["step"] == step]
        prefix.require(
            len(matches) == 1 and (state_sha is None or matches[0]["state_sha256"] == state_sha),
            "Every witness must match its saved decision",
        )
        for kind in counts:
            witness = matches[0][kind]
            if not witness["certified"]:
                continue
            interval = [q0, q0] if kind == "fixed_current" else config["finger_joint_range_m"]
            prefix.require(witness["joint_target_interval_m"] == interval, "The witness interval must match its claim")
            slack = independent_witness_slack(
                decision, axis, q0, interval, table_pose, table_size, witness["vertex_index"], config
            )
            prefix.require(
                slack > witness["numerical_guard_m"], "The independent angular bound must also prove strict interior"
            )
            counts[kind] += 1
            checks.append({"group": group, "mode": mode, "step": step, "kind": kind, "independent_slack_m": slack})
    prefix.require(len(records) == 30, "All 30 saved endpoints must be covered")
    sources = [
        prefix.reference(Path(__file__).resolve()),
        prefix.reference(ROOT / "tests/research/response_models/test_slide_range_table_collision_audit.py"),
    ]
    print(
        publish(
            OUTPUT,
            dict(
                complete=True,
                passed=True,
                control_qualified=False,
                result_ref=result_ref,
                source_refs=sources,
                endpoints_checked=len(records),
                positive_witnesses=counts,
                checks=checks,
                native_models=0,
                forward_queries=0,
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
                scope="positive saved-target witnesses under registered assumptions; negative cases remain inconclusive",
            ),
        ),
        flush=True,
    )


if __name__ == "__main__":
    execute()
