"""Audit saved WAM targets for sufficient table-collision certificates."""

import argparse
from pathlib import Path

import numpy as np

from s2a.infrastructure.publication import publish_result as publish
from s2a.research.joint_control.tracking.slide_range_box_collision_certificate import (
    slide_range_vertex_box_certificate,
)
from s2a.research.wam.training.native.native_training import state_sha256
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import prepare_early_box_face_feedback as early_admission
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.response_models import directional_slide_boxes_study as fresh_admission
from scripts.verification.response_models import audit_early_box_face_feedback as early_audit

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/response_models/slide_range_table_collision_20261111.json"
CONFIG_SHA = "3080e8f0be946be8ebeb4fcdc6cfc4bbf014f918186a18f0a948a8d7390969a8"
PLAN = ROOT / "plans/v0.0.1/tracking/response_models/slide_range_table_collision_20261111.md"
OUTPUT = ROOT / "outputs/slide_range_table_collision_20261111"
DRY = ROOT / "outputs/slide_range_table_collision_dry_20261111/result.json"
QA = ROOT / "outputs/qa/slide_range_table_collision_20261111/result.json"


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__).resolve(),
            ROOT / "src/s2a/research/joint_control/tracking/slide_range_box_collision_certificate.py",
            ROOT / "src/s2a/research/joint_control/tracking/pose_box_collision_certificate.py",
            ROOT / "tests/research/joint_control/tracking/test_slide_range_box_collision_certificate.py",
            ROOT / "tests/research/response_models/test_slide_range_table_collision.py",
        )
    ]


def validate_config(config):
    prefix.require(
        config["schema"] == "s2a.slide-range-table-collision.v1"
        and config["output"] == str(OUTPUT.relative_to(ROOT))
        and (config["case_index"], config["seed"]) == (8, 42)
        and config["finger_joint_range_m"] == [-0.04, 0.0]
        and (
            config["position_tolerance_m"],
            config["rotation_tolerance_rad"],
            config["additional_shape_uncertainty_m"],
            config["numerical_margin_m"],
        )
        == (0.01, 0.05, 0.0, 1e-9)
        and (config["max_fresh_endpoints"], config["max_early_endpoints"]) == (16, 14)
        and config["max_native_models"] == config["max_forward_queries"] == config["physics_steps"] == 0
        and config["world_model_calls"] == config["fresh_cases"] == 0
        and config["automatic_retry"] is False,
        "The exact saved endpoints, conditional bounds and zero-native budget are required",
    )


def load_inputs():
    fresh, early = fresh_admission.load_inputs(), early_admission.load_inputs()
    consumed = {**fresh["source_sha256"], **early["early_box_face_feedback"]["source_sha256"]}
    config_ref = {"path": str(CONFIG), "sha256": CONFIG_SHA}
    config = prefix.read(config_ref, consumed)
    validate_config(config)
    refs = {
        key: {"path": str(ROOT / config[key]), "sha256": config[key + "_sha256"]}
        for key in ("fresh_directional", "early_owner", "early_audit")
    }
    directional, owner, audit = (
        prefix.read(refs[key], consumed) for key in ("fresh_directional", "early_owner", "early_audit")
    )
    prefix.require(
        directional["complete"] is directional["passed"] is directional["sources_unchanged"] is True
        and directional["inputs"] == fresh_admission.input_receipt(fresh)
        and owner["complete"] is owner["passed"] is owner["sources_unchanged"] is True
        and audit["complete"] is audit["passed"] is audit["common_start_and_reference"] is True
        and audit["owner_ref"] == refs["early_owner"]
        and directional["control_qualified"] is owner["control_qualified"] is audit["control_qualified"] is False,
        "Both saved studies and their original source audits are required",
    )
    for ref in directional["source_refs"]:
        prefix.checked(ref, consumed)
    early_producers, early_checkers = {}, {}
    for mode in ("original", "box_face"):
        early_producers[mode] = prefix.read(owner["producer_refs"][mode], consumed)
        early_checkers[mode] = prefix.read(owner["checker_refs"][mode], consumed)
        early_audit.audit_replay(early_producers[mode], early_checkers[mode])
    early_audit.compare_common_reference(
        *(early_producers[mode]["runtime"]["continuation"] for mode in ("original", "box_face"))
    )
    boxes = directional["analysis"]["box_anchors"]
    prefix.require([row[0] for row in boxes] == ["table", "stove"], "The verified fixed table identity is required")
    np.testing.assert_allclose(
        boxes[0][1], early["early_box_face_feedback"]["config"]["table_pose"], atol=1e-12, rtol=0
    )
    np.testing.assert_array_equal(boxes[0][2], early["early_box_face_feedback"]["config"]["table_half_size_m"])
    np.testing.assert_array_equal(
        directional["analysis"]["topology"]["joint_interval_m"], config["finger_joint_range_m"]
    )
    np.testing.assert_array_equal(
        early["early_box_face_feedback"]["config"]["finger_joint_range_m"], config["finger_joint_range_m"]
    )
    return {
        "config": config,
        "config_ref": config_ref,
        "refs": refs,
        "fresh": fresh,
        "directional": directional,
        "early": early,
        "early_producers": early_producers,
        "early_checkers": early_checkers,
        "source_sha256": consumed,
    }


def input_receipt(inputs):
    return {
        "config_ref": inputs["config_ref"],
        "refs": inputs["refs"],
        "fresh": fresh_admission.input_receipt(inputs["fresh"]),
        "early": early_admission.input_receipt(inputs["early"]),
        "source_sha256": inputs["source_sha256"],
    }


def assess_target(decision, axis, q, joint_range, table_pose, table_size, config):
    """Keep fixed-current and entire nominal target ranges separate."""
    options = dict(
        position_tolerance=config["position_tolerance_m"],
        rotation_tolerance=config["rotation_tolerance_rad"],
        additional_shape_uncertainty=config["additional_shape_uncertainty_m"],
        numerical_margin=config["numerical_margin_m"],
    )
    args = (decision["original_target_tcp"], decision["decision_vertices_tcp"], axis, q)
    fixed = slide_range_vertex_box_certificate(*args, q, q, table_pose, table_size, **options)
    whole = slide_range_vertex_box_certificate(*args, *joint_range, table_pose, table_size, **options)
    if joint_range[0] <= q <= joint_range[1]:
        prefix.require(
            not whole["certified"] or fixed["certified"],
            "A whole-range certificate must imply its fixed-current subset",
        )
    return {"fixed_current": fixed, "full_joint_range": whole}


def analyze(inputs):
    config, directional = inputs["config"], inputs["directional"]["analysis"]
    _, table_pose, table_size = directional["box_anchors"][0]
    joint_range = config["finger_joint_range_m"]
    lookup = {(row["mode"], row["step"]): row for row in directional["rows"]}
    prefix.require(len(lookup) == config["max_fresh_endpoints"], "All fresh-reference endpoints must be unique")
    fresh_rows, early_rows = [], []
    for mode, branch in inputs["fresh"]["base"]["branches"].items():
        for period in branch["producer"]["runtime"]["continuation"]["probe"]["periods"]:
            old = lookup[(mode, period["step"])]
            result = assess_target(
                period["decision"],
                old["slide_axis_tcp"],
                old["slide_position_m"],
                joint_range,
                table_pose,
                table_size,
                config,
            )
            fresh_rows.append({"mode": mode, "step": period["step"], "state_sha256": old["state_sha256"], **result})
    prefix.require(len(fresh_rows) == config["max_fresh_endpoints"], "All fresh-reference endpoints must be assessed")
    start = np.asarray(inputs["early"]["reference_table_geometry"]["states"][38], dtype=float)
    for mode, producer in inputs["early_producers"].items():
        periods = producer["runtime"]["continuation"]["probe"]["periods"]
        states = [start] + [row["actual_state"] for row in periods[:-1]]
        for period, state in zip(periods, states, strict=True):
            decision = period["decision"]
            result = assess_target(
                decision,
                decision["decision_slide_axis_tcp"],
                decision["decision_joint_position_m"],
                joint_range,
                table_pose,
                table_size,
                config,
            )
            early_rows.append(
                {
                    "mode": mode,
                    "step": period["step"],
                    "state_sha256": state_sha256(np.asarray(state, dtype=float)),
                    **result,
                }
            )
    prefix.require(
        len(early_rows) == config["max_early_endpoints"], "Every previously completed early endpoint must be assessed"
    )
    return {
        "fresh_reference_rows": fresh_rows,
        "early_reference_rows": early_rows,
        "table_pose": table_pose,
        "table_half_size_m": table_size,
    }


def dry_run():
    inputs = load_inputs()
    print(
        publish(
            DRY,
            dict(
                complete=True,
                passed=True,
                inputs=input_receipt(inputs),
                source_refs=source_refs(),
                endpoints=0,
                native_models=0,
                forward_queries=0,
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
                automatic_retry=False,
            ),
        ),
        flush=True,
    )


def execute():
    inputs = load_inputs()
    before = snapshot(ROOT)["sources"]
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and not qa["collection"]["collection_errors"]
        and prefix.read(qa["source_before_ref"])["sources"] == before
        and dry["complete"] is dry["passed"] is True
        and dry["inputs"] == input_receipt(inputs)
        and dry["source_refs"] == source_refs()
        and dry["native_models"] == dry["forward_queries"] == dry["physics_steps"] == 0,
        "Source-matched 32-worker QA and zero-work dry are required",
    )
    OUTPUT.mkdir(exist_ok=False)
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        automatic_retry=False,
        native_models=0,
        forward_queries=0,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        inputs=input_receipt(inputs),
        source_refs=source_refs(),
        qa_ref=prefix.reference(QA),
        dry_ref=prefix.reference(DRY),
    )
    try:
        result["analysis"] = analyze(inputs)
        result["endpoints"] = len(result["analysis"]["fresh_reference_rows"]) + len(
            result["analysis"]["early_reference_rows"]
        )
        prefix.require(result["endpoints"] == 30, "The registered saved endpoint count is required")
        result["sources_unchanged"] = snapshot(ROOT)["sources"] == before
        prefix.require(result["sources_unchanged"], "Sources changed during collision analysis")
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(OUTPUT / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    dry_run() if args.dry else execute()


if __name__ == "__main__":
    main()
