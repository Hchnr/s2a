"""Compare face-based and vertical projections on the same saved states."""

import argparse
from pathlib import Path

import numpy as np

from s2a.infrastructure.publication import publish_result as publish
from s2a.research.joint_control.tracking.box_face_reference_projection import project_box_faces_reference
from s2a.research.wam.training.native.native_training import state_sha256
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.response_models import directional_slide_boxes_study as prior

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/response_models/box_face_projection_20261111.json"
CONFIG_SHA = "ef2782134df3182b7930462c694db3d1e8fe66d16f4ebb88c8de99a10793a482"
PLAN = ROOT / "plans/v0.0.1/tracking/response_models/box_face_projection_20261111.md"
OUTPUT = ROOT / "outputs/box_face_projection_20261111"
DRY = ROOT / "outputs/box_face_projection_dry_20261111/result.json"
QA = ROOT / "outputs/qa/box_face_study_20261111/result.json"


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__).resolve(),
            Path(prior.__file__).resolve(),
            ROOT / "src/s2a/research/joint_control/tracking/box_face_reference_projection.py",
            ROOT / "src/s2a/research/joint_control/tracking/directional_slide_box_projection.py",
            ROOT / "src/s2a/research/joint_control/tracking/parallel_box_reference_projection.py",
            ROOT / "src/s2a/research/joint_control/tracking/tabletop_reference_projection.py",
            ROOT / "tests/research/joint_control/tracking/test_box_face_reference_projection.py",
            ROOT / "tests/research/response_models/test_box_face_projection_study.py",
        )
    ]


def load_inputs():
    base = prior.load_inputs()
    consumed = dict(base["source_sha256"])
    config_ref = {"path": str(CONFIG), "sha256": CONFIG_SHA}
    config = prefix.read(config_ref, consumed)
    prefix.require(
        config["schema"] == "s2a.box-face-projection-study.v1"
        and config["output"] == str(OUTPUT.relative_to(ROOT))
        and (config["case_index"], config["seed"], config["first_step"], config["periods"]) == (8, 42, 120, 8)
        and (config["position_uncertainty_m"], config["rotation_uncertainty_rad"]) == (0.01, 0.05)
        and config["additional_shape_uncertainty_m"] == 0.0
        and (config["clearance_m"], config["numerical_margin_m"], config["max_translation_m"]) == (0.001, 1e-9, 0.05)
        and (config["max_faces_per_box"], config["max_box_combinations_per_endpoint"]) == (6, 36)
        and config["max_native_models"] == config["max_forward_queries"] == config["physics_steps"] == 0
        and config["world_model_calls"] == config["fresh_cases"] == 0
        and config["automatic_retry"] is False,
        "The exact previously recorded conditions and zero-native budget are required",
    )
    previous_ref = {"path": str(ROOT / config["previous_study"]), "sha256": config["previous_study_sha256"]}
    previous = prefix.read(previous_ref, consumed)
    prefix.require(
        previous["complete"] is previous["passed"] is previous["sources_unchanged"] is True
        and previous["control_qualified"] is False
        and previous["inputs"] == prior.input_receipt(base)
        and previous["stats"]
        == dict(
            models_attempted=1,
            models_completed=1,
            forward_queries_attempted=16,
            forward_queries_completed=16,
            physics_steps_attempted=0,
        )
        and len(previous["analysis"]["rows"]) == 16,
        "The exact source-pinned directional study and its complete endpoints are required",
    )
    for ref in previous["source_refs"]:
        prefix.checked(ref, consumed)
    return {
        "config": config,
        "config_ref": config_ref,
        "previous_ref": previous_ref,
        "previous": previous,
        "base": base,
        "source_sha256": consumed,
    }


def input_receipt(inputs):
    return {
        "config_ref": inputs["config_ref"],
        "previous_ref": inputs["previous_ref"],
        "prior": prior.input_receipt(inputs["base"]),
        "source_sha256": inputs["source_sha256"],
    }


def compare_endpoint(decision, previous, joint_range, boxes, config):
    """Compare the two conditional projections without claiming physical execution."""
    q = previous["slide_position_m"]
    lower, upper = joint_range
    vertical = previous["directional"]
    prefix.require((vertical is not None) == (lower <= q <= upper), "The same joint-range admission must hold")
    if vertical is None:
        return {"status": "joint_out_of_range", "vertical": None, "faces": None}
    faces = project_box_faces_reference(
        decision["original_target_tcp"],
        decision["decision_vertices_tcp"],
        previous["slide_axis_tcp"],
        q,
        lower,
        upper,
        boxes,
        position_uncertainty=config["position_uncertainty_m"],
        rotation_uncertainty=config["rotation_uncertainty_rad"],
        additional_shape_uncertainty=config["additional_shape_uncertainty_m"],
        clearance=config["clearance_m"],
        numerical_margin=config["numerical_margin_m"],
        max_translation=config["max_translation_m"],
    )
    prefix.require(
        faces["face_combinations_checked"] == config["max_box_combinations_per_endpoint"]
        and faces["required_translation_m"] <= vertical["required_translation_m"] + 1e-10,
        "The same conditions and bounded set of faces must include the previous upward candidate",
    )
    return {"status": "evaluated", "vertical": vertical, "faces": faces}


def analyze(inputs):
    config, previous = inputs["config"], inputs["previous"]["analysis"]
    baseline = {(row["mode"], row["step"]): row for row in previous["rows"]}
    prefix.require(len(baseline) == 16, "All previous endpoints must be unique")
    boxes = [(name, np.asarray(pose), np.asarray(size)) for name, pose, size in previous["box_anchors"]]
    prefix.require(
        [box[0] for box in boxes] == ["table", "stove"], "The two previously verified fixed boxes are required"
    )
    joint_range = previous["topology"]["joint_interval_m"]
    rows = []
    for mode, branch in inputs["base"]["base"]["branches"].items():
        continuation = branch["producer"]["runtime"]["continuation"]
        periods = continuation["probe"]["periods"]
        states = [continuation["pre_model_replay"]["probe"]["periods"][-1]["actual_state"]] + [
            row["actual_state"] for row in periods[:-1]
        ]
        for row, state in zip(periods, states, strict=True):
            old = baseline[(mode, row["step"])]
            digest = state_sha256(np.asarray(state, dtype=float))
            prefix.require(digest == old["state_sha256"], "Each result must use its own audited saved state")
            comparison = compare_endpoint(row["decision"], old, joint_range, boxes, config)
            rows.append(
                {
                    "mode": mode,
                    "step": row["step"],
                    "state_sha256": digest,
                    "slide_position_m": old["slide_position_m"],
                    **comparison,
                }
            )
    prefix.require(len(rows) == 16, "All registered endpoints must be included")
    return {"rows": rows, "box_anchors": previous["box_anchors"], "joint_interval_m": joint_range}


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
                numerical_endpoints=0,
                face_combinations=0,
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
        "Source-matched 32-worker tests and a zero-work dry are required",
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
        rows = result["analysis"]["rows"]
        result["numerical_endpoints"] = len(rows)
        result["face_combinations"] = sum(
            row["faces"]["face_combinations_checked"] for row in rows if row["faces"] is not None
        )
        prefix.require(result["face_combinations"] <= 16 * 36, "The registered face budget must hold")
        result["sources_unchanged"] = snapshot(ROOT)["sources"] == before
        prefix.require(result["sources_unchanged"], "Sources changed during the study")
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
