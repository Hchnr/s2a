"""Run one separately budgeted repair of saved object-state analysis."""

import argparse
from pathlib import Path

from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.response_models import early_box_face_bowl_effect as original

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/response_models/early_box_face_bowl_effect_repair_20261111.json"
CONFIG_SHA = "8a3ab84ac4e507b63fd456a1bd9af6c5943bf7db542592268d12ebf0c5e8ca23"
PLAN = ROOT / "plans/v0.0.1/tracking/response_models/early_box_face_bowl_effect_repair_20261111.md"
OUTPUT = ROOT / "outputs/early_box_face_bowl_effect_repair_20261111"
DRY = ROOT / "outputs/early_box_face_bowl_effect_repair_dry_20261111/result.json"
QA = ROOT / "outputs/qa/early_box_face_bowl_effect_repair_20261111/result.json"
OLD_SOURCE = ROOT / "docs/history/experiments/early_box_face_bowl_effect_before_quat_fix_20261111.py.txt"
OLD_SOURCE_SHA = "ff582a7c5310c0d5427754e910cbaae8bd35aa9d8b4dd0dca0ca0c5219358b0e"


def source_refs():
    return original.source_refs() + [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__).resolve(),
            OLD_SOURCE,
            ROOT / "tests/research/response_models/test_bowl_quaternion_normalization.py",
            ROOT / "tests/research/response_models/test_early_box_face_bowl_effect_repair.py",
        )
    ]


def validate_config(config):
    prefix.require(
        config["schema"] == "s2a.early-box-face-bowl-effect-repair.v1"
        and config["output"] == str(OUTPUT.relative_to(ROOT))
        and config["quaternion_atol"] == 1e-12
        and (
            config["prior_models"],
            config["prior_forward_queries"],
            config["repair_max_models"],
            config["repair_max_forward_queries"],
        )
        == (1, 9, 1, 13)
        and (config["total_max_models"], config["total_max_forward_queries"]) == (2, 22)
        and config["physics_steps"] == config["world_model_calls"] == config["fresh_cases"] == 0
        and config["automatic_retry"] is False,
        "The separately registered normalization repair and cumulative budget are required",
    )


def load_inputs():
    inputs = original.load_inputs()
    consumed = dict(inputs["source_sha256"])
    config_ref = {"path": str(CONFIG), "sha256": CONFIG_SHA}
    config = prefix.read(config_ref, consumed)
    validate_config(config)
    failed_ref = {"path": str(ROOT / config["failed_result"]), "sha256": config["failed_result_sha256"]}
    failed = prefix.read(failed_ref, consumed)
    prefix.require(
        failed["complete"] is failed["passed"] is False
        and failed["automatic_retry"] is False
        and failed["error"]["type"] == "AssertionError"
        and "Mismatched elements: 4 / 41" in failed["error"]["message"]
        and "Max absolute difference: 2.22044605e-16" in failed["error"]["message"]
        and failed["stats"]
        == dict(
            models_attempted=1,
            models_completed=1,
            forward_queries_attempted=9,
            forward_queries_completed=9,
            physics_steps_attempted=0,
        )
        and failed["world_model_calls"] == failed["fresh_cases"] == 0
        and failed["inputs"] == original.input_receipt(inputs),
        "The exact original normalization failure and spent budget must be preserved",
    )
    original_path = str(Path(original.__file__).resolve())
    for ref in failed["source_refs"]:
        if ref["path"] == original_path:
            prefix.require(ref["sha256"] == OLD_SOURCE_SHA, "The failed analysis source identity must match")
            prefix.checked({"path": str(OLD_SOURCE), "sha256": OLD_SOURCE_SHA}, consumed)
        else:
            prefix.checked(ref, consumed)
    inputs["bowl_effect_repair"] = {
        "config": config,
        "config_ref": config_ref,
        "failed_ref": failed_ref,
        "source_sha256": consumed,
    }
    return inputs


def input_receipt(inputs):
    item = inputs["bowl_effect_repair"]
    return {
        "prior": original.input_receipt(inputs),
        "config_ref": item["config_ref"],
        "failed_ref": item["failed_ref"],
        "source_sha256": item["source_sha256"],
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
    item = inputs["bowl_effect_repair"]
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
        "Source-matched 32-worker QA and zero-native dry are required",
    )
    OUTPUT.mkdir(exist_ok=False)
    stats = dict(
        models_attempted=0,
        models_completed=0,
        forward_queries_attempted=0,
        forward_queries_completed=0,
        physics_steps_attempted=0,
    )
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        automatic_retry=False,
        world_model_calls=0,
        fresh_cases=0,
        inputs=input_receipt(inputs),
        source_refs=source_refs(),
        qa_ref=prefix.reference(QA),
        dry_ref=prefix.reference(DRY),
        prior_failure_ref=item["failed_ref"],
        prior_models=1,
        prior_forward_queries=9,
        stats=stats,
    )
    try:
        result["analysis"] = original.analyze(inputs, stats)
        prefix.require(
            stats
            == dict(
                models_attempted=1,
                models_completed=1,
                forward_queries_attempted=13,
                forward_queries_completed=13,
                physics_steps_attempted=0,
            ),
            "Only the registered repair budget may execute",
        )
        checks = result["analysis"]["quaternion_normalization_checks"]
        prefix.require(
            len(checks) == 13
            and all(row["maximum_qpos_change"] <= item["config"]["quaternion_atol"] for row in checks),
            "All 13 saved states must satisfy the normalization contract",
        )
        result["total_models"] = 1 + stats["models_completed"]
        result["total_forward_queries"] = 9 + stats["forward_queries_completed"]
        result["sources_unchanged"] = snapshot(ROOT)["sources"] == before
        prefix.require(result["sources_unchanged"], "Sources changed during repair analysis")
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
