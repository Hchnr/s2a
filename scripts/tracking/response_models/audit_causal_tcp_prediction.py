"""Score causal TCP model-error diagnostics on independently replayed feedback."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import finite
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.response_models.causal.tcp_prediction_innovation import CausalTCPPredictionInnovation
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import prepare_quality_rate_repair_recorded_feedback as physical
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/response_models/causal_tcp_prediction_audit_20261111.json"
PLAN = ROOT / "plans/v0.0.1/tracking/response_models/causal_tcp_prediction_audit_20261111.md"
OUTPUT = ROOT / "outputs/causal_tcp_prediction_audit_20261111/result.json"
QA = ROOT / "outputs/qa/causal_tcp_prediction_audit_20261111/result.json"
FIXED = {
    "schema": "s2a.causal-tcp-prediction-audit.v1",
    "output": "outputs/causal_tcp_prediction_audit_20261111/result.json",
    "recorded_audit": "outputs/quality_rate_repair_recorded_feedback_audit_20261111/result.json",
    "recorded_audit_sha256": "bd33ed32be8e4bb4a9ef0c3fb753a8d0883e7cb6bd3ef22fedd9729807b4c802",
    "producer": "outputs/quality_rate_repair_recorded_feedback_20261111/producer/simulation/result.json",
    "producer_sha256": "e936271d98ba46fc975c2eea3e2808930f6d7d895c0c4e18db254e08b1be154d",
    "case_index": 8,
    "first_step": 80,
    "periods": 46,
    "position_cap_m": 0.01,
    "rotation_cap_rad": 0.05,
    "candidates": ["raw", "previous_innovation", "past_maximum_bound"],
    "model_calls": 0,
    "physics_steps": 0,
    "fresh_cases": 0,
    "automatic_retry": False,
}


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__).resolve(),
            ROOT / "src/s2a/research/response_models/causal/tcp_prediction_innovation.py",
            ROOT / "tests/research/response_models/test_tcp_prediction_innovation.py",
            ROOT / "tests/research/native_runtime/test_causal_tcp_prediction_audit.py",
        )
    ]


def load_inputs():
    prefix.environment()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    prefix.require(
        type(config) is dict
        and set(config) == set(FIXED)
        and all(type(config[key]) is type(value) and config[key] == value for key, value in FIXED.items()),
        "The exact causal TCP prediction audit scope is required",
    )
    consumed = {config_ref["path"]: config_ref["sha256"]}
    refs = {
        key: {"path": str(ROOT / config[key]), "sha256": config[key + "_sha256"]}
        for key in ("recorded_audit", "producer")
    }
    previous, producer = (prefix.read(refs[key], consumed) for key in ("recorded_audit", "producer"))
    prefix.require(
        previous["complete"] is previous["passed"] is previous["checker_policy_isolated"] is True
        and previous["continuation_periods"] == 46
        and previous["replayed_continuation_inner_samples"] == 1150
        and previous["control_qualified"] is False
        and producer["complete"] is producer["passed"] is True,
        "The exact independent replay audit and producer are required",
    )
    rows = physical.periods(producer)
    prefix.require(
        len(rows) == 46 and [row["step"] for row in rows] == list(range(80, 126)),
        "Forty-six consecutively completed physical periods are required",
    )
    prefix.reference(PLAN)
    return {"config": config, "refs": refs, "rows": rows, "source_sha256": consumed}


def _distance(predicted, actual):
    predicted, actual = finite(predicted, (6,)), finite(actual, (6,))
    rotation = Rotation.from_rotvec(predicted[3:]).inv() * Rotation.from_rotvec(actual[3:])
    return float(np.linalg.norm(predicted[:3] - actual[:3])), float(rotation.magnitude())


def _safe(error):
    return bool(np.linalg.norm(error[:3]) <= 0.01 + 1e-10 and np.linalg.norm(error[3:]) <= 0.05 + 1e-10)


def evaluate(periods):
    """Stage each prediction before reading that period's measured TCP error."""
    history, rows = CausalTCPPredictionInnovation(), []
    for period in periods:
        quality = period["decision"]["quality"]
        prefix.require(quality["verification"]["passed"] is True, "A checked model prediction is required")
        prediction = finite(quality["first_tcp_error"], (6,))
        staged = history.stage(period["step"], prediction)
        # The measured endpoint is read only after staging the causal estimates.
        actual = finite(period["tcp_error"], (6,))
        actual_safe = bool(np.linalg.norm(actual[:3]) <= 0.01 and np.linalg.norm(actual[3:]) <= 0.05)
        prefix.require(period["pose_caps_satisfied"] is actual_safe, "Saved physical pose gate changed")
        raw_position, raw_rotation = _distance(staged["raw"], actual)
        row = {
            "step": period["step"],
            "completed_history": staged["completed_history"],
            "history_available": staged["history_available"],
            "actual_error": actual.copy(),
            "actual_pose_caps_satisfied": actual_safe,
            "raw": {
                "predicted_error": staged["raw"],
                "position_prediction_error_m": raw_position,
                "rotation_prediction_error_rad": raw_rotation,
                "predicted_pose_caps_satisfied": _safe(staged["raw"]),
            },
            "previous_innovation": None,
            "past_maximum_bound": None,
        }
        if staged["history_available"]:
            corrected = staged["previous_innovation"]
            position, rotation = _distance(corrected, actual)
            row["previous_innovation"] = {
                "predicted_error": corrected,
                "position_prediction_error_m": position,
                "rotation_prediction_error_rad": rotation,
                "predicted_pose_caps_satisfied": _safe(corrected),
            }
            upper_position, upper_rotation = staged["past_maximum_bound_m"], staged["past_maximum_bound_rad"]
            row["past_maximum_bound"] = {
                "position_upper_m": upper_position,
                "rotation_upper_rad": upper_rotation,
                "predicted_pose_caps_satisfied": bool(
                    upper_position <= 0.01 + 1e-10 and upper_rotation <= 0.05 + 1e-10
                ),
                "position_covered": bool(np.linalg.norm(actual[:3]) <= upper_position + 1e-12),
                "rotation_covered": bool(np.linalg.norm(actual[3:]) <= upper_rotation + 1e-12),
                "future_guarantee": False,
            }
        row["completed_innovation"] = history.complete(period["step"], actual)
        rows.append(row)
    return rows


def summarize(rows):
    scored = [row for row in rows if row["history_available"]]
    prefix.require(scored, "At least one completed historical prediction is required")
    candidates = {}
    for name in ("raw", "previous_innovation", "past_maximum_bound"):
        predicted_safe = [row[name]["predicted_pose_caps_satisfied"] for row in scored]
        actual_safe = [row["actual_pose_caps_satisfied"] for row in scored]
        summary = {
            "periods": len(scored),
            "predicted_safe": sum(predicted_safe),
            "false_safe": sum(pred and not actual for pred, actual in zip(predicted_safe, actual_safe, strict=True)),
            "false_stop": sum(not pred and actual for pred, actual in zip(predicted_safe, actual_safe, strict=True)),
        }
        if name != "past_maximum_bound":
            for axis, unit in (("position", "m"), ("rotation", "rad")):
                errors = np.asarray([row[name][f"{axis}_prediction_error_{unit}"] for row in scored])
                summary[f"{axis}_prediction_rmse_{unit}"] = float(np.sqrt(np.mean(errors**2)))
        else:
            summary["position_covered"] = sum(row[name]["position_covered"] for row in scored)
            summary["rotation_covered"] = sum(row[name]["rotation_covered"] for row in scored)
        candidates[name] = summary
    windows = []
    first = rows[0]["step"]
    for offset in range(0, len(rows), 8):
        window = rows[offset : offset + 8]
        windows.append(
            {
                "start_step": first + offset,
                "periods": len(window),
                "actual_pose_caps_satisfied": sum(row["actual_pose_caps_satisfied"] for row in window),
                "position_prediction_rmse_m": float(
                    np.sqrt(np.mean([row["raw"]["position_prediction_error_m"] ** 2 for row in window]))
                ),
                "rotation_prediction_rmse_rad": float(
                    np.sqrt(np.mean([row["raw"]["rotation_prediction_error_rad"] ** 2 for row in window]))
                ),
            }
        )
    return {"candidates": candidates, "windows": windows}


def run():
    inputs = load_inputs()
    rows = evaluate(inputs["rows"])
    return {
        "complete": True,
        "passed": True,
        "source_refs": source_refs(),
        "source_sha256": inputs["source_sha256"],
        "model_calls": 0,
        "physics_steps": 0,
        "fresh_cases": 0,
        "task_labels_used": False,
        "control_qualified": False,
        "automatic_retry": False,
        "summary": summarize(rows),
        "periods": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    qa = prefix.read(prefix.reference(QA))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == 0
        and not qa["collection"]["collection_errors"]
        and snapshot(ROOT)["sources"] == prefix.read(qa["source_before_ref"])["sources"],
        "A completed source-pinned 32-worker CPU QA is required",
    )
    print(publish(OUTPUT, run()), flush=True)


if __name__ == "__main__":
    main()
