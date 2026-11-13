"""Link the new seed-43 sample to previously audited development feedback."""

import copy
from pathlib import Path

import numpy as np

from s2a.infrastructure.publication import publish_result as publish
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro

ROOT = prefix.REPO
OUTPUT = ROOT / "outputs/audits/seed43_historical_control_20261113"
NEW = {
    "path": str(ROOT / "outputs/adaptive_multiseed_real43_20261113/producer/result.json"),
    "sha256": "64b018f5fdbd72b3a055e6a60da3194b14522edc321591e5ac291946ab5cad71",
}
V4 = {
    "path": str(
        ROOT
        / "outputs/tracking_preview_wam_development_v4_20261027/development/service/requests/150_task0_init0_s43_raw_q00/completed.json"
    ),
    "sha256": "c7eee69e343d6726b4f1a2ef0443c3293f33dd5ddbb16053ce03369a9ebd28e5",
}
V5 = {
    "path": str(
        ROOT
        / "outputs/tracking_preview_wam_development_v5_20261028/development/service/requests/150_task0_init0_s43_raw_q00/completed.json"
    ),
    "sha256": "cfcae25a27c75f0ef5249378018ce1893eeb007d4878b4778d5b668b15b65c12",
}
HISTORICAL_AUDIT = {
    "path": str(ROOT / "outputs/reset_start_feedback_audit_20261111/result.json"),
    "sha256": "61606588d8298f915b5b043a9c1fc6d13f695c83dc96b51f64a63633ed339b97",
}


def same_request_and_response(old_request, old_response, new_request, new_response):
    """Allow only names and image-copy paths to differ for the same seed."""
    old, new = copy.deepcopy(old_request), copy.deepcopy(new_request)
    for key in ("name", "vision_path"):
        old.pop(key)
        new.pop(key)
    prefix.require(
        old == new and old["seed"] == 43 and old["prompt"] == "put the bowl on the plate",
        "The historical and new requests must have identical seed and model conditioning",
    )
    compared = repro.compare_responses(old_response, new_response)
    prefix.require(
        all(value["elementwise_equal"] and value["different_elements"] == 0 for value in compared.values()),
        "The historical and new model output arrays must match elementwise",
    )
    return compared


def compare_replays(producer, checker):
    """Recalculate recorded tracking and replay differences from all eight periods."""
    prefix.require(
        len(producer) == len(checker) == 8
        and [row["step"] for row in producer] == [row["step"] for row in checker] == list(range(8)),
        "Eight corresponding recorded control periods are required",
    )
    maxima = {key: 0.0 for key in ("command", "actual_state", "tcp_error", "inner_time", "inner_q", "inner_qvel")}
    for actual, replayed in zip(producer, checker, strict=True):
        prefix.require(
            len(actual["actual_inner"]) == len(replayed["actual_inner"]) == 25
            and actual["endpoint_receipt"] == replayed["endpoint_receipt"]
            and actual["pose_caps_satisfied"] == replayed["pose_caps_satisfied"]
            and actual["task_success"] == replayed["task_success"]
            and actual["target_grasp"] == replayed["target_grasp"],
            "The recorded endpoints, labels and 25 inner samples must correspond",
        )
        for key in ("command", "actual_state", "tcp_error"):
            left, right = np.asarray(actual[key], dtype=float), np.asarray(replayed[key], dtype=float)
            prefix.require(
                left.shape == right.shape and np.isfinite(left).all() and np.isfinite(right).all(),
                "Finite replay arrays",
            )
            maxima[key] = max(maxima[key], float(np.max(np.abs(left - right))))
        for key, output_key in (("time", "inner_time"), ("q", "inner_q"), ("qvel", "inner_qvel")):
            left = np.asarray([row[key] for row in actual["actual_inner"]], dtype=float)
            right = np.asarray([row[key] for row in replayed["actual_inner"]], dtype=float)
            prefix.require(
                left.shape == right.shape and np.isfinite(left).all() and np.isfinite(right).all(),
                "Finite inner samples",
            )
            maxima[output_key] = max(maxima[output_key], float(np.max(np.abs(left - right))))
    errors = np.asarray([row["tcp_error"] for row in producer], dtype=float)
    position, rotation = np.linalg.norm(errors[:, :3], axis=1), np.linalg.norm(errors[:, 3:], axis=1)
    return dict(
        control_periods=8,
        inner_samples=200,
        position_rmse_m=float(np.sqrt(np.mean(position**2))),
        rotation_rmse_rad=float(np.sqrt(np.mean(rotation**2))),
        position_max_m=float(np.max(position)),
        rotation_max_rad=float(np.max(rotation)),
        pose_cap_endpoints=sum(bool(row["pose_caps_satisfied"]) for row in producer),
        task_success_endpoints=sum(bool(row["task_success"]) for row in producer),
        target_grasp_endpoints=sum(bool(row["target_grasp"]) for row in producer),
        minimum_inner_margin_rad=float(min(row["minimum_inner_margin_rad"] for row in producer)),
        replay_max_absolute_difference=maxima,
    )


def audit():
    """Bind three saved samples to one previously independently replayed short window."""
    new, v4, v5, historical = (prefix.read(ref) for ref in (NEW, V4, V5, HISTORICAL_AUDIT))
    prefix.require(
        new["complete"] is new["passed"] is historical["complete"] is historical["passed"] is True
        and all(
            row["complete"] is True and row["case_id"] == "task0_init0_s43_raw" and row["sampling_seed"] == 43
            for row in (v4, v5)
        ),
        "The completed seed-43 model and historical development cases are required",
    )
    model = prefix.read(new["model_ref"])
    new_request = prefix.read(model["responses"][0]["model_request_ref"])
    new_response = prefix.read(model["responses"][0]["actual_response_ref"])
    comparisons = {}
    for name, completed in (("v4", v4), ("v5", v5)):
        request, response = prefix.read(completed["request_ref"]), prefix.read(completed["response_ref"])
        prefix.checked(completed["image_ref"])
        prefix.require(
            completed["image_ref"]["sha256"] == new["audit"]["image_sha256"]
            and repro.sha256(new_request["vision_path"]) == completed["image_ref"]["sha256"],
            "The historical and new input image bytes must match",
        )
        comparisons[name] = same_request_and_response(request, response, new_request, new_response)
    case = historical["cases"][2]
    prefix.require(case["case"]["case_id"] == "task0_init0_s43_raw", "The corresponding historical case is required")
    produced, replayed = prefix.read(case["producer_ref"]), prefix.read(case["checker_ref"])
    producer_summary, checker_summary = prefix.read(historical["producer_ref"]), prefix.read(historical["checker_ref"])
    for phase, summary, branch_ref in (
        ("producer", producer_summary, case["producer_ref"]),
        ("checker", checker_summary, case["checker_ref"]),
    ):
        lifecycle = prefix.read(summary["workers_ref"])
        rows = [row for row in lifecycle["workers"] if row["key"] == "case_02_feedback"]
        prefix.require(
            summary["complete"] is summary["passed"] is summary["sources_unchanged"] is True
            and branch_ref in summary["branch_refs"]
            and len(rows) == 1
            and rows[0]["actual_wait"] is rows[0]["reaped"] is True
            and rows[0]["returncode"] == 0
            and rows[0]["signal_sent"] is False,
            f"The historical {phase} worker must have exited and been reaped",
        )
        prefix.checked(rows[0]["log_ref"])
    prefix.require(
        produced["complete"] is produced["passed"] is replayed["complete"] is replayed["passed"] is True
        and produced["case"] == replayed["case"] == case["case"]
        and replayed["candidate_modules_unloaded"] is True
        and produced["runtime"]["reset_image_match"] is replayed["runtime"]["reset_image_match"] is True
        and produced["runtime"]["reset_endpoint_comparison"]["response_sha256"] == v4["response_ref"]["sha256"]
        and produced["runtime"]["reset_endpoint_comparison"]["recorded_state_sha256"]
        == produced["runtime"]["reset_endpoint_comparison"]["actual_state_sha256"]
        == new["audit"]["recorded_initial"]["state_sha256"]
        and produced["runtime"]["controller_contract"] == replayed["runtime"]["controller_contract"]
        and produced["runtime"]["controller_contract"]["control_freq"] == 20
        and produced["runtime"]["controller_contract"]["model_timestep"] == 0.002,
        "The same recorded reset, historical reference and fixed control interface are required",
    )
    metrics = compare_replays(
        produced["runtime"]["continuation"]["periods"], replayed["runtime"]["continuation"]["periods"]
    )
    expected = case["candidate"]
    prefix.require(
        all(value == 0 for value in metrics["replay_max_absolute_difference"].values())
        and all(
            abs(metrics[key] - expected[key]) <= 1e-15
            for key in ("position_rmse_m", "rotation_rmse_rad", "position_max_m", "rotation_max_rad")
        )
        and metrics["pose_cap_endpoints"] == expected["pose_cap_endpoints"]
        and metrics["task_success_endpoints"] == expected["task_success_endpoints"]
        and metrics["target_grasp_endpoints"] == expected["target_grasp_endpoints"]
        and abs(metrics["minimum_inner_margin_rad"] - case["minimum_inner_margin_rad"]) <= 1e-15,
        "Recomputed eight-period tracking and independent replay must match the original audit",
    )
    return dict(
        complete=True,
        passed=True,
        evidence_level="retrospective_reference_equivalence_not_new_closed_loop",
        control_qualified=False,
        new_result_ref=NEW,
        historical_v4_ref=V4,
        historical_v5_ref=V5,
        historical_feedback_audit_ref=HISTORICAL_AUDIT,
        historical_producer_ref=case["producer_ref"],
        historical_checker_ref=case["checker_ref"],
        audit_source_ref=prefix.reference(Path(__file__).resolve()),
        same_saved_reference=comparisons,
        historical_short_window=metrics,
        new_model_initializations=0,
        new_model_requests=0,
        new_simulation_models=0,
        new_physics_steps=0,
        fresh_cases=0,
    )


def main():
    prefix.environment()
    print(publish(OUTPUT / "result.json", audit()), flush=True)


if __name__ == "__main__":
    main()
