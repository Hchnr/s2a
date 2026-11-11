"""Audit bounded reset feedback against independent replay and reviewed baseline."""

from pathlib import Path

import numpy as np

from s2a.geometry import pose_error
from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_start_feedback as study

ROOT = prefix.REPO
PRODUCER = {
    "path": str(ROOT / "outputs/reset_start_feedback_20261111/producer/result.json"),
    "sha256": "2b7fc9fa25241d938245e14bce7f485d9c887f21bb0110893dd212a128d2cc36",
}
CHECKER = {
    "path": str(ROOT / "outputs/reset_start_feedback_20261111/checker/result.json"),
    "sha256": "f94148242784bba0d0802ddb4db93c8f9fac921ef564e5b36680821820fb64f1",
}
QA = ROOT / "outputs/qa/reset_start_feedback_audit_v2_20261111/result.json"
OUTPUT = ROOT / "outputs/reset_start_feedback_audit_20261111/result.json"


def metrics(rows):
    """Report vector-norm RMSE and endpoint labels, never task qualification."""
    errors = np.asarray([row["error"] for row in rows], dtype=float)
    opening = np.asarray([row["opening_error"] for row in rows], dtype=float)
    if not rows or errors.shape != (len(rows), 6) or not np.isfinite(errors).all() or not np.isfinite(opening).all():
        raise ValueError("Finite six-dimensional errors and opening errors are required")
    pos = np.linalg.norm(errors[:, :3], axis=1)
    rot = np.linalg.norm(errors[:, 3:], axis=1)
    return dict(
        endpoints=len(rows),
        position_rmse_m=float(np.sqrt(np.mean(pos**2))),
        rotation_rmse_rad=float(np.sqrt(np.mean(rot**2))),
        position_max_m=float(pos.max()),
        rotation_max_rad=float(rot.max()),
        opening_mae=float(np.mean(np.abs(opening))),
        pose_cap_endpoints=int(np.count_nonzero((pos <= 0.01) & (rot <= 0.05))),
        task_success_endpoints=sum(row["task_success"] is True for row in rows),
        target_grasp_endpoints=sum(row["target_grasp"] is True for row in rows),
    )


def compare_replay(producer, checker):
    """Compare physical quantities, labels, and all sampled inner states."""
    maxima = {}

    def compare(name, left, right):
        a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"Finite matching replay shape required: {name}")
        delta = float(np.max(np.abs(a - b))) if a.size else 0.0
        maxima[name] = max(maxima.get(name, 0.0), delta)
        if delta > 1e-10:
            raise ValueError(f"Replay differs: {name}")

    if len(producer) != 8 or len(checker) != 8:
        raise ValueError("Exactly eight physical periods required in each replay")
    for i, (left, right) in enumerate(zip(producer, checker, strict=True)):
        if left["step"] != right["step"] or left["step"] != i:
            raise ValueError("Replay periods must be aligned")
        for key in ("command", "actual_state", "tcp_error", "minimum_inner_margin_rad"):
            compare(key, left[key], right[key])
        for key in (
            "pose_caps_satisfied",
            "task_success",
            "target_grasp",
            "inner_outside_samples",
            "inner_below_buffer_samples",
        ):
            if left[key] != right[key]:
                raise ValueError(f"Replay label differs: {key}")
        if left["endpoint_receipt"] != right["endpoint_receipt"]:
            raise ValueError("Independent endpoint receipts differ")
        if len(left["actual_inner"]) != 25 or len(right["actual_inner"]) != 25:
            raise ValueError("Each period must contain all 25 inner samples")
        for a, b in zip(left["actual_inner"], right["actual_inner"], strict=True):
            for key in ("time", "q", "qvel"):
                compare("inner_" + key, a[key], b[key])
        for key in ("opening", "opening_reference", "opening_error", "qpos", "qvel", "motor_target"):
            compare(
                "gripper_" + key, left["completed_feedback"]["score"][key], right["completed_feedback"]["score"][key]
            )
    return maxima


def lifecycle(owner):
    value = prefix.read(owner["workers_ref"])
    workers = value["workers"]
    prefix.require(
        value["launch_error"] is None
        and len(workers) == 12
        and all(
            row["key"] == study.branch_key(i, "feedback")
            and row["returncode"] == 0
            and row["actual_wait"] is row["reaped"] is True
            and row["signal_sent"] is False
            for i, row in enumerate(workers)
        ),
        "All twelve workers must exit and be waited and reaped",
    )
    for worker in workers:
        prefix.checked(worker["log_ref"])


def execute():
    prefix.environment()
    before = snapshot(ROOT)["sources"]
    refs = [
        prefix.reference(p)
        for p in (Path(__file__).resolve(), ROOT / "tests/research/native_runtime/test_reset_start_feedback_audit.py")
    ]
    qa = prefix.read(prefix.reference(QA))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and not qa["collection"]["collection_errors"]
        and prefix.read(qa["source_before_ref"])["sources"] == before,
        "A source-matched 32-worker CPU regression is required",
    )
    inputs = study.load_inputs(verify_assets=True)
    review = prefix.read(inputs["study_inputs"]["refs"]["review"])
    owners = [prefix.read(PRODUCER), prefix.read(CHECKER)]
    for owner, phase in zip(owners, ("producer", "checker"), strict=True):
        totals = owner["counts"]["totals"]
        prefix.require(
            owner["complete"] is owner["passed"] is owner["sources_unchanged"] is True
            and owner["control_qualified"] is owner["automatic_retry"] is False
            and owner["phase"] == phase
            and owner["world_model_calls"] == owner["fresh_cases"] == 0
            and len(owner["branch_refs"]) == 12
            and totals["models"] == totals["contexts"] == 36
            and totals["control_steps"] == 96
            and totals["native_steps"] == 2400
            and totals["short_window_passes"] == 12,
            "Both fixed physical phases must have completed",
        )
        lifecycle(owner)
        registration = prefix.read(owner["registration_ref"])
        prefix.require(
            registration["phase"] == phase and registration["inputs"] == study.input_receipt(inputs),
            "Registered inputs must match the admitted original cases",
        )
        for ref in registration["source_refs"]:
            prefix.checked(ref)
        if phase == "checker":
            prefix.require(
                registration["producer_branch_refs"]
                == {study.branch_key(i, "feedback"): ref for i, ref in enumerate(owners[0]["branch_refs"])},
                "Checker must replay these producer commands",
            )
    rows, all_baseline, all_candidate, all_times = [], [], [], []
    replay_max = {}
    for i, case in enumerate(inputs["cases"]):
        producer, checker = [prefix.read(owner["branch_refs"][i]) for owner in owners]
        prefix.require(
            producer["case"] == checker["case"] == case["case"]
            and producer["complete"] is producer["passed"] is checker["complete"] is checker["passed"] is True
            and checker["candidate_modules_unloaded"] is True,
            "Aligned cases and an isolated checker are required",
        )
        pr, cr = producer["runtime"], checker["runtime"]
        prefix.require(
            pr["reset_observation_verified"] is cr["reset_observation_verified"] is True
            and pr["reset_image_match"] is cr["reset_image_match"] is True
            and pr["previous_saved_command"] is cr["previous_saved_command"] is None
            and pr["continuation"]["stop"] is cr["continuation"]["stop"] is None,
            "Both phases must start at the admitted reset and complete without a stop",
        )
        actual, replay = pr["continuation"]["periods"], cr["continuation"]["periods"]
        for key, value in compare_replay(actual, replay).items():
            replay_max[key] = max(replay_max.get(key, 0.0), value)
        manifests = [p for p in case["original_source_sha256"] if p.endswith("/manifest.json")]
        prefix.require(len(manifests) == 1, "One original manifest required")
        baseline_ref = prefix.reference(Path(manifests[0]).parent / "steps.jsonl")
        prefix.require(
            review["source_and_artifact_sha256"].get(baseline_ref["path"]) == baseline_ref["sha256"],
            "Baseline log must be independently source-pinned",
        )
        original = prefix.read_recorded_prefix(prefix.checked(baseline_ref), 8, 600)
        baseline_rows, candidate_rows = [], []
        for step, (old, new) in enumerate(zip(original, actual, strict=True)):
            target = np.asarray(case["chunk"]["effective_targets_tcp"][step])
            opening = case["chunk"]["openings"][step]
            np.testing.assert_allclose(old["effective_desired_tcp"], target, atol=1e-10, rtol=0)
            np.testing.assert_allclose(old["opening_target"], opening, atol=1e-10, rtol=0)
            error = pose_error(target, np.asarray(old["after_source"]) @ np.asarray(case["tool"]))
            np.testing.assert_allclose(error, old["tcp_error"], atol=1e-10, rtol=0)
            np.testing.assert_allclose(
                new["completed_feedback"]["score"]["opening_reference"], opening, atol=1e-10, rtol=0
            )
            baseline_rows.append(
                dict(
                    error=error,
                    opening_error=old["opening_after"] - opening,
                    task_success=old["success"],
                    target_grasp=old["objects"][case["target_object"]]["grasp_contact"],
                )
            )
            candidate_rows.append(
                dict(
                    error=new["tcp_error"],
                    opening_error=new["completed_feedback"]["score"]["opening_error"],
                    task_success=new["task_success"],
                    target_grasp=new["target_grasp"],
                )
            )
            all_times.append(float(new["decision"]["total_decision_seconds"]))
        all_baseline.extend(baseline_rows)
        all_candidate.extend(candidate_rows)
        rows.append(
            dict(
                case=case["case"],
                baseline_ref=baseline_ref,
                producer_ref=owners[0]["branch_refs"][i],
                checker_ref=owners[1]["branch_refs"][i],
                baseline=metrics(baseline_rows),
                candidate=metrics(candidate_rows),
                minimum_inner_margin_rad=min(p["minimum_inner_margin_rad"] for p in actual),
            )
        )
    prefix.require(np.isfinite(all_times).all() and min(all_times) >= 0, "Finite measured decision times required")
    prefix.require(
        snapshot(ROOT)["sources"] == before and refs == [prefix.reference(p["path"]) for p in refs],
        "Sources must remain unchanged",
    )
    print(
        publish(
            OUTPUT,
            dict(
                complete=True,
                passed=True,
                control_qualified=False,
                producer_ref=PRODUCER,
                checker_ref=CHECKER,
                qa_ref=prefix.reference(QA),
                source_refs=refs,
                sources_unchanged=True,
                cases=rows,
                baseline=metrics(all_baseline),
                candidate=metrics(all_candidate),
                replay_max_absolute_difference=replay_max,
                decision_seconds=dict(
                    min=float(np.min(all_times)), median=float(np.median(all_times)), max=float(np.max(all_times))
                ),
                native_models=0,
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
                historical_baseline_has_inner_samples=False,
            ),
        ),
        flush=True,
    )


if __name__ == "__main__":
    execute()
