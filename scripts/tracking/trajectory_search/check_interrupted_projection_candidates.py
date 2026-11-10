"""Independently replay the retained buffered candidates of an interrupted run."""

import os
import signal
import traceback

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment, strict_native_environment
from s2a.infrastructure.publication import publish_result
from s2a.research.trajectory_search.restored_anchor_sqp_evidence import REPO, read, reference, require
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.tracking.trajectory_search.check_cap_distance_gn import phase_deadline
from scripts.tracking.trajectory_search.check_replay_order_stencil import compare_values
from scripts.tracking.trajectory_search.check_reserve_entry_search import native_metric_entry
from scripts.tracking.trajectory_search.response_probe_runtime import ResponseProbeRuntime
from scripts.verification.trajectory_search.check_tracking_v5_prefix_trust_control import check_full_candidate
from scripts.verification.trajectory_search.check_tracking_v5_saved_direction_backtracking import case_engine
from scripts.verification.trajectory_search.check_tracking_v5_tail_translation_reserve_v2 import semantic_compare

OUTPUT = REPO / "outputs/interrupted_projection_candidate_replay_20261110"
SOURCE = REPO / "outputs/nonlinear_feasibility_projection_20261102"
ASSESSMENT = REPO / "outputs/nonlinear_projection_interrupted_assessment_20261110/assessment.json"
CONFIG = REPO / "configs/tracking/trajectory_search/interrupted_projection_candidate_replay_20261110.json"
PLAN = REPO / "plans/v0.0.1/tracking/trajectory_search/interrupted_projection_candidate_replay_20261110.md"
QA = REPO / "outputs/qa/interrupted_projection_candidate_replay_20261110/result.json"
SAFE_INDICES = [73, 74, 79, 80, 110, 125, 141, 147, 153, 214, 219, 220, 226]
ADDED = {
    "configs/tracking/trajectory_search/interrupted_projection_candidate_replay_20261110.json",
    "plans/v0.0.1/tracking/trajectory_search/interrupted_projection_candidate_replay_20261110.md",
    "scripts/tracking/trajectory_search/check_interrupted_projection_candidates.py",
    "tests/research/trajectory_search/test_interrupted_projection_candidates.py",
}


def source_delta(old, current):
    """Allow only the four declared replay files, no drift in original sources."""
    require(all(old[k] == current[k] for k in ("current_map_ref", "driver_ref")), "Original map and CPU driver")
    require(
        all(current["sources"].get(p) == sha for p, sha in old["sources"].items()), "Every original source unchanged"
    )
    added = set(current["sources"]) - set(old["sources"])
    require(added == ADDED, "Exactly four registered replay files")
    return dict(changed=[], added=sorted(added), original_sources_unchanged=True)


def registered_indices(assessment, spec):
    expected = dict(
        seed=42,
        assessment_sha256="4e1151091730a4da382a101a7be8442f7fb700a95e1ca9e17ad33e093f48da93",
        safe_indices=SAFE_INDICES,
        unsafe_control_index=0,
        repeat_index=110,
        independent_response_budget=15,
        independent_inner_step_budget=4500,
        independent_query_budget=4681,
        independent_model_budget=1,
        timeout_seconds=1200,
        automatic_retry=False,
    )
    require(spec == expected, "Exact fixed replay selection and budgets")
    require(
        assessment["complete"]
        and assessment["all_records_self_consistent"]
        and assessment["source_unchanged"]
        and assessment["saved_production_responses"] == 236
        and assessment["safe_buffered_count"] == 13
        and assessment["original_result_and_ledger_missing"],
        "Exact interrupted producer assessment",
    )
    rows = assessment["saved_responses"]
    require(len(rows) == 236 and [r["index"] for r in rows] == list(range(236)), "Every contiguous saved record")
    require([r["index"] for r in rows if r["safe_buffered"]] == SAFE_INDICES, "All and only saved buffered candidates")
    require(not rows[0]["safe_buffered"] and assessment["best_minimax_safe"][0]["index"] == 110, "Fixed controls")
    return [0, *SAFE_INDICES, 110]


def summarize(entry, baseline):
    q = np.asarray(entry["normalized_squared_errors"])
    safety = np.asarray(entry["safety_slack"])
    error = np.asarray(entry["tcp_errors"])
    require(q.shape == (12, 2) and safety.shape == (4369,) and error.shape == (12, 6), "Complete native metrics")
    safe = bool(entry["audit"]["feasible"] and np.all(safety >= 1e-8))
    base = np.asarray(baseline["q"])
    excess = np.zeros_like(q)
    outside = q > 1
    excess[outside] = (q[outside] - 1) / (np.sqrt(q[outside]) + 1)
    return dict(
        safe_buffered=safe,
        complete_witness=bool(safe and np.all(q <= 1)),
        caps=int(np.count_nonzero(q <= 1)),
        prefix=entry["prefix_length"],
        max_q=float(q.max()),
        cap_loss=float(0.5 * np.sum(excess**2)),
        terminal_position_m=float(np.linalg.norm(error[-1, :3])),
        terminal_rotation_rad=float(np.linalg.norm(error[-1, 3:])),
        dominates_baseline_all_24=bool(safe and np.all(q <= base) and np.any(q < base)),
    )


def main():
    phase_deadline(1200)
    spec = read(reference(CONFIG))
    runtime = ResponseProbeRuntime(OUTPUT, QA)
    assessment = runtime.load(reference(ASSESSMENT), spec["assessment_sha256"])
    indices = registered_indices(assessment, spec)
    require(
        not (SOURCE / "result.json").exists() and not (SOURCE / "producer_final.json").exists(),
        "Original final result and solver ledger remain missing",
    )
    registration = runtime.load(assessment["registration_ref"])
    delta = source_delta(runtime.load(registration["source_before_ref"]), runtime.before)
    baseline = runtime.load(assessment["baseline_repair_ref"])
    require(baseline["passed"] and baseline["verified_responses"] == 36, "Independent baseline remains qualified")
    entries = [runtime.load(assessment["saved_responses"][i]["ref"]) for i in indices]
    runtime.context.finish()
    registered = publish_result(
        OUTPUT / "registration.json",
        dict(
            config_ref=reference(CONFIG),
            plan_ref=reference(PLAN),
            qa_ref=runtime.qa_ref,
            assessment_ref=reference(ASSESSMENT),
            indices=indices,
            source_delta=delta,
            source_before_ref=publish_result(OUTPUT / "source_before.json", runtime.before),
            input_refs=list(runtime.refs.values()),
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        pid=os.getpid(),
        registration_ref=registered,
        models=0,
        model_attempts=0,
        verified_responses=0,
        new_production_responses=0,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        control_qualified=False,
        original_experiment_complete=False,
        original_solver_ledger_available=False,
        automatic_retry=False,
        records=[],
    )
    native = dict(
        mj_forward_queries=0,
        numerical_inner_steps=0,
        max_jit_vs_python_goal_difference=0.0,
        jit_python_goal_differences=0,
    )
    try:
        result["environment_admission"] = restore_cv2_environment(
            runtime.guard["exact_probe_ref"], runtime.guard["runtime_source_refs"]
        )
        os.environ.pop("MUJOCO_GL", None)
        os.environ.pop("PYOPENGL_PLATFORM", None)
        result["model_attempts"] += 1
        reconstruct, compare_rollout, classify, check_audit, _, _ = case_engine(
            "prefix",
            runtime.source,
            runtime.model_reg,
            runtime.prefix_reg,
            native,
            runtime.refs,
            lambda n, a, b, tolerance=1e-10: semantic_compare(n, a, b, tolerance),
            input_context=runtime.context,
        )
        result["models"] += 1
        fresh = []
        for ordinal, (index, entry) in enumerate(zip(indices, entries, strict=True)):
            commands = np.asarray(entry["commands"])
            rollout, _ = reconstruct(commands)
            metric = classify(commands, rollout)
            check_full_candidate(
                entry,
                commands,
                rollout,
                metric,
                compare_rollout,
                check_audit,
                compare=lambda n, a, b, tolerance=1e-10: semantic_compare(n, a, b, tolerance),
            )
            require(
                all(r["exact"] for r in compare_values(entry["rollout"], rollout)), "Exact complete native response"
            )
            verified = native_metric_entry(entry, rollout, metric)
            fresh.append(verified)
            summary = summarize(verified, assessment["baseline"])
            expected = assessment["saved_responses"][index]
            require(
                summary["safe_buffered"] == expected["safe_buffered"]
                and summary["caps"] == expected["caps"]
                and summary["prefix"] == expected["prefix"],
                "Exact saved decision labels",
            )
            record_ref = publish_result(
                OUTPUT / "independent" / f"response_{ordinal:02d}_{index:03d}.json",
                dict(rollout=rollout, metrics=metric, summary=summary),
            )
            result["records"].append(dict(index=index, ordinal=ordinal, ref=record_ref, **summary))
            result["verified_responses"] += 1
            print(dict(verified=ordinal + 1, total=15, index=index), flush=True)
        require(
            all(r["exact"] for r in compare_values(fresh[indices.index(110)], fresh[-1])), "Exact independent repeat"
        )
        require(
            not result["records"][0]["safe_buffered"] and all(r["safe_buffered"] for r in result["records"][1:]),
            "Unsafe control and thirteen independently buffered candidates",
        )
        require(
            result["verified_responses"] == 15
            and native["mj_forward_queries"] == 4681
            and native["numerical_inner_steps"] == 4500
            and result["models"] == 1,
            "Exact replay costs",
        )
        result.update(
            complete=True,
            passed=True,
            exact_native_responses=True,
            repeat_exact=True,
            independently_safe_candidates=13,
            independently_qualified_witnesses=sum(r["complete_witness"] for r in result["records"][1:-1]),
            independent_dominates_baseline=sum(r["dominates_baseline_all_24"] for r in result["records"][1:-1]),
        )
    except BaseException as error:
        result.update(error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        signal.alarm(0)
        result["independent_counts"] = native
        try:
            runtime.context.finish()
            strict_native_environment()
            after = snapshot(REPO)
            result["source_after_ref"] = publish_result(OUTPUT / "source_after.json", after)
            result["sources_unchanged"] = all(
                runtime.before[k] == after[k] for k in ("sources", "current_map_ref", "driver_ref")
            )
            result["passed"] &= result["sources_unchanged"]
        except BaseException as error:
            result.update(source_error=repr(error), passed=False)
        print(publish_result(OUTPUT / "result.json", result), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
