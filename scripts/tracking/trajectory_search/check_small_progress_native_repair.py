"""Native-only completion of a retained producer whose config audit stopped."""

import copy
import os
import signal
import traceback

import numpy as np

from s2a.controllers.osc import OSCInterface
from s2a.infrastructure.native_environment import restore_cv2_environment, strict_native_environment
from s2a.infrastructure.publication import publish_result, result_json
from s2a.research.trajectory_search.restored_anchor_sqp_evidence import REPO, read, reference, require
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.tracking.trajectory_search.cap_distance_search_audit import audit_cap_search
from scripts.tracking.trajectory_search.check_cap_distance_gn import phase_deadline
from scripts.tracking.trajectory_search.check_replay_order_stencil import compare_values
from scripts.tracking.trajectory_search.check_reserve_entry_search import native_metric_entry
from scripts.tracking.trajectory_search.reserve_aligned_cap_audit import ReserveAlignedCapAuditPolicy
from scripts.tracking.trajectory_search.response_probe_runtime import ResponseProbeRuntime
from scripts.verification.trajectory_search.check_tracking_v5_prefix_trust_control import check_full_candidate
from scripts.verification.trajectory_search.check_tracking_v5_saved_direction_backtracking import case_engine
from scripts.verification.trajectory_search.check_tracking_v5_tail_translation_reserve_v2 import semantic_compare

OUTPUT = REPO / "outputs/small_progress_native_repair_20261102"
CONFIG = REPO / "configs/tracking/trajectory_search/small_progress_native_repair_20261102.json"
PLAN = REPO / "plans/v0.0.1/tracking/trajectory_search/small_progress_native_repair_20261102.md"
QA = REPO / "outputs/qa/small_progress_native_repair_20261102/result.json"
CHANGED = {
    "scripts/tracking/trajectory_search/objective_search_audit.py",
    "scripts/tracking/trajectory_search/cap_distance_search_audit.py",
    "scripts/tracking/trajectory_search/check_reserve_entry_search.py",
}
ADDED = {
    "scripts/tracking/trajectory_search/check_small_progress_native_repair.py",
    "tests/research/trajectory_search/test_independent_progress_config.py",
    "configs/tracking/trajectory_search/small_progress_native_repair_20261102.json",
    "plans/v0.0.1/tracking/trajectory_search/small_progress_native_repair_20261102.md",
}


def source_delta(old, current):
    require(all(old[k] == current[k] for k in ("current_map_ref", "driver_ref")), "Same approved layout and CPU driver")
    changed = []
    for path, sha in old["sources"].items():
        require(path in current["sources"], "No deletion from original execution sources")
        if sha != current["sources"][path]:
            require(path in CHANGED, "Original numerical/control source unchanged: " + path)
            changed.append(path)
    added = set(current["sources"]) - set(old["sources"])
    require(added <= ADDED, "Only explicitly registered repair files added")
    return dict(changed=sorted(changed), added=sorted(added), numerical_sources_unchanged=True)


def main():
    phase_deadline(300)
    spec = read(reference(CONFIG))
    require(
        spec
        == dict(
            seed=42,
            source_result_sha256="844c4e05a24b41dda33e36aac180bb421afa73e63f88f8eb70b213615f6e4c5a",
            source_production_responses=36,
            new_independent_responses=36,
            new_query_budget=11233,
            new_inner_step_budget=10800,
            new_model_budget=1,
            new_qp_budget=0,
            new_nnls_budget=0,
            timeout_seconds=1200,
            automatic_retry=False,
        ),
        "Fixed native-only budget",
    )
    runtime = ResponseProbeRuntime(OUTPUT, QA)
    failed_ref = reference(REPO / "outputs/small_progress_search_20261102/small_progress/result.json")
    failed = runtime.load(failed_ref, spec["source_result_sha256"])
    require(
        not failed["complete"]
        and not failed["passed"]
        and failed["sources_unchanged"]
        and failed["models"] == 1
        and failed["verified_responses"] == 0
        and "minimum_predicted_reduction" in failed["error"],
        "Exact retained configuration-audit failure",
    )
    producer = runtime.load(failed["producer_final_ref"])
    answer = producer["answer"]
    require(
        producer["complete"] and producer["response_evaluations"] == 36 and producer["query_count"] == 11233,
        "Complete retained producer, no rerun",
    )
    registered = runtime.load(failed["registration_ref"])
    delta = source_delta(runtime.load(registered["source_before_ref"]), runtime.before)
    stencil = runtime.load(
        registered["initial_model_ref"], "f4bc18043a6c1de5ae40e3c266eb7b5cb4cbe1ea9ac9af137ecf6e2058cac417"
    )
    prior = runtime.load(
        reference(REPO / "outputs/cap_distance_gn_20261102/result.json"),
        "80f58cde84778d1e3b2f7ff3ca0b85c5414419da976a057710421d23f03012e9",
    )
    cap = runtime.load(prior["producer_final_ref"])["answer"]
    root = cap["evaluated_candidates"][0]
    models = {k: np.asarray(stencil["matrices"][k]) for k in ("Js", "Jq", "Je")}
    previous = np.asarray(runtime.source["previous_command"])[:6]
    interface = OSCInterface(*(runtime.control[k] for k in ("input_min", "input_max", "output_min", "output_max")))

    def audit(record):
        return audit_cap_search(
            record,
            models["Js"],
            models["Jq"],
            models["Je"],
            stencil["base"],
            previous,
            interface.input_min,
            interface.input_max,
            answer["config"],
            domain_reference=root,
            preserve_pose_reserve=True,
            reserve_safety_feedback=True,
            objective_audit=ReserveAlignedCapAuditPolicy(),
            progress_reference=cap["config"],
        )

    runtime.context.finish()
    registration = publish_result(
        OUTPUT / "registration.json",
        dict(
            spec=spec,
            config_ref=reference(CONFIG),
            plan_ref=reference(PLAN),
            qa_ref=runtime.qa_ref,
            failed_source_ref=failed_ref,
            source_producer_ref=failed["producer_final_ref"],
            source_delta=delta,
            source_before_ref=publish_result(OUTPUT / "source_before.json", runtime.before),
            input_refs=list(runtime.refs.values()),
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        pid=os.getpid(),
        registration_ref=registration,
        models=0,
        model_attempts=0,
        verified_responses=0,
        new_qp_solves=0,
        new_nnls_fits=0,
        source_failed_result_preserved=True,
        control_qualified=False,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    native = dict(
        mj_forward_queries=0,
        numerical_inner_steps=0,
        max_jit_vs_python_goal_difference=0.0,
        jit_python_goal_differences=0,
    )
    try:
        result["production_ledger"] = audit(answer)
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
        phase_deadline(1200)
        fresh = copy.deepcopy(answer)
        exact = True
        for i, entry in enumerate(answer["evaluated_candidates"]):
            command = np.asarray(entry["commands"])
            rollout, _ = reconstruct(command)
            metric = classify(command, rollout)
            check_full_candidate(
                entry,
                command,
                rollout,
                metric,
                compare_rollout,
                check_audit,
                compare=lambda n, a, b, tolerance=1e-10: semantic_compare(n, a, b, tolerance),
            )
            exact = exact and all(r["exact"] for r in compare_values(entry["rollout"], rollout))
            fresh["evaluated_candidates"][i] = native_metric_entry(entry, rollout, metric)
            result["verified_responses"] += 1
            publish_result(OUTPUT / "independent" / f"response_{i:03d}.json", dict(rollout=rollout, metrics=metric))
            if i % 12 == 0:
                print(result_json(dict(verified=i + 1, total=36)), flush=True)
        require(exact, "Exact every independent response")
        result["native_decision_ledger"] = audit(fresh)
        repeat = all(
            r["exact"] for r in compare_values(fresh["evaluated_candidates"][0], fresh["evaluated_candidates"][-1])
        )
        require(
            repeat and answer["initial_comparison"]["exact"] and answer["repeatability"]["exact"],
            "Exact initial/history/repeat",
        )
        qps = [s["qp"] for s in answer["stages"] if s["qp"] is not None]
        require(
            result["verified_responses"] == 36
            and native["mj_forward_queries"] == 11233
            and native["numerical_inner_steps"] == 10800,
            "Exact native-only costs",
        )
        result.update(
            complete=True,
            passed=True,
            exact_native_responses=True,
            native_repeat_exact=repeat,
            retained_qp_solves=len(qps),
            retained_nnls_fit_attempts=sum(q["nnls_fit_attempts"] for q in qps),
            cumulative_responses=producer["response_evaluations"] + result["verified_responses"],
            cumulative_queries=producer["query_count"] + native["mj_forward_queries"],
            cumulative_inner_steps=producer["numerical_inner_steps"] + native["numerical_inner_steps"],
            cumulative_models=2,
        )
    except BaseException as error:
        result.update(error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        signal.alarm(0)
        try:
            runtime.context.finish()
            strict_native_environment()
            after = snapshot(REPO)
            result["source_after_ref"] = publish_result(OUTPUT / "source_after.json", after)
            result["sources_unchanged"] = all(
                runtime.before[k] == after[k] for k in ("sources", "current_map_ref", "driver_ref")
            )
            result["passed"] = result["passed"] and result["sources_unchanged"]
        except BaseException as error:
            result.update(source_error=repr(error), passed=False)
        result["independent_counts"] = native
        receipt = publish_result(OUTPUT / "result.json", result)
        print(result_json(dict(result_ref=receipt, complete=result["complete"], passed=result["passed"])), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
