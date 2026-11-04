"""Collect and independently qualify a command-feasible coupled local stencil."""

import concurrent.futures
import multiprocessing
import os
import signal
import traceback

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment, strict_native_environment
from s2a.infrastructure.publication import publish_result, result_json
from s2a.research.trajectory_search.command_domain_stencil import fit_causal_response, select_interval_basis
from s2a.research.trajectory_search.interior_trajectory_search import causal_periods
from s2a.research.trajectory_search.restored_anchor_sqp_evidence import REPO, read, reference, require
from s2a.research.trajectory_search.state_vector_gn_search import normalized_error_vector
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.tracking.trajectory_search.check_replay_order_stencil import compare_values
from scripts.tracking.trajectory_search.coupled_stencil_runtime import (
    exact_prefix_observations,
    independent_causal_fit,
    production_batch,
)
from scripts.tracking.trajectory_search.response_probe_runtime import ResponseProbeRuntime
from scripts.verification.trajectory_search.check_tracking_v5_prefix_trust_control import check_full_candidate
from scripts.verification.trajectory_search.check_tracking_v5_saved_direction_backtracking import case_engine
from scripts.verification.trajectory_search.check_tracking_v5_tail_translation_reserve_v2 import semantic_compare

OUTPUT = REPO / "outputs/anchor85_coupled_stencil_20261102"
CONFIG = REPO / "configs/tracking/trajectory_search/anchor85_coupled_stencil_20261102.json"
PLAN = REPO / "plans/v0.0.1/tracking/trajectory_search/anchor85_coupled_stencil_20261102.md"
QA = REPO / "outputs/qa/anchor85_coupled_stencil_20261102/result.json"


def interrupted(signum, frame):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise TimeoutError("Owned CPU experiment terminated; reap workers before exit")


def output_arrays(entry):
    return [
        np.asarray(entry["safety_slack"]),
        np.asarray(entry["normalized_squared_errors"]).ravel(),
        normalized_error_vector(entry),
    ]


def main():
    signal.signal(signal.SIGTERM, interrupted)
    spec = read(reference(CONFIG))
    require(
        spec
        == dict(
            seed=42,
            jobs=32,
            anchor_index=85,
            nominal_step=1e-4,
            max_prefix_condition=10.0,
            production_responses=74,
            independent_responses=74,
            response_budget=148,
            inner_step_budget=44400,
            query_budget=46209,
            model_budget=33,
            qp_budget=0,
            nnls_budget=0,
            linear_fit_groups=6,
            timeout_seconds=1800,
            automatic_retry=False,
        ),
        "Fixed collection and fit budget",
    )
    runtime = ResponseProbeRuntime(OUTPUT, QA)
    old = runtime.load(
        reference(REPO / "outputs/cap_distance_gn_20261102/result.json"),
        "80f58cde84778d1e3b2f7ff3ca0b85c5414419da976a057710421d23f03012e9",
    )
    cap = runtime.load(old["producer_final_ref"])["answer"]
    anchor = cap["evaluated_candidates"][85]
    x = np.asarray(anchor["commands"])
    basis = select_interval_basis(
        x,
        np.asarray(runtime.source["previous_command"])[:6],
        np.asarray(runtime.control["input_min"]),
        np.asarray(runtime.control["input_max"]),
        np.asarray(cap["global_lower"]),
        np.asarray(cap["global_upper"]),
    )
    saved = runtime.load(
        reference(REPO / "outputs/anchor85_sampling_domain_20261102/coupled_basis.json"),
        "a35a48af6fed010eea746c4e6c4d4463de7a4ed817d88c9cfeed4c86ad22ddc7",
    )
    for new, old_case in zip(basis["chosen"], saved["chosen"], strict=True):
        for key in ("length", "start", "channel", "sign"):
            require(new[key] == old_case[key], "Fixed chosen direction")
        np.testing.assert_array_equal(new["commands"], old_case["commands"])
        np.testing.assert_array_equal(new["actual_displacement"], old_case["actual_displacement"])
    commands = [x, x.copy()] + [c["commands"] for c in basis["chosen"]]
    runtime.context.finish()
    registration = publish_result(
        OUTPUT / "registration.json",
        dict(
            spec=spec,
            config_ref=reference(CONFIG),
            plan_ref=reference(PLAN),
            qa_ref=runtime.qa_ref,
            input_refs=list(runtime.refs.values()),
            source_snapshot_ref=publish_result(OUTPUT / "source_before.json", runtime.before),
            basis=basis,
            commands=commands,
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        pid=os.getpid(),
        registration_ref=registration,
        batches=[],
        processes=[],
        new_responses=0,
        new_queries=0,
        new_inner_steps=0,
        independent_responses=0,
        model_attempts=0,
        models=0,
        mj_forward_queries=0,
        numerical_inner_steps=0,
        max_jit_vs_python_goal_difference=0.0,
        jit_python_goal_differences=0,
        linear_fit_groups=0,
        physics_steps=0,
        world_model_calls=0,
        qp_solves=0,
        nnls_fits=0,
        fresh_cases=0,
        control_qualified=False,
        automatic_retry=False,
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
            result,
            runtime.refs,
            lambda n, a, b, tolerance=1e-10: semantic_compare(n, a, b, tolerance),
            input_context=runtime.context,
        )
        result["models"] += 1
        batches = [list(enumerate(commands))[rank::32] for rank in range(32)]
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=32, mp_context=multiprocessing.get_context("spawn")
        )
        handles = []
        try:
            futures = [
                executor.submit(
                    production_batch,
                    OUTPUT,
                    rank,
                    batch,
                    runtime.source,
                    runtime.control,
                    runtime.model_reg["xml_ref"],
                    runtime.source["site_from_tcp"],
                )
                for rank, batch in enumerate(batches)
            ]
            handles = list(executor._processes.values())
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch = future.result()
                except BaseException as error:
                    result.setdefault("worker_errors", []).append(dict(error=repr(error), costs_unknown=True))
                    continue
                result["batches"].append(batch)
                for target, key in [
                    ("new_responses", "responses"),
                    ("new_queries", "queries"),
                    ("new_inner_steps", "inner_steps"),
                    ("models", "models"),
                    ("model_attempts", "model_attempts"),
                ]:
                    result[target] += batch[key]
                print(result_json(dict(completed_batches=len(result["batches"]), total=32)), flush=True)
        finally:
            executor.shutdown(wait=True)
            for process in handles:
                process.join()
                result["processes"].append(
                    dict(pid=process.pid, exitcode=process.exitcode, actual_wait=True, reaped=True)
                )
        require(
            len(result["batches"]) == len(result["processes"]) == 32 and all(b["passed"] for b in result["batches"]),
            "All batches completed",
        )
        require(all(p["exitcode"] == 0 for p in result["processes"]), "All worker processes reaped normally")
        records = sorted([r for b in result["batches"] for r in b["records"]], key=lambda r: r["index"])
        require([r["index"] for r in records] == list(range(74)), "Exactly every registered response once")
        entries = []
        independent = []
        exact = True
        for index, record in enumerate(records):
            e = read(record["ref"])
            native, _ = reconstruct(commands[index])
            result["independent_responses"] += 1
            metric = classify(commands[index], native)
            check_full_candidate(
                e,
                commands[index],
                native,
                metric,
                compare_rollout,
                check_audit,
                compare=lambda n, a, b, tolerance=1e-10: semantic_compare(n, a, b, tolerance),
            )
            comparison = compare_values(e["rollout"], native)
            exact = exact and all(r["exact"] for r in comparison)
            native_entry = dict(
                commands=commands[index],
                safety_slack=metric["safety_slack"],
                normalized_squared_errors=metric["normalized_squared_errors"],
                tcp_errors=metric["tcp_errors"],
            )
            for left, right in zip(output_arrays(e), output_arrays(native_entry), strict=True):
                require(np.array_equal(left, right), "Exact independently observed model outputs")
            publish_result(
                OUTPUT / "independent" / f"response_{index:02d}.json",
                dict(entry=native_entry, metrics=metric, rollout=native, comparison=comparison),
            )
            entries.append(e)
            independent.append(native_entry)
            if index % 12 == 0:
                print(result_json(dict(verified=index + 1, total=74)), flush=True)
        require(
            exact and all(r["exact"] for r in compare_values(entries[0], anchor)),
            "Exact responses and historical anchor",
        )
        require(all(r["exact"] for r in compare_values(entries[0], entries[1])), "Exact complete initial repeat")
        D = basis["displacements"]
        sp, qp = causal_periods()
        matrices = {}
        fits = {}
        for k, periods, name in zip(range(3), [sp, qp, np.repeat(np.arange(12), 6)], ["Js", "Jq", "Je"], strict=True):
            base = output_arrays(entries[0])[k]
            Y = np.asarray([output_arrays(e)[k] - base for e in entries[2:]])
            native_base = output_arrays(independent[0])[k]
            native_Y = np.asarray([output_arrays(e)[k] - native_base for e in independent[2:]])
            causal = exact_prefix_observations(D, Y, periods)
            fit = fit_causal_response(D, Y, periods)
            result["linear_fit_groups"] += 1
            audit = independent_causal_fit(D, native_Y, periods)
            result["linear_fit_groups"] += 1
            semantic_compare(name + "_independent_J", fit["jacobian"], audit["jacobian"], 1e-10)
            semantic_compare(name + "_independent_residual", fit["residual"], audit["residual"], 1e-10)
            matrices[name] = fit["jacobian"]
            fits[name] = dict(production=fit, independent=audit, observed_causality=causal)
        result["stencil_ref"] = publish_result(
            OUTPUT / "stencil.json",
            dict(
                base=entries[0],
                basis=basis,
                probe_refs=records[2:],
                matrices=matrices,
                fit_quality=fits,
                registration_ref=registration,
            ),
        )
        result.update(
            exact_native_responses=True,
            anchor_repeat_exact=True,
            historical_anchor_exact=True,
            total_queries=result["new_queries"] + result["mj_forward_queries"],
        )
        require(result["new_responses"] == result["independent_responses"] == 74, "All response costs")
        require(
            result["new_inner_steps"] + result["numerical_inner_steps"] == 44400 and result["total_queries"] == 46209,
            "All step/query costs",
        )
        require(result["models"] == 33 and result["linear_fit_groups"] == 6, "All model and fit costs")
        result.update(complete=True, passed=True)
    except BaseException as error:
        result.update(error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        try:
            runtime.context.finish()
            strict_native_environment()
            after = snapshot(REPO)
            result["source_after_ref"] = publish_result(OUTPUT / "source_after.json", after)
            result["sources_unchanged"] = all(
                runtime.before[k] == after[k] for k in ["sources", "current_map_ref", "driver_ref"]
            )
            result["passed"] = result["passed"] and result["sources_unchanged"]
        except BaseException as error:
            result.update(source_error=repr(error), passed=False)
        receipt = publish_result(OUTPUT / "result.json", result)
        print(result_json(dict(result_ref=receipt, complete=result["complete"], passed=result["passed"])), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
