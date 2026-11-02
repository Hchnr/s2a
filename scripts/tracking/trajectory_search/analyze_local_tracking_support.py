"""Optimistic per-cap support bounds of saved local models, never control actions."""

import os
import signal
import traceback

import numpy as np
from scipy.optimize import linprog

from s2a.infrastructure.publication import publish_result
from s2a.research.trajectory_search.restored_anchor_sqp_evidence import REPO, read, reference, require
from scripts.maintenance.run_cpu_tests import CPU_ENV, snapshot
from scripts.tracking.trajectory_search.probe_rotation_constraints import reconstruct_models

OUTPUT = REPO / "outputs/local_tracking_support_20261102"
CONFIG = REPO / "configs/tracking/trajectory_search/local_tracking_support_20261102.json"
PLAN = REPO / "plans/v0.0.1/tracking/trajectory_search/local_tracking_support_20261102.md"
QA = REPO / "outputs/qa/local_tracking_support_20261102/result.json"


def support_rows(entry, Js, previous, lower, upper, config, fraction):
    """Keep original linear safety/command/rate rows; omit radius and pose ceiling."""
    require(fraction in (0.0, 0.9), "Only diagnostic buffer or registered reserve floor")
    x = np.asarray(entry["commands"], dtype=float).ravel()
    safety = np.asarray(entry["safety_slack"], dtype=float)
    Js = np.asarray(Js, dtype=float)
    lower, upper = np.asarray(lower).ravel(), np.asarray(upper).ravel()
    previous = np.asarray(previous, dtype=float)
    require(x.shape == lower.shape == upper.shape == (72,), "Original H12 command domain")
    require(safety.shape == (4369,) and Js.shape == (4369, 72), "All original safety rows")
    require(previous.shape == (6,) and np.all(lower <= upper), "Original previous command and ordered domain")
    identity = np.eye(72)
    difference = identity - np.eye(72, k=-6)
    center = np.r_[previous, np.zeros(66)]
    weights = np.r_[np.ones(4368), config["terminal_time_scale"]]
    floor = config["safety_buffer"] + fraction * config["reserve_target"] / weights
    A = np.vstack((Js, difference, -difference, identity, -identity))
    b = np.r_[safety - floor, 0.25 + difference @ x - center, 0.25 - difference @ x + center, x - lower, upper - x]
    require(np.isfinite(A).all() and np.isfinite(b).all(), "Finite optimistic constraints")
    return A, b


def audit_support(A, b, c, point, dual, tolerance=1e-8):
    """Independent primal/dual algebra in original units; no solver status trusted."""
    A, b, c, point, dual = (np.asarray(v, dtype=float) for v in (A, b, c, point, dual))
    require(A.ndim == 2, "Constraint matrix")
    require(b.shape == dual.shape == (len(A),) and c.shape == point.shape == (A.shape[1],), "LP dimensions")
    require(all(np.isfinite(v).all() for v in (A, b, c, point, dual)), "Finite LP certificate")
    require(0 < tolerance <= 1e-8, "Fixed or stricter diagnostic tolerance")
    scale = np.maximum(1.0, np.maximum(np.max(np.abs(A), axis=1), np.abs(b)))
    slack = A @ point + b
    residual = c - A.T @ dual
    primal, bound = float(c @ point), float(-b @ dual)
    stationarity = float(np.max(np.abs(residual)) / max(1.0, np.max(np.abs(c)), np.max(np.abs(A.T @ dual))))
    gap = abs(primal - bound) / max(1.0, abs(primal), abs(bound))
    require(np.min(slack / scale) >= -tolerance, "LP primal feasibility")
    require(np.min(dual) >= -tolerance, "Nonnegative original-unit LP dual")
    require(stationarity <= tolerance and gap <= tolerance, "LP stationarity and objective gap")
    return dict(
        primal_min_scaled=float(np.min(slack / scale)),
        dual_min=float(np.min(dual)),
        stationarity=stationarity,
        relative_gap=gap,
        primal_objective=primal,
        dual_bound=bound,
    )


def solve_support(A, b, c, spec):
    scale = np.maximum(1.0, np.maximum(np.max(np.abs(A), axis=1), np.abs(b)))
    result = linprog(
        c,
        A_ub=-A / scale[:, None],
        b_ub=b / scale,
        bounds=[(None, None)] * len(c),
        method="highs",
        options=dict(
            time_limit=spec["lp_time_limit_seconds"],
            maxiter=spec["lp_max_iterations"],
            primal_feasibility_tolerance=1e-9,
            dual_feasibility_tolerance=1e-9,
        ),
    )
    record = dict(
        success=bool(result.success), status=int(result.status), message=result.message, iterations=int(result.nit)
    )
    if result.success:
        point, dual = result.x, -result.ineqlin.marginals / scale
        record.update(
            point=point, dual=dual, certificate=audit_support(A, b, c, point, dual, spec["certificate_tolerance"])
        )
    return record


def box_corrected_bound(A, b, c, dual, lower, upper):
    """Conservatively charge any dual residual over explicit finite box bounds."""
    lower, upper = np.asarray(lower), np.asarray(upper)
    require(lower.shape == upper.shape == np.asarray(c).shape and np.all(lower <= upper), "Finite step box")
    require(np.isfinite(lower).all() and np.isfinite(upper).all(), "Finite step bounds")
    nonnegative = np.maximum(np.asarray(dual), 0.0)
    residual = c - A.T @ nonnegative
    correction = float(np.sum(np.minimum(residual * lower, residual * upper)))
    return dict(bound=float(-b @ nonnegative + correction), residual_box_correction=correction)


def main():
    for key, value in CPU_ENV.items():
        require(os.environ.get(key) == value, "Explicit CPU environment: " + key)
    spec = read(reference(CONFIG))
    require(
        spec
        == dict(
            seed=42,
            anchor_indices=[0, 23],
            reserve_fractions=[0.0, 0.9],
            lp_budget=48,
            lp_time_limit_seconds=10,
            lp_max_iterations=10000,
            certificate_tolerance=1e-8,
            owner_timeout_seconds=1200,
            automatic_retry=False,
            new_response_budget=0,
            new_model_budget=0,
        ),
        "Exact diagnostic budget",
    )
    OUTPUT.mkdir(exist_ok=False)
    before = snapshot(REPO)
    qa = read(reference(QA))
    require(qa["passed"] and qa["all_tests_passed"] and qa["source_unchanged"] and qa["reaped"], "Closed diagnostic QA")
    qa_sources = read(qa["source_after_ref"])
    require(
        all(qa_sources[k] == before[k] for k in ("sources", "current_map_ref", "driver_ref")),
        "QA current source binding",
    )
    refs = []

    def load(path, sha=None):
        ref = reference(path)
        require(sha is None or ref["sha256"] == sha, "Fixed input SHA")
        refs.append(ref)
        return read(ref)

    repaired = load(
        REPO / "outputs/small_progress_native_repair_20261102/result.json",
        "52a41511b116e958a3bb7ba43adbfaf1485eb351be1e88a5349e4a79eae0ac90",
    )
    require(
        repaired["passed"] and repaired["sources_unchanged"] and repaired["verified_responses"] == 36,
        "Complete independent response admission",
    )
    failed = load(
        REPO / "outputs/small_progress_search_20261102/small_progress/result.json",
        "844c4e05a24b41dda33e36aac180bb421afa73e63f88f8eb70b213615f6e4c5a",
    )
    producer = load(failed["producer_final_ref"]["path"], failed["producer_final_ref"]["sha256"])
    answer = producer["answer"]
    stencil = load(
        REPO / "outputs/anchor85_coupled_stencil_20261102/stencil.json",
        "f4bc18043a6c1de5ae40e3c266eb7b5cb4cbe1ea9ac9af137ecf6e2058cac417",
    )
    bundle = load(
        REPO / "outputs/constraint_residual_repair_inputs_20261029/bundle.json",
        "eadf18a3ed90c29ee70d773741cc310304f4385189baea360e534ca6ef5bd023",
    )
    registration = publish_result(
        OUTPUT / "registration.json",
        dict(
            spec=spec,
            config_ref=reference(CONFIG),
            plan_ref=reference(PLAN),
            qa_ref=reference(QA),
            input_refs=refs,
            source_before_ref=publish_result(OUTPUT / "source_before.json", before),
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        pid=os.getpid(),
        registration_ref=registration,
        lp_attempts=0,
        responses=0,
        models=0,
        physics_steps=0,
        gpu_calls=0,
        control_qualified=False,
        records=[],
    )
    previous = np.asarray(bundle["source"]["previous_command"])[:6]
    try:
        for index in spec["anchor_indices"]:
            stop = 0 if index == 0 else answer["total_stages"]
            models = reconstruct_models(answer, stencil, stop)
            entry = answer["evaluated_candidates"][index]
            error = (np.asarray(entry["tcp_errors"]) / np.array([0.01, 0.01, 0.01, 0.05, 0.05, 0.05])).reshape(24, 3)
            require(np.count_nonzero(np.linalg.norm(error, axis=1) > 1) == 12, "Fixed twelve violated caps")
            for fraction in spec["reserve_fractions"]:
                A, b = support_rows(
                    entry,
                    models["Js"],
                    previous,
                    answer["global_lower"],
                    answer["global_upper"],
                    answer["config"],
                    fraction,
                )
                constraints = publish_result(OUTPUT / f"constraints_{index}_{fraction}.json", dict(A=A, b=b))
                for group in np.flatnonzero(np.linalg.norm(error, axis=1) > 1):
                    norm = np.linalg.norm(error[group])
                    normal = error[group] / norm
                    c = normal @ models["Je"][3 * group : 3 * group + 3]
                    require(result["lp_attempts"] < spec["lp_budget"], "LP budget")
                    result["lp_attempts"] += 1
                    record = solve_support(A, b, c, spec)
                    record.update(
                        anchor_index=index,
                        reserve_fraction=fraction,
                        group=int(group),
                        period=int(group // 2),
                        kind="position" if group % 2 == 0 else "rotation",
                        error_norm=float(norm),
                        normal=normal,
                        c=c,
                        constraints_ref=constraints,
                    )
                    if record["success"]:
                        x = np.asarray(entry["commands"]).ravel()
                        corrected = box_corrected_bound(
                            A,
                            b,
                            c,
                            record["dual"],
                            np.asarray(answer["global_lower"]).ravel() - x,
                            np.asarray(answer["global_upper"]).ravel() - x,
                        )
                        record["corrected_dual"] = corrected
                        record["support_lower_bound"] = float(norm + corrected["bound"])
                        record["linear_model_excluded"] = (
                            record["support_lower_bound"] > 1 + spec["certificate_tolerance"]
                        )
                    ref = publish_result(OUTPUT / f"lp_{result['lp_attempts']:02d}.json", record)
                    result["records"].append(
                        dict(
                            ref=ref,
                            **{k: record[k] for k in ["anchor_index", "reserve_fraction", "group", "success"]},
                            support_lower_bound=record.get("support_lower_bound"),
                            linear_model_excluded=record.get("linear_model_excluded"),
                        )
                    )
                print(f"anchor={index} reserve={fraction} LPs={result['lp_attempts']}", flush=True)
        result.update(
            complete=True, passed=result["lp_attempts"] == 48 and all(r["success"] for r in result["records"])
        )
    except BaseException as error:
        result.update(error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        after = snapshot(REPO)
        result["sources_unchanged"] = all(before[k] == after[k] for k in ("sources", "current_map_ref", "driver_ref"))
        result["inputs_unchanged"] = all(reference(r["path"]) == r for r in refs)
        result["passed"] &= result["sources_unchanged"] and result["inputs_unchanged"]
        result["source_after_ref"] = publish_result(OUTPUT / "source_after.json", after)
        print(publish_result(OUTPUT / "result.json", result), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    signal.alarm(1200)
    raise SystemExit(main())
