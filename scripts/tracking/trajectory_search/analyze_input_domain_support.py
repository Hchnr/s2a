"""Saved terminal support model on the full input box; no response or dispatch."""

import os
import traceback

import numpy as np

from s2a.infrastructure.publication import publish_result
from s2a.research.trajectory_search.restored_anchor_sqp_evidence import REPO, read, reference, require
from scripts.maintenance.run_cpu_tests import CPU_ENV, snapshot
from scripts.tracking.trajectory_search.analyze_local_tracking_support import box_corrected_bound, solve_support

OUTPUT = REPO / "outputs/input_domain_support_20261102"
CONFIG = REPO / "configs/tracking/trajectory_search/input_domain_support_20261102.json"
PLAN = REPO / "plans/v0.0.1/tracking/trajectory_search/input_domain_support_20261102.md"
QA = REPO / "outputs/qa/input_domain_support_20261102/result.json"


def input_domain_rows(A, b, entry, old_lower, old_upper, input_min, input_max, config, fraction):
    """Change only safety offsets and command offsets in the saved relaxation."""
    A, b = np.asarray(A).copy(), np.asarray(b).copy()
    x = np.asarray(entry["commands"]).ravel()
    lo, hi = np.asarray(input_min), np.asarray(input_max)
    require(A.shape == (4657, 72) and b.shape == (4657,) and x.shape == (72,), "Complete original relaxation")
    require(
        lo.shape == hi.shape == (6,) and np.isfinite(lo).all() and np.isfinite(hi).all() and np.all(lo < hi),
        "Strict finite controller input domain",
    )
    lower, upper = np.tile(lo, 12), np.tile(hi, 12)
    require(
        np.all(lower <= np.asarray(old_lower).ravel()) and np.all(upper >= np.asarray(old_upper).ravel()),
        "Full input box contains original algorithm domain",
    )
    require(fraction in (0.0, 0.9), "Only original buffer and registered reserve diagnostic")
    np.testing.assert_array_equal(A[4513:4585], np.eye(72))
    np.testing.assert_array_equal(A[4585:4657], -np.eye(72))
    weights = np.r_[np.ones(4368), config["terminal_time_scale"]]
    b[:4369] = (
        np.asarray(entry["safety_slack"]) - config["safety_buffer"] - fraction * config["reserve_target"] / weights
    )
    b[4513:4585], b[4585:4657] = x - lower, upper - x
    require(np.isfinite(A).all() and np.isfinite(b).all(), "Finite full-input relaxation")
    return A, b, lower - x, upper - x


def main():
    require(all(os.environ.get(k) == v for k, v in CPU_ENV.items()), "Explicit CPU environment")
    spec = read(reference(CONFIG))
    require(
        spec
        == dict(
            seed=42,
            anchor_index=23,
            group=23,
            reserve_fractions=[0.0, 0.9],
            lp_budget=2,
            lp_time_limit_seconds=10,
            lp_max_iterations=10000,
            certificate_tolerance=1e-8,
            timeout_seconds=300,
            automatic_retry=False,
        ),
        "Exact diagnostic budget",
    )
    OUTPUT.mkdir(exist_ok=False)
    before = snapshot(REPO)
    qa = read(reference(QA))
    require(
        qa["passed"] and qa["all_tests_passed"] and qa["reaped"] and qa["source_unchanged"], "Closed corresponding QA"
    )
    qs = read(qa["source_after_ref"])
    require(all(qs[k] == before[k] for k in ("sources", "current_map_ref", "driver_ref")), "QA source binding")
    refs = []

    def load(path, sha):
        ref = reference(path)
        require(ref["sha256"] == sha, "Fixed source SHA")
        refs.append(ref)
        return read(ref)

    support = load(
        REPO / "outputs/local_tracking_support_20261102/result.json",
        "f029085ade6a4c846fffb50ce0eba294b0a87cbbf2aedc5cf694192555d90203",
    )
    require(
        support["passed"] and support["sources_unchanged"] and support["inputs_unchanged"], "Qualified saved support"
    )
    lp = load(
        REPO / "outputs/local_tracking_support_20261102/lp_48.json",
        "6ff37e4cf90c1763cdffe0c961c6078c8838371351909f9b38b1470a2630b09b",
    )
    old_rows = load(lp["constraints_ref"]["path"], lp["constraints_ref"]["sha256"])
    failed = load(
        REPO / "outputs/small_progress_search_20261102/small_progress/result.json",
        "844c4e05a24b41dda33e36aac180bb421afa73e63f88f8eb70b213615f6e4c5a",
    )
    answer = load(failed["producer_final_ref"]["path"], failed["producer_final_ref"]["sha256"])["answer"]
    bundle = load(
        REPO / "outputs/constraint_residual_repair_inputs_20261029/bundle.json",
        "eadf18a3ed90c29ee70d773741cc310304f4385189baea360e534ca6ef5bd023",
    )
    entry = answer["evaluated_candidates"][23]
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
    try:
        for fraction in spec["reserve_fractions"]:
            A, b, lower, upper = input_domain_rows(
                old_rows["A"],
                old_rows["b"],
                entry,
                answer["global_lower"],
                answer["global_upper"],
                bundle["control"]["input_min"],
                bundle["control"]["input_max"],
                answer["config"],
                fraction,
            )
            result["lp_attempts"] += 1
            record = solve_support(A, b, np.asarray(lp["c"]), spec)
            record.update(
                reserve_fraction=fraction,
                constraints_ref=publish_result(OUTPUT / f"constraints_{fraction}.json", dict(A=A, b=b)),
            )
            if record["success"]:
                corrected = box_corrected_bound(A, b, np.asarray(lp["c"]), record["dual"], lower, upper)
                record["corrected_dual"] = corrected
                record["support_lower_bound"] = lp["error_norm"] + corrected["bound"]
                record["linear_model_excluded"] = record["support_lower_bound"] > 1 + spec["certificate_tolerance"]
            receipt = publish_result(OUTPUT / f"lp_{fraction}.json", record)
            result["records"].append(
                dict(
                    ref=receipt,
                    success=record["success"],
                    reserve_fraction=fraction,
                    support_lower_bound=record.get("support_lower_bound"),
                )
            )
        result.update(complete=True, passed=result["lp_attempts"] == 2 and all(r["success"] for r in result["records"]))
    except BaseException as error:
        result.update(error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        after = snapshot(REPO)
        result["source_after_ref"] = publish_result(OUTPUT / "source_after.json", after)
        result["sources_unchanged"] = all(before[k] == after[k] for k in ("sources", "current_map_ref", "driver_ref"))
        result["inputs_unchanged"] = all(reference(r["path"]) == r for r in refs)
        result["passed"] &= result["sources_unchanged"] and result["inputs_unchanged"]
        print(publish_result(OUTPUT / "result.json", result), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
