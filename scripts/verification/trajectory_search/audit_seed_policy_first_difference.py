"""Compare two replayed controller cohorts at their first action divergence."""

import argparse
import sys
from pathlib import Path

import numpy as np

from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import prepare_reset_correct_reference_cohort as preparation
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_correct_reference as original

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/joint_control/seed_policy_first_difference_20261116.json"
PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/seed_policy_first_difference_20261116.md"
OUTPUT = ROOT / "outputs/seed_policy_first_difference_20261116"
DRY = ROOT / "outputs/seed_policy_first_difference_dry_20261116.json"
QA = ROOT / "outputs/qa/seed_policy_first_difference_final_20261116/result.json"
MODULE = "scripts.verification.trajectory_search.audit_seed_policy_first_difference"
FAST = dict(
    path="outputs/fast_seed_cohort_20261116/audit.json",
    sha256="bc0cc238fc48707c7629bc5248b40f1c5f2083a57c3e09be2a83e838f9a7f103",
)
BOUNDED = dict(
    path="outputs/bounded_seed_preference_20261116/audit.json",
    sha256="efebb7ae26d844e3caaadaf33f5038ac6ff96b819fe6f018d5cf42b40954c42f",
)


def validate_config(config):
    expected = dict(
        schema="s2a.seed-policy-first-difference.v1",
        output="outputs/seed_policy_first_difference_20261116",
        fast_cohort_audit=FAST,
        bounded_cohort_audit=BOUNDED,
        cases=18,
        max_workers=18,
        tolerance=1e-10,
        worker_timeout_seconds=1200,
        physics_steps=0,
        model_responses=0,
        world_model_calls=0,
        gpu_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    if config != expected or any(type(config.get(key)) is not type(value) for key, value in expected.items()):
        raise ValueError("The recorded cohort identities, tolerance and zero-execution budget are fixed")


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__).resolve(),
            ROOT / "tests/research/trajectory_search/test_seed_policy_first_difference.py",
            ROOT / "scripts/simulation/runtime/audit_reset_multitask_live_recovery.py",
            ROOT / "scripts/simulation/runtime/run_reset_correct_reference.py",
            ROOT / "src/s2a/infrastructure/owned_processes.py",
        )
    ]


def inputs():
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    validate_config(config)
    refs = [preparation.configured_ref(config[key]) for key in ("fast_cohort_audit", "bounded_cohort_audit")]
    audits = [prefix.read(ref) for ref in refs]
    registrations = []
    for audit in audits:
        producer = prefix.read(audit["producer_ref"])
        prefix.require(
            producer["passed"] is producer["owner_closed"] is producer["sources_unchanged"] is True,
            "The original producer owner must have closed with unchanged sources",
        )
        registrations.append(prefix.read(producer["registration_ref"])["inputs"])
    prefix.require(
        all(
            audit["complete"] is audit["passed"] is True
            and audit["registered_cases"] == len(audit["cases"]) == 18
            and all(
                row["replayed"] is True and row["producer_status"] == row["checker_status"] == "closed"
                for row in audit["cases"]
            )
            for audit in audits
        )
        and audits[0]["qualified_cases"] == 16
        and audits[1]["qualified_cases"] == 15
        and registrations[0]["cohort"] == registrations[1]["cohort"]
        and registrations[0]["baseline_ref"] == registrations[1]["baseline_ref"]
        and all(a["case"] == b["case"] for a, b in zip(audits[0]["cases"], audits[1]["cases"], strict=True)),
        "Both independently replayed 18-case source rosters are required",
    )
    return config, dict(config_ref=config_ref, fast_ref=refs[0], bounded_ref=refs[1]), audits


def first_difference(left, right, *, key, tolerance=1e-10):
    """Find the first finite command or selected-plan difference in the common prefix."""
    if not isinstance(left, list) or not isinstance(right, list) or tolerance <= 0:
        raise ValueError("Two recorded period lists and a positive tolerance are required")
    shape = (7,) if key == "command" else (8, 6) if key == "refined_plan" else None
    if shape is None:
        raise ValueError("Only recorded commands and full refined plans are comparable")
    for index, (a, b) in enumerate(zip(left, right)):
        av = a[key] if key == "command" else a["decision"][key]
        bv = b[key] if key == "command" else b["decision"][key]
        x, y = np.asarray(av, dtype=float), np.asarray(bv, dtype=float)
        if x.shape != shape or y.shape != shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Recorded values must have the fixed finite controller shape")
        if np.max(np.abs(x - y)) > tolerance:
            return index
    return None


def analyze_case(fast, bounded, fast_qualified, bounded_qualified, *, tolerance=1e-10):
    """Use only the common measured past to characterize a differing decision."""
    left, right = original.periods(fast), original.periods(bounded)
    step = first_difference(left, right, key="command", tolerance=tolerance)
    plan_step = first_difference(left, right, key="refined_plan", tolerance=tolerance)
    common = min(len(left), len(right)) if step is None else step
    parity = comparison.compare_periods(left[:common], right[:common])
    outcome = (
        "bounded_only"
        if bounded_qualified and not fast_qualified
        else "fast_only"
        if fast_qualified and not bounded_qualified
        else "both_pass"
        if fast_qualified
        else "both_fail"
    )
    result = dict(
        fast_qualified=fast_qualified,
        bounded_qualified=bounded_qualified,
        outcome=outcome,
        fast_periods=len(left),
        bounded_periods=len(right),
        common_executed_before_action_difference=common,
        first_action_difference=step,
        first_plan_difference=plan_step,
        common_physical_prefix_max_difference=parity,
        diagnostic=None,
    )
    if step is None:
        return result
    a, b = left[step], right[step]
    fast_traces = fast["runtime"]["segment_receipts"][0]["fast_seed_traces"]
    bounded_traces = bounded["runtime"]["segment_receipts"][0]["seed_preference_traces"]
    prefix.require(step < len(fast_traces) and step < len(bounded_traces), "Both actual decision traces must exist")
    ft, bt = fast_traces[step], bounded_traces[step]
    previous = left[step - 1] if step else None
    delta = np.asarray(a["command"], dtype=float) - np.asarray(b["command"], dtype=float)
    seed_slack, increment, default = (
        ft.get("selected_future_slack"),
        bt.get("unbounded_seed_future_slack_increase"),
        bt.get("default_future_slack"),
    )
    comparable = all(type(value) in (int, float) and np.isfinite(value) for value in (seed_slack, increment, default))
    result["diagnostic"] = dict(
        fast_early_selected=ft.get("early_selected"),
        bounded_seed_blocked=bt.get("seed_blocked_by_future_slack"),
        bounded_seed_in_eligible_cache=bt.get("seed_in_eligible_cache"),
        seed_future_slack=seed_slack,
        bounded_seed_slack_increase=increment,
        bounded_default_future_slack=default,
        seed_slack_agrees=comparable and abs(seed_slack - increment - default) <= tolerance,
        action_max_abs_difference=float(np.max(np.abs(delta))),
        arm_difference_l2=float(np.linalg.norm(delta[:6])),
        gripper_difference=float(delta[6]),
        prior_tcp_error=None if previous is None else previous["tcp_error"],
        prior_gripper_opening=None if previous is None else previous["completed_feedback"]["score"]["opening"],
        fast_predicted_first_tcp_error=a["decision"].get("quality", {}).get("first_tcp_error"),
        bounded_predicted_first_tcp_error=b["decision"].get("quality", {}).get("first_tcp_error"),
        physical_history_alone_proves_internal_history_equal=False,
    )
    return result


def threshold_observation(cases, field):
    """Describe first-divergence label ordering, without estimating policy outcomes."""
    groups = {name: [] for name in ("fast_only", "bounded_only")}
    for row in cases:
        if row["outcome"] in groups and row["diagnostic"] is not None:
            diagnostic = row["diagnostic"]
            value = diagnostic.get(field)
            if (
                diagnostic.get("fast_early_selected") is True
                and diagnostic.get("bounded_seed_blocked") is True
                and diagnostic.get("bounded_seed_in_eligible_cache") is True
                and diagnostic.get("seed_slack_agrees") is True
                and type(value) in (int, float)
                and np.isfinite(value)
            ):
                groups[row["outcome"]].append(float(value))
    if any(len(groups[key]) != sum(row["outcome"] == key for row in cases) for key in groups):
        return dict(available=False, values=groups)
    fast, bounded = groups["fast_only"], groups["bounded_only"]
    if not fast or not bounded:
        return dict(available=False, values=groups)
    return dict(
        available=True,
        values=groups,
        larger_values_select_bounded_separable=max(fast) < min(bounded),
        smaller_values_select_bounded_separable=max(bounded) < min(fast),
    )


def worker(index):
    prefix.environment()
    registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
    config, identity, audits = inputs()
    prefix.require(
        0 <= index < config["cases"]
        and identity == registration["inputs"]
        and source_refs() == registration["source_refs"],
        "The worker must use the registered case and sources",
    )
    a, b = (audit["cases"][index] for audit in audits)
    fast, bounded = prefix.read(a["producer_ref"]), prefix.read(b["producer_ref"])
    prefix.require(
        fast["case"] == bounded["case"] == a["case"] == b["case"]
        and fast["runtime"]["reset_endpoint_comparison"] == bounded["runtime"]["reset_endpoint_comparison"],
        "Case identity and reset state must match",
    )
    value = analyze_case(fast, bounded, a["qualified"], b["qualified"], tolerance=config["tolerance"])
    value.update(case=a["case"], fast_ref=a["producer_ref"], bounded_ref=b["producer_ref"], complete=True, passed=True)
    print(publish_result(OUTPUT / f"case{index:02d}/result.json", value), flush=True)


def execute():
    prefix.environment()
    qa_ref, dry_ref = prefix.reference(QA), prefix.reference(DRY)
    qa, dry = prefix.read(qa_ref), prefix.read(dry_ref)
    before, sources = snapshot(ROOT)["sources"], source_refs()
    config, identity, _audits = inputs()
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["configured_parallelism"] == 64
        and qa["skipped_count"] == 0
        and before == prefix.read(qa["source_before_ref"])["sources"]
        and dry["complete"] is dry["passed"] is True
        and dry["inputs"] == identity
        and dry["source_refs"] == sources,
        "Source-bound 64-worker QA and a matching zero-execution preflight are required",
    )
    OUTPUT.mkdir(parents=True, exist_ok=False)
    registration_ref = publish_result(
        OUTPUT / "registration.json",
        dict(inputs=identity, source_refs=sources, qa_ref=qa_ref, dry_ref=dry_ref, automatic_retry=False),
    )
    specs = [
        dict(
            key=str(index),
            argv=[sys.executable, "-m", MODULE, "--worker", str(index)],
            log=OUTPUT / f"case{index:02d}/worker.log",
        )
        for index in range(config["cases"])
    ]
    lifecycle = run_owned_workers(specs, cwd=ROOT, timeout_seconds=config["worker_timeout_seconds"])
    for child in lifecycle["workers"]:
        child["log_ref"] = prefix.reference(child.pop("log"))
    lifecycle_ref = publish_result(OUTPUT / "workers.json", lifecycle)
    result = dict(
        complete=False,
        passed=False,
        cases={},
        registration_ref=registration_ref,
        lifecycle_ref=lifecycle_ref,
        physics_steps=0,
        model_responses=0,
        world_model_calls=0,
        gpu_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        for index in range(config["cases"]):
            path = OUTPUT / f"case{index:02d}/result.json"
            row = result["cases"][str(index)] = dict(status="failed")
            if path.is_file():
                ref = prefix.reference(path)
                data = prefix.read(ref)
                row["available_ref"] = ref
                if data.get("complete") is data.get("passed") is True:
                    row.update(status="closed", result_ref=ref)
        prefix.require(
            lifecycle["launch_error"] is None
            and len(lifecycle["workers"]) == 18
            and all(
                w["returncode"] == 0 and w["actual_wait"] is w["reaped"] is True and not w["signal_sent"]
                for w in lifecycle["workers"]
            )
            and all(row["status"] == "closed" for row in result["cases"].values()),
            "All registered workers must complete and be reaped",
        )
        rows = [prefix.read(result["cases"][str(i)]["result_ref"]) for i in range(config["cases"])]
        _config, after, _audits = inputs()
        prefix.require(
            after == identity and source_refs() == sources and snapshot(ROOT)["sources"] == before,
            "Sources and inputs must remain unchanged",
        )
        result.update(
            complete=True,
            passed=True,
            sources_unchanged=True,
            first_action_differences=sum(row["first_action_difference"] is not None for row in rows),
            first_plan_before_action=sum(
                row["first_plan_difference"] is not None
                and row["first_action_difference"] is not None
                and row["first_plan_difference"] < row["first_action_difference"]
                for row in rows
            ),
            seed_absolute_slack_ordering=threshold_observation(rows, "seed_future_slack"),
            seed_relative_slack_ordering=threshold_observation(rows, "bounded_seed_slack_increase"),
        )
    except Exception as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
    print(publish_result(OUTPUT / "result.json", result), flush=True)
    if not result["passed"]:
        raise SystemExit("The registered comparison did not close; no automatic retry")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--worker", type=int)
    args = parser.parse_args()
    prefix.environment()
    if args.dry:
        _config, identity, _audits = inputs()
        print(
            publish_result(
                DRY,
                dict(
                    complete=True,
                    passed=True,
                    inputs=identity,
                    source_refs=source_refs(),
                    physics_steps=0,
                    model_responses=0,
                    world_model_calls=0,
                    gpu_calls=0,
                    fresh_cases=0,
                ),
            ),
            flush=True,
        )
    elif args.execute:
        execute()
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
