"""Score past-only TCP prediction corrections on independently replayed runs."""

import argparse
from pathlib import Path

from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.response_models import audit_causal_tcp_prediction as diagnostic

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/response_models/multitask_causal_tcp_innovation_20261113.json"
PLAN = ROOT / "plans/v0.0.1/tracking/response_models/multitask_causal_tcp_innovation_20261113.md"
OUTPUT = ROOT / "outputs/multitask_causal_tcp_innovation_20261113/result.json"
QA = ROOT / "outputs/qa/multitask_causal_tcp_innovation_20261113/result.json"
CONFIG_SHA = "d44e9eee11508b38fdc9999bd39f0c4a9e852b9ed64732b89b888e1577bdd07b"
PLAN_SHA = "91038e56a7174cf73ef3a2617f727ea4bcb64c149ee494b0a3e9dd596c25e746"
INDICES = (0, 2, 4, 6, 8, 10)


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__).resolve(),
            ROOT / "scripts/tracking/response_models/audit_causal_tcp_prediction.py",
            ROOT / "src/s2a/research/response_models/causal/tcp_prediction_innovation.py",
            ROOT / "tests/research/response_models/test_multitask_causal_tcp_innovation.py",
            ROOT / "tests/research/response_models/test_tcp_prediction_innovation.py",
            ROOT / "tests/research/native_runtime/test_causal_tcp_prediction_audit.py",
        )
    ]


def load_trajectories():
    """Bind each complete recorded branch to its independent replay audit."""
    prefix.environment()
    consumed = {}
    config = prefix.read({"path": str(CONFIG), "sha256": CONFIG_SHA}, consumed)
    prefix.checked({"path": str(PLAN), "sha256": PLAN_SHA}, consumed)
    prefix.require(
        config["schema"] == "s2a.multitask-causal-tcp-innovation.v1"
        and config["output"] == str(OUTPUT.relative_to(ROOT))
        and config["case_indices"] == list(INDICES)
        and config["sampling_seeds"] == [42, 43, 42, 43, 42, 43]
        and config["expected_adaptive_periods"] == 1025
        and config["expected_fixed_periods"] == 270
        and config["position_cap_m"] == 0.01
        and config["rotation_cap_rad"] == 0.05
        and config["candidates"] == ["raw", "previous_innovation", "past_maximum_bound"]
        and config["new_model_requests"] == config["new_physics_steps"] == config["fresh_cases"] == config["fits"] == 0
        and config["automatic_retry"] is False,
        "The registered retrospective roster and zero-work budget are required",
    )
    audits = {}
    for protocol in ("adaptive", "fixed"):
        ref = {"path": str(ROOT / config[f"{protocol}_audit"]), "sha256": config[f"{protocol}_audit_sha256"]}
        audits[protocol] = prefix.read(ref, consumed)
        prefix.require(
            audits[protocol]["complete"] is True
            and [row["case_index"] for row in audits[protocol]["cases"]] == list(INDICES),
            "Both independent audit rosters must be complete and ordered",
        )
    groups = []
    for protocol in ("adaptive", "fixed"):
        for index, recorded in zip(INDICES, audits[protocol]["cases"], strict=True):
            if protocol == "adaptive":
                branch = prefix.read(recorded["producer_ref"], consumed)
                replay = prefix.read(recorded["checker_ref"], consumed)
                differences = recorded["independent_max_absolute_difference"]
            else:
                owner = prefix.read(recorded["producer_ref"], consumed)
                replay = prefix.read(recorded["checker_ref"], consumed)
                branch = prefix.read(owner["branch_ref"], consumed)
                differences = recorded["replay_max_absolute_difference"]
                prefix.require(owner["complete"] is owner["passed"] is True, "Fixed reference owner must close")
            prefix.require(
                branch["complete"] is branch["passed"] is replay["complete"] is replay["passed"] is True
                and branch["case"]["case_index"] == index
                and branch["case"]["sampling_seed"] == config["sampling_seeds"][INDICES.index(index)]
                and all(value == 0 for value in differences.values()),
                "Each same-seed branch and independent replay must be complete and agree",
            )
            continuation = branch["runtime"]["continuation"]
            later = (
                continuation["adaptive"]["segments"] if protocol == "adaptive" else continuation["fixed"]["segments"]
            )
            rows = continuation["first_window"]["periods"] + [
                period for segment in later for period in segment["continuation"]["periods"]
            ]
            prefix.require(
                len(rows) == recorded["counts"]["control_steps"]
                and [row["step"] for row in rows] == list(range(len(rows))),
                "Every recorded physical period must remain consecutive and in the denominator",
            )
            groups.append(dict(protocol=protocol, case_index=index, case_id=branch["case"]["case_id"], periods=rows))
    for protocol in ("adaptive", "fixed"):
        total = sum(len(group["periods"]) for group in groups if group["protocol"] == protocol)
        prefix.require(total == config[f"expected_{protocol}_periods"], "The full physical denominator must match")
    return config, groups, consumed


def evaluate_groups(groups):
    """Reset the causal innovation history independently for every trajectory."""
    cases, pooled = [], {"adaptive": [], "fixed": []}
    for group in groups:
        protocol, periods = group["protocol"], group["periods"]
        prefix.require(protocol in pooled and len(periods) > 1, "A known protocol with completed history is required")
        rows = diagnostic.evaluate(periods)
        pooled[protocol].extend(rows)
        cases.append(
            dict(
                protocol=protocol,
                case_index=group["case_index"],
                case_id=group["case_id"],
                periods=len(rows),
                periods_with_history=sum(row["history_available"] for row in rows),
                candidates=diagnostic.summarize(rows)["candidates"],
                rows=rows,
            )
        )
    summaries = {name: diagnostic.summarize(rows)["candidates"] for name, rows in pooled.items() if rows}
    return cases, summaries


def run():
    config, groups, consumed = load_trajectories()
    before = source_refs()
    cases, summaries = evaluate_groups(groups)
    for path, digest in consumed.items():
        prefix.checked({"path": path, "sha256": digest})
    prefix.require(source_refs() == before, "Diagnostic source identities must remain unchanged")
    return dict(
        complete=True,
        passed=True,
        control_qualified=False,
        source_refs=before,
        input_sha256=consumed,
        cases=cases,
        protocol_summaries=summaries,
        shared_initial_states_and_prefixes=True,
        independent_samples_claimed=False,
        retrospective_selection=True,
        task_labels_used_for_estimation=False,
        new_model_requests=config["new_model_requests"],
        new_physics_steps=config["new_physics_steps"],
        fresh_cases=config["fresh_cases"],
        automatic_retry=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        _config, groups, _consumed = load_trajectories()
        print({"trajectories": len(groups), "periods": sum(len(group["periods"]) for group in groups)})
        return
    qa = prefix.read(prefix.reference(QA))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["reaped"] is True
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == 0
        and not qa["collection"]["collection_errors"]
        and snapshot(ROOT)["sources"] == prefix.read(qa["source_before_ref"])["sources"],
        "A source-matched 32-worker CPU regression is required",
    )
    print(publish(OUTPUT, run()), flush=True)


if __name__ == "__main__":
    main()
