"""Compare causal TCP prediction diagnostics on a fixed set of replayed trajectories."""

import argparse
from pathlib import Path

from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import prepare_quality_rate_repair_recorded_feedback as streaming
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_wam_two_window_feedback as historical
from scripts.tracking.response_models import audit_causal_tcp_prediction as diagnostic

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/response_models/cross_trajectory_tcp_innovation_20261111.json"
CONFIG_SHA = "4115ac617469a1a4c54d3d101501289a4081506c3dc9655f01702554c79d6ca4"
PLAN = ROOT / "plans/v0.0.1/tracking/response_models/cross_trajectory_tcp_innovation_20261111.md"
OUTPUT = ROOT / "outputs/cross_trajectory_tcp_innovation_20261111/result.json"
QA = ROOT / "outputs/qa/cross_trajectory_tcp_innovation_20261111/result.json"


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__).resolve(),
            ROOT / "scripts/tracking/response_models/audit_causal_tcp_prediction.py",
            ROOT / "src/s2a/research/response_models/causal/tcp_prediction_innovation.py",
            ROOT / "tests/research/response_models/test_cross_trajectory_tcp_innovation.py",
        )
    ]


def read_config(consumed):
    value = prefix.read({"path": str(CONFIG), "sha256": CONFIG_SHA}, consumed)
    prefix.require(
        value["schema"] == "s2a.cross-trajectory-tcp-innovation.v1"
        and value["output"] == str(OUTPUT.relative_to(ROOT))
        and value["position_cap_m"] == 0.01
        and value["rotation_cap_rad"] == 0.05
        and value["model_calls"] == value["physics_steps"] == value["fresh_cases"] == 0
        and value["automatic_retry"] is False,
        "The fixed retrospective scope and zero-work budget are required",
    )
    return value


def read_ref(path, digest, consumed):
    return prefix.read({"path": str(ROOT / path), "sha256": digest}, consumed)


def owner(value, phase, consumed):
    prefix.require(
        value["phase"] == phase
        and value["complete"] is value["passed"] is value["sources_unchanged"] is True
        and value["world_model_calls"] == value["fresh_cases"] == 0
        and value["automatic_retry"] is False,
        "The source owner must have closed without new model work or fresh cases",
    )
    workers = prefix.read(value["workers_ref"], consumed)
    prefix.require(
        workers["launch_error"] is None
        and bool(workers["workers"])
        and all(
            row["actual_wait"] is row["reaped"] is True and row["signal_sent"] is False and row["returncode"] == 0
            for row in workers["workers"]
        ),
        "Every historical worker must have exited and been reaped",
    )


def selected_branch(summary, key, consumed):
    refs = [ref for ref in summary["branch_refs"] if Path(ref["path"]).parent.name == key]
    prefix.require(len(refs) == 1, "Every registered branch must be present exactly once")
    return prefix.read(refs[0], consumed)


def compare_replay(produced, checked):
    prefix.require(len(produced) == len(checked) and bool(produced), "Both branches must cover every actual period")
    for index, (actual, replayed) in enumerate(zip(produced, checked, strict=True)):
        prefix.require(
            actual["step"] == replayed["step"] == produced[0]["step"] + index,
            "Actual and replayed period steps must be consecutive and identical",
        )
        historical.compare_first_period(actual, replayed)


def score(name, case, rows):
    values = diagnostic.evaluate(rows)
    scored = sum(row["history_available"] for row in values)
    return {
        "name": name,
        "case": case,
        "periods": len(rows),
        "periods_with_history": scored,
        "summary": diagnostic.summarize(values)["candidates"] if scored else None,
        "rows": values,
    }


def other_trajectories(config, consumed):
    trajectories = []
    for source in config["sources"]:
        producer = read_ref(source["producer"], source["producer_sha256"], consumed)
        checker = read_ref(source["checker"], source["checker_sha256"], consumed)
        owner(producer, "producer", consumed)
        owner(checker, "checker", consumed)
        for key in source["branches"]:
            actual, replay = (selected_branch(summary, key, consumed) for summary in (producer, checker))
            prefix.require(
                actual["complete"] is actual["passed"] is replay["complete"] is replay["passed"] is True
                and replay["runtime"]["behavioural_replay_verified"] is True
                and (
                    replay.get("candidate_modules_unloaded") is True
                    or (
                        replay.get("candidate_policy_imported") is False
                        and replay.get("quality_module_imported") is False
                    )
                ),
                "The replayed branch must be closed and isolated from candidate policies",
            )
            rows, copies = actual["runtime"]["continuation"]["periods"], replay["runtime"]["continuation"]["periods"]
            compare_replay(rows, copies)
            trajectories.append(score(source["name"] + "/" + key, key.split("_")[1], rows))
    return trajectories


def wam_trajectory(config, consumed):
    audited = read_ref(config["wam_audit"], config["wam_audit_sha256"], consumed)
    producer = read_ref(config["wam_producer"], config["wam_producer_sha256"], consumed)
    checker = read_ref(config["wam_checker"], config["wam_checker_sha256"], consumed)
    prefix.require(
        audited["complete"] is audited["passed"] is audited["checker_policy_isolated"] is True
        and audited["continuation_periods"] == 46
        and audited["control_qualified"] is False
        and producer["complete"] is producer["passed"] is producer["sources_unchanged"] is True
        and checker["complete"] is checker["passed"] is checker["sources_unchanged"] is True,
        "The independently audited WAM feedback is required",
    )
    actual = prefix.read(producer["branch_ref"], consumed)
    replay = prefix.read(checker["branch_ref"], consumed)
    isolation = prefix.read(checker["policy_isolation_ref"], consumed)
    prefix.require(isolation["before"] is isolation["after"] is True, "WAM checker policy isolation required")
    rows, copies = streaming.periods(actual), streaming.periods(replay)
    compare_replay(rows, copies)
    return score("recorded_wam/correct_08", "08", rows)


def analyze():
    prefix.environment()
    consumed = {}
    config = read_config(consumed)
    trajectories = other_trajectories(config, consumed) + [wam_trajectory(config, consumed)]
    changed = read_ref(config["changed_action_audit"], config["changed_action_audit_sha256"], consumed)
    prefix.require(
        changed["complete"] is changed["passed"] is changed["checker_policy_isolated"] is True
        and changed["prior_periods_matched"] == 42
        and changed["step122"]["step"] == 122,
        "The paired changed action must retain its independent audit",
    )
    all_rows = [row for trajectory in trajectories for row in trajectory["rows"]]
    prefix.require(len(trajectories) == 9 and len(all_rows) == 103, "The complete fixed trajectory roster is required")
    return {
        "complete": True,
        "passed": True,
        "source_refs": source_refs(),
        "source_sha256": consumed,
        "trajectories": trajectories,
        "total_periods": len(all_rows),
        "periods_with_history": sum(row["history_available"] for row in all_rows),
        "pooled_descriptive_summary": diagnostic.summarize(all_rows)["candidates"],
        "changed_step122_pair": changed["step122"],
        "shared_initial_states_and_branches": True,
        "independent_samples_claimed": False,
        "retrospective_selection": True,
        "task_labels_used_for_estimation": False,
        "model_calls": 0,
        "physics_steps": 0,
        "fresh_cases": 0,
        "control_qualified": False,
        "automatic_retry": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    qa = prefix.read(prefix.reference(QA))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and not qa["collection"]["collection_errors"]
        and snapshot(ROOT)["sources"] == prefix.read(qa["source_before_ref"])["sources"],
        "Completed source-pinned 32-worker CPU QA required",
    )
    print(publish(OUTPUT, analyze()), flush=True)


if __name__ == "__main__":
    main()
