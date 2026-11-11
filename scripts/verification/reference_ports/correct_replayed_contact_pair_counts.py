"""Correct contact-pair counts when an inner step contains multiple contact points."""

import argparse
from pathlib import Path

import numpy as np

from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.verification.reference_ports import audit_replayed_contact_torque as previous

ROOT, RUN = prefix.REPO, previous.RUN
LEGACY = ROOT / "outputs/replayed_contact_torque_audit_20261111/result.json"
LEGACY_SHA = "66cdf351c32b3182e5d57339fa945b45bfa9f45d77e19df7b1d7fa3246436013"
OUTPUT = ROOT / "outputs/replayed_contact_pair_counts_20261111/result.json"
QA = ROOT / "outputs/qa/replayed_contact_pair_counts_v2_20261111/result.json"


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            Path(__file__).resolve(),
            Path(previous.__file__).resolve(),
            ROOT / "tests/research/response_models/test_replayed_contact_pair_counts.py",
        )
    ]


def summarize_pairs(samples):
    """Count each geometry pair once per inner step and count contact points separately."""
    periods = []
    prefix.require(len(samples) % 25 == 0, "Complete 25-inner-step periods required")
    for start in range(0, len(samples), 25):
        chunk = samples[start : start + 25]
        prefix.require(
            len({item["step"] for item in chunk}) == 1 and [item["inner_index"] for item in chunk] == list(range(25)),
            "Each period must contain all ordered inner steps",
        )
        pairs = {}
        for sample in chunk:
            seen = set()
            for contact in sample["contacts"]:
                if contact["kind"] != "robot_environment" or not contact["active_constraint"]:
                    continue
                pair = tuple(sorted((contact["geom1"], contact["geom2"])))
                item = pairs.setdefault(pair, {"contact_samples": 0, "contact_points": 0, "maximum_force_n": 0.0})
                item["contact_points"] += 1
                item["maximum_force_n"] = max(
                    item["maximum_force_n"], float(np.linalg.norm(contact["wrench_contact_frame"][:3]))
                )
                seen.add(pair)
            for pair in seen:
                pairs[pair]["contact_samples"] += 1
        periods.append(
            {
                "step": chunk[0]["step"],
                "contact_pairs": [{"geometries": pair, **item} for pair, item in sorted(pairs.items())],
            }
        )
    return periods


def audit():
    prefix.environment()
    consumed = {}
    old = prefix.read({"path": str(LEGACY), "sha256": LEGACY_SHA}, consumed)
    prefix.require(old["complete"] is old["passed"] is True, "The original audit must be retained")
    for ref in old["source_refs"]:
        prefix.checked(ref, consumed)
    owner = prefix.read(old["owner_ref"], consumed)
    reports = {}
    for name in ("historical", "innovation"):
        branch = prefix.read(owner["summaries"][name]["branch_ref"], consumed)
        periods = summarize_pairs(branch["runtime"]["replay_dynamics"]["samples"])
        legacy = old["branches"][name]["periods"]
        prefix.require(len(periods) == len(legacy), "The original period roster must be unchanged")
        affected = 0
        for corrected, original in zip(periods, legacy, strict=True):
            prefix.require(corrected["step"] == original["step"], "The original period clock must match")
            before = {tuple(item["geometries"]): item for item in original["contact_pairs"]}
            after = {tuple(item["geometries"]): item for item in corrected["contact_pairs"]}
            prefix.require(before.keys() == after.keys(), "The original contact-pair roster must match")
            for pair, item in after.items():
                old_item = before[pair]
                prefix.require(
                    old_item["samples"] == item["contact_points"]
                    and old_item["maximum_force_n"] == item["maximum_force_n"]
                    and 0 < item["contact_samples"] <= 25
                    and item["contact_samples"] <= item["contact_points"],
                    "The old field must count points; corrected samples must count distinct inner steps",
                )
                affected += item["contact_samples"] != item["contact_points"]
        reports[name] = {"periods": periods, "pair_periods_with_multiple_points": affected}
    return {
        "complete": True,
        "passed": True,
        "source_refs": source_refs(),
        "legacy_ref": {"path": str(LEGACY), "sha256": LEGACY_SHA},
        "source_sha256": consumed,
        "legacy_pair_samples_actually_counted_contact_points": True,
        "branch_periods": reports,
        "world_model_calls": 0,
        "native_steps": 0,
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
        and qa["skipped_count"] == 0
        and not qa["collection"]["collection_errors"]
        and snapshot(ROOT)["sources"] == prefix.read(qa["source_before_ref"])["sources"],
        "Completed source-pinned 32-worker correction QA required",
    )
    print(publish(OUTPUT, audit()), flush=True)


if __name__ == "__main__":
    main()
