"""Shadow a bounded release-revision selector on fixed, saved trajectories."""

import argparse
import json
from collections import Counter

import numpy as np

from s2a.geometry import pose_error
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.wam.tracking.preview.bounded_release_revision import (
    BoundedReleaseRevision,
    ReferenceChunk,
    ReleaseRevisionConfig,
)
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.wam.tracking.preview import bind_bounded_release_shadow as binding


def selector_config(config):
    return ReleaseRevisionConfig(
        **{
            key: config[key]
            for key in (
                "stride",
                "bridge_steps",
                "release_clearance",
                "opening_revision_gap",
                "current_position_gate_m",
                "current_rotation_gate_rad",
                "joint_buffer_rad",
            )
        }
    )


def reference(chunk):
    return ReferenceChunk(chunk["step"], chunk["effective_targets_tcp"], chunk["openings"])


def recorded_decision(decision):
    """Store the executable and one bridge window without adding future observations."""
    return {
        **{key: value for key, value in decision.items() if key not in ("poses", "openings")},
        "available_endpoints": len(decision["openings"]),
        "poses": decision["poses"][:16].copy(),
        "openings": decision["openings"][:16].copy(),
    }


def shadow_case(item, chunks, rows, tool, config):
    selector = BoundedReleaseRevision(selector_config(config))
    stride, records = config["stride"], []
    for query, chunk in enumerate(chunks):
        step = query * stride
        latest = reference(chunk)
        committed = None if query == 0 else reference(chunks[query - 1])
        closing = False if step == 0 else rows[step - 1]["gripper"]["latch"]["closing_episode"]
        prefix.require(type(closing) is bool, "Only completed gripper history enters the selector")
        before = selector.state()
        decision = selector.prepare(
            latest,
            committed=committed,
            step=step,
            measured_tcp=np.asarray(rows[step]["measured_source"], dtype=float) @ tool,
            measured_opening=rows[step]["opening_before"],
            closing_episode=closing,
            previous_inner_margin_rad=None,
        )
        # No original baseline period contains all 25 actual inner samples.
        # Never fabricate a passing safety margin to activate this candidate.
        prefix.require(decision["source"] == "latest", "Unknown actual inner history cannot authorize a hold")
        available = min(stride, len(rows) - step)
        np.testing.assert_array_equal(decision["poses"][:available], latest.poses[:available])
        np.testing.assert_array_equal(decision["openings"][:available], latest.openings[:available])
        stored = recorded_decision(decision)
        if available == stride:
            selector.commit(decision, completed_steps=stride)
        else:
            selector.abort(decision)
        records.append(
            {
                **stored,
                "query": query,
                "closing_episode": closing,
                "baseline_periods": available,
                "terminal_partial_not_committed": available < stride,
                "state_before": before,
                "state_after": selector.state(),
                "baseline_observation_label": "unchanged_baseline",
            }
        )
    reasons = Counter(record["reason"] for record in records)
    summary = {
        "baseline_periods": len(rows),
        "queries": len(records),
        "conflicts_without_safety": sum(record["conflict_without_safety"] for record in records),
        "unknown_inner_conflicts": reasons["unknown_inner_margin"],
        "holds": sum(record["source"] == "committed" for record in records),
        "bridges": sum(record["source"] == "bridge" for record in records),
        "reasons": dict(sorted(reasons.items())),
        "first_intervention_step": None,
    }
    return {"records": records, "summary": summary}


def studied_case(inputs, config):
    data = binding.read_studied(inputs)
    selector = BoundedReleaseRevision(selector_config(config))
    decision = selector.prepare(
        reference(data["latest"]),
        committed=reference(data["old"]),
        step=data["step"],
        measured_tcp=data["measured_tcp"],
        measured_opening=data["measured_opening"],
        closing_episode=data["closing_episode"],
        previous_inner_margin_rad=data["previous_inner_margin_rad"],
    )
    stored = recorded_decision(decision)
    latest = np.asarray(data["latest"]["effective_targets_tcp"])[8:16]
    delta = pose_error(decision["poses"][:8], latest)
    summary = {
        "step": data["step"],
        "source": decision["source"],
        "reason": decision["reason"],
        "closing_episode": data["closing_episode"],
        "measured_opening": data["measured_opening"],
        "previous_inner_margin_rad": data["previous_inner_margin_rad"],
        "max_selected_vs_latest_position_m": float(np.max(np.linalg.norm(delta[:, :3], axis=1))),
        "max_selected_vs_latest_rotation_rad": float(np.max(np.linalg.norm(delta[:, 3:], axis=1))),
    }
    selector.abort(decision)
    return {"decision": stored, "summary": summary, "selection_executed": False, "state_after_abort": selector.state()}


def worker(index):
    prefix.environment()
    directory = binding.ROOT / "producer"
    registration = prefix.read(prefix.reference(directory / "registration.json"))
    prefix.require(
        registration["phase"] == "producer" and registration["source_refs"] == binding.source_refs(),
        "Frozen producer sources",
    )
    inputs = registration["inputs"]
    config = prefix.read(inputs["config_ref"])
    if index == 30:
        data = studied_case(inputs, config)
        output = {"complete": True, "passed": True, **data}
        path = directory / "studied/result.json"
    else:
        prefix.require(type(index) is int and 0 <= index < 30, "One registered baseline case")
        item = inputs["cases"][index]
        chunks, rows, tool = binding.read_case(item)
        data = shadow_case(item, chunks, rows, tool, config)
        for ref in item["refs"].values():
            prefix.checked(ref)
        output = {
            "complete": True,
            "passed": True,
            "case_id": item["case_id"],
            "family": item["family"],
            "input_refs": item["refs"],
            **data,
        }
        path = directory / f"case_{index:02d}/result.json"
    print(publish(path, output), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--producer", action="store_true")
    group.add_argument("--worker", type=int)
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(binding.dry_run()))
    elif args.producer:
        binding.execute_phase("producer", "scripts.wam.tracking.preview.replay_bounded_release_revision")
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
