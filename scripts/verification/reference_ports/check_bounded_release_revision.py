"""Independently verify release-revision shadow records and the studied state."""

import argparse
import math
import sys
from collections import Counter

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from s2a.infrastructure.publication import publish_result as publish
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.wam.tracking.preview import bind_bounded_release_shadow as binding

FORBIDDEN = (
    "s2a.research.wam.tracking.preview.bounded_release_revision",
    "scripts.wam.tracking.preview.replay_bounded_release_revision",
)


def independent():
    prefix.require(all(name not in sys.modules for name in FORBIDDEN), "Checker cannot import the selector or producer")


def distance(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return math.dist(a[:3, 3], b[:3, 3]), float(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude())


def close(a, b, *, atol=1e-10):
    np.testing.assert_allclose(a, b, atol=atol, rtol=0)


def expected_state(last):
    return {
        "mode": "idle",
        "last_completed_step": last,
        "latest_source_step": last,
        "held_source_step": None,
        "staged": False,
        "faulted": False,
    }


def verify_case(item, produced, chunks, rows, tool, config):
    records = produced["records"]
    prefix.require(len(records) == len(chunks) and produced["case_id"] == item["case_id"], "Complete original case")
    reasons = Counter()
    conflicts = 0
    for query, (chunk, record) in enumerate(zip(chunks, records, strict=True)):
        step = query * config["stride"]
        latest = np.asarray(chunk["openings"], dtype=float)
        closing = False if step == 0 else rows[step - 1]["gripper"]["latch"]["closing_episode"]
        prefix.require(type(closing) is bool, "Prior completed gripper state")
        old = None if query == 0 else chunks[query - 1]
        measured = np.asarray(rows[step]["measured_source"], dtype=float) @ tool
        opening = float(rows[step]["opening_before"])
        conflict, position, rotation = False, None, None
        reason = "not_closing" if not closing else "no_committed_reference" if old is None else "no_conflict"
        if closing and old is not None:
            old_index = step - old["step"]
            if old_index < len(old["openings"]):
                old_opening = old["openings"][old_index]
                position, rotation = distance(old["effective_targets_tcp"][old_index], measured)
                conflict = bool(
                    latest[0] > opening + config["release_clearance"]
                    and latest[0] > old_opening + config["opening_revision_gap"]
                    and old_opening <= opening + config["release_clearance"]
                )
            if conflict:
                reason = "unknown_inner_margin"
        count = min(config["stride"], len(rows) - step)
        before = expected_state(None if query == 0 else step - config["stride"])
        after = expected_state(step) if count == config["stride"] else before
        expected = {
            "step": step,
            "source": "latest",
            "reason": reason,
            "latest_source_step": chunk["step"],
            "committed_source_step": None if old is None else old["step"],
            "absolute_first_endpoint": step + 1,
            "conflict_without_safety": conflict,
            "previous_inner_margin_rad": None,
            "available_endpoints": len(latest),
            "query": query,
            "closing_episode": closing,
            "baseline_periods": count,
            "terminal_partial_not_committed": count < config["stride"],
            "state_before": before,
            "state_after": after,
            "baseline_observation_label": "unchanged_baseline",
        }
        for key, value in expected.items():
            prefix.require(record[key] == value, f"Independent baseline decision differs: {key}")
        for key, value in (("current_position_error_m", position), ("current_rotation_error_rad", rotation)):
            if value is None:
                prefix.require(record[key] is None, f"Unavailable pose distance: {key}")
            else:
                close(record[key], value)
        np.testing.assert_array_equal(record["poses"], np.asarray(chunk["effective_targets_tcp"])[:16])
        np.testing.assert_array_equal(record["openings"], latest[:16])
        reasons[reason] += 1
        conflicts += int(conflict)
    summary = {
        "baseline_periods": len(rows),
        "queries": len(records),
        "conflicts_without_safety": conflicts,
        "unknown_inner_conflicts": reasons["unknown_inner_margin"],
        "holds": 0,
        "bridges": 0,
        "reasons": dict(sorted(reasons.items())),
        "first_intervention_step": None,
    }
    prefix.require(produced["summary"] == summary, "Independent baseline summary")
    return summary


def verify_studied(inputs, produced, config):
    data = binding.read_studied(inputs)
    step, old, latest = data["step"], data["old"], data["latest"]
    oi, ni = step - old["step"], step - latest["step"]
    op, oo = np.asarray(old["effective_targets_tcp"])[oi:], np.asarray(old["openings"])[oi:]
    np_, no = np.asarray(latest["effective_targets_tcp"])[ni:], np.asarray(latest["openings"])[ni:]
    position, rotation = distance(op[0], data["measured_tcp"])
    conflict = bool(
        no[0] > data["measured_opening"] + config["release_clearance"]
        and no[0] > oo[0] + config["opening_revision_gap"]
        and oo[0] <= data["measured_opening"] + config["release_clearance"]
    )
    prefix.require(
        data["closing_episode"]
        and data["previous_inner_margin_rad"] >= config["joint_buffer_rad"]
        and conflict
        and position <= config["current_position_gate_m"]
        and rotation <= config["current_rotation_gate_rad"]
        and min(len(oo), len(no)) >= config["stride"] + config["bridge_steps"],
        "Studied state independently meets the registered gates",
    )
    expected_p, expected_o = np_.copy(), no.copy()
    hold, bridge = config["stride"], config["bridge_steps"]
    expected_p[:hold], expected_o[:hold] = op[:hold], oo[:hold]
    for i in range(bridge):
        alpha, index = (i + 1) / bridge, hold + i
        expected_p[index, :3, 3] = (1 - alpha) * op[index, :3, 3] + alpha * np_[index, :3, 3]
        rotation_pair = Rotation.from_matrix([op[index, :3, :3], np_[index, :3, :3]])
        expected_p[index, :3, :3] = Slerp([0, 1], rotation_pair)([alpha]).as_matrix()[0]
        expected_o[index] = (1 - alpha) * oo[index] + alpha * no[index]
    decision = produced["decision"]
    expected = {
        "step": step,
        "source": "committed",
        "reason": "bounded_hold",
        "latest_source_step": latest["step"],
        "committed_source_step": old["step"],
        "absolute_first_endpoint": step + 1,
        "conflict_without_safety": True,
        "available_endpoints": len(no),
    }
    for key, value in expected.items():
        prefix.require(decision[key] == value, f"Studied selection differs: {key}")
    close(decision["previous_inner_margin_rad"], data["previous_inner_margin_rad"])
    close(decision["current_position_error_m"], position)
    close(decision["current_rotation_error_rad"], rotation)
    close(decision["poses"], expected_p[:16])
    close(decision["openings"], expected_o[:16])
    maximum_position = max(distance(expected_p[i], np_[i])[0] for i in range(hold))
    maximum_rotation = max(distance(expected_p[i], np_[i])[1] for i in range(hold))
    summary = produced["summary"]
    for key, value in (
        ("step", step),
        ("source", "committed"),
        ("reason", "bounded_hold"),
        ("closing_episode", True),
    ):
        prefix.require(summary[key] == value, f"Studied summary differs: {key}")
    for key, value in (
        ("measured_opening", data["measured_opening"]),
        ("previous_inner_margin_rad", data["previous_inner_margin_rad"]),
        ("max_selected_vs_latest_position_m", maximum_position),
        ("max_selected_vs_latest_rotation_rad", maximum_rotation),
    ):
        close(summary[key], value)
    prefix.require(
        produced["selection_executed"] is False and produced["state_after_abort"] == expected_state(None),
        "No shadow action or memory commit",
    )
    return summary


def worker(index):
    prefix.environment()
    independent()
    directory = binding.ROOT / "checker"
    registration = prefix.read(prefix.reference(directory / "registration.json"))
    prefix.require(
        registration["phase"] == "checker" and registration["source_refs"] == binding.source_refs(),
        "Frozen checker sources",
    )
    inputs, producer_refs = registration["inputs"], registration["producer"]
    config = prefix.read(inputs["config_ref"])
    if index == 30:
        produced = prefix.read(producer_refs["studied"])
        summary = verify_studied(inputs, produced, config)
        output = {"complete": True, "passed": True, "summary": summary, "candidate_modules_unloaded": True}
        path = directory / "studied/result.json"
    else:
        prefix.require(type(index) is int and 0 <= index < 30, "One registered baseline case")
        item = inputs["cases"][index]
        produced = prefix.read(producer_refs["cases"][index])
        prefix.require(
            produced["complete"] is produced["passed"] is True and produced["input_refs"] == item["refs"],
            "Bound original case",
        )
        chunks, rows, tool = binding.read_case(item)
        summary = verify_case(item, produced, chunks, rows, tool, config)
        for ref in item["refs"].values():
            prefix.checked(ref)
        output = {
            "complete": True,
            "passed": True,
            "case_id": item["case_id"],
            "family": item["family"],
            "summary": summary,
            "candidate_modules_unloaded": True,
        }
        path = directory / f"case_{index:02d}/result.json"
    independent()
    print(publish(path, output), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checker", action="store_true")
    group.add_argument("--worker", type=int)
    args = parser.parse_args()
    independent()
    if args.checker:
        binding.execute_phase("checker", "scripts.verification.reference_ports.check_bounded_release_revision")
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
