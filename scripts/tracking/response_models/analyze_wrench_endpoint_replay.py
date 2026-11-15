"""Align completed endpoint wrench with the following recorded model error."""

import json
from pathlib import Path

import numpy as np

from s2a.infrastructure.publication import publish_result
from scripts.simulation.runtime import inspect_recorded_wrench as replay
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

ROOT = prefix.REPO
OUTPUT = replay.OUTPUT / "analysis.json"
TRACE_DIGESTS = {
    "historical": "b206b0a0f0ba98ce5c8c64b5ed9728e00b99e72651c1664fa9b32a28f696da96",
    "anticipatory": "0b2ca9485eb8fe8d7be5ffac27328f1c4ce198a1405200f1fe79fde91d9d2a82",
}


def aligned_periods(trace, branch):
    """Bind the input at t to action t and the measured error at t+1."""
    rows = {row["endpoint_step"]: row for row in trace["rows"]}
    if len(rows) != len(trace["rows"]):
        raise ValueError("Endpoint identities must be unique")
    periods = replay.admission.periods(branch)
    aligned = []
    for period in periods:
        step = period["step"]
        current, following = rows[step], rows[step + 1]
        endpoint = period["endpoint_receipt"]
        if (
            endpoint["endpoint_step"] != step + 1
            or endpoint["sim_state_sha256"] != following["sim_state_sha256"]
            or endpoint["sim_time"] != following["sim_time"]
            or current["sim_time"] >= following["sim_time"]
        ):
            raise ValueError("The completed action must bind its following endpoint")
        local = np.asarray(current["wrench"]["local"], dtype=float)
        predicted = np.asarray(period["decision"]["quality"]["first_tcp_error"], dtype=float)
        measured = np.asarray(period["tcp_error"], dtype=float)
        if any(value.shape != (6,) or not np.isfinite(value).all() for value in (local, predicted, measured)):
            raise ValueError("Finite six-dimensional wrench and TCP error are required")
        aligned.append(
            {
                "step": step,
                "input_endpoint_sha256": current["sim_state_sha256"],
                "result_endpoint_sha256": following["sim_state_sha256"],
                "force_norm_n": float(np.linalg.norm(local[:3])),
                "torque_norm_nm": float(np.linalg.norm(local[3:])),
                "position_innovation_m": float(np.linalg.norm(measured[:3] - predicted[:3])),
                "offline_gripper_contact": bool(current["offline_gripper_contacts"]),
            }
        )
    return aligned


def contact_summary(rows):
    result = {}
    for contact in (False, True):
        group = [row for row in rows if row["offline_gripper_contact"] is contact]
        if not group:
            raise ValueError("The descriptive comparison requires both endpoint groups")
        errors = np.asarray([row["position_innovation_m"] for row in group])
        forces = np.asarray([row["force_norm_n"] for row in group])
        result[str(contact).lower()] = {
            "periods": len(group),
            "median_position_innovation_m": float(np.median(errors)),
            "max_position_innovation_m": float(np.max(errors)),
            "median_force_norm_n": float(np.median(forces)),
            "max_force_norm_n": float(np.max(forces)),
        }
    return result


def main():
    traces, all_rows, refs = {}, {}, []
    for case, digest in TRACE_DIGESTS.items():
        ref = {"path": str(replay.OUTPUT / case / "wrench_trace.json"), "sha256": digest}
        trace = prefix.read(ref)
        branch = prefix.read(trace["producer_branch_ref"])
        prefix.require(trace["complete"] is trace["passed"] is True, "A complete wrench replay is required")
        refs.extend((ref, trace["producer_branch_ref"], trace["replay_ref"]))
        prefix.read(trace["replay_ref"])
        traces[case] = {row["endpoint_step"]: row for row in trace["rows"]}
        all_rows[case] = aligned_periods(trace, branch)
    shared = list(range(80, 122))
    old, new = traces["historical"], traces["anticipatory"]
    prefix.require(
        all(old[step]["sim_state_sha256"] == new[step]["sim_state_sha256"] for step in shared),
        "Shared endpoints must have identical state identities",
    )
    force_diff = max(
        float(np.max(np.abs(np.asarray(old[step]["wrench"]["local"]) - new[step]["wrench"]["local"])))
        for step in shared
    )
    prefix.require(force_diff == 0, "Shared endpoint sensor values must match")
    for ref in refs:
        prefix.checked(ref)
    result = {
        "complete": True,
        "passed": True,
        "source_refs": refs + [prefix.reference(Path(__file__).resolve())],
        "retrospective": True,
        "new_model_requests": 0,
        "new_physics_steps": 0,
        "shared_endpoint_count": len(shared),
        "shared_wrench_max_abs_difference": force_diff,
        "cases": {
            case: {"periods": rows, "offline_contact_summary": contact_summary(rows)} for case, rows in all_rows.items()
        },
    }
    print(json.dumps(publish_result(OUTPUT, result), ensure_ascii=False))


if __name__ == "__main__":
    main()
