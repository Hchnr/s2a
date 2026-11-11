"""Audit the saved CPU replay and identify the changed native state fields."""

import json
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.infrastructure.publication import publish_result as publish
from s2a.research.wam.training.native.native_training import state_sha256
from scripts.simulation.runtime import causal_endpoint_diagnostic_admission as admission
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

ROOT = prefix.REPO
OUTPUT = ROOT / "outputs/causal_stream_endpoint_audit_20261111/result.json"
OWNER = {
    "path": str(ROOT / "outputs/causal_stream_endpoint_diagnostic_20261111/result.json"),
    "sha256": "bdd07c6d29feec6fb3cc9f5c6c1d245abe71c52bfa7e8f354f8dcebd53df4fe1",
}
BRANCH = {
    "path": str(ROOT / "outputs/causal_stream_endpoint_diagnostic_20261111/simulation/result.json"),
    "sha256": "14ecfbf12d1237667b8e19fec81a8a6d72d8253cd2676c0540be471152c8f4de",
}


def joint_spans(xml):
    root = ET.fromstring(xml)
    world = root.find("worldbody")
    prefix.require(world is not None, "The registered native worldbody is required")
    widths = {"free": 7, "ball": 4, "slide": 1, "hinge": 1}
    offset, spans = 0, []
    for node in world.iter():
        if node.tag not in ("joint", "freejoint"):
            continue
        kind = "free" if node.tag == "freejoint" else node.get("type", "hinge")
        prefix.require(kind in widths, "Every registered native joint type must be known")
        spans.append({"name": node.get("name"), "type": kind, "qpos_start": offset, "qpos_count": widths[kind]})
        offset += widths[kind]
    prefix.require(offset == 41, "Joint qpos layout must match the recorded native state")
    return spans


def summarize(inputs, owner, branch):
    prefix.require(
        owner["complete"] is owner["passed"] is owner["sources_unchanged"] is True
        and owner["branch_ref"] == BRANCH
        and owner["comparison"]["reproduced"] is True
        and owner["comparison"]["completed_actions"] == 40
        and owner["comparison"]["native_steps"] == 3000
        and all(value == 0 for value in owner["comparison"]["max_difference"].values())
        and branch["complete"] is branch["passed"] is branch["candidate_modules_unloaded"] is True
        and branch["source_stop_was_derived"] is True
        and branch["new_model_requests"] == branch["new_denoise_calls"] == branch["newly_selected_commands"] == 0,
        "A closed independent 40-action diagnostic with exact physical replay is required",
    )
    lifecycle = prefix.read(owner["workers_ref"])
    prefix.require(
        not lifecycle["launch_error"]
        and len(lifecycle["workers"]) == 1
        and lifecycle["workers"][0]["returncode"] == 0
        and lifecycle["workers"][0]["actual_wait"] is lifecycle["workers"][0]["reaped"] is True
        and lifecycle["workers"][0]["signal_sent"] is False,
        "The CPU diagnostic worker must have exited and been reaped",
    )
    runtime, events = branch["runtime"], branch["endpoint_events"]
    prefix.require(
        runtime["complete"] is runtime["passed"] is False
        and runtime["failure_stage"] == "continuation"
        and runtime["error"]["message"] == "Replay endpoint observation changed the physical state"
        and len(events) == 2,
        "The independently reproduced runtime failure must remain visible",
    )
    old = inputs["live"]["old"]["old"]["base"]["previous"]
    spans = joint_spans(old["record_xml"])
    matches = [span for span in spans if span["name"] == "cream_cheese_1_joint0"]
    prefix.require(
        matches == [{"name": "cream_cheese_1_joint0", "type": "free", "qpos_start": 16, "qpos_count": 7}],
        "The changed state must map to the registered cream-cheese free joint",
    )
    details = []
    for index, event in enumerate(events):
        before, after = event["before"], event["after"]
        left, right = np.asarray(before["state"], dtype=np.float64), np.asarray(after["state"], dtype=np.float64)
        changed = np.flatnonzero(left != right).tolist()
        delta = right - left
        prefix.require(
            event["step"] == 120
            and event["call_index"] == index
            and event["completed"] is True
            and before["nq"] == after["nq"] == 41
            and before["nv"] == after["nv"] == 37
            and len(left) == len(right) == 79
            and state_sha256(left) == before["sha256"]
            and state_sha256(right) == after["sha256"]
            and before["time"] == after["time"] == left[0] == right[0]
            and event["time_changed"] is False
            and event["changed_indices"] == changed == [20, 21, 22, 23]
            and float(np.max(np.abs(delta))) == event["max_absolute_difference"]
            and event["max_absolute_difference"] <= 3e-16,
            "The recorded difference must be limited to four quaternion components",
        )
        left_q, right_q = left[20:24], right[20:24]
        order = [1, 2, 3, 0]
        angle = float((Rotation.from_quat(left_q[order]).inv() * Rotation.from_quat(right_q[order])).magnitude())
        prefix.require(angle <= 1e-14, "The changed quaternion must describe the same measured rotation")
        details.append(
            {
                "call_index": index,
                "before_sha256": before["sha256"],
                "after_sha256": after["sha256"],
                "changed_flat_state_indices": changed,
                "max_absolute_difference": event["max_absolute_difference"],
                "rotation_difference_rad": angle,
                "quaternion_norm_before": float(np.linalg.norm(left_q)),
                "quaternion_norm_after": float(np.linalg.norm(right_q)),
            }
        )
    prefix.require(
        events[0]["before"]["sha256"] == events[1]["after"]["sha256"]
        and events[0]["after"]["sha256"] == events[1]["before"]["sha256"],
        "Both endpoint calls must exhibit the observed two-value cycle",
    )
    periods = [
        period for row in runtime["streaming_replay_partial"]["segments"] for period in row["continuation"]["periods"]
    ]
    tail = 0
    for period in reversed(periods):
        if not period["task_success"]:
            break
        tail += 1
    prefix.require(len(periods) == 40 and tail == 33, "The replayed task labels must preserve the partial result")
    return {
        "complete": True,
        "passed": True,
        "control_qualified": False,
        "reproduced_original_failure": True,
        "replayed_actions": 40,
        "replayed_native_steps": 3000,
        "replay_max_difference": owner["comparison"]["max_difference"],
        "task_success_tail_periods": tail,
        "changed_joint": matches[0],
        "events": details,
        "new_model_requests": 0,
        "new_physics_steps": 0,
        "owner_ref": OWNER,
        "branch_ref": BRANCH,
    }


def load_audit():
    prefix.environment()
    inputs = admission.load_inputs()
    return summarize(inputs, prefix.read(OWNER), prefix.read(BRANCH))


def main():
    print(json.dumps(publish(OUTPUT, load_audit())), flush=True)


if __name__ == "__main__":
    main()
