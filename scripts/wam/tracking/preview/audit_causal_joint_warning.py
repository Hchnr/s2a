"""Audit fixed causal warnings against the twelve reviewed v4 development logs."""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from s2a.geometry import pose_error
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.joint_control.limits.causal_warning import RULES, joint_warnings, summarize_rule
from scripts.simulation.runtime import replay_multicase_development_prefix as safe

REPO = safe.REPO
OUTPUT = REPO / "outputs/causal_joint_warning_audit_20261110"
QA = REPO / "outputs/qa/causal_joint_warning_audit_20261110/result.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/causal_warning_audit_20261110.md"
CORE = REPO / "src/s2a/research/joint_control/limits/causal_warning.py"
TEST = REPO / "tests/research/joint_control/limits/test_causal_warning.py"
SUMMARY = {
    "path": str(REPO / "outputs/multicase_geometry_analysis_20261110/summary.json"),
    "sha256": "fa34f7271a8bb533ec4497e2167b83e8958c6e4a541e48c273ea16f01f372a5a",
}
EXPECTED_OUTSIDE = [2, 6, 7, 8, 9, 10, 11]


def sources():
    return [
        safe.reference(path)
        for path in (
            Path(__file__),
            PLAN,
            CORE,
            TEST,
            Path(safe.__file__),
            safe.CORE,
            REPO / "src/s2a/geometry/__init__.py",
            REPO / "src/s2a/infrastructure/publication.py",
            REPO / "src/s2a/infrastructure/native_environment.py",
            REPO / "src/s2a/infrastructure/repository_paths.py",
        )
    ]


def recorded_inputs(index, summary, review, seen):
    item = summary["cases"][index]
    data = safe.read(item["result_ref"], seen)
    safe.require(item["case"] == data["case"] == review["cases"][index]["case"], "Original case identities")
    safe.require(item["case"]["case_index"] == index, "Complete original case order")
    paths = {}
    for name in ("actual_trajectory.npz", "steps.jsonl", "model.xml"):
        matches = [path for path in data["input_sha256"] if path.endswith("/" + name)]
        safe.require(len(matches) == 1, "Unambiguous original source")
        path = matches[0]
        digest = data["input_sha256"][path]
        safe.require(review["source_and_artifact_sha256"].get(path) == digest, "Original independent-review SHA")
        paths[name] = safe.checked({"path": path, "sha256": digest}, seen)
    return item, paths


def xml_layout(path):
    root = ET.fromstring(path.read_bytes())
    safe.require(root.find("compiler").get("angle") == "radian", "Original joint units")
    joints = list(root.find("worldbody").iter("joint"))
    safe.require(
        [j.get("name") for j in joints[:7]] == [f"robot0_joint{i}" for i in range(1, 8)],
        "Seven first robot joint addresses",
    )
    types = [j.get("type", "hinge") for j in joints]
    safe.require(all(t in ("hinge", "slide", "free") for t in types), "Supported explicit joint widths")
    safe.require(all(t in ("hinge", "slide") for t in types[:7]), "One q and dof per robot joint")
    limits = np.asarray([[float(x) for x in j.get("range").split()] for j in joints[:7]])
    safe.require(limits.shape == (7, 2) and np.all(limits[:, 0] < limits[:, 1]), "Original seven limits")
    nq = sum(7 if t == "free" else 1 for t in types)
    nv = sum(6 if t == "free" else 1 for t in types)
    return limits, nq, nv


def analyze_case(item, paths):
    limits, nq, nv = xml_layout(paths["model.xml"])
    with paths["steps.jsonl"].open() as stream:
        rows = [json.loads(line) for line in stream]
    with np.load(paths["actual_trajectory.npz"], allow_pickle=False) as archive:
        states, times, tcp = archive["sim_states"].copy(), archive["sim_time"].copy(), archive["tcp_pose"].copy()
    safe.require(len(rows) == 600 and states.shape == (601, 1 + nq + nv), "Complete saved trajectory")
    safe.require(times.shape == (601,) and tcp.shape == (601, 4, 4), "Complete time and TCP states")
    np.testing.assert_array_equal([row["step"] for row in rows], np.arange(600))
    np.testing.assert_allclose(times, states[:, 0], atol=1e-12, rtol=0)
    np.testing.assert_allclose(np.diff(times), 0.05, atol=1e-10, rtol=0)
    np.testing.assert_allclose([row["physical_time_before_s"] for row in rows], times[:-1], atol=1e-10, rtol=0)
    q = states[:, 1:8]
    velocity = states[:-1, 1 + nq : 1 + nq + 7]
    np.testing.assert_allclose(
        q[:-1], [r["coupled_dynamics"]["robot_joint_position"] for r in rows], atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        velocity, [r["coupled_dynamics"]["robot_joint_velocity"] for r in rows], atol=1e-12, rtol=0
    )
    previous_errors = np.asarray([r["tcp_error"] for r in rows], dtype=float)
    recomputed = pose_error(np.asarray([r["desired_tcp"] for r in rows]), tcp[1:])
    np.testing.assert_allclose(previous_errors, recomputed, atol=1e-10, rtol=0)
    margins = np.minimum(q - limits[:, 0], limits[:, 1] - q).min(axis=1)
    safe.require(int((margins < 0).sum()) == item["endpoint_outside_any_joint"], "Original outside count")
    observations = []
    for t in range(600):
        prior = previous_errors[t - 1] if t else None
        observations.append(
            {
                "endpoint": t,
                "time_s": float(times[t]),
                "minimum_margin_rad": float(margins[t]),
                "previous_tcp_error": prior,
                "previous_pose_caps_satisfied": (
                    None
                    if prior is None
                    else bool(np.linalg.norm(prior[:3]) <= 0.01 and np.linalg.norm(prior[3:]) <= 0.05)
                ),
                "warnings": joint_warnings(q[t], velocity[t], limits),
            }
        )
    summaries = {}
    for rule in RULES:
        flags = [observation["warnings"][rule]["alarm"] for observation in observations]
        result = summarize_rule(flags, q, limits)
        first = result["first_alarm_while_buffer_safe_before_first_outside"]
        result["first_safe_alarm_observation"] = None if first is None else observations[first]
        summaries[rule] = result
    return {
        "case": item["case"],
        "original_endpoint_outside_count": item["endpoint_outside_any_joint"],
        "nq": nq,
        "nv": nv,
        "joint_limits_rad": limits,
        "rules": summaries,
        "observations": observations,
    }


def main():
    safe.environment()
    safe.require(not OUTPUT.exists(), "New output only; automatic retry prohibited")
    qa_ref = safe.reference(QA)
    qa = safe.read(qa_ref)
    safe.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "CPU QA closed",
    )
    source_refs = sources()
    qa_sources = safe.read(qa["source_before_ref"])["sources"]
    safe.require(
        all(qa_sources.get(str(Path(ref["path"]).relative_to(REPO))) == ref["sha256"] for ref in source_refs),
        "Audited source identity matches QA",
    )
    seen = {}
    summary, review = safe.read(SUMMARY, seen), safe.read(safe.REVIEW, seen)
    safe.require(len(summary["cases"]) == len(review["cases"]) == 12, "All twelve development cases")
    registration_ref = publish(
        OUTPUT / "registration.json",
        {
            "summary_ref": SUMMARY,
            "review_ref": safe.REVIEW,
            "qa_ref": qa_ref,
            "source_refs": source_refs,
            "cases": list(range(12)),
            "rules": RULES,
            "horizon_steps": 4,
            "physics_steps": 0,
            "model_initializations": 0,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
    )
    result = {"complete": False, "passed": False, "registration_ref": registration_ref, "case_refs": []}
    try:
        cases = []
        for index in range(12):
            item, paths = recorded_inputs(index, summary, review, seen)
            case = analyze_case(item, paths)
            result["case_refs"].append(publish(OUTPUT / f"case_{index:02d}.json", case))
            cases.append(case)
            print(json.dumps({"completed_case": index}), flush=True)
        outside = [i for i, case in enumerate(cases) if case["original_endpoint_outside_count"] > 0]
        safe.require(outside == EXPECTED_OUTSIDE, "Original seven-case endpoint violation set")
        safe.require(sources() == source_refs, "Sources unchanged")
        safe.require(all(safe.reference(path)["sha256"] == digest for path, digest in seen.items()), "Inputs unchanged")
        result.update(
            complete=True,
            passed=True,
            sources_unchanged=True,
            input_sha256=seen,
            case_indices_with_endpoint_outside=outside,
            physics_steps=0,
            model_initializations=0,
            world_model_calls=0,
            fresh_cases=0,
            control_qualified=False,
        )
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(OUTPUT / "result.json", result), flush=True)


if __name__ == "__main__":
    sys.exit(main())
