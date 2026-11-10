"""Describe saved development-limit crossings and nearby endpoint contacts."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from s2a.infrastructure.artifacts import REPO
from s2a.research.wam.tracking.preview.development_reference_geometry import StaticReferenceProbe, migrate_xml
from s2a.research.wam.training.clean.clean_training_runtime import publish, read, require
from scripts.wam.tracking.preview.diagnose_development_reference_geometry import (
    REVIEW,
    SNAPSHOT,
    asset_paths,
    safe_environment,
    safe_ref,
)

OUTPUT = REPO / "outputs/multicase_joint_limit_context_20261110"
SUMMARY = {
    "path": str(REPO / "outputs/multicase_geometry_analysis_20261110/summary.json"),
    "sha256": "fa34f7271a8bb533ec4497e2167b83e8958c6e4a541e48c273ea16f01f372a5a",
}
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/multicase_joint_limit_context_20261110.md"
QA = REPO / "outputs/qa/multicase_joint_limit_context_20261110/result.json"
TEST = REPO / "tests/research/wam/tracking/preview/test_multicase_joint_limit_context.py"
LIBRARY = REPO / "src/s2a/research/wam/tracking/preview/development_reference_geometry.py"
HELPER = REPO / "scripts/wam/tracking/preview/diagnose_development_reference_geometry.py"
EXPECTED = [2, 6, 7, 8, 9, 10, 11]


def first_limit_context(q, limits):
    q = np.asarray(q, dtype=float)
    limits = np.asarray(limits, dtype=float)
    require(q.ndim == 2 and q.shape[1] == 7 and len(q) > 0, "Seven joint positions per endpoint")
    require(limits.shape == (7, 2) and np.all(limits[:, 0] < limits[:, 1]), "Valid seven joint ranges")
    require(np.isfinite(q).all() and np.isfinite(limits).all(), "Finite physical positions and limits")
    margins = np.minimum(q - limits[:, 0], limits[:, 1] - q)
    outside = np.any(margins < 0, axis=1)
    events = np.flatnonzero(outside)
    if not len(events):
        return {"first_outside_endpoint": None, "outside_endpoints": 0, "minimum_margin_rad": float(margins.min())}
    first = int(events[0])
    bad = np.flatnonzero(margins[first] < 0)
    prior_high = np.flatnonzero(margins[:first, bad[0]] >= 0.05)
    last_high = None if not len(prior_high) else int(prior_high[-1])
    first_low = 0 if last_high is None else last_high + 1
    return {
        "first_outside_endpoint": first,
        "first_outside_joint_numbers": (bad + 1).tolist(),
        "outside_endpoints": int(outside.sum()),
        "minimum_margin_rad": float(margins.min()),
        "last_at_or_above_buffer_endpoint": last_high,
        "first_below_buffer_after_last_high_endpoint": first_low,
        "cycles_from_buffer_loss_to_first_outside": first - first_low,
    }


def robot_scene_contacts(model, data):
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(model.ngeom)]

    def robot(name):
        return name.startswith("robot0_") or name.startswith("gripper0_")

    contacts = []
    for contact in data.contact[: data.ncon]:
        first, second = names[int(contact.geom1)], names[int(contact.geom2)]
        if contact.dist <= 0 and robot(first) != robot(second):
            contacts.append(
                {
                    "robot_geom": first if robot(first) else second,
                    "scene_geom": second if robot(first) else first,
                    "distance_m": float(contact.dist),
                }
            )
    return contacts


def main():
    safe_environment()
    require(not OUTPUT.exists(), "New output only; no automatic retry")
    summary, review, snapshot = read(SUMMARY), read(REVIEW), read(SNAPSHOT)
    require(len(summary["cases"]) == len(review["cases"]) == 12, "All fixed development cases")
    selected = [i for i, case in enumerate(summary["cases"]) if case["endpoint_outside_any_joint"] > 0]
    require(selected == EXPECTED, "Original seven-case outside set")
    qa_ref = safe_ref(QA)
    qa = read(qa_ref)
    require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "QA closed",
    )
    sources = [safe_ref(p) for p in (Path(__file__), TEST, PLAN, LIBRARY, HELPER)]
    qa_sources = read(qa["source_before_ref"])["sources"]
    require(
        all(qa_sources.get(str(Path(r["path"]).relative_to(REPO))) == r["sha256"] for r in sources),
        "QA source identity",
    )
    paths = asset_paths(snapshot)
    reg = publish(
        OUTPUT / "registration.json",
        {
            "summary_ref": SUMMARY,
            "review_ref": REVIEW,
            "snapshot_ref": SNAPSHOT,
            "qa_ref": qa_ref,
            "source_refs": sources,
            "case_indices": selected,
            "max_models": 7,
            "max_forward_queries": 28,
            "physics_steps": 0,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
    )
    source_map = review["source_and_artifact_sha256"]
    all_refs = {}
    case_refs, queries, model_count = [], 0, 0
    for index, case in enumerate(summary["cases"]):
        require(safe_ref(case["result_ref"]["path"]) == case["result_ref"], "Prior case result identity")
        data = read(case["result_ref"])
        needed = {}
        for suffix in ("/actual_trajectory.npz", "/steps.jsonl", "/model.xml"):
            matches = [p for p in data["input_sha256"] if p.endswith(suffix)]
            require(len(matches) == 1, "Unambiguous source file")
            path = matches[0]
            ref = safe_ref(path)
            require(ref["sha256"] == data["input_sha256"][path] == source_map[path], "Original reviewed source SHA")
            needed[suffix] = Path(path)
            all_refs[path] = ref["sha256"]
        rows = [json.loads(line) for line in needed["/steps.jsonl"].read_text().splitlines()]
        xml = ET.fromstring(needed["/model.xml"].read_bytes())
        require(xml.find("compiler").get("angle") == "radian", "Original joint angle units")
        joints = {j.get("name"): j for j in xml.iter("joint")}
        limits = np.asarray([[float(x) for x in joints[f"robot0_joint{j}"].get("range").split()] for j in range(1, 8)])
        with np.load(needed["/actual_trajectory.npz"], allow_pickle=False) as arrays:
            states = arrays["sim_states"]
            require(len(states) == 601 and len(rows) == 600, "All saved states and steps")
            q = states[:, 1:8]
            np.testing.assert_allclose(
                q[:-1], [r["coupled_dynamics"]["robot_joint_position"] for r in rows], atol=1e-12, rtol=0
            )
            context = first_limit_context(q, limits)
            require(context["outside_endpoints"] == case["endpoint_outside_any_joint"], "Original outside count")
            context.update(
                {
                    "case": case["case"],
                    "target": case["target"],
                    "source_sha256": {str(p): all_refs[str(p)] for p in needed.values()},
                }
            )
            if index in selected:
                print(json.dumps({"case_index": index, "stage": "model_initialization_requested"}), flush=True)
                model = mujoco.MjModel.from_xml_string(
                    migrate_xml(needed["/model.xml"].read_bytes(), paths).decode("utf-8")
                )
                model_count += 1
                print(
                    json.dumps({"case_index": index, "stage": "model_initialized", "models": model_count}), flush=True
                )
                probe = StaticReferenceProbe(model, case["target"], budget=4)
                joint_ids = [
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{j}") for j in range(1, 8)
                ]
                require(min(joint_ids) >= 0, "All seven model joint names")
                np.testing.assert_array_equal([model.jnt_qposadr[j] for j in joint_ids], np.arange(7))
                np.testing.assert_allclose([model.jnt_range[j] for j in joint_ids], limits, atol=1e-12, rtol=0)
                velocity_indices = [model.jnt_dofadr[j] for j in joint_ids]
                qvel = states[:, 1 + model.nq + np.asarray(velocity_indices)]
                np.testing.assert_allclose(
                    qvel[:-1], [r["coupled_dynamics"]["robot_joint_velocity"] for r in rows], atol=1e-12, rtol=0
                )
                first = context["first_outside_endpoint"]
                bad = np.asarray(context["first_outside_joint_numbers"]) - 1
                context["first_outside_joint_values_rad"] = q[first, bad].tolist()
                context["first_outside_joint_velocities_rad_s"] = qvel[first, bad].tolist()
                context["first_outside_joint_ranges_rad"] = limits[bad].tolist()
                context["previous_dispatch_route"] = (
                    None if first == 0 else rows[first - 1]["tracking_refinement"]["route"]
                )
                context["next_dispatch_route"] = None if first == 600 else rows[first]["tracking_refinement"]["route"]
                context["target_grasp_after_previous_dispatch"] = (
                    None if first == 0 else bool(rows[first - 1]["objects"][case["target"]]["grasp_contact"])
                )
                context["task_success_after_previous_dispatch"] = (
                    None if first == 0 else bool(rows[first - 1]["success"])
                )
                endpoints = {context["first_below_buffer_after_last_high_endpoint"], first}
                if first > 0:
                    endpoints.add(first - 1)
                if context["last_at_or_above_buffer_endpoint"] is not None:
                    endpoints.add(context["last_at_or_above_buffer_endpoint"])
                context["saved_endpoint_contacts"] = []
                for endpoint in sorted(endpoints):
                    probe.restore(states[endpoint])
                    context["saved_endpoint_contacts"].append(
                        {
                            "endpoint": endpoint,
                            "time_s": float(states[endpoint, 0]),
                            "robot_scene_contacts": robot_scene_contacts(model, probe.data),
                        }
                    )
                    print(
                        json.dumps({"case_index": index, "endpoint": endpoint, "queries": queries + probe.queries}),
                        flush=True,
                    )
                queries += probe.queries
                context["mj_forward_queries"] = probe.queries
            else:
                context["mj_forward_queries"] = 0
        case_refs.append(publish(OUTPUT / f"case_{index:02d}/result.json", context))
    require(model_count == 7 and queries <= 28, "Bounded seven-case diagnostic")
    require(all(safe_ref(p)["sha256"] == h for p, h in all_refs.items()), "Inputs unchanged during analysis")
    require(
        asset_paths(read(SNAPSHOT)) == paths and all(safe_ref(r["path"]) == r for r in sources),
        "Assets and source unchanged",
    )
    print(
        publish(
            OUTPUT / "result.json",
            {
                "complete": True,
                "passed": True,
                "registration_ref": reg,
                "case_refs": case_refs,
                "case_indices_with_violation": selected,
                "model_initializations": model_count,
                "mj_forward_queries": queries,
                "physics_steps": 0,
                "world_model_calls": 0,
                "fresh_cases": 0,
                "sources_unchanged": True,
                "scope": "Saved simulator endpoints only; contact does not establish causation",
            },
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
