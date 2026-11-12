"""Reconstruct saved pad/bowl projections without stepping or changing a policy."""

import copy
import hashlib
from pathlib import Path

from s2a.research.native_runtime.pad_alignment import alignment, project_box
from s2a.research.native_runtime.saved_contacts import target_contacts
from s2a.research.native_runtime.saved_geometry_v2 import reconstruct_endpoint as reconstruct_geometry
from scripts.simulation.runtime import audit_reset_bowl_fixed_h40 as fixed_audit
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import diagnose_reset_bowl_saved_geometry_v2 as recovery
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/gripper/bowl_pad_alignment_20261112.json"
PLAN = ROOT / "plans/v0.0.1/tracking/gripper/bowl_pad_alignment_20261112.md"
OUTPUT = ROOT / "outputs/bowl_pad_alignment_20261112"
QA = ROOT / "outputs/qa/bowl_pad_alignment_20261112/result.json"
DRY = ROOT / "outputs/bowl_pad_alignment_dry_20261112/result.json"
MODULE = "scripts.simulation.runtime.diagnose_bowl_pad_alignment"
CONTACTS = {
    "path": str(ROOT / "outputs/reset_bowl_saved_contacts_audit_20261112/result.json"),
    "sha256": "678b9e3d3a2e56cc89808912e359c6cb6b3f94d356e46ff2e2adf42c9df33b50",
}
LEAD = {
    "path": str(ROOT / "outputs/bowl_h40_opening_lead_audit_20261112/result.json"),
    "sha256": "005cb8a5e91c8625f2bb7befee71cd9978d0d1fb1276b75629769ec4d67b8577",
}
FIXED = copy.deepcopy(recovery.FIXED)
FIXED.update(
    schema="s2a.bowl-pad-alignment.v1",
    output=str(OUTPUT.relative_to(ROOT)),
    prior_contacts_audit_sha256=CONTACTS["sha256"],
    lead_audit_sha256=LEAD["sha256"],
    prior_static_models=3,
    prior_static_forwards=204,
    left_pad="gripper0_finger1_pad_collision",
    right_pad="gripper0_finger2_pad_collision",
    target_contact_geoms=[f"akita_black_bowl_1_g{i}" for i in range(1, 41)],
)
_BASE_LOAD, _BASE_SOURCES, _BASE_RUN = recovery.load_inputs, recovery.source_refs, recovery.run_geometry
_PINNED = (
    (
        "scripts/simulation/runtime/diagnose_reset_bowl_saved_geometry_v2.py",
        "e19d6724fbea99d4819fa380aece6ecbfd867a4842f19659ce721e3f8765fd5b",
    ),
    (
        "src/s2a/research/native_runtime/saved_geometry_v2.py",
        "1dbd196e8e46218bb78305dc9eae4c7565473583fed28397097b475fda1edc59",
    ),
    (
        "src/s2a/research/native_runtime/saved_contacts.py",
        "e419cd2bdae97c7ef22c049ab5cc44901bdca2384e5d53c9c162ed95cdd15546",
    ),
)


def load_inputs(*, verify_assets=True):
    inputs = _BASE_LOAD(verify_assets=verify_assets)
    contacts, lead = prefix.read(CONTACTS), prefix.read(LEAD)
    prefix.require(
        contacts["complete"] is contacts["passed"] is lead["complete"] is lead["passed"] is True
        and contacts["control_qualified"] is lead["control_qualified"] is False
        and contacts["cumulative_models"] == inputs["config"]["prior_static_models"]
        and contacts["cumulative_forwards"] == inputs["config"]["prior_static_forwards"]
        and contacts["audit_forwards"] == contacts["audit_physics_steps"] == lead["audit_physics_steps"] == 0
        and lead["new_producer_periods"] == lead["new_checker_periods"] == 48,
        "Both audited development branches and actual prior work are required",
    )
    prior_contact = prefix.read(contacts["source_ref"])
    prior_worker = prefix.read(prior_contact["worker_ref"])
    candidate, checker = prefix.read(lead["producer_branch_ref"]), prefix.read(lead["checker_branch_ref"])
    candidate_first, candidate_periods = fixed_audit.flatten_fixed(candidate)
    checker_continuation = checker["runtime"]["continuation"]
    replay_first = checker_continuation["first_window"]["periods"]
    replay_periods = [row for segment in checker_continuation["replays"] for row in segment["periods"]]
    prefix.require(
        candidate["complete"] is candidate["passed"] is checker["complete"] is checker["passed"] is True
        and checker["candidate_modules_unloaded"] is True
        and len(candidate_first) == len(replay_first) == 8
        and len(candidate_periods) == len(replay_periods) == 40,
        "A completed 48-period lead and independent replay are required",
    )
    comparison.compare_periods(candidate_first, inputs["fixed_periods"][:8])
    comparison.compare_periods(candidate_first, replay_first)
    comparison.compare_periods(candidate_periods, replay_periods)
    inputs["old_periods"] = candidate_first + candidate_periods
    inputs["old"] = dict(
        runtime=dict(
            reset_render=candidate["runtime"]["reset_render"],
            streaming_captures=[candidate["runtime"]["fixed_reference_capture"]],
        )
    )
    inputs["prior_contacts"] = prior_worker["geometry"]
    refs = dict(
        prior_contacts_audit=CONTACTS,
        prior_contacts_source=contacts["source_ref"],
        prior_contacts_worker=prior_contact["worker_ref"],
        lead_audit=LEAD,
        lead_producer=lead["producer_branch_ref"],
        lead_checker=lead["checker_branch_ref"],
    )
    for key, ref in refs.items():
        prefix.checked(ref, inputs["source_sha256"])
        inputs["refs"][key] = ref
    return inputs


def source_refs():
    files = (
        recovery.CONFIG,
        recovery.PLAN,
        Path(__file__).resolve(),
        ROOT / "src/s2a/research/native_runtime/pad_alignment.py",
        ROOT / "src/s2a/research/native_runtime/saved_contacts.py",
        ROOT / "tests/research/native_runtime/test_pad_alignment.py",
        ROOT / "tests/research/native_runtime/test_diagnose_bowl_pad_alignment.py",
        ROOT / "reports/tracking/references/reset_bowl_saved_contacts_20261112.md",
        ROOT / "reports/tracking/gripper/bowl_h40_opening_lead_20261112.md",
    )
    return _BASE_SOURCES() + [prefix.reference(path) for path in files]


def reconstruct_endpoint(model, data, state, *, site_id, body_id, site_from_tcp, forward):
    import mujoco

    point = reconstruct_geometry(
        model, data, state, site_id=site_id, body_id=body_id, site_from_tcp=site_from_tcp, forward=forward
    )
    kind = mujoco.mjtObj.mjOBJ_GEOM
    ids = {
        name: int(mujoco.mj_name2id(model, kind, name))
        for name in [FIXED["left_pad"], FIXED["right_pad"], *FIXED["target_contact_geoms"]]
    }
    prefix.require(all(index >= 0 for index in ids.values()), "Every registered geom must exist")
    left, right = ids[FIXED["left_pad"]], ids[FIXED["right_pad"]]
    details = alignment(
        data.geom_xpos[left],
        data.geom_xpos[right],
        data.xpos[body_id],
        data.site_xpos[site_id],
        data.site_xmat[site_id].reshape(3, 3),
    )
    axis, origin = details["jaw_axis_world"], details["pad_midpoint_world_m"]
    boxes = []
    for name, index in ids.items():
        prefix.require(int(model.geom_type[index]) == int(mujoco.mjtGeom.mjGEOM_BOX), "Registered geoms must be boxes")
        center, rotation, size = (
            data.geom_xpos[index].copy(),
            data.geom_xmat[index].reshape(3, 3).copy(),
            model.geom_size[index].copy(),
        )
        boxes.append(
            dict(
                name=name,
                center_world_m=center,
                rotation_world=rotation,
                half_size_m=size,
                jaw_axis_interval_m=project_box(center, rotation, size, origin, axis),
            )
        )
    target = boxes[2:]
    details["box_projections"] = boxes
    details["target_axis_interval_m"] = [
        min(row["jaw_axis_interval_m"][0] for row in target),
        max(row["jaw_axis_interval_m"][1] for row in target),
    ]
    point["pad_alignment"] = details
    point["contacts"] = target_contacts(
        model,
        data,
        mujoco=mujoco,
        body_id=body_id,
        target_names=FIXED["target_contact_geoms"],
        left_pad=FIXED["left_pad"],
        right_pad=FIXED["right_pad"],
    )
    return point


def run_geometry(inputs, counters, *, mujoco):
    geometry = _BASE_RUN(inputs, counters, mujoco=mujoco)
    geometry["points"]["lead"] = geometry["points"].pop("reobserved")
    geometry["summaries"]["lead"] = geometry["summaries"].pop("reobserved")
    for row in geometry["capture_checks"]:
        if row["branch"] == "reobserved":
            row["branch"] = "lead"
    prior = inputs["prior_contacts"]
    prefix.require(geometry["reset"]["contacts"] == prior["reset"]["contacts"], "Prior reset contacts must match")
    for point, old in zip(geometry["points"]["fixed"], prior["points"]["fixed"], strict=True):
        prefix.require(
            point["contacts"] == old["contacts"] and point["state_sha256"] == old["state_sha256"],
            "All baseline endpoint contacts and states must match",
        )
    geometry["contact_label_mismatches"] = {
        name: [point["endpoint_step"] for point in rows if point["contacts"]["bilateral"] != point["target_grasp"]]
        for name, rows in geometry["points"].items()
    }
    return geometry


def bind():
    prefix.require(
        all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == sha for path, sha in _PINNED),
        "Prior geometry and contact sources must remain unchanged",
    )
    engine = recovery.bind_recovery()
    engine.CONFIG, engine.PLAN, engine.OUTPUT, engine.QA, engine.DRY = CONFIG, PLAN, OUTPUT, QA, DRY
    engine.MODULE, engine.FIXED = MODULE, FIXED
    engine.reconstruct_endpoint = reconstruct_endpoint
    engine.load_inputs, engine.source_refs, engine.run_geometry = load_inputs, source_refs, run_geometry
    return engine


def main():
    bind().main()


if __name__ == "__main__":
    main()
