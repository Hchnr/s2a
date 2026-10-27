"""187 current finger/motor state and contact geometry of69 training anchors."""

from pathlib import Path

import mujoco
import numpy as np
from audit_wam_position import Evidence

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish


def main():
    out = REPO / "outputs/finger_intervention_coverage_20261025"
    if out.exists():
        raise FileExistsError(out)
    e = Evidence()
    e.read(__file__)
    e.read("plans/v0.0.1/187_finger_intervention_coverage_20261025.md")
    equations = e.json("outputs/left_table_forward_equations_20261025/result.json")
    refs = []
    seen = set()
    for row in equations["contrast_sources"]:
        key = (row["dataset"], row["anchor_group"])
        if key not in seen:
            seen.add(key)
            refs.append(dict(dataset=key[0], group=key[1], split="fit", ref=row["baseline_ref"]))
    assert len(refs) == 69
    config = e.json("configs/left_table_forward_20261025.json")
    closure = e.json(config["paired_closure_ref"]["path"])
    summary = e.json(closure["summary_ref"]["path"])
    for group in config["validation_anchor_groups"]:
        refs.append(dict(dataset="086", group=group, split="validation", ref=summary["cases"][7 * group]["arrays_ref"]))
    rows = []
    for row in refs:
        ref = row["ref"]
        e.check(ref["path"], ref["sha256"])
        xml = Path(ref["path"]).parent / "model.xml"
        e.read(xml)
        with np.load(ref["path"], allow_pickle=False) as trace:
            state = trace["sim_states"][-2].copy()
            q = trace["gripper_qpos"][-2].copy()
            v = trace["gripper_qvel"][-2].copy()
            offsets = trace["gripper_current_action_offsets"]
            index = len(offsets) - 3
            motor = trace["gripper_current_action_flat"][offsets[index] : offsets[index + 1]].copy()
            previous = trace["sent_commands"][-2].copy() if len(trace["sent_commands"]) > 1 else np.zeros(7)
        assert motor.shape == (2,)
        model = mujoco.MjModel.from_xml_path(str(xml))
        data = mujoco.MjData(model)
        assert len(state) == 1 + model.nq + model.nv
        data.time = state[0]
        data.qpos[:] = state[1 : 1 + model.nq]
        data.qvel[:] = state[1 + model.nq :]
        mujoco.mj_forward(model, data)
        measured = []
        for name in ("gripper0_finger_joint1", "gripper0_finger_joint2"):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            measured.append(data.qpos[model.jnt_qposadr[jid]])
        np.testing.assert_allclose(measured, q, atol=1e-12, rtol=0)
        contacts = []
        for c in data.contact[: data.ncon]:
            bodies = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""
                for g in (c.geom1, c.geom2)
            ]
            if any("finger" in b for b in bodies):
                contacts.append(dict(bodies=bodies, distance=float(c.dist)))
        gap = float(q[0] - q[1])
        target_gap = float(0.04 + 0.02 * (motor[0] - motor[1]))
        rows.append(
            dict(
                dataset=row["dataset"],
                group=row["group"],
                split=row["split"],
                gap_m=gap,
                motor_target_gap_m=target_gap,
                gap_minus_target_m=gap - target_gap,
                finger_qvel=v.tolist(),
                previous_gripper_command=float(previous[-1]),
                contacts=contacts,
            )
        )
    e.recheck()
    publish(
        out / "verified.json",
        dict(
            complete=True,
            training_anchors=69,
            validation_anchors=3,
            rows=rows,
            source_sha256=e.hashes,
            physics_steps=0,
            fits=0,
            reconstructed_forces=False,
            limitation="Current geometry only; motor target gap is not measured gap or force, and contact existence alone does not prove retained object or future response",
        ),
    )
    for split in ("fit", "validation"):
        selected = [r for r in rows if r["split"] == split]
        print(
            split,
            "anchors",
            len(selected),
            "any finger contact",
            sum(bool(r["contacts"]) for r in selected),
            "target<=5mm and actualgap>=5mm",
            sum(r["motor_target_gap_m"] <= 0.005 and r["gap_m"] >= 0.005 for r in selected),
        )
    for r in rows[-3:]:
        print(r)


if __name__ == "__main__":
    main()
