"""Cross-check saved free-joint object poses and simulated contact forces."""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.infrastructure.publication import publish_result as publish
from s2a.research.wam.training.native.native_training import state_sha256
from scripts.simulation.runtime import prepare_early_box_face_feedback as admission
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.verification.response_models.audit_box_face_feedback import assert_same_observation

ROOT = prefix.REPO
RESULT = ROOT / "outputs/early_box_face_bowl_effect_repair_20261111/result.json"
RESULT_SHA = "d793d55960bfa601884d3d96121d4601c99aae040664dbe66273fd1d4f751c6d"
OUTPUT = ROOT / "outputs/early_box_face_bowl_effect_audit_20261111/result.json"


def raw_free_pose(state, address, nq=41):
    """Read a previously verified world-parent free-joint pose from saved qpos."""
    state = np.asarray(state, dtype=float)
    if (
        state.ndim != 1
        or not np.isfinite(state).all()
        or type(address) is not int
        or not 0 <= address <= nq - 7
        or len(state) < 1 + nq
    ):
        raise ValueError("A full finite state and valid free-joint qpos address are required")
    qpos = state[1 : 1 + nq]
    quaternion = qpos[address + 3 : address + 7]
    if np.linalg.norm(quaternion) == 0:
        raise ValueError("The saved free-joint quaternion must be nonzero")
    return qpos[address : address + 3].copy(), Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()


def raw_motion(reference_bowl, reference_cheese, current_bowl, current_cheese):
    """Compute world and relative displacement directly from saved free joints."""
    b0, rb0 = reference_bowl
    c0, rc0 = reference_cheese
    b1, rb1 = current_bowl
    c1, rc1 = current_cheese
    rel0, rel1 = rb0.T @ (c0 - b0), rb1.T @ (c1 - b1)
    rr0, rr1 = rb0.T @ rc0, rb1.T @ rc1
    return {
        "bowl_world": {
            "translation_m": float(np.linalg.norm(b1 - b0)),
            "rotation_rad": float(Rotation.from_matrix(rb0.T @ rb1).magnitude()),
        },
        "cheese_world": {
            "translation_m": float(np.linalg.norm(c1 - c0)),
            "rotation_rad": float(Rotation.from_matrix(rc0.T @ rc1).magnitude()),
        },
        "cheese_relative_to_bowl": {
            "translation_m": float(np.linalg.norm(rel1 - rel0)),
            "rotation_rad": float(Rotation.from_matrix(rr0.T @ rr1).magnitude()),
        },
    }


def direct_normal_forces(samples, pair):
    sums, records = [], []
    for sample in samples:
        normals = [
            float(c["wrench_contact_frame"][0])
            for c in sample["contacts"]
            if c["active_constraint"] and {c["geom1"], c["geom2"]} == pair
        ]
        if normals:
            prefix.require(
                all(np.isfinite(normals)) and min(normals) >= 0, "Finite nonnegative recorded normal forces required"
            )
            sums.append(sum(normals))
            records.extend(normals)
    return {
        "active_inner_samples": len(sums),
        "active_contact_records": len(records),
        "maximum_single_contact_normal_n": max(records, default=None),
        "maximum_sum_normal_n_per_sample": max(sums, default=None),
        "mean_sum_normal_n_per_active_sample": float(np.mean(sums)) if sums else None,
    }


def execute():
    inputs = admission.load_inputs()
    result_ref = {"path": str(RESULT), "sha256": RESULT_SHA}
    result = prefix.read(result_ref)
    prefix.require(
        result["complete"] is result["passed"] is result["sources_unchanged"] is True
        and result["control_qualified"] is False
        and result["total_models"] == 2
        and result["total_forward_queries"] == 22
        and result["world_model_calls"] == result["fresh_cases"] == 0
        and result["stats"]["physics_steps_attempted"] == 0,
        "The closed, cumulative static repair is required",
    )
    analysis = result["analysis"]
    addresses = {name: analysis["object_bindings"][name]["qpos_address"] for name in ("bowl", "cheese")}
    start_state = np.asarray(inputs["reference_table_geometry"]["states"][38], dtype=float)
    prefix.require(
        state_sha256(start_state) == inputs["early_box_face_feedback"]["config"]["branch_state_sha256"],
        "The common initial state SHA must match",
    )
    start = {name: raw_free_pose(start_state, address) for name, address in addresses.items()}
    for name in addresses:
        np.testing.assert_allclose(
            start[name][0], np.asarray(analysis["initial_poses"][name])[:3, 3], atol=1e-12, rtol=0
        )
        np.testing.assert_allclose(
            start[name][1], np.asarray(analysis["initial_poses"][name])[:3, :3], atol=1e-12, rtol=0
        )
    owner = prefix.read(
        {
            "path": str(ROOT / "outputs/early_box_face_feedback_20261111/result.json"),
            "sha256": "57cb3e4d35ecb5ba7f6e13d84006c11b4ad4b35cc7bb359ab71ec6071af69a20",
        }
    )
    raw = {}
    pair = {"gripper0_finger1_collision", "akita_black_bowl_1_g34"}
    for mode in ("original", "box_face"):
        producer = prefix.read(owner["producer_refs"][mode])
        continuation = producer["runtime"]["continuation"]
        raw[mode] = []
        for index, row in enumerate(analysis["rows"][mode]):
            state = np.asarray(continuation["probe"]["periods"][index]["actual_state"], dtype=float)
            prefix.require(
                state_sha256(state) == row["state_sha256"] and row["step"] == 119 + index,
                "Each audited object endpoint must match its saved state",
            )
            poses = {name: raw_free_pose(state, address) for name, address in addresses.items()}
            for name in addresses:
                np.testing.assert_allclose(poses[name][0], np.asarray(row[name + "_pose"])[:3, 3], atol=1e-12, rtol=0)
                np.testing.assert_allclose(poses[name][1], np.asarray(row[name + "_pose"])[:3, :3], atol=1e-12, rtol=0)
            assert_same_observation(
                raw_motion(start["bowl"], start["cheese"], poses["bowl"], poses["cheese"]),
                row["motion_from_start"],
                path=f"{mode}.{index}.motion",
            )
            raw[mode].append(poses)
        force = direct_normal_forces(continuation["inner_samples"][:150], pair)
        for field, value in force.items():
            assert_same_observation(value, analysis["finger_bowl_contact_forces"][mode][field], path=f"{mode}.{field}")
    for index, (left, right) in enumerate(zip(raw["original"], raw["box_face"], strict=True)):
        assert_same_observation(
            raw_motion(left["bowl"], left["cheese"], right["bowl"], right["cheese"]),
            analysis["same_endpoint_branch_differences"][index],
            path=f"branch_difference.{index}",
        )
    prefix.require(len(raw["original"]) == len(raw["box_face"]) == 6, "All six matched endpoints are required")
    sources = [
        prefix.reference(Path(__file__).resolve()),
        prefix.reference(ROOT / "tests/research/response_models/test_early_box_face_bowl_effect_audit.py"),
    ]
    print(
        publish(
            OUTPUT,
            dict(
                complete=True,
                passed=True,
                control_qualified=False,
                result_ref=result_ref,
                source_refs=sources,
                matched_endpoints=6,
                qpos_and_body_poses_match=True,
                direct_motion_and_forces_match=True,
                new_native_models=0,
                forward_queries=0,
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
                scope="saved free-joint qpos and solver receipts only; no policy or causal attribution",
            ),
        ),
        flush=True,
    )


if __name__ == "__main__":
    execute()
