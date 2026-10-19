"""Bound all gripper collision geometry independently of its orientation.

This is a kinematic diagnostic, not a controller or a task evaluation. Sliding
finger coordinates are visited only in a separate disposable environment.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from s2a.artifacts import code_fingerprint, load_config, sha256, write_json
from s2a.libero_simulation import current_observation, make_environment, source_pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/libero_wam_effect_20261019.json")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    c, out = load_config(args.config)
    if c["seed"] != args.seed:
        raise ValueError("seed differs")
    dest = out / "contact_bounds"
    dest.mkdir(exist_ok=False)
    record = json.loads(Path(c["fixture"]).read_text())["episodes"][0]
    env, _ = make_environment(c, dest, record, 0)
    try:
        model, data = env.sim.model, env.sim.data
        source = source_pose(current_observation(env))
        start_time = float(data.time)
        joints = []
        for i in range(model.njnt):
            name = model.joint_id2name(i)
            if name.startswith("gripper0_"):
                if model.jnt_type[i] != 2 or not model.jnt_limited[i]:
                    raise ValueError("bound requires limited sliding gripper joints")
                joints.append(dict(name=name, qpos_index=int(model.jnt_qposadr[i]), range=model.jnt_range[i]))
        if len(joints) != 2:
            raise ValueError("expected two parallel fingers")
        geometries = {}
        for i in range(model.ngeom):
            name = model.geom_id2name(i) or ""
            collidable = model.geom_contype[i] != 0 or model.geom_conaffinity[i] != 0
            if collidable and name.startswith("gripper0_"):
                if model.geom_rbound[i] <= 0:
                    raise ValueError("unbounded or empty collision geometry")
                geometries[name] = dict(id=i, geom_rbound_m=float(model.geom_rbound[i]), corner_centers=[])
        if len(geometries) < 5:
            raise ValueError("incomplete gripper collision geometry")
        for corner in itertools.product((0, 1), repeat=len(joints)):
            for joint, side in zip(joints, corner, strict=True):
                data.qpos[joint["qpos_index"]] = joint["range"][side]
            env.sim.forward()
            np.testing.assert_allclose(source_pose(current_observation(env)), source, atol=1e-12, rtol=0)
            for row in geometries.values():
                row["corner_centers"].append(source[:3, :3].T @ (data.geom_xpos[row["id"]] - source[:3, 3]))
        for row in geometries.values():
            row["radius_from_source_m"] = (
                float(np.max(np.linalg.norm(row["corner_centers"], axis=1))) + row["geom_rbound_m"]
            )
        bowl_center = data.get_body_xpos("akita_black_bowl_1_main").copy()
        bowl = []
        for i in range(model.ngeom):
            name = model.geom_id2name(i) or ""
            if name.startswith("akita_black_bowl_") and (model.geom_contype[i] or model.geom_conaffinity[i]):
                bowl.append(
                    dict(
                        name=name,
                        radius_from_body_m=float(
                            np.linalg.norm(data.geom_xpos[i] - bowl_center) + model.geom_rbound[i]
                        ),
                    )
                )
        if not bowl or float(data.time) != start_time:
            raise ValueError("invalid geometry diagnostic")
        result = dict(
            code=code_fingerprint(),
            inputs_sha256={c["fixture"]: sha256(c["fixture"])},
            model_xml_sha256=hashlib.sha256(model.get_xml().encode()).hexdigest(),
            joints=joints,
            collision_geometries=geometries,
            gripper_radius_m=max(r["radius_from_source_m"] for r in geometries.values()),
            finger_radius_m=max(r["radius_from_source_m"] for name, r in geometries.items() if "finger" in name),
            bowl_collision_geometries=bowl,
            bowl_radius_m=max(r["radius_from_body_m"] for r in bowl),
            proof="For limited slides, each geom center is affine in finger q; norm is convex, hence corner maximum bounds every legal q. MuJoCo geom_rbound encloses that geometry. Triangle inequality is invariant under arbitrary hand rotation.",
            limitation="A positive separation concerns this gripper, not all other robot links or arbitrary unprescribed excursions between endpoints.",
            physics_steps=0,
        )
        write_json(dest / "bounds.json", result)
        print(json.dumps({k: result[k] for k in ("gripper_radius_m", "finger_radius_m", "bowl_radius_m")}))
    finally:
        env.close()


if __name__ == "__main__":
    main()
