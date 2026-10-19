"""Conservative kinematic reach bound; no action model or controller fitting."""

import argparse
import json
from pathlib import Path

import numpy as np

from s2a.artifacts import code_fingerprint, sha256, write_json
from s2a.libero_simulation import current_observation, make_environment, source_pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/libero_wam_effect_20261019.json")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    c = json.loads(Path(args.config).read_text())
    if args.seed != c["seed"]:
        raise ValueError("seed differs")
    out = Path(c["output_dir"]) / "workspace_bound"
    out.mkdir(exist_ok=False)
    fixture = json.loads(Path(c["fixture"]).read_text())
    results = []
    for trial, record in enumerate(fixture["episodes"]):
        env, _ = make_environment(c, out / f"trial{trial}", record, trial)
        try:
            source = source_pose(current_observation(env))
            m = env.sim.model
            site = m.site_name2id(env.robots[0].controller.eef_name)
            body, chain = int(m.site_bodyid[site]), []
            while body != 0:
                chain.append(body)
                body = int(m.body_parentid[body])
            chain.reverse()
            first = next(i for i, b in enumerate(chain) if m.body_jntnum[b])
            entries = []
            for b in chain[first:]:
                joints = []
                for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
                    if m.jnt_type[j] != 3 or np.linalg.norm(m.jnt_pos[j]) > 1e-12:
                        raise ValueError("proof requires hinge joints at body origins")
                    joints.append(m.joint_id2name(j))
                entries.append(dict(body=m.body_id2name(b), translation=m.body_pos[b].copy(), joints=joints))
            # This first joint rotates about its own fixed world origin. Every
            # descendant offset has invariant length under arbitrary rotations.
            center = env.sim.data.body_xpos[chain[first]].copy()
            radius = float(
                sum(np.linalg.norm(m.body_pos[b]) for b in chain[first + 1 :]) + np.linalg.norm(m.site_pos[site])
            )
            if np.linalg.norm(source[:3, 3] - center) > radius + 1e-10:
                raise ValueError("observed pose violates reach bound")
            results.append(
                dict(
                    trial=trial,
                    center_world=center,
                    radius_m=radius,
                    upper_z_m=float(center[2] + radius),
                    chain=entries,
                    site_offset=m.site_pos[site].copy(),
                )
            )
        finally:
            env.close()
    result = dict(
        code=code_fingerprint(),
        inputs_sha256={c["fixture"]: sha256(c["fixture"])},
        cases=results,
        proof="Triangle inequality over fixed-length descendant body offsets and site offset; arbitrary hinge rotations preserve lengths. No joint-range or collision restriction is needed, so the enclosing sphere is conservative for every physically possible joint configuration.",
        physics_steps=0,
    )
    write_json(out / "bound.json", result)
    print(json.dumps(results, default=lambda v: v.tolist()))


if __name__ == "__main__":
    main()
