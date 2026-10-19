"""Compare on-demand cameras with actual robosuite camera observables."""

import argparse
import json
from pathlib import Path

import numpy as np

from s2a.artifacts import code_fingerprint, load_config, sha256, write_json
from s2a.cameras import render_current_cameras
from s2a.libero_simulation import make_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/libero_followup_20261019.json")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    c, output = load_config(args.config)
    if args.seed != c["seed"]:
        raise ValueError("seed differs from frozen config")
    policy = json.loads(Path(c["policy_config"]).read_text())
    fixture = json.loads(Path(policy["fixture"]).read_text())
    dest = output / "camera_parity"
    dest.mkdir(exist_ok=False)
    rows = []
    for trial in c["trials"]:
        env, obs = make_environment(
            {**policy, "camera_observations": True}, dest / f"trial{trial}", fixture["episodes"][trial], trial
        )
        try:
            for step in range(3):
                captured = render_current_cameras(env, policy["render_resolution"])
                for key, pixels in captured.items():
                    difference = int(np.max(abs(pixels.astype(int) - obs[key].astype(int))))
                    rows.append(dict(trial=trial, step=step, camera=key, max_abs=difference))
                    if difference:
                        raise ValueError("on-demand RGB differs from the robosuite observable")
                obs, _, _, _ = env.step(np.zeros(7))
        finally:
            env.close()
    write_json(
        dest / "summary.json",
        dict(
            code=code_fingerprint(),
            input_sha256={policy["fixture"]: sha256(policy["fixture"])},
            cases=rows,
            passed=True,
        ),
    )
    print(json.dumps(dict(comparisons=len(rows), max_abs=max(r["max_abs"] for r in rows))))


if __name__ == "__main__":
    main()
