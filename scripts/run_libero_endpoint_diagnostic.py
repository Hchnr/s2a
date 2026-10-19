"""Quasistatic replay of frozen endpoints; diagnostic, not a live WAM policy.

The same fixed OSC may take up to twelve physical ticks per generated target.
This deliberately changes the reference pace to assess tracking limitations.
"""

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from s2a.artifacts import code_fingerprint, load_config, sha256, write_json
from s2a.geometry import pose_error
from s2a.gripper import OpeningIntent, opening_position_target
from s2a.libero_audit import contact_counts
from s2a.libero_simulation import (
    aperture,
    controller_contract,
    current_observation,
    make_environment,
    measured_twist,
    source_pose,
)
from s2a.osc import OSCInterface, OSCStateToAction


def run(c, out, source, trial):
    root = out / "sync" / source / f"trial{trial}"
    original = json.loads((root / "result.json").read_text())
    if original["error"] is not None or original["steps"] != original["expected_steps"]:
        raise ValueError("source trajectory is incomplete")
    source_file = root / "steps.jsonl"
    targets = [json.loads(line) for line in source_file.read_text().splitlines()][: c["oracle_steps"]]
    if len(targets) != c["oracle_steps"]:
        raise ValueError("insufficient frozen targets")
    dest = out / "quasistatic" / source / f"trial{trial}"
    dest.mkdir(parents=True, exist_ok=False)
    p = c["endpoint_diagnostic"]
    fixture = json.loads(Path(c["fixture"]).read_text())
    write_json(dest / "references.json", [dict(pose=r["desired_source"], opening=r["opening_target"]) for r in targets])
    write_json(
        dest / "manifest.json",
        dict(
            config=c,
            source=source,
            trial=trial,
            code=code_fingerprint(),
            inputs_sha256={str(source_file): sha256(source_file), c["fixture"]: sha256(c["fixture"])},
            evidence="Exogenous replay of the first 300 frozen targets; privileged time dilation, not a live-image WAM policy or the 20Hz task benchmark.",
        ),
    )
    env, _ = make_environment(c, dest, fixture["episodes"][trial], trial)
    rows, endpoints, error = [], [], None
    started = time.monotonic()
    try:
        obs = current_observation(env)
        _, contract = controller_contract(env)
        converter = OSCStateToAction(
            OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max"))),
            contract["kp"],
            contract["kd"],
            1 / c["control_fps"],
            compensate=True,
            inverse_weight=c["inverse_weight"],
            max_command_rate=c["max_command_rate"],
        )
        env.robots[0].gripper.current_action = opening_position_target(aperture(obs))
        intent = OpeningIntent(aperture(obs), band=c["opening_hysteresis"])
        with (dest / "steps.jsonl").open("w") as stream:
            for index, target in enumerate(targets):
                desired = np.asarray(target["desired_source"])
                grip = intent.update(target["opening_target"])
                for substep in range(p["max_substeps"]):
                    before = source_pose(obs)
                    converted = converter.convert(before, desired, measured_twist(env))
                    command = np.r_[converted["command"], grip]
                    simulation_time = float(env.sim.data.time)
                    env.step(command)
                    obs = current_observation(env)
                    after = source_pose(obs)
                    e = pose_error(desired, after)
                    reached = (
                        np.linalg.norm(e[:3]) <= p["position_tolerance_m"]
                        and np.linalg.norm(e[3:]) <= p["rotation_tolerance_rad"]
                    )
                    row = dict(
                        step=len(rows),
                        reference_index=index,
                        substep=substep,
                        measured_source=before,
                        desired_source=desired,
                        after_source=after,
                        simulation_time=simulation_time,
                        actual_dt=float(env.sim.data.time) - simulation_time,
                        command=command,
                        opening_target=target["opening_target"],
                        opening_after=aperture(obs),
                        error=e,
                        reached=bool(reached),
                        success=bool(env.check_success()),
                        contacts=contact_counts(env),
                        bowl_position=env.sim.data.get_body_xpos("akita_black_bowl_1_main").copy(),
                    )
                    rows.append(row)
                    stream.write(json.dumps(row, default=lambda v: v.tolist(), allow_nan=False) + "\n")
                    if reached:
                        break
                endpoints.append(row)
                if index % 20 == 19:
                    stream.flush()
                    print(
                        json.dumps(dict(source=source, trial=trial, targets=index + 1, physics_steps=len(rows))),
                        flush=True,
                    )
    except Exception as exc:
        error = repr(exc)
    finally:
        env.close()
    result = dict(
        source=source,
        trial=trial,
        error=error,
        references=len(endpoints),
        physical_steps=len(rows),
        success=any(r["success"] for r in rows),
        terminal_success=bool(rows and rows[-1]["success"]),
        reached_endpoints=sum(r["reached"] for r in endpoints),
        wall_elapsed_s=time.monotonic() - started,
    )
    if endpoints:
        e = np.asarray([r["error"] for r in endpoints])
        result.update(
            endpoint_position_rmse_m=float(np.sqrt(np.mean(np.sum(e[:, :3] ** 2, axis=1)))),
            endpoint_rotation_rmse_rad=float(np.sqrt(np.mean(np.sum(e[:, 3:] ** 2, axis=1)))),
            finger_contact_steps=sum(r["contacts"]["finger_bowl"] > 0 for r in rows),
        )
    write_json(dest / "result.json", result)
    if error:
        raise RuntimeError(error)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/libero_wam_effect_20261019.json")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source", choices=["oracle_response", "wam_affine"])
    parser.add_argument("--trial", type=int)
    args = parser.parse_args()
    c, out = load_config(args.config)
    if args.seed != c["seed"]:
        raise ValueError("seed differs")
    if args.source:
        print(json.dumps(run(c, out, args.source, args.trial)), flush=True)
        return

    def execute_trial(trial):
        results = []
        for source in ("oracle_response", "wam_affine"):
            path = out / f"quasistatic_{source}_trial{trial}.log"
            with path.open("x") as stream:
                code = subprocess.call(
                    [
                        str(Path(c["simulation_python"]).absolute()),
                        __file__,
                        "--config",
                        args.config,
                        "--seed",
                        str(args.seed),
                        "--source",
                        source,
                        "--trial",
                        str(trial),
                    ],
                    env={
                        **os.environ,
                        "LD_LIBRARY_PATH": "",
                        "OPENBLAS_NUM_THREADS": "1",
                        "OMP_NUM_THREADS": "2",
                        "LP_NUM_THREADS": "2",
                    },
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            row = dict(source=source, trial=trial, exit_code=code)
            print(json.dumps(row), flush=True)
            results.append(row)
        return results

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = [r for group in executor.map(execute_trial, c["trials"]) for r in group]
    write_json(out / "matrix_quasistatic.json", dict(cases=results, complete=all(r["exit_code"] == 0 for r in results)))
    if any(r["exit_code"] for r in results):
        raise SystemExit("quasistatic diagnostic failed; evidence retained")


if __name__ == "__main__":
    main()
