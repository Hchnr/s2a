"""Offline motor-cache recurrence with logged measured q; not a physical rollout."""

import hashlib
import json
from pathlib import Path

import numpy as np
from residual_motor_scalar_replay import replay


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path("outputs/residual_motor_recurrence_diagnosis_20261025")
    roster_path = Path("outputs/state_conditioned_gripper_inverse_20261025/fit_result.json")
    param_path = Path("outputs/residual_motor_prior_20261025/parameters.json")
    paths = [roster_path, param_path, Path(__file__), Path("scripts/residual_motor_scalar_replay.py")]
    roster = json.loads(roster_path.read_text())["trajectories"]
    paths += [Path(row["ref"]["path"]) for row in roster]
    hashes = {str(p): digest(p) for p in paths}
    params = json.loads(param_path.read_text())
    cases = []
    for entry in roster:
        ref = entry["ref"]
        assert digest(ref["path"]) == ref["sha256"]
        with np.load(ref["path"], allow_pickle=False) as data:
            q = data["opening"].copy()
            offsets = data["gripper_current_action_offsets"]
            flat = data["gripper_current_action_flat"]
            motors = np.array([flat[offsets[k]] for k in range(len(q))])
        recurrent = float(motors[0])
        teacher_targets, recurrent_targets, commands = [], [], []
        for k in range(len(q) - 1):
            desired = q[k + 1 : k + 41].tolist()
            prior = None if k == 0 else float(q[k - 1])
            teacher = replay(params, float(q[k]), float(motors[k]), prior, desired)
            predicted = replay(params, float(q[k]), recurrent, prior, desired)
            teacher_targets.append(teacher["selected_motor_target"])
            commands.append(predicted["command"])
            for _ in range(25):
                recurrent = max(-1.0, min(1.0, recurrent - 0.01 * predicted["command"]))
            recurrent_targets.append(recurrent)
        error = np.asarray(recurrent_targets) - motors[1:]
        cases.append(
            dict(
                ref=ref,
                split=entry["split"],
                task=entry["task"],
                cycles=len(commands),
                teacher_motor_rmse=float(np.sqrt(np.mean((np.asarray(teacher_targets) - motors[1:]) ** 2))),
                recurrent_motor_rmse=float(np.sqrt(np.mean(error**2))),
                large_motor_deviation_cycles=int(np.sum(np.abs(error) > 0.25 + 1e-10)),
                commands=commands,
                recurrent_motor_targets=recurrent_targets,
            )
        )
    assert all(digest(p) == sha for p, sha in hashes.items())
    root.mkdir(exist_ok=False)
    output = dict(
        complete=True,
        evidence_level="logged q replay with synthetic commanded motor recurrence; no physical q evolution",
        measured_opening_is_logged_not_predicted=True,
        source_sha256=hashes,
        cases=cases,
        new_fits=0,
        model_calls=0,
        physics_steps=0,
        fresh_cases=0,
    )
    (root / "diagnosis.json").write_text(json.dumps(output, indent=2) + "\n")
    for d in cases:
        if d["split"] == "holdout":
            print(
                Path(d["ref"]["path"]).parent.name,
                "teacher",
                round(d["teacher_motor_rmse"], 5),
                "recurrent",
                round(d["recurrent_motor_rmse"], 5),
                "large_cycles",
                d["large_motor_deviation_cycles"],
            )


if __name__ == "__main__":
    main()
