"""065 fixed training-only gripper opening response identification."""

import json
from pathlib import Path

import numpy as np
from analyze_wam_release_horizon import reference
from scipy.optimize import lsq_linear

PREPARED = Path(
    "outputs/wam_stability_20261021/clean_residual_training_20261023/recovery_20261023/preparation/prepared.json"
)
ROOT = Path("outputs/gripper_forward_identification_20261025")


def main():
    before = reference(PREPARED)
    prepared = json.loads(PREPARED.read_text())
    tasks = {i: [] for i in range(3)}
    seen = set()
    for anchor in prepared["training_anchors"]:
        path = anchor["trajectory_ref"]["path"]
        if path not in seen:
            seen.add(path)
            tasks[anchor["source_identity"]["task_index"]].append(anchor["trajectory_ref"])
    if any(len(rows) != 10 for rows in tasks.values()):
        raise ValueError("Requires original30 training trajectories")
    datasets = []
    for task, refs in tasks.items():
        for index, ref in enumerate(refs):
            path = Path(ref["path"])
            if reference(path) != ref:
                raise ValueError("Training input changed")
            with np.load(path, allow_pickle=False) as data:
                q = data["opening"].copy()
                offsets = data["gripper_current_action_offsets"]
                flat = data["gripper_current_action_flat"]
                motor = []
                for k in range(1, len(q)):
                    cache = flat[offsets[k] : offsets[k + 1]]
                    if cache.shape != (2,):
                        raise ValueError("Actual post-command motor cache must have two values")
                    motor.append(float(np.clip((0.039 + 0.02 * (cache[0] - cache[1])) / 0.079, 0, 1)))
            m = np.array(motor)
            steps = np.arange(1, len(q) - 1)
            x = np.column_stack([m[1:] - q[1:-1], q[1:-1] - q[:-2]])
            y = q[2:] - q[1:-1]
            active = (m[1:] >= m[:-1] - 1e-12) & (m[1:] > q[1:-1] + 0.02)
            datasets.append(
                dict(task=task, split="fit" if index < 8 else "holdout", ref=ref, steps=steps, x=x, y=y, active=active)
            )
    x = np.concatenate([d["x"][d["active"]] for d in datasets if d["split"] == "fit"])
    y = np.concatenate([d["y"][d["active"]] for d in datasets if d["split"] == "fit"])
    fit = lsq_linear(x, y, bounds=(0, 1), tol=1e-12)
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError("Forward fit failed")
    summaries = {}
    for split in ("fit", "holdout"):
        for mode in ("opening_loaded", "all"):
            errors, baseline = [], []
            for d in datasets:
                if d["split"] != split:
                    continue
                mask = d["active"] if mode == "opening_loaded" else np.ones(len(d["y"]), dtype=bool)
                errors.extend((d["x"] @ fit.x - d["y"])[mask])
                baseline.extend(d["y"][mask])
            summaries[f"{split}_{mode}"] = dict(
                rows=len(errors),
                rmse=float(np.sqrt(np.mean(np.square(errors)))),
                hold_current_rmse=float(np.sqrt(np.mean(np.square(baseline)))),
            )
    if before != reference(PREPARED) or any(reference(Path(d["ref"]["path"])) != d["ref"] for d in datasets):
        raise RuntimeError("Consumed inputs changed")
    ROOT.mkdir(exist_ok=False)
    result = dict(
        seed=42,
        coefficients=fit.x.tolist(),
        coefficient_names=["beta_motor_gap", "gamma_previous_velocity"],
        prepared_ref=before,
        script_ref=reference(Path(__file__)),
        summaries=summaries,
        trajectories=[
            dict(task=d["task"], split=d["split"], ref=d["ref"], loaded_steps=d["steps"][d["active"]].tolist())
            for d in datasets
        ],
        model_calls=0,
        physics_steps=0,
        fresh_cases=0,
        closed_loop_qualified=False,
    )
    (ROOT / "fit_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(coefficients=fit.x.tolist(), summaries=summaries)))


if __name__ == "__main__":
    main()
