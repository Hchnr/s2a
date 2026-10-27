"""096 partial holdout prequential check; full local-response qualification pending."""

import json
from pathlib import Path

import numpy as np
from arm_history_independent import augment, prediction
from audit_wam_position import Evidence
from scipy.spatial.transform import Rotation
from validate_history_conditioned_arm_forward import scores

from s2a.causal_local_response import CausalLocalResponse
from s2a.history_conditioned_arm_forward import nominal

ROOT = Path("outputs/causal_local_response_20261025")


def main():
    e = Evidence()
    for p in (
        __file__,
        "src/s2a/causal_local_response.py",
        "plans/v0.0.1/096_causal_local_response_20261025.md",
        "scripts/arm_history_independent.py",
    ):
        e.read(p)
    prior_root = Path("outputs/history_conditioned_arm_forward_20261025")
    closure = e.json(prior_root / "closure_verified.json")
    assert closure["complete"] and closure["numerically_verified"] and closure["owner_normal_wait_verified"]
    fit = e.json(prior_root / "fit_result.json")
    ref = fit["parameter_ref"]
    e.check(ref["path"], ref["sha256"])
    params = e.json(ref["path"])
    config = e.json("configs/history_conditioned_arm_forward_20261025.json")
    ref = config["data_result_ref"]
    e.check(ref["path"], ref["sha256"])
    entries = [r for r in e.json(ref["path"])["trajectories"] if r["split"] == "holdout"]
    assert len(entries) == 6
    rows = []
    solves = 0
    for entry in entries:
        ref = entry["cache_ref"]
        e.check(ref["path"], ref["sha256"])
        with np.load(ref["path"], allow_pickle=False) as cache:
            raw = {k: cache[k].copy() for k in cache.files}
        data = augment(raw, params)
        prior = prediction(data, params)
        base = nominal(data, params)
        observed = np.c_[data["y_motion"], data["y_rotation_increment"], 0.05 * data["y_twist"]]
        action = (data["action_goal_delta"][:, :3] - np.array(params["translation_mean"])) / np.array(
            params["translation_scale"]
        )
        # Row i-1 targets used for adaptation must equal observations already available at i.
        np.testing.assert_allclose(
            data["y_motion"][:-1], np.diff(data["x_site_pose"][:, :3, 3], axis=0), rtol=0, atol=1e-10
        )
        np.testing.assert_allclose(data["y_twist"][:-1], data["x_site_twist"][1:], rtol=0, atol=1e-10)
        rotation = data["x_site_pose"][:, :3, :3]
        increments = Rotation.from_matrix(rotation[1:] @ rotation[:-1].transpose(0, 2, 1)).as_rotvec()
        np.testing.assert_allclose(increments, data["y_rotation_increment"][:-1], rtol=0, atol=1e-10)
        model = CausalLocalResponse()
        independent_history = []
        values = []
        for i in range(len(prior)):
            if i:
                model.observe(action[i - 1], prior[i - 1], observed[i - 1])
                residual = observed[i - 1] - prior[i - 1]
                actual_r = Rotation.from_rotvec(observed[i - 1, 3:6]).as_matrix()
                predicted_r = Rotation.from_rotvec(prior[i - 1, 3:6]).as_matrix()
                residual[3:6] = Rotation.from_matrix(actual_r @ predicted_r.T).as_rotvec()
                independent_history.append((np.r_[1, action[i - 1]], residual))
            value = model.predict(action[i], prior[i])
            if independent_history:
                h = independent_history[-20:]
                n = len(h)
                x = np.stack([r[0] for r in h]) / np.sqrt(n)
                y = np.stack([r[1] for r in h]) / np.sqrt(n)
                beta = np.linalg.lstsq(np.vstack([x, np.eye(4)]), np.vstack([y, np.zeros((4, 12))]), rcond=None)[0]
                correction = np.r_[1, action[i]] @ beta
                independent = prior[i] + correction
                independent[3:6] = Rotation.from_matrix(
                    Rotation.from_rotvec(correction[3:6]).as_matrix() @ Rotation.from_rotvec(prior[i, 3:6]).as_matrix()
                ).as_rotvec()
                np.testing.assert_allclose(value, independent, rtol=0, atol=1e-10)
                solves += 1
            else:
                np.testing.assert_allclose(
                    Rotation.from_rotvec(value[3:6]).as_matrix(),
                    Rotation.from_rotvec(prior[i, 3:6]).as_matrix(),
                    atol=1e-14,
                )
            values.append(value)
        metrics = dict(
            adapted=scores(np.stack(values), data), frozen090=scores(prior, data), nominal=scores(base, data)
        )
        gates = {k: metrics["adapted"][k] <= metrics["nominal"][k] for k in metrics["nominal"]}
        rows.append(
            dict(index=entry["index"], rows=len(prior), online_updates=model.updates, metrics=metrics, gates=gates)
        )
        print(entry["index"], sum(gates.values()), "/4", flush=True)
    e.recheck()
    ROOT.mkdir(exist_ok=False)
    result = dict(
        complete=True,
        scope="partial six-holdout prequential check only; local response and full lifecycle qualification pending",
        holdout=rows,
        online_updates=sum(r["online_updates"] for r in rows),
        independent_numerical_solves=solves,
        new_global_fits=0,
        physics_steps=0,
        model_calls=0,
        fresh_cases=0,
        control_qualification=False,
        offline_forward_eligible=False,
        source_sha256=e.hashes,
    )
    with (ROOT / "holdout_prequential.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
