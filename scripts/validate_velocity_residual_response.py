"""079 frozen one/eight-step response validation; no fitting or physical execution."""

import hashlib
import json
from pathlib import Path

import numpy as np

from s2a.velocity_residual_response import VelocityResidualResponse, motor_after_native_period

ROOT = Path("outputs/velocity_residual_response_20261025")
COLLECTION = Path("outputs/native_gripper_response_interventions_20261025_v3")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rms(values):
    if not len(values):
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        value = float(np.sqrt(np.mean(np.square(values))))
    return value if np.isfinite(value) else None


def old_predict(params, q, m, dq, u):
    nextm = motor_after_native_period(m, u)
    z = (np.array([q, m, nextm, dq]) - params["mean"]) / params["scale"]
    with np.errstate(over="ignore", invalid="ignore"):
        features = np.r_[z, [z[i] * z[j] for i in range(4) for j in range(i, 4)], 1.0]
        predicted = q + features @ params["coefficients"]
    if not np.isfinite(predicted):
        raise ValueError("075 nonfinite response")
    return float(np.clip(predicted, 0, 1)), nextm


def main():
    paths = [
        ROOT / "fit_result.json",
        ROOT / "parameters.json",
        ROOT / "independent_fit_verified.json",
        Path("outputs/gripper_forward_identification_20261025/fit_result.json"),
        Path("outputs/gripper_velocity_state_audit_20261025/audit.json"),
        Path("outputs/gripper_velocity_state_audit_20261025/velocity_traces.npz"),
        Path("outputs/all_phase_gripper_forward_20261025/parameters.json"),
        Path("outputs/all_phase_gripper_forward_20261025/fit_result.json"),
        Path(__file__),
        Path("src/s2a/velocity_residual_response.py"),
    ]
    hashes = {str(p): digest(p) for p in paths}
    fit = json.loads(paths[0].read_text())
    audit = json.loads(paths[2].read_text())
    assert audit["numerically_verified"]
    assert digest(paths[1]) == fit["parameters_sha256"]
    for p, s in audit["source_sha256"].items():
        assert digest(p) == s
    oldfit = json.loads(paths[7].read_text())
    assert digest(paths[6]) == oldfit["parameters_sha256"]
    old = json.loads(paths[6].read_text())
    model = VelocityResidualResponse(json.loads(paths[1].read_text()))
    roster = [d for d in json.loads(paths[3].read_text())["trajectories"] if d["split"] == "holdout"]
    assert len(roster) == 6
    velocity_audit = json.loads(paths[4].read_text())
    velocity_index = {d["ref"]["path"]: i for i, d in enumerate(velocity_audit["cases"])}
    specs = [dict(kind="original_internal_holdout", ref=d["ref"], task=d["task"], start=0) for d in roster]
    for case in (2, 6, 10):
        p = COLLECTION / f"case{case:02d}" / "collection_result.json"
        hashes[str(p)] = digest(p)
        d = json.loads(p.read_text())
        assert d["profile"]["branch"] == "retain"
        specs.append(
            dict(
                kind="new_retain_response_validation",
                ref=d["arrays_ref"],
                task=d["profile"]["task"],
                case=case,
                start=d["profile"]["anchor_cycle"],
            )
        )
    results = []
    with np.load(paths[5], allow_pickle=False) as velocity:
        for spec in specs:
            ref = spec["ref"]
            assert digest(ref["path"]) == ref["sha256"]
            hashes[ref["path"]] = ref["sha256"]
            with np.load(ref["path"], allow_pickle=False) as data:
                q = data["opening"].copy()
                u = data["sent_commands"][:, -1].copy()
                off = data["gripper_current_action_offsets"]
                flat = data["gripper_current_action_flat"]
                m = np.array([flat[off[k]] for k in range(len(q))])
                if spec["kind"] == "original_internal_holdout":
                    v = 0.05 * velocity[f"case{velocity_index[ref['path']]:02d}_velocity"]
                else:
                    v = 0.05 * (data["gripper_qvel"][:, 0] - data["gripper_qvel"][:, 1]) / 0.079
            start = spec["start"]
            stop = len(u) if spec["kind"] == "original_internal_holdout" else start + 24
            methods = {
                name: dict(
                    one_q=[], roll_q=[], one_v=[], roll_v=[], failed_starts=0, max_abs_predicted_period_velocity=0.0
                )
                for name in ("candidate079", "hold_current", "constant_velocity", "frozen075")
            }
            for k in range(start, stop):
                for name, metric in methods.items():
                    current = float(q[k])
                    vel = float(v[k])
                    motor = float(m[k])
                    dq = 0.0 if k == 0 else float(q[k] - q[k - 1])
                    for j in range(k, min(k + 8, stop)):
                        try:
                            if name == "candidate079":
                                predicted = model.predict(current, motor, vel, float(u[j]))
                                nq, nv, nm = (
                                    predicted.next_opening,
                                    predicted.next_period_velocity,
                                    predicted.next_motor_target,
                                )
                            elif name == "hold_current":
                                nq, nv, nm = current, vel, motor
                            elif name == "constant_velocity":
                                nq, nv, nm = float(np.clip(current + vel, 0, 1)), vel, motor
                            else:
                                nq, nm = old_predict(old, current, motor, dq, float(u[j]))
                                nv = vel
                        except ValueError:
                            metric["failed_starts"] += 1
                            break
                        error = nq - q[j + 1]
                        metric["roll_q"].append(error)
                        if j == k:
                            metric["one_q"].append(error)
                        if name != "frozen075":
                            metric["roll_v"].append(nv - v[j + 1])
                            metric["max_abs_predicted_period_velocity"] = max(
                                metric["max_abs_predicted_period_velocity"], abs(nv)
                            )
                            if j == k:
                                metric["one_v"].append(nv - v[j + 1])
                        dq = nq - current
                        current, vel, motor = nq, nv, nm
            metrics = {}
            for name, values in methods.items():
                metrics[name] = dict(
                    one_step_q_rmse=rms(values["one_q"]),
                    rollout_8_q_rmse=rms(values["roll_q"]),
                    one_step_V_rmse=rms(values["one_v"]),
                    rollout_8_V_rmse=rms(values["roll_v"]),
                    failed_starts=values["failed_starts"],
                    evaluated_rollout_endpoints=len(values["roll_q"]),
                    max_abs_predicted_period_velocity=values["max_abs_predicted_period_velocity"]
                    if name != "frozen075"
                    else None,
                )
            candidate, hold = metrics["candidate079"], metrics["hold_current"]
            eligible = candidate["failed_starts"] == 0 and all(
                candidate[key] is not None and candidate[key] <= hold[key]
                for key in ("one_step_q_rmse", "rollout_8_q_rmse")
            )
            results.append(dict(**spec, stop=stop, cycles=stop - start, metrics=metrics, performance_eligible=eligible))
    assert all(digest(p) == s for p, s in hashes.items())
    output = dict(
        complete=True,
        offline_performance_eligible=all(d["performance_eligible"] for d in results),
        physical_qualification=False,
        results=results,
        source_sha256=hashes,
        new_fits=0,
        model_calls=0,
        physics_steps=0,
        fresh_cases=0,
    )
    payload = json.dumps(output, indent=2, allow_nan=False) + "\n"
    with (ROOT / "validation_result.json").open("x") as stream:
        stream.write(payload)
    print("offline eligible", output["offline_performance_eligible"])
    for d in results:
        c = d["metrics"]["candidate079"]
        h = d["metrics"]["hold_current"]
        print(
            d.get("case", Path(d["ref"]["path"]).parent.name),
            d["performance_eligible"],
            "one",
            c["one_step_q_rmse"],
            h["one_step_q_rmse"],
            "roll8",
            c["rollout_8_q_rmse"],
            h["rollout_8_q_rmse"],
            "failures",
            c["failed_starts"],
        )


if __name__ == "__main__":
    main()
