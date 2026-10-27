"""Independent scalar079 validation from raw qvel, without producer model imports."""

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path("outputs/velocity_residual_response_20261025")
COLLECTION = Path("outputs/native_gripper_response_interventions_20261025_v3")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def design(values, parameters):
    z = [
        (value - mean) / scale
        for value, mean, scale in zip(values, parameters["mean"], parameters["scale"], strict=True)
    ]
    return z + [z[i] * z[j] for i in range(4) for j in range(i, 4)] + [1.0]


def main():
    path = ROOT / "validation_result.json"
    validation = json.loads(path.read_text())
    params = json.loads((ROOT / "parameters.json").read_text())
    oldpath = Path("outputs/all_phase_gripper_forward_20261025/parameters.json")
    old = json.loads(oldpath.read_text())
    hashes = dict(validation["source_sha256"])
    hashes[str(path)] = digest(path)
    hashes[__file__] = digest(__file__)
    assert all(digest(p) == s for p, s in hashes.items())
    roster = json.loads(Path("outputs/gripper_forward_identification_20261025/fit_result.json").read_text())[
        "trajectories"
    ]
    expected = [d["ref"] for d in roster if d["split"] == "holdout"]
    for case in (2, 6, 10):
        expected.append(
            json.loads((COLLECTION / f"case{case:02d}" / "collection_result.json").read_text())["arrays_ref"]
        )
    assert len(validation["results"]) == 9 and [d["ref"] for d in validation["results"]] == expected
    audit = json.loads(Path("outputs/gripper_velocity_state_audit_20261025/audit.json").read_text())
    case_refs = {d["ref"]["path"]: d["case_ref"] for d in audit["cases"]}
    eligible = True
    cases = []
    for reported in validation["results"]:
        source = Path(reported["ref"]["path"])
        if reported["kind"] == "original_internal_holdout":
            ref = case_refs[str(source)]
            assert digest(ref["path"]) == ref["sha256"]
            hashes[ref["path"]] = ref["sha256"]
            xml = json.loads(Path(ref["path"]).read_text())["record"]["model_xml"]
        else:
            model = COLLECTION / f"case{reported['case']:02d}" / "model.xml"
            hashes[str(model)] = digest(model)
            xml = model.read_text()
        nq = nv = 0
        indices = {}
        for node in ET.fromstring(xml).find("worldbody").iter():
            if node.tag not in ("joint", "freejoint"):
                continue
            kind = "free" if node.tag == "freejoint" else node.get("type", "hinge")
            if node.get("name") in ("gripper0_finger_joint1", "gripper0_finger_joint2"):
                indices[node.get("name")] = (nq, nv)
            a, b = {"free": (7, 6), "ball": (4, 3), "slide": (1, 1), "hinge": (1, 1)}[kind]
            nq += a
            nv += b
        with np.load(source, allow_pickle=False) as data:
            q = data["opening"].copy()
            u = data["sent_commands"][:, -1].copy()
            off = data["gripper_current_action_offsets"]
            flat = data["gripper_current_action_flat"]
            m = [float(flat[off[k]]) for k in range(len(q))]
            state = data["sim_states"]
            assert state.shape == (len(q), 1 + nq + nv)
            pos = [1 + indices[f"gripper0_finger_joint{i}"][0] for i in (1, 2)]
            vel = [1 + nq + indices[f"gripper0_finger_joint{i}"][1] for i in (1, 2)]
            np.testing.assert_array_equal(state[:, pos], data["gripper_qpos"])
            v = 0.05 * ((state[:, vel[0]] - state[:, vel[1]]) / 0.079)
            if "case" in reported:
                np.testing.assert_array_equal(data["gripper_qvel"], state[:, vel])
        start = reported["start"]
        stop = reported["stop"]
        assert stop == (len(u) if "case" not in reported else start + 24)
        metrics = {}
        for method in ("candidate079", "hold_current", "constant_velocity", "frozen075"):
            one_q = []
            roll_q = []
            one_v = []
            roll_v = []
            maxv = 0.0
            failures = 0
            for k in range(start, stop):
                current = float(q[k])
                speed = float(v[k])
                motor = m[k]
                dq = 0.0 if k == 0 else float(q[k] - q[k - 1])
                for j in range(k, min(stop, k + 8)):
                    nextmotor = motor
                    if method in ("candidate079", "frozen075"):
                        increment = -0.01 if u[j] > 0 else (0.01 if u[j] < 0 else 0.0)
                        for _ in range(25):
                            nextmotor = max(-1.0, min(1.0, nextmotor + increment))
                    if method == "candidate079":
                        x = design([current, motor, nextmotor, speed], params)
                        residual = [
                            math.fsum(x[i] * params["coefficients"][i][column] for i in range(15)) for column in (0, 1)
                        ]
                        nextq = current + speed + residual[0]
                        nextv = speed + residual[1]
                    elif method == "frozen075":
                        x = design([current, motor, nextmotor, dq], old)
                        nextq = current + math.fsum(a * b for a, b in zip(x, old["coefficients"], strict=True))
                        nextv = speed
                    elif method == "constant_velocity":
                        nextq = current + speed
                        nextv = speed
                    else:
                        nextq = current
                        nextv = speed
                    if not math.isfinite(nextq) or not math.isfinite(nextv):
                        failures += 1
                        break
                    nextq = max(0.0, min(1.0, nextq))
                    roll_q.append(nextq - q[j + 1])
                    if j == k:
                        one_q.append(nextq - q[j + 1])
                    if method != "frozen075":
                        roll_v.append(nextv - v[j + 1])
                        maxv = max(maxv, abs(nextv))
                        if j == k:
                            one_v.append(nextv - v[j + 1])
                    dq = nextq - current
                    current, speed, motor = nextq, nextv, nextmotor

            def rmse(values):
                return math.sqrt(math.fsum(float(v) ** 2 for v in values) / len(values)) if values else None

            got = dict(
                one_step_q_rmse=rmse(one_q),
                rollout_8_q_rmse=rmse(roll_q),
                one_step_V_rmse=rmse(one_v),
                rollout_8_V_rmse=rmse(roll_v),
                failed_starts=failures,
                evaluated_rollout_endpoints=len(roll_q),
                max_abs_predicted_period_velocity=maxv if method != "frozen075" else None,
            )
            for key, value in got.items():
                saved = reported["metrics"][method][key]
                if value is None:
                    assert saved is None
                else:
                    np.testing.assert_allclose(value, saved, rtol=0, atol=1e-10, err_msg=f"{source}:{method}:{key}")
            metrics[method] = got
        candidate, hold = metrics["candidate079"], metrics["hold_current"]
        passed = candidate["failed_starts"] == 0 and all(
            candidate[k] <= hold[k] for k in ("one_step_q_rmse", "rollout_8_q_rmse")
        )
        assert bool(passed) == reported["performance_eligible"]
        eligible &= passed
        cases.append(dict(ref=reported["ref"], performance_eligible=bool(passed), metrics=metrics))
    assert bool(eligible) == validation["offline_performance_eligible"]
    assert all(digest(p) == s for p, s in hashes.items())
    output = dict(
        numerically_verified=True,
        offline_performance_eligible=bool(eligible),
        verified_cases=9,
        cases=cases,
        source_sha256=hashes,
        physical_qualification=False,
        new_fits=0,
        model_calls=0,
        physics_steps=0,
        fresh_cases=0,
    )
    payload = json.dumps(output, indent=2, allow_nan=False) + "\n"
    with (ROOT / "independent_validation_verified.json").open("x") as stream:
        stream.write(payload)
    print("independent scalar/raw-velocity validation:9 cases verified;eligible", eligible)


if __name__ == "__main__":
    main()
