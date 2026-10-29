"""Read-only translational nominal-PD residual decomposition on verified081 logs."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm

ROOT = Path("outputs/loaded_velocity_control_20261025")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    config_path = Path("configs/loaded_velocity_control_20261025.json")
    config = json.loads(config_path.read_text())
    hashes = {str(config_path): digest(config_path), __file__: digest(__file__)}
    cases = []
    for i, profile in enumerate(config["profiles"]):
        exit_path = ROOT / "verification" / f"case{i:02d}" / "exit.json"
        if (
            not exit_path.exists()
            or json.loads(exit_path.read_text())["returncode"] != 0
            or profile["format"] != "native_endpoints"
        ):
            continue
        folder = ROOT / f"case{i:02d}"
        verified_path = folder / "independent_verified.json"
        verified = json.loads(verified_path.read_text())
        assert verified["mechanical_labels_images_passed"]
        log = next(folder.glob("simulation/*/*/steps.jsonl"))
        controller_path = log.parent / "controller.json"
        controller = json.loads(controller_path.read_text())
        c = controller["contract"]
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        source = Path(profile["trajectory_ref"]["path"])
        assert digest(source) == profile["trajectory_ref"]["sha256"]
        for p in (exit_path, verified_path, log, controller_path, source):
            hashes[str(p)] = digest(p)
        dt = 1 / c["control_freq"]
        assert dt == 0.05
        response = []
        momentum = []
        for kp, kd in zip(c["kp"][:3], c["kd"][:3], strict=True):
            evolution = expm(np.array([[0.0, 1.0], [-kp, -kd]]) * dt)
            response.append(1 - evolution[0, 0])
            momentum.append(evolution[0, 1])
        before = np.array([r["measured_source"] for r in rows])
        after = np.array([r["after_source"] for r in rows])
        desired = np.array([r["desired_source"] for r in rows])
        command = np.array([r["command"][:6] for r in rows])
        velocity = np.array([r["measured_twist"][:3] for r in rows])
        lower, upper = np.array(c["input_min"]), np.array(c["input_max"])
        outlo, outhi = np.array(c["output_min"]), np.array(c["output_max"])
        delta = outlo + (command - lower) * (outhi - outlo) / (upper - lower)
        predicted = np.array(response) * delta[:, :3] + np.array(momentum) * velocity
        actual = after[:, :3, 3] - before[:, :3, 3]
        wanted = desired[:, :3, 3] - before[:, :3, 3]
        blend = wanted - predicted
        model = predicted - actual
        tracking = wanted - actual
        np.testing.assert_allclose(blend + model, tracking, rtol=0, atol=1e-12)
        with np.load(source, allow_pickle=False) as data:
            native_close = int(np.flatnonzero(data["sent_commands"][:, -1] > 0)[0])
        summaries = {}
        for label, sl in [
            ("all", slice(None)),
            ("native_closing_window", slice(max(0, native_close - 5), min(len(rows), native_close + 15))),
        ]:
            b, m, t = blend[sl], model[sl], tracking[sl]
            selected = rows[sl]
            summaries[label] = dict(
                rows=len(t),
                source_position_tracking_rmse_m=float(np.sqrt(np.mean(np.sum(t * t, axis=1)))),
                nominal_model_residual_rmse_m=float(np.sqrt(np.mean(np.sum(m * m, axis=1)))),
                nominal_target_deficit_rmse_m=float(np.sqrt(np.mean(np.sum(b * b, axis=1)))),
                cross_energy_m2=float(2 * np.mean(np.sum(b * m, axis=1))),
                tracking_energy_m2=float(np.mean(np.sum(t * t, axis=1))),
                rate_limited_cycles=sum(r["rate_limited"] for r in selected),
                saturated_cycles=sum(r["saturated"] for r in selected),
            )
            assert (
                abs(
                    summaries[label]["tracking_energy_m2"]
                    - (
                        np.mean(np.sum(b * b, axis=1))
                        + np.mean(np.sum(m * m, axis=1))
                        + summaries[label]["cross_energy_m2"]
                    )
                )
                < 1e-12
            )
        cases.append(
            dict(
                case=i,
                stable=verified["control"]["outcome"]["stable"],
                native_first_close_cycle=native_close,
                response=response,
                momentum=momentum,
                summaries=summaries,
            )
        )
        w = summaries["native_closing_window"]
        print(
            i,
            "stable",
            verified["control"]["outcome"]["stable"],
            "closing tracking/model/target deficit mm",
            *[
                round(1000 * w[k], 3)
                for k in (
                    "source_position_tracking_rmse_m",
                    "nominal_model_residual_rmse_m",
                    "nominal_target_deficit_rmse_m",
                )
            ],
            "limited",
            w["rate_limited_cycles"],
        )
    assert all(digest(p) == sha for p, sha in hashes.items())
    out = Path("outputs/loaded_arm_response_audit_20261025")
    out.mkdir(exist_ok=True)
    result = dict(
        complete_verified_snapshot=True,
        evidence="algebraic observed source-position residual decomposition;nominal uncoupled PD is an assumption;no counterfactual execution",
        cases=cases,
        source_sha256=hashes,
        new_fits=0,
        physics_steps=0,
        model_calls=0,
        fresh_cases=0,
    )
    with (out / f"verified_prefix_{len(cases):02d}.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
