"""197 full native-history causality parity and training-only normalization."""

import numpy as np
from arm_wrench_data import attach_current_wrench
from audit_wam_position import Evidence

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish, sha
from s2a.finger_contact_history import contact_history


def independently_reconstruct(data):
    count = len(data["x_finger_qpos"])
    result = np.zeros((count, 39))
    available = data["x_previous_command_available"]
    for i in range(count):
        for lag in (1, 2, 3):
            j = i - lag
            if j < 0 or not all(bool(v) for v in available[j + 1 : i + 1]):
                continue
            result[i, 13 * (lag - 1) : 13 * lag] = [
                *[float(data["x_finger_qpos"][i, k]) - float(data["x_finger_qpos"][j, k]) for k in range(2)],
                *[float(data["x_finger_qvel"][j, k]) for k in range(2)],
                *[
                    float(data["x_current_wrench_world"][i, k]) - float(data["x_current_wrench_world"][j, k])
                    for k in range(6)
                ],
                *[float(data["x_motor_before"][i, k]) - float(data["x_motor_before"][j, k]) for k in range(2)],
                1.0,
            ]
    return result


def main():
    root = REPO / "outputs/measured_finger_history_20261025"
    if root.exists():
        raise FileExistsError(root)
    e = Evidence()
    e.read(__file__)
    e.read("src/s2a/finger_contact_history.py")
    e.read("plans/v0.0.1/196_measured_finger_contact_history_20261025.md")
    config = e.json("configs/left_table_forward_20261025.json")
    source = e.json(config["data_result_ref"]["path"])
    fit = []
    all_rows = []
    identities = []
    for entry in source["trajectories"]:
        ref = entry["cache_ref"]
        e.check(ref["path"], ref["sha256"])
        with np.load(ref["path"], allow_pickle=False) as a:
            data = {k: a[k].copy() for k in a.files}
        data = attach_current_wrench(data, entry, config, e)
        np.testing.assert_allclose(np.diff(data["time"]), 0.05, atol=1e-10, rtol=0)
        actual = contact_history(data)
        independent = independently_reconstruct(data)
        np.testing.assert_allclose(actual, independent, atol=1e-12, rtol=0)
        for length in sorted(set([1, min(2, len(actual)), min(5, len(actual)), len(actual)])):
            prefix = {
                k: np.asarray(data[k])[:length]
                for k in (
                    "x_finger_qpos",
                    "x_finger_qvel",
                    "x_motor_before",
                    "x_current_wrench_world",
                    "x_previous_command_available",
                )
            }
            np.testing.assert_array_equal(contact_history(prefix), actual[:length])
        if entry["split"] == "fit":
            fit.append(actual)
        all_rows.append(actual)
        identities.append(dict(native_index=entry["index"], split=entry["split"], rows=len(actual)))
    fit = np.vstack(fit)
    history = np.vstack(all_rows)
    assert fit.shape == (4865, 39) and history.shape == (6180, 39)
    mean = fit.mean(axis=0)
    scale = np.maximum(fit.std(axis=0), 1e-8)
    e.recheck()
    root.mkdir()
    path = root / "history.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, history=history, fit_mean=mean, fit_scale=scale)
    publish(
        root / "verified.json",
        dict(
            complete=True,
            independently_reconstructed=True,
            prefix_invariance_verified=True,
            trajectories=30,
            rows=6180,
            history_dimension=39,
            fit_statistic_rows=4865,
            identities=identities,
            history_ref=dict(path=str(path), sha256=sha(path)),
            source_sha256=e.hashes,
            physics_steps=0,
            fits=0,
            control_qualified=False,
            limitation="Causally available measured history only; no claim of observability sufficiency, improved local response or control",
        ),
    )
    print("197 6180history rows independently verified;prefix causal;39dims;fit-only statistics")


if __name__ == "__main__":
    main()
