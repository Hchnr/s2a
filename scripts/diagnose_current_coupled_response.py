"""Compare nominal and corrected predictions on an independently verified actual case."""

import argparse
import json

import numpy as np
from audit_wam_position import Evidence
from scipy.spatial.transform import Rotation

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish


def rms(values):
    return float(np.sqrt(np.mean(np.sum(np.asarray(values) ** 2, axis=1))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-index", type=int, choices=range(18), required=True)
    args = parser.parse_args()
    e = Evidence()
    e.read(__file__)
    label = f"case{args.case_index:02d}"
    root = REPO / "outputs/current_coupled_control_20261026"
    directory = root / label
    verified = e.json(directory / "independent_verified.json")
    assert verified["complete"] and verified["mechanical_labels_images_passed"]
    for path, signature in verified["input_sha256"].items():
        e.check(path, signature)
    config = e.json("configs/current_coupled_control_registered_20261026.json")
    assert config["seed"] == 42
    profile = config["profiles"][args.case_index]
    array_ref = e.json(directory / "collection_result.json")["arrays_ref"]
    e.check(array_ref["path"], array_ref["sha256"])
    with np.load(array_ref["path"], allow_pickle=False) as archive:
        poses = archive["controller_site_pose"].copy()
    case = next(directory.glob("simulation/*/*"))
    rows = [json.loads(line) for line in e.read(case / "steps.jsonl").splitlines()]
    assert len(rows) == profile["physics_steps"] == len(poses) - 1
    errors = {
        kind: {"position": [], "rotation": []}
        for kind in ("original150_nominal", "coupled228_nominal", "coupled228_completed_innovation")
    }
    for index, row in enumerate(rows):
        actual_p = poses[index + 1, :3, 3] - poses[index, :3, 3]
        actual_r = Rotation.from_matrix(poses[index + 1, :3, :3] @ poses[index, :3, :3].T)
        predictions = {
            "original150_nominal": np.asarray(row["nominal_predicted_motion"]),
            "coupled228_nominal": np.asarray(row["coupled_response_history"]["commit"]["nominal_response"])[:6],
            "coupled228_completed_innovation": np.asarray(row["coupled_inverse"]["predicted_response"])[:6],
        }
        for kind, prediction in predictions.items():
            assert prediction.shape == (6,) and np.isfinite(prediction).all()
            errors[kind]["position"].append(actual_p - prediction[:3])
            errors[kind]["rotation"].append((actual_r * Rotation.from_rotvec(prediction[3:6]).inv()).as_rotvec())
        # An independent endpoint subtraction must equal the logged new history.
        residual = row["coupled_response_history"]["innovation_after"]
        np.testing.assert_allclose(errors["coupled228_nominal"]["position"][-1], residual[:3], atol=1e-9, rtol=0)
        np.testing.assert_allclose(errors["coupled228_nominal"]["rotation"][-1], residual[3:6], atol=1e-9, rtol=0)
    result = dict(
        complete=True,
        case=args.case_index,
        seed=42,
        rows=len(rows),
        source_sha256=e.hashes,
        actual_outcome=verified["control"]["outcome"],
        actual_tracking=verified["control"]["tracking"]["admitted"],
        actual_performance_qualified=verified["performance_qualified"],
        prediction_errors={
            kind: dict(position_rmse_m=rms(values["position"]), so3_rmse_rad=rms(values["rotation"]))
            for kind, values in errors.items()
        },
        interpretation="same actual executed commands/states;prediction diagnostics only;no alternate controller trajectory",
        new_physics_steps=0,
        model_calls=0,
        new_fits=0,
    )
    destination = REPO / "outputs/current_coupled_response_diagnosis_20261026"
    destination.mkdir(exist_ok=True)
    assert not (destination / f"{label}.json").exists()
    e.recheck()
    publish(destination / f"{label}.json", result)
    print(json.dumps({key: result[key] for key in ("case", "rows", "prediction_errors")}))


if __name__ == "__main__":
    main()
