"""238 quantify predictable angular tracking error on the closed actual18-case run."""

import json

import numpy as np
from audit_wam_position import Evidence
from scipy.spatial.transform import Rotation

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish


def rms(values):
    return float(np.sqrt(np.mean(np.sum(np.asarray(values) ** 2, axis=1))))


def main():
    root = REPO / "outputs/current_coupled_rotation_tracking_20261026"
    assert not root.exists()
    e = Evidence()
    e.read(__file__)
    closed = REPO / "outputs/current_coupled_control_20261026"
    closure = e.json(closed / "closure_verified.json")
    assert closure["complete"] and closure["verified_cases"] == 18 and closure["owner_normal_wait_verified"]
    for path, signature in closure["source_sha256"].items():
        e.check(path, signature)
    groups = (
        "actual_tracking",
        "nominal_prediction",
        "completed_innovation_prediction",
        "predicted_tracking",
        "corrected_predicted_tracking",
    )
    pooled = {key: [] for key in groups}
    cases = []
    for index in range(18):
        folder = closed / f"case{index:02d}"
        verified = e.json(folder / "independent_verified.json")
        assert verified["complete"] and verified["mechanical_labels_images_passed"]
        case = next(folder.glob("simulation/*/*"))
        rows = [json.loads(line) for line in e.read(case / "steps.jsonl").splitlines()]
        controller = e.json(case / "controller.json")
        contract, mapping = controller["contract"], np.asarray(controller["source_from_site"])
        ref = e.json(folder / "collection_result.json")["arrays_ref"]
        e.check(ref["path"], ref["sha256"])
        with np.load(ref["path"], allow_pickle=False) as archive:
            poses = archive["controller_site_pose"].copy()
        assert len(poses) == len(rows) + 1
        errors = {key: [] for key in groups}
        details, zero_steps, forcing_differences, stiffness_differences, command_angles = [], [], [], [], []
        for step, row in enumerate(rows):
            current, after = (Rotation.from_matrix(poses[i, :3, :3]) for i in (step, step + 1))
            desired = np.asarray(row["desired_source"]) @ mapping
            np.testing.assert_allclose(desired, row["desired_site"], atol=1e-9, rtol=0)
            target = Rotation.from_matrix(desired[:3, :3])
            nominal = np.asarray(row["coupled_response_history"]["commit"]["nominal_response"])
            innovation = np.asarray(row["coupled_response_history"]["innovation_before"])
            predicted = Rotation.from_rotvec(nominal[3:6]) * current
            corrected = Rotation.from_rotvec(innovation[3:6]) * predicted
            actual_innovation = (after * predicted.inv()).as_rotvec()
            np.testing.assert_allclose(
                actual_innovation, row["coupled_response_history"]["innovation_after"][3:6], atol=1e-8, rtol=0
            )
            values = dict(
                actual_tracking=(target * after.inv()).as_rotvec(),
                nominal_prediction=actual_innovation,
                completed_innovation_prediction=(after * corrected.inv()).as_rotvec(),
                predicted_tracking=(target * predicted.inv()).as_rotvec(),
                corrected_predicted_tracking=(target * corrected.inv()).as_rotvec(),
            )
            for key, value in values.items():
                errors[key].append(value)
                pooled[key].append(value)
            command = np.asarray(row["command"])[:6]
            delta = np.asarray(contract["output_min"]) + (command - contract["input_min"]) * (
                np.asarray(contract["output_max"]) - contract["output_min"]
            ) / (np.asarray(contract["input_max"]) - contract["input_min"])
            forcing_difference = float(
                np.linalg.norm(np.asarray(row["coupled_response_model"]["rotation_forcing"]) - delta[3:])
            )
            stiffness_difference = float(
                np.linalg.norm(np.asarray(row["coupled_response_model"]["rotation_stiffness"]) - np.eye(3), ord=2)
            )
            forcing_differences.append(forcing_difference)
            stiffness_differences.append(stiffness_difference)
            command_angles.append(float(np.linalg.norm(delta[3:])))
            if np.all(delta[3:] == 0):
                zero_steps.append(step)
            details.append(
                dict(
                    step=step,
                    actual_tracking_norm_rad=float(np.linalg.norm(values["actual_tracking"])),
                    nominal_prediction_error_norm_rad=float(np.linalg.norm(actual_innovation)),
                    corrected_prediction_error_norm_rad=float(
                        np.linalg.norm(values["completed_innovation_prediction"])
                    ),
                    actual_goal_forcing_minus_physical_rotvec_norm=forcing_difference,
                    actual_goal_stiffness_minus_identity_norm=stiffness_difference,
                )
            )
        stats = {key + "_rmse_rad": rms(values) for key, values in errors.items()}
        np.testing.assert_allclose(
            stats["actual_tracking_rmse_rad"],
            verified["control"]["tracking"]["admitted"]["so3_rmse_rad"],
            atol=1e-8,
            rtol=0,
        )
        cases.append(
            dict(
                case=index,
                rows=len(rows),
                performance_qualified=verified["performance_qualified"],
                statistics=stats,
                maximum_physical_rotation_command_norm_rad=max(command_angles),
                maximum_actual_forcing_minus_physical_rotvec_norm=max(forcing_differences),
                maximum_actual_stiffness_minus_identity_norm=max(stiffness_differences),
                native_zero_rotation_steps=zero_steps,
                details=details,
            )
        )
    assert len(cases) == 18 and sum(case["rows"] for case in cases) == 4907
    e.recheck()
    root.mkdir()
    result = dict(
        complete=True,
        cases=cases,
        rows=4907,
        statistics={key + "_rmse_rad": rms(values) for key, values in pooled.items()},
        source_sha256=e.hashes,
        seed=42,
        physics_steps=0,
        new_fits=0,
        model_calls=0,
        evidence_level="same recorded actual commands/states;angular prediction diagnostic;no six-dimensional inverse or counterfactual physics",
        approximation_qualified=False,
        six_dimensional_controller_qualified=False,
    )
    publish(root / "verified.json", result)
    print(
        json.dumps(
            dict(
                rows=result["rows"],
                statistics=result["statistics"],
                case11=cases[11]["statistics"],
                max_forcing_difference=max(case["maximum_actual_forcing_minus_physical_rotvec_norm"] for case in cases),
                max_stiffness_difference=max(case["maximum_actual_stiffness_minus_identity_norm"] for case in cases),
                zero_rotation_steps=sum(len(case["native_zero_rotation_steps"]) for case in cases),
            )
        )
    )


if __name__ == "__main__":
    main()
