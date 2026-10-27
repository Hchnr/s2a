"""156 independent current-conditioned inverse log replay, no production inverse."""

import numpy as np
from arm_joint_wrench_independent import nominal_prediction
from late_clipped_arm_inverse_independent import features, inputs
from late_clipped_arm_inverse_independent import nominal as inverse_nominal
from scipy.spatial.transform import Rotation


class LateClippedArmLogReplay:
    def __init__(self, parameters, contract, source_from_site, inverse_parameters):
        self.inverse_parameters = inverse_parameters
        for key in ("kp", "kd", "input_min", "input_max", "output_min", "output_max"):
            np.testing.assert_allclose(
                contract[key], inverse_parameters["controller_contract"][key], rtol=0, atol=1e-12
            )
        self.parameters = parameters
        self.contract = contract
        self.mapping = np.asarray(source_from_site)
        self.previous = np.zeros(7)
        self.last = None
        self.steps = 0

    def verify(self, row, arm, desired, site_velocity):
        data = {k: np.asarray(v) for k, v in row["qualified_forward_input"].items()}
        pose = data["x_site_pose"][0]
        twist = data["x_site_twist"][0]
        np.testing.assert_allclose(pose, np.asarray(row["measured_source"]) @ self.mapping, rtol=0, atol=1e-8)
        np.testing.assert_allclose(twist, row["measured_twist"], rtol=0, atol=1e-9)
        np.testing.assert_allclose(data["x_previous_command"][0], self.previous, rtol=0, atol=1e-12)
        assert bool(data["x_previous_command_available"][0]) == bool(self.steps)
        expected = np.zeros(12)
        if self.last is not None:
            previous_pose, nominal = self.last
            expected[:3] = pose[:3, 3] - previous_pose[:3, 3] - nominal[:3]
            expected[3:6] = Rotation.from_matrix(
                (pose[:3, :3] @ previous_pose[:3, :3].T) @ Rotation.from_rotvec(nominal[3:6]).as_matrix().T
            ).as_rotvec()
            expected[6:] = 0.05 * twist - nominal[6:]
        np.testing.assert_allclose(data["x_response_residual"][0], expected, rtol=0, atol=1e-9)
        np.testing.assert_allclose(row["response_residual"]["innovation"], expected, rtol=0, atol=1e-9)
        assert row["response_residual"]["completed_periods"] == self.steps
        command = np.asarray(row["command"])
        np.testing.assert_allclose(data["action_command"][0], command, rtol=0, atol=1e-12)
        np.testing.assert_allclose(command[3:6], arm["command"][3:6], rtol=0, atol=1e-9)
        target = np.asarray(desired) @ self.mapping
        np.testing.assert_allclose(row["desired_site"], target, rtol=0, atol=1e-9)
        for k, v in site_velocity.items():
            if k == "velocity":
                np.testing.assert_allclose(row["site_velocity_reference"][k], v, rtol=0, atol=1e-9)
            else:
                assert row["site_velocity_reference"][k] == v
        logged = row["direct_inverse"]
        values = inputs(data, target[None], command[-1:])
        base = inverse_nominal(data, target[None], self.contract)[0]
        residual = (features(values, self.inverse_parameters) @ np.asarray(self.inverse_parameters["coefficients"]))[0]
        lo, hi, outlo, outhi = (
            np.asarray(self.contract[k])[:3] for k in ("input_min", "input_max", "output_min", "output_max")
        )
        prediction = np.minimum(hi, np.maximum(lo, base + residual))
        lower = np.maximum(lo, self.previous[:3] - 0.25)
        upper = np.minimum(hi, self.previous[:3] + 0.25)
        executed = np.array([min(float(upper[j]), max(float(lower[j]), float(prediction[j]))) for j in range(3)])
        physical = outlo + (executed - lo) * (outhi - outlo) / (hi - lo)
        for key, value in dict(
            command=executed,
            nominal_command=base,
            residual_command=residual,
            unconstrained_rate_command=prediction,
            lower=lower,
            upper=upper,
            physical_delta=physical,
        ).items():
            np.testing.assert_allclose(logged[key], value, rtol=0, atol=1e-9)
        assert logged["rate_limited"] == bool(np.any(executed != prediction))
        assert row["prediction_kind"] == "physical nominal only;direct inverse does not predict next measured state"
        np.testing.assert_allclose(command[:3], executed, rtol=0, atol=1e-9)
        np.testing.assert_allclose(data["action_goal_delta"][0, :3], physical, rtol=0, atol=1e-12)
        nominal = nominal_prediction(data, self.parameters)[0]
        np.testing.assert_allclose(row["nominal_predicted_motion"], nominal[:6], rtol=0, atol=1e-9)
        np.testing.assert_allclose(row["nominal_predicted_linear_velocity"], nominal[6:9] / 0.05, rtol=0, atol=1e-9)
        np.testing.assert_allclose(row["predicted_linear_velocity"], nominal[6:9] / 0.05, rtol=0, atol=1e-9)
        self.last = (pose.copy(), nominal.copy())
        self.previous = command.copy()
        self.steps += 1
