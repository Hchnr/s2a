"""156 independent current-conditioned inverse log replay, no production inverse."""

import numpy as np
from arm_joint_wrench_independent import nominal_prediction
from guarded_loading_acceptance_checker import independent_reference_velocity
from rate_preview_independent import verify_plan
from scipy.spatial.transform import Rotation


class PreviewArmLogReplay:
    def __init__(self, parameters, contract, source_from_site):
        self.parameters = parameters
        self.contract = contract
        self.mapping = np.asarray(source_from_site)
        self.previous = np.zeros(7)
        self.last = None
        self.steps = 0

    def verify(self, row, arm, desired, site_velocity, generated, cursor, anchor):
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
        logged = row["translation_preview"]
        generated = np.asarray(generated)
        assert type(cursor) is int and 0 <= cursor < len(generated)
        horizon = min(6, len(generated) - cursor)
        assert logged["horizon"] == horizon
        assert logged["generated_indices"] == list(range(cursor, cursor + horizon))
        refs = generated[cursor : cursor + horizon, :3, 3]
        velocities = np.array(
            [independent_reference_velocity(generated, cursor + j, anchor, 0.05)["velocity"] for j in range(horizon)]
        )
        innovation = np.vstack([expected[:3], expected[6:9] / 0.05])
        np.testing.assert_allclose(generated[cursor], target, rtol=0, atol=1e-9)
        np.testing.assert_allclose(logged["reference_positions"], refs, rtol=0, atol=1e-9)
        np.testing.assert_allclose(logged["reference_velocities"], velocities, rtol=0, atol=1e-9)
        np.testing.assert_allclose(logged["completed_innovation"], innovation, rtol=0, atol=1e-9)
        verify_plan(logged, pose[:3, 3], twist[:3], refs, velocities, self.previous[:3], self.contract, innovation)
        np.testing.assert_allclose(command[:3], logged["command"], rtol=0, atol=1e-12)
        lo, hi, outlo, outhi = (
            np.asarray(self.contract[k])[:3] for k in ("input_min", "input_max", "output_min", "output_max")
        )
        physical = outlo + (command[:3] - lo) * (outhi - outlo) / (hi - lo)
        np.testing.assert_allclose(logged["physical_delta"], physical, rtol=0, atol=1e-12)
        np.testing.assert_allclose(data["action_goal_delta"][0, :3], physical, rtol=0, atol=1e-12)
        assert row["prediction_kind"] == "219 physical preview;150 completed innovation held;not measured future"
        assert row["rate_limited"] == bool(
            np.any(np.isclose(np.abs(command[:3] - self.previous[:3]), 0.25, atol=1e-10))
        )
        nominal = nominal_prediction(data, self.parameters)[0]
        np.testing.assert_allclose(row["nominal_predicted_motion"], nominal[:6], rtol=0, atol=1e-9)
        np.testing.assert_allclose(row["nominal_predicted_linear_velocity"], nominal[6:9] / 0.05, rtol=0, atol=1e-9)
        np.testing.assert_allclose(
            row["predicted_linear_velocity"], np.asarray(logged["predicted_velocities"])[0], rtol=0, atol=1e-9
        )
        np.testing.assert_allclose(
            np.asarray(logged["predicted_positions"])[0] - pose[:3, 3], nominal[:3] + expected[:3], rtol=0, atol=1e-9
        )
        self.last = (pose.copy(), nominal.copy())
        self.previous = command.copy()
        self.steps += 1
