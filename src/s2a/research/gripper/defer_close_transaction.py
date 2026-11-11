"""Stage a causal close deferral against the physically executed command."""

import copy
from dataclasses import asdict

from s2a.research.gripper.defer_close_release import propose_deferred_close
from s2a.research.gripper.loaded_gripper_transaction import LoadedGripperTransaction


class DeferringLoadedGripperTransaction(LoadedGripperTransaction):
    """Commit the observed latch update only after the selected command completes."""

    def prepare(
        self,
        opening,
        motor_cache,
        period_velocity,
        generated_poses,
        generated_openings,
        current_index,
        measured_tcp,
        *,
        previous_command,
        dt,
    ):
        decision = super().prepare(
            opening,
            motor_cache,
            period_velocity,
            generated_poses,
            generated_openings,
            current_index,
            measured_tcp,
            previous_command=previous_command,
            dt=dt,
        )
        try:
            baseline = copy.deepcopy(decision["detail"])
            proposal = propose_deferred_close(
                generated_poses,
                generated_openings,
                current_index,
                measured_tcp,
                opening,
                motor_cache,
                decision["command"],
                baseline["latch"]["closing_episode"],
            )
            command = proposal["command"]
            controller = self._staged[1]
            if proposal["deferred"]:
                prediction = controller.planner.model.predict(opening, float(motor_cache[0]), period_velocity, command)
                envelope = controller.planner.velocity_envelope
                decision["detail"]["executed_response"] = asdict(prediction)
                decision["detail"]["executed_prediction_in_velocity_envelope"] = bool(
                    abs(period_velocity) <= envelope and abs(prediction.next_period_velocity) <= envelope
                )
            decision["command"] = decision["detail"]["command"] = command
            decision["detail"]["baseline"] = baseline
            decision["detail"]["deferral"] = proposal
            self._staged = (decision, controller, copy.deepcopy(decision))
            return decision
        except BaseException:
            self.abort(decision)
            raise
