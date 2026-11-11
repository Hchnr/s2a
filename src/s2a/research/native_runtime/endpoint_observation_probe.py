"""Record the actual native state around a selected endpoint observation."""

import numpy as np

from s2a.research.wam.training.native.native_training import state_sha256


class EndpointObservationProbe:
    """Observe endpoint calls without restoring state or dispatching actions."""

    def __init__(self, endpoint, *, step):
        if not callable(endpoint) or type(step) is not int or step < 0:
            raise ValueError("One endpoint observer and a nonnegative step are required")
        self.endpoint, self.step, self.events = endpoint, step, []

    @staticmethod
    def _snapshot(env):
        sim = env.sim
        state = np.asarray(sim.get_state().flatten()).copy()
        time = float(sim.data.time)
        nq, nv, na = int(sim.model.nq), int(sim.model.nv), int(sim.model.na)
        if (
            state.ndim != 1
            or state.dtype != np.float64
            or not np.isfinite(state).all()
            or not np.isfinite(time)
            or na != 0
            or len(state) != 1 + nq + nv
            or state[0] != time
            or not np.array_equal(state[1 : 1 + nq], np.asarray(sim.data.qpos))
            or not np.array_equal(state[1 + nq :], np.asarray(sim.data.qvel))
        ):
            raise ValueError("Finite native time, qpos and qvel must match the flattened state")
        return {"state": state, "time": time, "sha256": state_sha256(state), "nq": nq, "nv": nv}

    def __call__(self, env, index):
        if index != self.step:
            return self.endpoint(env, index)
        before = self._snapshot(env)
        event = {"step": index, "call_index": len(self.events), "before": before, "completed": False}
        self.events.append(event)
        try:
            value = self.endpoint(env, index)
            event["completed"] = True
            return value
        finally:
            after = self._snapshot(env)
            event["after"] = after
            delta = after["state"] - before["state"]
            event["changed_indices"] = np.flatnonzero(before["state"] != after["state"])
            event["max_absolute_difference"] = float(np.max(np.abs(delta)))
            event["time_changed"] = before["time"] != after["time"]
