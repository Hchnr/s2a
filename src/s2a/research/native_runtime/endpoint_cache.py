"""Reuse a measured endpoint only while its exact state and clock stay fixed."""

import copy

import numpy as np

from s2a.research.wam.training.native.native_training import state_sha256


class SameStateEndpointCache:
    """Avoid a second endpoint read without normalizing or restoring state.

    A caller must still decide when an observation is causally available.
    This cache checks the raw simulator snapshot on every use and exposes no
    future endpoint or state that differs from the current one.
    """

    def __init__(self, endpoint):
        if not callable(endpoint):
            raise ValueError("A measured endpoint callback is required")
        self.endpoint = endpoint
        self._last = None
        self.events = []

    @staticmethod
    def current_snapshot(env, step):
        if type(step) is not int or step < 0:
            raise ValueError("A nonnegative current step is required")
        state = np.asarray(env.sim.get_state().flatten())
        time = float(env.sim.data.time)
        if not np.isfinite(time) or time < 0:
            raise ValueError("A finite current simulation clock is required")
        return {"step": step, "time": time, "state_sha256": state_sha256(state)}

    def __call__(self, env, step):
        before = self.current_snapshot(env, step)
        if self._last is not None and step == self._last["snapshot"]["step"]:
            if before != self._last["snapshot"]:
                raise ValueError("A cached endpoint requires the exact same current state and time")
            self.events.append({**before, "kind": "reused"})
            return copy.deepcopy(self._last["sample"]), copy.deepcopy(self._last["receipt"])
        if self._last is not None and step < self._last["snapshot"]["step"]:
            raise ValueError("Endpoint observations must not go backwards")
        sample, receipt = self.endpoint(env, step)
        after = self.current_snapshot(env, step)
        if (
            type(sample) is not dict
            or type(receipt) is not dict
            or type(receipt.get("endpoint_step")) is not int
            or receipt.get("endpoint_step") != step
            or receipt.get("sim_time") != after["time"]
            or receipt.get("sim_state_sha256") != after["state_sha256"]
            or state_sha256(sample["sim_states"]) != after["state_sha256"]
        ):
            raise ValueError("The measured endpoint must bind the actual post-observation state")
        self._last = {"snapshot": after, "sample": copy.deepcopy(sample), "receipt": copy.deepcopy(receipt)}
        self.events.append({**after, "kind": "observed"})
        return copy.deepcopy(sample), copy.deepcopy(receipt)
