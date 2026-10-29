"""Prepare, commit and observe one actual tracking-preview control period."""

import copy

import numpy as np

from s2a.adaptive_plan_history import AdaptivePlanHistory
from s2a.geometry import finite


class TrackingPreviewTransaction:
    """No physics stepping: the caller owns dispatch and actual endpoint sampling.

    A staged future plan becomes available as a numerical warm start only after
    the dispatched first command has been observed. Reset for independent cases.
    """

    def __init__(self, *, refiner=None):
        if refiner is None:
            from s2a.tracking_margin_native import refine_native_preview

            refiner = refine_native_preview
        self._refiner = refiner
        self.reset()

    def reset(self):
        self.history = AdaptivePlanHistory()
        self.phase = "ready"
        self._prepared = None

    def prepare(
        self, env, dynamics, site, twist, poses, anchor, index, seed_plan, previous, contract, *, site_from_tcp
    ):
        if self.phase != "ready":
            raise ValueError("Observe the committed period or explicitly reset before preparing again")
        self.phase = "preparing"
        try:
            result = self._refiner(
                env,
                dynamics,
                site,
                twist,
                poses,
                anchor,
                index,
                seed_plan,
                previous,
                contract,
                self.history,
                site_from_tcp=site_from_tcp,
            )
            command = finite(result["command"], (6,)).copy()
            plan = finite(result["planned_commands"])
            if plan.ndim != 2 or plan.shape[1] != 6 or not np.array_equal(plan[0], command):
                raise ValueError("Actual first command must match the staged plan")
            if self.history.pending_plan is None or not np.array_equal(self.history.pending_plan, plan):
                raise ValueError("Refiner must stage the selected plan")
            self._prepared = dict(
                site=finite(site, (4, 4)).copy(),
                twist=finite(twist, (6,)).copy(),
                command=command,
                prediction=finite(result["nominal_response"], (12,)).copy(),
            )
            self.phase = "prepared"
            return copy.deepcopy(result)
        except BaseException:
            self.phase = "faulted"
            raise

    def commit(self, actual_arm_command):
        if self.phase != "prepared":
            raise ValueError("One prepared command required before dispatch commitment")
        command = finite(actual_arm_command, (6,))
        if not np.array_equal(command, self._prepared["command"]):
            raise ValueError("Dispatched arm command differs from the prepared prediction")
        p = self._prepared
        self.history.commit_prediction(p["site"], p["twist"], p["prediction"])
        self.phase = "committed"

    def observe(self, actual_site, actual_twist):
        if self.phase != "committed":
            raise ValueError("Actual command must be committed before observing its completed endpoint")
        innovation = self.history.observe_completed(actual_site, actual_twist)
        self._prepared = None
        self.phase = "ready"
        return innovation.copy()
