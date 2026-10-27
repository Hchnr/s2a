"""Independent cache-side comparator for live152 inputs; labels never enter reader."""

import numpy as np
from arm_joint_wrench_independent import INPUTS
from arm_wrench_data import attach_current_wrench
from audit_wam_position import Evidence

from s2a.current_model_state import current_model_state
from s2a.joint_wrench_condition_forward import with_history
from s2a.runtime_response_history import RuntimeResponseHistory


class RuntimePrefixChecker:
    def __init__(self):
        self.e = Evidence()
        for p in (
            __file__,
            "src/s2a/current_model_state.py",
            "src/s2a/runtime_response_history.py",
            "plans/v0.0.1/152_runtime_prefix_parity_20261025.md",
        ):
            self.e.read(p)
        q = self.e.json("outputs/left_table_forward_qualification_20261025/qualification.json")
        assert q["offline_forward_eligible"]
        ref = q["parameter_ref"]
        self.e.check(ref["path"], ref["sha256"])
        self.parameters = self.e.json(ref["path"])
        c = self.e.json("configs/left_table_forward_20261025.json")
        ref = c["data_result_ref"]
        self.e.check(ref["path"], ref["sha256"])
        entry = next(r for r in self.e.json(ref["path"])["trajectories"] if r["index"] == 23)
        ref = entry["cache_ref"]
        self.e.check(ref["path"], ref["sha256"])
        with np.load(ref["path"], allow_pickle=False) as a:
            self.expected = attach_current_wrench(
                with_history({k: a[k].copy() for k in a.files}, self.parameters), entry, c, self.e
            )
        self.history = RuntimeResponseHistory()
        self.last = None

    def check(self, k, env, sample, commands):
        if k:
            self.history.observe_completed(
                self.last, sample["controller_site_pose"], sample["source_twist"], self.parameters
            )
        if k < len(commands):
            self.last, _ = current_model_state(
                env,
                commands[k],
                commands[k - 1] if k else np.zeros(7),
                previous_available=bool(k),
                history=self.history,
            )
            for field in INPUTS:
                np.testing.assert_allclose(
                    self.last[field], self.expected[field][k : k + 1], rtol=0, atol=1e-8, err_msg=field
                )

    def finish(self):
        self.e.recheck()
        return dict(
            runtime_input_parity=True,
            completed_history_periods=self.history.completed_periods,
            parity_source_sha256=self.e.hashes,
        )
