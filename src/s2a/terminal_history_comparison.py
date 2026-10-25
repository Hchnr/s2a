"""Exact historical052 comparison for the terminal-only058 intervention."""

import numpy as np

INPUTS = ("sample_active10_fp32", "hidden", "raw_velocity_active10", "c0_active10")
OUTPUTS = ("clean_residual_active10", "deployed_velocity_active10")


def compare_point(current, baseline, step_index):
    if type(step_index) is not int or not 0 <= step_index < 35:
        raise ValueError("Actual35-node index required")
    same_inputs = {name: np.array_equal(current[name], baseline[name]) for name in INPUTS}
    protected = slice(None) if step_index < 34 else slice(3, 9)
    same_outputs = {name: np.array_equal(current[name][:, protected], baseline[name][:, protected]) for name in OUTPUTS}
    return dict(
        inputs=same_inputs,
        protected_outputs=same_outputs,
        passed=all(same_inputs.values()) and all(same_outputs.values()),
        changed_residual_values=int(
            np.count_nonzero(current["clean_residual_active10"] != baseline["clean_residual_active10"])
        ),
    )
