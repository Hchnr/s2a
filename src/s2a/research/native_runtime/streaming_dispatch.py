"""Track physical dispatch and its actual inner samples for a streaming owner."""

from contextlib import contextmanager

import numpy as np

from s2a.research.native_runtime.streaming_continuation import measured_snapshot


@contextmanager
def track_stream_dispatch(env, ledger, row):
    """Bracket each dispatched period, retaining partial evidence on failure.

    The enclosing continuation records its own samples independently. The
    owner compares both records and commits feedback before closing the ledger
    transaction. The simulator remains the only source of actual observations.
    """
    if type(row) is not dict or row.get("inflight") is not None:
        raise ValueError("One fresh dispatch receipt is required")
    missing = object()
    prior_attribute = vars(env).get("step", missing)
    original_env_step = env.step

    def tracked_step(command, *args, **kwargs):
        if row["inflight"] is not None:
            raise ValueError("A previous dispatched action remains incomplete")
        before = measured_snapshot(env, ledger.current["step"])
        ledger.begin_period(before=before)
        inflight = {
            "step": before["step"],
            "command": np.asarray(command, dtype=float).copy(),
            "inner_attempted": 0,
            "inner_completed": 0,
            "actual_inner": [],
        }
        row["inflight"] = inflight
        original_sim_step = env.sim.step

        def tracked_inner(*inner_args, **inner_kwargs):
            inflight["inner_attempted"] += 1
            value = original_sim_step(*inner_args, **inner_kwargs)
            inflight["inner_completed"] += 1
            robot, sim = env.robots[0], env.sim
            inflight["actual_inner"].append(
                {
                    "time": float(sim.data.time),
                    "q": np.asarray(sim.data.qpos)[robot._ref_joint_pos_indexes].copy(),
                    "qvel": np.asarray(sim.data.qvel)[robot._ref_joint_vel_indexes].copy(),
                }
            )
            return value

        env.sim.step = tracked_inner
        try:
            return original_env_step(command, *args, **kwargs)
        finally:
            env.sim.step = original_sim_step

    env.step = tracked_step
    try:
        yield
    finally:
        if prior_attribute is missing:
            del env.step
        else:
            env.step = prior_attribute
