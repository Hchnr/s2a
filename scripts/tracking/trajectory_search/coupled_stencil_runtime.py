"""Explicit-output CPU batch runtime; arithmetic follows the qualified stencil worker."""

import os
import traceback
from pathlib import Path

import numpy as np

from s2a.controllers.osc import OSCInterface
from s2a.dynamics.mujoco_robot import MujocoRobotDynamics
from s2a.infrastructure.native_environment import strict_native_environment
from s2a.infrastructure.publication import publish_result
from s2a.research.trajectory_search.constraint_residual_repair import evaluate_repair_entry
from s2a.research.trajectory_search.replay_order_trajectory import rollout_replay_order_trajectory
from s2a.research.trajectory_search.restored_anchor_sqp_evidence import (
    CPU_ENVIRONMENT,
    checked,
    require,
)
from scripts.verification.trajectory_search.check_tracking_v5_tail_translation_reserve_v2 import semantic_compare


def production_batch(output_root, rank, commands, source, control, xml_ref, tool):
    """One explicitly billed model and an exclusive directory for each batch."""
    strict_native_environment()
    require(all(os.environ.get(k) == v for k, v in CPU_ENVIRONMENT.items()), "Single-thread CPU worker")
    output = Path(output_root) / f"batch_{rank:02d}"
    (output / "tmp").mkdir(parents=True, exist_ok=False)
    os.environ["TMPDIR"] = str(output / "tmp")
    result = dict(
        complete=False,
        passed=False,
        pid=os.getpid(),
        model_attempts=0,
        models=0,
        responses=0,
        inner_steps=0,
        queries=0,
        records=[],
    )
    dynamics = None
    try:
        require(2 <= len(commands) <= 3, "Fixed 32-batch response partition")
        result["model_attempts"] += 1
        dynamics = MujocoRobotDynamics(
            checked(xml_ref),
            control["controller_site"],
            [f"robot0_joint{k}" for k in range(1, 8)],
            source["fixed_gripper_state"],
        )
        result["models"] += 1
        q, velocity = (np.asarray(source[key]) for key in ("current_q", "current_velocity"))
        for key, value in dynamics(q, velocity).items():
            semantic_compare("initial_" + key, value, source["current_quantities"][key], 1e-10)
        interface = OSCInterface(*(control[key] for key in ("input_min", "input_max", "output_min", "output_max")))
        limits, torque = (np.asarray(source[key]) for key in ("joint_limits", "torque_limits"))
        for index, command in commands:
            rollout = rollout_replay_order_trajectory(
                q,
                velocity,
                command,
                interface,
                np.asarray(source["initial_joint"]),
                np.asarray(source["kp"]),
                np.asarray(source["kd"]),
                dynamics,
                torque[:, 0],
                torque[:, 1],
                latched_rotation_goal=np.asarray(source["latched_rotation_goal"]),
            )
            result["responses"] += 1
            result["inner_steps"] += 300
            entry = evaluate_repair_entry(
                rollout,
                command,
                np.asarray(source["previous_command"])[:6],
                interface,
                limits[:, 0],
                limits[:, 1],
                source["desired_tcp"],
                tool,
                source["past_inner_error_rad"],
                source["terminal_normals"],
            )
            ref = publish_result(output / f"response_{index:02d}.json", entry)
            result["records"].append(dict(index=index, ref=ref))
        require(dynamics.queries == 1 + 312 * len(commands), "Every batch query charged")
        result.update(complete=True, passed=True)
    except BaseException as error:
        result.update(error=repr(error), traceback=traceback.format_exc())
    finally:
        if dynamics is not None:
            result["queries"] = dynamics.queries
        strict_native_environment()
        result["result_ref"] = publish_result(output / "result.json", result)
    return result


def independent_causal_fit(D, changes, periods):
    """Independent normal equations in each causal prefix, retaining residuals."""
    D, changes, periods = np.asarray(D, dtype=float), np.asarray(changes, dtype=float), np.asarray(periods)
    require(D.shape == (72, 72) and changes.shape == (72, len(periods)), "Complete independent fit inputs")
    require(np.isfinite(D).all() and np.isfinite(changes).all(), "Finite independent design/response")
    require(
        periods.ndim == 1 and periods.size > 0 and np.isin(periods, np.arange(12)).all(), "Explicit valid causal rows"
    )
    J = np.zeros((len(periods), 72))
    residual = np.zeros_like(changes)
    for period in range(12):
        rows = np.flatnonzero(periods == period)
        end = 6 * (period + 1)
        P = D[:, :end] / 1e-4
        require(np.linalg.matrix_rank(P) == end and np.linalg.cond(P) <= 10, "Independent prefix geometry")
        if len(rows):
            coefficients = np.linalg.solve(P.T @ P, P.T @ (changes[:, rows] / 1e-4))
            J[rows, :end] = coefficients.T
            residual[:, rows] = changes[:, rows] - D[:, :end] @ coefficients
    return dict(jacobian=J, residual=residual)


def exact_prefix_observations(D, changes, periods):
    """Check observed response invariance for identical represented prefixes."""
    D, changes, periods = np.asarray(D, dtype=float), np.asarray(changes, dtype=float), np.asarray(periods)
    require(D.shape == (72, 72) and changes.shape == (72, len(periods)), "Complete prefix observations")
    comparisons = 0
    zero_checks = 0
    for period in range(12):
        rows = np.flatnonzero(periods == period)
        end = 6 * (period + 1)
        groups = {}
        for i, prefix in enumerate(D[:, :end]):
            value = changes[i, rows]
            if np.all(prefix == 0):
                require(np.all(value == 0), "Zero observed prefix has exact zero output change")
                zero_checks += 1
            key = tuple(prefix)
            if key in groups:
                require(np.array_equal(value, groups[key]), "Same observed prefix has exact same output")
                comparisons += 1
            else:
                groups[key] = value
    return dict(exact=True, identical_prefix_comparisons=comparisons, zero_prefix_checks=zero_checks)
