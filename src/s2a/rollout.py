"""Counterfactual LOCAL MODEL rollouts: candidate commands advance every state.

These are not independent physics simulations or robot experiments. Only the
initial measured pose and an already generated oracle reference enter a run.
"""

from dataclasses import asdict
from itertools import product

import numpy as np

from s2a.artifacts import write_json
from s2a.control import PIDConfig, PosePID
from s2a.geometry import advance, euler_pose_to_transform, pose_error
from s2a.interface import DroidReferenceInterface
from s2a.trajectory import ReferenceTrajectory


def past_vector(commands, motions, history):
    return (
        np.concatenate(
            [np.r_[commands[-j], motions[-j]] if j <= len(commands) else np.zeros(12) for j in range(1, history + 1)]
        )
        if history
        else np.empty(0)
    )


def simulate(
    model, interface, reference, times, *, gains, dt_command, inverse_lambda, feedforward_kind, scenario, settings, seed
):
    rng = np.random.default_rng(seed)
    controller = PosePID(PIDConfig(**settings["pid_limits"], **gains))
    state = reference.sample(times[0])[0]
    initial = np.array(settings["initial_offset"]) if scenario in {"initial_offset", "stationary"} else np.zeros(6)
    state = advance(state, initial, 1.0)
    queue = [np.zeros(6) for _ in range(model.lag)]
    input_queue = [np.zeros(6) for _ in range(settings["extra_delay_steps"] if scenario == "delay" else 0)]
    applied_commands = []
    commands, motions, observed_motions = [], [], []
    previous_measured = None
    trace = {
        key: []
        for key in [
            "state",
            "reference",
            "error",
            "integral",
            "derivative",
            "command",
            "applied_command",
            "raw_command",
            "raw_velocity",
            "velocity",
            "saturated",
        ]
    }
    for i, now in enumerate(times[:-1]):
        measured = state.copy()
        if scenario == "observation_noise":
            measured = advance(measured, rng.normal(size=6) * np.array(settings["observation_noise_std"]), 1.0)
        if previous_measured is not None:
            observed_motions.append(pose_error(measured, previous_measured) / (now - times[i - 1]))
        desired, ff = reference.sample(now)
        if feedforward_kind == "linear":
            _, ff = reference.sample(now + model.lag * dt_command)
        history_observed = past_vector(commands, observed_motions, model.history)

        def apply(velocity):
            prior = interface.encode_velocity(velocity, measured, dt_command)
            raw_u = (
                model.inverse(velocity, history_observed, prior, inverse_lambda)
                if feedforward_kind == "linear"
                else prior
            )
            elapsed = dt_command if controller.state.previous_time is None else now - controller.state.previous_time
            previous = commands[-1] if commands else np.zeros(6)
            sent = interface.limit_rate(
                raw_u,
                previous,
                elapsed,
                controller.config.max_translation_acceleration * dt_command / interface.max_translation_delta,
                controller.config.max_rotation_acceleration * dt_command / interface.max_rotation_delta,
            )
            if feedforward_kind == "linear":
                # Difference due to ALL final downstream limits in physical response space.
                delivered = velocity + model.B @ (sent - raw_u)
            else:
                delivered = pose_error(interface.decode_target(sent, measured), measured) / dt_command
            return delivered, {"sent": sent, "raw": raw_u}

        try:
            result = controller.update(measured, desired, ff, float(now), apply=apply)
        except ValueError as error:
            return {"valid": False, "reason": str(error), "completed_transitions": i}, None
        command = result["command"]["sent"]
        input_queue.append(command.copy())
        applied_command = input_queue.pop(0)
        true_history = past_vector(applied_commands, motions, model.history)
        prediction = model.predict(applied_command, true_history)
        queue.append(prediction)
        motion = queue.pop(0)
        if scenario == "response_gain":
            motion = motion * settings["response_gain"]
        if scenario in {"constant_disturbance", "stationary"}:
            motion = motion + np.array(settings["constant_disturbance"])
        if not np.isfinite(motion).all() or np.linalg.norm(motion[:3]) > 5 or np.linalg.norm(motion[3:]) > 20:
            return {
                "valid": False,
                "reason": "local response escaped declared velocity domain",
                "completed_transitions": i,
            }, None
        true_error = pose_error(desired, state)
        for key, value in [
            ("state", state),
            ("reference", desired),
            ("error", true_error),
            ("integral", result["integral"]),
            ("derivative", result["derivative"]),
            ("command", command),
            ("applied_command", applied_command),
            ("raw_command", result["command"]["raw"]),
            ("raw_velocity", result["raw_velocity"]),
            ("velocity", motion),
            ("saturated", result["saturated"]),
        ]:
            trace[key].append(np.array(value).copy())
        commands.append(command.copy())
        applied_commands.append(applied_command.copy())
        motions.append(motion.copy())
        previous_measured = measured.copy()
        state = advance(state, motion, float(times[i + 1] - now))
    trace = {key: np.array(value) for key, value in trace.items()}
    trace["times"] = times[:-1].copy()
    trace["endpoint_state"] = state.copy()
    warmup = settings["warmup_transitions"]
    error = trace["error"][warmup:]
    position = np.linalg.norm(error[:, :3], axis=1)
    rotation = np.linalg.norm(error[:, 3:], axis=1)
    score = np.sqrt(
        np.mean((position / settings["position_scale_m"]) ** 2 + (rotation / settings["rotation_scale_rad"]) ** 2) / 2
    )
    end_error = pose_error(reference.sample(times[-1])[0], state)
    rates = np.diff(trace["command"], axis=0) / np.diff(trace["times"])[:, None]
    result = {
        "valid": True,
        "scored_transitions": len(error),
        "warmup_transitions": warmup,
        "balanced_tracking_error": float(score),
        "position_rmse_m": float(np.sqrt(np.mean(position**2))),
        "rotation_rmse_rad": float(np.sqrt(np.mean(rotation**2))),
        "endpoint_position_m": float(np.linalg.norm(end_error[:3])),
        "endpoint_rotation_rad": float(np.linalg.norm(end_error[3:])),
        "saturation_fraction": float(np.mean(trace["saturated"][warmup:])),
        "saturation_duration_s": float(np.sum(np.diff(times)[warmup:] * trace["saturated"][warmup:])),
        "command_rate_rms_per_s": float(np.sqrt(np.mean(rates[max(0, warmup - 1) :] ** 2))),
        "command_total_variation": float(np.sum(np.linalg.norm(np.diff(trace["command"][warmup:], axis=0), axis=1))),
    }
    if scenario == "stationary":
        remaining = np.linalg.norm(trace["error"][:, :3], axis=1)
        rotation_remaining = np.linalg.norm(trace["error"][:, 3:], axis=1)
        settled = (remaining < settings["settle_position_m"]) & (rotation_remaining < settings["settle_rotation_rad"])
        indices = [j for j in range(len(settled)) if settled[j:].all()]
        initial_error = trace["error"][0]

        def overshoot(part):
            unit = initial_error[part] / max(np.linalg.norm(initial_error[part]), 1e-12)
            return float(max(0, -np.min(trace["error"][:, part] @ unit)))

        result.update(
            {
                "steady_position_m": float(np.mean(remaining[-5:])),
                "steady_rotation_rad": float(np.mean(rotation_remaining[-5:])),
                "settling_time_s": float(trace["times"][indices[0]] - times[0]) if indices else None,
                "position_overshoot_m": overshoot(slice(0, 3)),
                "rotation_overshoot_rad": overshoot(slice(3, 6)),
            }
        )
    return result, trace


def gain_candidates(method, settings):
    zero = (0.0,) * 6
    if method == "FF":
        return [{"kp": zero, "ki": zero, "kd": zero}]
    ps = settings["kp_grid"]
    ins = settings["ki_grid"] if "I" in method else [0.0]
    ds = settings["kd_grid"] if "D" in method else [0.0]
    return [{"kp": (p,) * 6, "ki": (ii,) * 6, "kd": (d,) * 6} for p, ii, d in product(ps, ins, ds)]


def compare_controllers(config, arrays, model, params, out):
    settings = config["rollout"]
    if not params["response_gate_passed"]:
        result = {
            "evidence_level": "local_identified_model_only",
            "executed": False,
            "reason": "validation response gate failed",
        }
        write_json(out / "closed_loop_model_metrics.json", result)
        write_json(out / "pid_config.json", {"status": "not_tuned", "reason": result["reason"]})
        return result
    f, clock = arrays["frame_index"], arrays["source_physical_timestamp"]
    poses = euler_pose_to_transform(arrays["r1_observation_state_cartesian_position"])
    interface = DroidReferenceInterface(**config["reference_interface"]["parameters"])
    scenarios = settings["scenarios"]

    def run(gains, partition, feedforward_kind, save=False):
        lo, hi = config["split_intervals_half_open"][partition]
        loc = np.flatnonzero((f >= lo) & (f <= hi))
        reference = ReferenceTrajectory(clock[loc], poses[loc])
        metrics, traces = {}, {}
        for j, scenario in enumerate(scenarios):
            ref, times = reference, clock[loc]
            if scenario == "stationary":
                end_pose = reference.sample(times[-1])[0]
                times = np.arange(settings["stationary_steps"] + 1) * params["dt_command_planned_s"]
                ref = ReferenceTrajectory(np.array([times[0], times[-1]]), np.array([end_pose, end_pose]))
            value, trace = simulate(
                model,
                interface,
                ref,
                times,
                gains=gains,
                dt_command=params["dt_command_planned_s"],
                inverse_lambda=params["inverse_regularization"],
                feedforward_kind=feedforward_kind,
                scenario=scenario,
                settings=settings,
                seed=config["seed"] + j,
            )
            metrics[scenario] = value
            if save and trace is not None:
                traces[scenario] = trace
        return metrics, traces

    selected, search = {}, {}
    for feedforward in settings["feedforward_kinds"]:
        selected[feedforward], search[feedforward] = {}, {}
        for method in settings["methods"]:
            candidates = []
            for gains in gain_candidates(method, settings):
                validation, _ = run(gains, "validation", feedforward)
                valid = all(v["valid"] for v in validation.values())
                objective = (
                    float(np.mean([v["balanced_tracking_error"] for v in validation.values()])) if valid else None
                )
                candidates.append(
                    {"gains": gains, "validation_objective": objective, "validation_scenarios": validation}
                )
            usable = [v for v in candidates if v["validation_objective"] is not None]
            best = dict(min(usable, key=lambda v: v["validation_objective"]) if usable else candidates[0])
            best["selection_status"] = (
                "selected_on_validation" if usable else "no_valid_gain_candidate_fixed_diagnostic"
            )
            selected[feedforward][method] = best
            search[feedforward][method] = candidates
    # Freeze every choice before opening the test comparison.
    write_json(
        out / "pid_config.json",
        {
            "selected_on": "validation only; independent state per candidate/scenario/split",
            "selected": selected,
            "settings": settings,
            "config_defaults": asdict(PIDConfig()),
        },
    )
    results = {}
    trace_arrays = {}
    for feedforward in settings["feedforward_kinds"]:
        results[feedforward] = {}
        for method, best in selected[feedforward].items():
            test, traces = run(best["gains"], "test", feedforward, save=True)
            for scenario, trace in traces.items():
                for key, value in trace.items():
                    trace_arrays[f"{feedforward}__{method}__{scenario}__{key}"] = value
            results[feedforward][method] = {
                "gains": best["gains"],
                "selection_status": best["selection_status"],
                "validation": best["validation_scenarios"],
                "test": test,
            }
    acceptance = {}
    for feedforward, methods in results.items():
        acceptance[feedforward] = {}
        for method, values in methods.items():
            if method == "FF":
                continue
            comparisons = {}
            for scenario, value in values["test"].items():
                baseline = methods["FF"]["test"][scenario]
                if not value["valid"] or not baseline["valid"]:
                    comparisons[scenario] = {"passed": False, "reason": "invalid candidate or baseline rollout"}
                    continue
                ratio = value["balanced_tracking_error"] / max(baseline["balanced_tracking_error"], 1e-12)
                safeguards = {
                    "tracking_improved_20pct": ratio <= 0.8,
                    "position_no_regression": value["position_rmse_m"] <= 1.1 * baseline["position_rmse_m"] + 1e-6,
                    "rotation_no_regression": value["rotation_rmse_rad"] <= 1.1 * baseline["rotation_rmse_rad"] + 1e-6,
                    "saturation_no_regression": value["saturation_fraction"] <= baseline["saturation_fraction"] + 0.05,
                    "command_rate_no_regression": value["command_rate_rms_per_s"]
                    <= 1.25 * baseline["command_rate_rms_per_s"] + 1e-3,
                }
                if scenario == "stationary":
                    safeguards["position_overshoot_no_regression"] = (
                        value["position_overshoot_m"] <= baseline["position_overshoot_m"] + 0.002
                    )
                    safeguards["rotation_overshoot_no_regression"] = (
                        value["rotation_overshoot_rad"] <= baseline["rotation_overshoot_rad"] + 0.01
                    )
                comparisons[scenario] = {
                    "tracking_ratio": ratio,
                    "checks": safeguards,
                    "passed": all(safeguards.values()),
                }
            acceptance[feedforward][method] = {
                "scenarios": comparisons,
                "all_perturbations_passed": values["selection_status"] == "selected_on_validation"
                and all(v["passed"] for k, v in comparisons.items() if k != "nominal"),
            }
    result = {
        "evidence_level": "local_identified_model_only",
        "executed": True,
        "oracle_reference": True,
        "results": results,
        "acceptance": acceptance,
        "validation_search": search,
        "limitations": [
            "single episode; no independent simulation/hardware",
            "reference public interface differs from historical command algebra",
            "model lacks independent joint/contact dynamics; state perturbations do not identify new physics",
        ],
        "unexecuted": {
            "gain_scheduling": "P4 requires verified same-device multiple episodes",
            "direct_inverse_MLP": "P4 requires episode-level independent splits",
            "residual_MLP": "P4 requires episode-level independent splits",
            "independent_simulator": "no verified matching controller/robot configuration",
            "hardware": "robot serial, controller config and execution channel unverified",
            "world_model_reference": "frozen predicted chunks not provided",
        },
    }
    write_json(out / "closed_loop_model_metrics.json", result)
    np.savez_compressed(out / "closed_loop_traces.npz", **trace_arrays)
    return result
