"""P2 temporal selection, forward-response gates and command diagnostics."""

import numpy as np

from s2a.artifacts import write_json
from s2a.data import split_indices
from s2a.geometry import advance, error_summary, euler_pose_to_transform, pose_error
from s2a.identification import LinearResponse, evaluate_one_step, history_vector, prediction_metrics, replay_model
from s2a.interface import DroidReferenceInterface


def command_metrics(prediction, target, fit_scale, moving):
    e = prediction - target
    rmse = np.sqrt(np.mean(e**2, axis=0))
    active = fit_scale > 1e-4
    normalized = [float(rmse[i] / fit_scale[i]) if active[i] else None for i in range(6)]
    correlation = []
    for i in range(6):
        if moving.sum() < 3 or np.std(prediction[moving, i]) < 1e-8 or np.std(target[moving, i]) < 1e-8:
            correlation.append(None)
        else:
            correlation.append(float(np.corrcoef(prediction[moving, i], target[moving, i])[0, 1]))
    return {
        "samples": len(e),
        "mae_per_axis": np.mean(np.abs(e), axis=0),
        "rmse_per_axis": rmse,
        "nrmse_per_axis": normalized,
        "near_constant_axes": np.flatnonzero(~active),
        "balanced_nrmse": float(np.sqrt(np.mean((e[:, active] / fit_scale[active]) ** 2))) if active.any() else None,
        "dynamic_samples": int(moving.sum()),
        "dynamic_correlation_per_axis": correlation,
        "saturation_fraction": float(
            np.mean(
                (np.linalg.norm(prediction[:, :3], axis=1) >= 1 - 1e-8)
                | (np.linalg.norm(prediction[:, 3:], axis=1) >= 1 - 1e-8)
            )
        ),
    }


def velocities_from_poses(poses, times):
    y = np.zeros((len(poses), 6))
    for i, dt in enumerate(np.diff(times)):
        y[i] = pose_error(poses[i + 1], poses[i]) / dt
    return y


def replay_metrics(model, commands, velocities, poses, times, warmup):
    prediction = replay_model(model, commands)
    predicted_poses = [poses[0]]
    for i, dt in enumerate(np.diff(times)):
        predicted_poses.append(advance(predicted_poses[-1], prediction[i], dt))
    predicted_poses = np.array(predicted_poses)
    count = len(times) - 1
    score = slice(warmup, count)
    return (
        {
            "velocity": prediction_metrics(prediction[score], velocities[score], model.response_scale),
            "pose": error_summary(predicted_poses[warmup:], poses[warmup:]),
            "hold_pose": error_summary(np.broadcast_to(poses[0], poses[warmup:].shape), poses[warmup:]),
            "warmup_transitions": warmup,
        },
        predicted_poses,
        prediction,
    )


def calibrate(config, arrays, out):
    settings = config["identification"]
    f, times = arrays["frame_index"], arrays["source_physical_timestamp"]
    poses = euler_pose_to_transform(arrays["r1_observation_state_cartesian_position"])
    u = arrays["r1_action_cartesian_velocity"]
    y = velocities_from_poses(poses, times)
    intervals = config["split_intervals_half_open"]
    fit = split_indices(f, intervals["fit"])
    scale = np.maximum(np.std(y[fit], axis=0), 1e-5)
    command_scale = np.std(u[fit], axis=0)
    dt_command = float(np.median(np.diff(times)[fit]))
    warmup = settings["warmup_transitions"]
    val_labels = np.flatnonzero((f >= intervals["validation"][0] + warmup) & (f < intervals["validation"][1]))
    candidates = []
    models = []
    for structure in settings["structures"]:
        for history in settings["history_steps"]:
            for lag in settings["lag_steps"]:
                ids = split_indices(f, intervals["fit"], history=history, lag=lag)
                for ridge in settings["ridge_grid"]:
                    model = LinearResponse.fit(u, y, ids, lag=lag, history=history, structure=structure, ridge=ridge)
                    model.response_scale = scale.copy()
                    val = evaluate_one_step(model, u, y, val_labels - lag)
                    candidates.append(
                        {
                            "structure": structure,
                            "history": history,
                            "lag": lag,
                            "ridge": ridge,
                            "fit_samples": len(ids),
                            "validation": val,
                        }
                    )
                    models.append(model)
    # Fixed ordering resolves ties conservatively: diagonal, shorter history/lag, stronger ridge.
    best_index = min(range(len(candidates)), key=lambda i: candidates[i]["validation"]["balanced_nrmse"])
    model = models[best_index]
    partitions = {}
    replay_arrays = {}
    for name in ["validation", "test"]:
        lo, hi = intervals[name]
        labels = np.flatnonzero((f >= lo + warmup) & (f < hi))
        ids = labels - model.lag
        metrics = evaluate_one_step(model, u, y, ids)
        zero = prediction_metrics(np.zeros_like(y[labels]), y[labels], scale)
        persistence = prediction_metrics(y[labels - 1], y[labels], scale)
        loc = np.flatnonzero((f >= lo) & (f <= hi))
        replay, predicted, motion = replay_metrics(model, u[loc], y[loc], poses[loc], times[loc], warmup)
        partitions[name] = {
            "one_step": metrics,
            "zero_velocity": zero,
            "previous_velocity": persistence,
            "logged_command_rollout": replay,
            "lag_ms_min_median_max": [
                float(v * 1000)
                for v in (
                    np.min(times[labels] - times[ids]),
                    np.median(times[labels] - times[ids]),
                    np.max(times[labels] - times[ids]),
                )
            ],
        }
        replay_arrays[name + "_predicted_pose"] = predicted
        replay_arrays[name + "_predicted_velocity"] = motion
        replay_arrays[name + "_frame_index"] = f[loc]
    v = partitions["validation"]
    gates = {
        "one_step_beats_zero": v["one_step"]["balanced_nrmse"] < v["zero_velocity"]["balanced_nrmse"],
        "one_step_beats_previous_motion": v["one_step"]["balanced_nrmse"] < v["previous_velocity"]["balanced_nrmse"],
        "rollout_velocity_beats_zero": v["logged_command_rollout"]["velocity"]["balanced_nrmse"]
        < v["zero_velocity"]["balanced_nrmse"],
        "rollout_position_beats_hold": v["logged_command_rollout"]["pose"]["position_rmse_m"]
        < v["logged_command_rollout"]["hold_pose"]["position_rmse_m"],
        "rollout_rotation_beats_hold": v["logged_command_rollout"]["pose"]["rotation_rmse_rad"]
        < v["logged_command_rollout"]["hold_pose"]["rotation_rmse_rad"],
        "B_condition_acceptable": model.diagnostics["B_scaled_condition"] < settings["max_B_condition"],
    }
    iface = DroidReferenceInterface(**config["reference_interface"]["parameters"])
    commands = {}
    inverse_lambda = settings["inverse_regularization"]
    for name in ["fit", "validation", "test"]:
        ids = split_indices(f, intervals[name], history=max(settings["history_steps"]), lag=max(settings["lag_steps"]))
        a = iface.encode_target(poses[ids + 1], poses[ids])
        unlimited = np.array(
            [
                model.inverse(y[i + model.lag], history_vector(u, y, i, model.history), prior, inverse_lambda)
                for i, prior in zip(ids, a, strict=True)
            ]
        )
        calibrated = iface.limit(unlimited)
        moving = (np.linalg.norm(y[ids, :3], axis=1) > settings["moving_translation_m_s"]) | (
            np.linalg.norm(y[ids, 3:], axis=1) > settings["moving_rotation_rad_s"]
        )
        commands[name] = {
            key: command_metrics(p, u[ids], command_scale, moving)
            for key, p in {
                "hold_zero": np.zeros_like(a),
                "previous_command": u[ids - 1],
                "analytic_A": a,
                "linear_calibrated": calibrated,
            }.items()
        }
        commands[name]["inverse_projected_fraction"] = float(
            np.mean(np.any(np.abs(unlimited - calibrated) > 1e-10, axis=1))
        )
    params = {
        "model": model.as_dict(),
        "selection": "validation one-step balanced NRMSE on common label frames",
        "inverse_regularization": inverse_lambda,
        "inverse_constraints": "unconstrained solve then explicit vector projection; not constrained optimum",
        "dt_command_planned_s": dt_command,
        "lag_for_online_s": model.lag * dt_command,
        "gates": gates,
        "response_gate_passed": all(gates.values()),
        "config": settings,
    }
    metrics = {
        "evidence_level": "held_out_log_response_and_local_model_replay",
        "candidates": candidates,
        "selected_candidate_index": best_index,
        "forward": partitions,
        "command_diagnostics": commands,
        "response_gate_passed": all(gates.values()),
        "gates": gates,
        "contact_segmentation": "not available; no verified contact labels",
        "oracle_reference": True,
        "fit_only_response_scale": scale,
        "fit_only_command_scale": command_scale,
    }
    write_json(out / "adapter_params.json", params)
    write_json(out / "calibration_metrics.json", metrics)
    np.savez_compressed(out / "forward_replay.npz", **replay_arrays)
    return model, params, metrics
