"""Replay an already executed command branch in the migrated CPU runtime.

This establishes only bounded behavioural regression evidence. No planner is
loaded, no reference is regenerated, and no saved future state is restored.
The initialization helpers below reuse the frozen bootstrap's path/controller
checks without changing its zero-step entry point or its source files.
"""

import os
import random
import sys
import tempfile
import xml.etree.ElementTree as ET
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from s2a import native_runtime_bootstrap as bootstrap
from s2a.native_runtime_bootstrap_v2 import _private_asset_records

REPO = bootstrap.REPO
PREFIX = 368
STEPS = 377
INNER_PER_STEP = 25
ATOL = 1e-10
LIMITS = dict(configure=1, model_construct=3, context=3, control_step=377, mj_step=9425, mj_step2=0)


class ReplayFailure(RuntimeError):
    """An unsuccessful attempt retaining all completed and attempted work."""

    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


@dataclass
class ReplayObservations:
    endpoint: object
    render_anchor: object
    render_cameras: object
    current: object
    source_pose: object
    object_progress: object
    current_dynamics: object = None


def validate_saved_branch(commands, states, periods):
    commands, states = np.asarray(commands), np.asarray(states)
    bootstrap._require(commands.shape == (STEPS, 7) and states.shape == (STEPS + 1, 79), "Exact archived branch shape")
    bootstrap._require(np.isfinite(commands).all() and np.isfinite(states).all(), "Finite archived branch")
    bootstrap._require(np.max(abs(commands)) <= 1 + 1e-12, "Archived command box")
    bootstrap._require(
        np.max(abs(np.diff(np.vstack((np.zeros(7), commands)), axis=0)[:, :6])) <= 0.25 + 1e-12,
        "Archived arm rate",
    )
    np.testing.assert_array_equal(commands[PREFIX:, 6], np.repeat(commands[PREFIX - 1, 6], STEPS - PREFIX))
    bootstrap._require(len(periods) == STEPS - PREFIX, "Nine archived periods")
    for index, period in enumerate(periods):
        bootstrap._require(period["step"] == PREFIX + index and period["cursor"] == index, "Consecutive archived phase")
        np.testing.assert_array_equal(period["command"], commands[PREFIX + index])
        actual = period["actual_inner"]
        bootstrap._require(len(actual) == INNER_PER_STEP, "25 archived inner samples")
        for key, shape in (("q", (25, 7)), ("qvel", (25, 7)), ("time", (25,))):
            value = np.asarray([item[key] for item in actual])
            bootstrap._require(value.shape == shape and np.isfinite(value).all(), "Finite complete inner samples")


def replay_commands(env, commands, states, periods, targets, tool, output, observations, receipt, *, progress=None):
    """Use saved commands only; archived future values are comparison targets.

    The current state is never restored during replay. The observation schedule
    is the historical checker schedule, including its extra endpoint forward.
    Caller observes native mj_step separately, including partially failed steps.
    """
    from s2a.geometry import pose_error
    from s2a.gripper_motor_state import preserve_motor_target

    validate_saved_branch(commands, states, periods)
    commands, states = np.asarray(commands), np.asarray(states)
    targets, tool = np.asarray(targets), np.asarray(tool)
    bootstrap._require(targets.shape == (40, 4, 4) and tool.shape == (4, 4), "Original reference and TCP layout")
    robot = env.robots[0]
    robot.gripper.current_action = preserve_motor_target(robot.gripper.current_action)
    limits = np.asarray([env.sim.model.jnt_range[env.sim.model.joint_name2id(f"robot0_joint{k}")] for k in range(1, 8)])
    profile = np.zeros((25, 7))
    inner, actual_states, metrics = [], [], []
    receipt.update(
        checked_endpoints=0,
        checked_inner_samples=0,
        last_checked_endpoint=None,
        maximum_state_difference=0.0,
        maximum_inner_q_difference=0.0,
        maximum_inner_qvel_difference=0.0,
        metrics=metrics,
    )
    observations.current(env)
    original_step = env.sim.step

    def observed_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        inner.append(
            dict(
                time=float(env.sim.data.time),
                q=np.asarray(env.sim.data.qpos)[robot._ref_joint_pos_indexes].copy(),
                qvel=np.asarray(env.sim.data.qvel)[robot._ref_joint_vel_indexes].copy(),
            )
        )
        return result

    primary_error = None
    try:
        for step in range(STEPS + 1):
            receipt["current_endpoint"] = step
            sample, endpoint_receipt = observations.endpoint(env, step)
            state = bootstrap._state(env)
            actual_states.append(state)
            difference = float(np.max(abs(state - states[step])))
            receipt["maximum_state_difference"] = max(receipt["maximum_state_difference"], difference)
            np.testing.assert_allclose(state, states[step], atol=ATOL, rtol=0)
            receipt["checked_endpoints"] += 1
            receipt["last_checked_endpoint"] = step
            if step == PREFIX:
                # This additional result is needed before reusing a frozen
                # actual368 numerical candidate; tolerance agreement is weaker.
                receipt["actual368_state_exact"] = state.tobytes() == states[step].astype(np.float64).tobytes()
                receipt["actual368_controller"] = {
                    key: np.asarray(getattr(robot.controller, key)).copy()
                    for key in ("initial_joint", "goal_pos", "goal_ori", "joint_pos", "joint_vel")
                }
                receipt["actual368_motor_cache"] = np.asarray(robot.gripper.current_action).copy()
                receipt["actual368_previous_command"] = commands[step - 1].copy()
                if observations.current_dynamics is not None:
                    current = observations.current_dynamics(env, commands[step - 1])
                    expected = periods[0]["current_dynamics"]
                    bootstrap._require(current.keys() == expected.keys(), "Original current dynamics fields")
                    for key, value in current.items():
                        if isinstance(value, (str, bool)):
                            bootstrap._require(value == expected[key], f"Current dynamics {key} changed")
                        else:
                            np.testing.assert_allclose(value, expected[key], atol=ATOL, rtol=0)
                    receipt["actual368_current_dynamics"] = current
                    receipt["actual368_joint_state_exact"] = all(
                        np.array_equal(current[key], expected[key])
                        for key in ("robot_joint_position", "robot_joint_velocity")
                    )
            if step in (0, 16, 56):
                observations.render_anchor(env, step, sample, endpoint_receipt, output)
            if progress is not None:
                progress(receipt)
            if step == STEPS:
                break
            if step % 8 == 0:
                observations.render_cameras(env, 256)
            inner.clear()
            if step >= PREFIX:
                env.sim.step = observed_step
            count = receipt["counters"]["control_step"]
            count["attempted"] += 1
            bootstrap._require(count["attempted"] <= STEPS, "Control replay budget exhausted")
            env.step(commands[step].copy())
            count["completed"] += 1
            env.sim.step = original_step
            obs = observations.current(env)
            if step < PREFIX:
                continue
            period = periods[step - PREFIX]
            bootstrap._require(len(inner) == 25, "Exactly 25 completed inner steps")
            q = np.asarray([item["q"] for item in inner])
            for key in ("q", "qvel", "time"):
                actual = np.asarray([item[key] for item in inner])
                expected = np.asarray([item[key] for item in period["actual_inner"]])
                difference = float(np.max(abs(actual - expected)))
                if key != "time":
                    label = f"maximum_inner_{key}_difference"
                    receipt[label] = max(receipt[label], difference)
                np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=0)
            receipt["checked_inner_samples"] += 25
            np.testing.assert_allclose(profile, period["history_before"], atol=ATOL, rtol=0)
            profile = np.maximum.accumulate(np.maximum(profile, abs(q - np.asarray(period["predicted_inner"]))), axis=0)
            np.testing.assert_allclose(profile, period["history_after"], atol=ATOL, rtol=0)
            error = pose_error(targets[step - PREFIX], observations.source_pose(obs) @ tool)
            margin = float(np.minimum(q - limits[:, 0], limits[:, 1] - q).min())
            np.testing.assert_allclose(error, period["actual_tcp_error"], atol=ATOL, rtol=0)
            np.testing.assert_allclose(margin, period["minimum_inner_margin_rad"], atol=ATOL, rtol=0)
            task = observations.object_progress(env)
            success, grasp = bool(task["success"]), bool(task["objects"]["akita_black_bowl_1"]["grasp_contact"])
            bootstrap._require(success == period["success"] and grasp == period["grasp"], "Archived task predicates")
            good = margin >= 0.05 and np.linalg.norm(error[:3]) <= 0.01 and np.linalg.norm(error[3:]) <= 0.05
            metrics.append(
                dict(
                    step=step,
                    cursor=step - PREFIX,
                    actual_tcp_error=error,
                    minimum_inner_margin_rad=margin,
                    original_caps_satisfied=bool(good),
                    success=success,
                    grasp=grasp,
                    actual_inner=list(inner),
                )
            )
            bootstrap._require(good, "Original actual caps violated; stop before next dispatch")
    except BaseException as error:
        primary_error = error
        raise
    finally:
        env.sim.step = original_step
        receipt["partial_period_inner_samples"] = list(inner)
        if actual_states:
            try:
                with (Path(output) / "states.npz").open("xb") as stream:
                    np.savez_compressed(stream, sim_states=np.asarray(actual_states))
            except BaseException as error:
                receipt["states_archive_error"] = dict(type=type(error).__name__, message=str(error))
                if primary_error is None:
                    raise
    bootstrap._require(
        receipt["checked_endpoints"] == 378 and receipt["checked_inner_samples"] == 225, "Full archived replay"
    )


def _observations():
    from collect_wam_native_training import endpoint, render_anchor

    from s2a.cameras import render_current_cameras
    from s2a.checkpoint_simulation import object_progress
    from s2a.current_coupled_state import read_current_dynamics
    from s2a.libero_simulation import current_observation, measured_twist, source_pose

    def dynamics(env, previous):
        robot, controller = env.robots[0], env.robots[0].controller
        site = np.eye(4)
        site[:3, :3] = env.sim.data.get_site_xmat(controller.eef_name)
        site[:3, 3] = env.sim.data.get_site_xpos(controller.eef_name)
        current = dict(
            x_joint_position=[np.asarray(env.sim.data.qpos)[robot._ref_joint_pos_indexes]],
            x_joint_velocity=[np.asarray(env.sim.data.qvel)[robot._ref_joint_vel_indexes]],
            x_site_pose=[site],
            x_site_twist=[measured_twist(env)],
            x_previous_goal_ori=[np.asarray(controller.goal_ori).copy()],
        )
        return read_current_dynamics(env, current, previous)

    return ReplayObservations(
        endpoint, render_anchor, render_current_cameras, current_observation, source_pose, object_progress, dynamics
    )


def run_native_replay(
    config,
    record,
    record_xml,
    commands,
    states,
    periods,
    targets,
    tool,
    output,
    *,
    allowed_asset_roots,
    asset_refs,
    expected_controller=None,
    expected_initial_joint=None,
    runtime_factory=None,
    observation_factory=None,
    progress=None,
):
    """Construct once and replay one archived branch, preserving failed costs."""
    receipt = dict(
        schema_version="s2a.native-runtime-replay.v1",
        complete=False,
        passed=False,
        runtime_loaded=False,
        control_qualified=False,
        historical_runtime_equivalence_claimed=False,
        limits=dict(LIMITS),
        counters={key: dict(attempted=0, completed=0) for key in LIMITS},
        model_records=[],
        world_model_calls=0,
        fresh_cases=0,
        newly_selected_commands=0,
        automatic_retry=False,
        cleanup_attempted=False,
        cleanup_complete=False,
        forbidden_control_attempts=0,
        integrated_time_seconds=0.0,
    )
    env, partial = None, None
    output = Path(output)
    stage = "admission"
    active = [False]
    closing = [False]

    def file_guard(event, args):
        if not active[0]:
            return
        writes = []
        if event == "open":
            path, mode, flags = args
            if isinstance(path, int):
                path = os.readlink(f"/proc/self/fd/{path}")
                if not path.startswith("/"):
                    return
            bootstrap._require(not os.fsdecode(path).startswith("/share/project/"), "Retired runtime path")
            if (mode and any(letter in mode for letter in "wax+")) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
                writes.append((path, None))
        elif event == "os.mkdir":
            writes.append((args[0], args[2]))
        elif event in ("os.remove", "os.rmdir"):
            writes.append((args[0], args[1]))
        elif event == "os.rename":
            writes.extend(((args[0], args[2]), (args[1], args[3])))
        elif event in ("os.link", "os.symlink"):
            raise ValueError("Runtime link creation forbidden")
        for path, directory_fd in writes:
            target = Path(os.fsdecode(path))
            if not target.is_absolute():
                base = Path.cwd() if directory_fd in (None, -1) else Path(os.readlink(f"/proc/self/fd/{directory_fd}"))
                target = base / target
            bootstrap._safe_path(target, (output,), exists=False)

    def counted(name, function, *args, **kwargs):
        counter = receipt["counters"][name]
        counter["attempted"] += 1
        bootstrap._require(counter["attempted"] <= LIMITS[name], f"{name} budget exhausted")
        value = function(*args, **kwargs)
        counter["completed"] += 1
        return value

    def cleanup():
        if receipt["cleanup_attempted"]:
            return
        target = env if env is not None and vars(env).get("env") is not None else partial
        if target is None:
            return
        receipt["cleanup_attempted"] = True
        closing[0] = True
        try:
            target.close()
            receipt["cleanup_complete"] = True
        except Exception as error:
            receipt["cleanup_error"] = repr(error)

    try:
        bootstrap._cpu_environment()
        bootstrap._require(
            runtime_factory is not None or (expected_controller is not None and expected_initial_joint is not None),
            "Production requires original controller and initial-joint bindings",
        )
        validate_saved_branch(commands, states, periods)
        bootstrap._require(
            config["seed"] == 42
            and config["camera_observations"] is False
            and config["refresh_controller_after_restore"] is True
            and config["render_resolution"] == 256,
            "Fixed historical runtime configuration",
        )
        output = bootstrap._safe_path(output, (REPO / "outputs",), exists=False)
        bootstrap._require(not output.exists(), "No output overwrite or automatic retry")
        roots = tuple(bootstrap._safe_path(path, (REPO,)) for path in allowed_asset_roots)
        _private_asset_records(record_xml, roots, asset_refs)
        bootstrap._require(
            bootstrap._xml_structure(ET.fromstring(record_xml))
            == bootstrap._xml_structure(ET.fromstring(record["model_xml"])),
            "Only record file paths may differ",
        )
        initial = np.asarray(record["initial_state"], dtype=np.float64)
        bootstrap._require(
            initial.shape == (79,) and bootstrap._hash(initial.tobytes()) == record["initial_state_sha256"],
            "Original initial-state identity",
        )
        np.testing.assert_array_equal(initial, np.asarray(states)[0])
        output.mkdir()
        for name in ("tmp", "cache"):
            (output / name).mkdir()
        tempfile.tempdir = str(output / "tmp")
        os.environ.update(
            TMPDIR=str(output / "tmp"),
            XDG_CACHE_HOME=str(output / "cache"),
            MUJOCO_GL="osmesa",
            PYOPENGL_PLATFORM="osmesa",
        )
        sys.addaudithook(file_guard)
        active[0] = True
        stage = "configure"
        runtime = counted("configure", runtime_factory or bootstrap._default_runtime, config, output)
        receipt["runtime_loaded"] = True
        bootstrap._cpu_environment()
        task = runtime.benchmark.get_benchmark_dict()[config["task_suite"]]().get_task(config["task_id"])
        bddl = bootstrap._safe_path(
            Path(runtime.get_path("bddl_files")) / task.problem_folder / task.bddl_file, (REPO,)
        )
        with ExitStack() as stack:
            original_compile = runtime.sim_class.from_xml_string

            def compile_model(cls, xml):
                assets = _private_asset_records(xml, roots, asset_refs)
                item = dict(stage=stage, xml_sha256=bootstrap._hash(xml.encode()), assets=assets, completed=False)
                receipt["model_records"].append(item)
                value = counted("model_construct", original_compile, xml)
                item["completed"] = True
                return value

            bootstrap._patch(stack, runtime.sim_class, "from_xml_string", classmethod(compile_model))
            original_context = runtime.context_class.__init__
            bootstrap._patch(
                stack,
                runtime.context_class,
                "__init__",
                lambda *a, **kw: counted("context", original_context, *a, **kw),
            )
            for name in ("mj_step", "mj_step2"):
                original = getattr(runtime.mujoco, name)

                def step(*args, _name=name, _original=original, **kwargs):
                    before = float(args[1].time)
                    try:

                        def invoke():
                            bootstrap._require(
                                stage == "replay" and not closing[0],
                                "No integration during initialization or cleanup",
                            )
                            return _original(*args, **kwargs)

                        return counted(_name, invoke)
                    finally:
                        after = float(args[1].time)
                        receipt["integrated_time_seconds"] += after - before
                        receipt["last_native_step"] = dict(operation=_name, before=before, after=after)

                bootstrap._patch(stack, runtime.mujoco, name, step)
            for cls in (runtime.env_class, runtime.env_base_class):
                original = cls.step

                def control_step(*args, _original=original, **kwargs):
                    if stage != "replay" or closing[0]:
                        receipt["forbidden_control_attempts"] += 1
                        raise ValueError("No control dispatch during initialization or cleanup")
                    return _original(*args, **kwargs)

                bootstrap._patch(stack, cls, "step", control_step)
            original_inner = runtime.env_base_class.__init__

            def initialize_inner(instance, *args, **kwargs):
                nonlocal partial
                partial = instance
                return original_inner(instance, *args, **kwargs)

            bootstrap._patch(stack, runtime.env_base_class, "__init__", initialize_inner)
            stack.callback(cleanup)
            random.seed(42)
            np.random.seed(42)
            stage = "constructor"
            env = runtime.env_class(
                bddl_file_name=str(bddl),
                camera_heights=256,
                camera_widths=256,
                render_gpu_device_id=-1,
                use_camera_obs=False,
            )
            env.seed(42)
            stage = "hard_reset"
            env.env.reset()
            stage = "record_xml"
            env.reset_from_xml_string(record_xml)
            stage = "restore_state"
            env.set_init_state(initial.copy())
            env.robots[0].controller.update(force=True)
            bootstrap._require(bootstrap._state(env).tobytes() == initial.tobytes(), "Exact original initial state")
            receipt["controller"] = bootstrap._controller_check(runtime, env)
            if expected_controller is not None:
                bootstrap._require(receipt["controller"] == expected_controller, "Full bootstrap controller contract")
            if expected_initial_joint is not None:
                np.testing.assert_array_equal(env.robots[0].controller.initial_joint, expected_initial_joint)
            receipt["renderer"] = runtime.renderer_info(env)
            bootstrap._require(
                receipt["renderer"]["backend"] == "osmesa"
                and any(
                    token in receipt["renderer"]["renderer"].lower() for token in ("llvmpipe", "softpipe", "swrast")
                ),
                "Software OSMesa required",
            )
            stage = "replay"
            observations = (observation_factory or _observations)()
            replay_commands(
                env, commands, states, periods, targets, tool, output, observations, receipt, progress=progress
            )
            bootstrap._require(receipt["counters"]["mj_step"]["completed"] == 9425, "Every historical 2ms integration")
            stage = "close"
            cleanup()
            bootstrap._require(receipt["cleanup_complete"], "Runtime cleanup failed")
        receipt["loaded_libraries"] = bootstrap._mapped_libraries() if runtime_factory is None else []
        receipt["loaded_sources"] = bootstrap._loaded_sources() if runtime_factory is None else []
        if runtime_factory is None:
            osmesa = bootstrap._reference(config["osmesa_library"])
            loaded = {item["path"]: item["sha256"] for item in receipt["loaded_libraries"]}
            bootstrap._require(loaded.get(osmesa["path"]) == osmesa["sha256"], "Registered OSMesa actually mapped")
        for name in ("model_construct", "context"):
            bootstrap._require(receipt["counters"][name]["completed"] == 3, "Original three-stage initialization")
        for model in receipt["model_records"]:
            for asset in model["assets"]:
                bootstrap._require(
                    bootstrap._reference(asset["reference"]["path"]) == asset["reference"], "Runtime asset changed"
                )
        bootstrap._cpu_environment()
        receipt.update(complete=True, passed=True, behavioural_replay_verified=True)
        return receipt
    except BaseException as error:
        receipt.update(failure_stage=stage, error=dict(type=type(error).__name__, message=str(error)))
        raise ReplayFailure(str(error), receipt) from error
    finally:
        active[0] = False
        if progress is not None:
            progress(receipt)
