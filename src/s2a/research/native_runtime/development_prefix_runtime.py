"""One bounded OSMesa saved-prefix replay with explicit lifecycle accounting."""

import os
import random
import sys
import tempfile
import xml.etree.ElementTree as ET
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from s2a.research.native_runtime import bootstrap
from s2a.research.native_runtime.bootstrap_v2 import _private_asset_records
from s2a.research.native_runtime.development_prefix_replay import ReplayObservers, replay_saved_prefix


class PrefixReplayFailure(RuntimeError):
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


def run_prefix_runtime(
    config,
    record,
    record_xml,
    commands,
    saved,
    receipts,
    output,
    asset_refs,
    controller,
    *,
    continuation=None,
    continuation_budget=0,
    continuation_limit=8,
    progress=None,
):
    """Use only a saved prefix; failed stages and partial integration remain visible."""
    count = len(commands)
    if (
        type(continuation_limit) is not int
        or continuation_limit not in (8, 16)
        or type(continuation_budget) is not int
        or (continuation is None and continuation_budget != 0)
        or (continuation is not None and not 0 < continuation_budget <= continuation_limit)
    ):
        raise ValueError("Explicit bounded continuation budget required")
    limits = {
        "configure": 1,
        "model_construct": 3,
        "context": 3,
        "mj_step": 25 * (count + continuation_budget),
        "mj_step2": 0,
    }
    receipt = {
        "complete": False,
        "passed": False,
        "control_qualified": False,
        "historical_runtime_equivalence_claimed": False,
        "initial_joint_historical_value_available": False,
        "limits": limits,
        "counters": {key: {"attempted": 0, "completed": 0} for key in limits},
        "model_records": [],
        "world_model_calls": 0,
        "fresh_cases": 0,
        "newly_selected_commands": 0,
        "automatic_retry": False,
        "cleanup_attempted": False,
        "cleanup_complete": False,
    }
    output = Path(output)
    env, partial = None, None
    stage = "admission"
    active, closing = [False], [False]

    def guard(event, args):
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
        bootstrap._require(counter["attempted"] <= limits[name], f"{name} budget exhausted")
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
        bootstrap._require(config["seed"] == 43 and config["task_id"] in (8, 9), "Fixed seed and development tasks")
        bootstrap._require(
            config["camera_observations"] is False
            and config["refresh_controller_after_restore"] is True
            and config["render_resolution"] == 256,
            "Original camera and controller refresh contract",
        )
        output = bootstrap._safe_path(output, (bootstrap.REPO / "outputs",), exists=False)
        bootstrap._require(not output.exists(), "No overwrite or automatic retry")
        roots = (
            bootstrap.REPO / "outputs/native_runtime_path_snapshot_20261029/LIBERO/libero/libero/assets",
            bootstrap.REPO / "outputs/libero_env311/lib/python3.11/site-packages/robosuite/models/assets",
        )
        _private_asset_records(record_xml, roots, asset_refs)
        bootstrap._require(
            bootstrap._xml_structure(ET.fromstring(record_xml))
            == bootstrap._xml_structure(ET.fromstring(record["model_xml"])),
            "Record XML changed outside asset paths",
        )
        initial = np.asarray(record["initial_state"], dtype=np.float64)
        bootstrap._require(
            initial.shape == (79,) and bootstrap._hash(initial.tobytes()) == record["initial_state_sha256"],
            "Original initial state identity",
        )
        np.testing.assert_array_equal(initial, saved["sim_states"][0])
        output.mkdir()
        (output / "tmp").mkdir()
        (output / "cache").mkdir()
        tempfile.tempdir = str(output / "tmp")
        os.environ.update(
            TMPDIR=str(output / "tmp"),
            XDG_CACHE_HOME=str(output / "cache"),
            MUJOCO_GL="osmesa",
            PYOPENGL_PLATFORM="osmesa",
        )
        sys.addaudithook(guard)
        active[0] = True
        stage = "configure"
        runtime = counted("configure", bootstrap._default_runtime, config, output)
        task = runtime.benchmark.get_benchmark_dict()[config["task_suite"]]().get_task(config["task_id"])
        bddl = bootstrap._safe_path(
            Path(runtime.get_path("bddl_files")) / task.problem_folder / task.bddl_file, (bootstrap.REPO,)
        )
        receipt["bddl_ref"] = bootstrap._reference(bddl)
        with ExitStack() as stack:
            original_compile = runtime.sim_class.from_xml_string

            def compile_model(_cls, xml):
                assets = _private_asset_records(xml, roots, asset_refs)
                model = {
                    "stage": stage,
                    "xml_sha256": bootstrap._hash(xml.encode()),
                    "assets": assets,
                    "completed": False,
                }
                receipt["model_records"].append(model)
                result = counted("model_construct", original_compile, xml)
                model["completed"] = True
                return result

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
                    bootstrap._require(
                        stage in ("replay", "continuation") and not closing[0], "Integration outside registered work"
                    )
                    return counted(_name, _original, *args, **kwargs)

                bootstrap._patch(stack, runtime.mujoco, name, step)
            for cls in (runtime.env_class, runtime.env_base_class):
                original = cls.step

                def control_step(*args, _original=original, **kwargs):
                    bootstrap._require(
                        stage in ("replay", "continuation") and not closing[0], "Dispatch outside registered work"
                    )
                    return _original(*args, **kwargs)

                bootstrap._patch(stack, cls, "step", control_step)
            original_inner = runtime.env_base_class.__init__

            def initialize_inner(instance, *args, **kwargs):
                nonlocal partial
                partial = instance
                return original_inner(instance, *args, **kwargs)

            bootstrap._patch(stack, runtime.env_base_class, "__init__", initialize_inner)
            stack.callback(cleanup)
            random.seed(config["seed"])
            np.random.seed(config["seed"])
            stage = "constructor"
            env = runtime.env_class(
                bddl_file_name=str(bddl),
                camera_heights=256,
                camera_widths=256,
                render_gpu_device_id=-1,
                use_camera_obs=False,
            )
            env.seed(config["seed"])
            stage = "hard_reset"
            env.env.reset()
            stage = "record_xml"
            env.reset_from_xml_string(record_xml)
            stage = "restore_state"
            env.set_init_state(initial.copy())
            env.robots[0].controller.update(force=True)
            bootstrap._require(bootstrap._state(env).tobytes() == initial.tobytes(), "Exact original initial state")
            current_contract = bootstrap._controller_check(runtime, env)
            bootstrap._require(
                {key: current_contract[key] for key in controller["contract"]} == controller["contract"],
                "Original controller contract",
            )
            receipt["controller_contract"] = current_contract
            receipt["runtime_initial_joint"] = np.asarray(env.robots[0].controller.initial_joint).tolist()
            renderer = runtime.renderer_info(env)
            bootstrap._require(
                renderer["backend"] == "osmesa"
                and any(x in renderer["renderer"].lower() for x in ("llvmpipe", "softpipe", "swrast")),
                "Software OSMesa required",
            )
            receipt["renderer"] = renderer
            from s2a.backends.libero.cameras import render_current_cameras
            from s2a.backends.libero.simulation import current_observation
            from s2a.research.gripper.gripper_motor_state import preserve_motor_target
            from s2a.research.wam.training.native_observations import endpoint, render_anchor

            robot = env.robots[0]

            def initialize_gripper(_env):
                robot.gripper.current_action = preserve_motor_target(robot.gripper.current_action)
                np.testing.assert_allclose(
                    robot.gripper.current_action, controller["aligned_initial_motor_target"], atol=1e-10, rtol=0
                )

            stage = "replay"
            observers = ReplayObservers(endpoint, current_observation, render_anchor, render_current_cameras)
            terminal = {}

            def save_terminal(sample, endpoint_receipt):
                terminal.update(sample=sample, endpoint_receipt=endpoint_receipt)

            replay_saved_prefix(
                env,
                commands,
                saved,
                receipts,
                output,
                observers,
                receipt,
                initialize=initialize_gripper,
                on_terminal=save_terminal,
                progress=progress,
            )
            receipt["prefix_control_steps"] = count
            if continuation is not None:
                bootstrap._require(bool(terminal), "Terminal endpoint must be captured before continuation")
                stage = "continuation"
                receipt["continuation"] = continuation(env, terminal, observers, receipt, progress=progress)
                receipt.pop("continuation_partial", None)
                bootstrap._require(
                    receipt["control_steps_completed"] <= count + continuation_budget,
                    "Continuation control budget exceeded",
                )
            bootstrap._require(
                receipt["counters"]["mj_step"]["completed"] == receipt["control_steps_completed"] * 25,
                "Complete 25 inner steps per completed command",
            )
            receipt["endpoint_controller"] = {
                name: np.asarray(getattr(robot.controller, name)).tolist()
                for name in ("initial_joint", "goal_pos", "goal_ori", "joint_pos", "joint_vel")
            }
            receipt["endpoint_gripper_cache"] = np.asarray(robot.gripper.current_action).tolist()
            receipt["previous_saved_command"] = np.asarray(commands[-1]).tolist()
            stage = "cleanup"
            cleanup()
            bootstrap._require(receipt["cleanup_complete"], "Runtime cleanup")
        receipt["loaded_libraries"] = bootstrap._mapped_libraries()
        receipt["loaded_sources"] = bootstrap._loaded_sources()
        osmesa = bootstrap._reference(config["osmesa_library"])
        loaded = {item["path"]: item["sha256"] for item in receipt["loaded_libraries"]}
        bootstrap._require(loaded.get(osmesa["path"]) == osmesa["sha256"], "Registered software renderer loaded")
        bootstrap._require(bootstrap._reference(bddl) == receipt["bddl_ref"], "Task BDDL changed during replay")
        for name in ("model_construct", "context"):
            bootstrap._require(receipt["counters"][name]["completed"] == 3, "Original three-stage initialization")
        bootstrap._cpu_environment()
        receipt.update(complete=True, passed=True, behavioural_replay_verified=True)
        return receipt
    except BaseException as error:
        receipt.update(failure_stage=stage, error={"type": type(error).__name__, "message": str(error)})
        raise PrefixReplayFailure(str(error), receipt) from error
    finally:
        active[0] = False
        if progress is not None:
            progress(receipt)
