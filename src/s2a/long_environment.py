"""Isolated017 factory: unchanged reset/restore with explicit long horizon.

No pre-roll or done-state mutation is introduced. The old factory stays frozen.
"""

import random
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from s2a.libero_simulation import configure_simulator, sha256_bytes


def validate_episode_settings(config):
    if type(config.get("episode_horizon")) is not int or config["episode_horizon"] != 1300:
        raise ValueError("017 requires explicit episode_horizon1300")
    if config.get("ignore_done") is not False:
        raise ValueError("017 preserves ignore_done=False")


def make_environment(config, output, record, trial):
    validate_episode_settings(config)
    benchmark, get_path, env_class = configure_simulator(config, output)
    task = benchmark.get_benchmark_dict()[config["task_suite"]]().get_task(config["task_id"])
    random.seed(config["seed"] + trial)
    np.random.seed(config["seed"] + trial)
    env = env_class(
        bddl_file_name=str(Path(get_path("bddl_files")) / task.problem_folder / task.bddl_file),
        camera_heights=config["render_resolution"],
        camera_widths=config["render_resolution"],
        render_gpu_device_id=-1,
        use_camera_obs=config.get("camera_observations", True),
        horizon=config["episode_horizon"],
        ignore_done=config["ignore_done"],
    )
    env.seed(config["seed"] + trial)
    env.reset()
    from libero.libero.utils.utils import postprocess_model_xml

    xml = ET.fromstring(postprocess_model_xml(record["model_xml"], {}))
    for node in xml.iter():
        file = node.get("file")
        if file and "/assets/" in file and not Path(file).is_file():
            asset = Path(get_path("assets")) / file.split("/assets/", 1)[1]
            if not asset.is_file():
                raise FileNotFoundError(asset)
            node.set("file", str(asset))
    env.reset_from_xml_string(ET.tostring(xml, encoding="unicode"))
    initial = np.asarray(record["initial_state"], dtype=float)
    if sha256_bytes(initial.tobytes()) != record["initial_state_sha256"]:
        env.close()
        raise ValueError("fixture initial state hash mismatch")
    obs = env.set_init_state(initial)
    # MuJoCo state restore does not invalidate robosuite's cached controller
    # feedback. Keep the historical protocol opt-in reproducible; new runs
    # explicitly enable refresh before the first goal is formed.
    if config.get("refresh_controller_after_restore", False):
        env.robots[0].controller.update(force=True)
    return env, obs


def episode_snapshot(env):
    """Read only actual counters; never clear done or advance physics."""
    core = env.env
    snapshot = dict(
        effective_horizon=int(core.horizon),
        ignore_done=bool(core.ignore_done),
        internal_timestep=int(core.timestep),
        internal_done=bool(core.done),
        control_freq=float(core.control_freq),
        control_timestep_s=float(core.control_timestep),
        model_timestep_s=float(core.model_timestep),
        sim_time_s=float(env.sim.data.time),
    )
    if snapshot["effective_horizon"] != 1300 or snapshot["ignore_done"]:
        raise ValueError("actual long environment settings differ")
    if snapshot["control_freq"] != 20 or not np.isclose(snapshot["control_timestep_s"], 0.05, atol=1e-12, rtol=0):
        raise ValueError("actual017 control clock differs")
    return snapshot


def episode_contract(initial, final, physical_steps):
    """Describe actual policy clocks separately from the1200 scoring budget."""
    if type(physical_steps) is not int or not 0 <= physical_steps <= 1200:
        raise ValueError("invalid observed017 physical-step count")
    observed = final["internal_timestep"] - initial["internal_timestep"]
    return dict(
        schema_version="s2a.long-episode-contract.v1",
        effective_horizon=initial["effective_horizon"],
        ignore_done=initial["ignore_done"],
        initial_timestep=initial["internal_timestep"],
        final_timestep=final["internal_timestep"],
        observed_policy_steps=observed,
        physical_steps=physical_steps,
        scored_physical_step_budget=1200,
        control_freq=initial["control_freq"],
        control_timestep_s=initial["control_timestep_s"],
        model_timestep_s=initial["model_timestep_s"],
        initial_sim_time_s=initial["sim_time_s"],
        final_sim_time_s=final["sim_time_s"],
        initialization_policy_steps_added=0,
        initialization_excluded_from_scoring=True,
        initial_internal_done=initial["internal_done"],
        final_internal_done=final["internal_done"],
        clock_consistent=bool(
            initial["effective_horizon"] == final["effective_horizon"] == 1300
            and initial["ignore_done"] is final["ignore_done"] is False
            and initial["internal_timestep"] == 0
            and observed == physical_steps
            and not initial["internal_done"]
            and not final["internal_done"]
            and np.isclose(final["sim_time_s"] - initial["sim_time_s"], physical_steps / 20, atol=1e-9, rtol=0)
        ),
        initial_observation=initial,
        final_observation=final,
    )
