"""Restore frozen native states and compare geometry/RGB without scored dynamics.

Use outputs/libero_env311/bin/python. Environment initialization is separately
counted; scored diagnostics restore state, refresh observations and render only.
"""

import argparse
import contextlib
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ORIENTATIONS = ["identity", "vertical_flip", "horizontal_flip", "both_flip"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def resolve(value):
    return (REPO / Path(value)).resolve()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    def convert(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(type(obj).__name__)

    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False, default=convert)
        stream.write("\n")


def verify_inputs(config):
    paths = {}
    for name, spec in config["inputs"].items():
        path = resolve(spec["path"])
        if sha256(path) != spec["sha256"]:
            raise ValueError(f"changed frozen input: {name}: {path}")
        paths[name] = path
    return paths


def validate(config, seed):
    if seed != 42 or config["seed"] != 42:
        raise ValueError("requires seed 42")
    if Path(sys.prefix).resolve() != resolve(Path(config["reader_python"]).parent.parent):
        raise ValueError("requires configured outputs/libero_env311/bin/python")
    fixed = dict(
        record="task1_init2",
        native_demo="demo_4",
        r1_episode=683,
        r2_episode=1139,
        frame_count=158,
        target_frames=[13, 73, 90],
        candidate_state_index_minus_target_frame=[0, 1, 2],
        resolutions=[128, 256],
        orientation_hypotheses=ORIENTATIONS,
        restore_absolute_tolerance=1e-12,
        restore_time_tolerance_s=0.0,
        expected_image_convention="opengl",
        scored_control_steps=0,
        roles=[
            dict(role="auxiliary_0", native_rgb_dataset="agentview_rgb", render_key="agentview_image"),
            dict(role="left_wrist", native_rgb_dataset="eye_in_hand_rgb", render_key="robot0_eye_in_hand_image"),
        ],
    )
    if any(config.get(key) != value for key, value in fixed.items()):
        raise ValueError("fixed static-restore protocol differs")
    destination = resolve(config["output_dir"])
    if destination != REPO / "outputs/wam_video_pose_restore_20261020":
        raise ValueError("unexpected output directory")
    if destination.exists():
        raise FileExistsError(destination)
    return destination


def load_sources(config, paths):
    import h5py
    import numpy as np

    fixtures = read_json(paths["fixtures"])["records"]
    record = next(row for row in fixtures if row["name"] == config["record"])
    source = next(row for row in read_json(paths["source_audit"])["cases"] if row["record"] == config["record"])
    if (
        record["demo"] != "demo_4"
        or resolve(record["source_hdf5"]) != paths["native_hdf5"]
        or record["source_hdf5_sha256"] != config["inputs"]["native_hdf5"]["sha256"]
        or len(source["r1_matches"]) != 1
        or source["r1_matches"][0]["r1_episode"] != 683
        or [row["r2_episode"] for row in source["r2"]] != [1139]
    ):
        raise ValueError("fixture/source identity differs")
    samples = [row for row in read_json(paths["sample_manifest"])["cases"] if row["episode"] == 1139]
    if len(samples) != 4 or any(
        resolve(row["arrays_path"]) != paths["published_npz"]
        or row["arrays_sha256"] != config["inputs"]["published_npz"]["sha256"]
        for row in samples
    ):
        raise ValueError("published image/state cache differs")
    with h5py.File(paths["native_hdf5"], "r") as stream:
        group = stream["data/demo_4"]
        xml = group.attrs["model_file"]
        convention = stream["data"].attrs["macros_image_convention"]
        if isinstance(xml, bytes):
            xml = xml.decode()
        if isinstance(convention, bytes):
            convention = convention.decode()
        if xml != record["model_xml"] or convention != "opengl":
            raise ValueError("original XML or native image convention differs")
        native = {
            key: group["obs"][key][:]
            for key in ("ee_pos", "ee_ori", "joint_states", "gripper_states", "agentview_rgb", "eye_in_hand_rgb")
        }
        native["states"] = group["states"][:]
    if native["states"].shape != (158, 79) or not np.isfinite(native["states"]).all():
        raise ValueError("incomplete/nonfinite native flattened states")
    for role in config["roles"]:
        pixels = native[role["native_rgb_dataset"]]
        if pixels.shape != (158, 128, 128, 3) or pixels.dtype != np.uint8:
            raise ValueError("native RGB shape/dtype differs")
    with np.load(paths["published_npz"], allow_pickle=False) as stream:
        published = {key: stream[key] for key in ("states", "actions", "auxiliary_0", "left_wrist")}
    for role in config["roles"]:
        pixels = published[role["role"]]
        if pixels.shape != (158, 256, 256, 3) or pixels.dtype != np.uint8:
            raise ValueError("published RGB shape/dtype differs")
    if published["states"].shape != (158, 128) or published["actions"].shape != (158, 128):
        raise ValueError("published state/action shape differs")
    return record, native, published


def asset_fingerprints(record):
    from libero.libero import get_libero_path
    from libero.libero.utils.utils import postprocess_model_xml

    # Match make_environment's exact order: LIBERO first relocates robosuite
    # assets, then S2A resolves remaining obsolete LIBERO asset paths.
    assets = Path(get_libero_path("assets"))
    xml = ET.fromstring(postprocess_model_xml(record["model_xml"], {}))
    result = {}
    for node in xml.iter():
        original = node.get("file")
        if original is None:
            continue
        path = Path(original)
        if not path.is_file() and "/assets/" in original:
            path = assets / original.split("/assets/", 1)[1]
            node.set("file", str(path))
        if not path.is_file():
            raise FileNotFoundError(path)
        result[str(path.resolve())] = sha256(path)
    return dict(
        assets_sha256=result,
        processed_xml_sha256=hashlib.sha256(ET.tostring(xml, encoding="unicode").encode()).hexdigest(),
        original_xml_sha256=hashlib.sha256(record["model_xml"].encode()).hexdigest(),
    )


@contextlib.contextmanager
def guard_dynamics():
    import mujoco

    phase = ["setup"]
    counts = {name: {key: 0 for key in ("mj_step", "mj_step1", "mj_step2")} for name in ("setup", "static")}
    original = {name: getattr(mujoco, name) for name in counts["setup"]}

    def wrap(name):
        def call(*args, **kwargs):
            counts[phase[0]][name] += 1
            if phase[0] != "setup":
                raise RuntimeError(f"forbidden dynamics call during static restoration: {name}")
            return original[name](*args, **kwargs)

        return call

    for name in original:
        setattr(mujoco, name, wrap(name))
    try:
        yield phase, counts
    finally:
        for name, function in original.items():
            setattr(mujoco, name, function)


def snapshot(env):
    state = env.sim.get_state()
    flat = state.flatten()
    return dict(
        time_s=float(state.time), qpos=state.qpos.copy(), qvel=state.qvel.copy(), flat=flat, sha256=array_sha(flat)
    )


def restore_check(expected, actual, env, config):
    import numpy as np

    nq, nv = env.sim.model.nq, env.sim.model.nv
    difference = actual["flat"] - expected
    return dict(
        exact=bool(np.array_equal(expected, actual["flat"])),
        passed=bool(
            np.max(abs(difference)) <= config["restore_absolute_tolerance"]
            and abs(actual["time_s"] - expected[0]) <= config["restore_time_tolerance_s"]
        ),
        full_state_max_abs=float(np.max(abs(difference))),
        qpos_max_abs=float(np.max(abs(actual["qpos"] - expected[1 : 1 + nq]))),
        qvel_max_abs=float(np.max(abs(actual["qvel"] - expected[1 + nq : 1 + nq + nv]))),
        time_error_s=float(actual["time_s"] - expected[0]),
    )


def observation_raw(observation):
    return {
        key: observation[key].copy()
        for key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_joint_pos", "robot0_gripper_qpos")
    }


def label_rows(native, published, index):
    fingers = native["gripper_states"][index]
    return dict(
        index=index,
        HDF_obs={key: native[key][index] for key in ("ee_pos", "ee_ori", "joint_states", "gripper_states")},
        R2_state=published["states"][index],
        R2_action=published["actions"][index],
        HDF_normalized_aperture=float(min(1, max(0, (fingers[0] - fingers[1] - 0.001) / 0.079))),
        R2_current_aperture=float(published["states"][index, 9]),
        R2_next_endpoint_action_aperture=float(published["actions"][index, 9]),
    )


def pose_comparison(reference, actual):
    import numpy as np

    from s2a.geometry import pose_error

    error = pose_error(reference, actual)
    return dict(
        position_error_m=float(np.linalg.norm(error[:3])),
        rotation_error_rad=float(np.linalg.norm(error[3:])),
        error6=error,
    )


def orientation(image, name):
    if name == "identity":
        return image
    if name == "vertical_flip":
        return image[::-1]
    if name == "horizontal_flip":
        return image[:, ::-1]
    if name == "both_flip":
        return image[::-1, ::-1]
    raise ValueError(name)


def image_metrics(reference, actual):
    import numpy as np

    delta = actual.astype(np.float64) - reference
    mse = float(np.mean(delta * delta))
    return dict(
        mae_rgb_0_255=float(np.mean(abs(delta))),
        mse_rgb_0_255=mse,
        psnr_db=None if mse == 0 else 10 * math.log10(255**2 / mse),
        pixel_identical=mse == 0,
    )


def save_image(path, image):
    from PIL import Image

    with path.open("xb") as stream:
        Image.fromarray(image).save(stream, format="PNG")
    return dict(
        path=str(path),
        file_sha256=sha256(path),
        pixel_sha256=array_sha(image),
        shape=list(image.shape),
        dtype=str(image.dtype),
    )


def save_panel(path, rendered, references, roles, resolution):
    from PIL import Image, ImageDraw

    panel = Image.new("RGB", (5 * resolution, 2 * (resolution + 24)), "white")
    draw = ImageDraw.Draw(panel)
    for row, role in enumerate(roles):
        key = role["render_key"]
        images = [references[key]] + [orientation(rendered[key], name) for name in ORIENTATIONS]
        for column, (title, image) in enumerate(zip(["reference"] + ORIENTATIONS, images, strict=True)):
            top = row * (resolution + 24)
            draw.text((column * resolution + 2, top + 4), f"{role['role']} {title}", fill="black")
            panel.paste(Image.fromarray(image), (column * resolution, top + 24))
    with path.open("xb") as stream:
        panel.save(stream, format="PNG")


def run_state(env, config, native, published, tool, resolution, frame, offset, destination):
    import numpy as np
    from scipy.spatial.transform import Rotation

    from s2a.cameras import render_current_cameras
    from s2a.geometry import pose9_to_transform, transform
    from s2a.libero_simulation import aperture, current_observation, source_pose

    index = frame + offset
    requested = native["states"][index].copy()
    if env.sim.model.na != 0 or requested.shape != (1 + env.sim.model.nq + env.sim.model.nv,):
        raise ValueError("flat state must contain exactly time/qpos/qvel with na=0")
    initial_obs = env.set_init_state(requested.copy())
    after_set = snapshot(env)
    env.sim.forward()
    obs = current_observation(env)
    before_render = snapshot(env)
    robot = env.robots[0]
    direct_source = transform(
        env.sim.data.site_xpos[robot.eef_site_id].copy(), env.sim.data.get_body_xmat(robot.robot_model.eef_name).copy()
    )
    actual_tcp = direct_source @ tool
    rendered = render_current_cameras(env, resolution)
    after_render = snapshot(env)
    checks = {
        name: restore_check(requested, value, env, config)
        for name, value in (
            ("after_set_init_state", after_set),
            ("after_forward_current_observation", before_render),
            ("after_render", after_render),
        )
    }
    rows = {
        name: label_rows(native, published, idx)
        for name, idx in (
            ("target_frame", frame),
            ("stored_state_index", index),
            ("stored_state_index_minus_one", index - 1),
        )
    }
    comparisons = {}
    for name, values in rows.items():
        label = values["HDF_obs"]
        hdf_tcp = transform(label["ee_pos"], Rotation.from_rotvec(label["ee_ori"]).as_matrix()) @ tool
        r2_tcp = pose9_to_transform(values["R2_state"][:9])
        comparisons[name] = dict(
            HDF_TCP=hdf_tcp,
            R2_TCP=r2_tcp,
            actual_vs_HDF=pose_comparison(hdf_tcp, actual_tcp),
            actual_vs_R2=pose_comparison(r2_tcp, actual_tcp),
        )
    references = {
        role["render_key"]: (
            native[role["native_rgb_dataset"]][frame] if resolution == 128 else published[role["role"]][frame]
        )
        for role in config["roles"]
    }
    image_rows, images = [], {}
    for role in config["roles"]:
        key = role["render_key"]
        if rendered[key].shape != (resolution, resolution, 3) or rendered[key].dtype != np.uint8:
            raise ValueError("unexpected render shape/dtype")
        images[key] = dict(
            rendered=save_image(destination / f"{role['role']}_render.png", rendered[key]),
            reference=save_image(destination / f"{role['role']}_reference.png", references[key]),
        )
        for hypothesis in ORIENTATIONS:
            image_rows.append(
                dict(
                    role=role["role"],
                    orientation_hypothesis=hypothesis,
                    reference_frame=frame,
                    stored_state_index=index,
                    **image_metrics(references[key], orientation(rendered[key], hypothesis)),
                )
            )
    panel = destination / "all_orientations.png"
    save_panel(panel, rendered, references, config["roles"], resolution)
    result = dict(
        target_frame=frame,
        stored_state_index=index,
        state_index_minus_target_frame=offset,
        resolution=resolution,
        requested_state=requested,
        requested_state_sha256=array_sha(requested),
        snapshots=dict(after_set_init_state=after_set, before_render=before_render, after_render=after_render),
        restore_checks=checks,
        render_preserved_flat_state_exactly=bool(np.array_equal(before_render["flat"], after_render["flat"])),
        set_init_observation=observation_raw(initial_obs),
        current_observation=observation_raw(obs),
        direct_sim_source=direct_source,
        direct_sim_TCP=actual_tcp,
        observable_source=source_pose(obs),
        observable_TCP=source_pose(obs) @ tool,
        observable_vs_direct_source=pose_comparison(direct_source, source_pose(obs)),
        robot_joint_qpos_indices=list(robot._ref_joint_pos_indexes),
        finger_qpos_indices=list(robot._ref_gripper_joint_pos_indexes),
        robot_joint_qpos=env.sim.data.qpos[robot._ref_joint_pos_indexes].copy(),
        finger_qpos=env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes].copy(),
        observed_normalized_aperture=aperture(obs),
        label_rows=rows,
        geometry_comparisons=comparisons,
        camera_world_positions=env.sim.data.cam_xpos.copy(),
        camera_world_axes=env.sim.data.cam_xmat.copy(),
        model_camera_positions=env.sim.model.cam_pos.copy(),
        model_camera_quaternions=env.sim.model.cam_quat.copy(),
        model_camera_fovy=env.sim.model.cam_fovy.copy(),
        image_comparisons=image_rows,
        images=images,
        panel=dict(path=str(panel), sha256=sha256(panel)),
        physical_timing_verdict="unresolved",
        scored_control_steps=0,
    )
    write_json(destination / "result.json", result)
    if not all(check["passed"] for check in checks.values()) or not result["render_preserved_flat_state_exactly"]:
        raise RuntimeError(f"static restore/causality check failed: {destination}")
    return result


def main():
    os.environ.update(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REPO / "src"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wam_video_pose_restore_20261020.json")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = read_json(config_path)
    destination = validate(config, args.seed)
    paths = verify_inputs(config)
    code_sha, config_sha = sha256(__file__), sha256(config_path)
    record, native, published = load_sources(config, paths)
    assets = {}
    template = read_json(paths["policy_template"])
    if resolve(read_json(paths["main_config"])["policy_template"]) != paths["policy_template"]:
        raise ValueError("policy template differs from main study")
    if any(resolve(template[name]) != paths[name] for name in ("tool_contract", "libero_config", "osmesa_library")):
        raise ValueError("environment dependencies differ from frozen inputs")
    destination.mkdir(parents=True, exist_ok=False)
    write_json(
        destination / "launch.json",
        dict(
            config=config,
            code_sha256=code_sha,
            config_sha256=config_sha,
            input_files_sha256={str(path): sha256(path) for path in paths.values()},
            original_xml_sha256=hashlib.sha256(record["model_xml"].encode()).hexdigest(),
            model_loaded=False,
            scored_control_steps=0,
        ),
    )
    results, setups = [], []
    try:
        import numpy as np

        from s2a.libero_simulation import configure_simulator, make_environment

        tool = np.asarray(read_json(paths["tool_contract"])["source_from_tcp"])
        for resolution in config["resolutions"]:
            env_dir = destination / f"resolution{resolution}"
            env_dir.mkdir()
            effective = dict(
                template,
                task_id=record["task_id"],
                task_text=record["task"],
                seed=42,
                camera_observations=False,
                render_resolution=resolution,
            )
            configure_simulator(effective, env_dir)
            asset_binding = asset_fingerprints(record)
            if assets and assets != asset_binding["assets_sha256"]:
                raise ValueError("loaded asset bindings differ between the two environments")
            assets = asset_binding["assets_sha256"]
            env = None
            with guard_dynamics() as (phase, counts):
                try:
                    env, _ = make_environment(effective, env_dir, record, 0)
                    phase[0] = "static"
                    import robosuite.macros as macros
                    from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING

                    if macros.IMAGE_CONVENTION != "opengl" or IMAGE_CONVENTION_MAPPING["opengl"] != 1:
                        raise ValueError("render helper/native opengl convention differs")
                    setup = dict(
                        resolution=resolution,
                        effective_config=effective,
                        initialization_mj_calls=dict(counts["setup"]),
                        state_after_setup=snapshot(env),
                        model_dimensions=dict(nq=env.sim.model.nq, nv=env.sim.model.nv, na=env.sim.model.na),
                        image_convention=macros.IMAGE_CONVENTION,
                        asset_binding=asset_binding,
                        note="Construction/reset may perform internal dynamics; counted separately, never scored.",
                    )
                    write_json(env_dir / "setup.json", setup)
                    setups.append(setup)
                    for frame in config["target_frames"]:
                        for offset in config["candidate_state_index_minus_target_frame"]:
                            case_dir = env_dir / f"frame{frame:04d}_state{frame + offset:04d}"
                            case_dir.mkdir()
                            results.append(
                                run_state(env, config, native, published, tool, resolution, frame, offset, case_dir)
                            )
                    if any(counts["static"].values()):
                        raise RuntimeError("dynamics attempted during static scoring")
                finally:
                    write_json(env_dir / "dynamics_counts.json", counts)
                    if env is not None:
                        env.close()
        verify_inputs(config)
        if any(sha256(path) != digest for path, digest in assets.items()):
            raise ValueError("XML-referenced assets changed")
        if sha256(__file__) != code_sha or sha256(config_path) != config_sha:
            raise ValueError("diagnostic code/config changed")
        if len(results) != 18 or sum(len(row["image_comparisons"]) for row in results) != 144:
            raise ValueError("fixed state/resolution/image hypothesis matrix incomplete")
        write_json(
            destination / "summary.json",
            dict(
                complete=True,
                environments=2,
                candidate_states=9,
                restored_cases=18,
                image_comparisons=144,
                scored_control_steps=0,
                WAM_requests=0,
                new_video_decoding=False,
                all_restore_checks_passed=True,
                render_state_preserved_exactly=True,
                physical_image_pose_time="unresolved",
                automatic_offset_or_orientation_selection=False,
                initialization=[
                    dict(resolution=row["resolution"], mj_calls=row["initialization_mj_calls"]) for row in setups
                ],
                results=results,
                limitations=config["notes"],
                psnr_convention="psnr_db=null with pixel_identical=true means positive infinity; all hypotheses retained.",
            ),
        )
        artifacts = {
            str(path): sha256(path)
            for path in sorted(destination.rglob("*"))
            if path.is_file()
            and path.suffix in {".json", ".png"}
            and not {"numba_cache", "matplotlib", "libero_config"}.intersection(path.relative_to(destination).parts)
        }
        versions = {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "mujoco", "robosuite", "h5py", "Pillow")
        }
        write_json(
            destination / "manifest.json",
            dict(
                complete=True,
                created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                seed=42,
                code_sha256=code_sha,
                config_sha256=config_sha,
                config=config,
                input_files_sha256={str(path): sha256(path) for path in paths.values()},
                assets_sha256=assets,
                artifact_files_sha256=artifacts,
                runtime_cache_bytes_hashed=False,
                runtime_versions=versions,
                python=sys.version,
                python_prefix=sys.prefix,
                environment={
                    key: os.environ[key]
                    for key in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "MUJOCO_GL", "PYOPENGL_PLATFORM")
                },
                scored_control_steps=0,
                WAM_requests=0,
                physical_image_pose_time="unresolved",
            ),
        )
    except BaseException as error:
        write_json(
            destination / "failure.json",
            dict(
                complete=False,
                error=repr(error),
                completed_cases=len(results),
                scored_control_steps=0,
                model_failure=False,
            ),
        )
        raise
    print(json.dumps(dict(output_dir=str(destination), complete=True, restored_cases=18, scored_control_steps=0)))


if __name__ == "__main__":
    main()
