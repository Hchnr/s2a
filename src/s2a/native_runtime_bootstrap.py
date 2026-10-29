"""Bounded CPU OSMesa bootstrap, without actions or integrated physical steps.

Only ``run_native_bootstrap`` loads optional simulation dependencies. The
injected runtime interface permits synthetic tests without loading a library.
This mirrors the successful historical initialization path, except that the
LIBERO retrying outer reset is replaced by one inner reset. It establishes
runtime readiness only, never a control or trajectory qualification.
"""

import hashlib
import importlib
import os
import random
import sys
import tempfile
import xml.etree.ElementTree as ET
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LIMITS = dict(
    configure=1,
    model_construct=3,
    context=3,
    forward=32,
    step1=2,
    reset=1,
    render=1,
    read_pixels=1,
    control_step=0,
    mj_step=0,
    mj_step2=0,
)
MODEL_STAGES = ("constructor", "hard_reset", "record_xml")


class BootstrapFailure(RuntimeError):
    """A failed single attempt, with completed and attempted calls preserved."""

    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


@dataclass
class BootstrapRuntime:
    """Bindings returned after configure, before any environment construction."""

    benchmark: object
    get_path: object
    env_class: object
    mujoco: object
    sim_class: object
    context_class: object
    env_base_class: object
    postprocess_xml: object
    controller_contract: object
    renderer_info: object


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _reference(path):
    path = Path(path)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return dict(path=str(path), sha256=digest.hexdigest(), size_bytes=size)


def _cpu_environment():
    expected = dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    actual = {name: os.environ.get(name) for name in expected}
    _require(actual == expected and Path.cwd().resolve() == REPO, "CPU environment or repository cwd changed")
    return actual


def _safe_path(path, roots, *, exists=True):
    path = Path(path)
    _require(path.is_absolute() and ".." not in path.parts, f"non-absolute or traversing path: {path}")
    _require(not str(path).startswith("/share/project/"), f"retired path forbidden: {path}")
    _require(not any(part.is_symlink() for part in (path, *path.parents)), f"symlink path forbidden: {path}")
    resolved = path.resolve(strict=exists)
    _require(any(resolved.is_relative_to(root) for root in roots), f"path outside approved roots: {path}")
    return resolved


def _mapped_libraries():
    """Record actual mapped ELF paths; their identity is current, not historical."""
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        pieces = line.split(maxsplit=5)
        if len(pieces) == 6 and pieces[5].startswith("/") and ".so" in pieces[5]:
            paths.add(pieces[5])
    result = []
    for name in sorted(paths):
        _require(not name.endswith(" (deleted)"), f"mapped library disappeared: {name}")
        path = Path(name)
        _require(not name.startswith("/share/project/"), f"retired mapped library: {name}")
        result.append({**_reference(path), "identity": "current_loaded"})
    return result


def _loaded_sources():
    paths = {
        str(Path(module.__file__).resolve())
        for name, module in list(sys.modules.items())
        if name.split(".")[0] in {"libero", "robosuite", "mujoco", "OpenGL"} and getattr(module, "__file__", None)
    }
    records = []
    for path in sorted(paths):
        _safe_path(path, (REPO,))
        records.append(_reference(path))
    return records


def _default_runtime(config, output):
    # No simulation import at module import or dry/metadata admission time.
    from s2a.libero_simulation import configure_simulator, controller_contract

    _require("robosuite" not in sys.modules and "libero" not in sys.modules, "runtime must be freshly imported")
    benchmark, get_path, env_class = configure_simulator(config, output)
    binding = importlib.import_module("robosuite.utils.binding_utils")
    base = importlib.import_module("robosuite.environments.base")
    macros = importlib.import_module("robosuite.macros")
    _require(macros.FILE_LOGGING_LEVEL is None, "external robosuite file logging is forbidden")
    _require("robosuite.macros_private" not in sys.modules, "unregistered private macros are forbidden")
    utils = importlib.import_module("libero.libero.utils.utils")
    mujoco = importlib.import_module("mujoco")

    def renderer_info(env):
        from OpenGL import GL

        context = env.sim._render_context_offscreen
        gl = context.gl_ctx
        # robosuite subclasses the MuJoCo OSMesa context; inspect the full MRO.
        ancestry = [f"{cls.__module__}.{cls.__name__}" for cls in type(gl).__mro__]
        _require(any(name.startswith("mujoco.osmesa.") for name in ancestry), "non-OSMesa context loaded")
        values = {}
        for key in ("VENDOR", "RENDERER", "VERSION"):
            value = GL.glGetString(getattr(GL, f"GL_{key}"))
            _require(isinstance(value, bytes) and bool(value), f"missing GL {key}")
            values[key.lower()] = value.decode("utf-8", errors="strict")
        return dict(backend="osmesa", context_ancestry=ancestry, **values)

    return BootstrapRuntime(
        benchmark,
        get_path,
        env_class,
        mujoco,
        binding.MjSim,
        binding.MjRenderContext,
        base.MujocoEnv,
        utils.postprocess_model_xml,
        controller_contract,
        renderer_info,
    )


def _xml_structure(node):
    return (
        node.tag,
        {key: value for key, value in node.attrib.items() if key != "file"},
        node.text,
        node.tail,
        tuple(_xml_structure(child) for child in node),
    )


def _asset_records(xml, roots, expected):
    _require(isinstance(xml, str), "model XML must be text")
    _require("<!DOCTYPE" not in xml.upper() and "<!ENTITY" not in xml.upper(), "XML entities forbidden")
    root = ET.fromstring(xml)
    _require(not list(root.iter("include")), "unexpanded XML include forbidden")
    # Keep historical compiler meshdir="meshes/" unchanged. Every file below
    # must be absolute, so MuJoCo does not resolve it through that directory.
    result = []
    for index, node in enumerate(root.iter()):
        declared = node.get("file")
        if declared is None:
            continue
        path = _safe_path(declared, roots)
        _require(path.is_file(), f"asset is not a file: {path}")
        ref = _reference(path)
        if expected is not None:
            _require(expected.get(str(path)) == ref["sha256"], f"unbound or changed asset: {path}")
        result.append(dict(element_index=index, tag=node.tag, declared_path=declared, reference=ref))
    return result


def _patch(stack, target, name, replacement):
    existed = name in vars(target)
    original = vars(target).get(name)
    setattr(target, name, replacement)

    def restore():
        if existed:
            setattr(target, name, original)
        else:
            delattr(target, name)

    stack.callback(restore)


def _state(env):
    return np.asarray(env.sim.get_state().flatten(), dtype=np.float64).copy()


def _controller_check(runtime, env):
    controller, info = runtime.controller_contract(env)
    info = dict(info)
    info.update(
        use_delta=bool(controller.use_delta),
        impedance_mode=controller.impedance_mode,
        interpolator_pos=None if controller.interpolator_pos is None else type(controller.interpolator_pos).__name__,
        interpolator_ori=None if controller.interpolator_ori is None else type(controller.interpolator_ori).__name__,
        uncoupling=bool(controller.uncoupling),
    )
    _require(info["use_delta"] and info["impedance_mode"] == "fixed" and info["uncoupling"], "controller mode changed")
    _require(info["interpolator_pos"] is None and info["interpolator_ori"] is None, "interpolator present")
    expected = dict(
        input_min=np.full(6, -1.0),
        input_max=np.ones(6),
        output_min=np.array([-0.05, -0.05, -0.05, -0.5, -0.5, -0.5]),
        output_max=np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5]),
        kp=np.full(6, 150.0),
        kd=np.full(6, 2 * np.sqrt(150.0)),
    )
    for key, value in expected.items():
        _require(np.array_equal(np.asarray(info[key]), value), f"fixed controller {key} mismatch")
    _require(info["control_freq"] == 20 and info["model_timestep"] == 0.002, "controller timing mismatch")
    _require(info["robot_model"] == "MountedPanda", "robot model mismatch")
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in info.items()}


def run_native_bootstrap(
    runtime_config,
    record,
    output,
    *,
    allowed_asset_roots,
    asset_refs=None,
    runtime_factory=None,
    progress=None,
):
    """One bounded bootstrap and one agentview render; never dispatch actions.

    The caller must admit all input/QA references and the registered OpenCV
    import mutation first. ``output`` is a new exclusive directory below this
    repository's outputs. Every failed attempt raises ``BootstrapFailure``;
    its receipt is suitable for the caller's permanent failure result.
    ``runtime_factory`` is the only dependency seam; production leaves it None.
    """
    receipt = dict(
        schema_version="s2a.native-runtime-bootstrap.v1",
        complete=False,
        passed=False,
        runtime_loaded=False,
        runtime_load_attempted=False,
        bootstrap_verified=False,
        control_qualified=False,
        limits=dict(LIMITS),
        counters={name: dict(attempted=0, completed=0) for name in LIMITS},
        stages=[],
        model_records=[],
        time_checks=[],
        new_s2a_actions=0,
        new_physics_steps=0,
        new_world_model_calls=0,
        new_fresh_episodes=0,
        automatic_retries=0,
    )
    stage = "admission"
    env = None
    inner_env = None
    cleanup_complete = False
    cleanup_attempted = False
    audit_active = [False]
    output = Path(output)

    def cleanup():
        nonlocal cleanup_complete, cleanup_attempted
        if env is not None and not cleanup_attempted:
            cleanup_attempted = True
            try:
                target = env
                if vars(env).get("env") is None and inner_env is not None:
                    target = inner_env
                receipt["cleanup_target"] = "inner_partial" if target is not env else "outer"
                target.close()
                cleanup_complete = True
            except Exception as error:
                receipt["cleanup_error"] = dict(type=type(error).__name__, message=str(error))

    def file_guard(event, args):
        if not audit_active[0]:
            return
        writes = []
        if event == "open":
            path, mode, flags = args
            if isinstance(path, int):
                path = os.readlink(f"/proc/self/fd/{path}")
                if not path.startswith("/"):  # Inherited stdout/stderr pipes, not files.
                    return
            _require(not os.fsdecode(path).startswith("/share/project/"), "retired path read attempted")
            if (mode and any(letter in mode for letter in "wax+")) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
                writes.append((path, None))
        elif event == "os.mkdir":
            writes.append((args[0], args[2]))
        elif event in ("os.remove", "os.rmdir"):
            writes.append((args[0], args[1]))
        elif event == "os.rename":
            writes.extend(((args[0], args[2]), (args[1], args[3])))
        elif event in ("os.link", "os.symlink"):
            raise ValueError("runtime link creation forbidden")
        for path, directory_fd in writes:
            target = Path(os.fsdecode(path))
            if not target.is_absolute():
                parent = (
                    Path.cwd() if directory_fd in (None, -1) else Path(os.readlink(f"/proc/self/fd/{directory_fd}"))
                )
                target = parent / target
            _safe_path(target, (output,), exists=False)

    def notify():
        if progress is not None:
            progress(receipt)

    def call(name, function, *args, **kwargs):
        counter = receipt["counters"][name]
        counter["attempted"] += 1
        notify()
        _require(counter["attempted"] <= LIMITS[name], f"{name} call budget exceeded")
        result = function(*args, **kwargs)
        counter["completed"] += 1
        return result

    try:
        receipt["environment_before"] = _cpu_environment()
        output = _safe_path(output, (REPO / "outputs",), exists=False)
        _require(not output.exists(), "bootstrap output already exists")
        roots = tuple(_safe_path(path, (REPO,), exists=True) for path in allowed_asset_roots)
        _require(bool(roots), "asset roots are required")
        config = dict(runtime_config)
        _require(
            config.get("camera_observations") is False
            and config.get("refresh_controller_after_restore") is True
            and config.get("render_resolution") == 256
            and config.get("seed") == 42,
            "fixed bootstrap configuration mismatch",
        )
        for key in ("libero_root", "libero_config", "osmesa_library"):
            _safe_path(config[key], (REPO,), exists=True)
        # Reject a potential external /tmp file handler before robosuite import.
        private = (
            Path(sys.prefix)
            / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/robosuite/macros_private.py"
        )
        _require(not private.exists(), "private robosuite logging/macro overrides forbidden")
        initial = np.asarray(record["initial_state"], dtype=np.float64)
        _require(initial.shape == (79,) and np.isfinite(initial).all(), "initial state must be finite float64[79]")
        _require(_hash(initial.tobytes()) == record["initial_state_sha256"], "initial state SHA mismatch")
        output.mkdir(parents=False, exist_ok=False)
        for name in ("tmp", "cache", "models"):
            (output / name).mkdir()
        # gettempdir() may have cached /tmp before this entry point was called.
        tempfile.tempdir = str(output / "tmp")
        os.environ.update(
            TMPDIR=str(output / "tmp"),
            XDG_CACHE_HOME=str(output / "cache"),
            MUJOCO_GL="osmesa",
            PYOPENGL_PLATFORM="osmesa",
            NUMBA_CACHE_DIR=str(output / "numba_cache"),
            MPLCONFIGDIR=str(output / "matplotlib"),
        )
        sys.addaudithook(file_guard)
        audit_active[0] = True
        receipt["write_guard_scope"] = "Python audit events; not an OS syscall sandbox"
        receipt["loaded_libraries_before"] = _mapped_libraries()
        stage = "configure"
        receipt["runtime_load_attempted"] = True
        runtime = call("configure", runtime_factory or _default_runtime, config, output)
        receipt["runtime_loaded"] = True
        receipt["environment_after_import"] = _cpu_environment()
        _require(os.environ["MUJOCO_GL"] == os.environ["PYOPENGL_PLATFORM"] == "osmesa", "GL backend changed")
        task = runtime.benchmark.get_benchmark_dict()[config["task_suite"]]().get_task(config["task_id"])
        bddl = _safe_path(Path(runtime.get_path("bddl_files")) / task.problem_folder / task.bddl_file, (REPO,))
        receipt["bddl_ref"] = _reference(bddl)
        original_compile = runtime.sim_class.from_xml_string
        with ExitStack() as stack:

            def compile_model(cls, xml):
                def construct():
                    _require(stage in MODEL_STAGES, f"model construction outside fixed stages: {stage}")
                    assets = _asset_records(xml, roots, asset_refs)
                    path = output / "models" / f"{len(receipt['model_records']):02d}_{stage}.xml"
                    with path.open("xb") as stream:
                        stream.write(xml.encode("utf-8"))
                    item = dict(stage=stage, xml_ref=_reference(path), assets=assets, completed=False)
                    receipt["model_records"].append(item)
                    model = original_compile(xml)
                    item["completed"] = True
                    return model

                return call("model_construct", construct)

            _patch(stack, runtime.sim_class, "from_xml_string", classmethod(compile_model))
            original_context = runtime.context_class.__init__

            def context(*args, **kwargs):
                return call("context", original_context, *args, **kwargs)

            _patch(stack, runtime.context_class, "__init__", context)
            names = {
                "mj_forward": "forward",
                "mj_step1": "step1",
                "mj_resetData": "reset",
                "mj_step": "mj_step",
                "mj_step2": "mj_step2",
                "mjr_render": "render",
                "mjr_readPixels": "read_pixels",
            }
            for attribute, name in names.items():
                original = getattr(runtime.mujoco, attribute)

                def observed(*args, _original=original, _name=name, **kwargs):
                    before = float(args[1].time) if _name in ("step1", "forward") else None
                    value = call(_name, _original, *args, **kwargs)
                    if before is not None:
                        after = float(args[1].time)
                        receipt["time_checks"].append(dict(operation=_name, before=before, after=after))
                        _require(after == before, f"{_name} advanced simulation time")
                    return value

                _patch(stack, runtime.mujoco, attribute, observed)

            def forbidden_control_step(*args, **kwargs):
                return call("control_step", lambda: None)

            _patch(stack, runtime.env_class, "step", forbidden_control_step)
            _patch(stack, runtime.env_base_class, "step", forbidden_control_step)
            # ControlEnv assigns self.env only after the inner constructor returns.
            # Observe the base constructor too, so failures still expose its sim.
            original_inner_init = runtime.env_base_class.__init__

            def initialize_inner(instance, *args, **kwargs):
                nonlocal inner_env
                inner_env = instance
                return original_inner_init(instance, *args, **kwargs)

            _patch(stack, runtime.env_base_class, "__init__", initialize_inner)
            # Capture even a partially initialized environment for one cleanup.
            original_init = runtime.env_class.__init__

            def initialize(instance, *args, **kwargs):
                nonlocal env
                env = instance
                return original_init(instance, *args, **kwargs)

            _patch(stack, runtime.env_class, "__init__", initialize)
            stack.callback(cleanup)  # Run while counter/forbidden-step observers remain installed.
            random.seed(config["seed"])
            np.random.seed(config["seed"])
            stage = "constructor"
            receipt["stages"].append(dict(name=stage, completed=False))
            env = runtime.env_class(
                bddl_file_name=str(bddl),
                camera_heights=256,
                camera_widths=256,
                render_gpu_device_id=-1,
                use_camera_obs=False,
            )
            receipt["stages"][-1]["completed"] = True
            env.seed(config["seed"])
            stage = "hard_reset"
            receipt["stages"].append(dict(name=stage, completed=False))
            # Do not enter ControlEnv.reset's exception-swallowing retry loop.
            env.env.reset()
            receipt["stages"][-1]["completed"] = True
            stage = "record_xml"
            receipt["stages"].append(dict(name=stage, completed=False))
            _require(
                "<!DOCTYPE" not in record["model_xml"].upper() and "<!ENTITY" not in record["model_xml"].upper(),
                "XML entities forbidden",
            )
            original_xml = ET.fromstring(record["model_xml"])
            xml = ET.fromstring(runtime.postprocess_xml(record["model_xml"], {}))
            _require(_xml_structure(original_xml) == _xml_structure(xml), "postprocess changed non-path XML content")
            redirects = []
            for node in xml.iter():
                file = node.get("file")
                if file:
                    _require(Path(file).is_absolute() and ".." not in Path(file).parts, "invalid declared XML path")
                # Match the frozen fallback without probing any external old path.
                # The bound historical record also contains /Users/.../assets/.
                if file and not any(Path(file).is_relative_to(root) for root in roots):
                    _require("/assets/" in file, "unrecognized external XML path")
                    replacement = str(Path(runtime.get_path("assets")) / file.split("/assets/", 1)[1])
                    redirects.append(dict(declared=file, replacement=replacement))
                    node.set("file", replacement)
            receipt["record_xml_redirects"] = redirects
            receipt["original_record_xml_sha256"] = _hash(record["model_xml"].encode())
            env.reset_from_xml_string(ET.tostring(xml, encoding="unicode"))
            receipt["stages"][-1]["completed"] = True
            stage = "restore_state"
            env.set_init_state(initial.copy())
            env.robots[0].controller.update(force=True)
            actual = _state(env)
            receipt["initial_state_check"] = dict(
                shape=list(actual.shape),
                exact=bool(actual.shape == initial.shape and actual.tobytes() == initial.tobytes()),
                expected_sha256=_hash(initial.tobytes()),
                actual_sha256=_hash(actual.tobytes()),
                max_abs_difference=float(np.max(np.abs(actual - initial))) if actual.shape == initial.shape else None,
            )
            _require(receipt["initial_state_check"]["exact"], "restored initial state differs")
            receipt["controller_contract"] = _controller_check(runtime, env)
            receipt["renderer"] = runtime.renderer_info(env)
            _require(receipt["renderer"]["backend"] == "osmesa", "OSMesa renderer not verified")
            _require(
                any(token in receipt["renderer"]["renderer"].lower() for token in ("llvmpipe", "softpipe", "swrast")),
                "software renderer not verified",
            )
            stage = "render"
            before = _state(env)
            image = np.asarray(env.sim.render(width=256, height=256, camera_name="agentview"))
            _require(image.shape == (256, 256, 3) and image.dtype == np.uint8, "invalid agentview frame")
            pixel_min, pixel_max = int(image.min()), int(image.max())
            _require(pixel_min < pixel_max, "constant agentview frame")
            _require(before.tobytes() == _state(env).tobytes(), "render changed initial state")
            frame_path = output / "agentview.rgb"
            with frame_path.open("xb") as stream:
                stream.write(image.tobytes())
            receipt["frame"] = dict(
                camera="agentview",
                shape=list(image.shape),
                dtype=str(image.dtype),
                pixel_min=pixel_min,
                pixel_max=pixel_max,
                reference=_reference(frame_path),
            )
            receipt["loaded_libraries_after"] = _mapped_libraries()
            receipt["source_refs"] = _loaded_sources() if runtime_factory is None else []
            if runtime_factory is None:
                loaded = {item["path"]: item["sha256"] for item in receipt["loaded_libraries_after"]}
                osmesa = _reference(config["osmesa_library"])
                _require(loaded.get(osmesa["path"]) == osmesa["sha256"], "registered OSMesa not mapped")
            for name, expected in dict(model_construct=3, context=3, step1=2, reset=1, render=1, read_pixels=1).items():
                _require(receipt["counters"][name]["completed"] == expected, f"unexpected completed {name} count")
            _require(
                [item["stage"] for item in receipt["model_records"]] == list(MODEL_STAGES), "model stage order mismatch"
            )
            for model in receipt["model_records"]:
                for asset in model["assets"]:
                    ref = asset["reference"]
                    _safe_path(ref["path"], roots)
                    _require(_reference(ref["path"]) == ref, "model asset changed during bootstrap")
            stage = "close"
            cleanup()
            _require(cleanup_complete, "environment cleanup failed")
        receipt["environment_after"] = _cpu_environment()
        for key in ("loaded_libraries_before", "loaded_libraries_after", "source_refs"):
            for ref in receipt.get(key, []):
                _require(_reference(ref["path"])["sha256"] == ref["sha256"], f"loaded source changed: {ref['path']}")
        receipt["loaded_sources_unchanged"] = True
        receipt.update(complete=True, passed=True, bootstrap_verified=True)
        return receipt
    except Exception as error:
        receipt["failure_stage"] = stage
        receipt["error"] = dict(type=type(error).__name__, message=str(error))
        if receipt["runtime_load_attempted"]:
            for key, capture in (
                ("loaded_libraries_failure", _mapped_libraries),
                ("source_refs_failure", _loaded_sources),
            ):
                try:
                    receipt[key] = capture()
                except Exception as capture_error:
                    receipt[key + "_capture_error"] = dict(
                        type=type(capture_error).__name__, message=str(capture_error)
                    )
        raise BootstrapFailure(str(error), receipt) from error
    finally:
        audit_active[0] = False
        receipt["cleanup_attempted"] = cleanup_attempted
        receipt["cleanup_complete"] = cleanup_complete
        notify()
