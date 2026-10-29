"""Synthetic bootstrap call graph; no MuJoCo, GL library, or environment loads."""

import hashlib
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from s2a import native_runtime_bootstrap as bootstrap


@pytest.fixture
def case(tmp_path, monkeypatch):
    for key, value in dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1").items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(bootstrap.REPO)
    monkeypatch.setattr(bootstrap, "_mapped_libraries", lambda: [])
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    for key in ("TMPDIR", "XDG_CACHE_HOME", "MUJOCO_GL", "PYOPENGL_PLATFORM", "NUMBA_CACHE_DIR", "MPLCONFIGDIR"):
        monkeypatch.setenv(key, "")  # Restore the caller's environment after the core's changes.
    assets = tmp_path / "assets"
    assets.mkdir()
    asset = assets / "mesh.stl"
    asset.write_bytes(b"synthetic geometry")
    bddl = tmp_path / "problem.bddl"
    bddl.write_text("synthetic task")
    configfile = tmp_path / "config.yaml"
    configfile.write_text("synthetic config")
    library = tmp_path / "libOSMesa.so"
    library.write_bytes(b"not a real library")
    initial = np.linspace(0, 0.78, 79)
    initial[0] = 0
    xml = f'<mujoco><compiler meshdir="meshes/"/><asset><mesh file="{asset}"/></asset></mujoco>'
    record = dict(
        initial_state=initial.tolist(),
        initial_state_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
        model_xml=xml,
    )
    config = dict(
        libero_root=str(tmp_path),
        libero_config=str(configfile),
        osmesa_library=str(library),
        seed=42,
        render_resolution=256,
        camera_observations=False,
        refresh_controller_after_restore=True,
        task_suite="synthetic",
        task_id=8,
    )
    actions = SimpleNamespace(reset_calls=0, close_calls=0, native_steps=0, failure=None)
    mj = SimpleNamespace()

    def no_op(*args, **kwargs):
        return None

    def forbidden_native(*args, **kwargs):
        actions.native_steps += 1

    mj.mj_forward = no_op
    mj.mj_step1 = no_op
    mj.mj_step = mj.mj_step2 = forbidden_native
    mj.mj_resetData = no_op
    mj.mjr_render = mj.mjr_readPixels = no_op

    class Sim:
        def __init__(self):
            self.flat = np.zeros(79)
            self.data = SimpleNamespace(time=0.0)
            self.model = object()

        @classmethod
        def from_xml_string(cls, text):
            return cls()

        def forward(self):
            mj.mj_forward(self.model, self.data)

        def get_state(self):
            return SimpleNamespace(flatten=lambda: self.flat.copy())

        def render(self, **kwargs):
            assert kwargs == dict(width=256, height=256, camera_name="agentview")
            mj.mjr_render()
            mj.mjr_readPixels()
            if actions.failure == "render_moves":
                self.flat[1] += 1
            image = np.zeros((256, 256, 3), dtype=np.uint8)
            if actions.failure != "constant_frame":
                image[0, 0] = 255
            return image

    class Context:
        def __init__(self, sim):
            sim.forward()

    class Base:
        def step(self, command):
            actions.native_steps += 1

    class Controller:
        use_delta = True
        impedance_mode = "fixed"
        interpolator_pos = interpolator_ori = None
        uncoupling = True

        def __init__(self, sim):
            self.sim = sim
            sim.forward()
            sim.forward()

        def update(self, force=False):
            assert force
            self.sim.forward()

    class Env(Base):
        def __init__(self, **kwargs):
            assert kwargs["use_camera_obs"] is False and kwargs["render_gpu_device_id"] == -1
            self.env = SimpleNamespace(reset=self.inner_reset)
            self.build(xml, False)
            if actions.failure in ("mj_step", "mj_step2"):
                getattr(mj, actions.failure)(self.sim.model, self.sim.data)
            elif actions.failure == "control_step":
                self.step(np.zeros(7))
            elif actions.failure == "forward_budget":
                for _ in range(40):
                    self.sim.forward()

        def build(self, text, deterministic):
            self.sim = Sim.from_xml_string(text)
            self.sim.forward()
            if deterministic:
                mj.mj_resetData(self.sim.model, self.sim.data)
            Context(self.sim)
            self.robots = [SimpleNamespace(controller=Controller(self.sim))]
            if not deterministic:
                mj.mj_step1(self.sim.model, self.sim.data)

        def seed(self, seed):
            assert seed == 42

        def reset(self):
            raise AssertionError("must not invoke unbounded outer reset")

        def inner_reset(self):
            actions.reset_calls += 1
            if actions.failure == "reset_error":
                raise RuntimeError("synthetic reset failure")
            self.build(xml, False)
            self.sim.forward()

        def reset_from_xml_string(self, text):
            self.build(text, True)
            self.sim.forward()

        def set_init_state(self, state):
            self.sim.flat = state.copy()
            self.sim.data.time = float(state[0])
            self.sim.forward()
            if actions.failure == "state_mismatch":
                self.sim.flat[1] = np.nextafter(state[1], np.inf)
            if actions.failure == "signed_zero":
                self.sim.flat[0] = -0.0

        def close(self):
            actions.close_calls += 1
            if actions.failure == "close_step":
                mj.mj_step(self.sim.model, self.sim.data)

    def contract(env):
        return env.robots[0].controller, dict(
            input_min=np.full(6, -1.0),
            input_max=np.ones(6),
            output_min=np.array([-0.05] * 3 + [-0.5] * 3),
            output_max=np.array([0.05] * 3 + [0.5] * 3),
            kp=np.full(6, 150.0),
            kd=np.full(6, 2 * np.sqrt(150.0)),
            control_freq=20,
            model_timestep=0.002,
            robot_model="MountedPanda",
            gripper_model="PandaGripper",
            controller_site="gripper0_grip_site",
            control_frame="MuJoCo world",
        )

    benchmark = SimpleNamespace(
        get_benchmark_dict=lambda: {
            "synthetic": lambda: SimpleNamespace(
                get_task=lambda task: SimpleNamespace(problem_folder="", bddl_file=bddl.name)
            )
        }
    )
    runtime = bootstrap.BootstrapRuntime(
        benchmark,
        lambda name: str(assets if name == "assets" else tmp_path),
        Env,
        mj,
        Sim,
        Context,
        Base,
        lambda text, cameras: text,
        contract,
        lambda env: dict(backend="osmesa", renderer="llvmpipe (synthetic)"),
    )

    def factory(cfg, output):
        if actions.failure == "outside_write":
            (tmp_path / "forbidden.log").write_text("unexpected")
        if actions.failure == "old_read":
            Path("/share/project/forbidden_asset").read_bytes()
        return runtime

    kwargs = dict(
        runtime_config=config,
        record=record,
        output=tmp_path / "run",
        allowed_asset_roots=[assets],
        runtime_factory=factory,
    )
    return SimpleNamespace(kwargs=kwargs, runtime=runtime, actions=actions, assets=assets, asset=asset, xml=xml)


def test_single_reset_three_models_one_frame_and_restore_observers(case):
    before = case.runtime.sim_class.from_xml_string.__func__
    result = bootstrap.run_native_bootstrap(**case.kwargs)
    assert result["passed"] and result["bootstrap_verified"] and not result["control_qualified"]
    counts = {key: item["completed"] for key, item in result["counters"].items()}
    assert counts == dict(
        configure=1,
        model_construct=3,
        context=3,
        forward=16,
        step1=2,
        reset=1,
        render=1,
        read_pixels=1,
        control_step=0,
        mj_step=0,
        mj_step2=0,
    )
    assert case.actions.reset_calls == case.actions.close_calls == 1
    assert case.runtime.sim_class.from_xml_string.__func__ is before
    assert [entry["stage"] for entry in result["model_records"]] == list(bootstrap.MODEL_STAGES)
    assert all(entry["before"] == entry["after"] for entry in result["time_checks"])
    assert result["initial_state_check"]["exact"]
    assert result["controller_contract"]["use_delta"]
    assert result["frame"]["shape"] == [256, 256, 3]


@pytest.mark.parametrize("failure", ["mj_step", "mj_step2", "control_step"])
def test_forbidden_step_counted_before_dispatch_and_partial_env_closed(case, failure):
    case.actions.failure = failure
    with pytest.raises(bootstrap.BootstrapFailure) as caught:
        bootstrap.run_native_bootstrap(**case.kwargs)
    result = caught.value.receipt
    assert result["counters"][failure] == dict(attempted=1, completed=0)
    assert case.actions.native_steps == 0
    assert case.actions.close_calls == 1
    assert not result["bootstrap_verified"] and result["failure_stage"] == "constructor"


def test_inner_reset_failure_never_retries(case):
    case.actions.failure = "reset_error"
    with pytest.raises(bootstrap.BootstrapFailure) as caught:
        bootstrap.run_native_bootstrap(**case.kwargs)
    assert case.actions.reset_calls == 1
    assert caught.value.receipt["counters"]["model_construct"]["completed"] == 1
    assert caught.value.receipt["automatic_retries"] == 0
    assert case.actions.close_calls == 1


@pytest.mark.parametrize("failure", ["state_mismatch", "signed_zero", "render_moves"])
def test_state_and_render_require_byte_exact_original(case, failure):
    case.actions.failure = failure
    with pytest.raises(bootstrap.BootstrapFailure):
        bootstrap.run_native_bootstrap(**case.kwargs)
    assert case.actions.close_calls == 1


def test_forward_budget_stops_before_extra_call(case):
    case.actions.failure = "forward_budget"
    with pytest.raises(bootstrap.BootstrapFailure) as caught:
        bootstrap.run_native_bootstrap(**case.kwargs)
    assert caught.value.receipt["counters"]["forward"] == dict(attempted=33, completed=32)


@pytest.mark.parametrize("failure", ["outside_write", "old_read"])
def test_loader_python_io_guard_blocks_before_side_effect(case, failure):
    case.actions.failure = failure
    with pytest.raises(bootstrap.BootstrapFailure) as caught:
        bootstrap.run_native_bootstrap(**case.kwargs)
    assert caught.value.receipt["failure_stage"] == "configure"
    assert not (case.asset.parent.parent / "forbidden.log").exists()


@pytest.mark.parametrize("old_prefix", ["/share/project/old", "/Users/fixture/libero"])
def test_xml_retired_asset_is_redirected_without_reading_old_path(case, old_prefix):
    case.kwargs["record"]["model_xml"] = (
        f'<mujoco><compiler meshdir="meshes/"/><asset><mesh file="{old_prefix}/assets/mesh.stl"/></asset></mujoco>'
    )
    result = bootstrap.run_native_bootstrap(**case.kwargs)
    assert result["record_xml_redirects"][0]["replacement"] == str(case.asset)
    assert result["model_records"][2]["assets"][0]["reference"]["path"] == str(case.asset)


@pytest.mark.parametrize(
    "bad_xml",
    [
        '<mujoco><asset><mesh file="/share/project/old/mesh.stl"/></asset></mujoco>',
        '<mujoco><include file="model.xml"/></mujoco>',
        '<!DOCTYPE mujoco [<!ENTITY x "bad">]><mujoco/>',
    ],
)
def test_bad_model_is_rejected_before_model_compile(case, bad_xml):
    case.kwargs["record"]["model_xml"] = bad_xml
    with pytest.raises(bootstrap.BootstrapFailure) as caught:
        bootstrap.run_native_bootstrap(**case.kwargs)
    assert caught.value.receipt["counters"]["model_construct"]["completed"] == 2


def test_only_paths_can_change_in_xml_postprocess(case):
    case.runtime.postprocess_xml = lambda xml, cameras: xml.replace('meshdir="meshes/"', 'meshdir="different/"')
    with pytest.raises(bootstrap.BootstrapFailure, match="non-path"):
        bootstrap.run_native_bootstrap(**case.kwargs)


def test_asset_digest_binding_and_symlink_escape(case):
    case.kwargs["asset_refs"] = {str(case.asset): "0" * 64}
    with pytest.raises(bootstrap.BootstrapFailure, match="unbound or changed"):
        bootstrap.run_native_bootstrap(**case.kwargs)
    assert not (case.kwargs["output"] / "models/00_constructor.xml").exists()


def test_cleanup_forbidden_step_is_observed_without_retry(case):
    case.actions.failure = "close_step"
    with pytest.raises(bootstrap.BootstrapFailure) as caught:
        bootstrap.run_native_bootstrap(**case.kwargs)
    assert caught.value.receipt["counters"]["mj_step"] == dict(attempted=1, completed=0)
    assert case.actions.close_calls == 1
    assert case.actions.native_steps == 0


def test_gpu_renderer_claim_is_rejected(case):
    case.runtime.renderer_info = lambda env: dict(backend="osmesa", renderer="unexpected hardware driver")
    with pytest.raises(bootstrap.BootstrapFailure, match="software renderer"):
        bootstrap.run_native_bootstrap(**case.kwargs)
    assert case.actions.close_calls == 1


def test_step1_time_advance_is_not_hidden(case):
    def bad_step1(model, data):
        data.time += 0.002

    case.runtime.mujoco.mj_step1 = bad_step1
    with pytest.raises(bootstrap.BootstrapFailure, match="advanced simulation time") as caught:
        bootstrap.run_native_bootstrap(**case.kwargs)
    assert caught.value.receipt["counters"]["step1"] == dict(attempted=1, completed=1)
    assert caught.value.receipt["time_checks"][-1]["after"] == 0.002


def test_asset_symlink_never_reaches_compile(case):
    alias = case.assets / "alias.stl"
    alias.symlink_to(case.asset)
    case.kwargs["record"]["model_xml"] = f'<mujoco><asset><mesh file="{alias}"/></asset></mujoco>'
    with pytest.raises(bootstrap.BootstrapFailure, match="symlink") as caught:
        bootstrap.run_native_bootstrap(**case.kwargs)
    assert caught.value.receipt["counters"]["model_construct"]["completed"] == 2


def test_path_traversal_is_not_normalized_into_a_safe_asset(case):
    case.kwargs["record"]["model_xml"] = '<mujoco><asset><mesh file="/Users/old/../assets/mesh.stl"/></asset></mujoco>'
    with pytest.raises(bootstrap.BootstrapFailure, match="invalid declared XML path"):
        bootstrap.run_native_bootstrap(**case.kwargs)


def test_partial_inner_constructor_is_closed_before_outer_assignment(case):
    runtime = case.runtime

    class Inner:
        def __init__(self):
            self.sim = runtime.sim_class.from_xml_string(case.xml)
            self.sim.forward()
            runtime.context_class(self.sim)
            raise RuntimeError("inner constructor failed before outer .env assignment")

        def close(self):
            case.actions.close_calls += 1

    class Outer:
        def __init__(self, **kwargs):
            self.env = Inner()

        def close(self):
            raise AssertionError("partial outer close must not be called")

    runtime.env_base_class = Inner
    runtime.env_class = Outer
    with pytest.raises(bootstrap.BootstrapFailure, match="inner constructor failed") as caught:
        bootstrap.run_native_bootstrap(**case.kwargs)
    result = caught.value.receipt
    assert result["counters"]["model_construct"] == dict(attempted=1, completed=1)
    assert result["counters"]["context"] == dict(attempted=1, completed=1)
    assert result["cleanup_target"] == "inner_partial"
    assert result["cleanup_complete"]
    assert case.actions.close_calls == 1


def test_cached_temp_directory_is_redirected_before_loader(case):
    original = case.kwargs["runtime_factory"]
    seen = []

    def factory(config, output):
        with tempfile.TemporaryDirectory() as directory:
            seen.append(Path(directory))
            (Path(directory) / "cache").write_bytes(b"temporary")
        return original(config, output)

    case.kwargs["runtime_factory"] = factory
    bootstrap.run_native_bootstrap(**case.kwargs)
    assert len(seen) == 1 and seen[0].is_relative_to(case.kwargs["output"] / "tmp")
    assert not seen[0].exists()


def test_constant_frame_does_not_qualify_renderer(case):
    case.actions.failure = "constant_frame"
    with pytest.raises(bootstrap.BootstrapFailure, match="constant agentview frame") as caught:
        bootstrap.run_native_bootstrap(**case.kwargs)
    assert caught.value.receipt["counters"]["render"]["completed"] == 1
    assert not caught.value.receipt["bootstrap_verified"]
