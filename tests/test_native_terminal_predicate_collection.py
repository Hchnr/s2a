"""Static diagnostic risks verified with pure fixtures, never a simulator."""

import importlib.util
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from s2a.artifacts import REPO
from s2a.native_training import state_sha256


def collector():
    spec = importlib.util.spec_from_file_location(
        "terminal_collector_fixture", REPO / "scripts/collect_native_terminal_predicate_diagnostic.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registry_class_required_instead_of_decorated_module_symbol():
    module = collector()

    class Base:
        pass

    actual = type(
        "Libero_Tabletop_Manipulation",
        (Base,),
        {
            "__module__": "libero.libero.envs.problems.libero_tabletop_manipulation",
            "_check_success": lambda self: True,
        },
    )
    assert module.registered_tabletop_class({"libero_tabletop_manipulation": actual}, Base) is actual
    for invalid in (None, Base, SimpleNamespace(_check_success=lambda: True)):
        with pytest.raises(ValueError, match="registered tabletop class"):
            module.registered_tabletop_class({"libero_tabletop_manipulation": invalid}, Base)


@pytest.mark.parametrize(
    "kind",
    [
        "mj_step_calls",
        "mj_step2_calls",
        "env_step_calls",
        "sim_step_calls",
        "render_calls",
        "source_hdf5_reads",
        "controller_command_calls",
    ],
)
def test_forbidden_entry_is_persisted_before_original_delegation(tmp_path, kind):
    module, called = collector(), []
    guarded = SimpleNamespace(operation=lambda: called.append("should-not-delegate"))
    path = tmp_path / "ledger.jsonl"
    guard = module.StaticCallGuard(path)
    guard.patch(guarded, "operation", kind)
    try:
        with guard.context("setup", "fixture_case"):
            with pytest.raises(RuntimeError, match="before delegation"):
                guarded.operation()
        assert not called and guard.counts[kind] == 1 and guard.counts[kind + "_delegated"] == 0
        assert guard.violation["kind"] == kind
        event = json.loads(path.read_text().splitlines()[-1])
        assert event["event"] == "rejected_before_delegate" and event["delegated"] is False
    finally:
        guard.close()
    guarded.operation()
    assert called == ["should-not-delegate"]  # The fixture patch was removed.


def test_step1_setup_only_and_unchanged_clock_is_distinct_from_state_install(tmp_path):
    module = collector()
    data, calls = SimpleNamespace(time=1.0), []
    functions = SimpleNamespace(step1=lambda model, data: calls.append(data.time))
    guard = module.StaticCallGuard(tmp_path / "ledger.jsonl")
    guard.patch(functions, "step1", "mj_step1_calls")
    try:
        with guard.context("setup"):
            functions.step1(None, data)
        assert calls == [1.0] and guard.phase_counts["setup"]["mj_step1_calls_delegated"] == 1
        # Installing an existing clock value is observed separately; no time integration.
        data.time = 8.7
        guard.event(
            dict(event="installed_saved_state", api="fixture", time_before=1.0, time_after=8.7, integration=False)
        )
        with guard.context("static", "fixture_case", 174):
            with pytest.raises(RuntimeError, match="before delegation"):
                functions.step1(None, data)
        assert calls == [1.0] and data.time == 8.7 and guard.counts["mj_step_calls"] == 0
    finally:
        guard.close()


def test_single_reset_propagates_inner_exception_and_never_uses_retry_wrapper(tmp_path):
    module, calls = collector(), []

    class Wrapper:
        def reset(self):
            raise AssertionError("old swallowing/retry wrapper must not run")

    def one_reset():
        calls.append("one_actual_delegate")
        raise ValueError("setup failure preserved")

    wrapper = Wrapper()
    wrapper.env = SimpleNamespace(reset=one_reset)
    guard = module.StaticCallGuard(tmp_path / "ledger.jsonl")
    guard.patch(Wrapper, "reset", "single_reset_calls", replacement=module.single_reset)
    try:
        with guard.context("setup"):
            with pytest.raises(ValueError, match="setup failure preserved"):
                wrapper.reset()
        assert calls == ["one_actual_delegate"]
        assert guard.counts["single_reset_calls"] == guard.counts["single_reset_calls_delegated"] == 1
        assert "delegate_raised" in (tmp_path / "ledger.jsonl").read_text()
    finally:
        guard.close()


def test_literal_region_rotation_and_strict_positive_margins():
    module = collector()
    angle = np.pi / 6
    r = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    point, size = np.array([0.07, 0.02, 0.06]), np.array([0.06, 0.06, 0.02])
    value = module.under_margins(np.zeros(3), r, size, point, 0.1)
    assert np.all(np.array(value["xy_lower"]) > 0) and value["z_lower"] > 0 and value["z_upper"] > 0
    assert np.any(np.abs(r.T @ point)[:2] >= size[:2])  # Transposing changes the actual source rule.
    for point, key in (
        ([0, 0, size[2] - 0.005], "z_lower"),
        ([0, 0, size[2] + 0.1], "z_upper"),
        ([size[0], 0, 0.06], "xy_lower"),
    ):
        value = module.under_margins(np.zeros(3), np.eye(3), size, point, 0.1)
        assert not np.all(np.asarray(value[key]) > 0)  # Exact boundary never qualifies.


def test_fixed_full_denominator_and_official_phase_budget(tmp_path):
    module = collector()
    guard = module.StaticCallGuard(tmp_path / "ledger.jsonl")
    try:
        guard.counts.update(environment_initializations=4, installed_saved_endpoints=16, official_success_calls=20)
        guard.phase_counts["static"]["official_success_calls"] = 16
        guard.phase_counts["setup"]["official_success_calls"] = 4
        module.validate_counts(guard, [None] * 16)
        with pytest.raises(ValueError, match="denominator"):
            module.validate_counts(guard, [None] * 15)
        guard.phase_counts["setup"]["official_success_calls"] = 17
        with pytest.raises(ValueError, match="official success"):
            module.validate_counts(guard, [None] * 16)
    finally:
        guard.close()


def test_capture_preserves_original_mismatch_and_all_active_positive_distance_contact(tmp_path, monkeypatch):
    """Source values are synthetic; this only exercises serialization/order."""
    module = collector()
    import s2a.libero_simulation as simulation

    raw = np.array([0.0, 0.0], dtype=np.float64)
    state = np.array([8.7, 0.25], dtype=np.float64)

    class Sim:
        def __init__(self):
            self.data = SimpleNamespace(
                time=0.0,
                body_xpos=np.array([[0, 0, 0.06], [0, 0, 0]]),
                body_xmat=np.array([np.eye(3), np.eye(3)]),
                body_xquat=np.array([[1, 0, 0, 0], [1, 0, 0, 0]]),
                geom_xpos=np.zeros((2, 3)),
                geom_xmat=np.array([np.eye(3), np.eye(3)]),
                ncon=1,
                contact=[SimpleNamespace(geom1=0, geom2=1, dist=0.001, pos=np.zeros(3), frame=np.eye(3))],
            )
            self.data.get_site_xpos = lambda _: np.zeros(3)
            self.data.get_site_xmat = lambda _: np.eye(3)
            self.model = SimpleNamespace(
                geom_name2id=lambda name: {"wine_geom": 0, "rack_geom": 1}[name],
                geom_id2name=lambda index: ["wine_geom", "rack_geom"][index],
                body_id2name=lambda index: ["wine_body", "rack_body"][index],
                geom_bodyid=np.array([0, 1]),
                geom_size=np.ones((2, 3)),
                geom_type=np.ones(2),
                site_name2id=lambda _: 0,
                site_bodyid=np.array([1]),
                site_size=np.array([[0.1, 0.1, 0.02]]),
            )

        def get_state(self):
            return SimpleNamespace(flatten=lambda: raw.copy())

        def set_state_from_flattened(self, value):
            raw[:] = value
            self.data.time = value[0]

        def forward(self):
            pass

    class Region:
        size = np.array([0.1, 0.1, 0.02])
        parent_name = "wine_rack_1"

        def under(self, pos, rotation, point, other_height=0.10):
            return True

    sim = Sim()
    models = {
        "wine_bottle_1": SimpleNamespace(contact_geoms=["wine_geom"], root_body="wine_body"),
        "wine_rack_1": SimpleNamespace(contact_geoms=["rack_geom"], root_body="rack_body"),
    }
    inner = SimpleNamespace(
        sim=sim,
        obj_body_id={"wine_bottle_1": 0, "wine_rack_1": 1},
        get_object=lambda name: models[name],
        parsed_problem={"goal_state": [["on", "wine_bottle_1", "wine_rack_1_top_region"]]},
        object_states_dict={
            "wine_rack_1_top_region": SimpleNamespace(object_state_type="site", parent_name="wine_rack_1")
        },
        object_sites_dict={"wine_rack_1_top_region": Region()},
        check_contact=lambda a, b: True,
        _eval_predicate=lambda goal: True,
    )
    env = SimpleNamespace(sim=sim, env=inner, check_success=lambda: False)
    monkeypatch.setattr(simulation, "current_observation", lambda _: {})
    monkeypatch.setattr(simulation, "source_pose", lambda _: np.eye(4))
    spec = dict(name="fixture", role="failure", arrays_ref={"fixture": True}, endpoint_receipts_ref={"fixture": True})
    guard = module.StaticCallGuard(tmp_path / "ledger.jsonl")
    try:
        with guard.context("static", "fixture", 174):
            result = module.capture_endpoint(
                env, spec, 174, state, np.eye(4), True, {"sim_state_sha256": state_sha256(state)}, guard
            )
        assert result["original_success"] is True and result["official_success"] is False
        assert result["static_success_reproduced"] is False  # Never replace the original native truth.
        assert (
            result["actual_raw_state_before"]["sha256"]
            == result["actual_raw_state_after"]["sha256"]
            == state_sha256(state)
        )
        assert result["sim_time_before"] == result["sim_time_after"] == 8.7
        assert result["contacts"][0]["dist"] == 0.001 and result["ncon"] == 1
        assert result["atoms"]["contact"] is True  # Positive-distance active contacts are not filtered.
        assert guard.counts["additional_atomic_predicate_calls"] == 3 and guard.counts["installed_saved_endpoints"] == 1
        assert guard.counts["mj_step_calls"] == guard.counts["mj_step2_calls"] == 0
    finally:
        guard.close()


def test_actual_cli_help_and_refuse_existing_output_artifact(tmp_path):
    module = collector()
    path = tmp_path / "receipt.json"
    module.write_once(path, dict(original=True))
    with pytest.raises(FileExistsError):
        module.write_once(path, dict(original=False))
    result = subprocess.run(
        [sys.executable, "scripts/collect_native_terminal_predicate_diagnostic.py", "--help"],
        cwd=REPO,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0 and "--prepared-sha256" in result.stdout


def test_complete_sixteen_mismatch_returns_zero_for_independent_negative_audit(tmp_path):
    module = collector()
    guard = module.StaticCallGuard(tmp_path / "ledger.jsonl")
    try:
        guard.counts.update(environment_initializations=4, installed_saved_endpoints=16, official_success_calls=20)
        guard.phase_counts["static"]["official_success_calls"] = 16
        guard.phase_counts["setup"]["official_success_calls"] = 4
        result = dict(complete=False, passed=False, endpoints=[])
        for index in range(16):
            bundle = module.write_once(tmp_path / f"point{index}.json", dict(static_success_reproduced=index != 3))
            result["endpoints"].append(dict(bundle=bundle))
        code = module.finalize_collection(result, guard)
        assert code == 0 and result["complete"] is True and result["passed"] is False
        assert result["all_saved_states_reproduced"] is False and len(result["endpoints"]) == 16
        assert result["error"]["type"] == "StaticPredicateReproductionMismatch"
        assert guard.counts["environment_initializations"] == 4
        assert guard.counts["installed_saved_endpoints"] == 16
    finally:
        guard.close()


def test_osmesa_preload_and_low_level_guards_precede_runtime_imports(tmp_path, monkeypatch):
    module, events = collector(), []
    import s2a.libero_simulation as simulation

    monkeypatch.chdir(REPO)
    for key, value in dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1").items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)
    library = "/fixture/not-loaded/libOSMesa.so"
    fake_mujoco, fake_guard = object(), object()

    def preload(path, mode):
        assert path == library and mode == module.ctypes.RTLD_GLOBAL
        assert os.environ["MUJOCO_GL"] == os.environ["PYOPENGL_PLATFORM"] == "osmesa"
        events.append("preload_osmesa")

    def import_module(name):
        assert name == "mujoco" and events == ["preload_osmesa"]
        events.append("import_mujoco")
        return fake_mujoco

    def low_level(guard, mujoco):
        assert guard is fake_guard and mujoco is fake_mujoco
        events.append("low_level_guards")

    def configure(policy, output):
        assert policy["osmesa_library"] == library and output == tmp_path / "runtime_bootstrap"
        assert events[-1] == "low_level_guards"
        events.append("configure_runtime")

    monkeypatch.setattr(module.ctypes, "CDLL", preload)
    monkeypatch.setattr(module.importlib, "import_module", import_module)
    monkeypatch.setattr(module, "install_low_level_guards", low_level)
    monkeypatch.setattr(simulation, "configure_simulator", configure)
    monkeypatch.setattr(module, "install_runtime_guards", lambda guard: events.append("runtime_guards"))
    module.bootstrap_static_runtime(dict(osmesa_library=library), tmp_path, fake_guard)
    assert events == ["preload_osmesa", "import_mujoco", "low_level_guards", "configure_runtime", "runtime_guards"]
