"""Finite replay semantics; fake dynamics never import a simulator or renderer."""

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from s2a import native_runtime_replay as replay


def saved_branch():
    commands = np.zeros((377, 7))
    states = np.zeros((378, 79))
    periods = []
    now = 0.0
    for step in range(377):
        inner = []
        for _ in range(25):
            now += 0.002
            inner.append(dict(time=now, q=np.zeros(7), qvel=np.zeros(7)))
        states[step + 1, 0] = now
        if step >= 368:
            periods.append(
                dict(
                    step=step,
                    cursor=step - 368,
                    command=commands[step],
                    actual_inner=inner,
                    history_before=np.zeros((25, 7)),
                    history_after=np.zeros((25, 7)),
                    predicted_inner=np.zeros((25, 7)),
                    actual_tcp_error=np.zeros(6),
                    minimum_inner_margin_rad=2.0,
                    success=False,
                    grasp=False,
                )
            )
    return commands, states, periods, np.tile(np.eye(4), (40, 1, 1)), np.eye(4)


class FakeSim:
    def __init__(self):
        self.data = SimpleNamespace(time=0.0, qpos=np.zeros(7), qvel=np.zeros(7))
        self.model = SimpleNamespace(
            jnt_range=np.tile([-2.0, 2.0], (7, 1)), joint_name2id=lambda name: int(name[-1]) - 1
        )
        self.integration = None
        self.inner_steps = 0

    def get_state(self):
        state = np.zeros(79)
        state[0] = self.data.time
        state[1:8] = self.data.qpos
        return state

    def step(self):
        if self.integration is not None:
            return self.integration(None, self.data)
        self.data.time += 0.002
        self.inner_steps += 1


class FakeEnv:
    def __init__(self):
        self.sim = FakeSim()
        self.commands = []
        self.fail_dispatch = None
        self.robot = SimpleNamespace(
            gripper=SimpleNamespace(current_action=np.zeros(1)),
            _ref_joint_pos_indexes=np.arange(7),
            _ref_joint_vel_indexes=np.arange(7),
            controller=SimpleNamespace(
                initial_joint=np.zeros(7),
                goal_pos=np.zeros(3),
                goal_ori=np.eye(3),
                joint_pos=np.zeros(7),
                joint_vel=np.zeros(7),
            ),
        )
        self.robots = [self.robot]

    def step(self, command):
        self.commands.append(command.copy())
        for index in range(25):
            self.sim.step()
            if self.fail_dispatch == len(self.commands) - 1 and index == 2:
                raise RuntimeError("partial dispatch failure")


def observers(events, *, corrupt_endpoint=None):
    def endpoint(env, index):
        events.append(("endpoint", index))
        if index == corrupt_endpoint:
            env.sim.data.qpos[0] = 0.01
        return {}, {}

    return replay.ReplayObservations(
        endpoint,
        lambda env, index, *_: events.append(("anchor", index)),
        lambda env, _: events.append(("camera", len(env.commands))),
        lambda env: events.append(("current", len(env.commands))) or {},
        lambda obs: np.eye(4),
        lambda env: dict(success=False, objects={"akita_black_bowl_1": dict(grasp_contact=False)}),
    )


def receipt():
    return dict(counters={key: dict(attempted=0, completed=0) for key in replay.LIMITS})


def test_full_saved_branch_order_and_counts(tmp_path):
    env, events, result = FakeEnv(), [], receipt()
    branch = saved_branch()
    replay.replay_commands(env, *branch, tmp_path, observers(events), result)
    assert result["checked_endpoints"] == 378
    assert result["checked_inner_samples"] == 225
    assert result["counters"]["control_step"] == dict(attempted=377, completed=377)
    assert len(env.commands) == 377 and env.sim.inner_steps == 9425
    assert result["actual368_state_exact"] is True
    assert [item for item in events if item[0] == "anchor"] == [("anchor", 0), ("anchor", 16), ("anchor", 56)]
    assert [item[1] for item in events if item[0] == "camera"] == list(range(0, 377, 8))
    assert events[:4] == [("current", 0), ("endpoint", 0), ("anchor", 0), ("camera", 0)]
    assert events[-2:] == [("current", 377), ("endpoint", 377)]
    np.testing.assert_array_equal(env.commands, branch[0])
    with np.load(tmp_path / "states.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["sim_states"], branch[1])


@pytest.mark.parametrize("failure_step", [0, 368, 376])
def test_failed_endpoint_never_dispatches_its_command(tmp_path, failure_step):
    env, result = FakeEnv(), receipt()
    with pytest.raises(AssertionError):
        replay.replay_commands(env, *saved_branch(), tmp_path, observers([], corrupt_endpoint=failure_step), result)
    assert len(env.commands) == failure_step
    assert result["counters"]["control_step"] == dict(attempted=failure_step, completed=failure_step)
    assert result["last_checked_endpoint"] == (failure_step - 1 if failure_step else None)
    assert env.sim.data.qpos[0] == 0.01  # Expected future state is never restored.


def test_partial_control_failure_records_samples_and_restores_step(tmp_path):
    env, result = FakeEnv(), receipt()
    env.fail_dispatch = 368
    original = env.sim.step
    with pytest.raises(RuntimeError, match="partial dispatch"):
        replay.replay_commands(env, *saved_branch(), tmp_path, observers([]), result)
    assert result["counters"]["control_step"] == dict(attempted=369, completed=368)
    assert len(result["partial_period_inner_samples"]) == 3
    assert env.sim.inner_steps == 368 * 25 + 3
    assert env.sim.step == original


def test_state_archive_error_does_not_replace_partial_dispatch_error(tmp_path, monkeypatch):
    env, result = FakeEnv(), receipt()
    env.fail_dispatch = 368

    def fail_archive(*args, **kwargs):
        raise OSError("archive unavailable")

    monkeypatch.setattr(np, "savez_compressed", fail_archive)
    with pytest.raises(RuntimeError, match="partial dispatch"):
        replay.replay_commands(env, *saved_branch(), tmp_path, observers([]), result)
    assert result["states_archive_error"]["message"] == "archive unavailable"
    assert result["counters"]["control_step"] == dict(attempted=369, completed=368)
    assert len(result["partial_period_inner_samples"]) == 3


def test_numerical_agreement_does_not_relax_actual_caps(tmp_path):
    env, result = FakeEnv(), receipt()
    branch = list(saved_branch())
    branch[3][0, 0, 3] = 0.01 + 1e-12
    branch[2][0]["actual_tcp_error"][0] = 0.01
    with pytest.raises(ValueError, match="Original actual caps"):
        replay.replay_commands(env, *branch, tmp_path, observers([]), result)
    assert len(env.commands) == 369
    assert result["metrics"][0]["original_caps_satisfied"] is False


@pytest.mark.parametrize("mutation", ["count", "phase", "command", "inner", "gripper", "nonfinite"])
def test_invalid_archive_rejected_before_dispatch(tmp_path, mutation):
    branch = list(saved_branch())
    if mutation == "count":
        branch[0] = branch[0][:-1]
    elif mutation == "phase":
        branch[2][0]["cursor"] = 9
    elif mutation == "command":
        branch[2][0]["command"] = np.ones(7)
    elif mutation == "inner":
        branch[2][0]["actual_inner"].pop()
    elif mutation == "gripper":
        branch[0][368, 6] = 0.1
    else:
        branch[1][1, 0] = np.nan
    env = FakeEnv()
    with pytest.raises((ValueError, AssertionError)):
        replay.replay_commands(env, *branch, tmp_path, observers([]), receipt())
    assert not env.commands


def fake_runtime(tmp_path, monkeypatch, *, fail_native=None):
    # Real constructor control flow, fake MuJoCo and fake renderer only.
    monkeypatch.setattr(replay, "REPO", tmp_path)
    monkeypatch.setattr(replay.bootstrap, "REPO", tmp_path)
    monkeypatch.setattr(replay.bootstrap, "_cpu_environment", lambda: {})
    monkeypatch.setattr(replay.bootstrap, "_controller_check", lambda *args: dict(fixed=True))
    (tmp_path / "outputs").mkdir()
    (tmp_path / "assets").mkdir()
    bddl = tmp_path / "tasks" / "case.bddl"
    bddl.parent.mkdir()
    bddl.write_text("fake")
    calls = dict(native=0, closed=0)

    def integrate(model, data):
        calls["native"] += 1
        data.time += 0.002
        if calls["native"] == fail_native:
            raise RuntimeError("native integration failed after advancing time")

    mujoco = SimpleNamespace(mj_step=integrate, mj_step2=lambda *args: None)

    class Context:
        def __init__(self):
            pass

    class Sim(FakeSim):
        @classmethod
        def from_xml_string(cls, xml):
            value = cls()
            value.integration = lambda *args: mujoco.mj_step(*args)
            return value

    class Inner(FakeEnv):
        def __init__(self):
            super().__init__()
            self.reset()
            self.robot.controller.update = lambda **kwargs: None

        def reset(self):
            self.sim = Sim.from_xml_string("<mujoco />")
            Context()

        def close(self):
            calls["closed"] += 1

    class Wrapper:
        def __init__(self, **kwargs):
            self.env = Inner()

        def __getattr__(self, key):
            return getattr(self.env, key)

        def step(self, command):
            return self.env.step(command)

        def seed(self, seed):
            assert seed == 42

        def reset_from_xml_string(self, xml):
            self.env.sim = Sim.from_xml_string(xml)
            Context()

        def set_init_state(self, value):
            self.sim.data.time = value[0]

    runtime = replay.bootstrap.BootstrapRuntime(
        SimpleNamespace(
            get_benchmark_dict=lambda: {
                "libero_goal": lambda: SimpleNamespace(
                    get_task=lambda _: SimpleNamespace(problem_folder="tasks", bddl_file="case.bddl")
                )
            }
        ),
        lambda _: str(tmp_path),
        Wrapper,
        mujoco,
        Sim,
        Context,
        Inner,
        lambda xml, _: xml,
        None,
        lambda env: dict(backend="osmesa", renderer="llvmpipe"),
    )
    record = dict(
        model_xml="<mujoco />",
        initial_state=np.zeros(79).tolist(),
        initial_state_sha256=hashlib.sha256(np.zeros(79).tobytes()).hexdigest(),
    )
    config = dict(
        seed=42,
        camera_observations=False,
        refresh_controller_after_restore=True,
        render_resolution=256,
        task_suite="libero_goal",
        task_id=8,
    )
    return config, record, runtime, calls


def test_complete_runtime_one_attempt_and_cleanup(tmp_path, monkeypatch):
    config, record, runtime, calls = fake_runtime(tmp_path, monkeypatch)
    result = replay.run_native_replay(
        config,
        record,
        "<mujoco />",
        *saved_branch(),
        tmp_path / "outputs" / "work",
        allowed_asset_roots=[tmp_path / "assets"],
        asset_refs={},
        runtime_factory=lambda *_: runtime,
        observation_factory=lambda: observers([]),
    )
    assert result["passed"] is result["cleanup_complete"] is True
    assert result["control_qualified"] is result["historical_runtime_equivalence_claimed"] is False
    assert result["counters"]["model_construct"] == result["counters"]["context"] == dict(attempted=3, completed=3)
    assert result["counters"]["mj_step"] == dict(attempted=9425, completed=9425)
    assert calls == dict(native=9425, closed=1)


@pytest.mark.parametrize("mismatch", ["controller", "initial_joint"])
def test_changed_initial_controller_stops_before_replay(tmp_path, monkeypatch, mismatch):
    config, record, runtime, calls = fake_runtime(tmp_path, monkeypatch)
    expected_controller = dict(fixed=False) if mismatch == "controller" else dict(fixed=True)
    initial_joint = np.ones(7) if mismatch == "initial_joint" else np.zeros(7)
    with pytest.raises(replay.ReplayFailure):
        replay.run_native_replay(
            config,
            record,
            "<mujoco />",
            *saved_branch(),
            tmp_path / "outputs" / "work",
            allowed_asset_roots=[tmp_path / "assets"],
            asset_refs={},
            expected_controller=expected_controller,
            expected_initial_joint=initial_joint,
            runtime_factory=lambda *_: runtime,
            observation_factory=lambda: observers([]),
        )
    assert calls == dict(native=0, closed=1)


def test_runtime_partial_native_failure_counts_and_time(tmp_path, monkeypatch):
    config, record, runtime, calls = fake_runtime(tmp_path, monkeypatch, fail_native=9203)
    with pytest.raises(replay.ReplayFailure) as failure:
        replay.run_native_replay(
            config,
            record,
            "<mujoco />",
            *saved_branch(),
            tmp_path / "outputs" / "work",
            allowed_asset_roots=[tmp_path / "assets"],
            asset_refs={},
            runtime_factory=lambda *_: runtime,
            observation_factory=lambda: observers([]),
        )
    result = failure.value.receipt
    assert result["counters"]["mj_step"] == dict(attempted=9203, completed=9202)
    assert result["counters"]["control_step"] == dict(attempted=369, completed=368)
    assert result["last_native_step"]["after"] > result["last_native_step"]["before"]
    assert result["cleanup_complete"] is True and calls["closed"] == 1
    assert "native integration failed" in result["error"]["message"]


def test_failed_replay_cleanup_cannot_dispatch_and_preserves_primary_error(tmp_path, monkeypatch):
    config, record, runtime, calls = fake_runtime(tmp_path, monkeypatch, fail_native=9203)

    def bad_close(self):
        calls["closed"] += 1
        self.step(np.zeros(7))

    monkeypatch.setattr(runtime.env_base_class, "close", bad_close)
    with pytest.raises(replay.ReplayFailure) as failure:
        replay.run_native_replay(
            config,
            record,
            "<mujoco />",
            *saved_branch(),
            tmp_path / "outputs" / "work",
            allowed_asset_roots=[tmp_path / "assets"],
            asset_refs={},
            runtime_factory=lambda *_: runtime,
            observation_factory=lambda: observers([]),
        )
    result = failure.value.receipt
    assert calls == dict(native=9203, closed=1)
    assert result["forbidden_control_attempts"] == 1
    assert result["cleanup_complete"] is False
    assert "No control dispatch" in result["cleanup_error"]
    assert "native integration failed" in result["error"]["message"]
    assert result["failure_stage"] == "replay"


def test_cli_dry_default_and_qa_gate(monkeypatch, capsys):
    import verify_native_runtime_replay as cli

    for key, value in cli.CPU_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(cli, "read", lambda _: dict(output_dir="unused"))
    monkeypatch.setattr(cli, "reference", lambda _: {})
    monkeypatch.setattr(cli, "inspect_inputs", lambda _: dict(refs={}))
    monkeypatch.setattr(cli, "run_native_replay", lambda *a, **kw: pytest.fail("Dry admission constructed a runtime"))
    cli.main()
    assert json.loads(capsys.readouterr().out)["runtime_loaded"] is False
    with pytest.raises(ValueError, match="component QA"):
        cli.main(execute=True)


def test_cli_forbids_retired_and_user_file_without_reading():
    import verify_native_runtime_replay as cli

    for path in ("/share/project/never/read", str(cli.REPO / "long_joint_inverse.py")):
        with pytest.raises(ValueError, match="Forbidden source"):
            cli.reference(path)


@pytest.mark.parametrize("mutation", ["environment", "command", "source"])
def test_qa_binding_rejects_changed_execution_contract(monkeypatch, mutation):
    import verify_native_runtime_replay as cli

    qa = dict(
        complete=True,
        passed=True,
        actual_wait=True,
        reaped=True,
        source_unchanged=True,
        returncode=0,
        signal_sent=False,
        automatic_retry=False,
        environment=dict(cli.CPU_ENVIRONMENT),
        python_version="3.11.15",
        python_executable=str(cli.NATIVE_PYTHON),
        command=[str(cli.NATIVE_PYTHON), "-m", "pytest", "-q", *cli.QA_TESTS],
        source_sha256={str(cli.REPO / path): "bound" for path in cli.SOURCE_PATHS},
        source_count=5,
        test_module_count=1,
        log_ref={},
        driver_ref={},
    )
    monkeypatch.setattr(cli, "read", lambda _: qa)
    monkeypatch.setattr(cli, "reference", lambda path: dict(path=str(path), sha256="bound"))
    monkeypatch.setattr(cli, "checked", lambda _: None)
    assert cli.verify_qa({}) is qa
    if mutation == "environment":
        qa["environment"]["CUDA_VISIBLE_DEVICES"] = "0"
    elif mutation == "command":
        qa["command"].extend(["-k", "nothing"])
    else:
        qa["source_sha256"][str(cli.REPO / cli.SOURCE_PATHS[0])] = "changed"
    with pytest.raises(ValueError):
        cli.verify_qa({})


def test_saved_input_admission_uses_current_files_only():
    import verify_native_runtime_replay as cli

    config = json.loads(cli.CONFIG.read_text())
    inputs = cli.inspect_inputs(config)
    assert inputs["commands"].shape == (377, 7)
    assert inputs["states"].shape == (378, 79)
    assert len(inputs["periods"]) == 9
    assert all(not path.startswith("/share/project/") for path in inputs["refs"])
    assert all(str(cli.REPO / relative) in inputs["refs"] for relative in cli.ADDITIONAL_RUNTIME_SOURCES)


@pytest.mark.parametrize("key,value", [("physics_budget", 378), ("absolute_tolerance", 1e-8), ("world_model_calls", 1)])
def test_metadata_rejects_changed_protocol(key, value):
    import verify_native_runtime_replay as cli

    config = copy.deepcopy(json.loads(cli.CONFIG.read_text()))
    config[key] = value
    with pytest.raises(ValueError):
        cli.inspect_inputs(config)
