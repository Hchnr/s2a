"""Wire archived and synthetic H12 responses without constructing a model."""

import copy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_prefix_callback_backtracking_execution import write_json
from test_prefix_trust_control_execution import CountingDynamics
from test_tail_translation_execution import complete_case, complete_rollout

from s2a.directional_derivative_probe import _canonical
from s2a.osc import OSCInterface
from s2a.terminal_only_tail_repair import ENTRY_FIELDS, probe_terminal_only_tail_repair


@pytest.fixture
def producer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("probe_tracking_v5_terminal_only_tail_repair")


@pytest.mark.parametrize("repairs", [False, True])
def test_real_core_wiring_resets_every_full_call_and_bills_the_repeat(producer, repairs):
    source, case = complete_case(restores=repairs)
    original = copy.deepcopy(source)
    saved = _canonical({key: case[key] for key in ("grid", "historical_anchor")})
    dynamics, calls = CountingDynamics(), []
    result = producer.run_probe(
        source,
        case["grid"],
        case["historical_anchor"],
        None,
        case["interface"],
        dynamics,
        probe_fn=probe_terminal_only_tail_repair,
        rollout_fn=complete_rollout(source, case, calls),
    )
    assert result["prediction_calls"] == result["response_evaluations"] == len(calls) == 14
    assert result["unique_command_count"] == 13
    assert result["numerical_inner_steps"] == 4200
    assert result["query_count"] == dynamics.queries == 4369
    assert result["answer"]["improvement_found"] is repairs
    assert result["answer"]["repeatability"]["exact"]
    assert not result["witness_found"] and not result["answer"]["control_qualified"]
    expected = [case["anchor"], *[row["commands"] for row in case["grid"]["grid_commands"]], case["anchor"]]
    assert [row["commands"].tobytes() for row in calls] == [command.tobytes() for command in expected]
    for row in calls:
        for key, source_key in (
            ("q", "current_q"),
            ("velocity", "current_velocity"),
            ("latch", "latched_rotation_goal"),
        ):
            np.testing.assert_array_equal(row[key], original[source_key])
    for key, value in original.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(source[key], value)
    assert _canonical({key: case[key] for key in ("grid", "historical_anchor")}) == saved


def test_bound_historical_schema_and_full_anchor_recompute_without_a_model(producer):
    # This artifact-backed check uses saved numbers only. It is distinct from
    # independent dynamics replay and produces no new numerical-model response.
    if not producer.INPUT.is_file():
        pytest.skip("Archived 3543 bundle is unavailable outside the registered experiment")
    raw = producer.INPUT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == producer.BUNDLE_SHA
    bundle = json.loads(raw)
    source, history, control = (bundle[key] for key in ("source", "historical_anchor", "control"))
    assert bundle["historical_response_index"] == 3543
    assert set(ENTRY_FIELDS) <= set(history)
    interface = OSCInterface(*(control[key] for key in ("input_min", "input_max", "output_min", "output_max")))
    calls = []

    def saved_response(commands):
        calls.append(commands.copy())
        return copy.deepcopy(history["rollout"])

    result = probe_terminal_only_tail_repair(
        saved_response,
        np.asarray(source["previous_command"])[:6],
        history["commands"],
        interface,
        np.asarray(source["joint_limits"])[:, 0],
        np.asarray(source["joint_limits"])[:, 1],
        source["desired_tcp"],
        source["site_from_tcp"],
        past_inner_error_rad=source["past_inner_error_rad"],
        terminal_normals=source["terminal_normals"],
        grid=bundle["grid"],
        historical_anchor=history,
    )
    assert len(calls) == 14
    assert result["historical_anchor_comparison"]["comparison"]["exact"]
    assert result["historical_anchor_comparison"]["compared_fields"] == list(ENTRY_FIELDS)
    assert result["terminal_values"][0] == history["safety_slack"][-1] < 0
    assert result["evaluated_candidates"][0]["prefix_length"] == 9
    assert not result["improvement_found"] and result["selected"] is None


def test_real_environment_guard_is_separate_from_the_case_controller_without_model_calls(producer, monkeypatch):
    if not producer.INPUT.is_file() or not Path(producer.CONTROL_REF["path"]).is_file():
        pytest.skip("Registered input and environment guard artifacts are unavailable")
    raw = producer.INPUT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == producer.BUNDLE_SHA
    bundle = json.loads(raw)
    case_ref = bundle["source_registration"]["control_ref"]
    assert Path(case_ref["path"]).is_relative_to(producer.REPO)
    assert case_ref != producer.CONTROL_REF
    case_controller = producer.read(case_ref)
    assert "contract" in case_controller and "cv2_guard_registration" not in case_controller

    def forbidden(*args, **kwargs):
        raise AssertionError("Environment guard metadata admission must not construct or execute a model")

    monkeypatch.setattr(producer, "MujocoRobotDynamics", forbidden)
    monkeypatch.setattr(producer, "run_probe", forbidden)
    monkeypatch.setattr(producer, "restore_cv2_environment", forbidden)
    refs = {}
    guard = producer.load_environment_guard(refs)
    assert guard == producer.read(producer.CONTROL_REF)["cv2_guard_registration"]
    for ref in (producer.CONTROL_REF, guard["exact_probe_ref"], *guard["runtime_source_refs"]):
        assert Path(ref["path"]).is_relative_to(producer.REPO)
        assert refs[ref["path"]] == ref


def setup_main(producer, monkeypatch, tmp_path):
    source, case = complete_case(restores=True)
    source["fixed_gripper_state"] = np.zeros(2)
    control = dict(
        **{key: getattr(case["interface"], key) for key in ("input_min", "input_max", "output_min", "output_max")},
        controller_site="synthetic_site",
    )
    guard_probe = write_json(tmp_path / "guard_probe.json", dict(synthetic=True))
    guard_source = write_json(tmp_path / "guard_source.json", dict(synthetic_source=True))
    control_ref = write_json(
        tmp_path / "control.json",
        dict(
            contract=_canonical(control),
            cv2_guard_registration=dict(exact_probe_ref=guard_probe, runtime_source_refs=[guard_source]),
        ),
    )
    states = np.zeros((369, 15))
    states[368, 1:8], states[368, 8:15] = source["current_q"], source["current_velocity"]
    states_path = tmp_path / "states.npz"
    np.savez(states_path, sim_states=states)
    source_reg = dict(control_ref=control_ref, states_ref=producer.reference(states_path))
    bundle = dict(
        source=source,
        source_registration=source_reg,
        control=control,
        historical_anchor=case["historical_anchor"],
        grid=case["grid"],
    )
    root, plan = tmp_path / "result", tmp_path / "plan.md"
    plan.write_text("synthetic test plan\n")
    monkeypatch.setattr(producer, "ROOT", root)
    monkeypatch.setattr(producer, "PLAN", plan)
    monkeypatch.setattr(producer, "CONTROL_REF", control_ref)
    monkeypatch.setattr(producer, "SOURCES", ())
    monkeypatch.setattr(producer, "load_inputs", lambda: (bundle, {}))
    monkeypatch.setattr(producer, "model_registration", lambda reg, refs: (dict(xml_ref=control_ref), {}))
    monkeypatch.setattr(producer, "strict_native_environment", lambda: None)
    monkeypatch.setattr(producer, "restore_cv2_environment", lambda *args: {})
    for key, value in producer.CPU_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    return source, case, root


def test_producer_dry_does_not_construct_model_or_call_predictor(producer, monkeypatch, tmp_path):
    _, _, root = setup_main(producer, monkeypatch, tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("No native model or numerical rollout during dry admission")

    monkeypatch.setattr(producer, "MujocoRobotDynamics", forbidden)
    monkeypatch.setattr(producer, "run_probe", forbidden)
    producer.main(dry=True)
    assert not root.exists()


def test_controller_only_record_is_rejected_during_dry_before_model_construction(producer, monkeypatch, tmp_path):
    _, _, root = setup_main(producer, monkeypatch, tmp_path)
    wrong_record = write_json(tmp_path / "controller_only.json", dict(contract={}))
    monkeypatch.setattr(producer, "CONTROL_REF", wrong_record)

    def forbidden(*args, **kwargs):
        raise AssertionError("Malformed environment guard must fail before model work")

    monkeypatch.setattr(producer, "MujocoRobotDynamics", forbidden)
    monkeypatch.setattr(producer, "run_probe", forbidden)
    with pytest.raises((KeyError, ValueError), match="cv2_guard_registration"):
        producer.main(dry=True)
    assert not root.exists()


def test_partial_rollout_failure_preserves_actual_queries_without_retry(producer, monkeypatch, tmp_path):
    source, case, root = setup_main(producer, monkeypatch, tmp_path)
    dynamics, calls, attempts = CountingDynamics(), [], []
    dynamics.qids = dynamics.dofs = dynamics.joints = np.arange(7)
    dynamics.model = SimpleNamespace(opt=SimpleNamespace(timestep=0.002), nq=7, jnt_range=source["joint_limits"].copy())
    monkeypatch.setattr(producer, "MujocoRobotDynamics", lambda *args: dynamics)
    successful = complete_rollout(source, case, calls)
    real_run_probe = producer.run_probe

    def rollout(*args, **kwargs):
        attempts.append(True)
        if len(attempts) == 3:
            dynamics.queries += 17
            raise RuntimeError("synthetic partial rollout")
        return successful(*args, **kwargs)

    def run(*args, **kwargs):
        return real_run_probe(*args, **kwargs, rollout_fn=rollout)

    monkeypatch.setattr(producer, "run_probe", run)
    with pytest.raises(RuntimeError, match="synthetic partial rollout"):
        producer.main()
    saved = json.loads((root / "verified.json").read_text())
    assert len(attempts) == saved["prediction_calls"] == 3
    assert len(calls) == saved["response_evaluations"] == 2
    # This is the completed-response inner count. The failed response's partial
    # work is retained by the actual dynamics query counter, not guessed away.
    assert saved["numerical_inner_steps"] == 600
    assert saved["query_count"] == dynamics.queries == 1 + 2 * 312 + 17
    assert saved["partial_response_queries"] == 17
    assert saved["numerical_inner_steps_scope"] == "Completed full responses; partial query cost recorded separately"
    assert saved["complete"] is saved["passed"] is saved["automatic_retry"] is False
    assert saved["error"] == "RuntimeError('synthetic partial rollout')"
    assert saved["physics_steps"] == saved["world_model_calls"] == saved["fresh_cases"] == 0
    assert "answer" not in saved and not saved["control_qualified"]
