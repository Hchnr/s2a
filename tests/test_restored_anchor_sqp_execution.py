"""CPU integration of exact initial admission, actual accounting and ownership."""

import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import test_prefix_callback_backtracking_execution as prior
from test_prefix_callback_backtracking_execution import write_json
from test_prefix_optimization_reserve_plan import inputs, response
from test_prefix_trust_control_execution import (
    CountingDynamics,
    captured_rollout,
    source_fixture,
    three_response_solver,
)

from s2a.accepted_point_sqp_plan import solve_accepted_point_sqp_response


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return SimpleNamespace(
        producer=importlib.import_module("probe_tracking_v5_restored_anchor_sqp"),
        owner=importlib.import_module("execute_tracking_v5_restored_anchor_sqp"),
        evidence=importlib.import_module("s2a.restored_anchor_sqp_evidence"),
        core=importlib.import_module("s2a.restored_anchor_sqp_plan"),
    )


def guard(ref):
    return dict(
        checked=True,
        passed=True,
        exact=True,
        hook_calls=1,
        request_index=0,
        response_index=0,
        request_count=1,
        response_count=1,
        stages_before=0,
        iterations_before=0,
        qp_iterations_before=0,
        expected_ref=ref,
        comparison=dict(exact=True),
    )


def full_fixture():
    arguments = inputs(horizon=12)
    expected = solve_accepted_point_sqp_response(**arguments)["initial"]
    source, _, _ = source_fixture()
    source.update(
        previous_command=np.r_[arguments["previous"], 0.0],
        desired_tcp=arguments["desired"],
        past_inner_error_rad=arguments["past_inner_error_rad"],
        terminal_normals=arguments["terminal_normals"],
    )
    calls = []

    def rollout(q, velocity, commands, interface, initial, kp, kd, dynamics, low, high, *, latched_rotation_goal):
        calls.append(commands.copy())
        dynamics.queries += 312
        return response(commands)

    return source, arguments["warm"], arguments["interface"], expected, calls, rollout


def test_real_solver_exact_first_full_record_is_billed_once_with_no_free_warm(modules, monkeypatch):
    source, warm, interface, expected, calls, rollout = full_fixture()
    monkeypatch.setattr(modules.core, "solve_epigraph_qp", lambda *_a, **_kw: pytest.fail("initial is a witness"))
    dynamics = CountingDynamics()
    result = modules.producer.run_search(
        source, warm, interface, dynamics, expected_initial=expected, rollout_fn=rollout
    )
    assert result["response_evaluations"] == result["prediction_calls"] == len(calls) == 1
    assert result["query_count"] == dynamics.queries == 313 and result["numerical_inner_steps"] == 300
    assert result["answer"]["witness_found"] is True and result["total_iterations"] == 0
    assert result["initial_record_guard"]["exact"] is True
    assert result["initial_record_guard"]["expected_ref"] == modules.evidence.TAIL_INITIAL_REF


@pytest.mark.parametrize("field", ["nonminimum_slack", "metric", "bool_label"])
def test_first_complete_record_mismatch_archives_actual_single_response_and_no_qp(modules, monkeypatch, field):
    source, warm, interface, expected, calls, rollout = full_fixture()
    expected = copy.deepcopy(expected)
    if field == "nonminimum_slack":
        expected["audit"]["groups"]["inner_upper"]["slack"][0, 0] += 1e-12
    elif field == "metric":
        expected["pose_loss"] += 1e-12
    else:
        expected["feasible"] = 1
    monkeypatch.setattr(modules.core, "solve_epigraph_qp", lambda *_a, **_kw: pytest.fail("guard must precede QP"))
    progress, dynamics = {}, CountingDynamics()
    with pytest.raises(modules.core.InitialRecordAdmissionError):
        modules.producer.run_search(
            source, warm, interface, dynamics, expected_initial=expected, rollout_fn=rollout, progress=progress
        )
    assert progress["answer"] is None and progress["total_iterations"] == 0
    assert progress["response_evaluations"] == progress["prediction_calls"] == len(calls) == 1
    assert progress["numerical_inner_steps"] == 300 and progress["query_count"] == dynamics.queries == 313
    failure = progress["initial_record_failure"]
    assert failure["answer"] is None and failure["phases"] == [] and failure["total_qp_iterations"] == 0
    assert failure["observed_entry"]["feasible"] is True
    assert len(failure["request_ledger"]) == len(failure["evaluated_candidates"]) == 1
    assert failure["comparison"]["exact"] is False


@pytest.mark.parametrize("defect", ["box", "rate", "nan", "wrong_expected_command"])
def test_invalid_warm_or_expected_command_is_rejected_before_current_query(modules, defect):
    source, warm, interface, expected, calls, rollout = full_fixture()
    if defect == "box":
        warm[0, 0] = 1.1
    elif defect == "rate":
        warm[1, 0] = 0.3
    elif defect == "nan":
        warm[0, 0] = np.nan
    else:
        expected["commands"][0, 0] = 1e-6
    dynamics = CountingDynamics()
    with pytest.raises(ValueError):
        modules.producer.run_search(source, warm, interface, dynamics, expected_initial=expected, rollout_fn=rollout)
    assert calls == [] and dynamics.queries == 0


def test_admission_preserves_legal_roundoff_outside_tight_box_bytes(modules):
    source, warm, interface = source_fixture()
    warm[0, 0] = np.nextafter(0.25, np.inf)
    saved = warm.tobytes()
    admitted = modules.producer.admit_warm(source, warm, interface)
    assert admitted.tobytes() == warm.tobytes() == saved
    assert not np.shares_memory(admitted, warm)
    assert np.clip(warm[0], -0.25, 0.25).tobytes() != warm[0].tobytes()


def test_every_later_response_resets_actual_state_and_latch_and_keeps_fixed_budget(modules):
    source, warm, interface = source_fixture()
    before = copy.deepcopy(source)
    dynamics, calls, solver_calls = CountingDynamics(), [], []
    standin = three_response_solver(source, dynamics, solver_calls)
    expected = dict(commands=warm.copy())

    def solver(*args, **kwargs):
        assert kwargs["max_stages"] == kwargs["max_iterations"] == 60
        proof = kwargs["initial_record_hook"](expected)
        assert proof["exact"] is True
        result = standin(*args, **kwargs)
        result["initial_record_guard"] = guard(modules.evidence.TAIL_INITIAL_REF)
        return result

    result = modules.producer.run_search(
        source,
        warm,
        interface,
        dynamics,
        expected_initial=expected,
        solver_fn=solver,
        rollout_fn=captured_rollout(source, calls),
    )
    assert result["response_evaluations"] == len(calls) == 3 and result["query_count"] == 937
    for row in calls:
        np.testing.assert_array_equal(row["q"], before["current_q"])
        np.testing.assert_array_equal(row["velocity"], before["current_velocity"])
        np.testing.assert_array_equal(row["latch"], before["latched_rotation_goal"])


def test_partial_technical_response_retains_actual_query_count_without_fabricated_completed_steps(modules):
    source, warm, interface, expected, calls, _ = full_fixture()
    progress, dynamics = {}, CountingDynamics()

    def interrupted(*args, **kwargs):
        dynamics.queries += 17
        raise RuntimeError("synthetic response failure after partial work")

    with pytest.raises(RuntimeError, match="partial work"):
        modules.producer.run_search(
            source, warm, interface, dynamics, expected_initial=expected, rollout_fn=interrupted, progress=progress
        )
    assert progress["answer"] is None and progress["query_count"] == 18
    assert (
        progress["prediction_calls"] == 1 and progress["response_evaluations"] == progress["numerical_inner_steps"] == 0
    )
    assert "initial_record_failure" not in progress


def decorate(result, stage, ref):
    result["prefix_optimization_margin"] = 1e-4
    if stage == "diagnose":
        result["initial_record_guard"] = guard(ref)
        result["answer"].update(
            method="restored_anchor_lagrangian_sqp",
            prefix_optimization_margin=1e-4,
            qp_iteration_limit=100,
            qp_certificate_tolerance=1e-6,
            total_qp_iterations=7,
            initial_record_guard=guard(ref),
            witness_found=False,
        )
        for row in result["answer"]["evaluated_candidates"]:
            row["feasible"] = False
    else:
        result["initial_record_check"] = dict(passed=True)
        result.update(
            ledger_check=dict(passed=True, witness_indices=[], initial_record_check=dict(passed=True)),
            native_quaternion_jit_signatures=["synthetic signature"],
            witness_found=False,
            source_refs_unchanged=[],
        )
    return result


def owner_run(modules, monkeypatch, tmp_path, stage, **options):
    original = prior.receipt
    monkeypatch.setattr(
        prior,
        "receipt",
        lambda stage, count=3: decorate(original(stage, count), stage, modules.evidence.TAIL_INITIAL_REF),
    )
    run = prior.owner_run(modules, monkeypatch, tmp_path, stage, **options)
    monkeypatch.setattr(modules.owner, "SOURCES", ())
    if stage == "check":
        owner, _ = modules.owner.closed_owner(None, {})
        old_ref = owner["registration_ref"]
        data = modules.owner.read(old_ref)
        data.update(
            tail_closure_ref=modules.evidence.TAIL_CLOSURE_REF, expected_initial_ref=modules.evidence.TAIL_INITIAL_REF
        )
        owner["registration_ref"] = write_json(Path(old_ref["path"]), data)
    original_child = modules.owner.owned_command

    def child(command, environment, capture, *, timeout):
        original_child(command, environment, capture, timeout=timeout)
        root = run.root(check=stage == "check")
        registration = dict(
            plan_ref=modules.owner.reference(modules.owner.PLAN),
            expected_initial_ref=modules.evidence.TAIL_INITIAL_REF,
            seed=42,
            source_refs=[],
            input_refs=[],
            algorithm_math_proof=dict(passed=True),
        )
        if stage == "check":
            registration.update(
                producer_ref=modules.owner.reference(run.root() / "verified.json"),
                producer_owner_ref=modules.owner.reference(run.root(owner=True) / "result.json"),
            )
        reg_ref = write_json(root / "registration.json", registration)
        receipt = json.loads((root / "verified.json").read_text())
        receipt["registration_ref"] = reg_ref
        write_json(root / "verified.json", receipt)
        return {
            name: write_json(capture / (name + ".json"), dict(synthetic=name)) for name in ("launch", "exit", "release")
        }

    monkeypatch.setattr(modules.owner, "owned_command", child)
    return run


@pytest.mark.parametrize("stage,timeout", [("diagnose", 1200), ("check", 5400)])
def test_owner_launches_only_registered_stage_with_fixed_cpu_and_budget(modules, monkeypatch, tmp_path, stage, timeout):
    run = owner_run(modules, monkeypatch, tmp_path, stage)
    run.run()
    assert len(run.calls) == 1 and run.calls[0]["timeout"] == timeout
    assert run.calls[0]["command"] == modules.owner.entry_command(stage)
    assert "restored_anchor_sqp.py" in run.calls[0]["command"][1]
    env = run.calls[0]["environment"]
    assert all(env[key] == value for key, value in modules.owner.CPU_ENVIRONMENT.items())
    result = json.loads((run.root(check=stage == "check", owner=True) / "result.json").read_text())
    assert result["complete"] is result["passed"] is True
    refs = {ref["path"]: ref for ref in result["source_refs_unchanged"]}
    assert refs[result["result_ref"]["path"]] == result["result_ref"]
    assert all(ref["path"] in refs for ref in result["owned_command"].values())


@pytest.mark.parametrize(
    "options,match,launched",
    [({"partial": True}, "No partial replay", 1), ({"different_qa": True}, "identical frozen", 0)],
)
def test_check_owner_rejects_partial_replay_or_different_qa(modules, monkeypatch, tmp_path, options, match, launched):
    run = owner_run(modules, monkeypatch, tmp_path, "check", **options)
    with pytest.raises(ValueError, match=match):
        run.run()
    assert len(run.calls) == launched


@pytest.mark.parametrize(
    "key,value",
    [
        ("exact", False),
        ("hook_calls", 2),
        ("request_count", 2),
        ("response_count", True),
        ("stages_before", 1),
        ("expected_ref", {}),
    ],
)
def test_receipt_cannot_bypass_or_reorder_first_record_gate(modules, key, value):
    saved = decorate(prior.receipt("diagnose"), "diagnose", modules.evidence.TAIL_INITIAL_REF)
    saved["initial_record_guard"][key] = value
    with pytest.raises(ValueError):
        modules.owner.validate_receipt(saved, "diagnose")


@pytest.mark.parametrize("defect", ["ledger_failed", "jit_empty", "different_guard", "false_witness"])
def test_checker_receipt_requires_completed_ledger_jit_and_strict_witness(modules, defect):
    saved = decorate(prior.receipt("check"), "check", modules.evidence.TAIL_INITIAL_REF)
    if defect == "ledger_failed":
        saved["ledger_check"]["passed"] = False
    elif defect == "jit_empty":
        saved["native_quaternion_jit_signatures"] = []
    elif defect == "different_guard":
        saved["ledger_check"]["initial_record_check"] = dict(passed=False)
    else:
        saved["witness_found"] = True
    with pytest.raises(ValueError):
        modules.owner.validate_receipt(saved, "check")


@pytest.mark.parametrize(
    "defect", [None, "missing_source", "missing_test", "old_subset_changed", "test_filter", "no_reap"]
)
def test_native_qa_binds_every_new_source_test_and_unchanged_prior_subset(modules, monkeypatch, tmp_path, defect):
    owner = modules.owner
    monkeypatch.setattr(owner, "REPO", tmp_path)
    monkeypatch.setattr(owner, "NATIVE_PYTHON", tmp_path / "python")
    sources = {path: write_json(tmp_path / path, dict(source=path))["sha256"] for path in owner.QA_SOURCES}
    old = write_json(tmp_path / "oldqa.json", dict(source_sha256={p: sources[p] for p in owner.PRIOR_QA_SOURCES}))
    monkeypatch.setattr(owner, "PRIOR_QA_REF", old)
    qa = dict(
        complete=True,
        passed=True,
        source_unchanged=True,
        prior_sources_unchanged=True,
        actual_wait=True,
        reaped=True,
        signal_sent=False,
        automatic_retry=False,
        returncode=0,
        source_count=110,
        test_module_count=33,
        source_sha256=sources,
        prior_native_component_qa_ref=old,
        prior_source_count=100,
        command=[str(owner.NATIVE_PYTHON), "-m", "pytest", "-q", *owner.QA_TESTS, "--basetemp=outputs/qa"],
    )
    monkeypatch.setattr(owner, "verify_accepted_qa", owner.read)
    if defect == "missing_source":
        del qa["source_sha256"][modules.evidence.SOURCES[0]]
    elif defect == "missing_test":
        qa["command"].remove(owner.QA_TESTS[-1])
    elif defect == "old_subset_changed":
        qa["source_sha256"][owner.PRIOR_QA_SOURCES[0]] = "0" * 64
    elif defect == "test_filter":
        qa["command"].extend(["-k", "small_subset"])
    elif defect == "no_reap":
        qa["reaped"] = False
    ref = write_json(tmp_path / "qa.json", qa)
    if defect is None:
        assert owner.verify_native_qa(ref) == qa
    else:
        with pytest.raises(ValueError):
            owner.verify_native_qa(ref)


def test_real_closed_tail_selection_is_hash_bound_without_any_response(modules):
    evidence = modules.evidence
    rows = [
        evidence.read(ref)
        for ref in (
            evidence.TAIL_PRODUCER_REF,
            evidence.TAIL_CHECK_REF,
            evidence.TAIL_CLOSURE_REF,
            evidence.TAIL_ANALYSIS_REF,
        )
    ]
    selected = evidence.validate_tail_archive(*rows)
    assert selected["prefix_length"] == 9 and selected["feasible"] is False
    assert np.asarray(selected["commands"]).shape == (12, 6)
    for key, value in (("sufficient_reserve_restored", False), ("selected_response_index", 4)):
        original = rows[0]["answer"][key]
        rows[0]["answer"][key] = value
        with pytest.raises(ValueError):
            evidence.validate_tail_archive(*rows)
        rows[0]["answer"][key] = original


@pytest.mark.parametrize("dry", [True, False])
def test_actual_entry_preflight_and_failure_publication_use_tail_record_and_actual368(
    modules, monkeypatch, tmp_path, dry, capsys
):
    producer = modules.producer
    source, warm, interface, expected, calls, rollout = full_fixture()
    expected["pose_loss"] += 1e-12  # Finite complete record, deliberately different after one response.
    source["registration_ref"] = write_json(tmp_path / "source.json", {})
    source["fixed_gripper_state"] = {"finger1": [0.02, 0.01], "finger2": [-0.02, -0.01]}
    control = dict(
        input_min=[-1.0] * 6,
        input_max=[1.0] * 6,
        output_min=[-1.0] * 6,
        output_max=[1.0] * 6,
        controller_site="same_site",
    )
    states = np.zeros((369, 19))
    states[368] = np.r_[0.0, source["current_q"], 0.02, -0.02, source["current_velocity"], 0.01, -0.01]
    state_path = tmp_path / "states.npz"
    np.savez(state_path, sim_states=states)
    registration = dict(
        control_ref=write_json(tmp_path / "control.json", dict(contract=control)),
        xml_ref=write_json(tmp_path / "model.xml", {}),
        states_ref=producer.reference(state_path),
    )
    saved_guard = write_json(
        tmp_path / "guard.json", dict(cv2_guard_registration=dict(exact_probe_ref={}, runtime_source_refs=[]))
    )
    monkeypatch.setattr(
        producer, "load_prerequisites", lambda: (source, registration, dict(answer=dict(selected=expected)), {})
    )
    monkeypatch.setattr(producer, "model_registration", lambda reg, refs: (reg, dict(synthetic=True)))
    monkeypatch.setattr(producer, "strict_native_environment", lambda: None)
    monkeypatch.setattr(producer, "restore_cv2_environment", lambda *args: {})
    monkeypatch.setattr(producer, "CONTROL_REF", saved_guard)
    monkeypatch.setattr(producer, "SOURCES", ())
    root = tmp_path / "published"
    monkeypatch.setattr(producer, "case_root", lambda _: root)
    for key, value in producer.CPU_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    models = []

    def model(*args):
        dynamics = CountingDynamics()
        dynamics.qids = dynamics.dofs = dynamics.joints = np.arange(7)
        dynamics.model = SimpleNamespace(
            opt=SimpleNamespace(timestep=0.002),
            nq=9,
            jnt_range=np.vstack((source["joint_limits"], [[-1, 1]] * 2)),
            jnt_qposadr=np.arange(9),
            jnt_dofadr=np.arange(9),
        )
        dynamics.mj = SimpleNamespace(
            mjtObj=SimpleNamespace(mjOBJ_JOINT=0), mj_name2id=lambda *args: {"finger1": 7, "finger2": 8}[args[-1]]
        )
        models.append(dynamics)
        return dynamics

    monkeypatch.setattr(producer, "MujocoRobotDynamics", model)
    run_search = producer.run_search
    monkeypatch.setattr(producer, "run_search", lambda *args, **kwargs: run_search(*args, **kwargs, rollout_fn=rollout))
    if dry:
        producer.main(dry=True)
        receipt = json.loads(capsys.readouterr().out.splitlines()[-1])
        assert receipt["dry_run"] is receipt["passed"] is True
        assert receipt["new_responses"] == receipt["new_queries"] == 0
        assert models == calls == [] and not root.exists()
    else:
        with pytest.raises(modules.core.InitialRecordAdmissionError):
            producer.main()
        result = json.loads((root / "verified.json").read_text())
        reg = json.loads((root / "registration.json").read_text())
        assert result["complete"] is result["passed"] is False
        assert result["answer"] is None and result["initial_record_failure"]["observed_entry"]["pose_loss"] == 0
        assert result["response_evaluations"] == len(calls) == len(models) == 1
        assert result["query_count"] == 313 and result["numerical_inner_steps"] == 300
        assert reg["expected_initial_ref"] == modules.evidence.TAIL_INITIAL_REF
        assert reg["algorithm_math_proof"]["passed"] is True and reg["endpoint"] == 368
        assert reg["response_budget"] == 4000 and reg["warm_locator"] == "answer.evaluated_candidates[5].commands"
