"""Synthetic admission, reset and billed-repeat wiring without native responses."""

import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_accepted_point_sqp_execution import native_qa
from test_prefix_callback_backtracking_execution import write_json
from test_prefix_trust_control_execution import CountingDynamics, captured_rollout, source_fixture


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return SimpleNamespace(
        producer=importlib.import_module("probe_tracking_v5_directional_derivative"),
        owner=importlib.import_module("execute_tracking_v5_directional_derivative"),
        evidence=importlib.import_module("s2a.directional_derivative_evidence"),
    )


def directions():
    result = np.zeros((3, 12, 6))
    for i in range(3):
        result[i, 1, i] = 0.004
    return result


def complete_synthetic_rollout(source, calls):
    capture = captured_rollout(source, calls)

    def rollout(*args, **kwargs):
        command = args[2].copy()
        capture(*args, **kwargs)
        pose = np.tile(np.eye(4), (12, 1, 1))
        pose[:, :3, 3] = command[:, :3] / 100
        return dict(
            inner_joint_position=np.zeros((300, 7)),
            inner_joint_velocity=np.zeros((300, 7)),
            joint_position=np.zeros((12, 7)),
            joint_velocity=np.zeros((12, 7)),
            pose=pose,
            twist=np.zeros((12, 6)),
            periods=[dict(goal=value.copy()) for value in pose],
        )

    return rollout


@pytest.mark.parametrize("parallel,expected_calls", [(False, 80), (True, 28)])
def test_default_probe_bills_grid_cache_and_resets_final_repeat(modules, parallel, expected_calls):
    source, anchor, interface = source_fixture()
    original, anchor_bytes = copy.deepcopy(source), anchor.tobytes()
    saved_directions = directions()
    if parallel:
        saved_directions[:] = saved_directions[0]
    before_directions = saved_directions.copy()
    dynamics, calls = CountingDynamics(), []
    result = modules.producer.run_probe(
        source,
        anchor,
        saved_directions,
        interface,
        dynamics,
        rollout_fn=complete_synthetic_rollout(source, calls),
    )
    assert result["prediction_calls"] == result["response_evaluations"] == len(calls) == expected_calls
    assert result["unique_command_count"] == expected_calls - 1
    assert result["numerical_inner_steps"] == expected_calls * 300
    assert result["query_count"] == dynamics.queries == 1 + expected_calls * 312
    assert result["answer"]["planned_request_count"] == 80
    assert sum(row["outcome"] == "cached" for row in result["answer"]["request_ledger"]) == 80 - expected_calls
    assert result["total_iterations"] == 0 and result["answer"]["repeatability"]["exact"]
    assert result["answer"]["source_stages"] == [24, 25, 26]
    assert len(result["answer"]["step_sizes"]) == 13
    assert calls[0]["commands"].tobytes() == calls[-1]["commands"].tobytes() == anchor_bytes == anchor.tobytes()
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
    np.testing.assert_array_equal(saved_directions, before_directions)


@pytest.mark.parametrize(
    "defect,expected_calls,match",
    [
        ("after_repeat", 2, "after the final anchor repeat"),
        ("nonanchor_duplicate", 2, "Only the final anchor"),
        ("wrong_first", 0, "First billed response"),
        ("over_budget", 80, "fixed80 budget"),
        ("bad_rate", 1, "command box and rate"),
    ],
)
def test_raw_predictor_rejects_unregistered_duplicate_budget_and_command_calls(modules, defect, expected_calls, match):
    source, anchor, interface = source_fixture()
    dynamics, calls, progress = CountingDynamics(), [], {}

    def invalid(predict, previous, start, *args, **kwargs):
        assert kwargs["max_responses"] == 80 and kwargs["source_stages"] == (24, 25, 26)
        if defect == "wrong_first":
            changed = start.copy()
            changed[0, 3] = 0.0  # Equal numeric value, different signed-zero bytes.
            predict(changed)
            return
        predict(start)
        changed = start.copy()
        changed[1, 0] += 0.01
        if defect == "after_repeat":
            predict(start.copy())
            predict(changed)
        elif defect == "nonanchor_duplicate":
            predict(changed)
            predict(changed.copy())
        elif defect == "bad_rate":
            changed[1, 0] = 0.6
            predict(changed)
        else:
            for index in range(1, 81):
                changed = start.copy()
                changed[1, 0] += index * 1e-5
                predict(changed)

    with pytest.raises(ValueError, match=match):
        modules.producer.run_probe(
            source,
            anchor,
            directions(),
            interface,
            dynamics,
            probe_fn=invalid,
            rollout_fn=captured_rollout(source, calls),
            progress=progress,
        )
    assert len(calls) == progress["response_evaluations"] == expected_calls
    assert progress["prediction_calls"] == expected_calls + 1
    assert progress["numerical_inner_steps"] == expected_calls * 300
    assert progress["query_count"] == 1 + expected_calls * 312


def test_wrapper_does_not_hide_wrong_per_response_query_count(modules):
    source, anchor, interface = source_fixture()
    dynamics, calls, progress = CountingDynamics(), [], {}

    def once(predict, previous, start, *args, **kwargs):
        predict(start)

    with pytest.raises(ValueError, match="bills300 inner"):
        modules.producer.run_probe(
            source,
            anchor,
            directions(),
            interface,
            dynamics,
            probe_fn=once,
            rollout_fn=captured_rollout(source, calls, query_error=1),
            progress=progress,
        )
    assert progress["prediction_calls"] == progress["response_evaluations"] == 1
    assert progress["query_count"] == 314


def receipt(stage, count=3):
    result = dict(
        complete=True,
        passed=True,
        experiment="directional_derivative_consistency",
        response_evaluations=count,
        unique_command_count=count - 1,
        numerical_inner_steps=300 * count,
        query_count=1 + 312 * count,
        total_iterations=0,
        control_qualified=False,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        actual_consumed_reference_count=0,
        source_refs_unchanged=[],
    )
    if stage == "check":
        result.update(
            verified_responses=count,
            mj_forward_queries=1 + 312 * count,
            repeated_responses_checked=1,
            ledger_check=dict(passed=True),
            boundary_classification_disagreements=[],
        )
        return result
    anchor = np.zeros((12, 6))
    commands = [anchor.copy() for _ in range(count)]
    for index in range(1, count - 1):
        commands[index][1, 0] = index / 10000
    ledger = []
    for index in range(80):
        role = "anchor" if index == 0 else "anchor_repeat" if index == 79 else "grid"
        charged = index < count - 1 or index == 79
        response_index = count - 1 if index == 79 else min(index, count - 2)
        ledger.append(
            dict(
                role=role,
                charged=charged,
                response_index=response_index,
                outcome="repeated" if index == 79 else "computed" if charged else "cached",
            )
        )
    owner = importlib.import_module("execute_tracking_v5_directional_derivative")
    result.update(
        prediction_calls=count,
        answer=dict(
            method="fixed_directional_derivative_probe",
            horizon=12,
            total_iterations=0,
            max_responses=80,
            planned_request_count=80,
            source_stages=[24, 25, 26],
            step_sizes=list(owner.STEP_SIZES),
            response_evaluations=count,
            unique_command_count=count - 1,
            control_qualified=False,
            anchor_commands=anchor.tolist(),
            request_ledger=ledger,
            evaluated_candidates=[dict(commands=value.tolist()) for value in commands],
            repeatability=dict(initial_response_index=0, repeated_response_index=count - 1),
        ),
    )
    return result


def owned_metadata(owner, root, stage, result_ref, registration, *, defect=None):
    registration_ref = write_json(root / "registration.json", registration)
    launch = write_json(
        root / "launch.json",
        dict(
            automatic_retry=False,
            selected_environment=dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
            command=owner.entry_command(stage),
            child=dict(pid=1234, start_ticks=5678),
        ),
    )
    exited = write_json(
        root / "exit.json",
        dict(
            automatic_retry=False,
            actual_wait=defect != "no_wait",
            returncode=0,
        ),
    )
    release = write_json(
        root / "release.json",
        dict(
            reaped=defect != "not_reaped",
            returncode=0,
            signal_sent=defect == "signal",
            original_error=None,
            pid=1234,
            start_ticks=5678,
        ),
    )
    return dict(
        complete=True,
        passed=True,
        experiment=owner.EXPERIMENT,
        stage=stage,
        registration_ref=registration_ref,
        result_ref=result_ref,
        owned_command=dict(launch=launch, exit=exited, release=release),
    )


def owner_run(modules, monkeypatch, tmp_path, stage, *, defect=None):
    owner = modules.owner
    monkeypatch.setattr(owner, "REPO", tmp_path)
    monkeypatch.setattr(owner, "NATIVE_PYTHON", tmp_path / "native_python")
    plan = write_json(tmp_path / "plan.json", {})
    monkeypatch.setattr(owner, "PLAN", Path(plan["path"]))
    monkeypatch.setattr(owner, "strict_native_environment", lambda: None)
    for key, value in owner.CPU_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("MUJOCO_GL", "forbidden")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "forbidden")

    def root(*, check=False, owner=False):
        return tmp_path / ("direction" + ("_check" if check else "") + ("_owner" if owner else ""))

    monkeypatch.setattr(owner, "case_root", root)
    log = write_json(tmp_path / "qa/pytest.log", {})
    qa_ref = write_json(tmp_path / "qa/result.json", dict(log_sha256=log["sha256"]))
    source = write_json(tmp_path / "source.json", dict(frozen=True))
    monkeypatch.setattr(owner, "verify_native_qa", lambda ref: owner.read(ref))
    monkeypatch.setattr(owner, "load_prerequisites", lambda: ({}, {}, {}, {}, {source["path"]: source}))
    if stage == "check":
        producer = receipt("diagnose")
        producer["source_refs_unchanged"] = [source]
        producer_ref = write_json(root() / "verified.json", producer)
        registration = dict(
            experiment=owner.EXPERIMENT,
            stage="diagnose",
            plan_ref=plan,
            native_component_qa_ref={} if defect == "different_qa" else qa_ref,
            source_sha256={source["path"]: source["sha256"]},
        )
        predecessor = owned_metadata(owner, root(owner=True), "diagnose", producer_ref, registration, defect=defect)
        write_json(root(owner=True) / "result.json", predecessor)
        if defect == "changed_source":
            Path(source["path"]).write_text("source changed after its closure")
    calls = []

    def child(command, environment, capture, *, timeout):
        calls.append(dict(command=command, environment=environment, timeout=timeout))
        if defect == "child_failure":
            raise ChildProcessError("synthetic child exit1; no retry")
        result_ref = write_json(
            root(check=stage == "check") / "verified.json", receipt(stage, 2 if defect == "partial" else 3)
        )
        lifecycle = owned_metadata(owner, capture, stage, result_ref, {})["owned_command"]
        if defect == "source_changed_during_child":
            Path(source["path"]).write_text("changed during owned work")
        return lifecycle

    monkeypatch.setattr(owner, "owned_command", child)
    return SimpleNamespace(run=lambda dry=False: owner.main(stage, qa_ref, dry), calls=calls, root=root)


@pytest.mark.parametrize("stage,timeout", [("diagnose", 600), ("check", 1800)])
def test_owner_fixed_cpu_timeout_and_complete_count_with_real_hash_lifecycle(
    modules, monkeypatch, tmp_path, stage, timeout
):
    run = owner_run(modules, monkeypatch, tmp_path, stage)
    run.run()
    assert len(run.calls) == 1 and run.calls[0]["timeout"] == timeout
    assert run.calls[0]["command"] == modules.owner.entry_command(stage)
    env = run.calls[0]["environment"]
    assert all(env[key] == value for key, value in modules.owner.CPU_ENVIRONMENT.items())
    assert "MUJOCO_GL" not in env and "PYOPENGL_PLATFORM" not in env
    owner_ref = modules.owner.reference(run.root(check=stage == "check", owner=True) / "result.json")
    result, report = modules.owner.closed_owner(owner_ref, {})
    assert result["complete"] is report["complete"] is True
    registration = modules.owner.read(result["registration_ref"])
    assert registration["automatic_retry"] is False
    assert registration["response_budget"] == registration["planned_request_count"] == 80
    assert registration["unique_command_budget"] == 79 and registration["mandatory_anchor_repeats"] == 1
    assert registration["iteration_budget"] == 0


@pytest.mark.parametrize("defect", ["different_qa", "no_wait", "not_reaped", "signal", "changed_source"])
def test_unclosed_or_changed_producer_cannot_launch_checker(modules, monkeypatch, tmp_path, defect):
    run = owner_run(modules, monkeypatch, tmp_path, "check", defect=defect)
    with pytest.raises(ValueError):
        run.run()
    assert run.calls == [] and not run.root(check=True, owner=True).exists()


@pytest.mark.parametrize("defect", ["partial", "child_failure", "source_changed_during_child"])
def test_failure_keeps_failed_owner_without_retry_or_partial_replay(modules, monkeypatch, tmp_path, defect):
    run = owner_run(modules, monkeypatch, tmp_path, "check", defect=defect)
    with pytest.raises((ValueError, ChildProcessError)):
        run.run()
    assert len(run.calls) == 1
    result = json.loads((run.root(check=True, owner=True) / "result.json").read_text())
    assert result["complete"] is result["passed"] is False and result["error"]


@pytest.mark.parametrize("stage", ["diagnose", "check"])
def test_owner_dry_only_validates_metadata_and_never_launches(modules, monkeypatch, tmp_path, stage):
    run = owner_run(modules, monkeypatch, tmp_path, stage)
    run.run(dry=True)
    assert run.calls == [] and not run.root(check=stage == "check", owner=True).exists()


def test_unmet_historical_prerequisite_blocks_before_launch(modules, monkeypatch, tmp_path):
    run = owner_run(modules, monkeypatch, tmp_path, "diagnose")

    def incomplete():
        raise ValueError("complete original SQP replay is required")

    monkeypatch.setattr(modules.owner, "load_prerequisites", incomplete)
    with pytest.raises(ValueError, match="complete original SQP replay"):
        run.run()
    assert run.calls == []


@pytest.mark.parametrize("stage", ["diagnose", "check"])
def test_existing_output_cannot_be_reused(modules, monkeypatch, tmp_path, stage):
    run = owner_run(modules, monkeypatch, tmp_path, stage)
    run.root(check=stage == "check").mkdir()
    with pytest.raises(ValueError, match="no overwrite or retry"):
        run.run()
    assert run.calls == []


def test_native_qa_binds_all62_sources_and20_modules(modules, monkeypatch, tmp_path):
    qa = native_qa(modules, monkeypatch, tmp_path)
    old_owner = importlib.import_module("execute_tracking_v5_accepted_point_sqp")
    monkeypatch.setattr(old_owner, "REPO", tmp_path)
    monkeypatch.setattr(old_owner, "NATIVE_PYTHON", tmp_path / "native_python")
    target = tmp_path / "qa/result.json"
    assert len(modules.owner.QA_SOURCES) == 62 and len(modules.owner.QA_TESTS) == 20
    assert modules.owner.verify_native_qa(write_json(target, qa)) == qa
    for source in modules.evidence.SOURCES:
        changed = copy.deepcopy(qa)
        del changed["source_sha256"][source]
        with pytest.raises(ValueError):
            modules.owner.verify_native_qa(write_json(target, changed))
    for test in modules.evidence.SOURCES[-3:]:
        changed = copy.deepcopy(qa)
        changed["command"].remove(test)
        with pytest.raises(ValueError):
            modules.owner.verify_native_qa(write_json(target, changed))
    ref = write_json(target, qa)
    (tmp_path / modules.evidence.SOURCES[0]).write_text("changed after synthetic QA")
    with pytest.raises(ValueError):
        modules.owner.verify_native_qa(ref)


@pytest.mark.parametrize(
    "stage,field,value",
    [
        ("diagnose", "prediction_calls", 4),
        ("diagnose", "unique_command_count", 3),
        ("diagnose", "total_iterations", 1),
        ("diagnose", "actual_consumed_reference_count", 1),
        ("check", "verified_responses", 2),
        ("check", "repeated_responses_checked", 0),
        ("check", "mj_forward_queries", 625),
        ("check", "boundary_classification_disagreements", [{}]),
    ],
)
def test_owner_rejects_unbilled_or_partial_or_nonzero_activity_receipt(modules, stage, field, value):
    saved = receipt(stage)
    saved[field] = value
    with pytest.raises(ValueError):
        modules.owner.validate_receipt(saved, stage)


def test_owner_rejects_signed_zero_change_in_mandatory_repeat(modules):
    saved = receipt("diagnose")
    saved["answer"]["evaluated_candidates"][-1]["commands"][0][0] = -0.0
    with pytest.raises(ValueError, match="exact bytes"):
        modules.owner.validate_receipt(saved, "diagnose")


def test_producer_dry_uses_only_prerequisite_metadata_and_never_constructs_dynamics(modules, monkeypatch, tmp_path):
    producer = modules.producer
    source, anchor, _ = source_fixture()
    control = dict(input_min=[-1.0] * 6, input_max=[1.0] * 6, output_min=[-0.05] * 6, output_max=[0.05] * 6)
    control_ref = write_json(tmp_path / "control.json", dict(contract=control))
    refs = {control_ref["path"]: control_ref}
    monkeypatch.setattr(producer, "strict_native_environment", lambda: None)
    for key, value in producer.CPU_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(producer, "case_root", lambda: tmp_path / "not-created")
    monkeypatch.setattr(
        producer, "load_prerequisites", lambda: (source, dict(control_ref=control_ref), dict(answer={}), {}, refs)
    )
    monkeypatch.setattr(producer, "saved_stencil", lambda answer: (anchor, directions(), None, None))
    monkeypatch.setattr(producer, "model_registration", lambda reg, bound: (reg, {}))

    def forbidden(*args, **kwargs):
        raise AssertionError("dry must not construct dynamics or evaluate a response")

    monkeypatch.setattr(producer, "MujocoRobotDynamics", forbidden)
    monkeypatch.setattr(producer, "run_probe", forbidden)
    producer.main(dry=True)
    assert not (tmp_path / "not-created").exists()


def saved_h12_answer():
    """Real core records with a small analytic fake rollout, never a simulator."""
    core = importlib.import_module("s2a.directional_derivative_probe")
    source, anchor, interface = source_fixture()
    fake = complete_synthetic_rollout(source, [])
    dynamics = CountingDynamics()

    def predict(commands):
        return fake(
            source["current_q"].copy(),
            source["current_velocity"].copy(),
            commands.copy(),
            interface,
            source["initial_joint"].copy(),
            source["kp"].copy(),
            source["kd"].copy(),
            dynamics,
            source["torque_limits"][:, 0],
            source["torque_limits"][:, 1],
            latched_rotation_goal=source["latched_rotation_goal"].copy(),
        )

    parallel = directions()
    parallel[:] = parallel[0]
    answer = core.probe_directional_derivatives(
        predict,
        source["previous_command"][:6],
        anchor,
        parallel,
        interface,
        source["joint_limits"][:, 0],
        source["joint_limits"][:, 1],
        source["desired_tcp"],
        source["site_from_tcp"],
        past_inner_error_rad=source["past_inner_error_rad"],
        terminal_normals=source["terminal_normals"],
        step_sizes=(1e-4,),
    )
    # Match persisted input shapes and scalars, including empty audit-index lists.
    return json.loads(json.dumps(core._canonical(answer)))


@pytest.mark.parametrize("defect", [None, "repeat_rollout", "safety_vector", "strict_label", "one_query_missing"])
def test_checker_full_replays_every_call_before_ledger_even_when_commands_repeat(modules, monkeypatch, defect):
    checker = importlib.import_module("check_tracking_v5_directional_derivative")
    frozen = importlib.import_module("check_tracking_v5_prefix_trust_control")
    answer = saved_h12_answer()
    reference_entries = copy.deepcopy(answer["evaluated_candidates"])
    assert len(reference_entries) == 4 and answer["unique_command_count"] == 3
    by_command = {frozen.command_key(entry["commands"], 12): entry for entry in reference_entries}
    result = dict(
        response_evaluations=0,
        verified_responses=0,
        numerical_inner_steps=0,
        query_count=1,
        mj_forward_queries=1,
        boundary_classification_disagreements=[],
    )
    calls, audits, ledger_calls, active = [], [], [], {}

    def reconstruct(commands):
        key = frozen.command_key(commands, 12)
        calls.append(key)
        active["entry"] = by_command[key]
        result["numerical_inner_steps"] += 300
        result["mj_forward_queries"] += 312
        if defect == "one_query_missing" and len(calls) == 4:
            result["mj_forward_queries"] -= 1
        return copy.deepcopy(active["entry"]["rollout"]), []

    def classify(commands, rollout):
        expected = active["entry"]
        q = frozen.normalized_errors(expected["tcp_errors"], 12)
        return dict(
            tcp_errors=expected["tcp_errors"],
            normalized_squared_errors=q,
            safety_slack=expected["safety_slack"],
            safety_min=float(np.min(expected["safety_slack"])),
            joint_feasible=expected["audit"]["feasible"],
            prefix_length=frozen.continuous_prefix(q),
            feasible=bool(expected["audit"]["feasible"] and np.all(q <= 1.0)),
            maximum_normalized_squared_error=float(np.max(q)),
            maximum_pose_cap_violation=float(max(0.0, np.max(np.sqrt(q)) - 1)),
            pose_loss=float(np.sum(q)),
        )

    def audit(saved, metrics):
        frozen.compare_contract(active["entry"]["audit"], saved, name="complete_synthetic_audit")
        audits.append(True)

    def ledger(saved):
        assert saved is answer and result["verified_responses"] == len(reference_entries)
        assert len(audits) == len(reference_entries)
        ledger_calls.append(True)
        return dict(passed=True)

    monkeypatch.setattr(checker, "audit_directional_derivative_ledger", ledger)
    if defect == "repeat_rollout":
        answer["evaluated_candidates"][-1]["rollout"]["pose"][0][0][3] += 1e-3
    elif defect == "safety_vector":
        answer["evaluated_candidates"][-1]["safety_slack"][147] += 1e-3
        assert min(answer["evaluated_candidates"][-1]["safety_slack"]) == min(reference_entries[-1]["safety_slack"])
    elif defect == "strict_label":
        answer["evaluated_candidates"][-1]["prefix_length"] = 0

    def run():
        checker.verify_response_trace(
            answer,
            reconstruct,
            frozen.compare_contract,
            classify,
            audit,
            result,
            compare=frozen.numerical_compare,
        )

    if defect is not None:
        with pytest.raises(ValueError):
            run()
        assert ledger_calls == []
        assert len(calls) == 4  # A duplicate key still reaches the independent replay.
    else:
        run()
        assert len(calls) == result["verified_responses"] == 4
        assert calls[0] == calls[-1] and len(set(calls)) == 3
        assert result["query_count"] == 1249 and result["numerical_inner_steps"] == 1200
        assert result["repeated_responses_checked"] == 1 and ledger_calls == [True]


@pytest.mark.parametrize("tamper", [False, True])
def test_historical_anchor_audit_comparison_survives_json_and_rejects_hidden_row_change(modules, tamper):
    checker = importlib.import_module("check_tracking_v5_directional_derivative")
    core = importlib.import_module("s2a.directional_derivative_probe")
    answer = saved_h12_answer()
    new = answer["evaluated_candidates"][0]
    projection = {
        key: copy.deepcopy(new[key])
        for key in ("commands", "tcp_errors", "safety_slack", "prefix_length", "feasible", "audit")
    }
    projection.update(safety_min=float(np.min(new["safety_slack"])), joint_feasible=new["audit"]["feasible"])
    old = copy.deepcopy(projection)
    compact = {key: value for key, value in old.items() if key != "audit"}
    sqp = dict(
        evaluated_candidates=[None] * 1819 + [compact], accepted_anchors=[dict(response_index=1819, audit=old["audit"])]
    )
    saved = dict(
        scope="Complete archived trace and accepted-anchor audit versus new response0; no old full rollout",
        historical_response_index=1819,
        new_response_index=0,
        comparison=core._repeat_comparison(old, projection, 0),
    )
    sqp, answer, saved = json.loads(json.dumps([sqp, answer, saved]))
    if tamper:
        sqp["accepted_anchors"][0]["audit"]["groups"]["inner_lower"]["slack"][147][3] += 1e-3
        with pytest.raises(ValueError):
            checker.verify_historical_anchor(sqp, answer, saved)
    else:
        result = checker.verify_historical_anchor(sqp, answer, saved)
        assert result["passed"] is result["exact"] is True
        assert "audit" in result["compared_fields"] and result["max_abs_difference"] == 0.0
