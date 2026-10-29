"""Synthetic fixed-call ownership, reset, full replay and provenance contracts."""

import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import test_directional_derivative_execution as previous
from test_accepted_point_sqp_execution import native_qa
from test_prefix_callback_backtracking_execution import write_json
from test_prefix_trust_control_execution import CountingDynamics, captured_rollout, source_fixture


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return SimpleNamespace(
        owner=importlib.import_module("execute_tracking_v5_terminal_secant_probe"),
        evidence=importlib.import_module("s2a.terminal_secant_evidence"),
        producer=importlib.import_module("probe_tracking_v5_terminal_secant_probe"),
        checker=importlib.import_module("check_tracking_v5_terminal_secant_probe"),
    )


def billed_receipt():
    return dict(
        response_evaluations=17,
        unique_command_count=16,
        numerical_inner_steps=5100,
        query_count=5305,
        total_iterations=0,
        control_qualified=False,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        actual_consumed_reference_count=0,
    )


def test_fixed17_calls_include_one_billed_repeat_and_one_current_query(modules):
    modules.owner.validate_call_counts(billed_receipt())
    partial = billed_receipt()
    partial.update(response_evaluations=16, unique_command_count=15, numerical_inner_steps=4800, query_count=4993)
    with pytest.raises(ValueError):
        modules.owner.validate_call_counts(partial)


@pytest.mark.parametrize(
    "key,value",
    [
        ("unique_command_count", 17),
        ("numerical_inner_steps", 4800),
        ("query_count", 5304),
        ("total_iterations", 1),
        ("physics_steps", 1),
        ("world_model_calls", 1),
        ("fresh_cases", 1),
        ("actual_consumed_reference_count", 1),
        ("control_qualified", True),
    ],
)
def test_confirmed_accounting_contract_rejects_free_repeat_or_extra_activity(modules, key, value):
    report = billed_receipt()
    report[key] = value
    with pytest.raises(ValueError):
        modules.owner.validate_call_counts(report)


@pytest.mark.parametrize("stage,prefix", [("diagnose", "probe"), ("check", "check")])
def test_new_entry_commands_do_not_route_to_prior_diagnostic(modules, stage, prefix):
    command = modules.owner.entry_command(stage)
    assert command == [
        str(modules.owner.NATIVE_PYTHON),
        str(modules.owner.REPO / f"scripts/{prefix}_tracking_v5_terminal_secant_probe.py"),
    ]


def test_native_qa_covers85_sources26_modules_and_preserves_prior75_digests(modules, monkeypatch, tmp_path):
    owner = modules.owner
    qa = native_qa(modules, monkeypatch, tmp_path)
    for name in (
        "execute_tracking_v5_directional_derivative_check_v3",
        "execute_tracking_v5_directional_derivative_v2",
        "execute_tracking_v5_directional_derivative",
        "execute_tracking_v5_accepted_point_sqp",
    ):
        module = importlib.import_module(name)
        monkeypatch.setattr(module, "REPO", tmp_path)
        monkeypatch.setattr(module, "NATIVE_PYTHON", tmp_path / "native_python")
    previous_owner = importlib.import_module("execute_tracking_v5_directional_derivative_check_v3")
    assert len(owner.QA_SOURCES) == 85 and len(owner.QA_TESTS) == 26
    new_sources = set(owner.QA_SOURCES) - set(owner.V3_QA_SOURCES)
    new_tests = set(owner.QA_TESTS) - set(previous_owner.QA_TESTS)
    assert len(new_sources) == 10 and len(new_tests) == 3
    prior = copy.deepcopy(qa)
    prior["source_sha256"] = {path: qa["source_sha256"][path] for path in owner.V3_QA_SOURCES}
    prior["command"] = [str(tmp_path / "native_python"), "-m", "pytest", *previous_owner.QA_TESTS]
    prior_log = write_json(tmp_path / "prior_qa/pytest.log", {})
    prior["log_sha256"] = prior_log["sha256"]
    prior_ref = write_json(tmp_path / "prior_qa/result.json", prior)
    monkeypatch.setattr(owner, "V3_QA_REF", prior_ref)
    target = tmp_path / "qa/result.json"
    assert owner.verify_native_qa(write_json(target, qa)) == qa
    assert owner.verify_prior_qa_subset(qa, {}) == prior
    for source in new_sources:
        changed = copy.deepcopy(qa)
        del changed["source_sha256"][source]
        with pytest.raises(ValueError):
            owner.verify_native_qa(write_json(target, changed))
    for test in new_tests:
        changed = copy.deepcopy(qa)
        changed["command"].remove(test)
        with pytest.raises(ValueError):
            owner.verify_native_qa(write_json(target, changed))
    changed = copy.deepcopy(qa)
    changed["source_sha256"][owner.V3_QA_SOURCES[0]] = "0" * 64
    with pytest.raises(ValueError, match="identical digest subset"):
        owner.verify_prior_qa_subset(changed, {})
    ref = write_json(target, qa)
    (tmp_path / sorted(new_sources)[0]).write_text("new source changed after QA")
    with pytest.raises(ValueError):
        owner.verify_native_qa(ref)


def terminal_case():
    """Real geometry/core inputs with an analytic non-robot terminal velocity."""
    core = importlib.import_module("s2a.terminal_secant_probe")
    source, anchor, interface = source_fixture()
    units = np.zeros((3, 12, 6))
    for index in range(3):
        units[index, 1, index] = 1.0
    geometry = core.build_terminal_secant_geometry(
        anchor, source["previous_command"][:6], interface.input_min, interface.input_max, units
    )
    historical = previous.saved_h12_answer()["evaluated_candidates"][0]
    for period in historical["rollout"]["periods"]:
        period["clipped"] = np.zeros((25, 7), dtype=bool)
    fit = core.fit_terminal_secant_row(
        anchor,
        units[0],
        np.zeros(72),
        0.0,
        [anchor + step * units[0] for step in core.FIT_STEP_SIZES],
        np.zeros(2),
    )
    return source, geometry, fit, historical, interface


def terminal_rollout(source, geometry, calls):
    original = previous.complete_synthetic_rollout(source, calls)

    def rollout(*args, **kwargs):
        command = args[2].copy()
        result = original(*args, **kwargs)
        # Deliberately unseen orthogonal terminal dependence: scientific failure
        # still requires the fixed fifteen evaluations and final billed repeat.
        result["joint_velocity"][-1, 6] = -float(np.sum((command - geometry["anchor_commands"]) * geometry["v"]))
        for period in result["periods"]:
            period["clipped"] = np.zeros((25, 7), dtype=bool)
        return result

    return rollout


def canonical(value):
    core = importlib.import_module("s2a.terminal_secant_probe")
    return json.loads(json.dumps(core._canonical(value)))


def complete_receipt(stage, answer=None):
    report = dict(
        **billed_receipt(),
        complete=True,
        passed=True,
        experiment="terminal_secant_probe",
        source_refs_unchanged=[],
        witness_indices=[],
        witness_found=False,
    )
    if stage == "check":
        report.update(
            verified_responses=17,
            mj_forward_queries=5305,
            repeated_responses_checked=1,
            geometry_check=dict(passed=True),
            fit_check=dict(passed=True),
            ledger_check=dict(passed=True),
            boundary_classification_disagreements=[],
            native_quaternion_jit_signatures=["(Array(float64, 1, 'C'),) -> array(float32, 2d, C)"],
        )
        return report
    if answer is None:
        owner = importlib.import_module("execute_tracking_v5_terminal_secant_probe")
        anchor = np.zeros((12, 6))
        commands = [anchor.copy() for _ in range(17)]
        for index in range(1, 16):
            commands[index][1, 0] = index / 10000
        ledger = []
        for index, command in enumerate(commands):
            grid = 1 <= index <= 15
            ledger.append(
                dict(
                    request_index=index,
                    response_index=index,
                    calls_before=index,
                    calls_after=index + 1,
                    charged=True,
                    role="grid" if grid else "anchor" if index == 0 else "anchor_repeat",
                    outcome="repeated" if index == 16 else "computed",
                    repeat_of=0 if index == 16 else None,
                    commands=command.tolist(),
                    command_key=command.tobytes().hex(),
                    direction_index=(index - 1) // 5 if grid else None,
                    direction_name=owner.DIRECTION_NAMES[(index - 1) // 5] if grid else None,
                    step_index=(index - 1) % 5 if grid else None,
                    step=owner.STEP_SIZES[(index - 1) % 5] if grid else None,
                )
            )
        answer = dict(
            method="fixed_terminal_secant_probe",
            horizon=12,
            max_responses=17,
            planned_request_count=17,
            response_evaluations=17,
            unique_command_count=16,
            total_iterations=0,
            control_qualified=False,
            step_sizes=list(owner.STEP_SIZES),
            direction_names=list(owner.DIRECTION_NAMES),
            anchor_commands=anchor.tolist(),
            evaluated_candidates=[dict(commands=value.tolist(), feasible=False) for value in commands],
            request_ledger=ledger,
            repeatability=dict(initial_response_index=0, repeated_response_index=16),
            predictions=[{} for _ in range(15)],
            clipping_checks=[{} for _ in range(17)],
            qualification=dict(extrapolation_passed=False, safety_accuracy_passed=False),
            extrapolation_passed=False,
            safety_accuracy_passed=False,
        )
    report.update(prediction_calls=17, answer=answer)
    report["witness_indices"] = [i for i, entry in enumerate(answer["evaluated_candidates"]) if entry["feasible"]]
    report["witness_found"] = bool(report["witness_indices"])
    return report


def test_real_core_default_wrapper_resets17_calls_and_preserves_scientific_failure(modules):
    source, geometry, fit, historical, interface = terminal_case()
    original = copy.deepcopy(source)
    geometry_before, fit_before = canonical(geometry), canonical(fit)
    dynamics, calls = CountingDynamics(), []
    result = modules.producer.run_probe(
        source, geometry, fit, historical, interface, dynamics, rollout_fn=terminal_rollout(source, geometry, calls)
    )
    assert result["prediction_calls"] == result["response_evaluations"] == len(calls) == 17
    assert result["unique_command_count"] == 16
    assert result["numerical_inner_steps"] == 5100 and result["query_count"] == dynamics.queries == 5305
    assert result["answer"]["extrapolation_passed"] is False
    assert result["answer"]["safety_accuracy_passed"] is False
    expected = [
        geometry["anchor_commands"],
        *[r["commands"] for r in geometry["grid_commands"]],
        geometry["anchor_commands"],
    ]
    assert [row["commands"].tobytes() for row in calls] == [value.tobytes() for value in expected]
    assert calls[0]["commands"].tobytes() == calls[-1]["commands"].tobytes()
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
    assert canonical(geometry) == geometry_before and canonical(fit) == fit_before
    modules.owner.validate_receipt(complete_receipt("diagnose", canonical(result["answer"])), "diagnose")


@pytest.mark.parametrize(
    "defect,completed", [("wrong_first", 0), ("reordered_grid", 1), ("early_repeat", 1), ("eighteenth", 17)]
)
def test_predictor_rejects_wrong_order_signed_zero_early_repeat_and_extra_call(modules, defect, completed):
    source, geometry, fit, historical, interface = terminal_case()
    dynamics, calls, progress = CountingDynamics(), [], {}

    def invalid(predict, previous_command, anchor, *args, **kwargs):
        expected = [anchor, *[r["commands"] for r in geometry["grid_commands"]], anchor]
        if defect == "wrong_first":
            command = anchor.copy()
            command[0, 3] = 0.0
            predict(command)
        elif defect == "reordered_grid":
            predict(anchor)
            predict(expected[2])
        elif defect == "early_repeat":
            predict(anchor)
            predict(anchor.copy())
        else:
            for command in expected:
                predict(command)
            predict(anchor.copy())

    with pytest.raises(ValueError, match="call order|after the final anchor"):
        modules.producer.run_probe(
            source,
            geometry,
            fit,
            historical,
            interface,
            dynamics,
            probe_fn=invalid,
            rollout_fn=captured_rollout(source, calls),
            progress=progress,
        )
    assert len(calls) == progress["response_evaluations"] == completed
    assert progress["prediction_calls"] == completed + 1
    assert progress["numerical_inner_steps"] == 300 * completed
    assert progress["query_count"] == 1 + 312 * completed


def test_all_grid_admission_precedes_even_the_current_query(modules):
    source, geometry, fit, historical, interface = terminal_case()
    geometry["grid_commands"][-1]["commands"][1, 0] = 0.6
    dynamics = CountingDynamics()
    with pytest.raises(ValueError, match="command box and rate"):
        modules.producer.run_probe(source, geometry, fit, historical, interface, dynamics)
    assert dynamics.queries == 0


def test_wrapper_query_bill_is_not_inferred_from_successful_return(modules):
    source, geometry, fit, historical, interface = terminal_case()
    dynamics, calls, progress = CountingDynamics(), [], {}

    def one(predict, previous_command, anchor, *args, **kwargs):
        predict(anchor)

    with pytest.raises(ValueError, match="bills300 inner"):
        modules.producer.run_probe(
            source,
            geometry,
            fit,
            historical,
            interface,
            dynamics,
            probe_fn=one,
            rollout_fn=captured_rollout(source, calls, query_error=-1),
            progress=progress,
        )
    assert progress["response_evaluations"] == 1 and progress["query_count"] == 312


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
        return tmp_path / ("secant" + ("_check" if check else "") + ("_owner" if owner else ""))

    monkeypatch.setattr(owner, "case_root", root)
    log = write_json(tmp_path / "qa/pytest.log", {})
    qa_ref = write_json(tmp_path / "qa/result.json", dict(log_sha256=log["sha256"]))
    source = write_json(tmp_path / "source.json", dict(frozen=True))
    monkeypatch.setattr(owner, "verify_native_qa", lambda ref: owner.read(ref))
    monkeypatch.setattr(owner, "verify_prior_qa_subset", lambda qa, refs: {})
    monkeypatch.setattr(owner, "load_prerequisites", lambda: ({}, {}, {}, {}, {source["path"]: source}))
    if stage == "check":
        producer = complete_receipt("diagnose")
        producer["source_refs_unchanged"] = [source]
        producer_ref = write_json(root() / "verified.json", producer)
        registration = dict(
            experiment=owner.EXPERIMENT,
            stage="diagnose",
            plan_ref=plan,
            native_component_qa_ref={} if defect == "different_qa" else qa_ref,
            source_sha256={source["path"]: source["sha256"]},
        )
        predecessor = previous.owned_metadata(
            owner, root(owner=True), "diagnose", producer_ref, registration, defect=defect
        )
        write_json(root(owner=True) / "result.json", predecessor)
        if defect == "changed_source":
            Path(source["path"]).write_text("source changed after closure")
    calls = []

    def child(command, environment, capture, *, timeout):
        calls.append(dict(command=command, environment=environment, timeout=timeout))
        if defect == "child_failure":
            raise ChildProcessError("synthetic child exit1; no retry")
        report = complete_receipt(stage)
        if defect == "partial":
            report.update(response_evaluations=16, verified_responses=16)
        if defect == "witness_mismatch":
            report.update(witness_indices=[0], witness_found=True)
        result_ref = write_json(root(check=stage == "check") / "verified.json", report)
        lifecycle = previous.owned_metadata(owner, capture, stage, result_ref, {})["owned_command"]
        if defect == "source_changed_during_child":
            Path(source["path"]).write_text("changed during owned work")
        return lifecycle

    monkeypatch.setattr(owner, "owned_command", child)
    return SimpleNamespace(run=lambda dry=False: owner.main(stage, qa_ref, dry), calls=calls, root=root)


@pytest.mark.parametrize("stage,timeout", [("diagnose", 600), ("check", 1800)])
def test_owner_fixed_cpu_timeout_complete_counts_and_real_hash_lifecycle(
    modules, monkeypatch, tmp_path, stage, timeout
):
    run = owner_run(modules, monkeypatch, tmp_path, stage)
    run.run()
    assert len(run.calls) == 1 and run.calls[0]["timeout"] == timeout
    assert run.calls[0]["command"] == modules.owner.entry_command(stage)
    environment = run.calls[0]["environment"]
    assert all(environment[key] == value for key, value in modules.owner.CPU_ENVIRONMENT.items())
    assert "MUJOCO_GL" not in environment and "PYOPENGL_PLATFORM" not in environment
    result, receipt = modules.owner.closed_owner(
        modules.owner.reference(run.root(check=stage == "check", owner=True) / "result.json"), {}
    )
    assert result["complete"] is receipt["complete"] is True
    registration = modules.owner.read(result["registration_ref"])
    assert registration["response_budget"] == registration["planned_request_count"] == 17
    assert registration["unique_command_budget"] == 16 and registration["mandatory_anchor_repeats"] == 1
    assert registration["expected_numerical_inner_steps"] == 5100 and registration["expected_queries"] == 5305
    assert registration["iteration_budget"] == 0 and registration["automatic_retry"] is False
    assert registration["prior_qa_subset_count"] == 75


@pytest.mark.parametrize("defect", ["different_qa", "no_wait", "not_reaped", "signal", "changed_source"])
def test_unclosed_or_changed_prior_producer_cannot_launch_checker(modules, monkeypatch, tmp_path, defect):
    run = owner_run(modules, monkeypatch, tmp_path, "check", defect=defect)
    with pytest.raises(ValueError):
        run.run()
    assert not run.calls and not run.root(check=True, owner=True).exists()


@pytest.mark.parametrize("defect", ["partial", "child_failure", "source_changed_during_child", "witness_mismatch"])
def test_child_failure_is_preserved_without_retry_or_partial_acceptance(modules, monkeypatch, tmp_path, defect):
    run = owner_run(modules, monkeypatch, tmp_path, "check", defect=defect)
    with pytest.raises((ValueError, ChildProcessError)):
        run.run()
    assert len(run.calls) == 1
    failed = json.loads((run.root(check=True, owner=True) / "result.json").read_text())
    assert failed["complete"] is failed["passed"] is False and failed["error"]


@pytest.mark.parametrize("stage", ["diagnose", "check"])
def test_owner_dry_validates_without_model_or_child(modules, monkeypatch, tmp_path, stage):
    run = owner_run(modules, monkeypatch, tmp_path, stage)
    run.run(dry=True)
    assert not run.calls and not run.root(check=stage == "check", owner=True).exists()


def test_unclosed_prior_diagnostic_blocks_before_launch(modules, monkeypatch, tmp_path):
    run = owner_run(modules, monkeypatch, tmp_path, "diagnose")

    def incomplete():
        raise ValueError("complete original diagnostic replay required")

    monkeypatch.setattr(modules.owner, "load_prerequisites", incomplete)
    with pytest.raises(ValueError, match="original diagnostic replay"):
        run.run()
    assert not run.calls


@pytest.mark.parametrize("stage", ["diagnose", "check"])
def test_existing_output_cannot_be_overwritten(modules, monkeypatch, tmp_path, stage):
    run = owner_run(modules, monkeypatch, tmp_path, stage)
    run.root(check=stage == "check").mkdir()
    with pytest.raises(ValueError, match="no overwrite or retry"):
        run.run()
    assert not run.calls


@pytest.mark.parametrize("defect", ["repeat_free", "wrong_order", "signature", "geometry", "fit", "ledger", "strict"])
def test_count_only_receipts_cannot_qualify(modules, defect):
    stage = "diagnose" if defect in ("repeat_free", "wrong_order") else "check"
    report = complete_receipt(stage)
    if defect == "repeat_free":
        report["answer"]["request_ledger"][-1]["charged"] = False
    elif defect == "wrong_order":
        report["answer"]["request_ledger"][1]["step"] = 5e-4
    elif defect == "signature":
        report["native_quaternion_jit_signatures"] = []
    elif defect == "strict":
        report["boundary_classification_disagreements"] = [{}]
    else:
        report[defect + "_check"]["passed"] = False
    with pytest.raises(ValueError):
        modules.owner.validate_receipt(report, stage)


@pytest.mark.parametrize("defect", [None, "repeat_rollout", "safety_vector", "strict_label", "one_query_missing"])
def test_checker_replays_all17_including_repeat_and_rejects_full_record_tampering(modules, defect):
    source, geometry, fit, historical, interface = terminal_case()
    result = modules.producer.run_probe(
        source,
        geometry,
        fit,
        historical,
        interface,
        CountingDynamics(),
        rollout_fn=terminal_rollout(source, geometry, []),
    )
    answer = canonical(result["answer"])
    reference_entries = copy.deepcopy(answer["evaluated_candidates"])
    frozen = importlib.import_module("check_tracking_v5_prefix_trust_control")
    by_command = {frozen.command_key(row["commands"], 12): row for row in reference_entries}
    replay = dict(
        response_evaluations=0,
        verified_responses=0,
        numerical_inner_steps=0,
        query_count=1,
        mj_forward_queries=1,
        boundary_classification_disagreements=[],
    )
    calls, audits, active = [], [], {}

    def reconstruct(commands):
        key = frozen.command_key(commands, 12)
        calls.append(key)
        active["entry"] = by_command[key]
        replay["numerical_inner_steps"] += 300
        replay["mj_forward_queries"] += 312
        if defect == "one_query_missing" and len(calls) == 17:
            replay["mj_forward_queries"] -= 1
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

    def check_audit(saved, metrics):
        frozen.compare_contract(active["entry"]["audit"], saved, name="complete_synthetic_audit")
        audits.append(True)

    if defect == "repeat_rollout":
        answer["evaluated_candidates"][-1]["rollout"]["pose"][0][0][3] += 1e-3
    elif defect == "safety_vector":
        answer["evaluated_candidates"][-1]["safety_slack"][147] += 1e-3
        assert min(answer["evaluated_candidates"][-1]["safety_slack"]) == min(reference_entries[-1]["safety_slack"])
    elif defect == "strict_label":
        answer["evaluated_candidates"][-1]["prefix_length"] = 0

    def run():
        modules.checker.verify_response_trace(
            answer,
            reconstruct,
            frozen.compare_contract,
            classify,
            check_audit,
            replay,
            compare=frozen.numerical_compare,
        )

    if defect is not None:
        with pytest.raises(ValueError):
            run()
        assert replay["verified_responses"] <= 17
    else:
        run()
        assert replay["verified_responses"] == len(audits) == 17
        assert replay["numerical_inner_steps"] == 5100 and replay["query_count"] == 5305
        assert replay["repeated_responses_checked"] == 1 and replay["unique_command_count"] == 16
        assert replay["boundary_classification_disagreements"] == []
    assert len(calls) == 17 and calls[0] == calls[-1] and len(set(calls)) == 16


def test_producer_dry_does_geometry_and_fit_before_model_construction(modules, monkeypatch, tmp_path):
    producer = modules.producer
    source, geometry, fit, historical, interface = terminal_case()
    control_ref = write_json(
        tmp_path / "control.json",
        dict(
            contract=dict(
                input_min=interface.input_min.tolist(),
                input_max=interface.input_max.tolist(),
                output_min=interface.input_min.tolist(),
                output_max=interface.input_max.tolist(),
            )
        ),
    )
    monkeypatch.setattr(producer, "strict_native_environment", lambda: None)
    for key, value in producer.CPU_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(producer, "case_root", lambda: tmp_path / "not-created")
    monkeypatch.setattr(producer, "load_prerequisites", lambda: (source, dict(control_ref=control_ref), {}, {}, {}))
    prepared = []

    def prepare(original, diagnostic, original_interface):
        assert original is source
        np.testing.assert_array_equal(original_interface.input_min, interface.input_min)
        prepared.append(True)
        return geometry, fit, {}

    def forbidden(*args, **kwargs):
        raise AssertionError("dry must not construct a native model or request a response")

    monkeypatch.setattr(producer, "prepare_case", prepare)
    monkeypatch.setattr(producer, "model_registration", lambda reg, refs: ({}, {}))
    monkeypatch.setattr(producer, "MujocoRobotDynamics", forbidden)
    monkeypatch.setattr(producer, "run_probe", forbidden)
    producer.main(dry=True)
    assert prepared == [True] and not (tmp_path / "not-created").exists()
