"""Synthetic ownership and complete-call wiring for fixed tail translations."""

import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import test_terminal_secant_execution as prior
from test_accepted_point_sqp_execution import native_qa
from test_prefix_callback_backtracking_execution import write_json
from test_prefix_trust_control_execution import CountingDynamics, captured_rollout, source_fixture


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return SimpleNamespace(
        owner=importlib.import_module("execute_tracking_v5_tail_translation_reserve"),
        evidence=importlib.import_module("s2a.tail_translation_reserve_evidence"),
        producer=importlib.import_module("probe_tracking_v5_tail_translation_reserve"),
        checker=importlib.import_module("check_tracking_v5_tail_translation_reserve"),
    )


def billed_receipt():
    return dict(
        response_evaluations=14,
        unique_command_count=13,
        numerical_inner_steps=4200,
        query_count=4369,
        total_iterations=0,
        control_qualified=False,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        actual_consumed_reference_count=0,
    )


def complete_receipt(stage, answer=None):
    report = dict(
        **billed_receipt(),
        complete=True,
        passed=True,
        experiment="tail_translation_reserve",
        source_refs_unchanged=[],
        witness_indices=[],
        witness_found=False,
    )
    if stage == "check":
        report.update(
            verified_responses=14,
            mj_forward_queries=4369,
            repeated_responses_checked=1,
            grid_check=dict(passed=True),
            ledger_check=dict(passed=True, improvement_found=False, sufficient_reserve_restored=False),
            independent_restoration_check=dict(
                passed=True,
                response_evaluations=14,
                improvement_found=False,
                sufficient_reserve_restored=False,
                selected_response_index=None,
                control_qualified=False,
                strict_disagreements=[],
            ),
            boundary_classification_disagreements=[],
            native_quaternion_jit_signatures=["synthetic compiled signature"],
        )
        return report
    if answer is None:
        owner = importlib.import_module("execute_tracking_v5_tail_translation_reserve")
        anchor = np.zeros((12, 6))
        anchor[0, 3] = -0.0
        entries, ledger = [], []
        for index in range(14):
            grid = 1 <= index <= 12
            axis, slot = ((index - 1) // 4, (index - 1) % 4) if grid else (None, None)
            command = anchor.copy()
            if grid:
                command[11, axis] += owner.SIGNED_STEPS[slot]
            entries.append(dict(commands=command.tolist(), feasible=False))
            ledger.append(
                dict(
                    request_index=index,
                    response_index=index,
                    calls_before=index,
                    calls_after=index + 1,
                    charged=True,
                    role="grid" if grid else "anchor" if index == 0 else "anchor_repeat",
                    outcome="repeated" if index == 13 else "computed",
                    repeat_of=0 if index == 13 else None,
                    commands=command.tolist(),
                    command_key=command.tobytes().hex(),
                    axis_index=axis,
                    axis_name=owner.AXIS_NAMES[axis] if grid else None,
                    step_index=slot,
                    signed_step=owner.SIGNED_STEPS[slot] if grid else None,
                )
            )
        answer = dict(
            method="fixed_tail_translation_reserve",
            horizon=12,
            max_responses=14,
            planned_request_count=14,
            response_evaluations=14,
            unique_command_count=13,
            total_iterations=0,
            control_qualified=False,
            anchor_commands=anchor.tolist(),
            request_ledger=ledger,
            evaluated_candidates=entries,
            repeatability=dict(initial_response_index=0, repeated_response_index=13),
            candidate_checks=[{} for _ in range(12)],
            causal_checks=[{} for _ in range(14)],
            clipping_checks=[{} for _ in range(14)],
            selected=None,
            selected_response_index=None,
            improvement_found=False,
            sufficient_reserve_restored=False,
        )
    report.update(prediction_calls=14, answer=answer)
    report["witness_indices"] = [
        index for index, entry in enumerate(answer["evaluated_candidates"]) if entry["feasible"]
    ]
    report["witness_found"] = bool(report["witness_indices"])
    return report


def owner_run(modules, monkeypatch, tmp_path, stage, **kwargs):
    # Reuse hash-bound lifecycle fixtures, never alter frozen source bytes.
    monkeypatch.setattr(prior, "complete_receipt", complete_receipt)
    return prior.owner_run(modules, monkeypatch, tmp_path, stage, **kwargs)


@pytest.mark.parametrize("stage,timeout", [("diagnose", 600), ("check", 1800)])
def test_owner_preserves_complete_scientific_failure_with_fixed_cpu_and_lifecycle(
    modules, monkeypatch, tmp_path, stage, timeout
):
    run = owner_run(modules, monkeypatch, tmp_path, stage)
    run.run()
    assert len(run.calls) == 1 and run.calls[0]["timeout"] == timeout
    assert run.calls[0]["command"] == modules.owner.entry_command(stage)
    env = run.calls[0]["environment"]
    assert all(env[key] == value for key, value in modules.owner.CPU_ENVIRONMENT.items())
    assert "MUJOCO_GL" not in env and "PYOPENGL_PLATFORM" not in env
    result, receipt = modules.owner.closed_owner(
        modules.owner.reference(run.root(check=stage == "check", owner=True) / "result.json"), {}
    )
    assert result["complete"] is receipt["passed"] is True
    registration = modules.owner.read(result["registration_ref"])
    assert registration["response_budget"] == registration["planned_request_count"] == 14
    assert registration["unique_command_budget"] == 13 and registration["mandatory_anchor_repeats"] == 1
    assert registration["expected_numerical_inner_steps"] == 4200 and registration["expected_queries"] == 4369
    assert registration["prior_qa_subset_count"] == 85 and registration["iteration_budget"] == 0
    assert registration["automatic_retry"] is False
    scientific = receipt["answer"] if stage == "diagnose" else receipt["ledger_check"]
    assert scientific["improvement_found"] is scientific["sufficient_reserve_restored"] is False


@pytest.mark.parametrize("defect", ["different_qa", "no_wait", "not_reaped", "signal", "changed_source"])
def test_missing_prior_closure_or_changed_sources_block_before_launch(modules, monkeypatch, tmp_path, defect):
    run = owner_run(modules, monkeypatch, tmp_path, "check", defect=defect)
    with pytest.raises(ValueError):
        run.run()
    assert not run.calls and not run.root(check=True, owner=True).exists()


@pytest.mark.parametrize("defect", ["partial", "child_failure", "source_changed_during_child", "witness_mismatch"])
def test_failed_child_is_retained_without_retry_or_partial_acceptance(modules, monkeypatch, tmp_path, defect):
    run = owner_run(modules, monkeypatch, tmp_path, "check", defect=defect)
    with pytest.raises((ValueError, ChildProcessError)):
        run.run()
    assert len(run.calls) == 1
    result = json.loads((run.root(check=True, owner=True) / "result.json").read_text())
    assert result["complete"] is result["passed"] is False and result["error"]


@pytest.mark.parametrize("stage", ["diagnose", "check"])
def test_dry_only_checks_metadata_and_never_launches(modules, monkeypatch, tmp_path, stage):
    run = owner_run(modules, monkeypatch, tmp_path, stage)
    run.run(dry=True)
    assert not run.calls and not run.root(check=stage == "check", owner=True).exists()


def test_unmet_complete_prior_secant_blocks_before_launch(modules, monkeypatch, tmp_path):
    run = owner_run(modules, monkeypatch, tmp_path, "diagnose")

    def fail():
        raise ValueError("prior full secant closure missing")

    monkeypatch.setattr(modules.owner, "load_prerequisites", fail)
    with pytest.raises(ValueError, match="prior full secant closure"):
        run.run()
    assert not run.calls


@pytest.mark.parametrize("stage", ["diagnose", "check"])
def test_existing_outputs_cannot_be_reused(modules, monkeypatch, tmp_path, stage):
    run = owner_run(modules, monkeypatch, tmp_path, stage)
    run.root(check=stage == "check").mkdir()
    with pytest.raises(ValueError, match="no overwrite or retry"):
        run.run()
    assert not run.calls


@pytest.mark.parametrize(
    "key,value",
    [
        ("response_evaluations", 13),
        ("unique_command_count", 14),
        ("numerical_inner_steps", 3900),
        ("query_count", 4368),
        ("total_iterations", 1),
        ("physics_steps", 1),
        ("world_model_calls", 1),
        ("fresh_cases", 1),
        ("actual_consumed_reference_count", 1),
        ("control_qualified", True),
    ],
)
def test_fixed14_counter_contract_rejects_free_repeat_and_extra_activity(modules, key, value):
    receipt = billed_receipt()
    receipt[key] = value
    with pytest.raises(ValueError):
        modules.owner.validate_call_counts(receipt)


@pytest.mark.parametrize(
    "defect",
    [
        "free_repeat",
        "changed_other_axis",
        "changed_signed_zero",
        "sign_order",
        "jit",
        "grid",
        "ledger",
        "strict",
        "fresh",
        "fresh_count",
    ],
)
def test_complete_counts_do_not_hide_changed_grid_or_missing_replay(modules, defect):
    stage = (
        "diagnose" if defect in ("free_repeat", "changed_other_axis", "changed_signed_zero", "sign_order") else "check"
    )
    report = complete_receipt(stage)
    if defect == "free_repeat":
        report["answer"]["request_ledger"][-1]["charged"] = False
    elif defect == "sign_order":
        report["answer"]["request_ledger"][1]["signed_step"] = -1e-3
    elif defect in ("changed_other_axis", "changed_signed_zero"):
        changed = np.asarray(report["answer"]["evaluated_candidates"][1]["commands"])
        if defect == "changed_signed_zero":
            changed[0, 3] = 0.0
        else:
            changed[10, 2] += 1e-8
        report["answer"]["evaluated_candidates"][1]["commands"] = changed.tolist()
        report["answer"]["request_ledger"][1].update(commands=changed.tolist(), command_key=changed.tobytes().hex())
    elif defect == "jit":
        report["native_quaternion_jit_signatures"] = []
    elif defect == "strict":
        report["boundary_classification_disagreements"] = [{}]
    elif defect == "fresh":
        report["independent_restoration_check"]["passed"] = False
    elif defect == "fresh_count":
        report["independent_restoration_check"]["response_evaluations"] = 13
    else:
        report[defect + "_check"]["passed"] = False
    with pytest.raises(ValueError):
        modules.owner.validate_receipt(report, stage)


def test_native_qa95_sources29_modules_preserves_original85_digest_subset(modules, monkeypatch, tmp_path):
    owner = modules.owner
    qa = native_qa(modules, monkeypatch, tmp_path)
    for name in (
        "execute_tracking_v5_terminal_secant_probe",
        "execute_tracking_v5_directional_derivative_check_v3",
        "execute_tracking_v5_directional_derivative_v2",
        "execute_tracking_v5_directional_derivative",
        "execute_tracking_v5_accepted_point_sqp",
    ):
        module = importlib.import_module(name)
        monkeypatch.setattr(module, "REPO", tmp_path)
        monkeypatch.setattr(module, "NATIVE_PYTHON", tmp_path / "native_python")
    previous_owner = importlib.import_module("execute_tracking_v5_terminal_secant_probe")
    assert len(owner.QA_SOURCES) == 95 and len(owner.QA_TESTS) == 29
    new_sources = set(owner.QA_SOURCES) - set(owner.PRIOR_QA_SOURCES)
    new_tests = set(owner.QA_TESTS) - set(previous_owner.QA_TESTS)
    assert len(new_sources) == 10 and len(new_tests) == 3
    old_qa = copy.deepcopy(qa)
    old_qa["source_sha256"] = {path: qa["source_sha256"][path] for path in owner.PRIOR_QA_SOURCES}
    old_qa["command"] = [str(tmp_path / "native_python"), "-m", "pytest", *previous_owner.QA_TESTS]
    old_log = write_json(tmp_path / "prior_qa/pytest.log", {})
    old_qa["log_sha256"] = old_log["sha256"]
    old_ref = write_json(tmp_path / "prior_qa/result.json", old_qa)
    monkeypatch.setattr(owner, "PRIOR_QA_REF", old_ref)
    target = tmp_path / "qa/result.json"
    assert owner.verify_native_qa(write_json(target, qa)) == qa
    assert owner.verify_prior_qa_subset(qa, {}) == old_qa
    for source in new_sources:
        changed = copy.deepcopy(qa)
        del changed["source_sha256"][source]
        with pytest.raises(ValueError):
            owner.verify_native_qa(write_json(target, changed))
    for module in new_tests:
        changed = copy.deepcopy(qa)
        changed["command"].remove(module)
        with pytest.raises(ValueError):
            owner.verify_native_qa(write_json(target, changed))
    changed = copy.deepcopy(qa)
    changed["source_sha256"][owner.PRIOR_QA_SOURCES[0]] = "0" * 64
    with pytest.raises(ValueError, match="identical digest subset"):
        owner.verify_prior_qa_subset(changed, {})
    ref = write_json(target, qa)
    (tmp_path / sorted(new_sources)[0]).write_text("source changed after QA")
    with pytest.raises(ValueError):
        owner.verify_native_qa(ref)


def prior_secant_fixture(source, anchor, interface):
    """Minimal saved algebra archive; no predictor or simulator is called."""
    groups = {
        "inner_lower": dict(slack=np.zeros((300, 7)).tolist()),
        "inner_upper": dict(slack=np.zeros((300, 7)).tolist()),
        "braking_lower": dict(slack=np.zeros((12, 7)).tolist()),
        "braking_upper": dict(slack=np.zeros((12, 7)).tolist()),
        "terminal": dict(slack=[0.02]),
    }
    historical = dict(
        commands=anchor.tolist(),
        safety_slack=np.zeros(4369).tolist(),
        audit=dict(groups=groups),
        rollout=dict(joint_velocity=np.zeros((12, 7)).tolist()),
    )
    historical["safety_slack"][-1] = 0.02
    historical["rollout"]["joint_velocity"][-1][6] = -0.02
    entries, ledger, predictions = [historical], [dict(role="anchor")], []
    old_row = np.zeros(72)
    old_row[6] = 2.0
    for index in range(1, 16):
        entry = copy.deepcopy(historical)
        command = anchor.copy()
        command[1, 0] += index * 1e-5
        predicted = float(old_row @ (command - anchor).ravel())
        terminal = 0.02 + predicted - index * 1e-7
        observed, residual = terminal - 0.02, (terminal - 0.02) - predicted
        entry["commands"] = command.tolist()
        entry["safety_slack"][-1] = entry["audit"]["groups"]["terminal"]["slack"][0] = terminal
        entry["rollout"]["joint_velocity"][-1][6] = -terminal
        entries.append(entry)
        ledger.append(dict(role="grid", response_index=index, request_index=index, command_key=command.tobytes().hex()))
        predictions.append(
            dict(
                request_index=index,
                original_predicted_change=predicted,
                observed_change=observed,
                original_residual=residual,
                corrected_residual=1e-12,
            )
        )
    entries.append(copy.deepcopy(historical))
    ledger.append(dict(role="anchor_repeat"))
    return dict(
        answer=dict(
            geometry=dict(anchor_commands=anchor.tolist()),
            input_min=interface.input_min.tolist(),
            input_max=interface.input_max.tolist(),
            previous_command=source["previous_command"][:6].tolist(),
            evaluated_candidates=entries,
            request_ledger=ledger,
            terminal_row_index=4368,
            fit=dict(original_terminal_row=old_row.tolist()),
            predictions=predictions,
        )
    )


@pytest.mark.parametrize(
    "defect", [None, "missing_last", "response_binding", "wrong_original_residual", "analysis_binding"]
)
def test_prepare_case_uses_all15_original_residuals_and_cross_checks_saved_analysis(
    modules, monkeypatch, tmp_path, defect
):
    source, anchor, interface = source_fixture()
    saved = prior_secant_fixture(source, anchor, interface)
    original = copy.deepcopy(saved)
    residuals = [row["original_residual"] for row in saved["answer"]["predictions"]]
    assert abs(residuals[-1]) == max(abs(value) for value in residuals)
    analysis = dict(
        analysis=dict(points=[dict(request_index=i + 1, old_residual=value) for i, value in enumerate(residuals)])
    )
    if defect == "missing_last":
        saved["answer"]["predictions"].pop()
    elif defect == "response_binding":
        saved["answer"]["request_ledger"][-2]["response_index"] = 1
    elif defect == "wrong_original_residual":
        saved["answer"]["predictions"][-1]["original_residual"] = 1e-12
    elif defect == "analysis_binding":
        analysis["analysis"]["points"][-1]["old_residual"] = residuals[0]
    ref = write_json(tmp_path / "prior_analysis.json", analysis)
    monkeypatch.setattr(modules.producer, "PRIOR_ANALYSIS_REF", ref)
    if defect:
        with pytest.raises(ValueError):
            modules.producer.prepare_case(source, saved, interface)
    else:
        grid, error, binding = modules.producer.prepare_case(source, saved, interface)
        assert error == max(abs(value) for value in residuals) > 1e-12
        assert binding["original_residuals"] == residuals and binding["request_indices"] == list(range(1, 16))
        assert binding["reserve_target"] == 10.0 * error and binding["terminal_slack_target"] == 1e-8 + 10.0 * error
        assert grid["admitted"] and len(grid["grid_commands"]) == 12
        assert saved == original  # In particular, corrected residuals never substitute for the original row.


def grid_case():
    source, anchor, interface = source_fixture()
    core = importlib.import_module("s2a.tail_translation_reserve")
    grid = core.build_tail_translation_grid(
        anchor, source["previous_command"][:6], interface.input_min, interface.input_max
    )
    return source, grid, interface


@pytest.mark.parametrize(
    "defect,completed",
    [
        ("wrong_first", 0),
        ("reordered_axis", 1),
        ("early_repeat", 1),
        ("fifteenth", 14),
    ],
)
def test_predictor_rejects_signed_zero_order_early_repeat_and_over_budget(modules, defect, completed):
    source, grid, interface = grid_case()
    dynamics, calls, progress = CountingDynamics(), [], {}

    def invalid(predict, previous_command, anchor, *args, **kwargs):
        expected = [anchor, *[row["commands"] for row in grid["grid_commands"]], anchor]
        if defect == "wrong_first":
            changed = anchor.copy()
            changed[0, 3] = 0.0
            predict(changed)
        elif defect == "reordered_axis":
            predict(anchor)
            predict(expected[5])
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
            grid,
            {},
            1e-6,
            interface,
            dynamics,
            probe_fn=invalid,
            rollout_fn=captured_rollout(source, calls),
            progress=progress,
        )
    assert len(calls) == progress["response_evaluations"] == completed
    assert progress["prediction_calls"] == completed + 1
    assert progress["numerical_inner_steps"] == 300 * completed and progress["query_count"] == 1 + 312 * completed


def test_all12_grid_commands_admitted_before_current_query(modules):
    source, grid, interface = grid_case()
    grid["grid_commands"][-1]["commands"][11, 2] = 0.6
    dynamics = CountingDynamics()
    with pytest.raises(ValueError):
        modules.producer.run_probe(source, grid, {}, 1e-6, interface, dynamics)
    assert dynamics.queries == 0


def test_legal_command_in_wrong_period_is_independently_rejected_before_current_query(modules):
    source, grid, interface = grid_case()
    command = grid["anchor_commands"].copy()
    command[10, 0] += 1e-3
    # Original box/rate admission alone permits this command. The registered
    # tail-only geometry must still reject it before any native query.
    admitted = modules.evidence.admit_anchor(source, command, interface)
    assert admitted.tobytes() == command.tobytes()
    grid["grid_commands"][0].update(commands=command, command_key=command.tobytes().hex())
    dynamics = CountingDynamics()
    with pytest.raises(ValueError):
        modules.producer.run_probe(source, grid, {}, 1e-6, interface, dynamics)
    assert dynamics.queries == 0


def test_actual_query_bill_is_not_inferred_from_response_counter(modules):
    source, grid, interface = grid_case()
    dynamics, calls, progress = CountingDynamics(), [], {}

    def once(predict, previous_command, anchor, *args, **kwargs):
        predict(anchor)

    with pytest.raises(ValueError, match="bills300 inner"):
        modules.producer.run_probe(
            source,
            grid,
            {},
            1e-6,
            interface,
            dynamics,
            probe_fn=once,
            rollout_fn=captured_rollout(source, calls, query_error=-1),
            progress=progress,
        )
    assert progress["prediction_calls"] == progress["response_evaluations"] == 1
    assert progress["query_count"] == 312


def test_producer_dry_prepares_full_grid_and_prior_error_without_model(modules, monkeypatch, tmp_path):
    source, grid, interface = grid_case()
    producer = modules.producer
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

    def prepare(original, saved_prior, current_interface):
        assert original is source
        np.testing.assert_array_equal(current_interface.input_max, interface.input_max)
        prepared.append(True)
        return grid, 1e-6, {}

    def forbidden(*args, **kwargs):
        raise AssertionError("No native model or response in dry")

    monkeypatch.setattr(producer, "prepare_case", prepare)
    monkeypatch.setattr(producer, "model_registration", lambda reg, refs: ({}, {}))
    monkeypatch.setattr(producer, "MujocoRobotDynamics", forbidden)
    monkeypatch.setattr(producer, "run_probe", forbidden)
    producer.main(dry=True)
    assert prepared == [True] and not (tmp_path / "not-created").exists()


def complete_case(*, restores=False, base=2e-8):
    from test_tail_translation_reserve import synthetic_case

    case = synthetic_case(base=base) if restores else synthetic_case(base=base, terminal=lambda commands: base)
    source, _, _ = source_fixture()
    source.update(
        previous_command=np.r_[case["previous"], 0.0],
        joint_limits=np.column_stack((case["lower"], case["upper"])),
        desired_tcp=case["desired"].copy(),
        site_from_tcp=case["site_from_tcp"].copy(),
        past_inner_error_rad=case["past_inner_error_rad"].copy(),
        terminal_normals=case["terminal_normals"].copy(),
    )
    return source, case


def complete_rollout(source, case, calls):
    capture = captured_rollout(source, calls)

    def rollout(*args, **kwargs):
        commands = args[2].copy()
        capture(*args, **kwargs)  # Counts312 and deliberately mutates all passed state copies.
        return case["predict"](commands)

    return rollout


def run_complete(modules, source, case, dynamics, calls):
    return modules.producer.run_probe(
        source,
        case["grid"],
        case["historical_anchor"],
        case["prior_max_abs_residual"],
        case["interface"],
        dynamics,
        rollout_fn=complete_rollout(source, case, calls),
    )


@pytest.mark.parametrize("restores", [False, True])
def test_default_core_completes14_and_resets_every_call_for_both_scientific_outcomes(modules, restores):
    source, case = complete_case(restores=restores)
    original = copy.deepcopy(source)
    input_before = prior.canonical({key: case[key] for key in ("grid", "historical_anchor")})
    dynamics, calls = CountingDynamics(), []
    result = run_complete(modules, source, case, dynamics, calls)
    assert result["prediction_calls"] == result["response_evaluations"] == len(calls) == 14
    assert result["unique_command_count"] == 13
    assert result["numerical_inner_steps"] == 4200 and result["query_count"] == dynamics.queries == 4369
    answer = result["answer"]
    assert answer["improvement_found"] is answer["sufficient_reserve_restored"] is restores
    assert (answer["selected"] is not None) is restores and answer["repeatability"]["exact"]
    expected = [case["anchor"], *[row["commands"] for row in case["grid"]["grid_commands"]], case["anchor"]]
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
    assert prior.canonical({key: case[key] for key in ("grid", "historical_anchor")}) == input_before
    modules.owner.validate_receipt(complete_receipt("diagnose", prior.canonical(answer)), "diagnose")


@pytest.mark.parametrize("defect", [None, "repeat_rollout", "safety_vector", "strict_label", "one_query_missing"])
def test_checker_replays_all14_including_repeat_and_checks_entire_saved_response(modules, defect):
    source, case = complete_case()
    answer = prior.canonical(run_complete(modules, source, case, CountingDynamics(), [])["answer"])
    reference_entries = copy.deepcopy(answer["evaluated_candidates"])
    frozen = importlib.import_module("check_tracking_v5_prefix_trust_control")
    by_command = {frozen.command_key(row["commands"], 12): row for row in reference_entries}
    result = dict(
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
        result["numerical_inner_steps"] += 300
        result["mj_forward_queries"] += 312
        if defect == "one_query_missing" and len(calls) == 14:
            result["mj_forward_queries"] -= 1
        return copy.deepcopy(active["entry"]["rollout"]), []

    def classify(commands, rollout):
        entry = active["entry"]
        q = frozen.normalized_errors(entry["tcp_errors"], 12)
        return dict(
            tcp_errors=entry["tcp_errors"],
            normalized_squared_errors=q,
            safety_slack=entry["safety_slack"],
            safety_min=float(np.min(entry["safety_slack"])),
            joint_feasible=entry["audit"]["feasible"],
            prefix_length=frozen.continuous_prefix(q),
            feasible=bool(entry["audit"]["feasible"] and np.all(q <= 1.0)),
            maximum_normalized_squared_error=float(np.max(q)),
            maximum_pose_cap_violation=float(max(0.0, np.max(np.sqrt(q)) - 1)),
            pose_loss=float(np.sum(q)),
        )

    def check_audit(saved, metrics):
        frozen.compare_contract(active["entry"]["audit"], saved, name="complete_synthetic_audit")
        audits.append(True)

    if defect == "repeat_rollout":
        answer["evaluated_candidates"][-1]["rollout"]["periods"][0]["requested_torque"][0][0] += 1e-3
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
            result,
            compare=frozen.numerical_compare,
        )

    if defect is not None:
        with pytest.raises(ValueError):
            run()
    else:
        run()
        assert result["verified_responses"] == len(audits) == 14
        assert result["numerical_inner_steps"] == 4200 and result["query_count"] == 4369
        assert result["repeated_responses_checked"] == 1 and result["unique_command_count"] == 13
        assert result["boundary_classification_disagreements"] == []
    assert len(calls) == 14 and calls[0] == calls[-1] and len(set(calls)) == 13


def fresh_inputs(answer):
    rollouts, metrics = [], []
    for entry in answer["evaluated_candidates"]:
        rollouts.append(copy.deepcopy(entry["rollout"]))
        metrics.append(
            dict(
                joint_feasible=entry["audit"]["feasible"],
                safety_slack=np.asarray(entry["safety_slack"]).copy(),
                normalized_squared_errors=np.asarray(entry["normalized_squared_errors"]).copy(),
                tcp_errors=np.asarray(entry["tcp_errors"]).copy(),
                groups=copy.deepcopy(entry["audit"]["groups"]),
                prefix_length=entry["prefix_length"],
                feasible=entry["feasible"],
                maximum_normalized_squared_error=entry["maximum_normalized_squared_error"],
            )
        )
    return rollouts, metrics


@pytest.mark.parametrize(
    "defect", [None, "prefix_torque", "rotation_goal", "repeat_metric", "buffer_one_ulp", "joint_flag"]
)
def test_fresh_replay_strict_flags_are_not_replaced_by_cross_run_float_tolerance(modules, defect):
    source, case = complete_case(base=1e-8 if defect == "buffer_one_ulp" else 2e-8)
    answer = prior.canonical(run_complete(modules, source, case, CountingDynamics(), [])["answer"])
    rollouts, metrics = fresh_inputs(answer)
    frozen = importlib.import_module("check_tracking_v5_prefix_trust_control")
    if defect == "prefix_torque":
        rollouts[1]["periods"][0]["requested_torque"][0][0] += 1e-13
        frozen.compare_contract(rollouts[1], answer["evaluated_candidates"][1]["rollout"])
    elif defect == "rotation_goal":
        rollouts[1]["periods"][-1]["goal"][0][1] += 1e-13
        frozen.compare_contract(rollouts[1], answer["evaluated_candidates"][1]["rollout"])
    elif defect == "repeat_metric":
        metrics[-1]["maximum_normalized_squared_error"] += 1e-13
        frozen.compare_contract(metrics[-1], metrics[0])
    elif defect == "buffer_one_ulp":
        before = metrics[1]["safety_slack"].copy()
        assert before[-1] == 1e-8 and answer["candidate_checks"][0]["safety_buffer_satisfied"] is True
        metrics[1]["safety_slack"][-1] = np.nextafter(1e-8, -np.inf)
        frozen.numerical_compare("sub-tolerance buffer crossing", before, metrics[1]["safety_slack"])
    elif defect == "joint_flag":
        metrics[1]["joint_feasible"] = False
    if defect is None:
        result = modules.checker.verify_replayed_restoration(answer, rollouts, metrics)
        assert result["passed"] and result["response_evaluations"] == 14
        assert result["strict_disagreements"] == []
        assert result["improvement_found"] is result["sufficient_reserve_restored"] is False
    else:
        with pytest.raises(ValueError, match="strict flag|strict restoration"):
            modules.checker.verify_replayed_restoration(answer, rollouts, metrics)


def test_consistent_fresh_and_saved_causal_failure_is_retained_as_scientific_failure(modules):
    source, case = complete_case()
    original, calls = case["predict"], []

    def with_causal_fault(commands):
        response = original(commands)
        calls.append(True)
        if len(calls) == 2:
            response["periods"][0]["requested_torque"][0, 0] += 1e-13
        return response

    case["predict"] = with_causal_fault
    answer = prior.canonical(run_complete(modules, source, case, CountingDynamics(), [])["answer"])
    assert answer["causal_checks"][1]["passed"] is False and not answer["improvement_found"]
    result = modules.checker.verify_replayed_restoration(answer, *fresh_inputs(answer))
    assert result["passed"] and result["response_evaluations"] == 14
    assert result["causal_checks"][1]["passed"] is False
    assert result["improvement_found"] is result["sufficient_reserve_restored"] is False


@pytest.mark.parametrize(
    "defect", [None, "changed_E17", "omitted_binding_point", "omitted_analysis_point", "corrected_residual"]
)
def test_independent_prior_binding_preserves_all15_points_and_original_E17(modules, monkeypatch, tmp_path, defect):
    source, anchor, interface = source_fixture()
    prior_result = prior_secant_fixture(source, anchor, interface)
    residuals = [row["original_residual"] for row in prior_result["answer"]["predictions"]]
    analysis = dict(
        analysis=dict(points=[dict(request_index=i + 1, old_residual=value) for i, value in enumerate(residuals)])
    )
    ref = write_json(tmp_path / "prior_analysis.json", analysis)
    monkeypatch.setattr(modules.producer, "PRIOR_ANALYSIS_REF", ref)
    monkeypatch.setattr(modules.checker, "PRIOR_ANALYSIS_REF", ref)
    _, error, binding = modules.producer.prepare_case(source, prior_result, interface)
    registration = dict(prior_error_binding=binding, prior_max_abs_residual=error)
    answer = dict(prior_max_abs_residual=error, reserve_target=10 * error, terminal_threshold=1e-8 + 10 * error)
    checked_prior = dict(original_residuals=residuals, prior_max_abs_residual=error)
    if defect == "changed_E17":
        answer["prior_max_abs_residual"] = np.nextafter(error, np.inf)
    elif defect == "omitted_binding_point":
        registration["prior_error_binding"]["request_indices"].pop()
    elif defect == "omitted_analysis_point":
        analysis["analysis"]["points"].pop()
    elif defect == "corrected_residual":
        analysis["analysis"]["points"][-1]["old_residual"] = 1e-12
    if defect is None:
        assert modules.checker.validate_prior_binding(registration, answer, checked_prior, analysis) == binding
    else:
        with pytest.raises(ValueError):
            modules.checker.validate_prior_binding(registration, answer, checked_prior, analysis)
