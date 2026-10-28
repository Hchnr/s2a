"""Synthetic callback-search admission, reset, billing and replay integration."""

import copy
import importlib
import json
import sys
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np
import pytest
from test_prefix_trust_asset_check_v2 import synthetic_main
from test_prefix_trust_control_execution import (
    CountingDynamics,
    captured_rollout,
    source_fixture,
    three_response_solver,
)

from s2a.prefix_trust_control_evidence import reference


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return SimpleNamespace(
        producer=importlib.import_module("probe_tracking_v5_prefix_callback_backtracking"),
        evidence=importlib.import_module("s2a.prefix_callback_backtracking_evidence"),
        owner=importlib.import_module("execute_tracking_v5_prefix_callback_backtracking"),
    )


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return reference(path)


def install_solver(monkeypatch, solver):
    monkeypatch.setitem(
        sys.modules,
        "s2a.prefix_callback_backtracking_plan",
        SimpleNamespace(solve_prefix_callback_backtracking_response=solver),
    )


def test_default_solver_is_new_and_every_actual_response_resets_and_is_billed(modules, monkeypatch):
    source, warm, interface = source_fixture()
    original, warm_bytes = copy.deepcopy(source), warm.tobytes()
    dynamics, calls, solver_calls, options = CountingDynamics(), [], [], []
    original_solver = three_response_solver(source, dynamics, solver_calls)

    def solver(*args, **kwargs):
        options.append(dict(kwargs))
        assert kwargs["max_stages"] == kwargs["max_iterations"] == 60
        return original_solver(*args, **kwargs)

    install_solver(monkeypatch, solver)
    result = modules.producer.run_search(source, warm, interface, dynamics, rollout_fn=captured_rollout(source, calls))
    assert solver_calls == [True] and len(options) == 1
    assert result["prediction_calls"] == result["response_evaluations"] == len(calls) == 3
    assert result["numerical_inner_steps"] == 900 and result["query_count"] == dynamics.queries == 937
    assert calls[0]["commands"].tobytes() == warm_bytes == warm.tobytes()
    for call in calls:
        for field, source_field in (
            ("q", "current_q"),
            ("velocity", "current_velocity"),
            ("latch", "latched_rotation_goal"),
        ):
            np.testing.assert_array_equal(call[field], original[source_field])
    for key, value in source.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, original[key])


def test_new_adapter_keeps_duplicate_raw_response_rejection_and_partial_counters(modules):
    source, warm, interface = source_fixture()
    dynamics, calls, progress = CountingDynamics(), [], {}

    def duplicate(predict, previous, start, *args, **kwargs):
        assert kwargs["max_stages"] == 60
        predict(start)
        predict(start.copy())

    with pytest.raises(ValueError, match="duplicate raw responses"):
        modules.producer.run_search(
            source,
            warm,
            interface,
            dynamics,
            solver_fn=duplicate,
            rollout_fn=captured_rollout(source, calls),
            progress=progress,
        )
    assert progress["prediction_calls"] == 2 and progress["response_evaluations"] == len(calls) == 1
    assert progress["numerical_inner_steps"] == 300 and progress["query_count"] == 313


def snapshot_fixture(monkeypatch, tmp_path, evidence):
    snapshot = {}
    for key in ("original_xml_ref", "migrated_xml_ref", "snapshot_ref", "compile_ref", "historical_manifest_ref"):
        snapshot[key] = write_json(tmp_path / (key + ".json"), {"synthetic": key})
    monkeypatch.setattr(evidence, "ORIGINAL_XML_REF", snapshot["original_xml_ref"])
    monkeypatch.setattr(evidence, "load_snapshot", lambda: snapshot)
    refs = {item["path"]: item for item in snapshot.values()}
    return snapshot, refs


def test_model_registration_changes_only_the_bound_xml_locator(modules, monkeypatch, tmp_path):
    snapshot, refs = snapshot_fixture(monkeypatch, tmp_path, modules.evidence)
    original = dict(xml_ref=snapshot["original_xml_ref"], states_ref={"fixed": 368}, controller={"latch": 367})
    before = copy.deepcopy(original)
    migrated, proof = modules.evidence.model_registration(original, refs)
    assert original == before and migrated is not original
    assert migrated["xml_ref"] == snapshot["migrated_xml_ref"]
    assert migrated["states_ref"] is original["states_ref"] and migrated["controller"] is original["controller"]
    assert all(proof[key] == value for key, value in snapshot.items())
    assert proof["complete_model_preserved"] is True and proof["model_body_or_geom_pruning"] is False
    del refs[snapshot["compile_ref"]["path"]]
    with pytest.raises(ValueError, match="already admitted"):
        modules.evidence.model_registration(original, refs)


def test_checker_engine_wrapper_preserves_all_frozen_inputs_except_xml(modules, monkeypatch, tmp_path):
    checker = importlib.import_module("check_tracking_v5_prefix_callback_backtracking")
    snapshot, refs = snapshot_fixture(monkeypatch, tmp_path, modules.evidence)
    source, _, _ = source_fixture()
    registration = dict(xml_ref=snapshot["original_xml_ref"], states_ref={"actual": 368}, controller={"latch": 367})
    prefix_registration, result, compare, sentinel = {}, {}, object(), object()
    calls = []

    def frozen(name, original, migrated, prefix, output, received_refs, received_compare):
        assert name == "candidate" and original is source and prefix is prefix_registration
        assert output is result and received_refs is refs and received_compare is compare
        assert migrated is not registration and migrated["xml_ref"] == snapshot["migrated_xml_ref"]
        assert migrated["states_ref"] is registration["states_ref"]
        assert migrated["controller"] is registration["controller"]
        calls.append(True)
        return sentinel

    monkeypatch.setattr(checker, "model_registration", modules.evidence.model_registration)
    monkeypatch.setattr(checker, "frozen_case_engine", frozen)
    assert (
        checker.case_engine("candidate", source, registration, prefix_registration, result, refs, compare) is sentinel
    )
    assert calls == [True] and registration["xml_ref"] == snapshot["original_xml_ref"]
    assert result["asset_loading_proof"]["snapshot_ref"] == snapshot["snapshot_ref"]


def trust_replay_fixture(modules, monkeypatch, tmp_path, defect=None):
    """Hash-bound synthetic receipts exercise the real closed_owner validator."""
    evidence = modules.evidence
    monkeypatch.setattr(evidence, "REPO", tmp_path)
    monkeypatch.setattr(evidence, "NATIVE_PYTHON", tmp_path / "python")

    def root(case, *, owner=False):
        return tmp_path / (case + ("_owner" if owner else "_check"))

    monkeypatch.setattr(evidence, "repair_root", root)
    snapshot, _ = snapshot_fixture(monkeypatch, tmp_path, evidence)
    qa = write_json(tmp_path / "qa.json", {"synthetic": True})
    common_source = write_json(tmp_path / "source.json", {"frozen": True})
    producer_refs, registrations = {}, {}
    for case, count in (("baseline", 4000), ("candidate", 3148)):
        producer_refs[case] = write_json(tmp_path / (case + "_producer.json"), {"synthetic": case})
        report_reg = write_json(root(case) / "registration.json", dict(producer_ref=producer_refs[case]))
        sources = {common_source["path"]: common_source["sha256"]}
        if defect == "different_sources" and case == "candidate":
            extra = write_json(tmp_path / "extra_source.json", {"changed_roster": True})
            sources[extra["path"]] = extra["sha256"]
        reg = dict(
            case_name=case,
            stage="check_v2",
            producer_ref=producer_refs[case],
            native_component_qa_ref=qa,
            source_sha256=sources,
            snapshot_ref=snapshot["snapshot_ref"],
            compile_ref=snapshot["compile_ref"],
        )
        registrations[case] = write_json(root(case, owner=True) / "registration.json", reg)
        proof = dict(snapshot, asset_count=96, complete_model_preserved=True, model_body_or_geom_pruning=False)
        report = dict(
            complete=True,
            passed=True,
            case_name=case,
            registration_ref=report_reg,
            response_evaluations=count,
            verified_responses=count,
            numerical_inner_steps=count * 300,
            query_count=count * 312 + 1,
            mj_forward_queries=count * 312 + 1,
            boundary_classification_disagreements=[],
            witness_found=False,
            control_qualified=False,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            asset_loading_proof=proof,
            source_refs_unchanged=[common_source],
        )
        if case == "candidate":
            if defect == "partial":
                report["verified_responses"] -= 1
            elif defect == "disagreement":
                report["boundary_classification_disagreements"] = [{"synthetic": True}]
            elif defect == "different_snapshot":
                report["asset_loading_proof"]["snapshot_ref"] = dict(path="other", sha256="other")
            elif defect == "physical_step":
                report["physics_steps"] = 1
        result_ref = write_json(root(case) / "verified.json", report)
        command = [
            str(evidence.NATIVE_PYTHON),
            str(tmp_path / "scripts/check_tracking_v5_prefix_trust_control_v2.py"),
            "--case",
            case,
        ]
        if defect == "wrong_launch" and case == "candidate":
            command[1] = str(tmp_path / "other_checker.py")
        launch = write_json(
            root(case, owner=True) / "launch.json",
            dict(
                automatic_retry=False,
                selected_environment=dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
                command=command,
                child=dict(pid=1234, start_ticks=5678),
            ),
        )
        exited = write_json(
            root(case, owner=True) / "exit.json",
            dict(automatic_retry=False, actual_wait=not (defect == "no_wait" and case == "candidate"), returncode=0),
        )
        release = write_json(
            root(case, owner=True) / "release.json",
            dict(
                reaped=True,
                returncode=0,
                signal_sent=defect == "signalled" and case == "candidate",
                original_error=None,
                pid=1234,
                start_ticks=5678,
            ),
        )
        owner = dict(
            complete=True,
            passed=True,
            case_name=case,
            stage="check_v2",
            registration_ref=registrations[case],
            result_ref=result_ref,
            owned_command=dict(launch=launch, exit=exited, release=release),
        )
        if case == "candidate":
            if defect == "running":
                owner["complete"] = owner["passed"] = False
            elif defect == "unregistered_owner":
                owner["registration_ref"] = write_json(tmp_path / "alternate_registration.json", reg)
        write_json(root(case, owner=True) / "result.json", owner)
    monkeypatch.setattr(evidence, "PRODUCER_REFS", producer_refs)
    monkeypatch.setattr(evidence, "TRUST_REGISTRATION_REFS", registrations)
    if defect == "missing_owner":
        (root("candidate", owner=True) / "result.json").unlink()


def test_closed_replay_admission_binds_both_actual_result_hashes(modules, monkeypatch, tmp_path):
    trust_replay_fixture(modules, monkeypatch, tmp_path)
    refs = {}
    result = modules.evidence.closed_trust_replays(refs)
    assert set(result) == {"baseline", "candidate"}
    for row in result.values():
        for ref in row.values():
            assert refs[ref["path"]] == reference(ref["path"]) == ref


@pytest.mark.parametrize(
    "defect",
    [
        "missing_owner",
        "running",
        "partial",
        "no_wait",
        "signalled",
        "wrong_launch",
        "unregistered_owner",
        "different_snapshot",
        "different_sources",
        "disagreement",
        "physical_step",
    ],
)
def test_live_partial_changed_or_unqualified_prior_replay_cannot_admit_search(modules, monkeypatch, tmp_path, defect):
    trust_replay_fixture(modules, monkeypatch, tmp_path, defect)
    with pytest.raises((ValueError, FileNotFoundError)):
        modules.evidence.closed_trust_replays({})


def receipt(stage, count=3):
    result = dict(
        complete=True,
        passed=True,
        case_name="candidate",
        response_evaluations=count,
        numerical_inner_steps=300 * count,
        query_count=1 + 312 * count,
        control_qualified=False,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
    )
    if stage == "diagnose":
        result.update(
            prediction_calls=count,
            total_iterations=2,
            answer=dict(
                response_evaluations=count,
                total_iterations=2,
                total_stages=2,
                evaluated_candidates=[{} for _ in range(count)],
            ),
            source_refs_unchanged=[],
        )
    else:
        result.update(
            verified_responses=count, mj_forward_queries=1 + 312 * count, boundary_classification_disagreements=[]
        )
    return result


@pytest.mark.parametrize("stage", ["diagnose", "check"])
def test_receipt_accepts_complete_billed_work_but_rejects_zero_responses(modules, stage):
    modules.owner.validate_receipt(receipt(stage), stage)
    with pytest.raises(ValueError):
        modules.owner.validate_receipt(receipt(stage, count=0), stage)


@pytest.mark.parametrize(
    "stage,field,value",
    [
        ("diagnose", "prediction_calls", 4),
        ("diagnose", "query_count", 938),
        ("check", "verified_responses", 2),
        ("check", "mj_forward_queries", 938),
        ("check", "boundary_classification_disagreements", [{}]),
        ("check", "control_qualified", True),
    ],
)
def test_owner_rejects_unbilled_or_partial_or_relaxed_receipts(modules, stage, field, value):
    saved = receipt(stage)
    saved[field] = value
    with pytest.raises(ValueError):
        modules.owner.validate_receipt(saved, stage)


def new_checker_run(modules, monkeypatch, tmp_path):
    checker = importlib.import_module("check_tracking_v5_prefix_callback_backtracking")
    frozen = importlib.import_module("check_tracking_v5_prefix_trust_control")
    isolated = FunctionType(
        checker.main.__code__, dict(vars(checker)), checker.main.__name__, checker.main.__defaults__
    )
    run = synthetic_main(
        SimpleNamespace(build_main=lambda: isolated, frozen=frozen), monkeypatch, tmp_path, "candidate"
    )
    metadata = sys.modules["s2a.prefix_trust_control_evidence"]
    metadata.load_prerequisites = isolated.__globals__["load_asset_repair_prerequisites"]

    def root(case_name="candidate", *, check=False, owner=False):
        assert case_name == "candidate"
        return tmp_path / ("callback" + ("_check" if check else "") + ("_owner" if owner else ""))

    metadata.case_root = root
    monkeypatch.setitem(sys.modules, "s2a.prefix_callback_backtracking_evidence", metadata)
    owner, _ = metadata.closed_owner(None, None)
    owner["result_ref"] = metadata.reference(root() / "verified.json")
    metadata.read(owner["owned_command"]["launch"])["command"] = [
        str(metadata.NATIVE_PYTHON),
        str(tmp_path / "scripts/probe_tracking_v5_prefix_callback_backtracking.py"),
    ]
    registration = metadata.read(run.producer["registration_ref"])
    proof = dict(full_model=True)
    registration.update(experiment="prefix_callback_backtracking", asset_loading_proof=proof)
    run.producer["asset_loading_proof"] = proof
    namespace = isolated.__globals__
    namespace["model_registration"] = lambda reg, refs: (dict(reg), proof)
    namespace["audit_callback_backtracking_ledger"] = namespace["audit_trust_search_ledger"]
    answer = run.producer["answer"]
    entry0 = copy.deepcopy(answer["initial"])
    entry1 = copy.deepcopy(entry0)
    entry1.update(commands=answer["evaluated_candidates"][1]["commands"].copy(), rollout=dict(index=1))
    attempts = [
        dict(response_index=0, commands=entry0["commands"], entry=copy.deepcopy(entry0), outcome="cache_hit"),
        dict(response_index=0, commands=entry0["commands"], entry=copy.deepcopy(entry0), outcome="cache_hit"),
        dict(response_index=1, commands=entry1["commands"], entry=entry1, outcome="new_response"),
        dict(response_index=None, commands=entry1["commands"], entry=None, outcome="budget_rejected"),
    ]
    answer["phases"] = [dict(ray_attempts=attempts)]
    run.checker, run.attempts = checker, attempts
    return run


def test_checker_replays_unique_commands_once_and_checks_every_full_ray_record(modules, monkeypatch, tmp_path):
    run = new_checker_run(modules, monkeypatch, tmp_path)
    run.main()
    result = run.published[-1][1]
    assert result["complete"] is result["passed"] is True
    assert result["response_evaluations"] == result["verified_responses"] == len(run.rows) == 2
    assert result["complete_ray_rollouts_checked"] == 3
    assert result["query_count"] == 625 and result["numerical_inner_steps"] == 600
    assert len(run.audits) == 7  # Three full retained records, one anchor, three rays.
    assert result["control_qualified"] is False


@pytest.mark.parametrize("defect", ["missing_entry", "wrong_key", "rollout", "audit", "false_budget"])
def test_checker_cannot_skip_or_trust_an_unverified_ray_record(modules, monkeypatch, tmp_path, defect):
    run = new_checker_run(modules, monkeypatch, tmp_path)
    attempt = run.attempts[1]  # A second cache-linked archive must be checked separately too.
    if defect == "missing_entry":
        attempt["entry"] = None
    elif defect == "wrong_key":
        attempt["commands"] = attempt["commands"] + 1e-5
    elif defect == "rollout":
        attempt["entry"]["rollout"] = dict(index=99)
    elif defect == "audit":
        attempt["entry"]["audit"] = dict(synthetic=False)
    else:
        run.attempts[-1]["outcome"] = "cache_hit"
    with pytest.raises((ValueError, AssertionError)):
        run.main()
    result = run.published[-1][1]
    assert result["complete"] is result["passed"] is False
    assert result["response_evaluations"] <= 1


def test_checker_rejects_empty_trace_before_querying_any_model(modules, monkeypatch, tmp_path):
    run = new_checker_run(modules, monkeypatch, tmp_path)
    run.producer["answer"]["evaluated_candidates"] = []
    with pytest.raises(ValueError, match="full response count"):
        run.main()
    assert run.rows == run.queries == run.published == []


@pytest.mark.parametrize("dry", [False, True])
def test_producer_entry_uses_only_snapshot_model_and_actual368_without_free_queries(
    modules, monkeypatch, tmp_path, dry
):
    producer = modules.producer
    source, warm, _ = source_fixture()
    snapshot, refs = snapshot_fixture(monkeypatch, tmp_path, modules.evidence)
    source["registration_ref"] = write_json(tmp_path / "old_registration.json", {})
    source["fixed_gripper_state"] = {"finger1": [0.02, 0.01], "finger2": [-0.02, -0.01]}
    control = dict(
        input_min=[-1.0] * 6,
        input_max=[1.0] * 6,
        output_min=[-0.05] * 6,
        output_max=[0.05] * 6,
        controller_site="same_site",
    )
    registration = dict(
        xml_ref=snapshot["original_xml_ref"],
        control_ref=write_json(tmp_path / "control.json", dict(contract=control)),
        states_ref=write_json(tmp_path / "states.json", {}),
    )
    guard = write_json(
        tmp_path / "guard.json", dict(cv2_guard_registration=dict(exact_probe_ref={}, runtime_source_refs=[]))
    )
    backtracking = dict(cases=dict(prefix=dict(entries=[{}] * 55 + [dict(request=dict(commands=warm))])))
    monkeypatch.setattr(producer, "load_prerequisites", lambda: (source, registration, backtracking, refs))
    monkeypatch.setattr(producer, "case_root", lambda _: tmp_path / "output")
    monkeypatch.setattr(producer, "strict_native_environment", lambda: None)
    monkeypatch.setattr(producer, "model_registration", modules.evidence.model_registration)
    monkeypatch.setattr(producer, "CONTROL_REF", guard)
    monkeypatch.setattr(producer, "SOURCES", ())
    monkeypatch.setattr(producer, "PLAN", Path(write_json(tmp_path / "plan.json", {})["path"]))
    for key, value in producer.CPU_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    models, state_indices, searches, guards, published = [], [], [], [], []
    monkeypatch.setattr(producer, "restore_cv2_environment", lambda *args: guards.append(args))

    def model(path, site, joints, fingers):
        assert str(path) == snapshot["migrated_xml_ref"]["path"]
        assert site == "same_site" and joints == [f"robot0_joint{k}" for k in range(1, 8)]
        assert fingers == source["fixed_gripper_state"]
        dynamics = CountingDynamics()
        dynamics.qids = dynamics.dofs = dynamics.joints = np.arange(7)
        dynamics.model = SimpleNamespace(
            opt=SimpleNamespace(timestep=0.002),
            nq=9,
            jnt_range=np.vstack((source["joint_limits"], [[-1.0, 1.0]] * 2)),
            jnt_qposadr=np.arange(9),
            jnt_dofadr=np.arange(9),
        )
        dynamics.mj = SimpleNamespace(
            mjtObj=SimpleNamespace(mjOBJ_JOINT=0),
            mj_name2id=lambda model, kind, name: {"finger1": 7, "finger2": 8}[name],
        )
        models.append(dynamics)
        return dynamics

    class States:
        def __getitem__(self, index):
            state_indices.append(index)
            assert index == 368
            return np.r_[0.0, source["current_q"], 0.02, -0.02, source["current_velocity"], 0.01, -0.01]

    class Archive:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __getitem__(self, key):
            assert key == "sim_states"
            return States()

    def search(supplied_source, supplied_warm, interface, dynamics, *, progress):
        assert supplied_source is source and supplied_warm.tobytes() == warm.tobytes()
        assert dynamics.queries == 0  # Current-state query belongs inside the billed runner.
        searches.append(True)
        progress.update(receipt("diagnose"), answer=dict(witness_found=False))

    def publish(path, value):
        published.append((path, copy.deepcopy(value)))
        return dict(path=str(path), sha256="synthetic-output")

    monkeypatch.setattr(producer, "MujocoRobotDynamics", model)
    monkeypatch.setattr(producer.np, "load", lambda *args, **kwargs: Archive())
    monkeypatch.setattr(producer, "run_search", search)
    monkeypatch.setattr(producer, "publish_result", publish)
    producer.main(dry=dry)
    assert registration["xml_ref"] == snapshot["original_xml_ref"]
    if dry:
        assert models == state_indices == searches == guards == published == []
    else:
        assert len(models) == len(searches) == len(guards) == 1 and state_indices == [368]
        result = published[-1][1]
        assert result["complete"] is result["passed"] is True
        assert result["asset_loading_proof"]["migrated_xml_ref"] == snapshot["migrated_xml_ref"]
        assert result["actual_consumed_reference_count"] == 0 and result["control_qualified"] is False


def owner_run(modules, monkeypatch, tmp_path, stage, *, partial=False, different_qa=False):
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

    def root(case_name="candidate", *, check=False, owner=False):
        return tmp_path / ("callback" + ("_check" if check else "") + ("_owner" if owner else ""))

    monkeypatch.setattr(owner, "case_root", root)
    log = write_json(tmp_path / "qa/pytest.log", {})
    qa_ref = write_json(tmp_path / "qa/result.json", dict(log_sha256=log["sha256"]))
    monkeypatch.setattr(owner, "verify_native_qa", lambda ref: owner.read(ref))
    monkeypatch.setattr(owner, "load_prerequisites", lambda: ({}, {}, {}, {}))
    if stage == "check":
        producer = receipt("diagnose")
        producer_ref = write_json(root() / "verified.json", producer)
        registration = write_json(
            root(owner=True) / "registration.json",
            dict(
                case_name="candidate",
                stage="diagnose",
                native_component_qa_ref={} if different_qa else qa_ref,
                plan_ref=plan,
            ),
        )
        launch = write_json(root(owner=True) / "launch.json", dict(command=owner.entry_command("diagnose")))
        prior_owner = dict(
            result_ref=producer_ref,
            case_name="candidate",
            stage="diagnose",
            registration_ref=registration,
            owned_command=dict(launch=launch),
        )
        write_json(root(owner=True) / "result.json", prior_owner)
        monkeypatch.setattr(owner, "closed_owner", lambda ref, refs: (prior_owner, producer))
    calls = []

    def child(command, environment, capture, *, timeout):
        calls.append(dict(command=command, environment=environment, timeout=timeout))
        write_json(root(check=stage == "check") / "verified.json", receipt(stage, count=2 if partial else 3))
        return dict(synthetic=True)

    monkeypatch.setattr(owner, "owned_command", child)
    return SimpleNamespace(run=lambda: owner.main(stage, qa_ref), calls=calls, root=root)


@pytest.mark.parametrize("stage,timeout", [("diagnose", 1200), ("check", 5400)])
def test_owner_binds_the_single_entry_and_cpu_environment_to_registered_timeout(
    modules, monkeypatch, tmp_path, stage, timeout
):
    run = owner_run(modules, monkeypatch, tmp_path, stage)
    run.run()
    assert len(run.calls) == 1 and run.calls[0]["timeout"] == timeout
    assert run.calls[0]["command"] == modules.owner.entry_command(stage)
    assert len(run.calls[0]["command"]) == 2  # Candidate is fixed; no inherited --case argument.
    environment = run.calls[0]["environment"]
    assert all(environment[key] == value for key, value in modules.owner.CPU_ENVIRONMENT.items())
    assert "MUJOCO_GL" not in environment and "PYOPENGL_PLATFORM" not in environment
    result = json.loads((run.root(check=stage == "check", owner=True) / "result.json").read_text())
    assert result["complete"] is result["passed"] is True


def test_check_owner_rejects_internally_consistent_but_partial_replay(modules, monkeypatch, tmp_path):
    run = owner_run(modules, monkeypatch, tmp_path, "check", partial=True)
    with pytest.raises(ValueError, match="No partial replay"):
        run.run()
    assert len(run.calls) == 1
    result = json.loads((run.root(check=True, owner=True) / "result.json").read_text())
    assert result["complete"] is result["passed"] is False and "No partial replay" in result["error"]


def test_check_owner_rejects_changed_qa_before_launch(modules, monkeypatch, tmp_path):
    run = owner_run(modules, monkeypatch, tmp_path, "check", different_qa=True)
    with pytest.raises(ValueError, match="identical frozen source-bound QA"):
        run.run()
    assert run.calls == [] and not run.root(check=True, owner=True).exists()


@pytest.mark.parametrize("defect", [None, "missing_source", "changed_source", "missing_test", "no_wait"])
def test_native_qa_covers_the_complete_new_source_and_test_roster(modules, monkeypatch, tmp_path, defect):
    owner = modules.owner
    native = tmp_path / "python"
    for module in (
        owner,
        importlib.import_module("execute_tracking_v5_prefix_trust_check_v2"),
        importlib.import_module("execute_tracking_v5_prefix_trust_control"),
    ):
        monkeypatch.setattr(module, "REPO", tmp_path)
        monkeypatch.setattr(module, "NATIVE_PYTHON", native)
    sources = {path: write_json(tmp_path / path, dict(source=path))["sha256"] for path in owner.QA_SOURCES}
    log = write_json(tmp_path / "qa/pytest.log", {})
    qa = dict(
        complete=True,
        passed=True,
        source_unchanged=True,
        actual_wait=True,
        returncode=0,
        command=[str(native), "-m", "pytest", *owner.QA_TESTS],
        runtime=dict(python="3.11.15", numpy="1.26.4", scipy="1.15.3"),
        environment=dict(owner.CPU_ENVIRONMENT),
        source_sha256=sources,
        log_sha256=log["sha256"],
    )
    if defect == "missing_source":
        del sources[owner.QA_SOURCES[-1]]
    elif defect == "changed_source":
        (tmp_path / owner.QA_SOURCES[-1]).write_text("changed after QA")
    elif defect == "missing_test":
        qa["command"].remove(owner.QA_TESTS[-1])
    elif defect == "no_wait":
        qa["actual_wait"] = False
    ref = write_json(tmp_path / "qa/result.json", qa)
    if defect is None:
        assert owner.verify_native_qa(ref) == qa
    else:
        with pytest.raises(ValueError):
            owner.verify_native_qa(ref)
