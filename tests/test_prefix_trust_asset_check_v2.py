"""Synthetic wiring and source-isolation checks; no model or real response."""

import ast
import copy
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def repair(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("check_tracking_v5_prefix_trust_control_v2")


def test_only_the_two_registered_main_ast_changes_are_applied(repair):
    source = ast.parse(Path(repair.frozen.__file__).read_bytes())
    original = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    changed, proof = repair._rewritten_main()
    expected = copy.deepcopy(original)
    calls = [node for node in ast.walk(expected) if isinstance(node, ast.Call)]
    call = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "load_prerequisites")
    call.func.id = "load_asset_repair_prerequisites"
    output = next(
        node
        for node in ast.walk(expected)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "output" for target in node.targets)
    )
    assert isinstance(output.value.right, ast.JoinedStr)
    assert output.value.right.values[-1].value == "_check_20261028"
    output.value.right.values[-1].value = "_check_v2_20261029"
    assert ast.dump(changed, include_attributes=False) == ast.dump(expected, include_attributes=False)
    assert proof["admission_call_replacements"] == proof["output_suffix_replacements"] == 1
    assert proof["inverse_ast_equal"] is True
    assert proof == repair.main_ast_proof()


@pytest.mark.parametrize("change", ["duplicate_call", "call_arguments", "missing_suffix", "duplicate_output"])
def test_source_drift_outside_the_exact_rewrite_contract_fails_closed(repair, monkeypatch, tmp_path, change):
    original = repair.frozen
    source = Path(original.__file__).read_text()
    if change == "duplicate_call":
        source = source.replace(
            "    original, prefix_reg, backtracking, refs = load_prerequisites()",
            "    load_prerequisites()\n    original, prefix_reg, backtracking, refs = load_prerequisites()",
        )
    elif change == "call_arguments":
        source = source.replace("refs = load_prerequisites()", "refs = load_prerequisites(1)")
    elif change == "missing_suffix":
        source = source.replace('_check_20261028"', '_unregistered_output"')
    else:
        line = '    output = REPO / f"outputs/tracking_v5_prefix_trust_{case}_check_20261028"'
        assert source.count(line) == 1
        source = source.replace(line, line + "\n" + line)
    path = tmp_path / "changed.py"
    path.write_text(source)
    monkeypatch.setattr(repair, "frozen", SimpleNamespace(__file__=str(path), require=original.require))
    with pytest.raises(ValueError):
        repair.build_main()


def test_compiled_namespace_preserves_frozen_module_and_numeric_functions(repair):
    before = dict(vars(repair.frozen))
    main = repair.build_main()
    assert main is not repair.frozen.main
    assert main.__globals__ is not vars(repair.frozen)
    assert set(vars(repair.frozen)) == set(before)
    assert all(vars(repair.frozen)[key] is value for key, value in before.items())
    for name in (
        "numerical_compare",
        "compare_contract",
        "validate_trace_entry",
        "check_full_candidate",
        "audit_trust_search_ledger",
        "audit_prefix_search_ledger",
        "native_math_proof",
    ):
        assert main.__globals__[name] is before[name]
    assert main.__globals__["case_engine"] is repair.case_engine
    assert main.__globals__["load_asset_repair_prerequisites"] is repair.load_asset_repair_prerequisites
    main.__globals__["REPO"] = Path("isolated-test-namespace")
    assert repair.frozen.REPO is before["REPO"]


def synthetic_snapshot():
    def ref(name):
        return dict(path=f"/synthetic/{name}", sha256=name)

    return dict(
        original_xml_ref=ref("original.xml"),
        migrated_xml_ref=ref("migrated.xml"),
        historical_manifest_ref=ref("historical.json"),
        snapshot_ref=ref("snapshot.json"),
        compile_ref=ref("compile.json"),
        assets=[ref("mesh"), ref("texture")],
    )


@pytest.mark.parametrize("invalid", [None, "original_xml", "unbound_migration", "unbound_compile"])
def test_engine_only_substitutes_bound_xml_and_preserves_original_inputs(repair, monkeypatch, invalid):
    snapshot = synthetic_snapshot()
    original, source_reg, prefix_reg = (
        {"state": [1, 2]},
        {"xml_ref": snapshot["original_xml_ref"], "control_ref": {"keep": True}},
        {"frozen": True},
    )
    refs = {ref["path"]: copy.deepcopy(ref) for ref in snapshot.values() if isinstance(ref, dict)}
    if invalid == "original_xml":
        source_reg["xml_ref"] = dict(path="different.xml", sha256="different")
    elif invalid == "unbound_migration":
        del refs[snapshot["migrated_xml_ref"]["path"]]
    elif invalid == "unbound_compile":
        del refs[snapshot["compile_ref"]["path"]]
    before, refs_before = copy.deepcopy(source_reg), copy.deepcopy(refs)
    result, observed, compare = {}, [], object()
    expected_return = tuple(object() for _ in range(6))

    def engine(name, old, migrated, check_reg, output, bound, comparator):
        assert name == "prefix" and old is original and check_reg is prefix_reg
        assert output is result and bound is refs and comparator is compare
        assert migrated is not source_reg
        assert migrated["xml_ref"] == snapshot["migrated_xml_ref"]
        assert migrated["control_ref"] is source_reg["control_ref"]
        assert {k: v for k, v in migrated.items() if k != "xml_ref"} == {
            k: v for k, v in source_reg.items() if k != "xml_ref"
        }
        observed.append(migrated)
        return expected_return

    original_frozen = repair.frozen
    monkeypatch.setattr(
        repair,
        "frozen",
        SimpleNamespace(__file__=original_frozen.__file__, require=original_frozen.require, case_engine=engine),
    )
    monkeypatch.setattr(repair, "load_snapshot", lambda: snapshot)
    if invalid:
        with pytest.raises(ValueError):
            repair.case_engine("prefix", original, source_reg, prefix_reg, result, refs, compare)
        assert observed == []
    else:
        assert repair.case_engine("prefix", original, source_reg, prefix_reg, result, refs, compare) is expected_return
        proof = result["asset_loading_proof"]
        for name in ("original_xml_ref", "migrated_xml_ref", "historical_manifest_ref", "snapshot_ref", "compile_ref"):
            assert proof[name] == snapshot[name]
        assert proof["asset_count"] == 2
        assert proof["complete_model_preserved"] is True and proof["model_body_or_geom_pruning"] is False
        assert proof["main_ast_proof"]["inverse_ast_equal"] is True
    assert source_reg == before and refs == refs_before


def synthetic_main(repair, monkeypatch, tmp_path, case, *, query_delta=0, wrong_prefix=False):
    """Run the actual rewritten main around two fake, fully charged responses."""
    main = repair.build_main()
    namespace = main.__globals__
    rows, queries, published, admissions, finishes, audits = [], [], [], [], [], []
    warm = np.zeros((12, 6))
    commands = [warm, warm + 1e-4]
    errors = np.tile([0.02, 0, 0, 0, 0, 0], (12, 1))
    metrics = dict(
        tcp_errors=errors,
        normalized_squared_errors=np.tile([4.0, 0.0], (12, 1)),
        safety_min=0.1,
        safety_slack=np.array([0.1]),
        joint_feasible=True,
        prefix_length=0,
        feasible=False,
        maximum_normalized_squared_error=4.0,
        maximum_pose_cap_violation=1.0,
        pose_loss=48.0,
    )
    trace = [
        dict(commands=command.copy(), **{key: metrics[key] for key in repair.frozen.TRACE_FIELDS - {"commands"}})
        for command in commands
    ]
    full = dict(commands=warm.copy(), rollout={"index": 0}, audit={"synthetic": True}, **metrics)
    lower, upper = np.full((12, 6), -1.0), np.full((12, 6), 1.0)
    answer = dict(
        initial=full,
        selected=full,
        closest_safety_attempt=full,
        supplied_warm=warm.copy(),
        used_start=warm.copy(),
        start_roundoff_clip=warm.copy(),
        horizon=12,
        pose_caps_horizon=12,
        all_pose_caps_enforced=True,
        max_evaluations=4000,
        max_iterations=60,
        max_stages=60,
        finite_difference_step=1e-4,
        optimization_safety_buffer=1e-8,
        terminal_normals=np.eye(7)[6:],
        past_inner_error_rad=np.zeros((25, 7)),
        lower_command=lower,
        upper_command=upper,
        evaluated_candidates=trace,
        response_evaluations=2,
        total_iterations=1,
        accepted_anchors=[dict(response_index=0, commands=warm.copy(), audit=full["audit"], **metrics)],
    )
    original = dict(
        previous_command=np.zeros(7),
        terminal_normals=answer["terminal_normals"],
        past_inner_error_rad=answer["past_inner_error_rad"],
        answer=dict(lower_command=lower, upper_command=upper),
    )
    current = dict(mass=np.eye(7))

    def ref(path):
        return dict(path=str(path), sha256="synthetic")

    plan, control = tmp_path / "plan.md", ref(tmp_path / "control.json")
    original_reg = dict(control_ref=control)
    registration_ref, launch_ref = ref(tmp_path / "producer_registration.json"), ref(tmp_path / "launch.json")
    producer_ref = ref(tmp_path / f"outputs/tracking_v5_prefix_trust_{case}_20261028/verified.json")
    producer = dict(
        case_name=case,
        registration_ref=registration_ref,
        warm=warm.copy(),
        answer=answer,
        prediction_calls=2,
        response_evaluations=2,
        total_iterations=1,
        numerical_inner_steps=600,
        query_count=625,
        control_qualified=False,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        current_quantities=current,
        source_refs_unchanged=[],
    )
    owner = dict(result_ref=producer_ref, owned_command=dict(launch=launch_ref))
    prefix_ref, prefix_reg_ref, guard_ref = (
        ref(tmp_path / "prefix_check.json"),
        ref(tmp_path / "prefix_registration.json"),
        ref(tmp_path / "guard.json"),
    )
    native = tmp_path / "python"
    archive = {
        prefix_ref["path"]: dict(registration_ref=prefix_reg_ref),
        prefix_reg_ref["path"]: {},
        launch_ref["path"]: dict(
            command=[str(native), str(tmp_path / "scripts/probe_tracking_v5_prefix_trust_control.py"), "--case", case]
        ),
        registration_ref["path"]: dict(
            case_name=case,
            plan_ref=ref(plan),
            seed=42,
            backtracking_ref=ref(tmp_path / "outputs/tracking_v5_saved_direction_backtracking_20261028/verified.json"),
            source_refs=[],
            input_refs=[],
        ),
        guard_ref["path"]: dict(
            cv2_guard_registration=dict(exact_probe_ref=ref(tmp_path / "probe.json"), runtime_source_refs=[])
        ),
        control["path"]: dict(contract=dict(input_min=[-1] * 6, input_max=[1] * 6)),
    }
    refs = {}

    def remember(item, bound):
        bound[item["path"]] = item
        return item

    def admit():
        admissions.append(True)
        return (
            original,
            original_reg,
            {"cases": {"prefix": {"entries": [None] * 55 + [dict(request=dict(commands=warm.copy()))]}}},
            refs,
        )

    def forbidden_original_admission():
        pytest.fail("The frozen admission call must be replaced")

    metadata = SimpleNamespace(
        CPU_ENVIRONMENT=dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
        NATIVE_PYTHON=native,
        PLAN=plan,
        PREFIX_CHECK_REF=prefix_ref,
        SOURCES=[],
        closed_owner=lambda *_: (owner, producer),
        finish=lambda bound: finishes.append(copy.deepcopy(bound)),
        load_prerequisites=forbidden_original_admission,
        read=lambda item: archive[item["path"]],
        reference=ref,
        remember=remember,
    )
    monkeypatch.setitem(sys.modules, "s2a.prefix_trust_control_evidence", metadata)
    for key, value in metadata.CPU_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM"):
        monkeypatch.delenv(key, raising=False)

    def engine(name, old, reg, prefix_reg, result, bound, compare):
        assert name == "prefix" and old is original and reg is original_reg and bound is refs
        queries.append("current")
        result["mj_forward_queries"] += 1

        def reconstruct(command):
            index = len(rows)
            np.testing.assert_array_equal(command, commands[index])
            rows.append(command.copy())
            result["numerical_inner_steps"] += 300
            result["mj_forward_queries"] += 312 + query_delta
            return dict(index=index), None

        def classify(command, rollout):
            return dict(metrics, prefix_length=1 if wrong_prefix else 0)

        def compare_rollout(actual, saved):
            assert actual == saved

        def check_audit(saved, actual):
            assert saved == full["audit"] and actual["joint_feasible"] is True
            audits.append(True)

        return (
            reconstruct,
            compare_rollout,
            classify,
            check_audit,
            current,
            SimpleNamespace(nopython_signatures=["synthetic-only"]),
        )

    def ledger(actual, classifications, *, compare):
        assert actual is answer and len(classifications) == len(rows) == 2
        return dict(witness_indices=[])

    def publish(path, payload):
        published.append((path, copy.deepcopy(payload)))
        return ref(path)

    namespace.update(
        REPO=tmp_path,
        CONTROL_REF=guard_ref,
        strict_native_environment=lambda: None,
        restore_cv2_environment=lambda *_: {"synthetic": True},
        load_asset_repair_prerequisites=admit,
        native_math_proof=lambda: dict(source_ref=ref(tmp_path / "frozen_math.py")),
        case_engine=engine,
        audit_prefix_search_ledger=ledger,
        audit_trust_search_ledger=ledger,
        publish=publish,
    )
    return SimpleNamespace(
        main=main,
        producer=producer,
        rows=rows,
        queries=queries,
        published=published,
        admissions=admissions,
        finishes=finishes,
        audits=audits,
    )


@pytest.mark.parametrize("case", ["baseline", "candidate"])
def test_real_main_replays_every_synthetic_row_with_original_billing(repair, monkeypatch, tmp_path, case):
    run = synthetic_main(repair, monkeypatch, tmp_path, case)
    run.main(case)
    assert len(run.rows) == 2 and run.queries == ["current"] and run.admissions == [True]
    assert len(run.published) == 2
    path, result = run.published[-1]
    assert path == tmp_path / f"outputs/tracking_v5_prefix_trust_{case}_check_v2_20261029/verified.json"
    registration = run.published[0][1]
    assert registration["producer_ref"]["path"].endswith(f"prefix_trust_{case}_20261028/verified.json")
    assert registration["expected_responses"] == result["verified_responses"] == result["response_evaluations"] == 2
    assert result["numerical_inner_steps"] == 600 and result["query_count"] == result["mj_forward_queries"] == 625
    assert result["complete"] is result["passed"] is True
    assert result["full_candidates_checked"] == ["closest_safety_attempt", "initial", "selected"]
    assert result["accepted_anchor_audits_checked"] == (1 if case == "candidate" else 0)
    assert len(run.audits) == (4 if case == "candidate" else 3)
    assert result["boundary_classification_disagreements"] == []
    assert result["control_qualified"] is result["witness_found"] is False
    assert all(result[key] == 0 for key in ("physics_steps", "world_model_calls", "fresh_cases"))


def test_dry_main_checks_admission_without_engine_output_or_responses(repair, monkeypatch, tmp_path, capsys):
    run = synthetic_main(repair, monkeypatch, tmp_path, "candidate")
    run.main("candidate", dry=True)
    output = json.loads(capsys.readouterr().out)
    assert run.admissions == [True] and run.rows == run.queries == run.published == []
    assert output["new_responses"] == output["new_queries"] == 0
    assert output["dry"] is output["ready"] is True


def test_original_producer_budget_gate_is_not_bypassed(repair, monkeypatch, tmp_path):
    run = synthetic_main(repair, monkeypatch, tmp_path, "candidate")
    run.producer["query_count"] += 1
    with pytest.raises(ValueError, match="charged exactly once"):
        run.main("candidate")
    assert run.rows == run.queries == run.published == []


@pytest.mark.parametrize("fault", ["query_count", "strict_prefix"])
def test_failed_independent_verification_keeps_actual_partial_accounting(repair, monkeypatch, tmp_path, fault):
    run = synthetic_main(
        repair,
        monkeypatch,
        tmp_path,
        "candidate",
        query_delta=int(fault == "query_count"),
        wrong_prefix=fault == "strict_prefix",
    )
    with pytest.raises(ValueError):
        run.main("candidate")
    assert len(run.rows) == 1
    result = run.published[-1][1]
    assert result["complete"] is result["passed"] is False
    assert result["response_evaluations"] == 1 and result["numerical_inner_steps"] == 300
    assert result["query_count"] == (314 if fault == "query_count" else 313)
    assert len(result["boundary_classification_disagreements"]) == int(fault == "strict_prefix")
    assert result["verified_responses"] == int(fault == "query_count")
    assert result["control_qualified"] is False
