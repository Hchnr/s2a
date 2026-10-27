import ast
import importlib

import numpy as np
import pytest

from s2a.artifacts import REPO


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    return tuple(
        importlib.import_module(n)
        for n in (
            "run_wam_terminal_development",
            "verify_wam_terminal_development",
            "verify_wam_terminal_development_points",
        )
    )


def test_owner_dry_fixes_case_matrix_and_uuid_without_creating_output(tmp_path, monkeypatch, modules):
    owner, _, _ = modules
    for name in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"):
        monkeypatch.setenv(name, "")
    monkeypatch.chdir(REPO)
    prepared = dict(
        root=str(tmp_path), model_config=dict(model_python="/actual/model/python", cosmos_source_root=str(tmp_path))
    )
    admission = dict(development_preconditions_passed=True)
    monkeypatch.setattr(owner, "admit_development", lambda *a, **k: (prepared, {}, {}, admission, {}))
    ref = dict(path="/explicit.json", sha256="a" * 64)
    r = owner.execute(ref, ref, ref, ref, dry_run=True)
    assert len(r["case_order"]) == 12 and sum(c["expected_requests"] for c in r["case_order"]) == 900
    assert r["service"][1] == "scripts/serve_wam_terminal_development.py"
    assert r["service"][r["service"].index("--gpu-uuid") + 1] == owner.GPU
    assert r["model_initializations"] == r["new_physics_steps"] == 0
    assert not (tmp_path / "development").exists()


def test_model_coordination_and_http_ports_are_distinct(modules, tmp_path):
    owner, _, _ = modules
    p = dict(root=str(tmp_path), model_config=dict(cosmos_source_root=str(tmp_path)))
    assert owner.PORT != owner.MASTER_PORT
    env = owner.training_environment(p, gpu=True)
    assert env["CUDA_VISIBLE_DEVICES"] == owner.GPU and env["LD_LIBRARY_PATH"] == ""
    assert env["MASTER_PORT"] == str(owner.MASTER_PORT)
    assert owner.cpu_environment(p)["CUDA_VISIBLE_DEVICES"] == ""


def test_independent_tail40_and_case_gate_are_unchanged(modules):
    _, audit, _ = modules
    assert not audit.outcome([False] * 560 + [True] * 39 + [False])["stable"]
    cases = [dict(case=c, outcome=audit.outcome([False] * 560 + [True] * 40)) for c in audit.order()]
    assert audit.score(cases)["development_stability_gate_passed"]
    for row in cases:
        if row["case"]["variant"] == "terminal058" and row["case"]["task_index"] == 2:
            row["outcome"] = audit.outcome([False] * 600)
    assert not audit.score(cases)["development_stability_gate_passed"]
    assert audit.score(cases)["fresh_performance_authorized"] is False


def test_numerical_head_accepts_only_fixed058_ridge(modules):
    _, _, points = modules
    parameters = dict(
        schema_version="s2a.clean-residual-readout.v1",
        coefficients=np.ones((1291, 10)).tolist(),
        center=np.zeros(1290).tolist(),
        scale=2.0,
        ridge=0.1 / 1290,
    )
    assert points.head_parameters(parameters)[2] == 2.0
    parameters["ridge"] = 0.1
    with pytest.raises(ValueError):
        points.head_parameters(parameters)


def test_complete_point_checker_changes_only_head_selection(modules):
    _, _, _ = modules
    original = (REPO / "scripts/verify_wam_clean_training_capture.py").read_text()
    expected = original[original.index("def audit_points(") : original.index("\ndef paired_initial(")]
    expected = expected.replace("head=head,", "head=step_head(head, index),")
    actual = ast.parse((REPO / "scripts/verify_wam_terminal_development_points.py").read_text())
    actual = next(n for n in actual.body if isinstance(n, ast.FunctionDef) and n.name == "audit_points")
    assert ast.dump(ast.parse(expected).body[0]) == ast.dump(actual)


def test_physical_outcome_score_and_receipt_rules_preserve_original():
    def functions(path):
        return {n.name: n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}

    old = functions(REPO / "scripts/verify_wam_clean_training_development.py")
    new = functions(REPO / "scripts/verify_wam_terminal_development.py")

    class Rename(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value == "w2":
                node.value = "terminal058"
            return node

    for name in (
        "order",
        "outcome",
        "score",
        "normal_owned",
        "verify_service_request",
        "label_current_mechanical_evidence",
    ):
        assert ast.dump(Rename().visit(old[name])) == ast.dump(new[name])
