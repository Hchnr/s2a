"""D2 dry path cannot load a model; the actual capture body preserves038 math."""

import ast
import builtins

import pytest

from s2a.artifacts import REPO
from s2a.clean_training_runtime import GPU


def capture_body(path):
    tree = ast.parse(path.read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    start = next(
        i
        for i, n in enumerate(run.body)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "bridge" for t in n.targets)
    )
    return ast.Module(body=run.body[start:], type_ignores=[])


class ExpectedD2Changes(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id == "admit_execution":
            return ast.copy_location(ast.Name(id="admit", ctx=node.ctx), node)
        return node

    def visit_IfExp(self, node):
        if (
            ast.dump(node.test) == ast.dump(ast.parse('round_name == "D0"', mode="eval").body)
            and isinstance(node.body, ast.Constant)
            and node.body.value == "raw"
            and isinstance(node.orelse, ast.Constant)
            and node.orelse.value == "w1"
        ):
            return ast.copy_location(ast.Constant("probe052"), node)
        return self.generic_visit(node)


def test_actual_capture_body_changes_only_experiment_variant_and_admission():
    original = capture_body(REPO / "scripts/collect_wam_clean_training.py")
    actual = capture_body(REPO / "scripts/collect_wam_clean_d2.py")
    assert ast.dump(ExpectedD2Changes().visit(original)) == ast.dump(actual)


def test_actual_worker_dry_does_not_import_torch_or_initialize_model(monkeypatch):
    import collect_wam_clean_d2 as worker

    calls = []

    def admitted(*args, **kwargs):
        calls.append((args, kwargs))
        return {}, {}, {}, object(), {}, {}

    monkeypatch.setattr(worker, "admit", admitted)
    original_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name == "torch" or name.startswith("cosmos_framework"):
            raise AssertionError("dry path attempted model import")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    monkeypatch.setattr(worker.np, "load", lambda *a, **k: pytest.fail("dry decoded an archive"))
    result = worker.run("prepared-ref", "registration-ref", gpu_uuid=GPU, dry=True)
    assert len(calls) == 1
    assert REPO / "scripts/collect_wam_clean_d2.py" in calls[0][1]["required_sources"]
    assert result["requests"] == 300 and result["model_initializations"] == result["future_label_reads"] == 0


def test_missing_execution_admission_stops_before_model_path(monkeypatch):
    import collect_wam_clean_d2 as worker

    def reject(*a, **k):
        raise ValueError("independent execution metadata missing")

    monkeypatch.setattr(worker, "admit", reject)
    monkeypatch.setattr(worker, "runtime_environment", lambda *a: pytest.fail("entered model path"))
    with pytest.raises(ValueError, match="metadata missing"):
        worker.run("prepared-ref", "registration-ref", gpu_uuid=GPU)


def test_dry_cannot_bypass_allowed_gpu_identity(monkeypatch):
    import collect_wam_clean_d2 as worker

    monkeypatch.setattr(worker, "admit", lambda *a, **k: ({}, {}, {}, None, {}, {}))
    with pytest.raises(ValueError, match="allowed UUID"):
        worker.run("prepared-ref", "registration-ref", gpu_uuid="0", dry=True)


def test_d2_audit_requires_registered052_ridge_and_finite_parameters():
    import numpy as np
    from verify_wam_clean_d2 import head_parameters

    params = dict(
        schema_version="s2a.clean-residual-readout.v1",
        coefficients=np.zeros((1291, 10)).tolist(),
        center=[0.0] * 1290,
        scale=1.0,
        ridge=0.1 / 1290,
    )
    assert head_parameters(params)[0].shape == (1291, 10)
    params["ridge"] = 0.1
    with pytest.raises(ValueError, match="scale/ridge"):
        head_parameters(params)
    params["ridge"] = 0.1 / 1290
    params["coefficients"][0][0] = float("nan")
    with pytest.raises(ValueError, match="layout"):
        head_parameters(params)


def test_lifecycle_proof_preserves_all_receipt_checks_for_new_worker():
    def function(path):
        return next(
            n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "lifecycle"
        )

    class ExpectedLifecycleChanges(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value == "scripts/collect_wam_clean_training.py":
                return ast.copy_location(ast.Constant("scripts/collect_wam_clean_d2.py"), node)
            return node

        def visit_Tuple(self, node):
            node.elts = [
                e
                for e in node.elts
                if not (
                    isinstance(e, ast.Tuple)
                    and e.elts
                    and isinstance(e.elts[0], ast.Constant)
                    and e.elts[0].value == "--round"
                )
            ]
            return self.generic_visit(node)

    original = function(REPO / "scripts/verify_wam_clean_training_capture.py")
    actual = function(REPO / "scripts/verify_wam_clean_d2.py")
    assert ast.dump(ExpectedLifecycleChanges().visit(original)) == ast.dump(actual)


def test_owner_command_parses_through_actual_worker_cli(monkeypatch, capsys):
    import sys

    import collect_wam_clean_d2 as worker
    from run_wam_clean_d2 import worker_command

    pref = {"path": str(REPO / "outputs/test/prepared.json"), "sha256": "1" * 64}
    regref = {"path": str(REPO / "outputs/test/registration.json"), "sha256": "2" * 64}
    command = worker_command({"model_config": {"model_python": "python"}}, pref, regref)
    received = []

    def run(prepared, registration, *, gpu_uuid, dry):
        received.append((prepared, registration, gpu_uuid, dry))
        return {"passed": True}

    monkeypatch.setattr(worker, "run", run)
    monkeypatch.setattr(sys, "argv", command[1:] + ["--dry"])
    worker.main()
    assert received == [(pref, regref, GPU, True)]
    assert '"passed": true' in capsys.readouterr().out


def test_independent_metadata_body_proof_rejects_sampler_budget_change(tmp_path, monkeypatch):
    import verify_wam_clean_d2_metadata as metadata

    assert metadata.body_proof()["passed"] is True
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("collect_wam_clean_training.py", "collect_wam_clean_d2.py"):
        source = (REPO / "scripts" / name).read_text()
        if name == "collect_wam_clean_d2.py":
            source = source.replace('counter.scope(entry["name"], 35)', 'counter.scope(entry["name"], 34)')
        (scripts / name).write_text(source)
    monkeypatch.setattr(metadata, "REPO", tmp_path)
    with pytest.raises(ValueError, match="capture body differs"):
        metadata.body_proof()
