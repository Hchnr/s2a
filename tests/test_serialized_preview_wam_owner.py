"""Owner command, resource boundary, and inherited process identity behavior."""

import ast
import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def owner():
    return importlib.import_module("run_serialized_preview_wam_development")


def test_dry_owner_uses_new_service_and_fixed_uuid_without_creating_output(tmp_path, monkeypatch, owner):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    p = dict(root=str(tmp_path), model_config=dict(model_python="/synthetic/python"))
    monkeypatch.setattr(
        owner, "admit_development", lambda *a, **k: (p, {}, {}, dict(development_preconditions_passed=True), {})
    )
    ref = dict(path="/synthetic/ref.json", sha256="a" * 64)
    result = owner.execute(ref, ref, ref, ref, dry_run=True)
    assert result["service"][1] == "scripts/serve_serialized_preview_wam_development.py"
    assert result["service"][result["service"].index("--gpu-uuid") + 1] == owner.GPU
    assert result["service"][result["service"].index("--port") + 1] == "31365"
    assert sum(c["expected_requests"] for c in result["case_order"]) == 900
    assert result["model_initializations"] == result["new_physics_steps"] == 0
    assert not (tmp_path / "development").exists()


def test_model_and_cpu_environment_use_only_registered_resources(tmp_path, owner):
    p = dict(root=str(tmp_path), model_config=dict(cosmos_source_root=str(tmp_path)))
    model = owner.training_environment(p, gpu=True)
    cpu = owner.cpu_environment(p)
    assert model["CUDA_VISIBLE_DEVICES"] == "GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f"
    assert model["LD_LIBRARY_PATH"] == cpu["LD_LIBRARY_PATH"] == cpu["CUDA_VISIBLE_DEVICES"] == ""
    assert model["MASTER_PORT"] == "31366" and owner.PORT == 31365


def test_rpc_readiness_and_environment_logic_match_proven_owner():
    functions = []
    for name in ("run_wam_terminal_development.py", "run_serialized_preview_wam_development.py"):
        tree = ast.parse((ROOT / "scripts" / name).read_text())
        functions.append(
            {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)}
        )
    for name in ("training_environment", "cpu_environment", "rpc", "ready", "_flags"):
        assert functions[0][name] == functions[1][name]


def test_native_interpreter_is_used_for_independent_physical_replay():
    tree = ast.parse((ROOT / "scripts/run_serialized_preview_wam_development.py").read_text())
    commands = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.List)
        and len(n.elts) >= 2
        and isinstance(n.elts[1], ast.Constant)
        and n.elts[1].value == "scripts/verify_serialized_preview_wam_development.py"
    ]
    assert len(commands) == 1
    assert "outputs/libero_env311/bin/python" in ast.unparse(commands[0].elts[0])
