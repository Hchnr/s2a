"""Exercise the new service with the established CPU HTTP and cleanup fixtures."""

import ast
import importlib
import json

import pytest
import test_terminal_development_service as prior

from s2a.artifacts import REPO
from s2a.clean_training_runtime import reference


@pytest.fixture
def server():
    return importlib.import_module("serve_serialized_preview_wam_development")


def test_fixed900_policy_on_new_service(tmp_path, server):
    prior.test_complete_fixed900_policy_has_no_variant_selection_or_early_finish(tmp_path, server)


@pytest.mark.parametrize("route,body", [("/generate", {}), ("/bind_case", {}), ("/finish", {})])
def test_actual_http_rejects_invalid_input_without_model(tmp_path, server, route, body):
    prior.test_real_http_rejection_stops_service_without_generating(tmp_path, server, route, body)


def test_actual_cleanup_failure_is_preserved(tmp_path, monkeypatch, server):
    prior.test_actual_run_preserves_cleanup_false_and_exits_nonzero_in_owned_cpu_child(tmp_path, monkeypatch, server)


def test_actual_dry_keeps_model_unimported(tmp_path, monkeypatch, server):
    prior.test_actual_dry_uses_real_identity_without_importing_model(tmp_path, monkeypatch, server)


def test_old_execution_registration_cannot_admit_new_service(tmp_path, server):
    path = tmp_path / "old.json"
    path.write_text(json.dumps(dict(schema_version="s2a.terminal-development-preparation.v1")))
    with pytest.raises(ValueError, match="Fixed preview WAM preparation"):
        server.run(
            reference(path),
            reference(path),
            state_owner_ref={},
            state_verification_ref={},
            host="127.0.0.1",
            port=31365,
            gpu_uuid=server.GPU,
        )
    assert not (tmp_path / "development").exists()


def test_service_model_and_http_functions_preserve_frozen_behavior():
    class Normalize(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                return ast.copy_location(
                    ast.Constant(
                        node.value.replace(
                            "serve_serialized_preview_wam_development.py", "serve_wam_terminal_development.py"
                        )
                    ),
                    node,
                )
            return node

    bodies = []
    for name in ("serve_wam_terminal_development.py", "serve_serialized_preview_wam_development.py"):
        tree = ast.parse((REPO / "scripts" / name).read_text())
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        bodies.append(ast.dump(Normalize().visit(ast.Module(body=nodes, type_ignores=[])), include_attributes=False))
    assert bodies[0] == bodies[1]
