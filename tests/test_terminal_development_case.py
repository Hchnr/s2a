import copy
import importlib
import io
import json

import pytest
import requests


@pytest.fixture
def worker(monkeypatch):
    from s2a.artifacts import REPO

    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    return importlib.import_module("run_wam_terminal_development_case")


def request(directory):
    return dict(
        name="current",
        model_mode="wam",
        prompt="task",
        vision_path=str(directory / "input_0000.png"),
        domain_name="domain",
        image_size=960,
        action_chunk_size=40,
        fps=20,
        num_frames=41,
        seed=42,
        num_steps=35,
        guidance=1.0,
        shift=5.0,
        extra=dict(
            action128=dict(
                action_domain="domain",
                action_dim_mask=[True] * 10 + [False] * 118,
                action_fps=20,
                task_title="task",
                selected_roles=[],
                camera_layout=[],
                return_video=False,
            )
        ),
    )


def test_transport_cap_keeps_original_json_and_restores_session(tmp_path, monkeypatch, worker):
    seen = []

    class Session:
        def request(self, method, url, **kwargs):
            seen.append((method, url, kwargs))
            return type("Response", (), dict(status_code=200))()

    monkeypatch.setattr(requests, "Session", Session)
    log = io.StringIO()
    body = request(tmp_path)
    saved = copy.deepcopy(body)
    with worker.bounded_http(log, worker.case_order()[0], tmp_path):
        requests.Session().request("POST", "http://127.0.0.1:31363/generate", json=body, timeout=900)
    assert requests.Session is Session and seen[0][2]["timeout"] == 300
    assert seen[0][2]["json"] is body and body == saved
    record = json.loads(log.getvalue())
    assert record["timeout_s"] == 300 and record["completed"] is True


@pytest.mark.parametrize("fault", ["host", "future", "seed", "other_image", "timeout"])
def test_transport_rejects_causal_or_resource_escape(tmp_path, monkeypatch, fault, worker):
    class Session:
        def request(self, *a, **k):
            raise AssertionError("invalid transport must fail before HTTP dispatch")

    monkeypatch.setattr(requests, "Session", Session)
    body = request(tmp_path)
    url = "http://127.0.0.1:31363/generate"
    timeout = 900
    if fault == "host":
        url = "https://external.example/generate"
    elif fault == "future":
        body["extra"]["future_state"] = [1]
    elif fault == "seed":
        body["seed"] = 43
    elif fault == "other_image":
        body["vision_path"] = str(tmp_path.parent / "future.png")
    else:
        timeout = None
    with pytest.raises(ValueError), worker.bounded_http(io.StringIO(), worker.case_order()[0], tmp_path):
        requests.Session().request("POST", url, json=body, timeout=timeout)
    assert requests.Session is Session


def test_dry_client_never_creates_physics_root_or_preloads_environment(tmp_path, monkeypatch, worker):
    from s2a.artifacts import REPO

    monkeypatch.chdir(REPO)
    for name, value in (("CUDA_VISIBLE_DEVICES", ""), ("LD_LIBRARY_PATH", ""), ("PYTHONDONTWRITEBYTECODE", "1")):
        monkeypatch.setenv(name, value)
    case = worker.case_order()[0]
    root = tmp_path / "development/cases" / case["case_id"]
    config = dict(output_dir=str(root))
    prepared = dict(root=str(tmp_path))
    monkeypatch.setattr(worker, "admit_development", lambda *a, **k: (prepared, {}, {}, {}, {}))
    monkeypatch.setattr(worker, "read", lambda ref: config)
    monkeypatch.setattr(worker, "make_case_config", lambda *a: config)
    monkeypatch.setattr(worker, "restore_cv2_environment", lambda *a: pytest.fail("dry preloaded native libraries"))
    result = worker.execute({}, {}, {}, {}, {}, case_index=0, dry=True)
    assert (
        result["passed"]
        and result["dry"]
        and result["new_model_initializations"] == result["physics_steps"] == result["requests"] == 0
    )
    assert not root.exists()


def test_complete_client_body_and_transport_preserve_original_executor():
    import ast

    from s2a.artifacts import REPO

    def functions(path):
        return {n.name: n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}

    old = functions(REPO / "scripts/execute_wam_clean_training_case.py")
    current = functions(REPO / "scripts/run_wam_terminal_development_case.py")
    assert ast.dump(old["bounded_http"]) == ast.dump(current["bounded_http"])
    expected = old["execute"]
    expected.args.kwonlyargs.append(ast.arg(arg="dry"))
    expected.args.kw_defaults.append(ast.Constant(False))

    class Rename(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                node.value = (
                    node.value.replace(
                        "scripts/run_wam_clean_training_development.py", "scripts/run_wam_terminal_development.py"
                    )
                    .replace("s2a.clean-training-development-case.v1", "s2a.terminal-development-case.v1")
                    .replace("W2 is an adapted model system", "terminal058 is an adapted model system")
                )
            return node

    expected = Rename().visit(expected)
    seam = ast.parse("""
if dry:
    return dict(passed=True,dry=True,case=case,new_model_initializations=0,physics_steps=0,requests=0)
""").body
    index = next(
        i
        for i, n in enumerate(expected.body)
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute)
        and n.value.func.attr == "mkdir"
    )
    expected.body[index:index] = seam
    assert ast.dump(expected) == ast.dump(current["execute"])
