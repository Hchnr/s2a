"""Private HTTP/current-input/budget policy tests without loading a model."""

import copy
import importlib
import json
import multiprocessing
import sys
import threading
from contextlib import nullcontext
from http.server import HTTPServer
from types import ModuleType, SimpleNamespace

import pytest
import requests

from s2a.artifacts import REPO
from s2a.wam_readout import LIBERO_DOMAIN


@pytest.fixture
def server(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    return importlib.import_module("serve_wam_clean_training")


def prepared(root):
    templates = []
    for task in range(3):
        request = dict(
            name="registered",
            model_mode="wam",
            prompt=f"task {task}",
            vision_path="/old/current.png",
            domain_name="franka_panda__parallel_gripper",
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
                    action_domain=LIBERO_DOMAIN,
                    action_dim_mask=[True] * 10 + [False] * 118,
                    action_fps=20,
                    task_title=f"task {task}",
                    selected_roles=["auxiliary_0", "left_wrist"],
                    camera_layout=[["auxiliary_0", "top", 640, 480], ["left_wrist", "bottom", 640, 480]],
                    return_video=False,
                )
            ),
        )
        templates.append(dict(request=request))
    return dict(
        root=str(root),
        zero_identity_cases=templates,
        validation_anchors=[dict(task_index=i, task=f"task {i}") for i in range(3)],
    )


def current_request(server, policy, *, pixels=b"identical current canvas"):
    case = policy.current
    directory = server.case_directory(policy.root, case)
    directory.mkdir(parents=True, exist_ok=True)
    image = directory / f"input_{policy.case_requests:04d}.png"
    image.write_bytes(pixels)
    return dict(
        copy.deepcopy(policy.templates[policy.tasks[case["task_index"]]]),
        name=f"s2a_{directory.name}_{policy.case_requests}",
        seed=case["sampling_seed"],
        vision_path=str(image),
    )


def complete(policy, request):
    pending = policy.reserve(request)
    policy.complete(dict(pending, complete=True, initial_ref=dict(path="/synthetic", sha256="a" * 64)))


def test_complete_fixed900_policy_has_no_variant_selection_or_early_finish(tmp_path, server):
    policy = server.DevelopmentPolicy(prepared(tmp_path))
    assert len(server.case_order()) == 12
    with pytest.raises(ValueError, match="complete"):
        policy.finish(dict(completed_cases=12, completed_requests=900))
    for case in server.case_order():
        policy.bind(case)
        for _ in range(75):
            complete(policy, current_request(server, policy))
        with pytest.raises(ValueError, match="budget"):
            policy.reserve(current_request(server, policy))
    assert [r["global_request_index"] for r in policy.requests] == list(range(900))
    assert [r["variant"] for r in policy.requests] == [c["variant"] for c in server.case_order() for _ in range(75)]
    policy.finish(dict(completed_cases=12, completed_requests=900))
    assert policy.finish_requested is policy.stopped is True and not policy.poisoned


@pytest.mark.parametrize("fault", ["skip_case", "change_variant", "early_next_case", "duplicate_bind", "wrong_count"])
def test_binding_cannot_skip_mix_or_truncate_cases(tmp_path, server, fault):
    policy = server.DevelopmentPolicy(prepared(tmp_path))
    cases = server.case_order()
    if fault in ("early_next_case", "duplicate_bind"):
        policy.bind(cases[0])
        candidate = cases[1] if fault == "early_next_case" else cases[0]
    else:
        candidate = copy.deepcopy(cases[1] if fault == "skip_case" else cases[0])
        if fault == "change_variant":
            candidate["variant"] = "w2"
        elif fault == "wrong_count":
            candidate["expected_requests"] = 74
    with pytest.raises(ValueError):
        policy.bind(candidate)


@pytest.mark.parametrize(
    "fault", ["future", "nested_future", "task", "seed", "name", "outside_image", "long_sampler", "duplicate_pending"]
)
def test_online_request_is_exact_current_image_and_task(tmp_path, server, fault):
    policy = server.DevelopmentPolicy(prepared(tmp_path))
    policy.bind(server.case_order()[0])
    request = current_request(server, policy)
    if fault == "future":
        request["future_state"] = [1]
    elif fault == "nested_future":
        request["extra"]["action128"]["future_action"] = [1]
    elif fault == "task":
        request["prompt"] = "another task"
    elif fault == "seed":
        request["seed"] = 43
    elif fault == "name":
        request["name"] = "wrong request number"
    elif fault == "outside_image":
        request["vision_path"] = str(tmp_path / "other.png")
    elif fault == "long_sampler":
        request["num_steps"] = 70
    else:
        policy.reserve(request)
    with pytest.raises(ValueError):
        policy.reserve(request)


def test_first_raw_w2_current_pixels_must_pair_even_with_valid_request_paths(tmp_path, server):
    policy = server.DevelopmentPolicy(prepared(tmp_path))
    cases = server.case_order()
    policy.bind(cases[0])
    for _ in range(75):
        complete(policy, current_request(server, policy))
    policy.bind(cases[1])
    with pytest.raises(ValueError, match="PNG"):
        policy.reserve(current_request(server, policy, pixels=b"changed first current image"))


@pytest.mark.parametrize(
    "route,body", [("/finish", dict(completed_cases=12, completed_requests=900)), ("/generate", dict(future_state=[1]))]
)
def test_real_http_rejection_stops_service_without_generating(tmp_path, server, route, body):
    policy = server.DevelopmentPolicy(prepared(tmp_path))
    aborts = []
    calls = []

    def generate(request):
        calls.append("entered")
        policy.reserve(request)
        raise AssertionError("invalid request must not reach a model")

    handler = server.handler_class(
        policy,
        generate,
        lambda: dict(checkpoint_path="/actual"),
        lambda: dict(ready=not policy.poisoned),
        aborts.append,
    )
    http = HTTPServer(("127.0.0.1", 0), handler)
    http.timeout = 0.1

    def serve():
        while not policy.stopped:
            http.handle_request()

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        with requests.Session() as session:
            session.trust_env = False
            url = f"http://127.0.0.1:{http.server_port}"
            assert session.get(url + "/healthz", timeout=2).json() == dict(status="ready")
            assert session.get(url + "/info", timeout=2).json()["checkpoint_path"] == "/actual"
            response = session.post(url + route, json=body, timeout=2)
            assert response.status_code == 422 and response.json()["status"] == "error"
        thread.join(timeout=2)
        assert not thread.is_alive() and policy.poisoned and len(aborts) == 1 and not policy.requests
        assert calls == (["entered"] if route == "/generate" else [])
    finally:
        policy.stopped = True
        thread.join(timeout=2)
        http.server_close()


def test_abort_is_explicit_and_preserves_case_identity(tmp_path, server):
    policy = server.DevelopmentPolicy(prepared(tmp_path))
    policy.bind(server.case_order()[0])
    with pytest.raises(ValueError):
        policy.abort(dict(case_id="other", reason="worker failed"))
    body = dict(case_id=policy.current["case_id"], reason="owned CPU worker exited7")
    assert policy.abort(body) == body and policy.poisoned and policy.stopped and not policy.finish_requested


def test_metadata_only_cli_cannot_initialize_model(tmp_path, monkeypatch, server):
    from s2a.clean_training_runtime import reference

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    p = tmp_path / "p.json"
    p.write_text(json.dumps(dict(schema_version="s2a.clean-training-preparation.v1")))
    pr = reference(p)
    r = tmp_path / "r.json"
    r.write_text(
        json.dumps(
            dict(
                schema_version="s2a.clean-training-execution-registration.v1",
                prepared_ref=pr,
                complete=True,
                passed=True,
                readiness=dict(ready_for_execution=False),
                automatic_retry=False,
            )
        )
    )
    with pytest.raises(ValueError, match="execution registration"):
        server.run(
            pr,
            reference(r),
            state_owner_ref={},
            state_verification_ref={},
            host="127.0.0.1",
            port=31339,
            gpu_uuid=server.GPU,
        )
    assert not (tmp_path / "development").exists()


def test_actual_run_preserves_cleanup_false_and_exits_nonzero_in_owned_cpu_child(tmp_path, monkeypatch, server):
    """Exercise real service.run/finally with a synthetic engine, not just its AST.

    Admission, engine and completed counters are stubs solely to reach teardown.
    No CUDA API, model forward, environment, HTTP generation or label reader runs.
    The child must actually exit nonzero after publishing the retained False.
    """
    numerics = importlib.import_module("audit_wam_current_numerics")
    features = importlib.import_module("run_wam_readout_features")
    sampling = importlib.import_module("run_wam_readout_sampling")
    runtime = importlib.import_module("s2a.wam_runtime")

    class Model:
        def parameters(self):
            return []

        def denoise(self, *args, **kwargs):
            raise AssertionError("the cleanup test cannot execute a model")

    class Bridge:
        total_head_calls = 31500  # Synthetic completed metadata, no actual calls.

        def __init__(self, *args, **kwargs):
            pass

        def restore(self):
            pass

    class Counter:
        def __init__(self, model, budget):
            self.original = model.denoise
            self.calls = [dict(completed=True)] * budget

    class Initial:
        def __init__(self, model):
            pass

        def remove(self):
            raise OSError("synthetic engine hook teardown failure")

    class HTTP:
        def __init__(self, *args):
            pass

        def handle_request(self):
            raise AssertionError("no HTTP request may be generated in this cleanup test")

        def server_close(self):
            (tmp_path / "http_close_observed.txt").write_text("closed before failed hook teardown")

    original_policy = server.DevelopmentPolicy

    def completed_policy(preparation):
        policy = original_policy(preparation)
        policy.requests = [dict(synthetic_cleanup_fixture=True)] * 900
        policy.case_index, policy.case_requests = 11, 75
        policy.finish_requested = policy.stopped = True
        return policy

    p = prepared(tmp_path)
    p.update(
        model_config=dict(checkpoint=dict(run=str(tmp_path / "checkpoint"), iteration="iter_000008400")),
        prerequisite_refs=dict(bridge_prepared=dict(path="/synthetic-qualified", sha256="a" * 64)),
        noise_roster=dict(training_seeds=[101], evaluation_seeds=[42, 43]),
    )
    reg = dict(development=dict(port=server.PORT, case_order=server.case_order()))
    cosmos = tmp_path / "cosmos"
    sources = {str((cosmos / name).resolve()): "a" * 64 for name in (server.SAMPLER_SOURCE, server.CONVERSION_SOURCE)}
    checkpoint = str(tmp_path / "checkpoint/checkpoints/iter_000008400/model")
    engine = SimpleNamespace(model=Model(), info=dict(checkpoint_path=checkpoint, ema_weights=False), spec={})
    monkeypatch.setattr(server, "admit_execution", lambda *a, **k: (p, reg, sources))
    monkeypatch.setattr(
        server,
        "admit_state",
        lambda *a, **k: (dict(w2_fit_ref={}, w2_verification_ref={}), None, dict(path="/params", sha256="b" * 64)),
    )
    monkeypatch.setattr(server, "read", lambda ref: dict(config=dict(cosmos_source_root=str(cosmos))))
    monkeypatch.setattr(server, "runtime_environment", lambda *a: None)
    monkeypatch.setattr(server, "qualified_grid", lambda *a: ((1.0, 1000),))
    monkeypatch.setattr(server, "DevelopmentPolicy", completed_policy)
    monkeypatch.setattr(server, "parameters_identity", lambda *a: dict(count=1, synthetic=True))
    monkeypatch.setattr(server, "imported_cosmos_sources", lambda *a: {})
    monkeypatch.setattr(server, "normalization_spec", lambda *a: dict(synthetic=True))
    monkeypatch.setattr(server, "ConversionWitness", Bridge)
    monkeypatch.setattr(server, "HTTPServer", HTTP)
    monkeypatch.setattr(server, "check_sources", lambda *a: None)
    monkeypatch.setattr(server, "checked", lambda *a: None)
    monkeypatch.setattr(numerics, "require_eval", lambda *a: None)
    monkeypatch.setattr(features, "BoundaryCounter", Counter)
    monkeypatch.setattr(sampling, "InitialCapture", Initial)
    monkeypatch.setattr(runtime, "load_service", lambda *a: engine)
    monkeypatch.setattr(runtime, "install_active_slot_mask", lambda *a: dict(synthetic=True))
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_properties=lambda _: SimpleNamespace(uuid=server.GPU),
        manual_seed_all=lambda _: None,
    )
    torch.manual_seed = lambda _: None
    torch.inference_mode = nullcontext
    monkeypatch.setitem(sys.modules, "torch", torch)
    for name in ("cosmos_framework", "cosmos_framework.inference", "cosmos_framework.inference.args"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["cosmos_framework.inference.args"].OmniSampleOverrides = object
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    ref = dict(path="/synthetic-input", sha256="a" * 64)
    child = multiprocessing.get_context("fork").Process(
        target=server.run,
        args=(ref, ref),
        kwargs=dict(
            state_owner_ref=ref, state_verification_ref=ref, host="127.0.0.1", port=server.PORT, gpu_uuid=server.GPU
        ),
    )
    child.start()
    child.join(timeout=10)
    if child.is_alive():
        child.terminate()
        child.join(timeout=5)
        pytest.fail("owned synthetic cleanup child did not terminate")
    assert child.exitcode == 1
    result = json.loads((tmp_path / "development/service/result.json").read_text())
    assert result["complete"] is result["passed"] is False
    assert result["cleanup_errors"] == [
        dict(stage="initial", type="OSError", message="synthetic engine hook teardown failure")
    ]
    assert (tmp_path / "http_close_observed.txt").is_file()
