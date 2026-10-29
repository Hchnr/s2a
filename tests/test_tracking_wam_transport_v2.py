"""Current-only transport and service rejection checks for the tracking entry."""

import copy
import importlib
import io
import json

import pytest
import requests
import test_terminal_development_service as service_checks
from test_serialized_preview_wam_collection import request


@pytest.fixture
def worker(monkeypatch):
    from s2a.artifacts import REPO

    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    return importlib.import_module("collect_tracking_preview_wam_development_v2")


def test_transport_preserves_current_request_and_caps_timeout(tmp_path, monkeypatch, worker):
    seen = []

    class Session:
        def request(self, method, url, **kwargs):
            seen.append((method, url, kwargs))
            return type("Response", (), dict(status_code=200))()

    monkeypatch.setattr(requests, "Session", Session)
    body = request(tmp_path)
    expected = copy.deepcopy(body)
    log = io.StringIO()
    with worker.bounded_http(log, worker.case_order()[0], tmp_path):
        requests.Session().request("POST", f"http://127.0.0.1:{worker.PORT}/generate", json=body, timeout=900)
    assert seen[0][2]["timeout"] == 300 and body == expected and requests.Session is Session
    assert json.loads(log.getvalue())["completed"] is True


@pytest.mark.parametrize("fault", ["future", "seed", "image", "host", "timeout"])
def test_invalid_current_request_never_dispatches(tmp_path, monkeypatch, worker, fault):
    class Session:
        def request(self, *args, **kwargs):
            pytest.fail("Rejected request reached HTTP")

    monkeypatch.setattr(requests, "Session", Session)
    body = request(tmp_path)
    url = f"http://127.0.0.1:{worker.PORT}/generate"
    timeout = 300
    if fault == "future":
        body["extra"]["next_state"] = [1]
    elif fault == "seed":
        body["seed"] = 43
    elif fault == "image":
        body["vision_path"] = str(tmp_path.parent / "future.png")
    elif fault == "host":
        url = "https://external.example/generate"
    else:
        timeout = None
    with pytest.raises(ValueError), worker.bounded_http(io.StringIO(), worker.case_order()[0], tmp_path):
        requests.Session().request("POST", url, json=body, timeout=timeout)
    assert requests.Session is Session


@pytest.mark.parametrize("route,body", [("/generate", {}), ("/bind_case", {}), ("/finish", {})])
def test_actual_tracking_service_http_rejects_without_model(tmp_path, monkeypatch, route, body):
    from s2a.artifacts import REPO

    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    service = importlib.import_module("serve_tracking_preview_wam_development_v2")
    service_checks.test_real_http_rejection_stops_service_without_generating(tmp_path, service, route, body)


def test_old_registration_cannot_start_tracking_model(tmp_path, monkeypatch):
    from s2a.artifacts import REPO
    from s2a.clean_training_runtime import reference

    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    service = importlib.import_module("serve_tracking_preview_wam_development_v2")
    path = tmp_path / "old_prepared.json"
    path.write_text(json.dumps(dict(schema_version="s2a.serialized-preview-wam-development-preparation.v1")))
    with pytest.raises(ValueError, match="Fixed preview WAM preparation"):
        service.run(
            reference(path),
            reference(path),
            state_owner_ref={},
            state_verification_ref={},
            host="127.0.0.1",
            port=service.PORT,
            gpu_uuid=service.GPU,
        )
    assert not (tmp_path / "development").exists()
