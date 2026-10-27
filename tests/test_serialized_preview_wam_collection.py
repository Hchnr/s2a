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
    return importlib.import_module("collect_serialized_preview_wam_development")


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
        requests.Session().request("POST", "http://127.0.0.1:31365/generate", json=body, timeout=900)
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
    url = "http://127.0.0.1:31365/generate"
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


def test_fixed12_configs_preserve_preview_parameters_and_no_expert_input(monkeypatch):
    from pathlib import Path

    from s2a import serialized_preview_wam_control as control

    base = json.loads(Path(control.BASE_REF["path"]).read_text())
    qualified = json.loads((control.REPO / "configs/coupled_preview_control_registered_20261026.json").read_text())
    ctrl_ref = dict(path="/synthetic/control.json", sha256="a" * 64)
    prepared_ref = dict(path="/synthetic/prepared.json", sha256="b" * 64)
    registration_ref = dict(path="/synthetic/registered.json", sha256="c" * 64)
    closure_ref = dict(path="/synthetic/closure.json", sha256="d" * 64)
    prepared = dict(
        root=str(control.ROOT),
        tool_relocation_ref=control.RELOCATION_REF,
        preview_control_closure_ref=closure_ref,
        candidate_params_ref=ctrl_ref,
        model_config=dict(
            cosmos_source_root="/readonly/cosmos",
            checkpoint=dict(run="/readonly/checkpoint", iteration="iter_000008400"),
        ),
    )
    registration = dict(
        prepared_ref=prepared_ref,
        preview_control_closure_ref=closure_ref,
        physical_control_admission=dict(control_config_ref=ctrl_ref),
        service_source_ref=ctrl_ref,
    )
    monkeypatch.setattr(control, "read", lambda r: copy.deepcopy(base if r == control.BASE_REF else qualified))
    monkeypatch.setattr(control, "safe_template", lambda r: dict(path="/synthetic/template.json"))
    original = copy.deepcopy(prepared)
    for case in control.case_order():
        config = control.make_case_config(
            prepared, registration, prepared_ref=prepared_ref, registration_ref=registration_ref, case=case
        )
        assert all(config[k] == qualified[k] for k in control.CONTROL_FIELDS)
        assert config["records"] == [case["record_name"]]
        assert config["sampling_seeds"] == [case["sampling_seed"]]
        assert config["online_max_steps"] == 600 and config["max_observation_interval_steps"] == 8
        assert config["gripper_preview_steps"] == 0
        assert config["checkpoints"][0]["gpu_uuid"] == control.GPU
        assert "gpu" not in config["checkpoints"][0]
        assert "profiles" not in config and "oracle_reference_root" not in config
        assert (control.REPO / config["output_dir"]).is_relative_to(control.ROOT)
    assert prepared == original
    with pytest.raises(ValueError, match="case binding"):
        control.make_case_config(
            prepared,
            registration,
            prepared_ref=prepared_ref,
            registration_ref=registration_ref,
            case=dict(control.case_order()[0], sampling_seed=43),
        )


def test_dry_collection_does_not_create_physics_or_preload(monkeypatch, tmp_path, worker):
    case = worker.case_order()[0]
    root = tmp_path / "development/cases" / case["case_id"]
    config = dict(output_dir=str(root))
    monkeypatch.setattr(worker, "strict_native_environment", lambda: {})
    monkeypatch.setattr(worker, "admit_development", lambda *a, **k: (dict(root=str(tmp_path)), {}, {}, {}, {}))
    monkeypatch.setattr(worker, "read", lambda _: config)
    monkeypatch.setattr(worker, "make_case_config", lambda *a, **k: config)
    monkeypatch.setattr(worker, "restore_cv2_environment", lambda *a: pytest.fail("preload on dry collection"))
    result = worker.execute({}, {}, {}, {}, {}, case_index=0, dry=True)
    assert result["dry"] and result["physics_steps"] == result["requests"] == 0
    assert not root.exists()
