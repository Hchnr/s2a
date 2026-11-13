"""Audit the real43 owner's evidence and uncertain partial costs on CPU."""

import copy
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.simulation.runtime import adaptive_multiseed_real43_admission as admission
from scripts.simulation.runtime import adaptive_multiseed_real43_owner as owner
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.wam.tracking.preview import adaptive_multiseed_contract as contract
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro
from tests.research.native_runtime.test_streaming_continuation import response_for


@pytest.fixture(scope="module")
def saved():
    return admission.load_inputs()


def final_model():
    return dict(
        model_initialization_attempts=1,
        model_initializations=1,
        model_requests_attempted=1,
        model_requests_completed=1,
        denoise_calls_attempted=35,
        denoise_calls_completed=35,
    )


def lifecycle(*, reaped=True):
    return {"workers": [{"name": "model", "actual_wait": reaped, "reaped": reaped}]}


def test_missing_worker_receipt_preserves_uncertain_actual_cost():
    unknown = owner.model_cost_bounds(lifecycle(), None, launch_attempted=True, sources_match=True)
    assert unknown["model_initializations"] == {"lower": 0, "upper": 1}
    assert unknown["denoise_calls_attempted"] == {"lower": 0, "upper": 35}
    not_started = owner.model_cost_bounds(None, None, launch_attempted=False, sources_match=True)
    assert all(row == {"lower": 0, "upper": 0} for row in not_started.values())
    not_reaped = owner.model_cost_bounds(
        lifecycle(reaped=False), final_model(), launch_attempted=True, sources_match=True
    )
    assert not_reaped == unknown
    changed_source = owner.model_cost_bounds(lifecycle(), final_model(), launch_attempted=True, sources_match=False)
    assert changed_source == unknown


def test_reaped_final_receipt_preserves_partial_actual_cost_and_rejects_impossible_values():
    model = final_model()
    model.update(model_requests_completed=0, denoise_calls_attempted=3, denoise_calls_completed=2)
    bounds = owner.model_cost_bounds(lifecycle(), model, launch_attempted=True, sources_match=True)
    assert bounds["model_requests_attempted"] == {"lower": 1, "upper": 1}
    assert bounds["model_requests_completed"] == {"lower": 0, "upper": 0}
    assert bounds["denoise_calls_attempted"] == {"lower": 3, "upper": 3}
    assert bounds["denoise_calls_completed"] == {"lower": 2, "upper": 2}
    for key, value in (
        ("denoise_calls_attempted", 36),
        ("model_initializations", True),
        ("model_requests_attempted", 0),
    ):
        bad = dict(model, **{key: value})
        with pytest.raises(ValueError, match="internally consistent"):
            owner.model_cost_bounds(lifecycle(), bad, launch_attempted=True, sources_match=True)


def sample_exchange(saved, tmp_path, monkeypatch, *, change=None):
    root = tmp_path / "trial"
    model_dir = root / "producer/model/model_run/step0"
    model_dir.mkdir(parents=True)
    monkeypatch.setattr(admission, "OUTPUT", root)
    image = model_dir / "current.png"
    shutil.copyfile(saved["refs"]["historical_image"]["path"], image)
    image_arg = {"step": 0, "path": str(image.relative_to(admission.ROOT)), "sha256": saved["config"]["image_sha256"]}
    request = contract.make_task_request(saved["template"], image_arg, saved["config"]["task"], 43)
    checkpoint = saved["prepared"]["model_config"]["checkpoint"]
    expected_model = str(Path(checkpoint["run"]) / "checkpoints" / checkpoint["iteration"] / "model")
    action = [[0.0] * 128 for _ in range(40)]
    identity = dict(
        seed=43,
        action_fps=20,
        checkpoint_path=expected_model,
        action_units="raw_physical_or_source_units",
        action_dim_mask=[True] * 10 + [False] * 118,
        normalized_action=copy.deepcopy(action),
    )
    response = {
        "status": "success",
        "args": copy.deepcopy(request),
        "outputs": [{"content": {"action": action, "action128": identity}}],
    }
    frame = dict(
        version=1,
        session=saved["config"]["session"],
        request_id=0,
        **saved["initial"],
        image=dict(path="current.png", sha256=saved["config"]["image_sha256"]),
        task=saved["config"]["task"],
        seed=43,
        model_sha256=saved["config"]["model_sha256"],
        denoise_steps=35,
    )
    envelope = response_for(frame)
    if change == "actual_seed":
        response["args"]["seed"] = 42
    elif change == "actual_identity":
        identity["seed"] = 42
    elif change == "envelope_array":
        envelope["action"][0][0] = 1.0
    elif change == "envelope_state":
        envelope["state_sha256"] = "f" * 64
    elif change == "model_count":
        pass
    repro.write_new(model_dir / "request.json", request)
    repro.write_new(model_dir / "response.json", response)
    model = {
        **final_model(),
        "complete": True,
        "passed": True,
        "gpu_uuid": admission.GPU,
        "sampling_seed": 43,
        "responses_sent": 1,
        "denoise_evidence": [{"completed": True} for _ in range(35)],
        "mask_evidence": [{"stage": "initial_noise", "inactive_rms_before": 0.1, "inactive_rms_after": 0.0}],
        "physics_steps": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
        "model_sequence_complete": True,
        "responses": [
            dict(
                step=0,
                reference_sent=True,
                model_request_ref=prefix.reference(model_dir / "request.json"),
                actual_response_ref=prefix.reference(model_dir / "response.json"),
            )
        ],
    }
    if change == "model_count":
        model["denoise_calls_completed"] = 34
    elif change == "mask_evidence":
        model["mask_evidence"][0]["inactive_rms_after"] = 0.1
    elif change == "denoise_evidence":
        model["denoise_evidence"][0]["completed"] = False
    client = dict(
        complete=True,
        passed=True,
        evidence_level="saved_development_image_model_replay",
        physics_steps=0,
        native_models=0,
        action_dispatches=0,
        exchange=dict(
            request=frame,
            response=envelope,
            compared_to_saved_seed42=repro.compare_responses(response, saved["previous_response"]),
            accepted_requests=1,
            completed_periods=0,
            physics_steps=0,
            native_models=0,
            action_dispatches=0,
            snapshot_scope="historical_recorded_endpoint_not_live",
        ),
    )
    return client, model


def test_audit_rechecks_actual_model_response_against_client_envelope(saved, tmp_path, monkeypatch):
    client, model = sample_exchange(saved, tmp_path, monkeypatch)
    audit = owner.audit_exchange(saved, client, model)
    assert audit["sampling_seed"] == 43 and audit["model_arrays_match_client_envelope"]
    assert audit["image_sha256"] == saved["config"]["image_sha256"]
    assert audit["compared_to_saved_seed42"] == client["exchange"]["compared_to_saved_seed42"]


@pytest.mark.parametrize(
    "change",
    [
        "actual_seed",
        "actual_identity",
        "envelope_array",
        "envelope_state",
        "model_count",
        "mask_evidence",
        "denoise_evidence",
    ],
)
def test_audit_rejects_tampering_or_incomplete_model_work(saved, tmp_path, monkeypatch, change):
    client, model = sample_exchange(saved, tmp_path, monkeypatch, change=change)
    with pytest.raises(ValueError):
        owner.audit_exchange(saved, client, model)


def test_actual_model_python_imports_entry_without_early_cv2_torch_or_cosmos(saved):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH="src:.")
    program = (
        "import sys; import scripts.simulation.runtime.run_adaptive_multiseed_real43; "
        "assert 'cv2' not in sys.modules and 'torch' not in sys.modules; "
        "assert not any(n == 'cosmos_framework' or n.startswith('cosmos_framework.') for n in sys.modules)"
    )
    result = subprocess.run(
        [saved["prepared"]["model_config"]["model_python"], "-c", program],
        cwd=admission.ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-1000:]


@pytest.mark.parametrize("succeed", [True, False])
def test_owner_publishes_complete_or_uncertain_failure_without_real_model(saved, tmp_path, monkeypatch, succeed):
    root = tmp_path / "trial"
    root.mkdir()
    monkeypatch.setattr(admission, "OUTPUT", root)
    monkeypatch.setattr(admission, "load_inputs", lambda: saved)
    monkeypatch.setattr(owner, "source_refs", lambda: [])
    monkeypatch.setattr(owner, "snapshot", lambda _root: {"sources": {}})
    monkeypatch.setattr(owner.registered, "gpu_observation", lambda _uuid: {"memory_mib": 0, "compute_apps": []})

    class Trial:
        def require_config(self, _inputs):
            pass

        def check_cpu_admission(self, _inputs):
            return {}

        def check_preflight(self, _inputs):
            pass

        def model_bundle(self, _inputs):
            return {"synthetic": True}

    monkeypatch.setattr(owner, "make_trial", Trial)

    def run_fake_pair(specs, *, timeout_seconds):
        assert timeout_seconds == 1800
        assert specs["simulation"]["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert specs["model"]["env"]["CUDA_VISIBLE_DEVICES"] == admission.GPU
        for spec in specs.values():
            (admission.ROOT / spec["log"]).write_text("synthetic worker\n")
        if succeed:
            client, model = sample_exchange(saved, tmp_path, monkeypatch)
            owner.publish(root / "producer/simulation/result.json", client)
            owner.publish(root / "producer/model/result.json", model)
        return dict(
            passed=succeed,
            launch_error=None,
            timed_out=False,
            interrupted=False,
            workers=[
                dict(
                    name=name,
                    actual_wait=True,
                    reaped=True,
                    returncode=0 if succeed else 1,
                    signal_sent=False,
                    log=spec["log"],
                )
                for name, spec in specs.items()
            ],
        )

    monkeypatch.setattr(owner, "run_owned_socket_pair", run_fake_pair)
    if succeed:
        owner.execute()
    else:
        with pytest.raises(ValueError, match="workers must exit"):
            owner.execute()
    result = json.loads((root / "producer/result.json").read_text())
    assert result["complete"] is result["passed"] is succeed
    assert result["gpu_released"] and result["physics_steps"] == result["action_dispatches"] == 0
    if succeed:
        assert result["audit"]["sampling_seed"] == 43
        assert result["model_cost_bounds"]["denoise_calls_completed"] == {"lower": 35, "upper": 35}
    else:
        assert result["model_cost_bounds"]["denoise_calls_attempted"] == {"lower": 0, "upper": 35}
