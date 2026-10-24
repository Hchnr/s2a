"""Substantive038 development admission, causal transport and score tests; no simulator."""

import ast
import copy
import io
import json
import os
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import execute_wam_clean_training_case as worker  # noqa: E402
import run_wam_clean_training_development as owner  # noqa: E402
import verify_wam_clean_training_development as verifier  # noqa: E402
from run_wam_readout import owned_command  # noqa: E402

from s2a.clean_training_runtime import reference  # noqa: E402


def save(path, value):
    path.write_text(json.dumps(value))
    return reference(path)


@pytest.fixture
def state(tmp_path):
    p, r = dict(path="/prepared", sha256="a" * 64), dict(path="/registration", sha256="b" * 64)
    commands = []
    for i in range(2):
        birth = dict(pid=123 + i, start_ticks=1000 + i)
        commands.append(
            dict(
                launch=save(tmp_path / f"launch{i}.json", dict(child=birth)),
                exit=save(tmp_path / f"exit{i}.json", dict(actual_wait=True, returncode=0, automatic_retry=False)),
                release=save(
                    tmp_path / f"release{i}.json",
                    dict(
                        **birth,
                        reaped=True,
                        returncode=0,
                        signal_sent=False,
                        original_error=None,
                        only_owned_child_signalled=True,
                    ),
                ),
            )
        )
    gpu = save(tmp_path / "gpu.json", dict(released=True, uuid=owner.GPU, memory_mib=0, compute_apps=[]))
    sampling = dict(path="/sampling", sha256="c" * 64)
    audit = dict(
        schema_version="s2a.clean-training-state-verification.v1",
        complete=True,
        passed=True,
        prepared_ref=p,
        execution_registration_ref=r,
        sampling_ref=sampling,
        actual_requests=99,
        actual_denoise_calls=3465,
        actual_head_calls=3465,
        model_initializations=1,
        paired_initial_conditions=48,
        zero_requests_verified=3,
        numerical_contract_passed=True,
        eligible_for_development=True,
        state_gate_passed=False,
        state_gate=dict(passed=False),
        worker_launch_ref=commands[0]["launch"],
        worker_exit_ref=commands[0]["exit"],
        worker_release_ref=commands[0]["release"],
        gpu_after_ref=gpu,
    )
    audit_ref = save(tmp_path / "audit.json", audit)
    envelope = dict(
        schema_version="s2a.clean-training-state-owner.v1",
        complete=True,
        passed=True,
        prepared_ref=p,
        execution_registration_ref=r,
        producer_ref=sampling,
        verification_ref=audit_ref,
        owned_commands=commands,
        gpu_after_ref=gpu,
    )
    envelope_ref = save(tmp_path / "owner.json", envelope)
    return p, r, envelope_ref, audit_ref, envelope, audit


def test_state_gate_false_does_not_block_complete_numeric_stage(state):
    p, r, envelope, audit, _, _ = state
    assert owner.admit_state(p, r, envelope, audit)["state_gate_passed"] is False


@pytest.mark.parametrize(
    "fault",
    [
        "partial99",
        "partial3465",
        "wrong_candidate_source",
        "owner_failed",
        "missing_wait",
        "gpu_not_released",
        "gate_rewritten",
    ],
)
def test_state_stage_cannot_be_replaced_or_shortened(state, tmp_path, fault):
    p, r, envelope_ref, audit_ref, envelope, audit = state
    if fault == "partial99":
        audit["actual_requests"] = 98
    elif fault == "partial3465":
        audit["actual_denoise_calls"] = 3430
    elif fault == "wrong_candidate_source":
        audit["sampling_ref"] = dict(path="/other", sha256="d" * 64)
    elif fault == "owner_failed":
        envelope["passed"] = False
    elif fault == "missing_wait":
        envelope["owned_commands"][0]["exit"] = save(
            tmp_path / "bad_exit.json", dict(actual_wait=False, returncode=0, automatic_retry=False)
        )
    elif fault == "gpu_not_released":
        bad = save(tmp_path / "bad_gpu.json", dict(released=False, uuid=owner.GPU, memory_mib=100, compute_apps=[1]))
        envelope["gpu_after_ref"] = audit["gpu_after_ref"] = bad
    else:
        audit["state_gate_passed"] = True
    audit_ref = save(tmp_path / "tampered_audit.json", audit)
    envelope["verification_ref"] = audit_ref
    envelope_ref = save(tmp_path / "tampered_owner.json", envelope)
    with pytest.raises(ValueError):
        owner.admit_state(p, r, envelope_ref, audit_ref)


def test_exact_case_order_and_gpu_is_only_uuid_in_launched_commands(tmp_path, monkeypatch):
    expected = [f"task{t}_init0_s{s}_{v}" for t in range(3) for s in (42, 43) for v in ("raw", "w2")]
    assert [x["case_id"] for x in owner.case_order()] == expected
    assert owner.case_order() == verifier.order()
    prepared = dict(root=str(tmp_path), model_config=dict(model_python="/registered/python"))
    monkeypatch.setattr(
        owner, "admit_development", lambda *a, **k: (prepared, {}, {}, dict(state_gate_passed=False), {})
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    ref = dict(path="/synthetic", sha256="a" * 64)
    result = owner.execute(ref, ref, ref, ref, dry_run=True)
    assert result["service"][result["service"].index("--gpu-uuid") + 1] == owner.GPU
    assert (
        result["case_order"] == owner.case_order()
        and result["model_initializations"] == result["new_physics_steps"] == 0
    )
    assert not tmp_path.exists() or not list(tmp_path.iterdir())


def test_case_config_preserves_qualified_control_and_isolates_state(tmp_path):
    prepared = dict(
        root=str(tmp_path), model_config=dict(checkpoint=dict(run="/registered/run", iteration="iter_000008400"))
    )
    a, b = (owner.make_case_config(prepared, case) for case in owner.case_order()[:2])
    assert (
        a["variants"]
        == b["variants"]
        == [
            dict(
                name="position_hold",
                rotation="response",
                gripper="position_hold",
                governor=False,
                model_rotation_alpha=1.0,
            )
        ]
    )
    assert (a["online_max_steps"], a["online_chunk_size"], a["max_observation_interval_steps"], a["control_fps"]) == (
        600,
        40,
        8,
        20,
    )
    assert a["output_dir"] != b["output_dir"]
    assert a["checkpoints"][0]["gpu"] == 4  # CPU-only legacy metadata, never a model launch environment.
    assert a["checkpoints"][0]["port"] == b["checkpoints"][0]["port"] == 31339
    assert a["model_reference_label"] == "raw_alpha1"


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


def test_transport_cap_keeps_original_json_and_restores_session(tmp_path, monkeypatch):
    seen = []

    class Session:
        def request(self, method, url, **kwargs):
            seen.append((method, url, kwargs))
            return type("Response", (), dict(status_code=200))()

    monkeypatch.setattr(requests, "Session", Session)
    log = io.StringIO()
    body = request(tmp_path)
    saved = copy.deepcopy(body)
    with worker.bounded_http(log, owner.case_order()[0], tmp_path):
        requests.Session().request("POST", "http://127.0.0.1:31339/generate", json=body, timeout=900)
    assert requests.Session is Session and seen[0][2]["timeout"] == 300
    assert seen[0][2]["json"] is body and body == saved
    record = json.loads(log.getvalue())
    assert record["timeout_s"] == 300 and record["completed"] is True


@pytest.mark.parametrize("fault", ["host", "future", "seed", "other_image", "timeout"])
def test_transport_rejects_causal_or_resource_escape(tmp_path, monkeypatch, fault):
    class Session:
        def request(self, *a, **k):
            raise AssertionError("invalid transport must fail before HTTP dispatch")

    monkeypatch.setattr(requests, "Session", Session)
    body = request(tmp_path)
    url = "http://127.0.0.1:31339/generate"
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
    with pytest.raises(ValueError), worker.bounded_http(io.StringIO(), owner.case_order()[0], tmp_path):
        requests.Session().request("POST", url, json=body, timeout=timeout)
    assert requests.Session is Session


def test_stable_requires_whole_actual_tail_and_preserves_ever_terminal():
    assert verifier.outcome([False] * 560 + [True] * 40) == dict(
        ever=True, terminal=True, stable=True, successful_steps=40, trailing_success_steps=40
    )
    transient = verifier.outcome([False] * 560 + [True] * 39 + [False])
    assert transient["ever"] is True and transient["terminal"] is transient["stable"] is False
    with pytest.raises(ValueError):
        verifier.outcome([True] * 599)
    with pytest.raises(ValueError):
        verifier.outcome([1] * 600)


def scored(raw, w2):
    cases = []
    for case in verifier.order():
        successes = raw if case["variant"] == "raw" else w2
        stable = (case["task_index"], case["sampling_seed"]) in successes
        cases.append(dict(case=case, outcome=dict(ever=stable, terminal=stable, stable=stable)))
    return cases


def test_fixed_stability_and_strict_gain_are_separate_no_seed_stitching():
    six = {(t, s) for t in range(3) for s in (42, 43)}
    five = six - {(2, 43)}
    result = verifier.score(scored(six, five))
    assert result["development_stability_gate_passed"] is True and result["strict_adaptation_gain_passed"] is False
    assert (
        result["eligible_for_later_fresh_candidate_freeze"] is True and result["fresh_performance_authorized"] is False
    )
    assert verifier.score(scored(set(), five))["strict_adaptation_gain_passed"] is True
    assert (
        verifier.score(scored(set(), {(0, 42), (0, 43), (1, 42), (1, 43)}))["development_stability_gate_passed"]
        is False
    )
    with pytest.raises(ValueError):
        verifier.score(scored(set(), six)[:-1])
    cases = scored(set(), six)
    cases[0], cases[1] = cases[1], cases[0]
    with pytest.raises(ValueError):
        verifier.score(cases)


def test_verifier_does_not_import_development_producers():
    source = Path(verifier.__file__).read_text()
    imports = {node.module for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ImportFrom)}
    assert not imports & {
        "run_wam_clean_training_development",
        "execute_wam_clean_training_case",
        "serve_wam_clean_training",
        "s2a.wam_clean_readout",
    }
    assert "wam_position_acceptance_checker" in imports


def test_new_service_evidence_cannot_be_reported_as_historical003_regression():
    mechanical = dict(
        outcome=dict(stable=False), steps=600, first_query003_regression=verifier.current_query_evidence_scope()
    )
    labeled = verifier.label_current_mechanical_evidence(mechanical)
    assert "first_query003_regression" not in labeled
    assert labeled["current_query_evidence_scope"]["historical_003_regression_performed"] is False
    assert labeled["outcome"] == mechanical["outcome"] and labeled["steps"] == 600
    assert "first_query003_regression" in mechanical  # Original014 return object is untouched.
    with pytest.raises(ValueError, match="unexpected claim"):
        verifier.label_current_mechanical_evidence(dict(mechanical, first_query003_regression=dict(verified=True)))


@pytest.mark.parametrize(
    "fault", [None, "variant", "readout", "seed", "case_index", "other_current_image", "response_replaced"]
)
def test_service_actual_http_binding_rejects_candidate_seed_or_source_mix(tmp_path, fault):
    case = verifier.order()[1]  # W2 of the first exact task/seed pair.
    image = tmp_path / "input_0000.png"
    image.write_bytes(b"current image bytes")
    head = dict(path="/the-frozen-W2.json", sha256="e" * 64)
    req = request(tmp_path)
    record = dict(
        complete=True,
        case_id=case["case_id"],
        case_index=case["case_index"],
        request_index=0,
        global_request_index=75,
        variant="w2",
        sampling_seed=42,
        anchor_name=case["case_id"],
        readout_ref=head,
        image_ref=reference(image),
    )
    response = dict(
        status="success",
        clean_training={
            key: record[key]
            for key in (
                "case_index",
                "case_id",
                "request_index",
                "global_request_index",
                "variant",
                "sampling_seed",
                "readout_ref",
            )
        },
    )
    record["request_ref"] = save(tmp_path / "request.json", req)
    record["response_ref"] = save(tmp_path / "response.json", response)
    if fault == "variant":
        record["variant"] = "raw"
    elif fault == "readout":
        record["readout_ref"] = None
    elif fault == "seed":
        record["sampling_seed"] = 43
    elif fault == "case_index":
        record["case_index"] = 0
    elif fault == "other_current_image":
        other = tmp_path / "future.png"
        other.write_bytes(b"same hash is insufficient without fixed current path")
        record["image_ref"] = reference(other)
    elif fault == "response_replaced":
        response["clean_training"]["sampling_seed"] = 43
    if fault is None:
        verifier.verify_service_request(
            verifier.numerics.Evidence(), record, case, 0, tmp_path, req, response, head_ref=head
        )
    else:
        with pytest.raises(ValueError):
            verifier.verify_service_request(
                verifier.numerics.Evidence(), record, case, 0, tmp_path, req, response, head_ref=head
            )


@pytest.mark.parametrize("code", [0, 7])
def test_real_cpu_child_wait_reap_and_retained_failure(tmp_path, code):
    folder = tmp_path / "owned"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa")
    command = [sys.executable, "-c", f"import sys;sys.exit({code})"]
    if code:
        with pytest.raises(ChildProcessError):
            owned_command(command, env, folder, timeout=5)
        assert json.loads((folder / "exit.json").read_text())["actual_wait"] is True
        assert json.loads((folder / "ownership/release.json").read_text())["reaped"] is True
    else:
        entry = owned_command(command, env, folder, timeout=5)
        verifier.normal_owned(verifier.numerics.Evidence(), entry, "")
