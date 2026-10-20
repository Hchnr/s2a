"""019 provenance must admit new stochastic outputs without admitting wrong inputs."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def audit(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("raw8_seed43_audit_test", root / "scripts/audit_wam_raw8_seed43.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def initial_pair(audit):
    request = {key: key for key in audit.FIRST_CONDITIONS}
    request.update(seed=42, extra={"action_dim_mask": [True] * 10 + [False] * 118})
    old = dict(
        request=request,
        record={"name": "same_fixture"},
        controller={"actual_initial_physics_sha256": "f" * 64},
        png=b"exact original pixels",
        chunk=dict(
            step=0,
            anchor_tcp=np.eye(4).tolist(),
            anchor_opening=0.8,
            physics_unchanged=True,
            physics_before_sha256="f" * 64,
            physics_after_sha256="f" * 64,
            physics_before_time_s=0.0,
            physics_after_time_s=0.0,
            raw128=[[1.0] * 128],
        ),
        row=dict(
            measured_source=np.eye(4).tolist(),
            measured_twist=[0.0] * 6,
            opening_before=0.8,
            gripper_qpos_before=[0.03, -0.03],
            motor_target_before=[0.5, -0.5],
        ),
    )
    current = copy.deepcopy(old)
    current["request"]["seed"] = 43
    current["chunk"]["raw128"] = [[-99.0] * 128]
    return old, current


def test_seed43_output_is_not_required_to_reproduce_seed42(audit):
    old, current = initial_pair(audit)
    result = audit.compare_first_inputs(old, current)
    assert not result["seed43_raw_equals_seed42_required"]
    assert result["actual_sampling_seed"] == 43 and result["initial_measured_state_exact"]
    assert old["chunk"]["raw128"] != current["chunk"]["raw128"]


def test_changed_seed_or_causal_initial_inputs_are_rejected(audit):
    old, original = initial_pair(audit)
    changed = []
    for key, value in (
        ("seed", 42),
        ("seed", 43.0),
        ("prompt", "different task"),
        ("shift", 3),
        ("extra", {"action_dim_mask": [True] * 128}),
    ):
        current = copy.deepcopy(original)
        current["request"][key] = value
        changed.append(current)
    for group, key, value in (
        ("chunk", "anchor_opening", 0.81),
        ("chunk", "physics_after_time_s", 0.05),
        ("controller", "actual_initial_physics_sha256", "0" * 64),
        ("row", "motor_target_before", [0.4, -0.4]),
    ):
        current = copy.deepcopy(original)
        current[group][key] = value
        changed.append(current)
    current = copy.deepcopy(original)
    current["png"] += b"changed"
    changed.append(current)
    for current in changed:
        with pytest.raises(AssertionError):
            audit.compare_first_inputs(old, current)


def service_log(pid=123, count=75, port=9744):
    return "\n".join(
        [
            f"INFO: Started server process [{pid}]",
            f"INFO: Uvicorn running on http://127.0.0.1:{port}",
            *['INFO: 127.0.0.1 - "POST /generate HTTP/1.1" 200 OK'] * count,
            "INFO: Shutting down",
            "INFO: Waiting for application shutdown.",
            "INFO: Application shutdown complete.",
            f"INFO: Finished server process [{pid}]",
        ]
    )


def test_exact_http_count_requires_owned_normal_lifetime(audit):
    assert audit.http_budget(service_log(), 123, 9744)["successful_http_requests"] == 75
    for bad in (
        service_log(count=74),
        service_log(count=76),
        service_log(pid=456),
        service_log().replace('"POST /generate HTTP/1.1" 200', '"POST /generate HTTP/1.1" 500', 1),
        service_log().replace("Waiting for application shutdown.", ""),
        '"POST /generate HTTP/1.1" 200 OK\n' + service_log(count=74),
        service_log(count=74) + '\n"POST /generate HTTP/1.1" 200 OK',
    ):
        with pytest.raises(AssertionError):
            audit.http_budget(bad, 123, 9744)


def test_incomplete_group_does_not_turn_pending_into_failure(audit):
    good = dict(status="verified", result=dict(outcome=dict(ever=True, terminal=True, stable=True)))
    assert audit.group_outcome([good]) is None
    assert audit.group_outcome([good, good, dict(status="pending")]) is None
    assert audit.group_outcome([good, good, dict(status="error")]) is None
    failed = dict(status="verified", result=dict(outcome=dict(ever=True, terminal=False, stable=False)))
    result = audit.group_outcome([good, failed, failed])
    assert result["stable"] == dict(successes=1, trials=3, rate=1 / 3)
    assert result["ever"]["rate"] == 1


def roster(audit):
    config = json.loads((audit.REPO / "configs/wam_raw8_seed43_task0_20261021.json").read_text())
    service, matrix = audit.expected_commands(config, 0)
    command = ["registered coordinator"]
    worker = [
        str((audit.REPO / config["simulation_python"]).absolute()),
        str(audit.REPO / "scripts/run_wam_position_matrix.py"),
        "--config",
        str(audit.REPO / "configs/wam_raw8_seed43_task0_20261021.json"),
        "--seed",
        "42",
        "--source",
        "wam",
        "--variant",
        "position_hold",
        "--record",
        "task0_init0",
        "--sampling-seed",
        "43",
        "--checkpoint",
        "8400_masked",
    ]
    identities = []
    for index, (role, cmd) in enumerate(
        zip(("coordinator", "model", "matrix", "physics"), (command, service, matrix, worker), strict=True)
    ):
        identities.append(
            dict(
                role=role,
                pid=100 + index,
                start_ticks=1000 + index,
                cwd=str(audit.REPO),
                proc_state="S",
                command=cmd,
                ppid=102 if role == "physics" else 100,
                environment=dict(
                    CUDA_VISIBLE_DEVICES="4" if role == "model" else "",
                    LD_LIBRARY_PATH="",
                    MUJOCO_GL="osmesa",
                    PYOPENGL_PLATFORM="osmesa",
                ),
            )
        )
    launch = dict(coordinator=dict(pid=100, start_ticks=1000), command=command)
    snapshot = dict(
        partition=0,
        ready=True,
        actual_http_info_equals_saved=True,
        physical_gpu=4,
        port=9744,
        local_registered_processes=identities,
    )
    manifest = dict(
        config=config, physical_gpu=4, coordinator_pid=100, model_command=service, jobs=[["position_hold", matrix]]
    )
    return config, launch, snapshot, manifest


def test_actual_ticks_parentage_and_physical_gpu_are_mandatory(audit):
    config, launch, snapshot, manifest = roster(audit)
    assert set(audit.check_identity_roster(0, config, launch, snapshot, manifest)) == {
        "coordinator",
        "model",
        "matrix",
        "physics",
    }
    changes = []
    for index, key, value in ((1, "ppid", 999), (0, "start_ticks", 999), (3, "start_ticks", 0), (3, "pid", 101)):
        bad = copy.deepcopy(snapshot)
        bad["local_registered_processes"][index][key] = value
        changes.append(bad)
    bad = copy.deepcopy(snapshot)
    bad["local_registered_processes"][1]["environment"]["CUDA_VISIBLE_DEVICES"] = "0"
    changes.append(bad)
    bad = copy.deepcopy(snapshot)
    bad["local_registered_processes"][3]["command"][-3] = "42"
    changes.append(bad)
    for bad in changes:
        with pytest.raises((AssertionError, ValueError)):
            audit.check_identity_roster(0, config, launch, bad, manifest)


def test_existing_literal_checker_is_reused_without_mutation(audit):
    import wam_position_acceptance_checker as original

    assert audit.verify_case is original.verify_case and audit.verify_query is original.verify_query
    assert audit.first_query is not original.verify_query
    proof = audit.source_proof(audit.Evidence())
    assert proof["all_other_ast_nodes_exact"] and not proof["dynamic_patch_or_generated_function_used"]


def test_malformed_active_json_is_pending_without_hashing_it(audit, tmp_path):
    path = tmp_path / "partial.json"
    path.write_text('{"incomplete":')
    evidence = audit.Evidence()
    assert audit.optional_json(path, evidence) is None and not evidence.hashes
    path.write_text('{"complete":true}')
    assert audit.optional_json(path, evidence) == {"complete": True}
    assert str(path.resolve()) in evidence.hashes


def test_opening_tracking_is_observed_aperture_error(audit):
    rows = [dict(opening_target=0.8, opening_after=0.6)] * 600
    result = audit.opening_tracking(rows)
    assert result["mae"] == pytest.approx(0.2) and result["rmse"] == pytest.approx(0.2)
    with pytest.raises(AssertionError):
        audit.opening_tracking(rows[:599])
