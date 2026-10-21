"""The021 outer contract cannot hide the original zero-step failure or weaken020."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def audit(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "temporal_continuation_audit_test", root / "scripts/audit_wam_temporal_continuation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_registered_output_port_and_appended_note_change(audit):
    old = json.loads((audit.REPO / "configs/wam_temporal_equal_task2_20261021.json").read_text())
    current = json.loads((audit.REPO / audit.CONFIG).read_text())
    assert audit.same_effect(old, current)
    alternatives = []
    for key, value in (("sampling_seeds", [43]), ("online_max_steps", 1200), ("guidance", 3.0)):
        alternatives.append(dict(current, **{key: value}))
    for key, value in (("gpu", 5), ("port", 9846)):
        changed = copy.deepcopy(current)
        changed["checkpoints"][0][key] = value
        alternatives.append(changed)
    changed = copy.deepcopy(current)
    changed["notes"][0] = "history rewritten"
    alternatives.append(changed)
    changed = copy.deepcopy(current)
    changed["variants"][0]["model_rotation_alpha"] = 0.0
    alternatives.append(changed)
    for changed in alternatives:
        with pytest.raises(AssertionError):
            audit.same_effect(old, changed)


def partial_ready(audit):
    return dict(
        schema_version="s2a.temporal-equal-live-ready-partial-snapshot.v1",
        complete_group_ready=False,
        captured_partitions=[0, 1],
        missing_partition=2,
        launch_sha256=audit.ORIGINAL_HASHES["launch"],
        local_network_operations="two GET /info requests only",
        new_model_requests=0,
        new_physics_steps=0,
        partitions=[
            dict(
                partition=i,
                local_registered_processes=[dict(pid=i * 4 + j, start_ticks=100 + i * 4 + j) for j in range(4)],
            )
            for i in (0, 1)
        ],
    )


def test_partial_ready_is_eight_historical_identities_never_whole_ready(audit):
    value = partial_ready(audit)
    assert audit.check_partial_ready(value)
    for key, changed in (
        ("schema_version", "s2a.temporal-equal-live-ready-snapshot.v1"),
        ("complete_group_ready", True),
        ("captured_partitions", [0, 1, 2]),
        ("missing_partition", 1),
        ("new_model_requests", 1),
        ("launch_sha256", "0" * 64),
    ):
        with pytest.raises(AssertionError):
            audit.check_partial_ready(dict(value, **{key: changed}))
    duplicated = copy.deepcopy(value)
    duplicated["partitions"][1]["local_registered_processes"][0] = duplicated["partitions"][0][
        "local_registered_processes"
    ][0]
    with pytest.raises(AssertionError):
        audit.check_partial_ready(duplicated)


def test_unknown_original_worker_is_scanned_by_exact_command_not_invented_pid(audit):
    command = [
        "python",
        str(audit.REPO / "scripts/run_wam_temporal_matrix.py"),
        "--config",
        str(audit.REPO / "configs/wam_temporal_equal_task2_20261021.json"),
        "--record",
        "task2_init0",
        "--sampling-seed",
        "42",
    ]
    assert audit.old_worker_matches(command)
    for index, replacement in (
        (1, str(audit.REPO / "scripts/run_wam_temporal_continuation_matrix.py")),
        (3, str(audit.REPO / audit.CONFIG)),
        (5, "task0_init0"),
        (7, "43"),
    ):
        changed = command.copy()
        changed[index] = replacement
        assert not audit.old_worker_matches(changed)
    assert not audit.old_worker_matches(command[:-1])
    assert not audit.old_worker_matches(["python"])


def test_original_zero_effect_rejects_generated_or_physics_evidence(audit, tmp_path):
    case, service = tmp_path / "case", tmp_path / "service"
    case.mkdir()
    service.mkdir()
    for filename in ("intervention.json", "runtime.json"):
        (service / filename).write_text("{}")
    log = 'INFO: 127.0.0.1 - "GET /info HTTP/1.1" 200 OK\n'
    assert audit.validate_zero_files(case, service, log)
    for filename in ("manifest.json", "steps.jsonl", "chunks.json"):
        path = case / filename
        path.write_text("{}")
        with pytest.raises(AssertionError):
            audit.validate_zero_files(case, service, log)
        path.unlink()
    generated = service / "generate_UUID"
    generated.mkdir()
    with pytest.raises(AssertionError):
        audit.validate_zero_files(case, service, log)
    generated.rmdir()
    for status in ("200 OK", "500 Internal Server Error"):
        with pytest.raises(AssertionError):
            audit.validate_zero_files(case, service, log + f'INFO: "POST /generate HTTP/1.1" {status}\n')


def handshake(audit):
    root = audit.REPO / "outputs/synthetic021"
    registry_path = root / "task2_init0_s42.process.json"
    config_path, snapshot_path = audit.REPO / audit.CONFIG, root / "execution_snapshot.json"
    env = dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa")
    identity = dict(
        pid=12, start_ticks=101, ppid=11, command=["python", "worker"], cwd=str(audit.REPO), environment=env
    )
    unknown = copy.deepcopy(identity)
    unknown["environment"]["CUDA_VISIBLE_DEVICES"] = None
    observations = [unknown, copy.deepcopy(identity)]
    parent = dict(pid=11, start_ticks=100)
    pending = dict(pid=12, start_ticks=101, ppid=11, command=identity["command"], automatic_retry=False)
    ready = dict(
        schema_version="s2a.cpu-worker-ready.v1",
        token="a" * 32,
        identity=identity,
        parent_identity=parent,
        config_path=str(config_path),
        config_sha256="c" * 64,
        snapshot_path=str(snapshot_path),
        snapshot_sha256="s" * 64,
        experiment_work_started=False,
        self_observations=observations,
    )
    registry = dict(
        **copy.deepcopy(identity),
        identity=copy.deepcopy(identity),
        automatic_retry=False,
        snapshot_sha256="s" * 64,
        ready_sha256="r" * 64,
        ready_path=str(root / "task2_init0_s42.handshake/ready.json"),
        parent_observations=copy.deepcopy(observations),
    )
    go = dict(
        schema_version="s2a.cpu-worker-go.v1",
        token=ready["token"],
        identity=identity,
        parent_identity=parent,
        registry_path=str(registry_path),
        registry_sha256="g" * 64,
        ready_sha256="r" * 64,
        config_sha256="c" * 64,
        snapshot_sha256="s" * 64,
    )
    kwargs = dict(
        config_path=config_path,
        config_sha="c" * 64,
        snapshot_path=snapshot_path,
        snapshot_sha="s" * 64,
        registry_path=registry_path,
        ready_sha="r" * 64,
        registry_sha="g" * 64,
        parent=parent,
    )
    return [ready, registry, go, pending], kwargs


def test_saved_go_binds_real_birth_parent_and_every_immutable_receipt(audit):
    values, kwargs = handshake(audit)
    result = audit.handshake_contract(*values, **kwargs)
    assert result["ready_go_registry_binding_passed"]
    assert result["no_unknown_environment_normalized_to_empty"]
    assert result["experiment_work_started_before_ready"] is False
    for index, key, changed in (
        (0, "token", "wrong-token"),
        (0, "experiment_work_started", True),
        (0, "config_sha256", "wrong"),
        (0, "parent_identity", dict(pid=11, start_ticks=999)),
        (1, "snapshot_sha256", "wrong"),
        (1, "automatic_retry", True),
        (2, "registry_sha256", "wrong"),
        (2, "ready_sha256", "wrong"),
        (3, "start_ticks", 999),
        (3, "ppid", 999),
    ):
        bad = copy.deepcopy(values)
        bad[index][key] = changed
        with pytest.raises(AssertionError):
            audit.handshake_contract(*bad, **kwargs)


def test_unknown_is_allowed_only_before_a_complete_unchanged_environment_observation(audit):
    values, kwargs = handshake(audit)
    for record_index, observations_key in ((0, "self_observations"), (1, "parent_observations")):
        for key, changed in (("CUDA_VISIBLE_DEVICES", "6"), ("LD_LIBRARY_PATH", "/external"), ("MUJOCO_GL", "egl")):
            bad = copy.deepcopy(values)
            bad[record_index][observations_key][0]["environment"][key] = changed
            with pytest.raises(AssertionError):
                audit.handshake_contract(*bad, **kwargs)
        for changed in (None, "6"):
            bad = copy.deepcopy(values)
            bad[record_index][observations_key][-1]["environment"]["CUDA_VISIBLE_DEVICES"] = changed
            with pytest.raises(AssertionError):
                audit.handshake_contract(*bad, **kwargs)
        bad = copy.deepcopy(values)
        bad[record_index][observations_key][0]["pid"] = 777
        with pytest.raises(AssertionError):
            audit.handshake_contract(*bad, **kwargs)


def test_case_mechanics_and_whole_ready_contract_remain_the_frozen_independent_code(audit):
    import audit_wam_temporal_wam as original
    import wam_temporal_position_checker as checker

    assert audit.original is original
    assert audit.original.verify_case is checker.verify_case
    assert audit.original.first_query is original.first_query
    assert audit.original.group_outcome([dict(status="verified", result={}), dict(status="pending")]) is None
    assert audit.original.BUDGET == dict(
        wam_trials=3, physics_steps=1800, maximum_h40_queries=225, fresh_initial_states=0
    )
