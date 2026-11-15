"""The registered physical worker admits only its own immutable case."""

import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.references.broader_recovery_timeline import broader_recovery_timeline
from scripts.simulation.runtime import run_broader_recovery_pilot as run


@pytest.fixture
def registration():
    return dict(
        phase="producer",
        eligible_cases=list(range(24)),
        source_refs=[dict(path="fixed", sha256="digest")],
        inputs=dict(cases=[dict(index=index) for index in range(24)]),
        producer_branch_refs={},
        automatic_retry=False,
    )


def test_worker_admission_preserves_exact_case_and_source(registration):
    assert run.validate_registration("producer", 7, registration, registration["source_refs"]) == dict(index=7)
    for mutate in (
        lambda data: data["eligible_cases"].remove(7),
        lambda data: data["inputs"]["cases"][7].update(index=8),
        lambda data: data.update(automatic_retry=True),
        lambda data: data["source_refs"][0].update(sha256="changed"),
        lambda data: data.update(producer_branch_refs={"7": {}}),
    ):
        altered = copy.deepcopy(registration)
        mutate(altered)
        with pytest.raises(ValueError):
            run.validate_registration("producer", 7, altered, registration["source_refs"])
    for index in (-1, 24, True):
        with pytest.raises(ValueError, match="One registered case"):
            run.validate_registration("producer", index, registration, registration["source_refs"])


def test_probe_is_reset_only_case_and_checker_requires_producer(registration):
    probe = copy.deepcopy(registration)
    probe.update(phase="probe", eligible_cases=[0])
    assert run.validate_registration("probe", 0, probe, registration["source_refs"]) == dict(index=0)
    with pytest.raises(ValueError, match="One registered case"):
        run.validate_registration("probe", 1, probe, registration["source_refs"])
    checker = copy.deepcopy(registration)
    checker.update(phase="checker", producer_branch_refs={"7": {"sha256": "producer"}})
    assert run.validate_registration("checker", 7, checker, registration["source_refs"]) == dict(index=7)
    with pytest.raises(ValueError, match="replayed producer"):
        run.validate_registration("checker", 8, checker, registration["source_refs"])


def test_preflight_requires_source_matched_64_worker_checks(monkeypatch):
    identity = dict(cases=[dict(index=0)])
    sources = [dict(path="source", sha256="digest")]
    hashes = {"source": "digest"}
    qa = dict(
        complete=True,
        passed=True,
        all_tests_passed=True,
        source_unchanged=True,
        actual_wait=True,
        reaped=True,
        configured_parallelism=64,
        skipped_count=0,
        collection=dict(collection_errors=[]),
        source_before_ref="snapshot",
    )
    dry = dict(
        complete=True,
        passed=True,
        physics_steps=0,
        world_model_calls=0,
        gpu_calls=0,
        fresh_cases=0,
        training_authorized=False,
        inputs=identity,
        source_refs=sources,
    )
    objects = {str(run.QA): qa, str(run.OUTPUT / "dry.json"): dry, "snapshot": dict(sources=hashes)}
    monkeypatch.setattr(run.prefix, "environment", lambda: None)
    monkeypatch.setattr(run.prefix, "reference", lambda path: str(path))
    monkeypatch.setattr(run.prefix, "read", lambda ref: objects[ref])
    monkeypatch.setattr(run, "snapshot", lambda _root: dict(sources=hashes))
    monkeypatch.setattr(run, "source_refs", lambda: sources)
    monkeypatch.setattr(run.pilot, "load_inputs", lambda: ({"config": "fixed"}, identity))
    assert run.preflight()[1] == identity
    for key, invalid in (("configured_parallelism", 32), ("reaped", False), ("skipped_count", 1)):
        original = qa[key]
        qa[key] = invalid
        with pytest.raises(ValueError, match="source-matched CPU"):
            run.preflight()
        qa[key] = original
    dry["physics_steps"] = 1
    with pytest.raises(ValueError, match="zero-physics"):
        run.preflight()
    dry["physics_steps"] = 0
    dry["inputs"] = dict(cases=[])
    with pytest.raises(ValueError, match="Dry identity"):
        run.preflight()


def test_probe_admission_requires_reaped_child_and_exact_registration(monkeypatch):
    identity = dict(cases=[dict(index=0)])
    sources, qa_ref, dry_ref = ["source"], "qa", "dry"
    child = dict(key="0", returncode=0, actual_wait=True, reaped=True, signal_sent=False, log_ref="log")
    registration = dict(
        phase="probe",
        eligible_cases=[0],
        inputs=identity,
        source_refs=sources,
        qa_ref=qa_ref,
        dry_ref=dry_ref,
        producer_branch_refs={},
        automatic_retry=False,
    )
    branch = dict(case=identity["cases"][0])
    checked = dict(endpoint_count=1)
    probe = dict(
        complete=True,
        passed=True,
        owner_closed=True,
        sources_unchanged=True,
        inputs=identity,
        source_refs=sources,
        qa_ref=qa_ref,
        dry_ref=dry_ref,
        lifecycle_ref="lifecycle",
        registration_ref="registration",
        branch_ref="branch",
        capture_check=checked,
    )
    objects = {"lifecycle": dict(launch_error=None, workers=[child]), "registration": registration, "branch": branch}
    monkeypatch.setattr(run.prefix, "read", lambda ref: objects[ref])
    monkeypatch.setattr(run.prefix, "checked", lambda ref: ref)
    monkeypatch.setattr(run.preparation, "shared_inputs", lambda: {})
    monkeypatch.setattr(run, "bind_case", lambda _identity, _shared: ({}, {}))
    monkeypatch.setattr(run.recovery, "verify_probe_branch", lambda *_args: checked)
    run.verify_probe_admission(probe, identity, sources, qa_ref, dry_ref)
    child["reaped"] = False
    with pytest.raises(ValueError, match="actually reaped"):
        run.verify_probe_admission(probe, identity, sources, qa_ref, dry_ref)
    child["reaped"] = True
    registration["automatic_retry"] = True
    with pytest.raises(ValueError, match="registered training reset"):
        run.verify_probe_admission(probe, identity, sources, qa_ref, dry_ref)


def test_checker_replays_only_closed_producers_without_losing_failed_cases():
    identity = dict(cases=[dict(index=index) for index in range(4)])
    producer = dict(
        phase="producer",
        owner_closed=True,
        sources_unchanged=True,
        cases={
            "0": dict(status="closed", branch_ref="first"),
            "1": dict(status="failed"),
            "2": dict(status="closed", branch_ref="third"),
            "3": dict(status="failed"),
        },
    )
    eligible, refs = run.checker_eligible(producer, identity)
    assert eligible == [0, 2] and refs == {"0": "first", "2": "third"}
    assert len(producer["cases"]) == 4
    producer["owner_closed"] = False
    with pytest.raises(ValueError, match="closed owner"):
        run.checker_eligible(producer, identity)
    producer["owner_closed"] = True
    del producer["cases"]["3"]
    with pytest.raises(ValueError, match="full producer roster"):
        run.checker_eligible(producer, identity)


def test_missing_or_partial_worker_costs_are_never_treated_as_zero():
    assert run.observed_resources(None) is None
    assert run.observed_resources(dict(runtime={})) is None
    counters = {
        key: dict(attempted=attempted, completed=completed)
        for key, attempted, completed in (
            ("mj_step", 51, 50),
            ("mj_step2", 0, 0),
            ("model_construct", 3, 3),
            ("context", 3, 2),
        )
    }
    branch = dict(
        runtime=dict(control_steps_attempted=3, control_steps_completed=2, counters=counters, cleanup_complete=True)
    )
    observed = run.observed_resources(branch)
    assert observed["control_steps_attempted"] == 3
    assert observed["control_steps_completed"] == 2
    assert observed["counters"]["mj_step"] == dict(attempted=51, completed=50)
    assert observed["cleanup_complete"] is True
    del counters["context"]["attempted"]
    assert run.observed_resources(branch) is None


def test_independent_owner_verification_checks_actual_child_lifecycle(monkeypatch):
    identity = dict(cases=[dict(index=0), dict(index=1)])
    sources, qa_ref, dry_ref, probe_ref = ["source"], "qa", "dry", "probe"
    registration = dict(
        phase="producer",
        inputs=identity,
        source_refs=sources,
        qa_ref=qa_ref,
        dry_ref=dry_ref,
        probe_ref=probe_ref,
        eligible_cases=[0, 1],
        automatic_retry=False,
    )
    workers = [
        dict(key="0", actual_wait=True, reaped=True, returncode=0, signal_sent=False, log_ref="log0"),
        dict(key="1", actual_wait=True, reaped=True, returncode=1, signal_sent=False, log_ref="log1"),
    ]
    owner = dict(
        phase="producer",
        owner_closed=True,
        sources_unchanged=True,
        automatic_retry=False,
        registration_ref="registration",
        lifecycle_ref="lifecycle",
        cases={
            "0": dict(status="closed", branch_ref="branch0", available_branch_ref="branch0"),
            "1": dict(status="failed"),
        },
    )
    objects = {"registration": registration, "lifecycle": dict(launch_error=None, workers=workers)}
    monkeypatch.setattr(run.prefix, "read", lambda ref: objects[ref])
    monkeypatch.setattr(run.prefix, "checked", lambda ref: ref)
    assert run.verify_owner(owner, "producer", identity, sources, qa_ref, dry_ref, probe_ref) == registration
    workers[1]["reaped"] = False
    with pytest.raises(ValueError, match="reaped worker"):
        run.verify_owner(owner, "producer", identity, sources, qa_ref, dry_ref, probe_ref)
    workers[1]["reaped"] = True
    workers[0]["signal_sent"] = True
    with pytest.raises(ValueError, match="successful reaped worker"):
        run.verify_owner(owner, "producer", identity, sources, qa_ref, dry_ref, probe_ref)


def test_recovery_metrics_use_tcp_pivot_and_real_reached_endpoints():
    native = np.repeat(np.eye(4)[None], 41, axis=0)
    tool = np.eye(4)
    tool[:3, 3] = [0.1, 0.0, 0.2]
    timeline, _ = broader_recovery_timeline(native, np.ones(41), tool, variant="base_yaw", sign=1)
    measured = np.repeat(np.eye(4)[None], 57, axis=0)
    measured[16, :3, :3] = Rotation.from_euler("z", 0.2).as_matrix()
    measured[16, 0, 3] = 0.03
    metrics = run.recovery_metrics(dict(source_pose=measured), tool, timeline)
    displacement = np.array([0.03 + 0.1 * (np.cos(0.2) - 1), 0.1 * np.sin(0.2), 0.0])
    np.testing.assert_allclose(metrics["perturbation"]["measured_tcp_displacement_m"], displacement, atol=1e-12)
    np.testing.assert_allclose(metrics["perturbation"]["measured_base_rotation_vector_rad"], [0, 0, 0.2], atol=1e-12)
    assert metrics["perturbation"]["position_error_m"] == pytest.approx(np.linalg.norm(displacement))
    assert metrics["perturbation"]["rotation_error_rad"] == pytest.approx(0.3)
    assert metrics["return"]["position_error_m"] == pytest.approx(0.0)
    assert metrics["return"]["rotation_error_rad"] == pytest.approx(0.0)
    partial = run.recovery_metrics(dict(source_pose=measured[:16]), tool, timeline)
    assert partial == {"perturbation": None, "return": None}
