"""The registered physical worker admits only its own immutable case."""

import copy

import pytest

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
