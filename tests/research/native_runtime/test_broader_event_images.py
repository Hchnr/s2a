"""Admission, physical receipts and real H40 bindings for event-image replay."""

import copy
import json

import numpy as np
import pytest

from scripts.simulation.runtime import run_broader_event_images as run


@pytest.fixture
def registration():
    return dict(
        phase="replay",
        eligible_cases=list(range(24)),
        source_refs=[dict(path="source", sha256="digest")],
        inputs=dict(cases=[dict(index=index) for index in range(24)]),
        automatic_retry=False,
    )


def test_worker_requires_exact_registered_case_roster_and_sources(registration):
    assert run.validate_registration("replay", 7, registration, registration["source_refs"]) == dict(index=7)
    for mutate in (
        lambda row: row["eligible_cases"].remove(7),
        lambda row: row["inputs"]["cases"][7].update(index=8),
        lambda row: row.update(automatic_retry=True),
        lambda row: row["source_refs"][0].update(sha256="changed"),
    ):
        changed = copy.deepcopy(registration)
        mutate(changed)
        with pytest.raises(ValueError, match="roster and source"):
            run.validate_registration("replay", 7, changed, registration["source_refs"])
    for index in (-1, 24, True):
        with pytest.raises(ValueError, match="One registered"):
            run.validate_registration("replay", index, registration, registration["source_refs"])
    probe = copy.deepcopy(registration)
    probe.update(phase="probe", eligible_cases=[0])
    assert run.validate_registration("probe", 0, probe, registration["source_refs"]) == dict(index=0)
    with pytest.raises(ValueError, match="One registered"):
        run.validate_registration("probe", 1, probe, registration["source_refs"])


def test_preflight_requires_source_matched_64_workers_and_no_physics(monkeypatch):
    identity, sources, hashes = dict(cases=[]), ["source"], {"source": "digest"}
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
    monkeypatch.setattr(run.stage, "load_inputs", lambda: ({}, identity))
    assert run.preflight()[1] == identity
    for key, invalid in (("configured_parallelism", 32), ("reaped", False), ("skipped_count", 1)):
        original = qa[key]
        qa[key] = invalid
        with pytest.raises(ValueError, match="source-matched"):
            run.preflight()
        qa[key] = original
    dry["gpu_calls"] = 1
    with pytest.raises(ValueError, match="zero-physics"):
        run.preflight()
    dry["gpu_calls"] = 0
    dry["inputs"] = dict(cases=[1])
    with pytest.raises(ValueError, match="Dry identity"):
        run.preflight()


def test_probe_admission_requires_exact_registration_and_reaped_worker(monkeypatch):
    identity, sources, qa_ref, dry_ref = dict(cases=[dict(index=0)]), ["source"], "qa", "dry"
    child = dict(key="0", returncode=0, actual_wait=True, reaped=True, signal_sent=False, log_ref="log")
    registration = dict(
        phase="probe",
        eligible_cases=[0],
        inputs=identity,
        source_refs=sources,
        qa_ref=qa_ref,
        dry_ref=dry_ref,
        probe_ref=None,
        automatic_retry=False,
    )
    entry = dict(status="closed", branch_ref="branch", available_branch_ref="branch", counts={}, images=[])
    probe = dict(
        complete=True,
        passed=True,
        owner_closed=True,
        sources_unchanged=True,
        resource_counts_complete=True,
        inputs=identity,
        source_refs=sources,
        qa_ref=qa_ref,
        dry_ref=dry_ref,
        registration_ref="registration",
        lifecycle_ref="lifecycle",
        cases={"0": entry},
    )
    objects = {"lifecycle": dict(launch_error=None, workers=[child]), "registration": registration, "branch": {}}
    monkeypatch.setattr(run.prefix, "read", lambda ref: objects[ref])
    monkeypatch.setattr(run.prefix, "checked", lambda ref: ref)
    monkeypatch.setattr(run.legacy, "validate_worker", lambda *_args: {})
    monkeypatch.setattr(run.legacy, "verify_images", lambda *_args: [])
    run.verify_probe(probe, identity, sources, qa_ref, dry_ref)
    child["reaped"] = False
    with pytest.raises(ValueError, match="reaped"):
        run.verify_probe(probe, identity, sources, qa_ref, dry_ref)
    child["reaped"] = True
    registration["automatic_retry"] = True
    with pytest.raises(ValueError, match="registered probe"):
        run.verify_probe(probe, identity, sources, qa_ref, dry_ref)


def test_replay_owner_requires_every_registered_child_to_close(monkeypatch):
    identity, sources, qa_ref, dry_ref, probe_ref = dict(cases=[]), ["source"], "qa", "dry", "probe"
    registration = dict(
        phase="replay",
        inputs=identity,
        source_refs=sources,
        qa_ref=qa_ref,
        dry_ref=dry_ref,
        probe_ref=probe_ref,
        eligible_cases=list(range(24)),
        automatic_retry=False,
    )
    workers = [
        dict(key=str(i), returncode=0, actual_wait=True, reaped=True, signal_sent=False, log_ref=f"log{i}")
        for i in range(24)
    ]
    cases = {
        str(i): dict(status="closed", branch_ref=f"branch{i}", available_branch_ref=f"branch{i}") for i in range(24)
    }
    owner = dict(
        complete=True,
        passed=True,
        owner_closed=True,
        sources_unchanged=True,
        resource_counts_complete=True,
        phase="replay",
        automatic_retry=False,
        inputs=identity,
        source_refs=sources,
        qa_ref=qa_ref,
        dry_ref=dry_ref,
        registration_ref="registration",
        lifecycle_ref="lifecycle",
        cases=cases,
    )
    objects = {"registration": registration, "lifecycle": dict(workers=workers, launch_error=None)}
    monkeypatch.setattr(run.prefix, "read", lambda ref: objects[ref])
    monkeypatch.setattr(run.prefix, "checked", lambda ref: ref)
    run.verify_replay_owner(owner, identity, sources, qa_ref, dry_ref, probe_ref)
    workers[13]["reaped"] = False
    with pytest.raises(ValueError, match="reaped child"):
        run.verify_replay_owner(owner, identity, sources, qa_ref, dry_ref, probe_ref)
    workers[13]["reaped"] = True
    del owner["cases"]["13"]
    with pytest.raises(ValueError, match="complete replay roster"):
        run.verify_replay_owner(owner, identity, sources, qa_ref, dry_ref, probe_ref)


def test_real_h40_matches_recorded_candidate_and_rejects_changed_labels():
    config = json.loads(run.stage.CONFIG.read_text())
    refs = {key: run.preparation.configured_ref(config[key]) for key in run.stage.REFERENCES}
    recovery = run.prefix.read(refs["recovery_audit"])
    events = run.prefix.read(refs["event_audit"])
    candidate = events["cases"][0]["candidates"][0]
    spec = recovery["cases"][0]["case"]
    producer = run.prefix.read(recovery["cases"][0]["producer_ref"])
    shared = run.preparation.shared_inputs()
    source = run.preparation.case_inputs(spec["original_identity"], shared)
    normalization = run.prefix.read(refs["normalization"])
    with np.load(run.prefix.checked(producer["arrays_ref"]), allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]).copy() for key in archive.files}
    label = run.verify_label(candidate, data, source["case"]["tool"], normalization)
    assert label["normalized_sha256"] == candidate["normalized_sha256"]
    assert label["clipped_count"] == 3
    changed = copy.deepcopy(candidate)
    changed["normalized_sha256"] = "changed"
    with pytest.raises(ValueError, match="original H40"):
        run.verify_label(changed, data, source["case"]["tool"], normalization)
    changed = copy.deepcopy(candidate)
    changed["clipped_count"] = 0
    with pytest.raises(ValueError, match="original H40"):
        run.verify_label(changed, data, source["case"]["tool"], normalization)


def test_partial_workers_keep_unknown_costs_and_all_registered_cases(tmp_path, monkeypatch):
    identity = dict(cases=[dict(index=i, replay_periods=10) for i in range(24)])
    config = dict(
        worker_timeout_seconds=30,
        max_replay_control_periods=240,
        max_replay_inner_steps=6000,
        max_replay_models=72,
        max_replay_contexts=72,
    )
    sources, before, published = ["source"], {"source": "digest"}, {}
    monkeypatch.setattr(run, "OUTPUT", tmp_path)
    monkeypatch.setattr(run, "preflight", lambda: (config, identity, before, sources, "qa", "dry"))
    monkeypatch.setattr(run, "verify_probe", lambda *_args: None)
    monkeypatch.setattr(run, "source_refs", lambda: sources)
    monkeypatch.setattr(run, "snapshot", lambda _root: dict(sources=before))
    monkeypatch.setattr(run.stage, "load_inputs", lambda: (config, identity))
    monkeypatch.setattr(run.prefix, "reference", lambda path: str(path))

    def publish(path, value):
        ref = str(path)
        published[ref] = copy.deepcopy(value)
        return ref

    def read(ref):
        if ref.endswith("probe/result.json"):
            return {}
        return published[ref] if ref in published else json.loads(run.Path(ref).read_text())

    def launch(specs, *, cwd, timeout_seconds):
        assert len(specs) == 24 and timeout_seconds == 30
        children = []
        for index, spec in enumerate(specs):
            if index < 2:
                path = tmp_path / "replay" / f"case{index:02d}" / "worker/result.json"
                path.parent.mkdir(parents=True)
                counters = {
                    key: dict(attempted=count, completed=count)
                    for key, count in (("mj_step", 50), ("mj_step2", 0), ("model_construct", 3), ("context", 3))
                }
                path.write_text(
                    json.dumps(
                        dict(runtime=dict(control_steps_attempted=2, control_steps_completed=2, counters=counters))
                    )
                )
            children.append(
                dict(
                    key=str(index),
                    returncode=0 if index == 0 else 1,
                    actual_wait=True,
                    reaped=True,
                    signal_sent=False,
                    log=str(spec["log"]),
                )
            )
        return dict(workers=children, launch_error=None)

    monkeypatch.setattr(run, "publish_result", publish)
    monkeypatch.setattr(run.prefix, "read", read)
    monkeypatch.setattr(run, "run_owned_workers", launch)
    monkeypatch.setattr(run.legacy, "validate_worker", lambda *_args: dict(control_steps=2, inner_steps=50, images=1))
    monkeypatch.setattr(run.legacy, "verify_images", lambda *_args: [])
    with pytest.raises(SystemExit, match="no automatic retry"):
        run.phase_run("replay")
    result = published[str(tmp_path / "replay/result.json")]
    assert result["owner_closed"] is result["sources_unchanged"] is result["complete"] is True
    assert result["passed"] is result["resource_counts_complete"] is False
    assert len(result["cases"]) == 24
    assert result["cases"]["0"]["status"] == "closed"
    assert result["cases"]["1"]["observed_resources"]["control_steps_completed"] == 2
    assert result["cases"]["2"]["observed_resources"] is None
    assert result["observed_resource_totals"]["control_steps_completed"] == 4
