"""043 metadata admission attacks using exclusively new synthetic files."""

import ast
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_wam_clean_training_postcapture.py"
SPEC = importlib.util.spec_from_file_location("independent_postcapture", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def ref(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))
    return ref(path)


def text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != content:
        path.write_text(content)
    return ref(path)


def done(schema, **values):
    return dict(schema_version=schema, complete=True, passed=True, **values)


@pytest.fixture(scope="module")
def source_workspace(tmp_path_factory):
    return tmp_path_factory.mktemp("synthetic_postcapture") / "repo"


@pytest.fixture
def scene(source_workspace, monkeypatch):
    # Immutable synthetic sources are reused; all mutable JSON is reconstructed
    # for each case. This avoids thousands of repeated source writes on NFS.
    repo = source_workspace
    repo.mkdir(exist_ok=True)
    monkeypatch.setattr(audit, "REPO", repo)
    monkeypatch.setattr(audit, "VERIFIER", repo / audit.NEW_SOURCES[-2])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    oldroot, root = repo / audit.OLD_REL, repo / audit.NEW_REL
    for relative in ["fit/W1", "review/W1_verified.json", "supervisor/D1/launch.json"]:
        artifact = oldroot / relative
        if artifact.exists():
            artifact.unlink()
    sources = {}
    verifier = text(repo / "scripts/verify_wam_clean_training_continuation.py", "# original synthetic source\n")
    sources[verifier["path"]] = verifier["sha256"]
    for i in range(483):
        entry = text(repo / f"legacy/source_{i}.txt", f"original source {i}\n")
        sources[entry["path"]] = entry["sha256"]
    snapshots, snapshot_refs = {}, {}
    for path, digest in sources.items():
        snapshot = oldroot / "execution/source/repo" / Path(path).relative_to(repo)
        entry = text(snapshot, Path(path).read_text())
        snapshots[str(snapshot)] = digest
        snapshot_refs[path] = entry
    entries, requests = [], []
    for i in range(300):
        original = 1000 + i
        anchor = f"anchor{i // 2}"
        payload = json.dumps(
            ["s2a.clean-training-request-seed31.v1", anchor, 101 + i % 2, original],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode()
        digest = hashlib.sha256(payload).digest()
        seed = int.from_bytes(digest[:4], "big") & ((1 << 31) - 1)
        entries.append(
            dict(
                index=i,
                name=f"train{i}",
                anchor_name=anchor,
                root_seed=101 + i % 2,
                original_sampling_seed=original,
                sampling_seed=seed,
                payload_sha256=digest.hex(),
            )
        )
        requests.append(dict(index=i, name=f"train{i}", sampling_seed=original))
    prepared = dict(
        schema_version="s2a.clean-training-preparation.v1",
        root=str(oldroot),
        snapshot_sha256=copy.deepcopy(snapshots),
        training_requests=requests,
    )
    prepared_ref = write(oldroot / "preparation/prepared.json", prepared)
    metadata_ref = write(
        oldroot / "metadata_verified.json",
        done(
            "s2a.clean-training-preparation-independent.v1",
            prepared_ref=prepared_ref,
            metadata_qualified=True,
            execution_authorized=False,
            execution_ready=False,
            consumed_before_after_exact=True,
        ),
    )
    seed_ref = write(oldroot / "execution/seed_plan.json", done("s2a.clean-training-seed-plan.v1", entries=entries))
    original = done(
        "s2a.clean-training-execution-proposal.v1",
        prepared_ref=prepared_ref,
        execution_authorized=False,
        readiness=dict(ready_for_execution=False),
        automatic_retry=False,
        fresh_performance_authorized=False,
        source_sha256=copy.deepcopy(sources),
        snapshot_sha256=copy.deepcopy(snapshots),
        snapshot_refs=copy.deepcopy(snapshot_refs),
        metadata_verification_ref=metadata_ref,
        training_seed_plan_ref=seed_ref,
    )
    original_ref = write(oldroot / "execution/proposal.json", original)
    verified_ref = write(
        oldroot / "execution/verified.json",
        done(
            "s2a.clean-training-execution-verification.v1",
            prepared_ref=prepared_ref,
            proposal_ref=original_ref,
            eligible_for_execution=True,
            verifier_ref=verifier,
        ),
    )
    old = copy.deepcopy(original)
    old.update(
        schema_version="s2a.clean-training-execution-registration.v1",
        execution_authorized=True,
        proposal_ref=original_ref,
        execution_verification_ref=verified_ref,
    )
    old["readiness"]["ready_for_execution"] = True
    old_ref = write(oldroot / "execution/admitted.json", old)

    def owned(directory, pid, *, code=0, parent=100, model=False, stage=False):
        launch = dict(
            child=dict(pid=pid, start_ticks=pid * 10, ppid=parent),
            command=[str(Path(sys.executable)), "scripts/fit_wam_clean_training.py"],
            selected_environment=dict(CUDA_VISIBLE_DEVICES=audit.GPU if model else "", LD_LIBRARY_PATH=""),
        )
        exit_ = dict(actual_wait=True, returncode=code)
        release = dict(
            pid=pid,
            start_ticks=pid * 10,
            reaped=True,
            returncode=code,
            signal_sent=False,
            original_error=None if code == 0 else "ChildProcessError('command failed with1')",
            only_owned_child_signalled=True,
        )
        refs = dict(
            launch=write(directory / "launch.json", launch),
            exit=write(directory / "exit.json", exit_),
            release=write(directory / ("release.json" if stage else "ownership/release.json"), release),
        )
        return SimpleNamespace(launch=launch, exit=exit_, release=release, refs=refs, directory=directory)

    model = owned(oldroot / "orchestration/D0/model", 201, model=True)
    independent = owned(oldroot / "orchestration/D0/independent", 202)
    gpu_ref = write(
        oldroot / "orchestration/D0/gpu_after.json",
        dict(released=True, uuid=audit.GPU, device_minor=6, memory_mib=0, compute_apps=[]),
    )
    collection = done(
        "s2a.clean-training-capture.v1",
        prepared_ref=prepared_ref,
        execution_registration_ref=old_ref,
        round_name="D0",
        readout_ref=None,
        d0_audit_ref=None,
        actual_requests=300,
        actual_denoise_calls=10500,
        actual_head_calls=10500,
        model_initialization_attempts=1,
        model_initializations=1,
        parameters_before={"unchanged": True},
        parameters_after={"unchanged": True},
        future_label_reads=0,
        producer_fits=0,
        physics_steps=0,
        fresh_cases=0,
        source_sha256=copy.deepcopy(sources),
        training_seed_plan_ref=seed_ref,
        requests=[
            dict(index=i, name=e["name"], sampling_seed=e["sampling_seed"], complete=True)
            for i, e in enumerate(entries)
        ],
    )
    labels = [e["name"] for e in entries for _ in range(35)]
    collection["boundary_calls"] = [dict(index=i, label=label, completed=True) for i, label in enumerate(labels)]
    collection_ref = write(oldroot / "capture/D0/result.json", collection)
    d0audit = done(
        "s2a.clean-training-capture-audit.v1",
        prepared_ref=prepared_ref,
        execution_registration_ref=old_ref,
        collection_ref=collection_ref,
        round_name="D0",
        actual_requests=300,
        actual_denoise_calls=10500,
        actual_online_points_verified=10500,
        changed_active_values=0,
        future_label_reads=0,
        producer_fits=0,
        physics_steps=0,
        fresh_cases=0,
        source_sha256=copy.deepcopy(sources),
        input_sha256={prepared_ref["path"]: prepared_ref["sha256"]},
        worker_launch_ref=model.refs["launch"],
        worker_exit_ref=model.refs["exit"],
        worker_release_ref=model.refs["release"],
        gpu_after_ref=gpu_ref,
        evidence_boundary="inherited saved action mathematics; omitted video remains a hash witness",
    )
    d0audit_ref = write(oldroot / "review/D0_verified.json", d0audit)
    d0owner = done(
        "s2a.clean-training-owner.v1",
        stage="D0",
        prepared_ref=prepared_ref,
        execution_registration_ref=old_ref,
        producer_ref=collection_ref,
        verification_ref=d0audit_ref,
        owned_commands=[model.refs, independent.refs],
        gpu_after_ref=gpu_ref,
    )
    d0owner_ref = write(oldroot / "orchestration/D0/result.json", d0owner)
    d0stage = owned(oldroot / "supervisor/D0", 200, stage=True)
    d0durable_ref = write(
        oldroot / "supervisor/D0/result.json",
        done(
            "s2a.durable-stage-owner.v1",
            stage_result_ref=d0owner_ref,
            source_sha256=copy.deepcopy(sources),
            **{k + "_ref": v for k, v in d0stage.refs.items()},
        ),
    )
    readonly_ref = write(
        repo / "outputs/d0_readonly.json",
        done(
            "s2a.clean-training-d0-readonly-review.v1",
            D0_data_reuse_eligible_with_original_frozen_contract=True,
            fit_rerun_authorized=False,
            prior_independent_math_audit_ref=d0audit_ref,
            D0_owner_ref=d0owner_ref,
            D0_durable_stage_owner_ref=d0durable_ref,
            new_array_decodes=0,
            consumed_metadata_before_after_exact=True,
        ),
    )
    stage = owned(oldroot / "supervisor/W1", 301, code=1, stage=True)
    producer = owned(oldroot / "orchestration/W1/producer", 302, code=1, parent=301)
    stage_value = dict(complete=False, passed=False, **{k + "_ref": v for k, v in stage.refs.items()})
    stage_ref = write(oldroot / "supervisor/W1/result.json", stage_value)
    supervisor_ref = write(
        oldroot / "supervisor/result.json",
        dict(complete=False, passed=False, active_stage="W1", failed_stage_ref=stage_ref),
    )
    log_ref = text(
        oldroot / "orchestration/W1/producer/command.log",
        "context = load_context(\ncheck_execution_proof(\ncheck_file(verifier)\nValueError: fit artifact escaped S2A outputs\n",
    )
    failure = done(
        "s2a.clean-training-w1-pre-fit-failure-audit.v1",
        exception="ValueError: fit artifact escaped S2A outputs",
        failure_before_load_context_return=True,
        original_failure_preserved=True,
        source_before_after_exact=True,
        source_sha256=copy.deepcopy(sources),
        producer_fits=0,
        independent_reconstructions=0,
        future_array_decodes=0,
        new_model_calls=0,
        d1_started=False,
        supervisor_ref=supervisor_ref,
        stage_owner_ref=stage_ref,
        producer_log_ref=log_ref,
        **{"stage_" + k + "_ref": v for k, v in stage.refs.items()},
        **{"producer_" + k + "_ref": v for k, v in producer.refs.items()},
    )
    failure_ref = write(root / "incident/audit.json", failure)
    fixed = dict(
        root=str(root),
        prepared_ref=prepared_ref,
        execution_registration_ref=old_ref,
        d0_collection_ref=collection_ref,
        d0_capture_audit_ref=d0audit_ref,
        d0_owner_ref=d0owner_ref,
        failure_ref=failure_ref,
        d0_readonly_audit_ref=readonly_ref,
        fit_output_dir=str(oldroot / "fit/W1"),
        verification_output=str(oldroot / "review/W1_verified.json"),
        budget=copy.deepcopy(audit.BUDGET),
    )
    config_ref = write(
        repo / audit.CONFIG_REL, dict(protocol="043", stage_scope="W1_only", automatic_retry=False, **fixed)
    )
    plan_ref = text(repo / audit.PLAN_REL, "043 W1-only synthetic metadata plan\n")
    new_sources = copy.deepcopy(sources)
    for relative in audit.NEW_SOURCES:
        entry = text(repo / relative, "# new explicitly synthetic source\n")
        new_sources[entry["path"]] = entry["sha256"]
    for entry in [config_ref, plan_ref]:
        new_sources[entry["path"]] = entry["sha256"]
    new_refs, new_snapshots = copy.deepcopy(snapshot_refs), copy.deepcopy(snapshots)
    for path, digest in new_sources.items():
        if path not in sources:
            entry = text(root / "source/repo" / Path(path).relative_to(repo), Path(path).read_text())
            new_refs[path] = entry
            new_snapshots[entry["path"]] = digest
    dry = dict(
        complete=True,
        passed=True,
        zero_array_decodes=True,
        array_decode_attempts=0,
        expected_rows=420000,
        callback_corrected_count=1,
        source_guard_completed=True,
        source_before_after_exact=True,
        source_verifier_ref=verifier,
        source_sha256=copy.deepcopy(new_sources),
        **{
            k: fixed[k]
            for k in ["prepared_ref", "execution_registration_ref", "d0_collection_ref", "d0_capture_audit_ref"]
        },
    )
    dry_ref = write(root / "metadata_dry.json", dry)
    log_refs = {
        name: text(root / f"qa/{name}.log", content)
        for name, content in [
            ("pytest", "30 passed in 1.0s\n"),
            ("ruff", "All checks passed!\n"),
            ("format", "2 files already formatted\n"),
        ]
    }
    qa = dict(
        complete=True,
        passed=True,
        source_sha256=copy.deepcopy(new_sources),
        tests_exit_code=0,
        ruff_exit_code=0,
        format_exit_code=0,
        tests_passed=30,
        test_paths=[str(repo / audit.NEW_SOURCES[-1])],
        log_refs=log_refs,
    )
    qa_ref = write(root / "qa/qa.json", qa)
    proposal = done(
        "s2a.clean-training-postcapture-proposal.v1",
        execution_authorized=False,
        **fixed,
        config_ref=config_ref,
        plan_ref=plan_ref,
        source_sha256=new_sources,
        snapshot_sha256=new_snapshots,
        source_snapshot_refs=new_refs,
        qa_ref=qa_ref,
        metadata_dry_ref=dry_ref,
    )

    def check():
        return audit.verify_postcapture(write(root / "proposal.json", proposal))

    def replace(key, value):
        target = Path(proposal[key]["path"])
        proposal[key] = write(target, value)
        if key in audit.FIXED:
            config = json.loads(Path(proposal["config_ref"]["path"]).read_text())
            config[key] = proposal[key]
            proposal["config_ref"] = write(Path(proposal["config_ref"]["path"]), config)

    return SimpleNamespace(
        repo=repo,
        root=root,
        oldroot=oldroot,
        sources=sources,
        snapshots=snapshots,
        proposal=proposal,
        check=check,
        replace=replace,
        old=old,
        original=original,
        prepared=prepared,
        collection=collection,
        d0audit=d0audit,
        d0owner=d0owner,
        failure=failure,
        dry=dry,
        qa=qa,
        model=model,
        independent=independent,
        stage=stage,
        producer=producer,
    )


def test_complete_metadata_admission_inherits_math_without_array_loading(scene):
    value = scene.check()
    assert value["eligible_for_execution"] is True and value["execution_authorized"] is False
    assert value["actual_denoise_calls"] == 10500 and value["expected_rows"] == 420000
    assert value["consumed_before_after_exact"] is True and value["future_array_decodes"] == value["new_fits"] == 0
    assert value["fresh_performance_authorized"] is value["goal_complete"] is False


@pytest.mark.parametrize(
    "key,value", [("new_model_initializations", 1), ("producer_fits", 2), ("automatic_retry", True)]
)
def test_scope_cannot_expand_into_models_or_retries(scene, key, value):
    scene.proposal["budget"][key] = value
    with pytest.raises(ValueError, match="scope/budget"):
        scene.check()


def test_old_source_bytes_are_not_replaceable_by_fresh_source_map(scene):
    Path(next(iter(scene.sources))).write_text("changed original source\n")
    with pytest.raises(ValueError, match="SHA differs"):
        scene.check()


def test_old_snapshot_bytes_must_remain_exact(scene):
    Path(next(iter(scene.snapshots))).write_text("changed original snapshot\n")
    with pytest.raises(ValueError, match="SHA differs"):
        scene.check()


def test_true_qa_does_not_authorize_an_unfrozen_source(scene):
    path = str(scene.repo / audit.NEW_SOURCES[0])
    del scene.proposal["source_sha256"][path]
    with pytest.raises(ValueError, match="one-to-one|closure missing"):
        scene.check()


@pytest.mark.parametrize(
    "key,value",
    [
        ("actual_requests", 299),
        ("actual_online_points_verified", 10499),
        ("changed_active_values", 1),
        ("future_label_reads", 1),
    ],
)
def test_d0_capture_audit_denominator_and_label_boundary_are_required(scene, key, value):
    scene.d0audit[key] = value
    scene.replace("d0_capture_audit_ref", scene.d0audit)
    with pytest.raises(ValueError):
        scene.check()


@pytest.mark.parametrize("field,value", [("actual_wait", False), ("returncode", 1)])
def test_d0_model_wait_is_not_a_self_attested_true(scene, field, value):
    scene.model.exit[field] = value
    scene.model.refs["exit"] = write(scene.model.directory / "exit.json", scene.model.exit)
    scene.replace("d0_owner_ref", scene.d0owner)
    with pytest.raises(ValueError):
        scene.check()


@pytest.mark.parametrize("field,value", [("start_ticks", 999999), ("reaped", False), ("signal_sent", True)])
def test_failed_fit_must_have_actual_same_birth_reaped_no_signal_receipts(scene, field, value):
    scene.producer.release[field] = value
    scene.failure["producer_release_ref"] = write(
        scene.producer.directory / "ownership/release.json", scene.producer.release
    )
    scene.replace("failure_ref", scene.failure)
    with pytest.raises(ValueError):
        scene.check()


@pytest.mark.parametrize("target", ["fit/W1", "review/W1_verified.json", "supervisor/D1/launch.json"])
def test_existing_fit_or_later_stage_blocks_metadata_readmission(scene, target):
    text(scene.oldroot / target, "already exists\n")
    with pytest.raises(ValueError, match="already exists|D1 was"):
        scene.check()


@pytest.mark.parametrize(
    "key,value",
    [
        ("zero_array_decodes", False),
        ("array_decode_attempts", 1),
        ("expected_rows", 840000),
        ("callback_corrected_count", 0),
        ("source_guard_completed", False),
    ],
)
def test_dry_requires_real_single_corrected_callback_and_no_decodes(scene, key, value):
    scene.dry[key] = value
    scene.replace("metadata_dry_ref", scene.dry)
    with pytest.raises(ValueError, match="dry proof"):
        scene.check()


def test_dry_must_use_original041_independent_verifier(scene):
    scene.dry["source_verifier_ref"] = ref(scene.repo / audit.NEW_SOURCES[0])
    scene.replace("metadata_dry_ref", scene.dry)
    with pytest.raises(ValueError, match="verifier closure"):
        scene.check()


@pytest.mark.parametrize("key,value", [("tests_exit_code", 1), ("tests_passed", 0), ("ruff_exit_code", 1)])
def test_qa_true_does_not_replace_successful_actual_checks(scene, key, value):
    scene.qa[key] = value
    scene.replace("qa_ref", scene.qa)
    with pytest.raises(ValueError, match="QA exits"):
        scene.check()


def test_qa_reported_count_must_match_the_actual_pytest_log(scene):
    scene.qa["tests_passed"] += 1
    scene.replace("qa_ref", scene.qa)
    with pytest.raises(ValueError, match="actual success count"):
        scene.check()


def test_original_envelope_cannot_gain_unreviewed_payload(scene):
    scene.old["extra_unreviewed_authority"] = True
    scene.replace("execution_registration_ref", scene.old)
    with pytest.raises(ValueError, match="changed its audited proposal"):
        scene.check()


def test_failure_true_cannot_replace_the_actual_pre_fit_trace(scene):
    path = Path(scene.failure["producer_log_ref"]["path"])
    scene.failure["producer_log_ref"] = text(path, "ValueError: another failure after fit\n")
    scene.replace("failure_ref", scene.failure)
    with pytest.raises(ValueError, match="pre-fit source-proof gate"):
        scene.check()


def test_json_parser_refuses_binary_and_duplicate_keys(scene):
    path = scene.root / "current.npz"
    path.write_bytes(b"not a decoded array")
    with pytest.raises(ValueError, match="refuses binary"):
        audit.Evidence().json(ref(path))
    with pytest.raises(ValueError, match="duplicate"):
        audit.parse_json('{"passed":false,"passed":true}')


def test_only_frozen_input_metadata_can_be_hashed_outside_current_repo(scene):
    external = text(scene.repo.parent / "external_readonly/config.yaml", "frozen: metadata\n")
    with pytest.raises(ValueError, match="permitted scope"):
        audit.Evidence().path(external)
    evidence = audit.Evidence()
    evidence.mapping({external["path"]: external["sha256"]}, text_only=True)
    assert evidence.consumed[external["path"]] == external["sha256"]
    with pytest.raises(ValueError, match="deleted root"):
        audit.actual_path("/share/project/eai_pwm/home/hcr/repos/s2a/metadata.json", allow_external=True)


def test_checker_has_no_production_or_array_imports():
    tree = ast.parse(SCRIPT.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [entry.name for entry in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module]
        else:
            continue
        assert not any(name.startswith(("s2a", "numpy", "torch")) for name in names)


def test_cli_help_has_no_side_effects():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
        check=False,
        timeout=10,
    )
    assert result.returncode == 0 and "--proposal-sha256" in result.stdout
