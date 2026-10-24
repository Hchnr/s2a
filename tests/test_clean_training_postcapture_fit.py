"""043 worker regressions use synthetic metadata and mocked math delegates only."""

import copy
import importlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

MAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN / "scripts"))
worker = importlib.import_module("execute_wam_clean_training_postcapture_fit")


def dump(filename, value):
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text(json.dumps(value, sort_keys=True) + "\n")
    return worker.reference(filename)


def read(ref):
    return json.loads(Path(ref["path"]).read_text())


def seal(fixture):
    proposal_ref = dump(fixture.root / "admission/proposal.json", fixture.proposal)
    verification = copy.deepcopy(fixture.verification)
    verification["proposal_ref"] = proposal_ref
    verification_ref = dump(fixture.root / "admission/verified.json", verification)
    registration = copy.deepcopy(fixture.proposal)
    registration.update(
        schema_version=worker.REGISTRATION_SCHEMA,
        execution_authorized=True,
        proposal_ref=proposal_ref,
        independent_verification_ref=verification_ref,
    )
    fixture.registration = registration
    fixture.ref = dump(fixture.root / "admission/admitted.json", registration)
    return fixture.ref


@pytest.fixture
def admission(tmp_path, monkeypatch):
    assert tmp_path.resolve().is_relative_to(MAIN / "outputs")
    repo = tmp_path / "synthetic_repo"
    repo.mkdir()
    monkeypatch.setattr(worker, "REPO", repo)
    monkeypatch.chdir(repo)
    for name, value in {
        "CUDA_VISIBLE_DEVICES": "",
        "LD_LIBRARY_PATH": "",
        "OMP_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("metadata worker must not decode any array"))
    root, original_root = repo / "outputs/repair043", repo / "outputs/original041"
    old_paths = list(dict.fromkeys((*worker._caller_sources("producer"), *worker._caller_sources("independent"))))
    old_verifier = repo / "scripts/verify_wam_clean_training_continuation.py"
    old_paths.append(old_verifier)
    old_paths.extend(repo / f"src/s2a/frozen_{i:03d}.py" for i in range(484 - len(old_paths)))
    old_sources = {}
    for source in old_paths:
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("# Synthetic immutable041 source. No execution.\n")
        old_sources[str(source)] = worker.sha(source)
    sources = dict(old_sources)
    for relative in (
        "scripts/execute_wam_clean_training_postcapture_fit.py",
        "tests/test_clean_training_postcapture_fit.py",
        "src/s2a/clean_training_source_proof.py",
        "scripts/verify_synthetic043.py",
    ):
        source = repo / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"# Synthetic043 source: {relative}\n")
        sources[str(source)] = worker.sha(source)
    snapshots = {}
    for digest in set(sources.values()):
        source = next(Path(p) for p, d in sources.items() if d == digest)
        snapshot = root / "source" / (digest + ".txt")
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(source.read_bytes())
        snapshots[str(snapshot)] = digest
    prepared_ref = dump(original_root / "preparation/prepared.json", dict(root=str(original_root), synthetic=True))
    old_verifier_ref = worker.reference(old_verifier)
    old_verified_ref = dump(original_root / "execution/verified.json", dict(verifier_ref=old_verifier_ref))
    old_ref = dump(
        original_root / "execution/admitted.json",
        dict(
            schema_version="s2a.clean-training-execution-registration.v1",
            complete=True,
            passed=True,
            execution_authorized=True,
            automatic_retry=False,
            fresh_performance_authorized=False,
            prepared_ref=prepared_ref,
            execution_verification_ref=old_verified_ref,
            source_sha256=old_sources,
        ),
    )
    d0 = dict(
        complete=True, passed=True, round_name="D0", prepared_ref=prepared_ref, execution_registration_ref=old_ref
    )
    collection_ref = dump(
        original_root / "capture/D0/result.json", dict(d0, schema_version="s2a.clean-training-capture.v1")
    )
    audit_ref = dump(
        original_root / "review/D0_verified.json",
        dict(
            d0,
            schema_version="s2a.clean-training-capture-audit.v1",
            collection_ref=collection_ref,
            actual_requests=300,
            actual_denoise_calls=10500,
            actual_online_points_verified=10500,
        ),
    )
    owner_ref = dump(
        original_root / "orchestration/D0/result.json",
        dict(
            schema_version="s2a.clean-training-owner.v1",
            complete=True,
            passed=True,
            stage="D0",
            prepared_ref=prepared_ref,
            execution_registration_ref=old_ref,
            producer_ref=collection_ref,
            verification_ref=audit_ref,
        ),
    )
    failure_ref = dump(
        root / "incident/audit.json",
        dict(
            schema_version="s2a.clean-training-w1-pre-fit-failure-audit.v1",
            complete=True,
            passed=True,
            failure_before_load_context_return=True,
            producer_fits=0,
            independent_reconstructions=0,
            future_array_decodes=0,
            new_model_calls=0,
            d1_started=False,
            original_failure_preserved=True,
            source_before_after_exact=True,
        ),
    )
    fixture = SimpleNamespace(repo=repo, root=root, original_root=original_root, old_verifier_ref=old_verifier_ref)
    fixture.proposal = dict(
        schema_version=worker.PROPOSAL_SCHEMA,
        complete=True,
        passed=True,
        execution_authorized=False,
        root=str(root),
        prepared_ref=prepared_ref,
        execution_registration_ref=old_ref,
        d0_collection_ref=collection_ref,
        d0_capture_audit_ref=audit_ref,
        d0_owner_ref=owner_ref,
        failure_ref=failure_ref,
        fit_output_dir=str(original_root / "fit/W1"),
        verification_output=str(original_root / "review/W1_verified.json"),
        source_sha256=sources,
        snapshot_sha256=snapshots,
        budget=copy.deepcopy(worker.BUDGET),
    )
    fixture.verification = dict(
        schema_version=worker.VERIFICATION_SCHEMA,
        complete=True,
        passed=True,
        eligible_for_execution=True,
        verifier_ref=worker.reference(repo / "scripts/verify_synthetic043.py"),
    )
    seal(fixture)
    return fixture


@pytest.fixture
def mock_math(admission, monkeypatch):
    state = SimpleNamespace(calls=[], contexts=[], load_roles=[], active=None, fail=None, mutate=None, callback_count=1)

    @contextmanager
    def proof_context(old_ref):
        assert old_ref == admission.registration["execution_registration_ref"]
        proof = SimpleNamespace(
            registration_ref=old_ref,
            source_verifier_ref=admission.old_verifier_ref,
            callback_corrected_count=0,
            source_guard_completed=False,
        )
        assert state.active is None
        state.active = proof
        try:
            yield proof
        finally:
            proof.source_guard_completed = True
            state.active = None

    def load_context(prepared, old, **kwargs):
        state.contexts.append((prepared, old, kwargs))
        state.active.callback_corrected_count += state.callback_count
        return SimpleNamespace(
            stage="W1", rounds=("D0",), expected_rows=420000, evidence=SimpleNamespace(finish=lambda: None)
        )

    def delegate(role):
        def invoke(prepared, old, *args, **kwargs):
            state.calls.append((role, prepared, old, args, kwargs))
            state.active.callback_corrected_count += state.callback_count
            if state.fail == "before":
                raise RuntimeError("synthetic pre-fit failure")
            reg = admission.registration
            if role == "producer":
                directory = Path(kwargs["output_dir"])
                directory.mkdir(parents=True, exist_ok=False)
                params_ref = dump(directory / "parameters.json", dict(synthetic=True, coefficients=[1]))
                payload = dict(
                    schema_version="s2a.clean-training-fit.v1",
                    complete=state.fail != "after",
                    passed=state.fail != "after",
                    stage="W1",
                    prepared_ref=prepared,
                    execution_registration_ref=old,
                    collection_refs=kwargs["collection_refs"],
                    capture_audit_refs=kwargs["capture_audit_refs"],
                    rounds=["D0"],
                    producer_fits=1,
                    fit_attempts=1,
                    actual_rows=420000,
                    params_ref=params_ref,
                )
                result_ref = dump(directory / "result.json", payload)
            else:
                assert len(args) == 1
                produced = read(args[0])
                result_ref = dump(
                    Path(kwargs["output"]),
                    dict(
                        schema_version="s2a.clean-training-fit-verification.v1",
                        complete=True,
                        passed=True,
                        stage="W1",
                        prepared_ref=prepared,
                        execution_registration_ref=old,
                        fit_ref=args[0],
                        params_ref=produced["params_ref"],
                        actual_rows=420000,
                        independent_reconstructions=1,
                        full_training_row_reconstruction=True,
                    ),
                )
            if state.mutate:
                state.mutate(result_ref, reg)
                result_ref = worker.reference(result_ref["path"])
            if state.fail == "after":
                state.original_failure_bytes = Path(result_ref["path"]).read_bytes()
                raise RuntimeError("synthetic failure after original result")
            return result_ref

        return invoke

    def load(role, registration):
        state.load_roles.append(role)
        assert registration == admission.registration
        return (
            delegate(role),
            SimpleNamespace(execution_source_proof=proof_context),
            SimpleNamespace(load_context=load_context),
            worker.reference(admission.repo / "scripts" / (worker.MODULES[role][0] + ".py")),
            worker.reference(admission.repo / "src/s2a/clean_training_source_proof.py"),
        )

    monkeypatch.setattr(worker, "_load_modules", load)
    return state


def test_metadata_admit_returns_new_registration_without_module_import(admission, monkeypatch):
    monkeypatch.setattr(worker.importlib, "import_module", lambda *a: pytest.fail("admit may not import delegates"))
    assert worker.admit(admission.ref) == admission.registration
    assert not (admission.root / "workers").exists()
    assert not Path(admission.proposal["fit_output_dir"]).exists()


@pytest.mark.parametrize("role", ["producer", "independent"])
def test_dry_calls_complete_metadata_once_without_fit_or_output(admission, mock_math, role):
    result = worker.execute(admission.ref, role=role, dry_run=True)
    assert result["schema_version"] == worker.DRY_SCHEMA
    assert result["complete"] is result["passed"] is result["metadata_only"] is True
    assert result["expected_rows"] == 420000
    assert result["source_proof"]["source_verifier_ref"] == admission.old_verifier_ref
    assert result["source_proof"]["callback_corrected_count"] == 1
    assert mock_math.load_roles == [role]
    assert mock_math.calls == []
    assert len(mock_math.contexts) == 1
    prepared, old, kwargs = mock_math.contexts[0]
    assert prepared == admission.registration["prepared_ref"]
    assert old == admission.registration["execution_registration_ref"]
    assert kwargs == dict(
        stage="W1",
        collection_refs=[admission.registration["d0_collection_ref"]],
        capture_audit_refs=[admission.registration["d0_capture_audit_ref"]],
        caller_sources=worker._caller_sources(role),
    )
    assert not (admission.root / "workers").exists()
    assert not Path(admission.proposal["fit_output_dir"]).exists()
    assert not Path(admission.proposal["verification_output"]).exists()
    assert result["producer_fits"] == result["independent_reconstructions"] == 0


def test_producer_then_independent_fixed_arguments_and_original_json_preserved(admission, mock_math):
    producer_ref = worker.execute(admission.ref, role="producer")
    producer = read(producer_ref)
    original_bytes = Path(producer["delegate_result_ref"]["path"]).read_bytes()
    independent_ref = worker.execute(admission.ref, role="independent")
    independent = read(independent_ref)
    assert mock_math.load_roles == ["producer", "independent"]
    assert producer["complete"] is producer["passed"] is independent["passed"] is True
    assert producer["producer_fits"] == independent["independent_reconstructions"] == 1
    assert producer["independent_reconstructions"] == independent["producer_fits"] == 0
    assert producer["delegate_calls_attempted"] == independent["delegate_calls_attempted"] == 1
    for result in (producer, independent):
        assert result["repair_ref"] == admission.ref
        assert result["source_proof"]["source_verifier_ref"] == admission.old_verifier_ref
        assert result["source_proof"]["callback_corrected_count"] == 1
        assert result["source_proof"]["source_guard_completed"] is True
        assert result["source_before_after_exact"] is True
        assert all(
            result[k] == 0
            for k in ("new_model_initializations", "new_denoise_calls", "new_physics_steps", "new_fresh_cases")
        )
        assert result["automatic_retry"] is False
    assert mock_math.calls[0][4] == dict(
        stage="W1",
        collection_refs=[admission.registration["d0_collection_ref"]],
        capture_audit_refs=[admission.registration["d0_capture_audit_ref"]],
        output_dir=admission.registration["fit_output_dir"],
    )
    assert mock_math.calls[1][3] == (producer["delegate_result_ref"],)
    assert mock_math.calls[1][4] == dict(stage="W1", output=admission.registration["verification_output"])
    assert Path(producer["delegate_result_ref"]["path"]).read_bytes() == original_bytes
    assert read(producer["delegate_result_ref"])["schema_version"] == "s2a.clean-training-fit.v1"
    assert read(independent["delegate_result_ref"])["schema_version"] == "s2a.clean-training-fit-verification.v1"


@pytest.mark.parametrize("role", ["producer", "independent"])
def test_exclusive_no_retry(admission, mock_math, role):
    worker.execute(admission.ref, role="producer")
    if role == "independent":
        worker.execute(admission.ref, role=role)
    calls = len(mock_math.calls)
    with pytest.raises((ValueError, FileExistsError), match="already exists|no retry|File exists"):
        worker.execute(admission.ref, role=role)
    assert len(mock_math.calls) == calls


@pytest.mark.parametrize("when", ["before", "after"])
def test_delegate_failure_never_becomes_passed_despite_source_guard(admission, mock_math, when):
    mock_math.fail = when
    with pytest.raises(RuntimeError, match="synthetic"):
        worker.execute(admission.ref, role="producer")
    result = read(worker.reference(admission.root / "workers/producer/result.json"))
    assert result["complete"] is result["passed"] is False
    assert result["delegate_calls_attempted"] == 1
    assert result["producer_fits"] == 0
    assert result["delegate_error"]["type"] == "RuntimeError"
    assert result["source_proof"]["source_guard_completed"] is True
    assert result["source_before_after_exact"] is True
    assert result["delegate_result_ref"] is None
    if when == "after":
        result_path = Path(result["available_delegate_result_ref"]["path"])
        assert result_path.read_bytes() == mock_math.original_failure_bytes
        assert read(result["available_delegate_result_ref"])["passed"] is False
    with pytest.raises((ValueError, FileExistsError)):
        worker.execute(admission.ref, role="producer")
    assert len(mock_math.calls) == 1


@pytest.mark.parametrize("count", [0, 2])
def test_exactly_one_corrected_callback_is_required(admission, mock_math, count):
    mock_math.callback_count = count
    with pytest.raises(ValueError, match="callback"):
        worker.execute(admission.ref, role="producer")
    result = read(worker.reference(admission.root / "workers/producer/result.json"))
    assert result["complete"] is result["passed"] is False
    assert result["available_delegate_result_ref"]


def test_source_drift_after_original_output_fails_guard(admission, mock_math):
    frozen = admission.repo / "src/s2a/geometry.py"
    mock_math.mutate = lambda *args: frozen.write_text("# synthetic source drift\n")
    with pytest.raises(ValueError, match="changed during043"):
        worker.execute(admission.ref, role="producer")
    result = read(worker.reference(admission.root / "workers/producer/result.json"))
    assert result["complete"] is result["passed"] is result["source_before_after_exact"] is False
    assert result["final_guard_error"]
    assert result["available_delegate_result_ref"]


def test_independent_needs_same_repair_producer_wrapper(admission, mock_math):
    with pytest.raises(FileNotFoundError):
        worker.execute(admission.ref, role="independent")
    assert mock_math.calls == []
    worker.execute(admission.ref, role="producer")
    filename = admission.root / "workers/producer/result.json"
    value = json.loads(filename.read_text())
    value["repair_ref"] = admission.registration["execution_registration_ref"]
    dump(filename, value)
    with pytest.raises(ValueError, match="another043"):
        worker.execute(admission.ref, role="independent")
    assert len(mock_math.calls) == 1


@pytest.mark.parametrize(
    "field,bad", [("actual_rows", 419999), ("stage", "W2"), ("fit_attempts", 2), ("producer_fits", 0)]
)
def test_original_producer_output_must_keep_w1_denominator(admission, mock_math, field, bad):
    def corrupt(ref, reg):
        payload = read(ref)
        payload[field] = bad
        dump(Path(ref["path"]), payload)

    mock_math.mutate = corrupt
    with pytest.raises(ValueError, match="lineage/denominator"):
        worker.execute(admission.ref, role="producer")


@pytest.mark.parametrize(
    "field,bad",
    [
        ("actual_rows", 419999),
        ("full_training_row_reconstruction", False),
        ("independent_reconstructions", 2),
        ("prepared_ref", {}),
    ],
)
def test_independent_output_must_bind_full_reconstruction(admission, mock_math, field, bad):
    worker.execute(admission.ref, role="producer")

    def corrupt(ref, reg):
        payload = read(ref)
        payload[field] = bad
        dump(Path(ref["path"]), payload)

    mock_math.mutate = corrupt
    with pytest.raises(ValueError, match="lineage/denominator"):
        worker.execute(admission.ref, role="independent")


@pytest.mark.parametrize(
    "field,bad",
    [
        ("execution_authorized", True),
        ("complete", False),
        ("budget", dict(worker.BUDGET, producer_fits=2)),
        ("budget", dict(worker.BUDGET, automatic_retry=True)),
        ("automatic_retry", True),
        ("fit_output_dir", "escape"),
        ("verification_output", "escape"),
        ("root", "escape"),
        ("snapshot_sha256", {}),
    ],
)
def test_proposal_rejects_extra_budget_or_path_or_retry(admission, mock_math, field, bad):
    admission.proposal[field] = bad
    seal(admission)
    with pytest.raises((ValueError, KeyError)):
        worker.execute(admission.ref, role="producer", dry_run=True)
    assert mock_math.calls == mock_math.contexts == []


@pytest.mark.parametrize(
    "field,bad", [("passed", False), ("complete", False), ("eligible_for_execution", False), ("verifier_ref", {})]
)
def test_new_independent_envelope_required(admission, field, bad):
    admission.verification[field] = bad
    seal(admission)
    with pytest.raises((ValueError, KeyError)):
        worker.admit(admission.ref)


def test_registration_is_exact_proposal_copy_and_old041_is_not_authorization(admission):
    with pytest.raises(ValueError, match="new043"):
        worker.admit(admission.registration["execution_registration_ref"])
    registration = copy.deepcopy(admission.registration)
    registration["unexpected_mutation"] = True
    new_ref = dump(Path(admission.ref["path"]), registration)
    with pytest.raises(ValueError, match="exact proposal"):
        worker.admit(new_ref)


@pytest.mark.parametrize(
    "relative",
    [
        "src/s2a/geometry.py",
        "src/s2a/clean_training_source_proof.py",
        "scripts/execute_wam_clean_training_postcapture_fit.py",
    ],
)
def test_frozen_old_and_new_source_drift_rejected_before_delegates(admission, mock_math, relative):
    (admission.repo / relative).write_text("# changed\n")
    with pytest.raises(ValueError, match="SHA"):
        worker.execute(admission.ref, role="producer")
    assert mock_math.load_roles == mock_math.calls == []


def test_missing_old484_source_and_missing_helper_rejected(admission):
    key = str(admission.repo / "src/s2a/geometry.py")
    del admission.proposal["source_sha256"][key]
    seal(admission)
    with pytest.raises(ValueError, match="omitted"):
        worker.admit(admission.ref)
    admission.proposal["source_sha256"][key] = worker.sha(key)
    del admission.proposal["source_sha256"][str(admission.repo / "src/s2a/clean_training_source_proof.py")]
    seal(admission)
    with pytest.raises(ValueError, match="not043-frozen"):
        worker.admit(admission.ref)


def test_d0_bad_audit_and_failure_evidence_rejected(admission):
    audit = read(admission.proposal["d0_capture_audit_ref"])
    audit["actual_denoise_calls"] = 10499
    admission.proposal["d0_capture_audit_ref"] = dump(Path(admission.proposal["d0_capture_audit_ref"]["path"]), audit)
    owner = read(admission.proposal["d0_owner_ref"])
    owner["verification_ref"] = admission.proposal["d0_capture_audit_ref"]
    admission.proposal["d0_owner_ref"] = dump(Path(admission.proposal["d0_owner_ref"]["path"]), owner)
    seal(admission)
    with pytest.raises(ValueError, match="denominator"):
        worker.admit(admission.ref)
    audit["actual_denoise_calls"] = 10500
    admission.proposal["d0_capture_audit_ref"] = dump(Path(admission.proposal["d0_capture_audit_ref"]["path"]), audit)
    owner["verification_ref"] = admission.proposal["d0_capture_audit_ref"]
    admission.proposal["d0_owner_ref"] = dump(Path(admission.proposal["d0_owner_ref"]["path"]), owner)
    failure = read(admission.proposal["failure_ref"])
    failure["future_array_decodes"] = 1
    admission.proposal["failure_ref"] = dump(Path(admission.proposal["failure_ref"]["path"]), failure)
    seal(admission)
    with pytest.raises(ValueError, match="pre-fit/pre-decode"):
        worker.admit(admission.ref)


@pytest.mark.parametrize(
    "key,value", [("CUDA_VISIBLE_DEVICES", "0"), ("LD_LIBRARY_PATH", "/bad"), ("OMP_NUM_THREADS", "1")]
)
def test_cpu_environment_is_explicit(admission, monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="CPU thread"):
        worker.admit(admission.ref)


def test_cwd_guard_and_symlink_escape(admission, monkeypatch):
    monkeypatch.chdir(admission.root)
    with pytest.raises(ValueError, match="main S2A cwd"):
        worker.admit(admission.ref)
    monkeypatch.chdir(admission.repo)
    alias = admission.root / "alias.json"
    alias.symlink_to(admission.ref["path"])
    with pytest.raises(ValueError, match="nonsymlink"):
        worker.admit(dict(path=str(alias), sha256=admission.ref["sha256"]))


def test_lazy_independent_import_does_not_import_producer_and_unwraps_helper(admission, monkeypatch):
    modules = {}
    for name, relative, text in (
        (
            "verify_wam_clean_training_fit",
            "scripts/verify_wam_clean_training_fit.py",
            "def verify_fit(*a, **k): pass\n",
        ),
        (
            "s2a.clean_training_source_proof",
            "src/s2a/clean_training_source_proof.py",
            "from contextlib import contextmanager\n@contextmanager\ndef execution_source_proof(ref):\n    yield None\n",
        ),
        (
            "s2a.clean_training_fit_contract",
            "src/s2a/clean_training_fit_contract.py",
            "# Synthetic metadata-only contract.\n",
        ),
    ):
        filename = admission.repo / relative
        filename.write_text(text)
        module = ModuleType(name)
        module.__file__ = str(filename)
        exec(compile(text, str(filename), "exec"), module.__dict__)
        modules[name] = module
        admission.registration["source_sha256"][str(filename)] = worker.sha(filename)
    imported = []

    def load(name):
        imported.append(name)
        assert name != "fit_wam_clean_training"
        return modules[name]

    monkeypatch.setattr(worker.importlib, "import_module", load)
    result = worker._load_modules("independent", admission.registration)
    assert result[0] is modules["verify_wam_clean_training_fit"].verify_fit
    assert imported == [
        "verify_wam_clean_training_fit",
        "s2a.clean_training_source_proof",
        "s2a.clean_training_fit_contract",
    ]
    modules["verify_wam_clean_training_fit"].__file__ = str(admission.repo / "outputs/fake_checker.py")
    with pytest.raises(ValueError, match="original main-repository"):
        worker._load_modules("independent", admission.registration)


def test_help_is_fresh_cpu_process_and_cli_has_no_delegate_overrides():
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        LD_LIBRARY_PATH="",
        OMP_NUM_THREADS="8",
        OPENBLAS_NUM_THREADS="8",
        MKL_NUM_THREADS="8",
        PYTHONDONTWRITEBYTECODE="1",
    )
    command = [sys.executable, str(MAIN / "scripts/execute_wam_clean_training_postcapture_fit.py")]
    result = subprocess.run([*command, "--help"], cwd=MAIN, env=env, text=True, capture_output=True, check=True)
    assert "--repair-registration" in result.stdout and "--dry-run" in result.stdout
    result = subprocess.run(
        [
            *command,
            "--repair-registration",
            "/unused.json",
            "--repair-registration-sha256",
            "0" * 64,
            "--role",
            "producer",
            "--seed",
            "42",
            "--stage",
            "W2",
        ],
        cwd=MAIN,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "unrecognized arguments: --stage W2" in result.stderr
