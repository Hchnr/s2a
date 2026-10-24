"""Synthetic proof regression only: no fit, QR, physics, model or future reads."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from s2a import clean_execution_proof as generic
from s2a import clean_training_fit_contract as contract
from s2a import clean_training_source_proof as repair
from s2a.artifacts import REPO


def dump(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return contract.reference(path)


@pytest.fixture
def admission(tmp_path, monkeypatch):
    assert tmp_path.resolve().is_relative_to(REPO / "outputs")
    for name, value in {
        "CUDA_VISIBLE_DEVICES": "",
        "LD_LIBRARY_PATH": "",
        "OMP_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("proof repair must never decode arrays"))
    source = tmp_path / "synthetic_execution_verifier.py"
    source.write_text("# Synthetic execution verifier, no actual qualification.\n")
    prepared_ref = dump(tmp_path / "prepared.json", dict(synthetic=True))
    fixture = SimpleNamespace(directory=tmp_path, source=source, prepared_ref=prepared_ref)
    fixture.payload = dict(
        complete=True,
        passed=True,
        prepared_ref=prepared_ref,
        source_sha256={
            str(Path(module.__file__).resolve()): contract.sha256(module.__file__) for module in (contract, generic)
        },
        budget=dict(producer_fits=2, independent_qr_rebuilds=2, denoise=55965),
    )
    seal(fixture)
    return fixture


def seal(fixture, *, source=None):
    source = source or fixture.source
    payload = copy.deepcopy(fixture.payload)
    payload["source_sha256"][str(source.resolve())] = contract.sha256(source)
    payload.update(
        schema_version="s2a.clean-training-execution-proposal.v1",
        execution_authorized=False,
        automatic_retry=False,
        fresh_performance_authorized=False,
        readiness=dict(ready_for_execution=False),
    )
    proposal_ref = dump(fixture.directory / "proposal.json", payload)
    verifier_ref = contract.reference(source)
    verified = dict(
        schema_version="s2a.clean-training-execution-verification.v1",
        complete=True,
        passed=True,
        eligible_for_execution=True,
        prepared_ref=fixture.prepared_ref,
        proposal_ref=proposal_ref,
        verifier_ref=verifier_ref,
    )
    verified_ref = dump(fixture.directory / "verified.json", verified)
    registration = copy.deepcopy(payload)
    registration.update(
        schema_version=contract.REGISTRATION_SCHEMA,
        execution_authorized=True,
        proposal_ref=proposal_ref,
        execution_verification_ref=verified_ref,
        readiness=dict(ready_for_execution=True),
    )
    fixture.registration = registration
    fixture.registration_ref = dump(fixture.directory / "registration.json", registration)
    fixture.verifier_ref = verifier_ref


def invoke(fixture, evidence=None, **kwargs):
    evidence = evidence or contract.Evidence()
    return contract.check_execution_proof(
        kwargs.pop("registration", fixture.registration),
        prepared_ref=kwargs.pop("prepared_ref", fixture.prepared_ref),
        read_json=kwargs.pop("read_json", evidence.json),
        check_file=kwargs.pop("check_file", evidence.path),
        **kwargs,
    )


def test_real_scripts_verifier_reproduces_bug_then_corrects_only_source(admission):
    source = REPO / "scripts/verify_wam_clean_training_continuation.py"
    seal(admission, source=source)
    original = contract.check_execution_proof
    evidence = contract.Evidence()
    with pytest.raises(ValueError, match="escaped S2A outputs"):
        invoke(admission, evidence)
    with repair.execution_source_proof(admission.registration_ref) as state:
        assert state.callback_corrected_count == 0
        assert state.source_guard_completed is False
        verified = invoke(admission, evidence)
        assert verified["verifier_ref"] == admission.verifier_ref == state.source_verifier_ref
        assert evidence.inputs[str(source)] == admission.verifier_ref["sha256"]
        assert state.callback_corrected_count == 1
        with pytest.raises(ValueError, match="escaped S2A outputs"):
            evidence.path(admission.verifier_ref)
        with pytest.raises(ValueError, match="escaped S2A outputs"):
            evidence.json(admission.verifier_ref)
    assert state.source_guard_completed is True
    assert contract.check_execution_proof is original
    with pytest.raises(ValueError, match="escaped S2A outputs"):
        invoke(admission)


def test_state_refs_are_copies_and_producer_checker_each_keep_original_proof(admission):
    with repair.execution_source_proof(admission.registration_ref) as state:
        source = state.source_verifier_ref
        source["sha256"] = "0" * 64
        bound = state.registration_ref
        bound["sha256"] = "0" * 64
        assert state.source_verifier_ref == admission.verifier_ref
        assert state.registration_ref == admission.registration_ref
        assert invoke(admission)["eligible_for_execution"] is True
        assert invoke(admission)["eligible_for_execution"] is True
        assert state.callback_corrected_count == 2
    assert state.source_guard_completed is True


def test_body_error_restores_original_and_allows_later_use(admission):
    original = contract.check_execution_proof
    with pytest.raises(RuntimeError, match="synthetic body failure"):
        with repair.execution_source_proof(admission.registration_ref) as state:
            invoke(admission)
            raise RuntimeError("synthetic body failure")
    assert contract.check_execution_proof is original
    assert state.source_guard_completed is True
    with repair.execution_source_proof(admission.registration_ref):
        invoke(admission)


def test_nested_context_rejected_without_breaking_outer_hook(admission):
    with repair.execution_source_proof(admission.registration_ref) as state:
        hook = contract.check_execution_proof
        with pytest.raises(ValueError, match="already active"):
            with repair.execution_source_proof(admission.registration_ref):
                pytest.fail("nested context was admitted")
        assert contract.check_execution_proof is hook
        invoke(admission)
        assert state.callback_corrected_count == 1


@pytest.mark.parametrize("fault", ["lambda_reader", "lambda_file", "other_owner", "subclass", "instance_override"])
def test_wrong_reader_or_evidence_owner_rejected(admission, fault):
    owner = contract.Evidence()
    kwargs = {}
    if fault == "lambda_reader":
        kwargs["read_json"] = lambda ref: owner.json(ref)
    elif fault == "lambda_file":
        kwargs["check_file"] = lambda ref: owner.path(ref)
    elif fault == "other_owner":
        kwargs["check_file"] = contract.Evidence().path
    elif fault == "subclass":

        class OtherEvidence(contract.Evidence):
            pass

        owner = OtherEvidence()
    else:
        owner.source_map = lambda *args, **kwargs: None
    with repair.execution_source_proof(admission.registration_ref) as state:
        with pytest.raises(ValueError, match="same original Evidence"):
            invoke(admission, owner, **kwargs)
        assert state.callback_corrected_count == 0


@pytest.mark.parametrize("fault", ["registration", "prepared"])
def test_cross_registration_or_preparation_rejected(admission, fault):
    kwargs = {}
    if fault == "registration":
        value = copy.deepcopy(admission.registration)
        value["budget"]["producer_fits"] = 3
        kwargs["registration"] = value
    else:
        kwargs["prepared_ref"] = dict(path="/unrelated", sha256="0" * 64)
    with repair.execution_source_proof(admission.registration_ref) as state:
        with pytest.raises(ValueError, match="bound registration/preparation"):
            invoke(admission, **kwargs)
        assert state.callback_corrected_count == 0


@pytest.mark.parametrize(
    "fault",
    ["failed", "not_admitted", "retry", "fresh", "source_missing", "source_sha", "proposal", "verifier_unfrozen"],
)
def test_incomplete_or_changed_envelope_rejected_before_install(admission, fault):
    value = copy.deepcopy(admission.registration)
    if fault == "failed":
        value["passed"] = False
    elif fault == "not_admitted":
        value["execution_authorized"] = False
    elif fault == "retry":
        value["automatic_retry"] = True
    elif fault == "fresh":
        value["fresh_performance_authorized"] = True
    elif fault == "source_missing":
        value["source_sha256"].pop(str(Path(contract.__file__).resolve()))
    elif fault == "source_sha":
        value["source_sha256"][str(Path(generic.__file__).resolve())] = "0" * 64
    elif fault == "proposal":
        value["budget"]["denoise"] += 1
    else:
        value["source_sha256"].pop(admission.verifier_ref["path"])
    ref = dump(admission.directory / "bad_registration.json", value)
    with pytest.raises(ValueError):
        with repair.execution_source_proof(ref):
            pytest.fail("bad envelope installed hook")
    assert contract.check_execution_proof is generic.check_execution_proof
    with repair.execution_source_proof(admission.registration_ref):
        invoke(admission)


def test_non_python_registered_file_cannot_be_verifier(admission):
    array = admission.directory / "future.npz"
    array.write_bytes(b"synthetic opaque bytes, never decoded")
    seal(admission, source=array)
    with pytest.raises(ValueError, match="Python source"):
        with repair.execution_source_proof(admission.registration_ref):
            pytest.fail("opaque artifact was treated as source")


def test_symlink_source_alias_rejected(admission):
    alias = admission.directory / "alias.py"
    alias.symlink_to(admission.source)
    verified_path = Path(admission.registration["execution_verification_ref"]["path"])
    verified = json.loads(verified_path.read_text())
    verified["verifier_ref"]["path"] = str(alias)
    # Preserve a valid complete envelope; reject the source alias itself.
    proposal_path = Path(admission.registration["proposal_ref"]["path"])
    proposal = json.loads(proposal_path.read_text())
    proposal["source_sha256"][str(alias)] = contract.sha256(alias)
    proposal_ref = dump(proposal_path, proposal)
    verified["proposal_ref"] = proposal_ref
    verified_ref = dump(verified_path, verified)
    value = copy.deepcopy(admission.registration)
    value.update(
        source_sha256=proposal["source_sha256"], proposal_ref=proposal_ref, execution_verification_ref=verified_ref
    )
    ref = dump(admission.directory / "alias_registration.json", value)
    with pytest.raises(ValueError, match="canonical"):
        with repair.execution_source_proof(ref):
            pytest.fail("source alias was admitted")


@pytest.mark.parametrize("timing", ["before_entry", "before_callback", "after_callback"])
def test_source_drift_rejected_and_hook_restored(admission, timing):
    original = contract.check_execution_proof
    if timing == "before_entry":
        admission.source.write_text("# changed synthetic verifier\n")
        with pytest.raises(ValueError, match="SHA changed"):
            with repair.execution_source_proof(admission.registration_ref):
                pytest.fail("changed source was admitted")
    else:
        with pytest.raises(ValueError, match="changed during run"):
            with repair.execution_source_proof(admission.registration_ref) as state:
                if timing == "after_callback":
                    invoke(admission)
                admission.source.write_text("# changed synthetic verifier\n")
                if timing == "before_callback":
                    invoke(admission)
        assert state.source_guard_completed is False
    assert contract.check_execution_proof is original


def test_metadata_drift_is_checked_even_after_a_successful_callback(admission):
    with pytest.raises(ValueError, match="changed during run"):
        with repair.execution_source_proof(admission.registration_ref) as state:
            invoke(admission)
            Path(admission.prepared_ref["path"]).write_text('{"changed":true}')
    assert state.callback_corrected_count == 1 and state.source_guard_completed is False
    assert contract.check_execution_proof is generic.check_execution_proof


def test_external_hook_change_restores_original_and_rejects(admission):
    with pytest.raises(ValueError, match="hook changed"):
        with repair.execution_source_proof(admission.registration_ref) as state:
            contract.check_execution_proof = lambda *args, **kwargs: None
    assert state.source_guard_completed is False
    assert contract.check_execution_proof is generic.check_execution_proof


def test_preexisting_hook_rejected_and_not_overwritten(admission, monkeypatch):
    def other(*args, **kwargs):
        return None

    monkeypatch.setattr(contract, "check_execution_proof", other)
    with pytest.raises(ValueError, match="already replaced"):
        with repair.execution_source_proof(admission.registration_ref):
            pytest.fail("unrelated existing hook was replaced")
    assert contract.check_execution_proof is other


@pytest.mark.parametrize(
    "name,value", [("CUDA_VISIBLE_DEVICES", "0"), ("LD_LIBRARY_PATH", "/tmp"), ("MKL_NUM_THREADS", "1")]
)
def test_cpu_environment_is_required_before_hook_install(admission, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="requires|require"):
        with repair.execution_source_proof(admission.registration_ref):
            pytest.fail("wrong CPU environment was admitted")
    assert contract.check_execution_proof is generic.check_execution_proof
