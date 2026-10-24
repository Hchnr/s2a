"""STATE ownership admission and diagnostic/task-gate separation."""

import copy
import importlib
import json

import pytest

from s2a.artifacts import REPO
from s2a.clean_training_runtime import reference


@pytest.fixture
def owner(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    return importlib.import_module("run_wam_clean_training_state")


def save(path, value):
    path.write_text(json.dumps(value))
    return reference(path)


@pytest.mark.parametrize("state_gate", [False, True])
@pytest.mark.parametrize(
    "fault", [None, "wrong_fit", "partial_budget", "failed_numerics", "wrong_sampling", "wrong_params"]
)
def test_exact_full_state_admission_keeps_pointwise_gate_diagnostic(tmp_path, owner, state_gate, fault):
    p, r = dict(path="/prepared", sha256="a" * 64), dict(path="/registration", sha256="b" * 64)
    params = dict(path="/fixed-w2", sha256="c" * 64)
    fit = save(tmp_path / "fit.json", dict(params_ref=params))
    verified = dict(path="/independent-w2", sha256="d" * 64)
    w2 = dict(producer_ref=fit, verification_ref=verified)
    common = dict(complete=True, passed=True, prepared_ref=p, execution_registration_ref=r)
    produced = dict(
        common,
        schema_version="s2a.clean-training-state-sampling.v1",
        actual_requests=99,
        actual_denoise_calls=3465,
        model_initialization_attempts=1,
        model_initializations=1,
        w2_fit_ref=fit,
        w2_verification_ref=verified,
        readout_ref=params,
    )
    if fault == "wrong_fit":
        produced["w2_fit_ref"] = verified
    if fault == "partial_budget":
        produced["actual_requests"] = 98
    produced_ref = save(tmp_path / "sampling.json", produced)
    gpu = dict(path="/gpu", sha256="e" * 64)
    audit = dict(
        common,
        schema_version="s2a.clean-training-state-verification.v1",
        sampling_ref=produced_ref,
        w2_fit_ref=fit,
        w2_verification_ref=verified,
        readout_ref=params,
        gpu_after_ref=gpu,
        actual_requests=99,
        actual_denoise_calls=3465,
        numerical_contract_passed=True,
        eligible_for_development=True,
        state_gate_passed=state_gate,
    )
    if fault == "failed_numerics":
        audit["numerical_contract_passed"] = False
    if fault == "wrong_sampling":
        audit["sampling_ref"] = fit
    if fault == "wrong_params":
        audit["readout_ref"] = verified
    audit_ref = save(tmp_path / "audit.json", audit)
    if fault is None:
        owner.check_artifacts(p, r, produced_ref, w2, audit_ref, gpu)
    else:
        with pytest.raises(ValueError):
            owner.check_artifacts(p, r, produced_ref, w2, audit_ref, gpu)


def test_state_commands_bind_only_the_completed_w2_and_actual_owned_receipts(tmp_path, owner, monkeypatch):
    p, r = dict(path="/prepared", sha256="a" * 64), dict(path="/registration", sha256="b" * 64)
    w2 = dict(producer_ref=dict(path="/fit", sha256="c" * 64), verification_ref=dict(path="/qr", sha256="d" * 64))
    monkeypatch.setattr(owner, "prior", lambda *args: copy.deepcopy(w2))
    prepared = dict(root=str(tmp_path), model_config=dict(model_python="/registered/python"))
    command, actual = owner.planned_worker(prepared, p, r)
    assert actual == w2 and command[:2] == ["/registered/python", "scripts/sample_wam_clean_training_state.py"]
    assert command[command.index("--w2-fit") + 1] == "/fit" and command[-4:] == [
        "--gpu-uuid",
        owner.GPU,
        "--seed",
        "42",
    ]
    owned = {name: dict(path=f"/{name}", sha256="e" * 64) for name in ("launch", "exit", "release")}
    audit = owner.planned_audit(
        prepared, p, r, dict(path="/sampling", sha256="f" * 64), owned, dict(path="/gpu", sha256="0" * 64)
    )
    assert audit[audit.index("--worker-launch") + 1] == "/launch"
    assert audit[audit.index("--output") + 1] == str(tmp_path / "review/STATE_verified.json")
