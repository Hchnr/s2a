"""028 requires both the frozen teacher and new actual precision admission."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]


def module():
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location("scaled_state_owner_test", REPO / "scripts/run_wam_scaled_state.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_teacher_failure_cannot_enter_new_gpu_stage(tmp_path, monkeypatch):
    owner = module()
    old = dict(
        complete=True,
        passed=True,
        teacher_gate=dict(passed=False),
        eligible_for_runtime_qualification=False,
        eligible_for_sampling=False,
    )
    audit = dict(
        passed=True,
        phase="fit",
        fit_verified=True,
        **{k: old[k] for k in ("teacher_gate", "eligible_for_runtime_qualification", "eligible_for_sampling")},
    )
    monkeypatch.setattr(owner, "read", lambda p, d: old if p == "fit" else audit)
    with pytest.raises(ValueError, match="teacher success"):
        owner.teacher_admission(
            dict(
                scaled_fit_ref=dict(path="fit", sha256="f"), scaled_fit_verification_ref=dict(path="audit", sha256="a")
            )
        )


def test_wrong_preparation_digest_rejected_before_any_gpu(tmp_path, monkeypatch):
    owner = module()
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps(dict(schema_version="s2a.scaled-state-preparation.v1")))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.setattr(owner, "owned_command", lambda *a, **k: pytest.fail("GPU started before admission"))
    with pytest.raises(ValueError, match="artifact SHA differs"):
        owner.execute(SimpleNamespace(prepared=path, prepared_sha256="0" * 64))


def test_planned_gpu_worker_preserves_model_venv_and_cpu_audit(tmp_path):
    owner = module()
    args = SimpleNamespace(stage="precision", prepared=tmp_path / "prepared.json", prepared_sha256="a" * 64)
    prepared = dict(output_root=str(tmp_path), config=dict(cosmos_repo="/readonly/cosmos", master_port=31086))
    worker, audit, producer = owner.planned_commands(args, prepared)
    assert worker[0] == "/readonly/cosmos/.venv/bin/python"
    assert audit[0] == str(REPO / ".venv/bin/python")
    assert producer == tmp_path / "precision/result.json"
    assert owner.environment(prepared, True)["CUDA_VISIBLE_DEVICES"] == "6"
    assert owner.environment(prepared, False)["CUDA_VISIBLE_DEVICES"] == ""


def test_precision_from_another_candidate_rejected_before_sampling(tmp_path):
    owner = module()
    prepared_path = tmp_path / "prepared.json"
    prepared_path.write_text("{}")
    precision = tmp_path / "precision.json"
    precision.write_text(
        json.dumps(
            dict(
                complete=True,
                passed=True,
                schema_version="s2a.scaled-readout-synthetic-cpu-cuda-precision.v1",
                prepared_ref=owner.reference(prepared_path),
                candidate_model_ref=dict(path="other", sha256="bad"),
            )
        )
    )
    with pytest.raises(ValueError, match="precision admission"):
        owner.sampling_admission(
            dict(adapted_model_ref=dict(path="expected", sha256="good")),
            prepared_path,
            precision,
            owner.sha256(precision),
            tmp_path / "unopened_audit.json",
            "unused",
        )
