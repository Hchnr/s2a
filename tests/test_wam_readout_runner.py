"""025 owner boundaries with real CPU children; no GPU/model/fit execution."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def owner():
    path = Path(__file__).resolve().parents[1] / "scripts/run_wam_readout.py"
    spec = importlib.util.spec_from_file_location("readout_owner_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def child_file(tmp_path, body):
    path = tmp_path / "cpu_child.py"
    path.write_text("import json, os, pathlib, sys, time\n" + body)
    return path


def cpu_environment():
    return dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")


def test_actual_cpu_child_startup_identity_and_successful_wait_release(owner, tmp_path):
    receipt = tmp_path / "started.json"
    child = child_file(
        tmp_path,
        "pathlib.Path(sys.argv[1]).write_text(json.dumps(dict(pid=os.getpid(),ppid=os.getppid(),cuda=os.environ['CUDA_VISIBLE_DEVICES'],ld=os.environ['LD_LIBRARY_PATH'])))\ntime.sleep(0.3)\n",
    )
    command = [sys.executable, str(child), str(receipt)]
    directory = tmp_path / "owned"
    result = owner.owned_command(command, cpu_environment(), directory, started=receipt, timeout=10)
    actual = owner.read(directory / "actual_child_identity.json")
    launch = owner.read(result["launch"]["path"])
    release = owner.read(result["release"]["path"])
    assert actual["command"] == command
    assert actual["environment"]["CUDA_VISIBLE_DEVICES"] == actual["environment"]["LD_LIBRARY_PATH"] == ""
    assert actual["start_ticks"] == launch["child"]["start_ticks"] == release["start_ticks"]
    assert release["reaped"] and release["returncode"] == 0 and not release["signal_sent"]
    assert owner.read(result["exit"]["path"]) == dict(returncode=0, actual_wait=True, automatic_retry=False)


def test_startup_environment_rejection_reaps_only_actual_child(owner, tmp_path):
    receipt = tmp_path / "started.json"
    child = child_file(
        tmp_path,
        "pathlib.Path(sys.argv[1]).write_text(json.dumps(dict(pid=os.getpid(),ppid=os.getppid(),cuda='0',ld='')))\ntime.sleep(20)\n",
    )
    directory = tmp_path / "rejected"
    # The actual process still has empty CUDA. Its deliberately false startup
    # receipt must fail before any experiment work can be admitted.
    with pytest.raises(ValueError, match="startup environment differs"):
        owner.owned_command(
            [sys.executable, str(child), str(receipt)], cpu_environment(), directory, started=receipt, timeout=10
        )
    release = owner.read(directory / "ownership/release.json")
    assert release["reaped"] is True and release["only_owned_child_signalled"] is True
    assert release["signal_sent"] is True and release["returncode"] != 0
    assert not (directory / "exit.json").exists()


def test_partial_startup_observation_waits_on_same_real_child(owner, tmp_path):
    receipt = tmp_path / "started.json"
    child = child_file(
        tmp_path,
        "path=pathlib.Path(sys.argv[1])\n"
        "with path.open('w') as stream:\n"
        " stream.write('{'); stream.flush(); time.sleep(0.15)\n"
        " item=dict(pid=os.getpid(),ppid=os.getppid(),cuda='',ld='')\n"
        " stream.seek(0); json.dump(item,stream); stream.flush()\n"
        "time.sleep(0.2)\n",
    )
    directory = tmp_path / "partial"
    result = owner.owned_command(
        [sys.executable, str(child), str(receipt)], cpu_environment(), directory, started=receipt, timeout=10
    )
    observations = owner.read(directory / "identity_observations.json")
    assert any(item.get("startup_receipt_incomplete") for item in observations)
    assert observations[-1]["pid"] == owner.read(result["release"]["path"])["pid"]
    assert owner.read(result["exit"]["path"])["returncode"] == 0


def test_nonzero_exit_is_preserved_without_automatic_retry(owner, tmp_path):
    child = child_file(tmp_path, "raise SystemExit(7)\n")
    directory = tmp_path / "failed"
    with pytest.raises(ChildProcessError, match="failed with7"):
        owner.owned_command([sys.executable, str(child)], cpu_environment(), directory, timeout=10)
    assert owner.read(directory / "exit.json") == dict(returncode=7, actual_wait=True, automatic_retry=False)
    release = owner.read(directory / "ownership/release.json")
    assert release["returncode"] == 7 and release["reaped"] is True and not release["signal_sent"]
    with pytest.raises(FileExistsError):
        owner.owned_command([sys.executable, str(child)], cpu_environment(), directory, timeout=10)


def test_sampling_requires_both_independent_and_producer_gate_and_exact_fit_binding(owner, tmp_path):
    prepared, features, fit_path, model = [
        tmp_path / name for name in ("prepared.json", "features.json", "fit.json", "weights.npz")
    ]
    for path in (prepared, features, fit_path, model):
        path.write_text("fixed synthetic source")
    fit = dict(
        complete=True,
        passed=True,
        prepared_ref=owner.reference(prepared),
        features_ref=owner.reference(features),
        producer_fit_groups=2,
        fit_attempts=2,
        eligible_for_sampling=True,
        teacher_gate=dict(passed=True),
        models=dict(adapt=owner.reference(model), bias=owner.reference(model)),
    )
    audit = dict(
        phase="fit",
        passed=True,
        fit_verified=True,
        eligible_for_sampling=True,
        teacher_gate=dict(passed=True),
        prepared=owner.reference(prepared),
        input_sha256={str(fit_path.resolve()): owner.sha256(fit_path)},
    )
    owner.sampling_admission(fit, audit, prepared, features, fit_path)
    for altered_fit, altered_audit in (
        (dict(fit, eligible_for_sampling=False), audit),
        (fit, dict(audit, eligible_for_sampling=False)),
        (fit, dict(audit, input_sha256={str(fit_path.resolve()): "0" * 64})),
    ):
        with pytest.raises(ValueError):
            owner.sampling_admission(altered_fit, altered_audit, prepared, features, fit_path)


def test_owner_refuses_nonempty_cuda_before_reading_sources(owner, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    with pytest.raises(ValueError, match="empty CUDA/LD"):
        owner.execute(SimpleNamespace())


def test_busy_gpu_gate_never_starts_or_touches_existing_processes(owner, monkeypatch, tmp_path):
    monkeypatch.setattr(owner, "REPO", tmp_path)
    monkeypatch.setattr(owner, "gpu_memory", lambda: 1024)
    monkeypatch.setattr(owner, "owned_command", lambda *args, **kwargs: pytest.fail("busy GPU must not launch"))
    directory = tmp_path / "stage"
    directory.mkdir()
    with pytest.raises(ValueError, match="memory0; existing processes remain untouched"):
        owner.run_gpu(dict(output_root=str(tmp_path)), {}, ["never", "launch"], "features", directory)
    assert owner.read(directory / "gpu_before.json") == dict(physical_gpu=6, memory_mib=1024)
    assert not (directory / "model").exists()


def test_interpreter_paths_keep_venv_symlink_and_commands_preserve_budgets(owner, tmp_path):
    cosmos = tmp_path / "cosmos"
    (cosmos / ".venv/bin").mkdir(parents=True)
    (cosmos / ".venv/bin/python").symlink_to(sys.executable)
    args = SimpleNamespace(stage="features", prepared=tmp_path / "prepared.json", prepared_sha256="a" * 64)
    prepared = dict(output_root=str(tmp_path / "result"), config=dict(cosmos_repo=str(cosmos)))
    worker, audit, result = owner.planned_commands(args, prepared)
    assert worker[0] == str(cosmos / ".venv/bin/python")
    assert worker[0] != str((cosmos / ".venv/bin/python").resolve())
    assert worker[1] == "scripts/run_wam_readout_features.py"
    assert audit[1:4] == ["scripts/verify_wam_readout.py", "--phase", "features"]
    assert str(result).endswith("features/result.json")


def test_independent_audit_must_bind_exact_prepared_and_producer_bytes(owner, tmp_path):
    prepared, actual, audit = [tmp_path / name for name in ("prepared.json", "features.json", "audit.json")]
    prepared.write_text("prepared")
    actual.write_text("actual")
    record = dict(
        passed=True,
        phase="features",
        prepared=owner.reference(prepared),
        input_sha256={str(actual.resolve()): owner.sha256(actual)},
    )
    audit.write_text(json.dumps(record))
    assert owner.admitted_audit(audit, "features", prepared, actual) == record
    actual.write_text("changed")
    with pytest.raises(ValueError, match="frozen source changed"):
        owner.admitted_audit(audit, "features", prepared, actual)
