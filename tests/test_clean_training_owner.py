"""Check038 stage dependencies and real CPU child ownership without WAM."""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_wam_clean_training as owner  # noqa: E402

from s2a.clean_training_runtime import GPU, read, reference  # noqa: E402


def prepared(tmp_path):
    return dict(
        root=str(tmp_path),
        model_config=dict(model_python="/registered/python", cosmos_source_root="/frozen/cosmos"),
        resources=dict(
            d0_timeout_s=86400,
            d1_timeout_s=86400,
            fit_timeout_s=7200,
            independent_fit_timeout_s=7200,
            audit_timeout_s=1800,
        ),
    )


def test_uuid_cpu_environment_and_thread_caps_do_not_follow_inherited_cuda(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    for gpu in (True, False):
        env = owner.environment(prepared(tmp_path), gpu=gpu)
        assert env["CUDA_VISIBLE_DEVICES"] == (GPU if gpu else "")
        assert env["LD_LIBRARY_PATH"] == ""
        assert [env[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")] == ["8"] * 3
        assert env["MASTER_PORT"] == "31338"
        assert env["PYTHONPATH"].startswith("/frozen/cosmos:")
        assert all(
            env[k].startswith(str(tmp_path)) for k in ("XDG_CACHE_HOME", "HF_HOME", "TORCH_HOME", "CUDA_CACHE_PATH")
        )


@pytest.mark.parametrize("stage", ["W1", "D1", "W2"])
def test_stage_cannot_skip_previous_owned_result(tmp_path, stage):
    with pytest.raises((ValueError, FileNotFoundError)):
        owner.planned_worker(
            prepared(tmp_path), dict(path="/p", sha256="a" * 64), dict(path="/r", sha256="b" * 64), stage
        )


def test_exact_d0_and_fixed_w1_d1_commands(tmp_path, monkeypatch):
    prep = prepared(tmp_path)
    p, r = dict(path="/p", sha256="a" * 64), dict(path="/r", sha256="b" * 64)
    command, result = owner.planned_worker(prep, p, r, "D0")
    assert command[:2] == ["/registered/python", "scripts/collect_wam_clean_training.py"]
    assert command[-4:] == ["--round", "D0", "--gpu-uuid", GPU]
    assert result == tmp_path / "capture/D0/result.json"
    refs = {
        stage: dict(
            producer_ref=dict(path=f"/{stage}/producer", sha256="c" * 64),
            verification_ref=dict(path=f"/{stage}/verified", sha256="d" * 64),
        )
        for stage in ("D0", "D1", "W1")
    }
    monkeypatch.setattr(owner, "prior", lambda prepared, p, r, stage: refs[stage])
    command, result = owner.planned_worker(prep, p, r, "D1")
    assert command[command.index("--d0-audit") + 1] == "/D0/verified"
    assert command[command.index("--w1-fit") + 1] == "/W1/producer"
    assert command[command.index("--w1-verification") + 1] == "/W1/verified"
    command, _ = owner.planned_worker(prep, p, r, "W2")
    assert command.count("--collection") == command.count("--capture-audit") == 2
    assert command[command.index("--collection") + 1] == "/D0/producer"
    assert command[command.index("--collection", command.index("--collection") + 1) + 1] == "/D1/producer"


@pytest.mark.parametrize("code", [0, 7])
def test_cpu_stage_owns_actual_wait_reap_and_keeps_failed_producer(tmp_path, monkeypatch, code):
    # Use the established owned_command unchanged and real Popen children. Only
    # model/source admission is synthetic; this test consumes no actual labels.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    prep = prepared(tmp_path)
    p, r = dict(path="/p", sha256="a" * 64), dict(path="/r", sha256="b" * 64)
    registration = dict(master_port=31338, stage_entrypoints=dict(W1="synthetic-owned-worker"))
    monkeypatch.setattr(owner, "admit_execution", lambda *a, **k: (prep, registration, {}))
    producer = tmp_path / "fit/W1/result.json"
    audit = tmp_path / "review/W1_verified.json"
    script = tmp_path / "cpu_child.py"
    script.write_text(
        "import json,sys,time,hashlib\nfrom pathlib import Path\np=Path(sys.argv[1])\n"
        "p.parent.mkdir(parents=True,exist_ok=True)\n"
        "v=dict(complete=True,passed=True,prepared_ref=json.loads(sys.argv[3]),"
        "execution_registration_ref=json.loads(sys.argv[4]),stage='W1',params_ref={'path':'/params','sha256':'e'*64})\n"
        "if sys.argv[5]=='producer':\n"
        " v.update(schema_version='s2a.clean-training-fit.v1',producer_fits=1,actual_rows=420000,rounds=['D0'])\n"
        "else:\n"
        " q=Path(sys.argv[6]); v.update(schema_version='s2a.clean-training-fit-verification.v1',"
        "fit_ref=dict(path=str(q),sha256=hashlib.sha256(q.read_bytes()).hexdigest()))\n"
        "p.write_text(json.dumps(v))\ntime.sleep(.1)\nsys.exit(int(sys.argv[2]))\n"
    )
    command = [
        sys.executable,
        "synthetic-owned-worker",
        str(script),
        str(producer),
        str(code),
        json.dumps(p),
        json.dumps(r),
        "producer",
    ]
    # planned entrypoint identity is checked before dispatch; the actual child
    # is a plain CPU Python script and launch.json retains its exact command.
    monkeypatch.setattr(owner, "planned_worker", lambda *a: (command, producer))
    real_owned = owner.owned_command

    def dispatch(cmd, env, directory, **kwargs):
        actual = [cmd[0], *cmd[2:]] if cmd[1] == "synthetic-owned-worker" else cmd
        return real_owned(actual, env, directory, **kwargs)

    monkeypatch.setattr(owner, "owned_command", dispatch)
    monkeypatch.setattr(
        owner,
        "planned_audit",
        lambda *a, **k: (
            [sys.executable, str(script), str(audit), "0", json.dumps(p), json.dumps(r), "audit", str(producer)],
            audit,
        ),
    )
    if code:
        with pytest.raises(ChildProcessError, match="failed"):
            owner.execute(p, r, stage="W1")
    else:
        value = read(owner.execute(p, r, stage="W1"))
        assert value["complete"] is value["passed"] is True
        assert len(value["owned_commands"]) == 2
        for item in value["owned_commands"]:
            owner.check_owned_command(item)
            assert read(item["launch"])["selected_environment"] == dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="")
    result = json.loads((tmp_path / "orchestration/W1/result.json").read_text())
    if code:
        assert result["passed"] is result["complete"] is False
        assert producer.is_file() and not audit.exists()
        exit_ref = reference(tmp_path / "orchestration/W1/producer/exit.json")
        release_ref = reference(tmp_path / "orchestration/W1/producer/ownership/release.json")
        assert read(exit_ref)["returncode"] == 7 and read(exit_ref)["actual_wait"] is True
        assert read(release_ref)["reaped"] is True and read(release_ref)["signal_sent"] is False
        assert result["available_failure_receipts"]["producer/exit.json"] == exit_ref
        assert result["available_failure_receipts"]["producer/ownership/release.json"] == release_ref
    assert result["automatic_retry"] is False
    assert result["fresh_performance_authorized"] is False
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_release_contract_cannot_shorten_registered_budget(tmp_path):
    with pytest.raises(ValueError, match="120"):
        owner.observe_release(tmp_path, timeout=60)


@pytest.mark.parametrize(
    "fault", [None, "producer_stage", "producer_budget", "audit_schema", "audit_stage", "audit_producer", "params"]
)
def test_true_receipt_must_bind_exact_stage_full_budget_and_producer(tmp_path, fault):
    p, r = dict(path="/p", sha256="a" * 64), dict(path="/r", sha256="b" * 64)
    producer = dict(
        schema_version="s2a.clean-training-fit.v1",
        stage="W1",
        complete=True,
        passed=True,
        prepared_ref=p,
        execution_registration_ref=r,
        producer_fits=1,
        actual_rows=420000,
        rounds=["D0"],
        params_ref=dict(path="/params", sha256="c" * 64),
    )
    if fault == "producer_stage":
        producer["stage"] = "W2"
    elif fault == "producer_budget":
        producer["actual_rows"] -= 40
    path = tmp_path / "producer.json"
    path.write_text(json.dumps(producer))
    pr = reference(path)
    audit = dict(
        schema_version="s2a.clean-training-fit-verification.v1",
        stage="W1",
        complete=True,
        passed=True,
        prepared_ref=p,
        execution_registration_ref=r,
        fit_ref=pr,
        params_ref=producer["params_ref"],
    )
    if fault == "audit_schema":
        audit["schema_version"] = "any.other.true-receipt"
    elif fault == "audit_stage":
        audit["stage"] = "W2"
    elif fault == "audit_producer":
        audit["fit_ref"] = dict(path="/other", sha256="d" * 64)
    elif fault == "params":
        audit["params_ref"] = dict(path="/wrongparams", sha256="e" * 64)
    ap = tmp_path / "audit.json"
    ap.write_text(json.dumps(audit))
    if fault is None:
        owner.check_stage_artifacts("W1", p, r, pr, reference(ap))
    else:
        with pytest.raises(ValueError):
            owner.check_stage_artifacts("W1", p, r, pr, reference(ap))
