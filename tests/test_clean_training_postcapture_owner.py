"""043 owner with real tiny CPU children and wholly synthetic fit/QR metadata.

Only admission and worker commands are substituted. Production owned_stage,
result checks, wait/reap, CPU environment and exclusive receipts remain real.
No original training futures, fit solver, model or physics are opened.
"""

import importlib
import json
import os
import signal
import sys
import textwrap
from types import SimpleNamespace

import pytest

from s2a import durable_stage_supervisor as durable

TINY_WORKER = r"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text())
role = sys.argv[2]
root = Path(config["registration"]["root"])
registration = config["registration"]
fault = config["fault"]
assert Path.cwd() == Path(config["repo"])
assert os.environ["CUDA_VISIBLE_DEVICES"] == os.environ["LD_LIBRARY_PATH"] == ""
assert all(os.environ[key] == "8" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"))

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def ref(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

# Ensure the parent can observe a live birth before this deliberately tiny job
# exits; this is bounded CPU bookkeeping, not computation or sampling.
time.sleep(.06)
with (root / "events.jsonl").open("a") as stream:
    stream.write(json.dumps(dict(role=role, event="start", pid=os.getpid())) + "\n")
if role == "producer" and fault == "producer_exit7":
    raise SystemExit(7)

base = dict(complete=True, passed=True, stage="W1", prepared_ref=registration["prepared_ref"],
            execution_registration_ref=registration["execution_registration_ref"], actual_rows=420000)
fit_path = Path(registration["fit_output_dir"]) / "result.json"
if role == "producer":
    params_ref = write(fit_path.parent / "parameters.json", dict(synthetic=True))
    delegate = dict(base, schema_version="s2a.clean-training-fit.v1", producer_fits=1,
                    rounds=["D0"], params_ref=params_ref)
    if fault == "producer_rows":
        delegate["actual_rows"] = 419999
    delegate_ref = write(fit_path, delegate)
else:
    # This assertion in the real second child proves that the first child's
    # wait and release receipts were committed before the next launch.
    producer_dir = root / "supervisor/producer"
    exit_record = json.loads((producer_dir / "exit.json").read_text())
    release = json.loads((producer_dir / "release.json").read_text())
    assert exit_record["actual_wait"] is True and exit_record["returncode"] == 0
    assert release["reaped"] is True and release["signal_sent"] is False
    assert release["actual_wait"] is True and release["returncode"] == 0
    fit = json.loads(fit_path.read_text())
    delegate = dict(base, schema_version="s2a.clean-training-fit-verification.v1", fit_ref=ref(fit_path),
                    params_ref=fit["params_ref"], independent_reconstructions=1,
                    full_training_row_reconstruction=True)
    if fault == "independent_rows":
        delegate["actual_rows"] = 419999
    delegate_ref = write(Path(registration["verification_output"]), delegate)

proof = dict(registration_ref=registration["execution_registration_ref"],
             source_verifier_ref=config["synthetic_verifier_ref"],
             callback_corrected_count=1, source_guard_completed=True)
if role == "producer":
    if fault == "proof_zero":
        proof["callback_corrected_count"] = 0
    elif fault == "proof_twice":
        proof["callback_corrected_count"] = 2
    elif fault == "not_restored":
        proof["source_guard_completed"] = False
worker = dict(schema_version="s2a.clean-training-postcapture-worker.v1", complete=True, passed=True,
              repair_ref=config["repair_ref"], role=role, prepared_ref=registration["prepared_ref"],
              execution_registration_ref=registration["execution_registration_ref"],
              source_before_after_exact=True, automatic_retry=False, source_proof=proof,
              producer_fits=int(role == "producer"), independent_reconstructions=int(role == "independent"),
              new_model_initializations=0, new_denoise_calls=0, new_physics_steps=0, new_fresh_cases=0,
              delegate_result_ref=delegate_ref, synthetic=True)
write(root / "workers" / role / "result.json", worker)
with (root / "events.jsonl").open("a") as stream:
    stream.write(json.dumps(dict(role=role, event="finish", pid=os.getpid())) + "\n")
"""


def json_at(path):
    return json.loads(path.read_text())


@pytest.fixture
def scene(tmp_path, monkeypatch):
    assert tmp_path.resolve().is_relative_to(durable.REPO / "outputs")
    for name in durable.ENV_KEYS:
        monkeypatch.setenv(name, durable.cpu_environment()[name])
    monkeypatch.syspath_prepend(str(durable.REPO / "scripts"))
    owner = importlib.import_module("run_wam_clean_training_postcapture")
    root = tmp_path / "repair"
    root.mkdir()
    prepared_ref = durable.publish(
        root / "prepared.json",
        dict(synthetic=True, resources=dict(fit_timeout_s=3, independent_fit_timeout_s=3)),
    )
    execution_ref = durable.publish(root / "execution.json", dict(synthetic=True))
    d0_owner_ref = durable.publish(root / "retained_d0.json", dict(synthetic=True))
    failure_ref = durable.publish(root / "original_failure.json", dict(synthetic=True))
    repair_ref = durable.publish(root / "registration.json", dict(synthetic=True))
    tiny_worker = root / "tiny_cpu_worker.py"
    tiny_worker.write_text(textwrap.dedent(TINY_WORKER))
    sources = {
        str(path): durable.sha(path) for path in (durable.REPO / owner.ENTRY, durable.REPO / owner.WORKER, tiny_worker)
    }
    registration = dict(
        root=str(root),
        prepared_ref=prepared_ref,
        execution_registration_ref=execution_ref,
        d0_owner_ref=d0_owner_ref,
        failure_ref=failure_ref,
        fit_output_dir=str(root / "synthetic_original/fit/W1"),
        verification_output=str(root / "synthetic_original/review/W1_verified.json"),
        source_sha256=sources,
    )
    # This fixture attests no real043 admission or detachment. The actual owner
    # receives a dict through its normal import API and owns genuine children.
    admissions = []

    def admit(ref):
        assert ref == repair_ref
        admissions.append(ref)
        return registration

    monkeypatch.setitem(sys.modules, "execute_wam_clean_training_postcapture_fit", SimpleNamespace(admit=admit))

    def accepted(directory, *, command, source_sha256):
        assert command == owner.argv(repair_ref, "--run")
        assert source_sha256 == sources
        directory.mkdir(exist_ok=False)
        return durable.publish(directory / "synthetic_launch.json", dict(synthetic=True, no_detachment_claim=True))

    monkeypatch.setattr(owner, "accept_detached_launch", accepted)
    config_path = root / "tiny_config.json"
    command_calls = []

    def worker_argv(ref, role):
        assert ref == repair_ref and role in owner.ROLES
        command_calls.append(role)
        return [sys.executable, str(tiny_worker), str(config_path), role]

    monkeypatch.setattr(owner, "worker_argv", worker_argv)
    return SimpleNamespace(
        owner=owner,
        root=root,
        repair_ref=repair_ref,
        registration=registration,
        command_calls=command_calls,
        admissions=admissions,
        config_path=config_path,
        configure=lambda fault=None: durable.publish(
            config_path,
            dict(
                registration=registration,
                repair_ref=repair_ref,
                repo=str(durable.REPO),
                synthetic_verifier_ref=dict(path=str(tiny_worker), sha256=durable.sha(tiny_worker)),
                fault=fault,
            ),
        ),
    )


def assert_actual_wait_reap(scene, role, code):
    directory = scene.root / "supervisor" / role
    launch = json_at(directory / "launch.json")
    exit_record = json_at(directory / "exit.json")
    release = json_at(directory / "release.json")
    result = json_at(directory / "result.json")
    child = launch["child"]
    assert child["ppid"] == os.getpid()
    assert child["sid"] == child["pgid"] == child["pid"]
    assert launch["start_new_session"] is True and launch["stdin"] == "DEVNULL"
    assert launch["command"] == child["command"]
    assert child["cwd"] == str(durable.REPO)
    assert child["environment"] == {key: durable.cpu_environment()[key] for key in durable.OBSERVED_ENV_KEYS}
    assert exit_record["actual_wait"] is True and exit_record["returncode"] == code
    assert release["actual_wait"] is release["reaped"] is True
    assert release["pid"] == child["pid"] and release["start_ticks"] == child["start_ticks"]
    assert release["returncode"] == code
    assert release["signal_sent"] is release["kill_sent"] is False
    assert release["only_owned_child_signalled"] is True
    assert result["automatic_retry"] is False
    try:
        current = durable.identity(child["pid"])
    except FileNotFoundError:
        pass
    else:
        assert current["start_ticks"] != child["start_ticks"], "the actual child must have been reaped"
    return result


def test_producer_then_independent_use_actual_wait_and_reap(scene):
    scene.configure()
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    result_ref = scene.owner.execute(scene.repair_ref, mode="run")
    result = durable.read(result_ref)
    assert result["complete"] is result["passed"] is True
    assert result["automatic_retry"] is result["fresh_performance_authorized"] is result["d1_authorized"] is False
    assert [entry["role"] for entry in result["owned_commands"]] == ["producer", "independent"]
    assert scene.command_calls == ["producer", "independent"]
    events = [json.loads(line) for line in (scene.root / "events.jsonl").read_text().splitlines()]
    assert [(entry["role"], entry["event"]) for entry in events] == [
        ("producer", "start"),
        ("producer", "finish"),
        ("independent", "start"),
        ("independent", "finish"),
    ]
    for role in ("producer", "independent"):
        role_result = assert_actual_wait_reap(scene, role, 0)
        assert role_result["complete"] is role_result["passed"] is True
        assert role_result["stage_result_ref"] == durable.reference(scene.root / "workers" / role / "result.json")
    assert {sig: signal.getsignal(sig) for sig in handlers} == handlers
    assert not (scene.root / "orchestration").exists()


@pytest.mark.parametrize(
    "fault,failed_role,code",
    [
        ("producer_exit7", "producer", 7),
        ("producer_rows", "producer", 0),
        ("proof_zero", "producer", 0),
        ("proof_twice", "producer", 0),
        ("not_restored", "producer", 0),
        ("independent_rows", "independent", 0),
    ],
)
def test_failure_is_false_reaped_preserved_and_never_retried(scene, monkeypatch, fault, failed_role, code):
    scene.configure(fault)
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    with pytest.raises(ValueError, match="owned stage failed"):
        scene.owner.execute(scene.repair_ref, mode="run")
    result_path = scene.root / "supervisor/result.json"
    result_ref = durable.reference(result_path)
    result = durable.read(result_ref)
    role_result = assert_actual_wait_reap(scene, failed_role, code)
    assert result["complete"] is result["passed"] is False
    assert role_result["complete"] is role_result["passed"] is False
    assert result["active_role"] == failed_role
    assert result["failed_role_owner_ref"] == durable.reference(scene.root / "supervisor" / failed_role / "result.json")
    assert result["automatic_retry"] is False
    expected = ["producer"] if failed_role == "producer" else ["producer", "independent"]
    assert scene.command_calls == expected
    if failed_role == "producer":
        assert not (scene.root / "supervisor/independent").exists()
        assert not (scene.root / "workers/independent").exists()
    assert {sig: signal.getsignal(sig) for sig in handlers} == handlers

    def forbidden_popen(*args, **kwargs):
        pytest.fail("a failed owner must never launch another worker or owner")

    monkeypatch.setattr(durable.subprocess, "Popen", forbidden_popen)
    with pytest.raises((ValueError, FileExistsError)):
        scene.owner.execute(scene.repair_ref, mode="launch")
    assert durable.reference(result_path) == result_ref
    assert scene.command_calls == expected
    assert not (scene.root / "orchestration").exists()


def test_owner_dry_constructs_two_commands_without_child_or_artifact(scene, monkeypatch):
    def forbidden_popen(*args, **kwargs):
        pytest.fail("owner dry run must not launch any process")

    monkeypatch.setattr(durable.subprocess, "Popen", forbidden_popen)
    result = scene.owner.execute(scene.repair_ref, mode="dry-run")
    assert result["complete"] is result["passed"] is True
    assert result["actual_worker_launches"] == result["new_model_calls"] == result["new_fits"] == 0
    assert set(result["commands"]) == {"producer", "independent"}
    assert result["timeouts_s"] == dict(producer=3, independent=3)
    assert scene.command_calls == ["producer", "independent"]
    assert not (scene.root / "supervisor").exists()
    assert not (scene.root / "workers").exists()
    assert not (scene.root / "events.jsonl").exists()


def test_unfrozen_owner_source_is_rejected_before_any_worker(scene, monkeypatch):
    scene.registration["source_sha256"][str(durable.REPO / scene.owner.ENTRY)] = "0" * 64

    def forbidden_popen(*args, **kwargs):
        pytest.fail("unfrozen owner must not start a process")

    monkeypatch.setattr(durable.subprocess, "Popen", forbidden_popen)
    with pytest.raises(ValueError, match="owner/worker source is not frozen"):
        scene.owner.execute(scene.repair_ref, mode="run")
    assert scene.command_calls == [] and not (scene.root / "supervisor").exists()
