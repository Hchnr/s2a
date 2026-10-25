"""046 routing and rejection checks; synthetic metadata never launches a model."""

import copy
import importlib
from pathlib import Path

import pytest

from s2a import clean_training_d1_admission as admission
from s2a.clean_training_w1_handoff import PREPARED_ROOT


@pytest.fixture(autouse=True)
def script_import_path(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))


def ref(name):
    return dict(path=name, sha256="a" * 64)


@pytest.mark.parametrize(
    "key,value",
    [
        ("budget", dict(admission.BUDGET, model_initialization_attempts=2)),
        ("stage_scope", "D0_D1_W2"),
        ("gpu_uuid", "2"),
        ("owner_timeout_s", 100000),
        ("private_port", 31339),
        ("execution_authorized", True),
    ],
)
def test_changed_proposal_rejected_before_any_handoff_or_model(monkeypatch, key, value):
    p = dict(
        schema_version=admission.PREFIX + "proposal.v1",
        complete=True,
        passed=True,
        execution_authorized=False,
        protocol="046",
        stage_scope="D1_only",
        root=str(admission.ROOT),
        budget=admission.BUDGET,
        automatic_retry=False,
        gpu_uuid=admission.GPU,
        private_port=31338,
        owner_timeout_s=88400,
    )
    p[key] = value
    monkeypatch.setattr(admission.Evidence, "json", lambda e, r: p)
    monkeypatch.setattr(admission, "check_cpu_environment", lambda e: None)
    monkeypatch.setattr(admission, "check_handoff", lambda p: pytest.fail("invalid metadata reached W1 lookup"))
    with pytest.raises(ValueError):
        admission.validate_proposal(ref("proposal"))


@pytest.fixture
def artifacts(monkeypatch):
    worker = importlib.import_module("execute_wam_clean_training_d1_handoff")
    registration = {
        key: ref(key)
        for key in (
            "prepared_ref",
            "execution_registration_ref",
            "w1_fit_ref",
            "w1_verification_ref",
            "w1_params_ref",
            "d0_capture_audit_ref",
        )
    }
    owner_ref = ref(str(PREPARED_ROOT / "orchestration/D1/result.json"))
    owner = dict(
        complete=True,
        passed=True,
        stage="D1",
        prepared_ref=registration["prepared_ref"],
        execution_registration_ref=registration["execution_registration_ref"],
        producer_ref=ref("capture"),
        verification_ref=ref("audit"),
    )
    common = dict(
        complete=True,
        passed=True,
        round_name="D1",
        prepared_ref=registration["prepared_ref"],
        execution_registration_ref=registration["execution_registration_ref"],
        actual_requests=300,
        actual_denoise_calls=10500,
        d0_audit_ref=registration["d0_capture_audit_ref"],
    )
    capture = dict(
        common,
        schema_version="s2a.clean-training-capture.v1",
        actual_head_calls=10500,
        model_initialization_attempts=1,
        model_initializations=1,
        future_label_reads=0,
        producer_fits=0,
        physics_steps=0,
        fresh_cases=0,
        w1_fit_ref=registration["w1_fit_ref"],
        w1_verification_ref=registration["w1_verification_ref"],
        readout_ref=registration["w1_params_ref"],
    )
    audit = dict(
        common,
        schema_version="s2a.clean-training-capture-audit.v1",
        collection_ref=ref("capture"),
        actual_online_points_verified=10500,
    )
    data = {owner_ref["path"]: owner, "capture": capture, "audit": audit}
    monkeypatch.setattr(worker, "reference", lambda p: ref(str(p)))
    monkeypatch.setattr(worker, "read", lambda r: copy.deepcopy(data[r["path"]]))
    return worker, registration, owner_ref, data


def test_complete_stage_binds_new_head_and_full_audit(artifacts):
    worker, registration, owner_ref, data = artifacts
    assert worker.check_stage(registration, owner_ref) == data[owner_ref["path"]]


@pytest.mark.parametrize(
    "which,key,value",
    [
        ("capture", "actual_requests", 299),
        ("capture", "actual_head_calls", 10499),
        ("capture", "model_initialization_attempts", 2),
        ("capture", "readout_ref", ref("other_params")),
        ("capture", "future_label_reads", 1),
        ("audit", "actual_online_points_verified", 10499),
        ("audit", "round_name", "D0"),
        ("audit", "d0_audit_ref", ref("other_D0")),
        ("audit", "passed", False),
        ("capture", "execution_registration_ref", ref("other_registration")),
    ],
)
def test_partial_capture_wrong_w1_and_incomplete_audit_rejected(artifacts, which, key, value):
    worker, registration, owner_ref, data = artifacts
    data[which][key] = value
    with pytest.raises(ValueError):
        worker.check_stage(registration, owner_ref)


def test_supervisor_routes_only_one_cpu_wrapper_and_no_extra_stage():
    owner = importlib.import_module("run_wam_clean_training_d1_handoff")
    command = owner.worker_argv(ref("046_admitted"))
    assert Path(command[0]).name == "python"
    assert command[1] == "scripts/execute_wam_clean_training_d1_handoff.py"
    assert command.count("--registration") == command.count("--registration-sha256") == 1
    assert "--stage" not in command and "--gpu-uuid" not in command
    assert owner.argv(ref("046_admitted"), "--run")[1] == "scripts/run_wam_clean_training_d1_handoff.py"


@pytest.mark.parametrize("fault", ["none", "exit7", "failed_result"])
def test_single_stage_supervisor_real_wait_and_failure_receipts(tmp_path, monkeypatch, fault):
    import json
    import sys

    from s2a import durable_stage_supervisor as durable

    assert tmp_path.is_relative_to(durable.REPO / "outputs")
    owner = importlib.import_module("run_wam_clean_training_d1_handoff")
    root = tmp_path / "synthetic046"
    root.mkdir()
    worker = root / "tiny_cpu.py"
    worker.write_text("""import json,sys,time
from pathlib import Path
time.sleep(.08)
root=Path(sys.argv[1]);fault=sys.argv[2]
if fault=="exit7":raise SystemExit(7)
path=root/"worker/result.json";path.parent.mkdir()
with path.open("x") as stream:json.dump(dict(complete=fault!="failed_result",passed=fault!="failed_result",synthetic=True),stream)
""")
    sources = {str(p): durable.sha(p) for p in (durable.REPO / owner.ENTRY, durable.REPO / owner.WORKER, worker)}
    registration = dict(
        root=str(root),
        source_sha256=sources,
        owner_timeout_s=3,
        w1_owner_ref=ref("synthetic_w1"),
        w1_params_ref=ref("synthetic_params"),
    )
    registered = durable.publish(root / "registration.json", dict(synthetic=True))
    launch = durable.publish(root / "synthetic_launch.json", dict(synthetic=True))
    monkeypatch.setattr(owner, "admit", lambda r: registration)
    monkeypatch.setattr(owner, "accept_detached_launch", lambda *args, **kwargs: launch)
    monkeypatch.setattr(owner, "worker_argv", lambda r: [sys.executable, str(worker), str(root), fault])
    monkeypatch.setattr(
        owner,
        "observe_descendant_cleanup",
        lambda p: dict(synthetic=True, no_actual_descendants=True, all_recorded_descendants_reaped=True),
    )
    for key, value in durable.cpu_environment().items():
        if key in durable.ENV_KEYS:
            monkeypatch.setenv(key, value)

    def result(registration, registered):
        r = durable.reference(root / "worker/result.json")
        durable.require(durable.read(r)["passed"] is True, "synthetic result rejected")
        return r

    monkeypatch.setattr(owner, "check_worker", result)
    if fault == "none":
        receipt = owner.execute(registered, mode="run")
        assert durable.read(receipt)["passed"] is True
    else:
        with pytest.raises(Exception):
            owner.execute(registered, mode="run")
        assert json.loads((root / "supervisor/result.json").read_text())["passed"] is False
    exited = json.loads((root / "supervisor/D1/exit.json").read_text())
    release = json.loads((root / "supervisor/D1/release.json").read_text())
    assert exited["actual_wait"] is release["actual_wait"] is release["reaped"] is True
    assert exited["returncode"] == release["returncode"] == (7 if fault == "exit7" else 0)
    assert release["signal_sent"] is False
    assert not (root / "supervisor/D1_2").exists()
