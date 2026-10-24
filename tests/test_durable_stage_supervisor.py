"""Real bounded CPU children test persistent ownership, failure and no retry."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from s2a import durable_stage_supervisor as durable

sys.path.insert(0, str(durable.REPO / "scripts"))
import run_wam_clean_training_continuation as continuation  # noqa: E402


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    for key, value in durable.cpu_environment().items():
        if key in (*durable.ENV_KEYS, "PYTHONDONTWRITEBYTECODE"):
            monkeypatch.setenv(key, value)


def sources():
    path = Path(durable.__file__).resolve()
    return {str(path): durable.sha(path)}


def json_at(path):
    return json.loads(path.read_text())


def test_atomic_exclusive_receipt_never_replaces_original(tmp_path):
    path = tmp_path / "receipt.json"
    first = durable.publish(path, dict(value="first"))
    with pytest.raises(FileExistsError):
        durable.publish(path, dict(value="second"))
    assert durable.reference(path) == first
    assert json_at(path) == dict(value="first")
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("code", [0, 7])
def test_actual_cpu_wait_reap_and_nonzero_is_not_complete(tmp_path, code):
    directory = tmp_path / "stage"
    command = [sys.executable, "-c", f"import time;time.sleep(.1);raise SystemExit({code})"]

    def call():
        return durable.owned_stage(
            command, environment=durable.cpu_environment(), directory=directory, timeout_s=3, source_sha256=sources()
        )

    if code:
        with pytest.raises(ValueError, match="owned stage failed"):
            call()
    else:
        call()
    launch = json_at(directory / "launch.json")
    exit_record = json_at(directory / "exit.json")
    release = json_at(directory / "release.json")
    result = json_at(directory / "result.json")
    assert exit_record["actual_wait"] and exit_record["returncode"] == code
    assert release["reaped"] and release["returncode"] == code
    assert release["signal_sent"] is False
    assert release["pid"] == launch["child"]["pid"]
    assert release["start_ticks"] == launch["child"]["start_ticks"]
    assert launch["child"]["ppid"] == os.getpid()
    assert launch["child"]["sid"] == launch["child"]["pgid"] == launch["child"]["pid"]
    assert launch["command"] == launch["child"]["command"] == command
    assert launch["child"]["environment"] == {key: durable.cpu_environment()[key] for key in durable.OBSERVED_ENV_KEYS}
    assert launch["stdin"] == "DEVNULL" and launch["start_new_session"] is True
    assert result["passed"] is (code == 0)
    assert result["automatic_retry"] is False


def test_actual_timeout_signals_only_owned_child_and_reaps(tmp_path):
    directory = tmp_path / "timed_out"
    started = time.monotonic()
    with pytest.raises(ValueError, match="owned stage failed"):
        durable.owned_stage(
            [sys.executable, "-c", "import time;time.sleep(20)"],
            environment=durable.cpu_environment(),
            directory=directory,
            timeout_s=0.15,
            source_sha256=sources(),
        )
    assert time.monotonic() - started < 4
    release = json_at(directory / "release.json")
    assert release["signal_sent"] and release["reaped"]
    assert release["returncode"] == -signal.SIGTERM
    result = json_at(directory / "result.json")
    assert result["timed_out"] and not result["passed"]
    assert json_at(directory / "exit.json")["actual_wait"] is True


def test_timeout_grace_allows_owner_to_reap_its_child_before_owner_exit(tmp_path):
    script = tmp_path / "cpu_owner.py"
    child_reaped = tmp_path / "child_reaped.json"
    ready = tmp_path / "owner_ready.json"
    script.write_text(
        "import json,signal,subprocess,sys,time\nfrom pathlib import Path\n"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'])\n"
        "def stop(signum,frame):\n"
        " time.sleep(3)\n"
        " child.terminate();code=child.wait(timeout=2)\n"
        " Path(sys.argv[1]).write_text(json.dumps(dict(pid=child.pid,reaped=True,code=code)))\n"
        " raise SystemExit(7)\n"
        "signal.signal(signal.SIGTERM,stop)\n"
        "Path(sys.argv[2]).write_text(json.dumps(dict(pid=child.pid)))\n"
        "while True:time.sleep(.1)\n"
    )
    directory = tmp_path / "patient_owner"
    with pytest.raises(ValueError, match="owned stage failed"):
        durable.owned_stage(
            [sys.executable, str(script), str(child_reaped), str(ready)],
            environment=durable.cpu_environment(),
            directory=directory,
            timeout_s=0.5,
            cleanup_grace_s=5,
            source_sha256=sources(),
        )
    assert ready.exists()
    reaped = json_at(child_reaped)
    assert reaped["reaped"] and reaped["code"] == -signal.SIGTERM
    assert not Path(f"/proc/{reaped['pid']}").exists()
    release = json_at(directory / "release.json")
    assert release["signal_sent"] and not release["kill_sent"]
    assert release["reaped"] and release["returncode"] == 7


def test_birth_read_failure_retains_kernel_ownership_for_cleanup(tmp_path, monkeypatch):
    original_identity = durable.identity

    def refuse_child_birth(pid):
        if pid != os.getpid():
            raise OSError("synthetic birth observation failure")
        return original_identity(pid)

    monkeypatch.setattr(durable, "identity", refuse_child_birth)
    directory = tmp_path / "birth_failure"
    with pytest.raises(ValueError, match="owned stage failed"):
        durable.owned_stage(
            [sys.executable, "-c", "import time;time.sleep(10)"],
            environment=durable.cpu_environment(),
            directory=directory,
            timeout_s=3,
            source_sha256=sources(),
        )
    release = json_at(directory / "release.json")
    assert release["reaped"] and release["signal_sent"]
    assert release["birth_capture_available"] is False and release["start_ticks"] is None
    assert release["ownership_basis"] in ("pidfd", "unreaped kernel child wait relationship")
    assert not Path(f"/proc/{release['pid']}").exists()


@pytest.mark.parametrize("field", ["start_ticks", "ppid", "pid", "boot_id"])
def test_changed_birth_or_parent_never_receives_signal(tmp_path, field):
    with (tmp_path / "child.log").open("x") as log:
        child = subprocess.Popen(
            [sys.executable, "-c", "import time;time.sleep(10)"],
            env=durable.cpu_environment(),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        birth = durable.identity(child.pid)
        wrong = dict(birth)
        wrong[field] = "different" if field == "boot_id" else birth[field] + 1
        try:
            with pytest.raises(ValueError, match="different birth or parent"):
                durable.signal_owned(child, wrong, signal.SIGTERM)
            assert child.poll() is None
        finally:
            release = durable.stop_owned(child, birth)
        assert release["signal_sent"] and release["reaped"]


def test_source_mutation_rejects_before_popen_and_existing_stage_cannot_restart(tmp_path, monkeypatch):
    original = tmp_path / "source.py"
    original.write_text("original\n")
    mapping = {str(original): durable.sha(original)}
    original.write_text("changed\n")
    seen = []
    monkeypatch.setattr(durable.subprocess, "Popen", lambda *a, **k: seen.append(a))
    with pytest.raises(ValueError, match="SHA changed"):
        durable.owned_stage(
            [sys.executable, "-c", "pass"],
            environment=durable.cpu_environment(),
            directory=tmp_path / "new",
            timeout_s=1,
            source_sha256=mapping,
        )
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        durable.owned_stage(
            [sys.executable, "-c", "pass"],
            environment=durable.cpu_environment(),
            directory=existing,
            timeout_s=1,
            source_sha256=sources(),
        )
    assert seen == []


def make_detached_helper(tmp_path):
    helper = tmp_path / "detached_cpu_helper.py"
    helper.write_text(
        "import json,os,sys,time\nfrom pathlib import Path\n"
        "from s2a import durable_stage_supervisor as d\n"
        "root=Path(sys.argv[1]);mode=sys.argv[2]\n"
        "mapping={str(Path(d.__file__).resolve()):d.sha(d.__file__)}\n"
        "command=[sys.executable,str(Path(__file__).resolve()),str(root),mode,'--run']\n"
        "if '--run' not in sys.argv:\n"
        " ref=d.launch_detached(command,directory=root,source_sha256=mapping,ready_timeout_s=5)\n"
        " print(json.dumps(ref),flush=True)\n"
        "else:\n"
        " d.accept_detached_launch(root,command=command,source_sha256=mapping,timeout_s=5)\n"
        " code=7 if mode=='exit7' else 0\n"
        " duration=20 if mode=='timeout' else 1.2\n"
        " child=[sys.executable,'-c',f'import time;time.sleep({duration});raise SystemExit({code})']\n"
        " ok=False\n"
        " try:\n"
        "  d.owned_stage(child,environment=d.cpu_environment(),directory=root/'child',timeout_s=.2 if mode=='timeout' else 4,source_sha256=mapping)\n"
        "  ok=True\n"
        " except ValueError:\n"
        "  pass\n"
        " finally:\n"
        "  d.publish(root/'supervisor_result.json',dict(complete=True,passed=ok,identity=d.identity(os.getpid())))\n"
    )
    return helper


@pytest.mark.parametrize("mode", ["normal", "exit7", "timeout"])
def test_detached_supervisor_outlives_short_launcher_and_reaps_its_real_child(tmp_path, mode):
    helper = make_detached_helper(tmp_path)
    directory = tmp_path / "detached"
    launcher = subprocess.run(
        [sys.executable, str(helper), str(directory), mode],
        cwd=durable.REPO,
        env=durable.cpu_environment(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=7,
    )
    assert launcher.returncode == 0, launcher.stderr
    launch = json_at(directory / "launch.json")
    supervisor_pid = launch["child"]["pid"]
    if mode != "timeout":
        # A separate exec command already returned and its launcher is gone.
        actual = durable.identity(supervisor_pid)
        assert actual["ppid"] == 1
        assert actual["start_ticks"] == launch["child"]["start_ticks"]
        assert not (directory / "supervisor_result.json").exists()
    deadline = time.monotonic() + 6
    while not (directory / "supervisor_result.json").exists():
        assert time.monotonic() < deadline
        time.sleep(0.02)
    release = json_at(directory / "child/release.json")
    child_launch = json_at(directory / "child/launch.json")
    assert child_launch["child"]["ppid"] == supervisor_pid
    assert release["actual_wait"] and release["reaped"]
    assert release["returncode"] == ({"normal": 0, "exit7": 7, "timeout": -signal.SIGTERM}[mode])
    assert release["signal_sent"] is (mode == "timeout")
    assert json_at(directory / "supervisor_result.json")["passed"] is (mode == "normal")
    first = durable.reference(directory / "launch.json")
    with pytest.raises(FileExistsError, match="durable launch already exists"):
        durable.launch_detached(launch["command"], directory=directory, source_sha256=sources())
    assert durable.reference(directory / "launch.json") == first
    rejected = list((directory / "rejected_launches").glob("*.json"))
    assert len(rejected) == 1 and json_at(rejected[0])["new_processes"] == 0


def test_commands_bind_real_admitted_sha_without_circular_template_and_six_timeouts():
    p = dict(path="/actual/prepared.json", sha256="1" * 64)
    r = dict(path="/actual/admitted.json", sha256="2" * 64)
    registration = dict(supervisor=continuation.expected_supervisor())
    assert tuple(registration["supervisor"]["stage_order"]) == continuation.STAGES
    assert registration["supervisor"]["stage_timeouts_s"] == continuation.TIMEOUTS
    for stage in continuation.STAGES:
        command = continuation.build_stage_argv(registration, p, r, stage)
        assert command[command.index("--execution-registration-sha256") + 1] == r["sha256"]
        assert command[command.index("--prepared-sha256") + 1] == p["sha256"]
        assert command[command.index("--seed") + 1] == "42"
        assert not any("placeholder" in value.lower() for value in command)
        if stage not in ("STATE", "DEVELOPMENT"):
            assert command[-2:] == ["--stage", stage]
        else:
            assert "--stage" not in command


def test_returned_contract_cannot_mutate_fixed_outer_timeout_constants():
    spec = continuation.expected_supervisor()
    spec["stage_timeouts_s"]["STATE"] = 1
    assert continuation.TIMEOUTS["STATE"] == 45500
    assert continuation.expected_supervisor()["stage_timeouts_s"]["STATE"] == 45500


def test_cpu_environment_sets_its_own_import_paths_instead_of_inheriting_caller(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/unregistered/caller/source")
    environment = durable.cpu_environment()
    assert environment["PYTHONPATH"] == os.pathsep.join(str(durable.REPO / name) for name in ("src", "scripts"))
    assert "/unregistered/caller/source" not in environment["PYTHONPATH"]
    # The caller need only supply the five CPU/thread admission variables;
    # each actual child receives the explicit source import environment.
    durable.check_cpu_environment({key: environment[key] for key in durable.ENV_KEYS})


def guarded_bundle_environment(tmp_path):
    guard = tmp_path / "import_guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(
        "import atexit,builtins,json,os\nfrom pathlib import Path\nimport numpy as np\n"
        "observed=dict(array_decodes=0,forbidden_imports=0,cuda=os.environ.get('CUDA_VISIBLE_DEVICES'),ld=os.environ.get('LD_LIBRARY_PATH'))\n"
        "def deny_arrays(*args,**kwargs):\n"
        " observed['array_decodes']+=1\n raise AssertionError('bundle metadata CLI decoded arrays')\n"
        "np.load=deny_arrays\noriginal=builtins.__import__\n"
        "def guarded(name,*args,**kwargs):\n"
        " if name.split('.')[0] in {'cosmos_framework','torch','mujoco','robosuite','libero'}:\n"
        "  observed['forbidden_imports']+=1\n  raise AssertionError('bundle metadata CLI imported model or physics')\n"
        " return original(name,*args,**kwargs)\n"
        "builtins.__import__=guarded\n"
        "atexit.register(lambda:Path(os.environ['S2A_CLI_GUARD_RECEIPT']).write_text(json.dumps(observed)))\n"
    )
    environment = durable.cpu_environment()
    # Only the test guard precedes the exact production import paths. Python
    # still puts the actually executed bundle directory at sys.path[0].
    environment["PYTHONPATH"] = str(guard) + os.pathsep + environment["PYTHONPATH"]
    return environment


def test_actual_published_bundle_fresh_help_bootstraps_with_production_cpu_environment(tmp_path):
    directory = durable.REPO / continuation.NEW_ROOT_REL / "source/entrypoints"
    environment = guarded_bundle_environment(tmp_path)
    names = (
        "run_wam_clean_training.py",
        "run_wam_clean_training_state.py",
        "run_wam_clean_training_development.py",
        "collect_wam_clean_training.py",
        "sample_wam_clean_training_state.py",
        "serve_wam_clean_training.py",
        "verify_wam_clean_training_capture.py",
        "verify_wam_clean_training_state.py",
        "verify_wam_clean_training_development.py",
        "execute_wam_clean_training_case.py",
    )
    for name in names:
        script = directory / name
        assert script.is_file(), "041 actual metadata publication must precede final execution QA"
        marker = tmp_path / f"{name}.guard.json"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=durable.REPO,
            env=dict(environment, S2A_CLI_GUARD_RECEIPT=str(marker)),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=3,
        )
        assert result.returncode == 0, (name, result.stderr)
        assert "usage:" in result.stdout
        assert json_at(marker) == dict(array_decodes=0, forbidden_imports=0, cuda="", ld="")


def test_actual_bundle_d0_dry_arguments_reach_admission_and_reject_metadata_only(tmp_path):
    environment = guarded_bundle_environment(tmp_path)
    p = durable.publish(
        tmp_path / "synthetic_prepared.json",
        dict(schema_version="s2a.clean-training-preparation.v1", root=str(durable.REPO / continuation.NEW_ROOT_REL)),
    )
    r = durable.publish(
        tmp_path / "synthetic_metadata_only_registration.json",
        dict(
            schema_version="s2a.clean-training-execution-registration.v1", prepared_ref=p, complete=False, passed=False
        ),
    )
    command = continuation.build_stage_argv(dict(supervisor=continuation.expected_supervisor()), p, r, "D0")
    command.append("--dry-run")
    marker = tmp_path / "d0_dry.guard.json"
    result = subprocess.run(
        command,
        cwd=durable.REPO,
        env=dict(environment, S2A_CLI_GUARD_RECEIPT=str(marker)),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert result.returncode != 0
    assert "a complete independent execution registration is required" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr and "unrecognized arguments" not in result.stderr
    assert json_at(marker) == dict(array_decodes=0, forbidden_imports=0, cuda="", ld="")
    # This is an explicit rejection probe. The real admitted D0 --dry-run is
    # performed later by the root and is not claimed by this synthetic test.


def test_development_mechanical_result_does_not_depend_on_performance_gate(tmp_path):
    p = dict(path="/p", sha256="a" * 64)
    r = dict(path="/r", sha256="b" * 64)
    value = dict(
        schema_version="s2a.clean-training-development-audit-owner.v1",
        complete=True,
        passed=True,
        prepared_ref=p,
        execution_registration_ref=r,
        score=dict(stability_gate_passed=False, adaptation_benefit_passed=False),
    )
    path = tmp_path / "orchestration/DEVELOPMENT/result.json"
    ref = durable.publish(path, value)
    registration = dict(supervisor=continuation.expected_supervisor())
    assert continuation.check_owner_result(tmp_path, registration, p, r, "DEVELOPMENT") == ref


def test_duplicate_run_admission_cannot_poison_active_pipeline_result(tmp_path, monkeypatch):
    directory = tmp_path / "supervisor"
    directory.mkdir()
    monkeypatch.setattr(continuation, "admission", lambda *a: (dict(root=str(tmp_path)), dict(source_sha256=sources())))
    monkeypatch.setattr(
        continuation, "accept_detached_launch", lambda *a, **k: (_ for _ in ()).throw(ValueError("wrong birth"))
    )
    with pytest.raises(ValueError, match="wrong birth"):
        continuation.execute(dict(path="/p", sha256="a" * 64), dict(path="/r", sha256="b" * 64), launch=False)
    assert not (directory / "result.json").exists()
    rejected = list((directory / "rejected_runs").glob("*.json"))
    assert len(rejected) == 1 and json_at(rejected[0])["passed"] is False


@pytest.mark.parametrize("has_release", [False, True])
def test_missing_descendant_release_is_unqualified_even_when_pid_is_gone(tmp_path, has_release):
    pending = tmp_path / "orchestration/D0/model/ownership/pending.json"
    durable.publish(pending, dict(pid=999999999, start_ticks=123))
    if has_release:
        durable.publish(pending.with_name("release.json"), dict(pid=999999999, start_ticks=123, reaped=True))
    observed = continuation.observe_descendant_cleanup(tmp_path)
    assert observed["recorded_descendants"] == 1
    assert observed["all_recorded_descendants_reaped"] is has_release
    assert observed["observer_signals"] == 0
    assert observed["no_recorded_descendants_proves_no_unrecorded_descendants"] is False


def test_supervisor_requires_empty_cuda_and_explicit_thread_limits():
    environment = durable.cpu_environment()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    with pytest.raises(ValueError, match="empty CUDA/LD"):
        durable.check_cpu_environment(environment)
    environment = durable.cpu_environment()
    environment.pop("MKL_NUM_THREADS")
    with pytest.raises(ValueError, match="thread limits8"):
        durable.check_cpu_environment(environment)


def synthetic_admission(monkeypatch):
    p = dict(path="/metadata/prepared.json", sha256="a" * 64)
    r = dict(path="/metadata/admitted.json", sha256="b" * 64)
    m = dict(path="/metadata/metadata_verified.json", sha256="c" * 64)
    verifier = dict(path=str(durable.REPO / "scripts/verify_wam_clean_training_preparation.py"), sha256="a" * 64)
    prep = dict(schema_version="s2a.clean-training-preparation.v1", root=str(durable.REPO / continuation.NEW_ROOT_REL))
    spec = continuation.expected_supervisor()
    files = [
        durable.REPO / "scripts/run_wam_clean_training_continuation.py",
        Path(durable.__file__),
        Path(verifier["path"]),
        *(durable.REPO / value["entrypoint"] for value in spec["stage_commands"].values()),
    ]
    reg = dict(
        schema_version="s2a.clean-training-execution-registration.v1",
        prepared_ref=p,
        complete=True,
        passed=True,
        execution_authorized=True,
        readiness=dict(ready_for_execution=True),
        automatic_retry=False,
        fresh_performance_authorized=False,
        continuation_contract=dict(
            protocol="041",
            original_execution_ref=dict(path="/metadata/old.json", sha256="d" * 64),
            incident_ref=dict(path="/metadata/incident.json", sha256="e" * 64),
            old_data_used_for_fit=False,
            one_new_complete_attempt=True,
            automatic_retry=False,
        ),
        supervisor=spec,
        metadata_verification_ref=m,
        source_sha256={str(path.resolve()): "a" * 64 for path in files},
        snapshot_sha256={"/snapshot": "a" * 64},
        root=prep["root"],
    )
    meta = dict(
        schema_version="s2a.clean-training-preparation-independent.v1",
        prepared_ref=p,
        complete=True,
        passed=True,
        metadata_qualified=True,
        execution_ready=False,
        execution_authorized=False,
        verifier_ref=verifier,
    )
    values = {p["path"]: prep, r["path"]: reg, m["path"]: meta}
    proof_calls = []
    monkeypatch.setattr(continuation, "read", lambda ref: values[ref["path"]])
    monkeypatch.setattr(continuation, "checked", lambda ref: Path(ref["path"]))
    monkeypatch.setattr(continuation, "check_sources", lambda mapping: None)
    monkeypatch.setattr(continuation, "sha", lambda path: "a" * 64)
    monkeypatch.setattr(continuation, "check_execution_proof", lambda *args, **kwargs: proof_calls.append(args))
    return p, r, prep, reg, meta, proof_calls


def test_admission_calls_real_proof_gate_and_binds_metadata_sources_and_fixed_plan(monkeypatch):
    p, r, prep, reg, _, calls = synthetic_admission(monkeypatch)
    assert continuation.admission(p, r) == (prep, reg)
    assert len(calls) == 1 and calls[0][0] is reg


@pytest.mark.parametrize(
    "change", ["root", "retry", "old_rows", "new_attempt", "fresh", "timeout", "argv", "source", "metadata"]
)
def test_admission_refuses_root_retry_source_future_or_plan_drift(monkeypatch, change):
    p, r, prep, reg, meta, _ = synthetic_admission(monkeypatch)
    if change == "root":
        prep["root"] += "_other"
    elif change == "retry":
        reg["automatic_retry"] = True
    elif change == "old_rows":
        reg["continuation_contract"]["old_data_used_for_fit"] = True
    elif change == "new_attempt":
        reg["continuation_contract"]["one_new_complete_attempt"] = False
    elif change == "fresh":
        reg["fresh_performance_authorized"] = True
    elif change == "timeout":
        reg["supervisor"]["stage_timeouts_s"] = dict(continuation.TIMEOUTS, D0=999999)
    elif change == "argv":
        reg["supervisor"]["stage_commands"]["D1"]["arguments"] = ["--stage", "D0"]
    elif change == "source":
        reg["source_sha256"][str(Path(durable.__file__).resolve())] = "f" * 64
    else:
        meta["execution_authorized"] = True
    with pytest.raises(ValueError):
        continuation.admission(p, r)


@pytest.mark.parametrize("fail_stage", [None, "D1"])
def test_serial_pipeline_real_cpu_owners_stop_at_first_failure(tmp_path, monkeypatch, fail_stage):
    p = dict(path="/p", sha256="a" * 64)
    r = dict(path="/r", sha256="b" * 64)
    helper = tmp_path / "cpu_stage_owner.py"
    helper.write_text(
        "import json,sys,time\nfrom pathlib import Path\n"
        "stage=sys.argv[-1]\n"
        "def flag(k):return sys.argv[sys.argv.index(k)+1]\n"
        "schema='s2a.clean-training-state-owner.v1' if stage=='STATE' else 's2a.clean-training-development-audit-owner.v1' if stage=='DEVELOPMENT' else 's2a.clean-training-owner.v1'\n"
        f"passed=stage!={fail_stage!r}\n"
        "value=dict(schema_version=schema,complete=passed,passed=passed,stage=stage,prepared_ref=dict(path=flag('--prepared'),sha256=flag('--prepared-sha256')),execution_registration_ref=dict(path=flag('--execution-registration'),sha256=flag('--execution-registration-sha256')))\n"
        "path=Path(__file__).parent/'orchestration'/stage/'result.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))\n"
        "time.sleep(.05)\nraise SystemExit(0 if passed else 7)\n"
    )
    spec = continuation.expected_supervisor()
    spec["python"] = sys.executable
    spec["stage_commands"] = {stage: dict(entrypoint=str(helper), arguments=[stage]) for stage in continuation.STAGES}
    spec["stage_timeouts_s"] = dict.fromkeys(continuation.STAGES, 3)
    registration = dict(source_sha256=sources(), supervisor=spec)
    (tmp_path / "supervisor").mkdir()
    monkeypatch.setattr(continuation, "admission", lambda *a: (dict(root=str(tmp_path)), registration))
    monkeypatch.setattr(continuation, "accept_detached_launch", lambda *a, **k: None)
    if fail_stage is None:
        continuation.execute(p, r, launch=False)
    else:
        with pytest.raises(ValueError, match="pipeline stopped"):
            continuation.execute(p, r, launch=False)
    value = json_at(tmp_path / "supervisor/result.json")
    assert value["passed"] is (fail_stage is None)
    expected = list(continuation.STAGES) if fail_stage is None else ["D0", "W1", "D1"]
    actual = [stage for stage in continuation.STAGES if (tmp_path / "supervisor" / stage / "launch.json").exists()]
    assert actual == expected
    assert value["automatic_retry"] is False and value["fresh_performance_authorized"] is False
    for stage in expected:
        directory = tmp_path / "supervisor" / stage
        assert json_at(directory / "exit.json")["actual_wait"] is True
        assert json_at(directory / "release.json")["reaped"] is True
        assert json_at(directory / "result.json")["cleanup_grace_s"] == 130
        assert json_at(directory / "descendant_cleanup.json")["observer_signals"] == 0
