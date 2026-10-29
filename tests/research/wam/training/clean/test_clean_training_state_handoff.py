"""048 production bridge routing and guards, without W2 or STATE execution."""

import ast
import copy
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from s2a.infrastructure.lifecycle.durable_stage_supervisor import REPO, reference
from s2a.research.wam.training.clean import clean_training_state_handoff as state
from tests.support.source_layout import historical_import_closure, historical_source_bytes, historical_source_ref


def archival_metadata_environment(tmp_path):
    """Isolate original imports for metadata regression; grant no runtime qualification."""
    historical = tmp_path / "historical_sources"
    closure = historical_import_closure(
        [
            "scripts/prepare_wam_clean_training_state_handoff.py",
            "scripts/run_wam_clean_development_repair.py",
            "scripts/run_wam_clean_training.py",
            "scripts/sample_wam_clean_training_state.py",
            "scripts/run_wam_clean_training_development.py",
            "scripts/serve_wam_clean_training.py",
            "scripts/execute_wam_clean_training_case.py",
            "scripts/verify_wam_clean_training_development.py",
        ],
        historical,
    )
    guard = tmp_path / "archival_guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(
        "import atexit,builtins,json,sys\nfrom pathlib import Path\nimport numpy as np\n"
        "attempts=dict(arrays=0,models=0,current_sources=0)\n"
        "def deny(*a,**kw):\n attempts['arrays']+=1\n raise AssertionError('archival metadata decoded arrays')\n"
        "np.load=deny\noriginal=builtins.__import__\n"
        "def guarded(name,*a,**kw):\n"
        " if name.split('.')[0] in {'torch','cosmos_framework','mujoco','robosuite','libero'}:\n"
        "  attempts['models']+=1\n  raise AssertionError('archival metadata imported model or environment')\n"
        " result=original(name,*a,**kw)\n"
        " for module in tuple(sys.modules.values()):\n"
        "  source=getattr(module,'__file__','') or ''\n"
        f"  if source.startswith({tuple(str(REPO / area) + os.sep for area in ('src', 'scripts'))!r}):\n"
        "   attempts['current_sources']+=1\n   raise AssertionError('current source entered archival imports')\n"
        " return result\nbuiltins.__import__=guarded\n"
        "from s2a import artifacts\n"
        f"artifacts.REPO=Path({str(REPO)!r})\n"
        "from s2a import durable_stage_supervisor as archival_supervisor\n"
        f"archival_supervisor.REPO=Path({str(REPO)!r})\n"
        f"atexit.register(lambda:Path({str(tmp_path / 'archival_guard.json')!r}).write_text(json.dumps(attempts)))\n"
    )
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        LD_LIBRARY_PATH="",
        PYTHONDONTWRITEBYTECODE="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTHONPATH=os.pathsep.join(map(str, (guard, historical / "src", historical / "scripts"))),
    )
    (tmp_path / "archival_scope.json").write_text(
        json.dumps(dict(scope="archival_metadata_regression_not_current_runtime_qualification", closure=closure))
    )
    return environment, historical


def archival_bridge_case(bridge_scene, node):
    directory = Path(bridge_scene[1]["path"]).parent
    environment, historical = archival_metadata_environment(directory)
    original_ref = historical_source_ref("tests/test_clean_training_state_handoff.py")
    source = directory / "test_original_state_bridge.py"
    source.write_bytes(historical_source_bytes(original_ref))
    before = role_receipt_fingerprints()
    result = subprocess.run(
        [
            str(REPO / ".venv/bin/python"),
            "-m",
            "pytest",
            "-q",
            str(source) + "::" + node,
            "-o",
            "pythonpath=",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            "--basetemp=" + str(directory / "archival_pytest_tmp"),
        ],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    (directory / "archival_pytest.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert source.read_bytes() == historical_source_bytes(original_ref)
    assert json.loads((directory / "archival_guard.json").read_text()) == dict(arrays=0, models=0, current_sources=0)
    verify_archival_metadata(directory)
    assert role_receipt_fingerprints() == before


def verify_archival_metadata(directory):
    scope = json.loads((directory / "archival_scope.json").read_text())
    assert scope["scope"] == "archival_metadata_regression_not_current_runtime_qualification"
    for row in scope["closure"]["sources"].values():
        assert reference(row["staged_path"])["sha256"] == row["original_ref"]["sha256"]


def role_receipt_fingerprints():
    """Dry metadata calls must preserve any already recorded production receipts."""
    folder = state.ROOT / "role_receipts"
    return {str(p.relative_to(folder)): reference(p)["sha256"] for p in folder.rglob("*") if p.is_file()}


def ref(name):
    return dict(path=str(name), sha256="a" * 64)


@pytest.fixture(autouse=True)
def script_path(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))


def test_production_bridge_matches_fixed_source_template():
    prepare = importlib.import_module("scripts.wam.training.clean.prepare_wam_clean_training_state_handoff")
    assert ast.dump(ast.parse(prepare.bridge_source())) == ast.dump(ast.parse(state.BRIDGE_PATH.read_text()))


@pytest.mark.parametrize(
    "key,value",
    [
        ("budget", dict(state.BUDGET, requests=100)),
        ("stage_scope", "STATE_DEVELOPMENT"),
        ("gpu_uuid", "2"),
        ("owner_timeout_s", 50000),
        ("execution_authorized", True),
    ],
)
def test_changed_scope_rejected_before_upstream_or_model(monkeypatch, key, value):
    p = dict(
        schema_version=state.PREFIX + "proposal.v1",
        complete=True,
        passed=True,
        protocol="048",
        execution_authorized=False,
        stage_scope="STATE_only",
        root=str(state.ROOT),
        budget=state.BUDGET,
        gpu_uuid=state.GPU,
        automatic_retry=False,
        owner_timeout_s=45500,
        private_port=31338,
    )
    p[key] = value
    monkeypatch.setattr(state.Evidence, "json", lambda e, r: p)
    monkeypatch.setattr(state, "qualified_w2", lambda *args: pytest.fail("invalid scope reached W2"))
    with pytest.raises(ValueError):
        state.validate(ref("proposal"))


def test_model_role_rejects_cpu_bootstrap_before_reading_metadata(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(key, "8")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(state.Evidence, "json", lambda *args: pytest.fail("CPU bootstrap read model-role registration"))
    with pytest.raises(ValueError, match="model bootstrap"):
        state.admit(ref("unavailable"), role="model")


@pytest.fixture
def bridge_scene(tmp_path, monkeypatch):
    for name in ("run_wam_clean_training", "continuation_runtime", "collect_wam_clean_training"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    a = {
        key: ref(key)
        for key in (
            "prepared_ref",
            "execution_registration_ref",
            "w2_owner_ref",
            "w2_fit_ref",
            "w2_verification_ref",
            "w2_params_ref",
        )
    }
    a.update(
        bridge_ref=reference(state.BRIDGE_PATH),
        original_training_owner_ref=reference(state.BUNDLE / "run_wam_clean_training.py"),
    )
    registered = ref(tmp_path / "048_admitted.json")
    monkeypatch.setattr(state, "admit", lambda r, role="owner": copy.deepcopy(a))
    parent = dict(producer_ref=a["w2_fit_ref"], verification_ref=a["w2_verification_ref"])
    monkeypatch.setattr(state, "qualified_w2", lambda p: copy.deepcopy(parent))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.setenv(state.DRY_ENV, "1")
    state.reset_dry_receipts()
    prepared = dict(
        root=str(state.PREPARED_ROOT),
        model_config=dict(model_python="synthetic_model_python"),
        prerequisite_refs=dict(bridge_qualification=ref("bridge_qualification")),
    )
    modules = dict(
        run_wam_clean_training=a["bridge_ref"],
        continuation_runtime=reference(state.BUNDLE / "continuation_runtime.py"),
        collect_wam_clean_training=reference(state.BUNDLE / "collect_wam_clean_training.py"),
    )
    return a, registered, prepared, modules, parent


def test_actual_owner_and_sampler_call_sites_use_production_bridge(bridge_scene, monkeypatch):
    archival_bridge_case(bridge_scene, "test_actual_owner_and_sampler_call_sites_use_production_bridge")


def test_gpu_environment_safe_path_and_independent_audit_unbridged(bridge_scene, monkeypatch):
    archival_bridge_case(bridge_scene, "test_gpu_environment_safe_path_and_independent_audit_unbridged")


def test_dry_model_environment_is_refused(bridge_scene):
    archival_bridge_case(bridge_scene, "test_dry_model_environment_is_refused")


def test_lookup_receipt_uses_actual_old_four_field_child_identity(monkeypatch, tmp_path):
    registered = ref("registered")
    a = {
        k: ref(k)
        for k in (
            "prepared_ref",
            "execution_registration_ref",
            "w2_owner_ref",
            "w2_fit_ref",
            "w2_verification_ref",
            "w2_params_ref",
            "bridge_ref",
            "original_training_owner_ref",
            "original_sampler_ref",
            "original_owner_ref",
        )
    }
    values = {}
    for role in ("owner", "model"):
        values[str(state.ROOT / "role_receipts" / f"{role}.json")] = dict(
            schema_version=state.PREFIX + "lookup.v1",
            complete=True,
            passed=True,
            role=role,
            dry_run=False,
            registration_ref=registered,
            **{
                k: a[k]
                for k in (
                    "prepared_ref",
                    "execution_registration_ref",
                    "w2_owner_ref",
                    "w2_fit_ref",
                    "w2_verification_ref",
                    "w2_params_ref",
                    "bridge_ref",
                    "original_training_owner_ref",
                )
            },
            source_guard_completed=True,
            sys_path_prefix=[str(state.BRIDGE_DIRECTORY), str(state.BUNDLE)],
            safe_path=role == "model",
            identity=dict(
                pid=11 if role == "owner" else 12,
                start_ticks=100,
                ppid=10 if role == "owner" else 11,
                environment=dict(CUDA_VISIBLE_DEVICES="" if role == "owner" else state.GPU),
            ),
            caller_source_ref=a["original_owner_ref"] if role == "owner" else a["original_sampler_ref"],
            caller_function="planned_worker" if role == "owner" else "admit_sampling",
        )
    values["launch"] = dict(child=dict(pid=12, ppid=11, proc_state="R", start_ticks=100))
    monkeypatch.setattr(state, "reference", lambda p: ref(p))
    monkeypatch.setattr(state.Evidence, "json", lambda e, r: copy.deepcopy(values[r["path"]]))
    stage = dict(owned_commands=[dict(launch=ref("launch"))])
    assert len(state.check_lookup_receipts(a, registered, stage)) == 2
    values[str(state.ROOT / "role_receipts/model.json")]["identity"]["start_ticks"] = 101
    with pytest.raises(ValueError, match="actual sampler child"):
        state.check_lookup_receipts(a, registered, stage)


@pytest.mark.parametrize("runtime", ["main", "model"])
def test_actual_original_sampler_help_imports_production_bridge_without_model(tmp_path, runtime):
    import json
    import subprocess

    receipts_before = role_receipt_fingerprints()
    prepared = json.loads((state.PREPARED_ROOT / "preparation/prepared.json").read_text())
    python = str(REPO / ".venv/bin/python") if runtime == "main" else prepared["model_config"]["model_python"]
    archival, _ = archival_metadata_environment(tmp_path)
    env = dict(
        archival,
        CUDA_VISIBLE_DEVICES="",
        LD_LIBRARY_PATH="",
        OMP_NUM_THREADS="8",
        OPENBLAS_NUM_THREADS="8",
        MKL_NUM_THREADS="8",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONSAFEPATH="1",
        PYTHONVERBOSE="1",
        PYTHONPATH=":".join(map(str, (state.BRIDGE_DIRECTORY, state.BUNDLE))) + ":" + archival["PYTHONPATH"],
    )
    out = subprocess.run(
        [python, str(state.BUNDLE / "sample_wam_clean_training_state.py"), "--help"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode == 0, out.stderr[-1000:]
    assert "usage:" in out.stdout and str(state.BRIDGE_PATH) in out.stderr
    assert "import 'torch'" not in out.stderr and "import 'mujoco'" not in out.stderr
    assert json.loads((tmp_path / "archival_guard.json").read_text()) == dict(arrays=0, models=0, current_sources=0)
    verify_archival_metadata(tmp_path)
    (tmp_path / f"{runtime}_help_stdout.log").write_text(out.stdout)
    (tmp_path / f"{runtime}_help_imports.log").write_text(out.stderr)
    assert role_receipt_fingerprints() == receipts_before


@pytest.mark.parametrize("fault", ["none", "exit7", "source_drift"])
def test_state_supervisor_real_cpu_child_and_false_final_guard(tmp_path, monkeypatch, fault):
    import json

    from s2a.infrastructure.lifecycle import durable_stage_supervisor as durable

    owner = importlib.import_module("scripts.wam.training.clean.run_wam_clean_training_state_handoff")
    root = tmp_path / "synthetic048"
    root.mkdir()
    monkeypatch.setattr(owner, "ROOT", root)
    script = root / "tiny.py"
    script.write_text("""import json,sys,time
from pathlib import Path
time.sleep(.08)
root=Path(sys.argv[1]);fault=sys.argv[2]
if fault=="exit7":raise SystemExit(7)
path=root/"worker/result.json";path.parent.mkdir()
with path.open("x") as stream:json.dump(dict(complete=True,passed=True,state_gate_passed=False,delegate_result_ref=dict(path="synthetic_state_owner",sha256="a"*64),verification_ref=dict(path="synthetic_state_audit",sha256="a"*64),synthetic=True),stream)
""")
    registered = durable.publish(root / "admitted.json", dict(synthetic=True))
    launch = durable.publish(root / "launch.json", dict(synthetic=True))
    sources = {str(p): durable.sha(p) for p in (REPO / owner.ENTRY, REPO / owner.WORKER, script)}
    a = dict(source_sha256=sources, owner_timeout_s=3, w2_owner_ref=ref("w2_owner"), w2_params_ref=ref("params"))
    monkeypatch.setattr(owner, "admit", lambda r, role="owner": a)
    monkeypatch.setattr(owner, "accept_detached_launch", lambda *args, **kw: launch)
    monkeypatch.setattr(owner, "worker_argv", lambda r: [sys.executable, str(script), str(root), fault])
    monkeypatch.setattr(owner, "check_worker", lambda a, r: reference(root / "worker/result.json"))
    monkeypatch.setattr(
        owner, "observe_descendant_cleanup", lambda p: dict(synthetic=True, all_recorded_descendants_reaped=True)
    )
    if fault == "source_drift":
        monkeypatch.setattr(
            owner, "check_sources", lambda s: (_ for _ in ()).throw(ValueError("synthetic source drift"))
        )
    for key in durable.ENV_KEYS:
        monkeypatch.setenv(key, durable.cpu_environment()[key])
    if fault == "none":
        result = owner.execute(registered, mode="run")
        assert durable.read(result)["passed"] is True
        assert durable.read(result)["state_gate_passed"] is False  # Diagnostic False is not a new rejection gate.
    else:
        with pytest.raises(Exception):
            owner.execute(registered, mode="run")
        assert json.loads((root / "supervisor/result.json").read_text())["passed"] is False
    exited = json.loads((root / "supervisor/STATE/exit.json").read_text())
    release = json.loads((root / "supervisor/STATE/release.json").read_text())
    assert exited["actual_wait"] is release["actual_wait"] is release["reaped"] is True
    assert release["signal_sent"] is False and exited["returncode"] == release["returncode"] == (
        7 if fault == "exit7" else 0
    )


@pytest.fixture
def state_artifacts(monkeypatch):
    worker = importlib.import_module("s2a.research.wam.training.state_stage_checks")
    a = {
        k: ref(k)
        for k in ("prepared_ref", "execution_registration_ref", "w2_fit_ref", "w2_verification_ref", "w2_params_ref")
    }
    a.update(
        original_sampler_ref=ref(state.BUNDLE / "sample_wam_clean_training_state.py"),
        original_verifier_ref=ref(state.BUNDLE / "verify_wam_clean_training_state.py"),
        gpu_uuid=state.GPU,
    )
    stage_ref = ref(state.PREPARED_ROOT / "orchestration/STATE/result.json")
    commands = [
        dict(launch=ref(role + "_launch"), exit=ref(role + "_exit"), release=ref(role + "_release"))
        for role in ("model", "audit")
    ]
    owner = dict(
        schema_version="s2a.clean-training-state-owner.v1",
        complete=True,
        passed=True,
        stage="STATE",
        prepared_ref=a["prepared_ref"],
        execution_registration_ref=a["execution_registration_ref"],
        numerical_contract_passed=True,
        eligible_for_development=True,
        producer_ref=ref("sample"),
        verification_ref=ref("verified"),
        state_gate_passed=False,
        owned_commands=commands,
        gpu_after_ref=ref("gpu"),
    )
    common = dict(
        complete=True,
        passed=True,
        prepared_ref=a["prepared_ref"],
        execution_registration_ref=a["execution_registration_ref"],
        actual_requests=99,
        actual_denoise_calls=3465,
        w2_fit_ref=a["w2_fit_ref"],
        w2_verification_ref=a["w2_verification_ref"],
        readout_ref=a["w2_params_ref"],
        actual_head_calls=3465,
        model_initializations=1,
        fresh_cases=0,
    )
    producer = dict(
        common,
        schema_version="s2a.clean-training-state-sampling.v1",
        model_initialization_attempts=1,
        future_label_reads=0,
        producer_fits=0,
        physics_steps=0,
    )
    audit = dict(
        common,
        schema_version="s2a.clean-training-state-verification.v1",
        sampling_ref=ref("sample"),
        numerical_contract_passed=True,
        eligible_for_development=True,
        state_gate_passed=False,
        paired_initial_conditions=48,
        zero_requests_verified=3,
        zero_fullpoint_nodes_verified=105,
        actual_solver_sigma_traces_verified=99,
        physics_tasks_executed=0,
        gpu_after_ref=ref("gpu"),
    )
    data = {
        stage_ref["path"]: owner,
        "sample": producer,
        "verified": audit,
        "prepared_ref": dict(model_config=dict(model_python="model_python")),
        "gpu": dict(released=True, uuid=state.GPU, memory_mib=0, compute_apps=[]),
    }
    for i, role in enumerate(("model", "audit")):
        data[role + "_launch"] = dict(
            child=dict(pid=i + 1, start_ticks=100),
            command=[
                "model_python" if i == 0 else str(REPO / ".venv/bin/python"),
                a["original_sampler_ref"]["path"] if i == 0 else a["original_verifier_ref"]["path"],
            ],
        )
        data[role + "_exit"] = dict(actual_wait=True, returncode=0, automatic_retry=False)
        data[role + "_release"] = dict(
            only_owned_child_signalled=True,
            reaped=True,
            returncode=0,
            signal_sent=False,
            original_error=None,
            pid=i + 1,
            start_ticks=100,
        )
    monkeypatch.setattr(worker, "reference", lambda path: ref(path))
    monkeypatch.setattr(worker, "read", lambda r: copy.deepcopy(data[r["path"]]))
    monkeypatch.setattr(state.Evidence, "json", lambda e, r: copy.deepcopy(data[r["path"]]))
    monkeypatch.setattr(state.Evidence, "finish", lambda e: None)  # The receipt reader above is deliberately synthetic.
    return worker, a, stage_ref, data


def test_complete_numerical_state_with_false_diagnostic_is_accepted(state_artifacts):
    worker, a, stage_ref, data = state_artifacts
    result = worker.check_stage(a, stage_ref)
    assert result["state_gate_passed"] is False and result == data[stage_ref["path"]]


@pytest.mark.parametrize(
    "target,key,value",
    [
        ("sample", "actual_requests", 98),
        ("sample", "model_initialization_attempts", 2),
        ("sample", "future_label_reads", 1),
        ("sample", "readout_ref", ref("other_w2")),
        ("verified", "zero_fullpoint_nodes_verified", 104),
        ("verified", "numerical_contract_passed", False),
    ],
)
def test_incomplete_state_or_online_leak_rejected(state_artifacts, target, key, value):
    worker, a, stage_ref, data = state_artifacts
    data[target][key] = value
    with pytest.raises(ValueError):
        worker.check_stage(a, stage_ref)
