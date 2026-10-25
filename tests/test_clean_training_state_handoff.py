"""048 production bridge routing and guards, without W2 or STATE execution."""

import ast
import copy
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from s2a import clean_training_state_handoff as state
from s2a.clean_training_import_bridge import ENV_PATH, ENV_SHA, load_frozen_delegate, metadata_import_scope
from s2a.durable_stage_supervisor import REPO, reference


def ref(name):
    return dict(path=str(name), sha256="a" * 64)


@pytest.fixture(autouse=True)
def script_path(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))


def test_production_bridge_matches_fixed_source_template():
    prepare = importlib.import_module("prepare_wam_clean_training_state_handoff")
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
    a, registered, prepared, modules, parent = bridge_scene
    with metadata_import_scope(
        bridge_directory=state.BRIDGE_DIRECTORY,
        bundle_directory=state.BUNDLE,
        module_refs=modules,
        registration_ref=registered,
    ) as (bridge, scope):
        old_prior = bridge._original.prior
        owner = load_frozen_delegate(
            reference(state.BUNDLE / "run_wam_clean_training_state.py"), name="_s2a048_test_actual_state_owner"
        )
        command, w2 = owner.planned_worker(prepared, a["prepared_ref"], a["execution_registration_ref"])
        assert w2 == parent and Path(command[1]).resolve() == state.BUNDLE / "sample_wam_clean_training_state.py"
        assert command[command.index("--w2-fit") + 1] == a["w2_fit_ref"]["path"]
        sampler = load_frozen_delegate(
            reference(state.BUNDLE / "sample_wam_clean_training_state.py"), name="_s2a048_test_actual_sampler"
        )
        objects = {
            a["w2_fit_ref"]["path"]: dict(
                rounds=["D0", "D1"],
                actual_rows=840000,
                collection_refs=[{}, {}],
                capture_audit_refs=[{}, {}],
                fit_metadata=dict(replay_sha256=dict(features="f", weights="w", target_clean_residual="t")),
            ),
            a["w2_verification_ref"]["path"]: dict(
                actual_rows=840000,
                independent_reconstructions=1,
                full_training_row_reconstruction=True,
                qr_metadata=dict(replay_sha256=dict(features="f", weights="w", targets="t")),
            ),
            "bridge_qualification": dict(complete=True, passed=True, bridge_qualified=True),
        }
        monkeypatch.setattr(sampler, "read", lambda r: objects[r["path"]])
        monkeypatch.setattr(
            sampler, "trained_head", lambda *args, **kwargs: (SimpleNamespace(center=[0] * 1290), a["w2_params_ref"])
        )
        monkeypatch.setattr(sampler, "sampling_entries", lambda *args: [{}] * 99)
        _, params, _, _, entries = sampler.admit_sampling(
            prepared, a["prepared_ref"], a["execution_registration_ref"], a["w2_fit_ref"], a["w2_verification_ref"]
        )
        assert params == a["w2_params_ref"] and len(entries) == 99
        assert bridge._original.prior is old_prior
        assert bridge.prior.__code__.co_filename == str(state.BRIDGE_PATH)
    assert scope["restored"] is scope["source_guard_completed"] is True
    receipts = state.dry_receipts()
    assert len(receipts) == 2
    assert {r["caller_function"] for r in receipts} == {"planned_worker", "admit_sampling"}
    assert all(r["dry_run"] is True and r["bridge_ref"] == a["bridge_ref"] for r in receipts)
    assert not (state.ROOT / "role_receipts").exists()


def test_gpu_environment_safe_path_and_independent_audit_unbridged(bridge_scene, monkeypatch):
    _, registered, _, modules, _ = bridge_scene
    with metadata_import_scope(
        bridge_directory=state.BRIDGE_DIRECTORY,
        bundle_directory=state.BUNDLE,
        module_refs=modules,
        registration_ref=registered,
    ):
        monkeypatch.delenv(state.DRY_ENV)
        base = dict(CUDA_VISIBLE_DEVICES=state.GPU, LD_LIBRARY_PATH="", PYTHONPATH=str(REPO / "src"))
        model = state.model_environment(base)
        assert model["PYTHONSAFEPATH"] == "1" and model["PYTHONPATH"].split(":")[:2] == [
            str(state.BRIDGE_DIRECTORY),
            str(state.BUNDLE),
        ]
        cpu = dict(
            base,
            CUDA_VISIBLE_DEVICES="",
            PYTHONSAFEPATH="1",
            **{ENV_PATH: registered["path"], ENV_SHA: registered["sha256"], state.DRY_ENV: "1"},
        )
        audit = state.audit_environment(cpu)
        assert audit["CUDA_VISIBLE_DEVICES"] == "" and audit["PYTHONPATH"] == base["PYTHONPATH"]
        assert all(k not in audit for k in (ENV_PATH, ENV_SHA, state.DRY_ENV, "PYTHONSAFEPATH"))
        with pytest.raises(ValueError, match="inherited bridge"):
            state.audit_environment(dict(model, CUDA_VISIBLE_DEVICES=""))


def test_dry_model_environment_is_refused(bridge_scene):
    _, registered, _, modules, _ = bridge_scene
    with metadata_import_scope(
        bridge_directory=state.BRIDGE_DIRECTORY,
        bundle_directory=state.BUNDLE,
        module_refs=modules,
        registration_ref=registered,
    ):
        with pytest.raises(ValueError, match="model environment"):
            state.model_environment(dict(CUDA_VISIBLE_DEVICES=state.GPU, LD_LIBRARY_PATH=""))


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
    import os
    import subprocess

    prepared = json.loads((state.PREPARED_ROOT / "preparation/prepared.json").read_text())
    python = str(REPO / ".venv/bin/python") if runtime == "main" else prepared["model_config"]["model_python"]
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        LD_LIBRARY_PATH="",
        OMP_NUM_THREADS="8",
        OPENBLAS_NUM_THREADS="8",
        MKL_NUM_THREADS="8",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONSAFEPATH="1",
        PYTHONVERBOSE="1",
        PYTHONPATH=":".join(map(str, (state.BRIDGE_DIRECTORY, state.BUNDLE, REPO / "src", REPO / "scripts"))),
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
    (tmp_path / f"{runtime}_help_stdout.log").write_text(out.stdout)
    (tmp_path / f"{runtime}_help_imports.log").write_text(out.stderr)
    assert not (state.ROOT / "role_receipts").exists()


@pytest.mark.parametrize("fault", ["none", "exit7", "source_drift"])
def test_state_supervisor_real_cpu_child_and_false_final_guard(tmp_path, monkeypatch, fault):
    import json

    from s2a import durable_stage_supervisor as durable

    owner = importlib.import_module("run_wam_clean_training_state_handoff")
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
    worker = importlib.import_module("execute_wam_clean_training_state_handoff")
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
