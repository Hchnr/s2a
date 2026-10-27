"""051 role binding; independent payload proof is separate from these fixtures."""

import copy
import sys
from pathlib import Path

import pytest

from s2a import development_repair_admission as repair
from s2a.durable_stage_supervisor import read, reference


@pytest.fixture
def role(tmp_path, monkeypatch):
    root = tmp_path / "repair"
    bundle = root / "source_runtime_py311"
    bundle.mkdir(parents=True)
    monkeypatch.setattr(repair, "ROOT", root)
    monkeypatch.setattr(repair, "BUNDLE", bundle)
    q = dict(source_sha256={"synthetic": "explicit-mock-of-independent-admission"})
    monkeypatch.setattr(repair, "admit", lambda ref: q)
    monkeypatch.setenv(repair.ENV_PATH, str(root / "execution/admitted.json"))
    monkeypatch.setenv(repair.ENV_SHA, "a" * 64)
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    return root, bundle, q


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("run_wam_clean_training_development.py", "owner"),
        ("serve_wam_clean_training.py", "model"),
        ("execute_wam_clean_training_case.py", "client_03"),
        ("verify_wam_clean_training_development.py", "audit"),
    ],
)
def test_actual_role_receipt_and_repeated_same_birth(role, monkeypatch, filename, expected):
    root, bundle, q = role
    path = bundle / filename
    path.write_text("# synthetic metadata role\n")
    monkeypatch.setattr(sys, "argv", [str(path), "--case-index", "3"])
    if expected == "model":
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", repair.GPU)
    before = copy.deepcopy(q)
    assert repair.current_qualification() == q
    ref = reference(root / "role_receipts" / (expected + ".json"))
    receipt = read(ref)
    assert receipt["role"] == expected and receipt["entrypoint_ref"] == reference(path)
    assert repair.current_qualification() == q and reference(ref["path"]) == ref and q == before


@pytest.mark.parametrize("fault", ("wrong_gpu", "wrong_entry", "wrong_case", "different_birth"))
def test_wrong_role_environment_or_identity_rejected(role, monkeypatch, fault):
    root, bundle, _ = role
    path = bundle / "execute_wam_clean_training_case.py"
    path.write_text("# role\n")
    monkeypatch.setattr(sys, "argv", [str(path), "--case-index", "12" if fault == "wrong_case" else "0"])
    if fault == "wrong_gpu":
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    if fault == "wrong_entry":
        monkeypatch.setattr(sys, "argv", [str(root / "unregistered.py")])
    if fault == "different_birth":
        repair.current_qualification()
        monkeypatch.setattr(repair, "identity", lambda pid: dict(pid=pid, start_ticks=-1, boot_id="wrong"))
    with pytest.raises(ValueError):
        repair.current_qualification()


def test_actual_owner_dry_does_not_consume_future_role_receipt(role, monkeypatch):
    root, bundle, _ = role
    path = bundle / "run_wam_clean_training_development.py"
    path.write_text("# role\n")
    monkeypatch.setattr(sys, "argv", [str(path), "--dry-run"])
    repair.current_qualification()
    assert not (root / "role_receipts").exists()


@pytest.mark.parametrize(
    "key,value",
    [("requests", 901), ("physics_steps", 1), ("fresh_cases", 1), ("cumulative_development_model_initializations", 3)],
)
def test_scope_budget_fails_before_any_upstream_reads(monkeypatch, key, value):
    p = dict(
        schema_version=repair.PREFIX + "proposal.v1",
        complete=True,
        passed=True,
        execution_authorized=False,
        root=str(repair.ROOT),
        stage_scope="DEVELOPMENT_only",
        budget=copy.deepcopy(repair.BUDGET),
        gpu_uuid=repair.GPU,
        automatic_retry=False,
    )
    p["budget"][key] = value

    class Evidence:
        def json(self, ref):
            return p

    monkeypatch.setattr(repair, "Evidence", Evidence)
    with pytest.raises(ValueError, match="finite development"):
        repair.validate({})


@pytest.mark.parametrize("fault", ("none", "exit7", "source_drift"))
def test_repair_supervisor_actual_cpu_child_wait_and_false_guard(tmp_path, monkeypatch, fault):
    import importlib
    import json

    from s2a import durable_stage_supervisor as durable

    owner = importlib.import_module("run_wam_clean_development_repair")
    root = tmp_path / "repair_owner"
    root.mkdir()
    monkeypatch.setattr(owner, "ROOT", root)
    script = root / "tiny.py"
    script.write_text("""import sys,time
from pathlib import Path
from s2a.durable_stage_supervisor import publish,identity
time.sleep(.08)
root=Path(sys.argv[1])
if sys.argv[2]=='exit7':raise SystemExit(7)
publish(root/'result.json',dict(score=dict(promotion_gate=False),synthetic=True))
publish(root/'role_receipts/owner.json',dict(identity=identity(__import__('os').getpid()),synthetic=True))
""")
    sources = {str(p): durable.sha(p) for p in (script, durable.REPO / owner.ENTRY)}
    a = dict(source_sha256=sources, snapshot_sha256=sources)
    ref = durable.publish(root / "admitted.json", dict(synthetic=True))
    launch = durable.publish(root / "launch.json", dict(synthetic=True))
    monkeypatch.setattr(owner, "admit", lambda ref: a)
    monkeypatch.setattr(owner, "command", lambda a: [sys.executable, str(script), str(root), fault])
    monkeypatch.setattr(owner, "environment", lambda ref: durable.cpu_environment())
    monkeypatch.setattr(owner, "accept_detached_launch", lambda *args, **kw: launch)
    monkeypatch.setattr(owner, "check_result", lambda a, r: durable.reference(root / "result.json"))
    monkeypatch.setattr(
        owner, "observe_descendant_cleanup", lambda p: dict(synthetic=True, all_recorded_descendants_reaped=True)
    )
    for key in durable.ENV_KEYS:
        monkeypatch.setenv(key, durable.cpu_environment()[key])
    if fault == "source_drift":

        def changed(s):
            raise ValueError("synthetic source drift")

        monkeypatch.setattr(owner, "check_sources", changed)
    if fault == "none":
        result = durable.read(owner.execute(ref, mode="run"))
        assert result["passed"] and not result["score"]["promotion_gate"]
    else:
        with pytest.raises(Exception):
            owner.execute(ref, mode="run")
        assert json.loads((root / "supervisor/result.json").read_text())["passed"] is False
    exited = json.loads((root / "supervisor/DEVELOPMENT/exit.json").read_text())
    released = json.loads((root / "supervisor/DEVELOPMENT/release.json").read_text())
    assert exited["actual_wait"] is released["actual_wait"] is released["reaped"] is True
    assert released["signal_sent"] is False and released["returncode"] == exited["returncode"] == (
        7 if fault == "exit7" else 0
    )


@pytest.mark.parametrize(
    "name,runtime",
    [
        ("run_wam_clean_training_development.py", "main"),
        ("serve_wam_clean_training.py", "model"),
        ("execute_wam_clean_training_case.py", "sim"),
        ("verify_wam_clean_training_development.py", "main"),
    ],
)
def test_actual_generated_entrypoint_help_uses_repair_runtime_without_inference(tmp_path, name, runtime):
    import os
    import subprocess

    from s2a.clean_training_state_handoff import BUNDLE as original_bundle
    from s2a.durable_stage_supervisor import REPO

    old = read(repair.OLD_REF)
    prepared = read(old["prepared_ref"])
    python = (
        prepared["model_config"]["model_python"]
        if runtime == "model"
        else str(REPO / ("outputs/libero_env311/bin/python" if runtime == "sim" else ".venv/bin/python"))
    )
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        LD_LIBRARY_PATH="",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONVERBOSE="1",
        OMP_NUM_THREADS="8",
        OPENBLAS_NUM_THREADS="8",
        MKL_NUM_THREADS="8",
        PYTHONPATH=os.pathsep.join(map(str, (repair.BUNDLE, original_bundle, REPO / "src", REPO / "scripts"))),
    )
    out = subprocess.run(
        [python, str(repair.BUNDLE / name), "--help"], cwd=REPO, env=env, capture_output=True, text=True, timeout=60
    )
    assert out.returncode == 0, out.stderr[-2000:]
    expected_import = (
        Path(repair.__file__)
        if name == "verify_wam_clean_training_development.py"
        else repair.BUNDLE / "continuation_runtime.py"
    )
    assert "usage:" in out.stdout and str(expected_import) in out.stderr
    assert "import 'torch'" not in out.stderr and "import 'mujoco'" not in out.stderr
    (tmp_path / (name + ".imports.log")).write_text(out.stderr)


def test_actual_preflight_external_readonly_inputs_are_hash_checked():
    preflight = read(reference(repair.ROOT / "preflight_runtime_py311/all_case_inputs.json"))
    external = {
        p: d
        for row in preflight["records"]
        for p, d in row["input_sha256"].items()
        if not Path(p).is_relative_to(repair.REPO)
    }
    assert external
    e = repair.Evidence()
    e.mapping(external)
    e.finish()
    changed = dict(external)
    changed[next(iter(changed))] = "0" * 64
    with pytest.raises(ValueError):
        repair.Evidence().mapping(changed)
    with pytest.raises(ValueError, match="forbidden old workspace"):
        repair.Evidence().file(dict(path="/share/project/eai_pwm/home/hcr/repos/s2a/forbidden.json", sha256="a" * 64))


def test_final_config_cannot_change_the_proposal_scope(monkeypatch):
    p = dict(
        schema_version=repair.PREFIX + "proposal.v1",
        complete=True,
        passed=True,
        execution_authorized=False,
        root=str(repair.ROOT),
        stage_scope="DEVELOPMENT_only",
        budget=repair.BUDGET,
        gpu_uuid=repair.GPU,
        automatic_retry=False,
        old049_ref=repair.OLD_REF,
        state_audit_ref=repair.STATE_AUDIT_REF,
        failure_audit_ref=repair.FAILURE_REF,
        config_ref=dict(path="synthetic-config", sha256="a" * 64),
    )

    class Evidence:
        def json(self, ref):
            return dict(stage_scope="FRESH") if ref == p["config_ref"] else p

    monkeypatch.setattr(repair, "Evidence", Evidence)
    with pytest.raises(ValueError, match="final config differs"):
        repair.validate({})
