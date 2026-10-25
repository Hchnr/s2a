"""Execution-boundary checks for the fixed development continuation."""

import os

import pytest

from s2a import clean_training_development_handoff as dev


def test_development_environment_drops_state_bridge_and_keeps_cpu_limits(monkeypatch):
    for name in ("S2A_STATE_HANDOFF_PATH", "S2A_STATE_HANDOFF_SHA256", "S2A_STATE_HANDOFF_DRY_RUN", "PYTHONSAFEPATH"):
        monkeypatch.setenv(name, "inherited-state")
    monkeypatch.setenv("PYTHONPATH", "inherited-bridge")
    env = dev.environment()
    assert env["CUDA_VISIBLE_DEVICES"] == env["LD_LIBRARY_PATH"] == ""
    assert all(env[key] == "8" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"))
    assert env["PYTHONPATH"].split(os.pathsep) == list(map(str, (dev.BUNDLE, dev.REPO / "src", dev.REPO / "scripts")))
    assert not any(key.startswith("S2A_STATE_HANDOFF") or key == "PYTHONSAFEPATH" for key in env)


@pytest.mark.parametrize(
    "output", ["supervisor", "development", "orchestration/DEVELOPMENT", "review/DEVELOPMENT_verified.json"]
)
def test_any_existing_actual_output_prevents_retry(tmp_path, monkeypatch, output):
    control = tmp_path / "control"
    prepared = tmp_path / "prepared"
    monkeypatch.setattr(dev, "ROOT", control)
    monkeypatch.setattr(dev, "PREPARED_ROOT", prepared)
    dev.absent_outputs()
    path = (control if output == "supervisor" else prepared) / output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    with pytest.raises(ValueError, match="no automatic retry"):
        dev.absent_outputs()


def test_owner_command_passes_all_four_independent_sha_bound_inputs():
    keys = ("prepared_ref", "execution_registration_ref", "state_owner_ref", "state_verification_ref")
    config = {key: dict(path=f"/test/{key}.json", sha256=str(i) * 64) for i, key in enumerate(keys)}
    args = dev.command(config)
    assert args[:2] == [str(dev.REPO / ".venv/bin/python"), str(dev.OWNER_PATH)]
    for name, key in zip(
        ("prepared", "execution-registration", "state-owner", "state-verification"), keys, strict=True
    ):
        assert args[args.index("--" + name) + 1] == config[key]["path"]
        assert args[args.index("--" + name + "-sha256") + 1] == config[key]["sha256"]
    assert args[-2:] == ["--seed", "42"]


@pytest.mark.parametrize("fault", ("none", "exit7", "source_drift"))
def test_development_supervisor_actual_child_exit_and_guard(tmp_path, monkeypatch, fault):
    import importlib
    import json
    import sys

    from s2a import durable_stage_supervisor as durable

    owner = importlib.import_module("run_wam_clean_training_development_handoff")
    root = tmp_path / "synthetic049"
    root.mkdir()
    monkeypatch.setattr(owner, "ROOT", root)
    script = root / "tiny.py"
    script.write_text("""import json,sys,time
from pathlib import Path
time.sleep(.08)
root=Path(sys.argv[1])
if sys.argv[2]=="exit7":raise SystemExit(7)
with (root/"result.json").open("x") as stream:json.dump(dict(score=dict(promotion_gate=False),synthetic=True),stream)
""")
    registered = durable.publish(root / "admitted.json", dict(synthetic=True))
    launch = durable.publish(root / "launch.json", dict(synthetic=True))
    sources = {str(p): durable.sha(p) for p in (dev.REPO / owner.ENTRY, script)}
    a = dict(source_sha256=sources, snapshot_sha256=sources, owner_timeout_s=3)
    monkeypatch.setattr(owner, "admit", lambda r: a)
    monkeypatch.setattr(owner, "accept_detached_launch", lambda *args, **kw: launch)
    monkeypatch.setattr(owner, "command", lambda a: [sys.executable, str(script), str(root), fault])
    monkeypatch.setattr(owner, "check_result", lambda a: durable.reference(root / "result.json"))
    monkeypatch.setattr(
        owner, "observe_descendant_cleanup", lambda p: dict(synthetic=True, all_recorded_descendants_reaped=True)
    )
    if fault == "source_drift":

        def changed(sources):
            raise ValueError("synthetic source drift")

        monkeypatch.setattr(owner, "check_sources", changed)
    for key in durable.ENV_KEYS:
        monkeypatch.setenv(key, durable.cpu_environment()[key])
    if fault == "none":
        r = owner.execute(registered, mode="run")
        assert durable.read(r)["passed"] is True
        assert durable.read(r)["score"]["promotion_gate"] is False
    else:
        with pytest.raises(Exception):
            owner.execute(registered, mode="run")
        assert json.loads((root / "supervisor/result.json").read_text())["passed"] is False
    exited = json.loads((root / "supervisor/DEVELOPMENT/exit.json").read_text())
    released = json.loads((root / "supervisor/DEVELOPMENT/release.json").read_text())
    assert exited["actual_wait"] is released["actual_wait"] is released["reaped"] is True
    assert released["signal_sent"] is False
    assert exited["returncode"] == released["returncode"] == (7 if fault == "exit7" else 0)


def test_original_development_help_imports_without_state_bridge_or_model(tmp_path):
    import subprocess

    env = dev.environment()
    env["PYTHONVERBOSE"] = "1"
    result = subprocess.run(
        [
            *dev.command(
                {
                    key: dict(path="unused", sha256="a" * 64)
                    for key in (
                        "prepared_ref",
                        "execution_registration_ref",
                        "state_owner_ref",
                        "state_verification_ref",
                    )
                }
            )[:2],
            "--help",
        ],
        cwd=dev.REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr[-1000:]
    assert "usage:" in result.stdout
    assert "import 'torch'" not in result.stderr and "import 'mujoco'" not in result.stderr
    assert str(dev.BUNDLE / "run_wam_clean_training.py") in result.stderr
    assert str(dev.STATE_ROOT / "import_bridge") not in result.stderr
    (tmp_path / "original_help_imports.log").write_text(result.stderr)


@pytest.mark.parametrize("fault", ("none", "wrong_uuid", "wrong_order", "nonzero_exit"))
def test_dry_requires_actual_original_entry_result(monkeypatch, fault):
    import importlib
    from types import SimpleNamespace

    owner = importlib.import_module("run_wam_clean_training_development_handoff")
    from s2a.durable_stage_supervisor import sha

    sources = {str(dev.REPO / owner.ENTRY): sha(dev.REPO / owner.ENTRY)}
    a = dict(
        source_sha256=sources,
        snapshot_sha256=sources,
        budget=dev.BUDGET,
        case_order=["registered-order"],
        gpu_uuid=dev.GPU,
    )
    monkeypatch.setattr(owner, "admit", lambda ref: a)
    monkeypatch.setattr(owner, "command", lambda a: ["old-python", "original-owner.py"])
    original = dict(
        case_order=a["case_order"], actual_model_gpu_uuid=dev.GPU, model_initializations=0, new_physics_steps=0
    )
    if fault == "wrong_uuid":
        original["actual_model_gpu_uuid"] = "forbidden"
    if fault == "wrong_order":
        original["case_order"] = ["different-order"]
    calls = []

    def run(argv, **kw):
        import json

        calls.append(argv)
        assert kw["env"]["CUDA_VISIBLE_DEVICES"] == ""
        return SimpleNamespace(returncode=7 if fault == "nonzero_exit" else 0, stdout=json.dumps(original), stderr="")

    monkeypatch.setattr(owner.subprocess, "run", run)
    for key in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"):
        monkeypatch.setenv(key, "")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(key, "8")
    if fault == "none":
        assert owner.execute(dict(path="synthetic", sha256="a" * 64), mode="dry-run")["original_dry"] == original
    else:
        with pytest.raises(ValueError):
            owner.execute(dict(path="synthetic", sha256="a" * 64), mode="dry-run")
    assert calls == [["old-python", "original-owner.py", "--dry-run"]]
