"""030 owner admission and real CPU-child exit/quality separation."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]


def load_owner():
    sys.path.insert(0, str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "native_training_owner_test", REPO / "scripts/run_wam_native_training.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def admitted(tmp_path, monkeypatch):
    owner = load_owner()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(owner, "REPO", tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    root = tmp_path / "outputs/native"
    root.mkdir(parents=True)
    prepared = dict(
        output_root=str(root),
        records=[dict(task_index=i) for i in range(3) for _ in range(10)],
        config=dict(simulation_python="outputs/simulator/bin/python"),
    )
    path = root / "prepared.json"
    path.write_text(json.dumps(prepared))

    def read_prepared(p, digest):
        if owner.sha256(p) != digest:
            raise ValueError("fixture preparation SHA changed")
        return json.loads(Path(p).read_text())

    monkeypatch.setattr(owner, "read_prepared", read_prepared)
    peer = root / "peer.json"
    peer.write_text(
        json.dumps(
            dict(
                schema_version="s2a.native-training-independent-verification.v1",
                phase="prepare",
                passed=True,
                preparation_verified=True,
                prepared=owner.reference(path),
                input_sha256={str(path): owner.sha256(path)},
                model_training_authorized=False,
                fresh_performance_authorized=False,
            )
        )
    )
    args = SimpleNamespace(
        prepared=path,
        prepared_sha256=owner.sha256(path),
        source_peer=peer,
        source_peer_sha256=owner.sha256(peer),
        dry_run=False,
        seed=42,
    )
    return owner, args, prepared, root


def miniature_commands(owner, args, prepared, tmp_path, *, qualified, fail_task=None):
    """Real processes test ownership only; these are not physics/audit evidence."""
    script = tmp_path / "miniature_child.py"
    script.write_text(
        "import hashlib,json,os,sys,time\n"
        "from pathlib import Path\n"
        "root,prep,mode,qual,fail=sys.argv[1:]\n"
        "root=Path(root);prep=Path(prep)\n"
        "assert os.environ['CUDA_VISIBLE_DEVICES']==os.environ['LD_LIBRARY_PATH']==''\n"
        "assert os.environ['PYTHONDONTWRITEBYTECODE']=='1'\n"
        "ref=lambda p:dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())\n"
        "if mode!='audit':\n"
        " i=int(mode);folder=root/f'collection/task{i}';folder.mkdir()\n"
        " (folder/'worker_started.json').write_text(json.dumps(dict(pid=os.getpid(),ppid=os.getppid(),cuda='',ld='')))\n"
        " time.sleep(.15)\n"
        " if mode==fail:sys.exit(7)\n"
        " result=dict(schema_version='s2a.native-training-task.v1',complete=True,passed=True,task_index=i,"
        "prepared_ref=ref(prep),cases=[{}]*10,environment_initializations=10,current_canvas_images=50,physics_steps=830,"
        "wam_initializations=0,denoise_calls=0,fits=0,fresh_cases=0,native_quality_passed=(qual=='yes'))\n"
        " (folder/'result.json').write_text(json.dumps(result))\n"
        "else:\n"
        " pub=root/'collection/result.json';data=json.loads(pub.read_text());assert len(data['task_results'])==3\n"
        " for item in data['task_results']:\n"
        "  end=json.loads(Path(item['lifecycle']['exit']['path']).read_text());assert end['actual_wait'] and end['returncode']==0\n"
        " result=dict(schema_version='s2a.native-training-independent-verification.v1',phase='collection',passed=True,"
        "collection_verified=True,prepared=ref(prep),input_sha256={str(pub.resolve()):ref(pub)['sha256']},"
        "eligible_for_model_training_registration=(qual=='yes'),model_training_authorized=False,fresh_performance_authorized=False)\n"
        " (root/'review').mkdir();(root/'review/collection_verified.json').write_text(json.dumps(result))\n"
    )
    common = [sys.executable, str(script), prepared["output_root"], str(args.prepared)]
    suffix = ["yes" if qualified else "no", str(fail_task)]
    return [common + [str(i)] + suffix for i in range(3)], common + ["audit"] + suffix


def test_dry_run_preserves_simulator_venv_and_never_creates_collection(tmp_path, monkeypatch, capsys):
    owner, args, prepared, root = admitted(tmp_path, monkeypatch)
    args.dry_run = True
    monkeypatch.setattr(owner, "owned_command", lambda *a, **k: pytest.fail("dry run launched a child"))
    owner.execute(args)
    result = json.loads(capsys.readouterr().out)
    assert [cmd[cmd.index("--task-index") + 1] for cmd in result["workers"]] == ["0", "1", "2"]
    assert all(cmd[0] == str(tmp_path / "outputs/simulator/bin/python") for cmd in result["workers"])
    assert not (root / "collection").exists() and not (root / "orchestration").exists()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    assert owner.cpu_environment()["CUDA_VISIBLE_DEVICES"] == ""


def test_unbound_metadata_peer_rejected_before_launch(tmp_path, monkeypatch):
    owner, args, prepared, root = admitted(tmp_path, monkeypatch)
    data = json.loads(args.source_peer.read_text())
    data["input_sha256"] = {}
    args.source_peer.write_text(json.dumps(data))
    args.source_peer_sha256 = owner.sha256(args.source_peer)
    monkeypatch.setattr(owner, "owned_command", lambda *a, **k: pytest.fail("invalid peer launched a child"))
    with pytest.raises(ValueError, match="independent metadata"):
        owner.execute(args)
    assert not (root / "collection").exists()


def test_existing_collection_cannot_be_replayed(tmp_path, monkeypatch):
    owner, args, prepared, root = admitted(tmp_path, monkeypatch)
    (root / "collection").mkdir()
    monkeypatch.setattr(owner, "owned_command", lambda *a, **k: pytest.fail("existing run launched a child"))
    with pytest.raises(ValueError, match="refuses existing"):
        owner.execute(args)
    assert not (root / "orchestration").exists()


@pytest.mark.parametrize("qualified", [False, True])
def test_real_waits_precede_publication_and_quality_is_separate(tmp_path, monkeypatch, qualified):
    owner, args, prepared, root = admitted(tmp_path, monkeypatch)
    commands = miniature_commands(owner, args, prepared, tmp_path, qualified=qualified)
    monkeypatch.setattr(owner, "planned_commands", lambda *a: commands)
    owner.execute(args)
    result = json.loads((root / "orchestration/collection/result.json").read_text())
    assert result["passed"] is result["complete"] is True
    assert result["eligible_for_model_training_registration"] is qualified
    assert result["model_training_authorized"] is result["fresh_performance_authorized"] is False
    assert len(result["owned_commands"]) == len(result["command_attempts"]) == 4
    for command in result["owned_commands"]:
        exited = json.loads(Path(command["exit"]["path"]).read_text())
        release = json.loads(Path(command["release"]["path"]).read_text())
        assert exited["actual_wait"] and exited["returncode"] == 0
        assert release["reaped"] and not release["signal_sent"]
    assert result["physics_steps"] == 2490  # Declared miniature budget, not a MuJoCo assertion.


def test_failed_actual_child_is_preserved_without_launching_remaining_tasks(tmp_path, monkeypatch):
    owner, args, prepared, root = admitted(tmp_path, monkeypatch)
    commands = miniature_commands(owner, args, prepared, tmp_path, qualified=False, fail_task=1)
    monkeypatch.setattr(owner, "planned_commands", lambda *a: commands)
    with pytest.raises(ChildProcessError):
        owner.execute(args)
    result = json.loads((root / "orchestration/collection/result.json").read_text())
    assert result["complete"] is result["passed"] is False
    assert len(result["owned_commands"]) == 1 and len(result["command_attempts"]) == 2
    failed = result["command_attempts"][1]["observed_lifecycle"]
    assert json.loads(Path(failed["exit"]["path"]).read_text())["returncode"] == 7
    release = json.loads(Path(failed["release"]["path"]).read_text())
    assert release["returncode"] == 7 and release["reaped"] and not release["signal_sent"]
    assert not (root / "collection/task2").exists() and not (root / "collection/result.json").exists()
