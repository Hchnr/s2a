"""Continuation ownership tests use real CPU children, never physics evidence."""

import importlib.util
import json
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]


def admitted(tmp_path, monkeypatch):
    sys.path.insert(0, str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "native_continuation_owner_test", REPO / "scripts/run_wam_native_training_continuation.py"
    )
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    import run_wam_readout

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(owner, "REPO", tmp_path)
    monkeypatch.setattr(run_wam_readout, "REPO", tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    root = tmp_path / "outputs/continuation"
    root.mkdir(parents=True)
    original = tmp_path / "original_failed_task.json"
    original.write_text(json.dumps(dict(passed=False, complete=False, actual_worker_returncode=1)))
    old_prepared = tmp_path / "original_prepared.json"
    old_prepared.write_text("{}")
    records = [
        dict(task_index=i, name=f"case{i}_{j}", native_steps=last if j == 9 else typical)
        for i, typical, last in ((0, 90, 118), (1, 180, 194), (2, 100, 138))
        for j in range(10)
    ]
    prepared = dict(
        schema_version="s2a.native-training-continuation-preparation.v1",
        seed=42,
        output_root=str(root),
        original_config=dict(simulation_python="outputs/simulator/bin/python"),
        original_prepared_ref=owner.reference(old_prepared),
        failed_task_ref=owner.reference(original),
        records=records,
        cached_records=records[:10],
        remaining_records=records[10:],
        remaining_tasks=[1, 2],
        original_task_passed=False,
        original_owner_passed=False,
        cached_quality_verified=False,
        model_training_authorized=False,
        fresh_performance_authorized=False,
    )
    path = root / "prepared.json"
    path.write_text(json.dumps(prepared))

    def read_prepared(p, digest):
        if owner.reference(p)["sha256"] != digest:
            raise ValueError("fixture preparation SHA changed")
        return json.loads(Path(p).read_text())

    monkeypatch.setattr(owner, "read_prepared", read_prepared)
    args = SimpleNamespace(prepared=path, prepared_sha256=owner.reference(path)["sha256"], dry_run=False, seed=42)
    return owner, args, prepared, root


def miniature_commands(args, prepared, tmp_path, *, qualified, fail=None):
    """Synthetic receipts isolate actual ordering, startup, wait and failure cleanup."""
    script = tmp_path / "miniature_continuation_child.py"
    script.write_text(
        textwrap.dedent("""\
        import hashlib,json,os,sys,time
        from pathlib import Path
        root,prep,mode,qual,fail=sys.argv[1:]
        root=Path(root);prep=Path(prep);metadata=json.loads(prep.read_text())
        assert os.environ['CUDA_VISIBLE_DEVICES']==os.environ['LD_LIBRARY_PATH']==''
        assert os.environ['PYTHONDONTWRITEBYTECODE']=='1'
        ref=lambda p:dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        with (root/'order.txt').open('a') as f:f.write(mode+'\\n')
        common=dict(schema_version='s2a.native-training-continuation-independent-verification.v1',
                    phase=mode,passed=True,prepared=ref(prep),input_sha256={str(prep):ref(prep)['sha256']},
                    model_training_authorized=False,fresh_performance_authorized=False)
        if mode=='prefix':
            if mode==fail:sys.exit(7)
            result=dict(common,prefix_verified=True,case_verification=[{}]*10,cached_case_refs=[{}]*10,
                        original_task_passed=False,original_owner_passed=False,original_worker_returncode=1,
                        eligible_for_model_training_registration=False)
            (root/'review/prefix_verified.json').write_text(json.dumps(result))
        elif mode in ('1','2'):
            assert (root/'review/prefix_verified.json').is_file()
            i=int(mode);folder=root/f'collection/task{i}';folder.mkdir()
            (folder/'worker_started.json').write_text(json.dumps(dict(pid=os.getpid(),ppid=os.getppid(),cuda='',ld='')))
            time.sleep(.15)
            if mode==fail:sys.exit(7)
            readiness=dict(schema_version='s2a.native-training-environment-readiness.v1',passed=True,complete=True,
                           continuation_prepared_ref=ref(prep),restore_performed=True,
                           after_restore=dict(CUDA_VISIBLE_DEVICES='',LD_LIBRARY_PATH='',cwd=os.getcwd()))
            (folder/'environment_readiness.json').write_text(json.dumps(readiness))
            result=dict(schema_version='s2a.native-training-continuation-task.v1',complete=True,passed=True,
                        prepared_ref=metadata['original_prepared_ref'],continuation_prepared_ref=ref(prep),task_index=i,
                        environment_readiness_ref=ref(folder/'environment_readiness.json'),cases=[{}]*10,
                        environment_initializations=10,current_canvas_images=50,physics_steps={1:2214,2:1438}[i],
                        cached_cases_recollected=0,wam_initializations=0,denoise_calls=0,fits=0,fresh_cases=0,
                        future_rgb_frames=0,source_hdf5_reads=0,native_quality_passed=(qual=='yes'))
            (folder/'result.json').write_text(json.dumps(result))
        else:
            pub=root/'collection/result.json';data=json.loads(pub.read_text())
            assert [t['task_index'] for t in data['task_results']]==[1,2]
            assert data['cached_task_ref']==metadata['failed_task_ref']
            assert json.loads(Path(data['cached_task_ref']['path']).read_text())['passed'] is False
            for item in data['task_results']:
                end=json.loads(Path(item['lifecycle']['exit']['path']).read_text())
                release=json.loads(Path(item['lifecycle']['release']['path']).read_text())
                assert end['actual_wait'] and end['returncode']==0 and release['reaped'] and not release['signal_sent']
            common['input_sha256'][str(pub)]=ref(pub)['sha256']
            result=dict(common,collection_verified=True,case_verification=[{}]*30,physics_steps=4980,
                        new_collection_physics_steps=3652,environment_initializations=30,current_canvas_images=150,
                        prefix_verification_ref=ref(root/'review/prefix_verified.json'),
                        eligible_for_model_training_registration=(qual=='yes'))
            (root/'review/collection_verified.json').write_text(json.dumps(result))
        """)
    )
    common = [sys.executable, str(script), prepared["output_root"], str(args.prepared)]
    suffix = ["yes" if qualified else "no", str(fail)]
    return common + ["prefix"] + suffix, [common + [str(i)] + suffix for i in (1, 2)], common + ["collection"] + suffix


def test_dry_run_only_schedules_remaining_tasks_and_keeps_venv(tmp_path, monkeypatch, capsys):
    owner, args, prepared, root = admitted(tmp_path, monkeypatch)
    args.dry_run = True
    monkeypatch.setattr(owner, "attempt", lambda *a, **k: pytest.fail("dry run launched child"))
    owner.execute(args)
    plan = json.loads(capsys.readouterr().out)
    assert [command[command.index("--task-index") + 1] for command in plan["workers"]] == ["1", "2"]
    assert all(command[0] == str(tmp_path / "outputs/simulator/bin/python") for command in plan["workers"])
    assert "--phase" in plan["prefix"] and "prefix" in plan["prefix"]
    assert not any((root / name).exists() for name in ("collection", "review", "orchestration"))


def test_wrong_sha_or_existing_output_cannot_launch_a_child(tmp_path, monkeypatch):
    owner, args, prepared, root = admitted(tmp_path, monkeypatch)
    monkeypatch.setattr(owner, "attempt", lambda *a, **k: pytest.fail("invalid admission launched child"))
    actual = args.prepared_sha256
    args.prepared_sha256 = "0" * 64
    with pytest.raises(ValueError, match="SHA changed"):
        owner.execute(args)
    args.prepared_sha256 = actual
    (root / "collection").mkdir()
    with pytest.raises(ValueError, match="refuses existing"):
        owner.execute(args)
    assert not (root / "orchestration").exists()


@pytest.mark.parametrize("qualified", [False, True])
def test_four_actual_children_keep_cached_failure_and_separate_native_quality(tmp_path, monkeypatch, qualified):
    owner, args, prepared, root = admitted(tmp_path, monkeypatch)
    original_sha = owner.reference(prepared["failed_task_ref"]["path"])
    commands = miniature_commands(args, prepared, tmp_path, qualified=qualified)
    monkeypatch.setattr(owner, "planned_commands", lambda *a: commands)
    owner.execute(args)
    result = json.loads((root / "orchestration/collection/result.json").read_text())
    assert result["passed"] is result["complete"] is True
    assert result["eligible_for_model_training_registration"] is qualified
    assert result["original_task_passed"] is result["original_owner_passed"] is False
    assert result["model_training_authorized"] is result["fresh_performance_authorized"] is False
    assert (root / "order.txt").read_text().splitlines() == ["prefix", "1", "2", "collection"]
    assert len(result["owned_commands"]) == len(result["command_attempts"]) == 4
    for item in result["owned_commands"]:
        end = json.loads(Path(item["exit"]["path"]).read_text())
        release = json.loads(Path(item["release"]["path"]).read_text())
        assert end["actual_wait"] and end["returncode"] == 0
        assert release["reaped"] and not release["signal_sent"]
    assert not (root / "collection/task0").exists()
    assert owner.reference(prepared["failed_task_ref"]["path"]) == original_sha
    assert result["physics_steps"] == 4980 and result["new_collection_physics_steps"] == 3652


@pytest.mark.parametrize("failed_mode,attempts,completed", [("prefix", 1, 0), ("1", 2, 1)])
def test_actual_nonzero_exit_stops_remaining_work_without_retry(
    tmp_path, monkeypatch, failed_mode, attempts, completed
):
    owner, args, prepared, root = admitted(tmp_path, monkeypatch)
    commands = miniature_commands(args, prepared, tmp_path, qualified=False, fail=failed_mode)
    monkeypatch.setattr(owner, "planned_commands", lambda *a: commands)
    with pytest.raises(ChildProcessError):
        owner.execute(args)
    result = json.loads((root / "orchestration/collection/result.json").read_text())
    assert result["passed"] is result["complete"] is False
    assert len(result["command_attempts"]) == attempts and len(result["owned_commands"]) == completed
    failed = result["command_attempts"][-1]["observed_lifecycle"]
    assert json.loads(Path(failed["exit"]["path"]).read_text())["returncode"] == 7
    release = json.loads(Path(failed["release"]["path"]).read_text())
    assert release["reaped"] and release["returncode"] == 7 and not release["signal_sent"]
    assert not (root / "collection/result.json").exists()
    assert not (root / "collection/task2").exists()
