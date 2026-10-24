"""Finite static-diagnostic admission and ownership, with real CPU children."""

import importlib.util
import json
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]


def load_owner(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "terminal_predicate_owner_test", REPO / "scripts/run_native_terminal_predicate_diagnostic.py"
    )
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    return owner


def admitted(tmp_path, monkeypatch):
    owner = load_owner(monkeypatch)
    import run_wam_readout

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(owner, "REPO", tmp_path)
    monkeypatch.setattr(run_wam_readout, "REPO", tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    root = tmp_path / "outputs/diagnostic"
    root.mkdir(parents=True)
    prepared = dict(output_root=str(root), config=dict(simulation_python="outputs/simenv/bin/python"))
    path = root / "prepared.json"
    path.write_text(json.dumps(prepared))

    def read_prepared(path, digest):
        owner.cpu_guard()
        if owner.reference(path)["sha256"] != digest:
            raise ValueError("fixture prepared SHA changed")
        return json.loads(Path(path).read_text())

    monkeypatch.setattr(owner, "read_prepared", read_prepared)
    args = SimpleNamespace(prepared=path, prepared_sha256=owner.reference(path)["sha256"], seed=42, dry_run=False)
    return owner, args, prepared, root


def test_budget_keeps_all_sixteen_and_never_promotes_native_quality(monkeypatch):
    owner = load_owner(monkeypatch)
    config = json.loads((REPO / "configs/native_terminal_predicate_diagnostic_20261021.json").read_text())
    owner.budget_guard(config)
    config["cases"][0]["frame_indices"].pop()
    with pytest.raises(ValueError, match="sixteen"):
        owner.budget_guard(config)
    config = json.loads((REPO / "configs/native_terminal_predicate_diagnostic_20261021.json").read_text())
    config["model_training_authorized"] = True
    with pytest.raises(ValueError, match="permissions"):
        owner.budget_guard(config)
    config["model_training_authorized"] = False
    config["budget"]["mj_step_calls"] = 1
    with pytest.raises(ValueError, match="budget"):
        owner.budget_guard(config)


@pytest.mark.parametrize("suffix", [".hdf5", ".png", ".npy", ".safetensors"])
def test_forbidden_source_is_rejected_before_any_content_read(tmp_path, monkeypatch, suffix):
    owner = load_owner(monkeypatch)
    monkeypatch.setattr(owner, "sha256", lambda *a: pytest.fail("forbidden content was opened"))
    with pytest.raises(ValueError, match="refuses HDF/image/expert/weight"):
        owner.check(tmp_path / ("source" + suffix), None, {})


def test_dry_run_keeps_venv_entrypoint_and_does_not_launch(tmp_path, monkeypatch, capsys):
    owner, args, prepared, root = admitted(tmp_path, monkeypatch)
    args.dry_run = True
    monkeypatch.setattr(owner, "owned_command", lambda *a, **k: pytest.fail("dry run launched child"))
    owner.execute(args)
    result = json.loads(capsys.readouterr().out)
    assert result["worker"][0] == str(tmp_path / "outputs/simenv/bin/python")
    assert result["model_calls"] == 0 and result["cuda_visible_devices"] == result["ld_library_path"] == ""
    assert not any((root / p).exists() for p in ("orchestration", "collection", "review"))


def test_bad_sha_or_existing_output_cannot_launch(tmp_path, monkeypatch):
    owner, args, prepared, root = admitted(tmp_path, monkeypatch)
    monkeypatch.setattr(owner, "owned_command", lambda *a, **k: pytest.fail("invalid admission launched child"))
    actual = args.prepared_sha256
    args.prepared_sha256 = "0" * 64
    with pytest.raises(ValueError, match="SHA changed"):
        owner.execute(args)
    args.prepared_sha256 = actual
    (root / "collection").mkdir()
    with pytest.raises(ValueError, match="refuses existing"):
        owner.execute(args)
    assert not (root / "orchestration").exists()


def miniature_commands(tmp_path, args, root, *, fail="none", mismatch=False):
    script = tmp_path / "miniature_diagnostic_child.py"
    script.write_text(
        textwrap.dedent("""\
        import hashlib,json,os,sys
        from pathlib import Path
        root,prep,phase,fail,mismatch=sys.argv[1:]
        root=Path(root);prep=Path(prep)
        assert os.environ['CUDA_VISIBLE_DEVICES']==os.environ['LD_LIBRARY_PATH']==''
        assert os.environ['PYTHONDONTWRITEBYTECODE']=='1'
        assert os.environ['MUJOCO_GL']==os.environ['PYOPENGL_PLATFORM']=='osmesa'
        ref=lambda p:dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        with (root/'order.txt').open('a') as f:f.write(phase+'\\n')
        if phase==fail:sys.exit(7)
        if phase=='worker':
            (root/'collection').mkdir()
            result=dict(schema_version='synthetic.worker',complete=True,prepared_ref=ref(prep))
            (root/'collection/result.json').write_text(json.dumps(result))
        else:
            lifecycle=json.loads((root/'orchestration/run/worker_lifecycle.json').read_text())
            assert lifecycle['collection_ref']==ref(root/'collection/result.json')
            end=json.loads(Path(lifecycle['worker_lifecycle']['exit']['path']).read_text())
            release=json.loads(Path(lifecycle['worker_lifecycle']['release']['path']).read_text())
            assert end['actual_wait'] and end['returncode']==0 and release['reaped'] and not release['signal_sent']
            result=dict(schema_version='s2a.native-terminal-predicate-independent-verification.v1',
                        passed=mismatch=='False',prepared=ref(prep),
                        input_sha256={str(root/'collection/result.json'):ref(root/'collection/result.json')['sha256']},
                        original_native_quality_passed=False,eligible_for_model_training_registration=False,
                        model_training_authorized=False,fresh_performance_authorized=False)
            (root/'review/verified.json').write_text(json.dumps(result))
        """)
    )
    common = [sys.executable, str(script), str(root), str(args.prepared)]
    return tuple(common + [phase, fail, str(mismatch)] for phase in ("worker", "independent"))


def test_two_actual_cpu_children_wait_reap_before_checking_and_leave_quality_false(tmp_path, monkeypatch):
    owner, args, prepared, root = admitted(tmp_path, monkeypatch)
    commands = miniature_commands(tmp_path, args, root)
    monkeypatch.setattr(owner, "planned_commands", lambda *a: commands)
    owner.execute(args)
    result = json.loads((root / "orchestration/run/result.json").read_text())
    assert result["complete"] is result["passed"] is True
    assert len(result["owned_commands"]) == len(result["command_attempts"]) == 2
    assert (root / "order.txt").read_text().splitlines() == ["worker", "independent"]
    assert all(result[k] is False for k in owner.FALSE_FLAGS)
    for lifecycle in result["owned_commands"]:
        owner.lifecycle_guard(lifecycle, {})


@pytest.mark.parametrize("fail,attempts", [("worker", 1), ("independent", 2)])
def test_actual_failure_is_reaped_and_never_retried(tmp_path, monkeypatch, fail, attempts):
    owner, args, prepared, root = admitted(tmp_path, monkeypatch)
    commands = miniature_commands(tmp_path, args, root, fail=fail)
    monkeypatch.setattr(owner, "planned_commands", lambda *a: commands)
    with pytest.raises(ChildProcessError):
        owner.execute(args)
    result = json.loads((root / "orchestration/run/result.json").read_text())
    assert result["complete"] is result["passed"] is False
    assert len(result["command_attempts"]) == attempts
    failed = result["command_attempts"][-1]["observed_lifecycle"]
    end = json.loads(Path(failed["exit"]["path"]).read_text())
    release = json.loads(Path(failed["release"]["path"]).read_text())
    assert end["actual_wait"] and end["returncode"] == 7
    assert release["reaped"] and not release["signal_sent"]
    assert all(result[k] is False for k in owner.FALSE_FLAGS)


def test_static_mismatch_is_preserved_and_does_not_promote_quality(tmp_path, monkeypatch):
    owner, args, prepared, root = admitted(tmp_path, monkeypatch)
    commands = miniature_commands(tmp_path, args, root, mismatch=True)
    monkeypatch.setattr(owner, "planned_commands", lambda *a: commands)
    with pytest.raises(ValueError, match="independent complete evidence"):
        owner.execute(args)
    review = json.loads((root / "review/verified.json").read_text())
    result = json.loads((root / "orchestration/run/result.json").read_text())
    assert review["passed"] is result["passed"] is False
    assert len(result["owned_commands"]) == 2
    assert all(result[k] is False for k in owner.FALSE_FLAGS)


def test_read_prepared_rejects_modified_snapshot_even_with_matching_metadata(tmp_path, monkeypatch):
    owner, args, prepared, root = admitted(tmp_path, monkeypatch)
    # Exercise the actual reader rather than admitted()'s fixture reader.
    original = load_owner(monkeypatch)
    monkeypatch.setattr(original, "REPO", tmp_path)
    snapshots = {}
    for relative in original.NEW_SOURCES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    metadata = dict(
        output_root=str(root),
        native_prepared=dict(snapshot_sha256={}),
        source_sha256={str(tmp_path / p): original.sha256(tmp_path / p) for p in original.NEW_SOURCES},
    )
    snapshots = original.snapshot_mapping(metadata)
    for path in snapshots:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((tmp_path / original.NEW_SOURCES[int(target.name[:2])]).read_bytes())
    metadata["snapshot_sha256"] = snapshots
    args.prepared.write_text(json.dumps(metadata))
    monkeypatch.setattr(
        original, "build_metadata", lambda: {k: v for k, v in metadata.items() if k != "snapshot_sha256"}
    )
    digest = original.sha256(args.prepared)
    assert original.read_prepared(args.prepared, digest) == metadata
    Path(next(iter(snapshots))).write_text("tampered")
    with pytest.raises(ValueError, match="frozen snapshot changed"):
        original.read_prepared(args.prepared, digest)
