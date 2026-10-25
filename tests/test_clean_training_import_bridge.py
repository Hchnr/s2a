"""Source-honest metadata aliases and real interpreter safe-path resolution."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from s2a import clean_training_import_bridge as bridge
from s2a.durable_stage_supervisor import REPO, reference


@pytest.fixture
def scene(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "run_wam_clean_training", raising=False)
    assert tmp_path.is_relative_to(REPO / "outputs")
    directory = tmp_path / "bridge"
    bundle = tmp_path / "bundle"
    directory.mkdir()
    bundle.mkdir()
    delegate = bundle / "original.py"
    delegate.write_text('def original():\n    return "unchanged"\n')
    source = directory / "run_wam_clean_training.py"
    source.write_text(f"""from s2a.clean_training_import_bridge import load_frozen_delegate
old=load_frozen_delegate({reference(delegate)!r},name="_s2a048_unit_delegate")
original=old.original
value="new metadata only"
""")
    refs = {"run_wam_clean_training": reference(source)}
    registration = dict(path=str(tmp_path / "future_registration.json"), sha256="a" * 64)
    return directory, bundle, refs, registration, delegate


def enter(scene):
    directory, bundle, refs, registration, _ = scene
    return bridge.metadata_import_scope(
        bridge_directory=directory, bundle_directory=bundle, module_refs=refs, registration_ref=registration
    )


def test_scope_preserves_true_source_and_restores_names_environment(scene):
    before_path = list(sys.path)
    before_env = {k: os.environ.get(k) for k in (bridge.ENV_PATH, bridge.ENV_SHA, "PYTHONPATH", "PYTHONSAFEPATH")}
    with enter(scene) as (module, receipt):
        assert module.original() == "unchanged"
        assert module.original.__code__.co_filename == str(scene[-1])
        assert module.old.__file__ == str(scene[-1])
        assert sys.modules["run_wam_clean_training"] is module
        assert sys.path[:2] == list(map(str, scene[:2]))
    assert receipt["restored"] is receipt["source_guard_completed"] is True
    assert receipt["imported_sources"] == scene[2]
    assert sys.path == before_path and all(os.environ.get(k) == v for k, v in before_env.items())
    assert "run_wam_clean_training" not in sys.modules and "_s2a048_unit_delegate" not in sys.modules


def test_body_exception_restores_and_unlocks(scene):
    with pytest.raises(RuntimeError, match="delegate error"):
        with enter(scene) as (_, receipt):
            raise RuntimeError("delegate error")
    assert receipt["restored"] is True and "run_wam_clean_training" not in sys.modules
    with enter(scene):
        pass


def test_cached_alias_is_rejected_without_changes(scene, monkeypatch):
    old = object()
    monkeypatch.setitem(sys.modules, "run_wam_clean_training", old)
    before = list(sys.path)
    with pytest.raises(ValueError, match="already cached"):
        with enter(scene):
            pass
    assert sys.modules["run_wam_clean_training"] is old and sys.path == before


def test_nested_scope_refuses_rebinding(scene):
    with enter(scene) as (module, _):
        with pytest.raises(ValueError, match="already active"):
            with enter(scene):
                pass
        assert sys.modules["run_wam_clean_training"] is module


def test_replacing_alias_is_rejected_and_restored(scene):
    with pytest.raises(ValueError, match="replaced"):
        with enter(scene) as (_, receipt):
            sys.modules["run_wam_clean_training"] = object()
    assert receipt["restored"] is True and receipt["source_guard_completed"] is False
    assert "run_wam_clean_training" not in sys.modules


def test_child_environment_is_explicit_deduplicated_and_cannot_conflict(scene):
    directory, bundle, _, registration, _ = scene
    original = {
        "CUDA_VISIBLE_DEVICES": "allowed_uuid",
        "LD_LIBRARY_PATH": "",
        "PYTHONPATH": str(bundle) + ":" + str(REPO / "src"),
    }
    updated = bridge.child_environment(
        original, bridge_directory=directory, bundle_directory=bundle, registration_ref=registration
    )
    assert updated["CUDA_VISIBLE_DEVICES"] == "allowed_uuid" and updated["LD_LIBRARY_PATH"] == ""
    assert updated["PYTHONPATH"].split(":") == [str(directory), str(bundle), str(REPO / "src")]
    assert updated["PYTHONSAFEPATH"] == "1" and bridge.ENV_PATH not in original
    with pytest.raises(ValueError, match="conflicting"):
        bridge.child_environment(
            dict(original, **{bridge.ENV_SHA: "b" * 64}),
            bridge_directory=directory,
            bundle_directory=bundle,
            registration_ref=registration,
        )


@pytest.mark.parametrize("safe", [True, False])
@pytest.mark.parametrize("runtime", ["main", "model"])
def test_actual_original_sampler_help_resolves_bridge_only_with_safe_path(tmp_path, safe, runtime):
    bundle = (
        REPO / "outputs/wam_stability_20261021/clean_residual_training_20261023/recovery_20261023/source/entrypoints"
    )
    sampler = bundle / "sample_wam_clean_training_state.py"
    owner = bundle / "run_wam_clean_training.py"
    before = {str(p): reference(p) for p in (sampler, owner)}
    directory = tmp_path / "bridge"
    directory.mkdir()
    witness = tmp_path / "import_witness.json"
    (directory / "run_wam_clean_training.py").write_text(f"""import atexit,inspect,json,os,sys
from pathlib import Path
from s2a.clean_training_import_bridge import load_frozen_delegate
old=load_frozen_delegate({reference(owner)!r},name="_s2a048_real_original_owner")
MASTER_PORT=old.MASTER_PORT
_flags=old._flags
environment=old.environment
observe_release=old.observe_release
def prior(*args,**kwargs):
    raise RuntimeError("this import probe cannot authorize a W2 lookup")
assert "torch" not in sys.modules
assert sys.argv[1:]==["--help"]
def record():
    main=sys.modules["__main__"]
    assert main.prior is prior
    Path({str(witness)!r}).write_text(json.dumps(dict(bridge_file=__file__,delegate_file=old.__file__,actual_safe_path=sys.flags.safe_path,original_command=sys.argv,pid=os.getpid(),sampler_file=main.__file__,sampler_run_source=inspect.getsourcefile(main.run),prior_source=inspect.getsourcefile(prior),scope="import resolution only; callback denies all W2 lookup; no inference")))
atexit.register(record)
""")
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        LD_LIBRARY_PATH="",
        OMP_NUM_THREADS="8",
        OPENBLAS_NUM_THREADS="8",
        MKL_NUM_THREADS="8",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=":".join(map(str, (directory, bundle, REPO / "src", REPO / "scripts"))),
    )
    if safe:
        environment["PYTHONSAFEPATH"] = "1"
    else:
        environment.pop("PYTHONSAFEPATH", None)
    prepared = json.loads((bundle.parents[1] / "preparation/prepared.json").read_text())
    python = str(REPO / ".venv/bin/python") if runtime == "main" else prepared["model_config"]["model_python"]
    result = subprocess.run(
        [python, str(sampler), "--help"],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert witness.exists() is safe
    if safe:
        value = json.loads(witness.read_text())
        assert value["actual_safe_path"] is True
        assert value["delegate_file"] == str(owner) and value["original_command"] == [str(sampler), "--help"]
        assert value["sampler_file"] == value["sampler_run_source"] == str(sampler)
        assert value["prior_source"] == str(directory / "run_wam_clean_training.py")
    assert {str(p): reference(p) for p in (sampler, owner)} == before


def test_caller_mutating_expected_sha_cannot_hide_source_change(scene):
    with pytest.raises(ValueError, match="origin changed|source changed"):
        with enter(scene) as (_, receipt):
            source = Path(scene[2]["run_wam_clean_training"]["path"])
            source.write_text(source.read_text() + "\n# actual file changed during scope\n")
            scene[2]["run_wam_clean_training"] = reference(source)
    assert receipt["restored"] is True and receipt["source_guard_completed"] is False
    assert "run_wam_clean_training" not in sys.modules
