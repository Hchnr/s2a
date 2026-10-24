"""Exact041 source edits and exclusive output; fixtures never execute a model."""

import ast
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from s2a import clean_training_continuation_bundle as BUNDLE

REPO = Path(__file__).resolve().parents[1]
NAMES = (
    "run_wam_clean_training.py",
    "collect_wam_clean_training.py",
    "verify_wam_clean_training_capture.py",
    "run_wam_clean_training_state.py",
    "sample_wam_clean_training_state.py",
    "verify_wam_clean_training_state.py",
    "run_wam_clean_training_development.py",
    "serve_wam_clean_training.py",
    "verify_wam_clean_training_development.py",
    "execute_wam_clean_training_case.py",
)
OLD = "outputs/wam_stability_20261021/clean_residual_training_20261023"
NEW = OLD + "/recovery_20261023"
DESTINATION = NEW + "/source/entrypoints"
ORIGINALS = tuple("scripts/" + name for name in NAMES) + ("src/s2a/clean_training_runtime.py",)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def gold(source, relative, repo):
    """An independent literal gold; never use the producer's edit plan."""
    text = source.decode("utf-8")
    pairs = [("scripts/" + name, DESTINATION + "/" + name) for name in NAMES]
    pairs += [("src/s2a/clean_training_runtime.py", DESTINATION + "/continuation_runtime.py"), (OLD, NEW)]
    # The frozen040 input uses ordinary quoted literals; comments, mathematical
    # expressions, imports of unchanged fit code, and other strings stay intact.
    for old, new in pairs:
        for quote in ("'", '"'):
            text = text.replace(quote + old + quote, quote + new + quote)
    text = text.replace("from s2a.clean_training_runtime import", "from continuation_runtime import")
    if relative in ("scripts/verify_wam_clean_training_capture.py", "scripts/verify_wam_clean_training_development.py"):
        text = text.replace("REPO = Path(__file__).resolve().parents[1]", "REPO = Path(" + json.dumps(str(repo)) + ")")
    return text


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "synthetic_repository"
    repo.mkdir()
    registration = dict(source_sha256={})
    original_bytes = {}
    for relative in ORIGINALS:
        data = (REPO / relative).read_bytes()
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        registration["source_sha256"][str(path)] = sha(data)
        original_bytes[str(path)] = data
    return repo, registration, original_bytes


def test_readonly_plan_matches_independent_gold_for_every_byte_and_ast(repository):
    repo, registration, before = repository
    planned = BUNDLE.expected_bundle_sources(registration, repo=repo)
    assert set(planned) == set(before)
    assert not (repo / NEW).exists()
    for original, entry in planned.items():
        path = Path(original)
        relative = str(path.relative_to(repo))
        expected = gold(before[original], relative, repo)
        assert entry["source_sha256"] == sha(before[original])
        assert entry["transformed_text"] == expected
        assert Path(entry["target_abs"]).parent == repo / DESTINATION
        assert Path(entry["target_abs"]).name == (
            "continuation_runtime.py" if path.name == "clean_training_runtime.py" else path.name
        )
        assert ast.dump(ast.parse(entry["transformed_text"])) == ast.dump(ast.parse(expected))
        compile(entry["transformed_text"], entry["target_abs"], "exec")
        replay = before[original].decode("utf-8")
        for edit in reversed(entry["edits"]):
            begin = edit["offset"]
            assert edit["count"] == 1 and replay[begin : begin + len(edit["old"])] == edit["old"]
            replay = replay[:begin] + edit["new"] + replay[begin + len(edit["old"]) :]
        assert replay == expected
    assert {p: Path(p).read_bytes() for p in before} == before
    owner = planned[str(repo / "scripts/run_wam_clean_training.py")]["transformed_text"]
    assert '"scripts/fit_wam_clean_training.py"' in owner
    assert '"scripts/verify_wam_clean_training_fit.py"' in owner
    assert (DESTINATION + "/source/entrypoints") not in "".join(e["transformed_text"] for e in planned.values())


def test_exclusive_publish_records_every_source_and_preserves_originals(repository):
    repo, registration, before = repository
    preparation = repo / NEW / "preparation/prepared.json"
    preparation.parent.mkdir(parents=True)
    preparation.write_bytes(b'{"synthetic_preparation": true}\n')
    manifest = BUNDLE.publish_bundle(registration, repo / NEW, repo=repo)
    assert manifest["schema_version"] == "s2a.clean-training-continuation-bundle.v1"
    assert manifest["complete"] and manifest["passed"] and manifest["original_sources_before_after_exact"]
    assert manifest["entrypoints_executed"] == manifest["future_archive_decodes"] == 0
    assert manifest["old_admitted_ref"] is None
    assert len(manifest["files"]) == len(manifest["source_sha256"]) == 11
    for record in manifest["files"]:
        actual, original = record["actual_ref"], record["original_ref"]
        assert actual["sha256"] == sha(Path(actual["path"]).read_bytes())
        assert manifest["source_sha256"][actual["path"]] == actual["sha256"]
        assert original["sha256"] == sha(before[original["path"]])
    assert not (repo / DESTINATION / "manifest.json").exists()
    assert preparation.read_bytes() == b'{"synthetic_preparation": true}\n'
    bundle_bytes = {p: p.read_bytes() for p in (repo / DESTINATION).iterdir()}
    with pytest.raises(ValueError, match="already exists"):
        BUNDLE.publish_bundle(registration, repo / NEW, repo=repo)
    assert {p: p.read_bytes() for p in bundle_bytes} == bundle_bytes
    assert {p: Path(p).read_bytes() for p in before} == before


@pytest.mark.parametrize("fault", ["missing", "drift", "binding", "syntax", "count", "escaped_quote"])
def test_missing_drifted_or_noncontract_sources_fail_before_outputs(repository, fault):
    repo, registration, _ = repository
    source = repo / "scripts/collect_wam_clean_training.py"
    if fault == "missing":
        source.unlink()
    elif fault == "binding":
        del registration["source_sha256"][str(source)]
    else:
        data = source.read_bytes()
        if fault == "drift":
            data += b"\n# unregistered source drift\n"
        elif fault == "syntax":
            data += b"\nthis is not valid python !!!\n"
        elif fault == "count":
            data += b'\nextra = "scripts/collect_wam_clean_training.py"\n'
        else:
            data = data.replace(b'"src/s2a/clean_training_runtime.py"', b'r"src/s2a/clean_training_runtime.py"')
        source.write_bytes(data)
        if fault != "drift":
            registration["source_sha256"][str(source)] = sha(data)
    with pytest.raises((ValueError, SyntaxError)):
        BUNDLE.publish_bundle(registration, repo / NEW, repo=repo)
    assert not (repo / NEW).exists()


@pytest.mark.parametrize("fault", ["root", "traversal", "source_symlink", "destination_symlink", "collision"])
def test_path_escape_and_existing_bundle_are_rejected(repository, tmp_path, fault):
    repo, registration, _ = repository
    output = repo / NEW
    if fault == "root":
        output = tmp_path / "outside"
    elif fault == "traversal":
        output = repo / NEW / ".." / "recovery_20261023"
    elif fault == "source_symlink":
        source = repo / "scripts/collect_wam_clean_training.py"
        outside = tmp_path / "outside.py"
        source.rename(outside)
        source.symlink_to(outside)
    elif fault == "destination_symlink":
        output.parent.mkdir(parents=True)
        output.symlink_to(tmp_path, target_is_directory=True)
    else:
        directory = repo / DESTINATION
        directory.mkdir(parents=True)
        (directory / "preserved.py").write_bytes(b"original collision bytes\n")
    with pytest.raises(ValueError):
        BUNDLE.publish_bundle(registration, output, repo=repo)
    if fault == "collision":
        assert (repo / DESTINATION / "preserved.py").read_bytes() == b"original collision bytes\n"


def test_old_admitted_ref_is_bound_to_exact_original_location_and_payload(repository):
    repo, registration, _ = repository
    path = repo / OLD / "execution/admitted.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(registration))
    reference = dict(path=str(path), sha256=sha(path.read_bytes()))
    bad = copy.deepcopy(registration)
    bad["extra_unregistered"] = True
    with pytest.raises(ValueError, match="payload differs"):
        BUNDLE.publish_bundle(bad, repo / NEW, repo=repo, old_admitted_ref=reference)
    assert not (repo / NEW).exists()
    manifest = BUNDLE.publish_bundle(registration, repo / NEW, repo=repo, old_admitted_ref=reference)
    assert manifest["old_admitted_ref"] == reference


def test_drift_during_planning_is_rejected_without_writing(repository, monkeypatch):
    repo, registration, _ = repository
    transform = BUNDLE._transform
    source = repo / "scripts/run_wam_clean_training.py"

    def mutate(*args, **kwargs):
        result = transform(*args, **kwargs)
        source.write_bytes(source.read_bytes() + b"\n# concurrent drift\n")
        return result

    monkeypatch.setattr(BUNDLE, "_transform", mutate)
    with pytest.raises(ValueError, match="changed during planning"):
        BUNDLE.publish_bundle(registration, repo / NEW, repo=repo)
    assert not (repo / NEW).exists()


def test_actual040_metadata_readonly_build_and_all_old_source_shas_remain_exact(monkeypatch):
    path = REPO / OLD / "execution/admitted.json"
    if not path.is_file():
        pytest.skip("actual040 evidence is unavailable in this checkout")
    original = path.read_bytes()
    registration = json.loads(original)
    # No numpy/torch is needed or imported by the builder. Forbid every output
    # write while processing the real registration, including future artifacts.
    original_open = Path.open

    def readonly(self, mode="r", *args, **kwargs):
        assert not any(c in mode for c in "wax+"), "readonly metadata build attempted an output write"
        assert self.suffix not in (".npz", ".npy", ".parquet"), "builder read a non-source data archive"
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", readonly)
    planned = BUNDLE.expected_bundle_sources(registration)
    assert len(planned) == 11 and path.read_bytes() == original
    for source, digest in registration["source_sha256"].items():
        assert sha(Path(source).read_bytes()) == digest


def test_all_ten_published_entrypoint_help_commands_are_cpu_and_do_not_decode_data(repository, tmp_path):
    repo, registration, _ = repository
    BUNDLE.publish_bundle(registration, repo / NEW, repo=repo)
    guard = tmp_path / "cli_import_guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(
        "import builtins\n"
        "import numpy as np\n"
        "def deny_arrays(*args, **kwargs):\n"
        "    raise AssertionError('CLI help decoded arrays')\n"
        "np.load = deny_arrays\n"
        "original = builtins.__import__\n"
        "def guarded(name, *args, **kwargs):\n"
        "    if name.split('.')[0] in {'cosmos_framework', 'mujoco', 'robosuite'}:\n"
        "        raise AssertionError('CLI help imported model or physics')\n"
        "    return original(name, *args, **kwargs)\n"
        "builtins.__import__ = guarded\n"
    )
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        LD_LIBRARY_PATH="",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="8",
        OPENBLAS_NUM_THREADS="8",
        MKL_NUM_THREADS="8",
        PYTHONPATH=os.pathsep.join(str(p) for p in (guard, REPO / "src", REPO / "scripts")),
    )
    for name in NAMES:
        result = subprocess.run(
            [sys.executable, str(repo / DESTINATION / name), "--help"],
            cwd=REPO,
            env=environment,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, (name, result.stdout, result.stderr)
        assert "usage:" in result.stdout
        assert not (repo / NEW / "capture").exists()
