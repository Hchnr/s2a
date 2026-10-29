"""Real local Git fixtures distinguish original bytes from migrated live code."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from s2a.infrastructure.repository_paths import FORBIDDEN_HISTORICAL_ROOT
from s2a.infrastructure.source_identity import SourceIdentityLocator


def git(root, *args):
    process = subprocess.Popen(
        ["git", "-C", str(root), *args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_TERMINAL_PROMPT="0"),
    )
    stdout, stderr = process.communicate(timeout=30)
    assert process.wait() == 0, stderr.decode()
    return stdout


def reference(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return reference(path)


@pytest.fixture
def baseline(tmp_path):
    root = tmp_path / "repository"
    (root / "src/s2a").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname = "s2a"\n')
    (root / "src/s2a/__init__.py").write_text('"""Fixture package."""\n')
    paths = ["src/s2a/subject.py", "src/s2a/other.py"]
    for index, name in enumerate(paths):
        (root / name).write_text(f"raise RuntimeError('source must not be imported {index}')\n")
    git(root, "init", "--quiet")
    git(root, "add", "pyproject.toml", "src")
    git(
        root,
        "-c",
        "user.name=S2A fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "base",
    )
    commit = git(root, "rev-parse", "HEAD").decode().strip()
    entries = []
    for name in paths:
        line = git(root, "ls-tree", "HEAD", "--", name).decode().strip()
        mode, _, oid = line.split("\t")[0].split()
        entries.append(
            dict(
                old_path=name,
                sha256=reference(root / name)["sha256"],
                size_bytes=(root / name).stat().st_size,
                git_blob_oid=oid,
                git_mode=mode,
                baseline_commit=commit,
            )
        )
    catalog = dict(
        schema_version=1,
        baseline_commit=commit,
        file_count=len(entries),
        total_size_bytes=sum(entry["size_bytes"] for entry in entries),
        entries=entries,
    )
    catalog_path = root / "docs/migrations/baseline.json"
    catalog_ref = write_json(catalog_path, catalog)
    return dict(root=root, commit=commit, catalog=catalog, catalog_ref=catalog_ref, paths=paths)


def locator(baseline, *, migrations=(), catalog_ref=None, cache=None):
    return SourceIdentityLocator(
        root=baseline["root"],
        baseline_commit=baseline["commit"],
        catalog_ref=baseline["catalog_ref"] if catalog_ref is None else catalog_ref,
        migration_refs=migrations,
        cache_directory=baseline["root"] / "outputs/cache" if cache is None else cache,
    )


def original_ref(baseline, *, path=None):
    entry = baseline["catalog"]["entries"][0]
    return dict(path=entry["old_path"] if path is None else str(path), sha256=entry["sha256"])


def migrate(baseline, *, modified=False):
    root = baseline["root"]
    old = root / baseline["paths"][0]
    new = root / "src/s2a/research/subject.py"
    new.parent.mkdir(parents=True)
    old.rename(new)
    if modified:
        new.write_text("# current code uses new imports\n")
    move = dict(
        old_path=baseline["paths"][0],
        original_sha256=original_ref(baseline)["sha256"],
        new_path=new.relative_to(root).as_posix(),
        current_sha256=reference(new)["sha256"],
    )
    manifest = dict(catalog_ref=baseline["catalog_ref"], moves=[move])
    manifest_ref = write_json(root / "docs/migrations/moves.json", manifest)
    return new, move, manifest, manifest_ref


def test_active_bytes_preserve_declaration_and_have_no_execution_qualification(baseline):
    reader = locator(baseline)
    declared = original_ref(baseline)
    result = reader.resolve(declared)
    assert result.declared_ref == declared
    assert result.storage_ref == reference(baseline["root"] / declared["path"])
    assert result.migration_ref is None
    assert result.resolution_kind == "active"
    assert result.identity_scope == "byte_identity_only"
    assert not (baseline["root"] / "outputs/cache").exists()
    assert all(row["actual_wait"] and row["reaped"] and row["returncode"] == 0 for row in reader.git_operations)
    reader.finish()


@pytest.mark.parametrize("historical", [False, True])
def test_absolute_current_and_old_declarations_keep_exact_identity_without_old_io(baseline, historical):
    root = FORBIDDEN_HISTORICAL_ROOT if historical else baseline["root"]
    declared = original_ref(baseline, path=root / baseline["paths"][0])
    result = locator(baseline).resolve(declared)
    assert result.declared_ref == declared
    assert Path(result.storage_ref["path"]).is_relative_to(baseline["root"])


def test_unknown_version_is_rejected_even_if_live_bytes_match(baseline):
    path = baseline["root"] / baseline["paths"][0]
    path.write_text("# unregistered historical version\n")
    reader = locator(baseline)
    before = len(reader.git_operations)
    with pytest.raises(ValueError, match="Unknown historical"):
        reader.resolve(reference(path))
    assert len(reader.git_operations) == before
    assert not (baseline["root"] / "outputs/cache").exists()


def test_registered_moved_unchanged_source_uses_exact_target(baseline):
    new, _, _, manifest_ref = migrate(baseline)
    result = locator(baseline, migrations=[manifest_ref]).resolve(original_ref(baseline))
    assert result.resolution_kind == "migrated"
    assert result.storage_ref == reference(new)
    assert result.migration_ref == manifest_ref
    assert not (baseline["root"] / "outputs/cache").exists()


def test_changed_live_bytes_fall_back_to_original_git_blob_and_reuse_cache(baseline):
    original = (baseline["root"] / baseline["paths"][0]).read_bytes()
    (baseline["root"] / baseline["paths"][0]).write_text("# changed live source\n")
    reader = locator(baseline)
    result = reader.resolve(original_ref(baseline))
    assert result.resolution_kind == "git_cache"
    assert Path(result.storage_ref["path"]).read_bytes() == original
    assert Path(result.storage_ref["path"]).is_relative_to(baseline["root"] / "outputs/cache")
    assert sum("cat-file" in row["command"] for row in reader.git_operations) == 1
    assert reader.resolve(original_ref(baseline)).storage_ref == result.storage_ref
    assert sum("cat-file" in row["command"] for row in reader.git_operations) == 1
    reader.finish()


def test_changed_migration_target_preserves_current_ref_and_returns_original_cache(baseline):
    original = (baseline["root"] / baseline["paths"][0]).read_bytes()
    new, move, _, manifest_ref = migrate(baseline, modified=True)
    reader = locator(baseline, migrations=[manifest_ref])
    result = reader.resolve(original_ref(baseline))
    assert result.resolution_kind == "git_cache"
    assert result.migration_ref == manifest_ref
    assert result.storage_ref["sha256"] == move["original_sha256"]
    assert result.storage_ref["sha256"] != move["current_sha256"]
    assert Path(result.storage_ref["path"]).read_bytes() == original
    assert reference(new)["sha256"] == move["current_sha256"]
    with pytest.raises(ValueError, match="Unknown historical"):
        reader.resolve(dict(path=move["old_path"], sha256=move["current_sha256"]))


def test_registered_current_target_tamper_does_not_fall_back_to_git_or_active_copy(baseline):
    old = baseline["root"] / baseline["paths"][0]
    original = old.read_bytes()
    new, _, _, manifest_ref = migrate(baseline, modified=True)
    old.write_bytes(original)
    new.write_text("# unregistered drift\n")
    reader = locator(baseline, migrations=[manifest_ref])
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        reader.resolve(original_ref(baseline))
    assert not any("cat-file" in row["command"] for row in reader.git_operations)
    assert not (baseline["root"] / "outputs/cache").exists()


def test_missing_current_target_is_not_hidden_by_original_git_blob(baseline):
    new, _, _, manifest_ref = migrate(baseline)
    new.unlink()
    with pytest.raises(FileNotFoundError):
        locator(baseline, migrations=[manifest_ref]).resolve(original_ref(baseline))


def test_locator_does_not_guess_a_same_basename_or_hash_elsewhere(baseline):
    root = baseline["root"]
    old = root / baseline["paths"][0]
    original = old.read_bytes()
    unregistered = root / "src/s2a/unregistered/subject.py"
    unregistered.parent.mkdir()
    old.rename(unregistered)
    result = locator(baseline).resolve(original_ref(baseline))
    assert result.resolution_kind == "git_cache"
    assert result.storage_ref["path"] != str(unregistered)
    assert Path(result.storage_ref["path"]).read_bytes() == original


@pytest.mark.parametrize("bad", ["duplicate_old", "duplicate_new", "wrong_original", "wrong_catalog", "traversal"])
def test_ambiguous_or_unbound_migration_maps_fail_closed(baseline, bad):
    _, move, manifest, manifest_ref = migrate(baseline)
    if bad == "duplicate_old":
        manifest["moves"].append(dict(move))
    elif bad == "duplicate_new":
        other = baseline["catalog"]["entries"][1]
        manifest["moves"].append(dict(move, old_path=other["old_path"], original_sha256=other["sha256"]))
    elif bad == "wrong_original":
        move["original_sha256"] = "0" * 64
        manifest["moves"] = [move]
    elif bad == "wrong_catalog":
        manifest["catalog_ref"] = dict(baseline["catalog_ref"], sha256="0" * 64)
    else:
        move["new_path"] = "src/../subject.py"
        manifest["moves"] = [move]
    manifest_ref = write_json(Path(manifest_ref["path"]), manifest)
    with pytest.raises(ValueError):
        locator(baseline, migrations=[manifest_ref])


@pytest.mark.parametrize("bad", ["duplicate", "wrong_blob", "wrong_mode", "wrong_size", "wrong_total"])
def test_catalog_identity_and_git_origin_are_independently_checked(baseline, bad):
    catalog = baseline["catalog"]
    if bad == "duplicate":
        catalog["entries"].append(dict(catalog["entries"][0]))
    elif bad == "wrong_blob":
        catalog["entries"][0]["git_blob_oid"] = catalog["entries"][1]["git_blob_oid"]
    elif bad == "wrong_mode":
        catalog["entries"][0]["git_mode"] = "120000"
    elif bad == "wrong_size":
        catalog["entries"][0]["size_bytes"] += 1
        catalog["total_size_bytes"] += 1
    else:
        catalog["total_size_bytes"] += 1
    catalog_ref = write_json(Path(baseline["catalog_ref"]["path"]), catalog)
    with pytest.raises(ValueError):
        locator(baseline, catalog_ref=catalog_ref).resolve(original_ref(baseline))


def test_corrupt_existing_cache_is_rejected_without_repair(baseline):
    (baseline["root"] / baseline["paths"][0]).unlink()
    first = locator(baseline).resolve(original_ref(baseline))
    cache = Path(first.storage_ref["path"])
    cache.write_bytes(b"corruption")
    reader = locator(baseline)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        reader.resolve(original_ref(baseline))
    assert cache.read_bytes() == b"corruption"
    assert not any("cat-file" in row["command"] for row in reader.git_operations)


def test_finish_rechecks_manifest_current_target_and_original_cache(baseline):
    new, _, _, manifest_ref = migrate(baseline, modified=True)
    reader = locator(baseline, migrations=[manifest_ref])
    reader.resolve(original_ref(baseline))
    new.write_text("# changed after resolve\n")
    with pytest.raises(ValueError, match="Consumed source or manifest changed"):
        reader.finish()


def test_replaced_manifest_is_rejected_before_resolving_another_source(baseline):
    _, _, manifest, manifest_ref = migrate(baseline)
    reader = locator(baseline, migrations=[manifest_ref])
    manifest["moves"] = []
    write_json(Path(manifest_ref["path"]), manifest)
    with pytest.raises(ValueError, match="Consumed source or manifest changed"):
        reader.resolve(original_ref(baseline))


def test_storage_and_cache_symlinks_or_escapes_are_rejected(baseline):
    root = baseline["root"]
    with pytest.raises(ValueError, match="inside repository outputs"):
        locator(baseline, cache=root / "src/cache")
    source = root / baseline["paths"][0]
    source.unlink()
    source.symlink_to(FORBIDDEN_HISTORICAL_ROOT / "never-opened.py")
    with pytest.raises(ValueError, match="Symlink"):
        locator(baseline).resolve(original_ref(baseline))


def test_failed_local_git_command_is_actually_reaped(baseline):
    reader = locator(baseline)
    with pytest.raises(ValueError, match="Local Git identity command failed"):
        reader._git(["cat-file", "blob", "0" * 40])
    receipt = reader.git_operations[-1]
    assert receipt["actual_wait"] is receipt["reaped"] is True
    assert receipt["returncode"] != 0
    assert receipt["signal_sent"] is False
