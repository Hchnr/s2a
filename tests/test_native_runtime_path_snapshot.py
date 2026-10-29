"""Synthetic filesystem contracts; no runtime imports, datasets, or native loading."""

from __future__ import annotations

import builtins
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from s2a import native_runtime_path_snapshot as snapshot


def file_ref(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return file_ref(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@pytest.fixture
def source(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    safe = tmp_path / "safe"
    old = Path("/historical/libero_sim")
    monkeypatch.setattr(snapshot, "REPO", repo)
    monkeypatch.setattr(snapshot, "SAFE_SOURCE_ROOT", safe)
    monkeypatch.setattr(snapshot, "OLD_SOURCE_ROOT", old)
    package_files = {
        "__init__.py": b"raise AssertionError('runtime must never be imported')\n",
        "libero/__init__.py": b"# fixture package\n",
        "libero/benchmark/__init__.py": b"TASK = 'fixture'\n",
        "libero/bddl_files/libero_goal/task.bddl": b"(define (problem fixture))\n",
        "libero/init_files/libero_goal/task.pruned_init": b"\x00original init states\xff",
        "libero/assets/models/fixture.xml": b"<mujoco model='fixture'/>\n",
        "libero/assets/meshes/fixture.stl": b"\x00synthetic mesh\xff",
    }
    package = safe / "LIBERO/libero"
    for relative, data in package_files.items():
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (package / "libero/assets/empty").mkdir()
    license_path = safe / "LIBERO/LICENSE"
    license_path.write_bytes(b"Synthetic license fixture\n")
    caches = {
        "__pycache__/module.cpython-311.pyc": b"ignored bytecode",
        ".pytest_cache/v/cache/nodeids": b"ignored pytest cache",
        "libero/.mypy_cache/nested/data.json": b"ignored mypy cache",
        "libero/.ruff_cache/nested/data": b"ignored ruff cache",
        "libero/orphan.pyc": b"ignored orphan pyc",
        "libero/orphan.pyo": b"ignored orphan pyo",
    }
    for relative, data in caches.items():
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    dataset = safe / "data"
    dataset.mkdir()
    (dataset / "never_read.bin").write_bytes(b"Dataset is outside the snapshot scope")
    benchmark = old / "LIBERO/libero/libero"
    config = {
        "benchmark_root": str(benchmark),
        "bddl_files": str(benchmark / "bddl_files"),
        "init_states": str(benchmark / "init_files"),
        "assets": str(benchmark / "assets"),
        "datasets": str(old / "data"),
        "fixture_option": {"seed": 42, "enabled": False, "labels": ["retain", "verbatim values"]},
    }
    config_path = safe / "config/config.yaml"
    config_ref = write_json(config_path, config)
    osmesa = safe / "systemlibs/usr/lib/x86_64-linux-gnu/libOSMesa.so.8.0.0"
    osmesa.parent.mkdir(parents=True)
    osmesa.write_bytes(b"\x7fELF synthetic bytes; never dynamically loaded\x00")
    historical = {
        str(old / "LIBERO/libero" / relative): file_ref(package / relative)["sha256"]
        for relative in ("__init__.py", "libero/benchmark/__init__.py")
    }
    historical["/historical/unrelated/robosuite/controller.py"] = "0" * 64
    manifest = {
        "runtime_config": {
            "libero_root": str(old / "LIBERO"),
            "libero_config": str(old / "config/config.yaml"),
            "osmesa_library": str(old / osmesa.relative_to(safe)),
            "seed": 42,
            "task_suite": "libero_goal",
            "task_id": 8,
            "control_fps": 20,
            "refresh_controller_after_restore": True,
            "nested_parameters": {"must_stay": [1, 2, 3]},
        },
        "simulator_source_sha256": historical,
        "record": {"initial_state": [1.0, 2.0], "source": "/historical/provenance-only.json"},
    }
    manifest_path = repo / "outputs/original/manifest.json"
    manifest_ref = write_json(manifest_path, manifest)
    return SimpleNamespace(
        repo=repo,
        safe=safe,
        old=old,
        package=package,
        package_files=package_files,
        caches=caches,
        dataset=dataset,
        config=config,
        config_path=config_path,
        osmesa=osmesa,
        manifest=manifest,
        manifest_path=manifest_path,
        refs=[manifest_ref, config_ref, file_ref(osmesa)],
        destination=repo / "outputs/runtime-snapshot",
    )


def prepare(source, **kwargs):
    return snapshot.prepare_runtime_snapshot(*source.refs, source.destination, **kwargs)


def updated_snapshot_ref(ref, manifest):
    return write_json(Path(ref["path"]), manifest)


def reject_without_destination(source, **kwargs):
    with pytest.raises(ValueError):
        prepare(source, **kwargs)
    assert not source.destination.exists()


def test_exact_path_only_migration_and_byte_preserving_snapshots(source):
    original_manifest = source.manifest_path.read_bytes()
    original_config = source.config_path.read_bytes()
    ref = prepare(source)
    result = snapshot.verify_runtime_snapshot(ref)
    assert result["file_snapshot_verified"] is True
    for flag in ("runtime_loaded", "bootstrap_verified", "control_qualified"):
        assert result[flag] is False
    assert ref == file_ref(Path(ref["path"]))
    assert Path(result["original_manifest_snapshot_ref"]["path"]).read_bytes() == original_manifest
    assert Path(result["original_config_snapshot_ref"]["path"]).read_bytes() == original_config
    runtime = read_json(result["runtime_manifest_ref"]["path"])
    expected = json.loads(original_manifest)
    expected["runtime_config"].update(
        libero_root=str(source.destination / "LIBERO"),
        libero_config=str(source.destination / "config/config.yaml"),
        osmesa_library=str(source.destination / "systemlibs" / source.osmesa.name),
    )
    assert runtime == expected
    migrated_config = read_json(result["runtime_config_ref"]["path"])
    expected_config = dict(source.config)
    for key in ("benchmark_root", "bddl_files", "init_states", "assets", "datasets"):
        expected_config[key] = str(source.destination / Path(source.config[key]).relative_to(source.old))
    assert migrated_config == expected_config
    mappings = result["path_mappings"]
    assert set(mappings["runtime_config"]) == {"libero_root", "libero_config", "osmesa_library"}
    assert set(mappings["libero_config"]) == {"benchmark_root", "bddl_files", "init_states", "assets", "datasets"}
    for key, mapping in mappings["runtime_config"].items():
        assert mapping == {
            "original_path": source.manifest["runtime_config"][key],
            "snapshot_path": runtime["runtime_config"][key],
        }
    for key, mapping in mappings["libero_config"].items():
        assert mapping == {"original_path": source.config[key], "snapshot_path": migrated_config[key]}
    assert source.manifest_path.read_bytes() == original_manifest
    assert source.config_path.read_bytes() == original_config


def test_full_package_init_files_empty_directories_and_historical_scope(source):
    result = snapshot.verify_runtime_snapshot(prepare(source))
    inventory = {entry["relative_path"]: entry for entry in result["file_inventory"]}
    expected_paths = {"LIBERO/libero/" + relative for relative in source.package_files}
    expected_paths |= {"LIBERO/LICENSE", "systemlibs/" + source.osmesa.name}
    assert set(inventory) == expected_paths
    bound = set()
    for relative, data in source.package_files.items():
        entry = inventory["LIBERO/libero/" + relative]
        assert (source.destination / "LIBERO/libero" / relative).read_bytes() == data
        assert entry["source_ref"] == file_ref(source.package / relative)
        assert entry["snapshot_ref"] == file_ref(source.destination / "LIBERO/libero" / relative)
        assert entry["size_bytes"] == len(data)
        if entry["historical_source_ref"] is not None:
            historical_path = str(source.old / "LIBERO/libero" / relative)
            assert entry["historical_source_ref"] == {
                "path": historical_path,
                "sha256": source.manifest["simulator_source_sha256"][historical_path],
            }
            bound.add(relative)
    assert bound == {"__init__.py", "libero/benchmark/__init__.py"}
    assert (source.destination / "LIBERO/libero/libero/assets/empty").is_dir()
    assert (source.destination / "data").is_dir()
    assert list((source.destination / "data").iterdir()) == []
    assert result["dataset"] == {"path": str(source.destination / "data"), "populated": False, "source_read": False}
    assert (source.destination / "systemlibs" / source.osmesa.name).read_bytes() == source.osmesa.read_bytes()


def test_inspection_has_no_destination_side_effect_and_reports_exclusions(source):
    result = snapshot.inspect_runtime_sources(*source.refs, max_files=10000, max_bytes=2**30)
    assert not source.destination.exists()
    assert sorted(path.name for path in (source.repo / "outputs").iterdir()) == ["original"]
    assert result["copied_file_count"] == len(source.package_files) + 2
    assert result["libero_file_count"] == len(source.package_files)
    assert result["historical_libero_file_count"] == 2
    assert result["license_included"] is True
    for flag in ("file_snapshot_verified", "runtime_loaded", "bootstrap_verified", "control_qualified"):
        assert result[flag] is False
    assert (
        result["copied_bytes"]
        == sum(map(len, source.package_files.values()))
        + (source.safe / "LIBERO/LICENSE").stat().st_size
        + source.osmesa.stat().st_size
    )
    excluded = json.dumps(result["excluded_cache_entries"], sort_keys=True)
    for relative in source.caches:
        assert relative in excluded
        for parent in Path(relative).parents:
            if parent != Path("."):
                assert str(parent) in excluded
    for entry in result["source_inventory"]:
        assert "snapshot_ref" not in entry
    prepared = snapshot.verify_runtime_snapshot(prepare(source))
    assert prepared["excluded_cache_entries"] == result["excluded_cache_entries"]
    for relative in source.caches:
        assert not (source.destination / "LIBERO/libero" / relative).exists()


def test_optional_license_absence_is_preserved(source):
    (source.safe / "LIBERO/LICENSE").unlink()
    inspected = snapshot.inspect_runtime_sources(*source.refs, max_files=10000, max_bytes=2**30)
    assert inspected["license_included"] is False
    result = snapshot.verify_runtime_snapshot(prepare(source))
    assert "LIBERO/LICENSE" not in {entry["relative_path"] for entry in result["file_inventory"]}


@pytest.mark.parametrize("limit", [{"max_files": 1}, {"max_bytes": 1}, {"max_files": 0}, {"max_bytes": -1}])
def test_initial_budget_rejection_never_creates_destination(source, limit):
    reject_without_destination(source, **limit)
    with pytest.raises(ValueError):
        snapshot.inspect_runtime_sources(*source.refs, **({"max_files": 10000, "max_bytes": 2**30} | limit))


def test_output_budget_accounts_for_backups_and_generated_metadata(source):
    payload_count = len(source.package_files) + 2
    source_bytes = (
        sum(map(len, source.package_files.values()))
        + (source.safe / "LIBERO/LICENSE").stat().st_size
        + source.osmesa.stat().st_size
        + source.manifest_path.stat().st_size
        + source.config_path.stat().st_size
    )
    reject_without_destination(source, max_files=payload_count + 5)
    reject_without_destination(source, max_bytes=source_bytes)
    result = snapshot.verify_runtime_snapshot(prepare(source, max_files=payload_count + 6))
    assert len(result["file_inventory"]) == payload_count
    assert len([path for path in source.destination.rglob("*") if path.is_file()]) == payload_count + 6


@pytest.mark.parametrize("which", [0, 1, 2])
def test_input_ref_hash_mismatch_is_rejected_before_copy(source, which):
    source.refs[which] = {**source.refs[which], "sha256": "f" * 64}
    reject_without_destination(source)


@pytest.mark.parametrize("which", [0, 1, 2])
def test_input_ref_disallowed_location_is_rejected(source, which):
    alternate = source.repo.parent / "elsewhere" / Path(source.refs[which]["path"]).name
    alternate.parent.mkdir()
    alternate.write_bytes(Path(source.refs[which]["path"]).read_bytes())
    source.refs[which] = file_ref(alternate)
    reject_without_destination(source)


@pytest.mark.parametrize("which", [1, 2])
def test_safe_input_must_match_exact_historical_mapping(source, which):
    alternate = source.safe / ("alternate-" + Path(source.refs[which]["path"]).name)
    alternate.write_bytes(Path(source.refs[which]["path"]).read_bytes())
    source.refs[which] = file_ref(alternate)
    reject_without_destination(source)


@pytest.mark.parametrize("key", ["benchmark_root", "bddl_files", "init_states", "assets", "datasets"])
def test_every_config_path_requires_the_original_mapping(source, key):
    config = dict(source.config)
    config[key] = str(source.old / "unexpected")
    source.refs[1] = write_json(source.config_path, config)
    reject_without_destination(source)


@pytest.mark.parametrize("key", ["libero_root", "libero_config", "osmesa_library"])
def test_every_runtime_path_requires_the_original_mapping(source, key):
    source.manifest["runtime_config"][key] = str(source.old / "unexpected")
    source.refs[0] = write_json(source.manifest_path, source.manifest)
    reject_without_destination(source)


@pytest.mark.parametrize("kind", ["hash", "missing", "escape"])
def test_historical_libero_python_claims_are_checked(source, kind):
    key = str(source.old / "LIBERO/libero/__init__.py")
    if kind == "hash":
        source.manifest["simulator_source_sha256"][key] = "1" * 64
    elif kind == "missing":
        source.manifest["simulator_source_sha256"][str(source.old / "LIBERO/libero/missing.py")] = "1" * 64
    else:
        source.manifest["simulator_source_sha256"][str(source.old / "LIBERO/libero/../../escape.py")] = "1" * 64
    source.refs[0] = write_json(source.manifest_path, source.manifest)
    reject_without_destination(source)


@pytest.mark.parametrize("kind", ["outside", "dotdot", "existing", "parent_symlink"])
def test_destination_must_be_new_and_canonical_under_repo_outputs(source, kind):
    if kind == "outside":
        source.destination = source.repo / "outside-outputs"
    elif kind == "dotdot":
        source.destination = source.repo / "outputs/../outputs/runtime-snapshot"
    elif kind == "existing":
        source.destination.mkdir()
        (source.destination / "keep.txt").write_bytes(b"do not replace")
    else:
        real_parent = source.repo / "outputs/real-parent"
        real_parent.mkdir()
        alias = source.repo / "outputs/alias"
        alias.symlink_to(real_parent, target_is_directory=True)
        source.destination = alias / "runtime-snapshot"
    with pytest.raises(ValueError):
        prepare(source)
    if kind == "existing":
        assert (source.destination / "keep.txt").read_bytes() == b"do not replace"
    else:
        assert not source.destination.exists()


@pytest.mark.parametrize("kind", ["file_symlink", "directory_symlink", "cache_symlink", "fifo"])
def test_unsupported_source_entries_are_rejected_before_copy(source, kind):
    if kind == "file_symlink":
        (source.package / "alias.py").symlink_to(source.package / "__init__.py")
    elif kind == "directory_symlink":
        (source.package / "alias_dir").symlink_to(source.package / "libero", target_is_directory=True)
    elif kind == "cache_symlink":
        (source.package / "alias.pyc").symlink_to(source.package / "__init__.py")
    else:
        os.mkfifo(source.package / "special.fifo")
    reject_without_destination(source)


@pytest.mark.parametrize("which", [0, 1, 2])
def test_symlink_input_refs_are_rejected(source, which):
    path = Path(source.refs[which]["path"])
    actual = path.with_suffix(path.suffix + ".real")
    path.rename(actual)
    path.symlink_to(actual)
    reject_without_destination(source)


@pytest.mark.parametrize("kind", ["added_file", "removed_file", "added_directory", "removed_directory", "changed_file"])
def test_source_tree_drift_is_detected_after_preparation(source, kind):
    ref = prepare(source)
    if kind == "added_file":
        (source.package / "new.py").write_bytes(b"new source\n")
    elif kind == "removed_file":
        (source.package / "libero/assets/meshes/fixture.stl").unlink()
    elif kind == "added_directory":
        (source.package / "new_empty").mkdir()
    elif kind == "removed_directory":
        (source.package / "libero/assets/empty").rmdir()
    else:
        (source.package / "libero/assets/meshes/fixture.stl").write_bytes(b"altered source bytes")
    with pytest.raises(ValueError):
        snapshot.verify_runtime_snapshot(ref)


@pytest.mark.parametrize("kind", ["added_file", "removed_file", "added_directory", "removed_directory", "changed_file"])
def test_snapshot_tree_drift_is_detected_after_preparation(source, kind):
    ref = prepare(source)
    package = source.destination / "LIBERO/libero"
    if kind == "added_file":
        (package / "new.py").write_bytes(b"undeclared copy\n")
    elif kind == "removed_file":
        (package / "libero/assets/meshes/fixture.stl").unlink()
    elif kind == "added_directory":
        (package / "new_empty").mkdir()
    elif kind == "removed_directory":
        (package / "libero/assets/empty").rmdir()
    else:
        (package / "libero/assets/meshes/fixture.stl").write_bytes(b"altered copy bytes")
    with pytest.raises(ValueError):
        snapshot.verify_runtime_snapshot(ref)


@pytest.mark.parametrize("kind", ["copy", "source", "config", "runtime", "original_manifest", "original_config"])
def test_rehashed_snapshot_manifest_cannot_bless_tampering(source, kind):
    ref = prepare(source)
    manifest = read_json(ref["path"])
    if kind in {"copy", "source"}:
        entry = next(
            entry
            for entry in manifest["file_inventory"]
            if entry["relative_path"] == "LIBERO/libero/libero/assets/meshes/fixture.stl"
        )
        ref_key = "snapshot_ref" if kind == "copy" else "source_ref"
        path = Path(entry[ref_key]["path"])
        path.write_bytes(b"changed but rehashed")
        entry[ref_key] = file_ref(path)
        entry["size_bytes"] = path.stat().st_size
    else:
        ref_key = {
            "config": "runtime_config_ref",
            "runtime": "runtime_manifest_ref",
            "original_manifest": "original_manifest_snapshot_ref",
            "original_config": "original_config_snapshot_ref",
        }[kind]
        path = Path(manifest[ref_key]["path"])
        value = read_json(path)
        if kind in {"runtime", "original_manifest"}:
            value["runtime_config"]["seed"] = 43
        else:
            value["fixture_option"]["seed"] = 43
        manifest[ref_key] = write_json(path, value)
    altered_ref = updated_snapshot_ref(ref, manifest)
    with pytest.raises(ValueError):
        snapshot.verify_runtime_snapshot(altered_ref)


def test_rehashed_manifest_inventory_cannot_hide_removed_source(source):
    ref = prepare(source)
    manifest = read_json(ref["path"])
    relative = "LIBERO/libero/libero/assets/meshes/fixture.stl"
    (source.safe / relative).unlink()
    (source.destination / relative).unlink()
    manifest["file_inventory"] = [entry for entry in manifest["file_inventory"] if entry["relative_path"] != relative]
    altered_ref = updated_snapshot_ref(ref, manifest)
    with pytest.raises(ValueError):
        snapshot.verify_runtime_snapshot(altered_ref)


@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize("kind", ["duplicate", "NaN", "Infinity", "-Infinity"])
def test_duplicate_json_keys_and_nonfinite_json_are_rejected(source, which, kind):
    path = Path(source.refs[which]["path"])
    value = read_json(path)
    if kind == "duplicate":
        key = "record" if which == 0 else "fixture_option"
        added_entry = json.dumps(key) + ": " + json.dumps(value[key])
    else:
        added_entry = '"extra_numeric_option": ' + kind
    content = json.dumps(value)[:-1] + ", " + added_entry + "}"
    path.write_text(content, encoding="utf-8")
    source.refs[which] = file_ref(path)
    reject_without_destination(source)


def test_snapshot_never_reads_dataset_or_imports_runtime(source, monkeypatch):
    def check_path(value):
        if isinstance(value, (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(value))
            assert path != source.dataset and source.dataset not in path.parents, "dataset was accessed"

    cache_files = {source.package / relative for relative in source.caches}

    def guarded(original, *, content_read):
        def call(path, *args, **kwargs):
            check_path(path)
            if content_read and isinstance(path, (str, bytes, os.PathLike)):
                assert Path(os.fsdecode(path)) not in cache_files, "cache contents were read"
            return original(path, *args, **kwargs)

        return call

    for owner, name in ((builtins, "open"), (io, "open"), (os, "open"), (os, "scandir")):
        monkeypatch.setattr(owner, name, guarded(getattr(owner, name), content_read=name != "scandir"))
    original_import = builtins.__import__

    def import_without_runtime(name, *args, **kwargs):
        assert name.split(".")[0] not in {"libero", "robosuite", "mujoco", "torch", "numba"}
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_runtime)
    snapshot.inspect_runtime_sources(*source.refs, max_files=10000, max_bytes=2**30)
    result = snapshot.verify_runtime_snapshot(prepare(source))
    assert result["dataset"]["source_read"] is False


@pytest.mark.parametrize("which", [1, 2], ids=["config_ref", "osmesa_ref"])
@pytest.mark.parametrize("entrypoint", ["prepare", "inspect"])
def test_dataset_input_refs_are_rejected_before_any_read(source, monkeypatch, which, entrypoint):
    source.refs[which] = file_ref(source.dataset / "never_read.bin")

    def guarded(original):
        def call(value, *args, **kwargs):
            if isinstance(value, (str, bytes, os.PathLike)):
                path = Path(os.fsdecode(value))
                assert path != source.dataset and source.dataset not in path.parents, "invalid dataset ref was accessed"
            return original(value, *args, **kwargs)

        return call

    for owner, name in ((builtins, "open"), (io, "open"), (os, "open"), (os, "scandir")):
        monkeypatch.setattr(owner, name, guarded(getattr(owner, name)))
    with pytest.raises(ValueError):
        if entrypoint == "prepare":
            prepare(source)
        else:
            snapshot.inspect_runtime_sources(*source.refs)
    assert not source.destination.exists()
