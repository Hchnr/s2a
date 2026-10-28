"""Prepare a file-only LIBERO/OSMesa snapshot with reversible path edits.

Source identities are rechecked, including files lacking historical hashes.
No simulator, dynamic library or training dataset is loaded by this module.
The returned manifest reference must be retained as the external trust anchor.
"""

import copy
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SAFE_SOURCE_ROOT = Path("/share-mixed/project/eai_pwm/datasets/cosmos3_action_tower_training/libero_sim")
OLD_SOURCE_ROOT = Path("/share/project/eai_pwm/datasets/cosmos3_action_tower_training/libero_sim")
SCHEMA = "s2a.native-runtime-path-snapshot.v1"
CACHE_DIRECTORIES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
CACHE_SUFFIXES = {".pyc", ".pyo"}
CONFIG_PATHS = {
    "benchmark_root": "LIBERO/libero/libero",
    "bddl_files": "LIBERO/libero/libero/bddl_files",
    "init_states": "LIBERO/libero/libero/init_files",
    "assets": "LIBERO/libero/libero/assets",
    "datasets": "data",
}
FLAGS = dict(runtime_loaded=False, bootstrap_verified=False, control_qualified=False)
CHUNK_SIZE = 1024 * 1024


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def number(token):
        result = float(token)
        _require(math.isfinite(result), "Finite JSON numbers required")
        return result

    def invalid(_token):
        raise ValueError("Nonfinite JSON constants forbidden")

    value = json.loads(raw, object_pairs_hook=pairs, parse_float=number, parse_constant=invalid)
    _require(isinstance(value, dict), "JSON object required")
    return value


def _encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _no_links(path):
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        _require(not stat.S_ISLNK(mode), "Symlink path components forbidden")


def _path(path, root):
    path, root = Path(path), Path(root)
    _require(path.is_absolute() and root.is_absolute(), "Absolute canonical paths required")
    _require(".." not in path.parts and ".." not in root.parts, "Path traversal forbidden")
    _require(path.is_relative_to(root), "Path escapes permitted root")
    _no_links(path)
    _require(path.resolve() == path and root.resolve() == root, "Canonical paths required")
    return path


def _regular(path):
    _no_links(path)
    try:
        details = path.lstat()
    except OSError as error:
        raise ValueError(f"Missing source or copied file: {path}") from error
    _require(stat.S_ISREG(details.st_mode), "Only regular files may be copied or read")
    return details.st_size


def _digest_file(path):
    size = _regular(path)
    digest, observed = hashlib.sha256(), 0
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as stream:
            _require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "Regular opened source required")
            while chunk := stream.read(CHUNK_SIZE):
                digest.update(chunk)
                observed += len(chunk)
    except OSError as error:
        raise ValueError(f"Cannot read regular file: {path}") from error
    _require(observed == size == _regular(path), "File size changed while reading")
    return dict(path=str(path), sha256=digest.hexdigest()), size


def _ref(ref, root):
    _require(
        isinstance(ref, dict)
        and set(ref) == {"path", "sha256"}
        and isinstance(ref["path"], str)
        and isinstance(ref["sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]),
        "Exact path/SHA256 reference required",
    )
    path = _path(ref["path"], root)
    _require(str(path) == ref["path"], "Canonical reference spelling required")
    actual, _ = _digest_file(path)
    _require(actual == ref, "Source or copied file SHA mismatch")
    return path


def _read(ref, root, max_bytes):
    path = _path(ref["path"], root)
    _require(_regular(path) <= max_bytes, "Input exceeds byte budget")
    path = _ref(ref, root)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError("Cannot read referenced JSON input") from error
    _require(hashlib.sha256(raw).hexdigest() == ref["sha256"], "Input changed while reading")
    return raw


def _budget(max_files, max_bytes):
    _require(type(max_files) is int and max_files > 0, "Positive integer file budget required")
    _require(type(max_bytes) is int and max_bytes > 0, "Positive integer byte budget required")
    return dict(max_files=max_files, max_bytes=max_bytes)


def _old_relative(value):
    _require(isinstance(value, str), "Declared source path must be a string")
    path = Path(value)
    _require(path.is_absolute() and ".." not in path.parts, "Absolute nontraversing historical path required")
    _require(str(path) == value and path.is_relative_to(OLD_SOURCE_ROOT), "Historical path escapes original root")
    return path.relative_to(OLD_SOURCE_ROOT)


def _walk(root, *, exclude_cache):
    """List regular files and empty directories; inspect but never read caches."""
    _no_links(root)
    try:
        _require(stat.S_ISDIR(root.lstat().st_mode), "Required package or snapshot directory missing")
    except OSError as error:
        raise ValueError("Required directory missing") from error
    files, directories, excluded = [], [root], []

    def visit(directory, inherited=False):
        try:
            with os.scandir(directory) as stream:
                entries = sorted(stream, key=lambda entry: entry.name)
        except OSError as error:
            raise ValueError("Cannot enumerate complete file tree") from error
        for entry in entries:
            path = directory / entry.name
            mode = entry.stat(follow_symlinks=False).st_mode
            _require(stat.S_ISDIR(mode) or stat.S_ISREG(mode), "Symlink and special file entries forbidden")
            is_directory = stat.S_ISDIR(mode)
            cached = exclude_cache and (
                inherited or (is_directory and entry.name in CACHE_DIRECTORIES) or path.suffix in CACHE_SUFFIXES
            )
            if cached:
                excluded.append(dict(path=str(path), kind="directory" if is_directory else "file", reason="cache"))
            elif is_directory:
                directories.append(path)
            else:
                files.append(path)
            if is_directory:
                visit(path, cached)

    visit(root)
    return sorted(files), sorted(directories), sorted(excluded, key=lambda item: item["path"])


def _sources(manifest_ref, config_ref, osmesa_ref, *, max_files, max_bytes):
    budgets = _budget(max_files, max_bytes)
    # Reject misplaced references before opening even an invalid caller input;
    # an alleged configuration or library must never make us read training data.
    for ref, subdirectory in ((config_ref, "config"), (osmesa_ref, "systemlibs")):
        _require(isinstance(ref, dict) and isinstance(ref.get("path"), str), "Explicit safe source reference required")
        _path(ref["path"], SAFE_SOURCE_ROOT / subdirectory)
    manifest_raw = _read(manifest_ref, REPO, max_bytes)
    config_raw = _read(config_ref, SAFE_SOURCE_ROOT, max_bytes)
    manifest, config = _json(manifest_raw), _json(config_raw)
    runtime = manifest.get("runtime_config")
    _require(isinstance(runtime, dict), "Original runtime_config object required")
    _require(_old_relative(runtime.get("libero_root")) == Path("LIBERO"), "Original complete LIBERO root required")
    for key, supplied in (("libero_config", config_ref), ("osmesa_library", osmesa_ref)):
        relative = _old_relative(runtime.get(key))
        _require(
            supplied["path"] == str(SAFE_SOURCE_ROOT / relative),
            "Source reference differs from exact old-to-safe mapping",
        )
    _require(
        _old_relative(runtime["osmesa_library"]).is_relative_to("systemlibs"), "OSMesa must be from safe systemlibs"
    )
    for key, relative in CONFIG_PATHS.items():
        _require(
            config.get(key) == str(OLD_SOURCE_ROOT / relative),
            "Exactly the registered five inner path mappings required",
        )
    osmesa_path = _ref(osmesa_ref, SAFE_SOURCE_ROOT)
    package = _path(SAFE_SOURCE_ROOT / "LIBERO/libero", SAFE_SOURCE_ROOT)
    files, directories, excluded = _walk(package, exclude_cache=True)
    entries = [(path, path.relative_to(SAFE_SOURCE_ROOT).as_posix(), "libero_package") for path in files]
    license_path = _path(SAFE_SOURCE_ROOT / "LIBERO/LICENSE", SAFE_SOURCE_ROOT)
    license_included = os.path.lexists(license_path)
    if license_included:
        _regular(license_path)
        entries.append((license_path, "LIBERO/LICENSE", "libero_license"))
    entries.append((osmesa_path, f"systemlibs/{osmesa_path.name}", "osmesa"))
    _require(len(entries) <= max_files, "Source inventory exceeds file budget")
    sizes = [_regular(path) for path, _, _ in entries]
    _require(sum(sizes) + len(manifest_raw) + len(config_raw) <= max_bytes, "Source inventory exceeds byte budget")

    declared = manifest.get("simulator_source_sha256")
    _require(isinstance(declared, dict), "Historical simulator source map required")
    history = {}
    historical_package = OLD_SOURCE_ROOT / "LIBERO/libero"
    for name, digest in declared.items():
        _require(isinstance(name, str), "Historical source map keys must be paths")
        path = Path(name)
        if not path.is_absolute() or not path.is_relative_to(historical_package):
            continue
        relative = _old_relative(name)
        _require(
            path.suffix == ".py" and isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
            "Valid historical LIBERO Python SHA required",
        )
        history[relative.as_posix()] = dict(path=name, sha256=digest)
    inventory = []
    for (path, relative, role), size in sorted(zip(entries, sizes, strict=True), key=lambda item: item[0][1]):
        source_ref, observed_size = _digest_file(path)
        _require(size == observed_size, "Source changed during inventory")
        historical = history.get(relative)
        if historical is not None:
            _require(source_ref["sha256"] == historical["sha256"], "Historical LIBERO source SHA mismatch")
        inventory.append(
            dict(
                relative_path=relative,
                role=role,
                source_ref=source_ref,
                size_bytes=size,
                historical_source_ref=historical,
            )
        )
    _require(
        set(history) <= {item["relative_path"] for item in inventory},
        "Historical LIBERO Python missing from copied inventory",
    )
    refs = {item["source_ref"]["path"]: item["source_ref"] for item in inventory}
    for ref in (manifest_ref, config_ref, osmesa_ref):
        _require(refs.get(ref["path"], ref) == ref, "One source identity per path")
        refs[ref["path"]] = dict(ref)
    result = dict(
        input_refs=dict(manifest_ref=manifest_ref, config_ref=config_ref, osmesa_ref=osmesa_ref),
        source_refs=[refs[key] for key in sorted(refs)],
        source_inventory=inventory,
        source_directories=[path.relative_to(SAFE_SOURCE_ROOT).as_posix() for path in directories],
        excluded_cache_entries=excluded,
        historical_libero_file_count=len(history),
        libero_file_count=len(files),
        copied_file_count=len(inventory),
        copied_bytes=sum(sizes),
        license_included=license_included,
        budgets=budgets,
        file_snapshot_verified=False,
        **FLAGS,
    )
    for ref in result["source_refs"]:
        _ref(ref, REPO if ref == manifest_ref else SAFE_SOURCE_ROOT)
    return result, manifest_raw, config_raw, manifest, config


def inspect_runtime_sources(manifest_ref, config_ref, osmesa_ref, *, max_files=10000, max_bytes=2**30):
    """Preflight complete sources only; no destination, dataset or runtime access."""
    return _sources(manifest_ref, config_ref, osmesa_ref, max_files=max_files, max_bytes=max_bytes)[0]


def _destination(destination, *, new):
    destination = Path(destination)
    if not destination.is_absolute():
        destination = REPO / destination
    destination = _path(destination, REPO / "outputs")
    _require(destination != REPO / "outputs", "A dedicated output directory is required")
    if new:
        _require(not os.path.lexists(destination), "Destination already exists; no overwrite or retry")
    return destination


def _bytes_ref(path, raw):
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())


def _layout(destination, sources, manifest_raw, config_raw, manifest, config):
    runtime_paths = dict(
        libero_root=destination / "LIBERO",
        libero_config=destination / "config/config.yaml",
        osmesa_library=destination / "systemlibs" / Path(sources["input_refs"]["osmesa_ref"]["path"]).name,
    )
    derived_manifest, derived_config = copy.deepcopy(manifest), copy.deepcopy(config)
    mappings = dict(runtime_config={}, libero_config={})
    for key, path in runtime_paths.items():
        mappings["runtime_config"][key] = dict(original_path=manifest["runtime_config"][key], snapshot_path=str(path))
        derived_manifest["runtime_config"][key] = str(path)
    for key, relative in CONFIG_PATHS.items():
        mappings["libero_config"][key] = dict(original_path=config[key], snapshot_path=str(destination / relative))
        derived_config[key] = str(destination / relative)
    payloads = {
        "original_manifest.json": manifest_raw,
        "original_config.yaml": config_raw,
        "runtime_manifest.json": _encoded(derived_manifest),
        "config/config.yaml": _encoded(derived_config),
        "source_inventory.json": _encoded(sources),
    }
    files = [
        dict(
            item, snapshot_ref=dict(path=str(destination / item["relative_path"]), sha256=item["source_ref"]["sha256"])
        )
        for item in sources["source_inventory"]
    ]
    result = dict(
        schema_version=SCHEMA,
        original_manifest_ref=sources["input_refs"]["manifest_ref"],
        config_source_ref=sources["input_refs"]["config_ref"],
        osmesa_source_ref=sources["input_refs"]["osmesa_ref"],
        original_manifest_snapshot_ref=_bytes_ref(destination / "original_manifest.json", manifest_raw),
        original_config_snapshot_ref=_bytes_ref(destination / "original_config.yaml", config_raw),
        runtime_manifest_ref=_bytes_ref(destination / "runtime_manifest.json", payloads["runtime_manifest.json"]),
        runtime_config_ref=_bytes_ref(destination / "config/config.yaml", payloads["config/config.yaml"]),
        source_inventory_ref=_bytes_ref(destination / "source_inventory.json", payloads["source_inventory.json"]),
        source_root=str(SAFE_SOURCE_ROOT),
        old_source_root=str(OLD_SOURCE_ROOT),
        snapshot_root=str(destination),
        file_inventory=files,
        source_directories=sources["source_directories"],
        excluded_cache_entries=sources["excluded_cache_entries"],
        historical_libero_file_count=sources["historical_libero_file_count"],
        copied_file_count=sources["copied_file_count"],
        copied_bytes=sources["copied_bytes"],
        budgets=sources["budgets"],
        path_mappings=mappings,
        dataset=dict(path=str(destination / "data"), populated=False, source_read=False),
        file_snapshot_verified=True,
        **FLAGS,
    )
    payloads["manifest.json"] = _encoded(result)
    _require(len(files) + len(payloads) <= sources["budgets"]["max_files"], "Complete output exceeds file budget")
    _require(
        sources["copied_bytes"] + sum(map(len, payloads.values())) <= sources["budgets"]["max_bytes"],
        "Complete output exceeds byte budget",
    )
    return result, payloads


def _directories(destination, sources, payloads):
    result = {destination, destination / "data"}
    for relative in sources["source_directories"]:
        result.add(destination / relative)
    for relative in [*payloads, *(item["relative_path"] for item in sources["source_inventory"])]:
        parent = (destination / relative).parent
        while parent != destination:
            result.add(parent)
            parent = parent.parent
    for directory in list(result):
        parent = directory.parent
        while parent != destination and parent.is_relative_to(destination):
            result.add(parent)
            parent = parent.parent
    return result


def _write(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)


def _copy(item, destination):
    source = _ref(item["source_ref"], SAFE_SOURCE_ROOT)
    target = _path(destination / item["relative_path"], destination)
    digest, count = hashlib.sha256(), 0
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as incoming, target.open("xb") as outgoing:
        _require(stat.S_ISREG(os.fstat(incoming.fileno()).st_mode), "Regular streaming source required")
        while chunk := incoming.read(CHUNK_SIZE):
            outgoing.write(chunk)
            digest.update(chunk)
            count += len(chunk)
    _require(
        count == item["size_bytes"] and digest.hexdigest() == item["source_ref"]["sha256"],
        "Source changed during streaming copy",
    )
    _ref(item["source_ref"], SAFE_SOURCE_ROOT)
    _require(_digest_file(target) == (dict(path=str(target), sha256=digest.hexdigest()), count), "Copied bytes differ")


def prepare_runtime_snapshot(manifest_ref, config_ref, osmesa_ref, destination, *, max_files=10000, max_bytes=2**30):
    """Create one exclusive file snapshot after all source and output-budget checks."""
    destination = _destination(destination, new=True)
    sources, *originals = _sources(manifest_ref, config_ref, osmesa_ref, max_files=max_files, max_bytes=max_bytes)
    _, payloads = _layout(destination, sources, *originals)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _destination(destination, new=True)
    destination.mkdir()
    for directory in sorted(
        _directories(destination, sources, payloads) - {destination}, key=lambda path: (len(path.parts), str(path))
    ):
        directory.mkdir()
    for item in sources["source_inventory"]:
        _copy(item, destination)
    for name, raw in payloads.items():
        if name != "manifest.json":
            _write(destination / name, raw)
    fresh, *_ = _sources(manifest_ref, config_ref, osmesa_ref, max_files=max_files, max_bytes=max_bytes)
    _require(fresh == sources, "Source inventory changed while preparing snapshot")
    _write(destination / "manifest.json", payloads["manifest.json"])
    ref = _bytes_ref(destination / "manifest.json", payloads["manifest.json"])
    verify_runtime_snapshot(ref)
    return ref


def verify_runtime_snapshot(ref):
    """Rebuild every source and copy identity plus the exact three/five path edits."""
    path = _ref(ref, REPO / "outputs")
    destination = _destination(path.parent, new=False)
    _require(path == destination / "manifest.json", "Canonical snapshot manifest required")
    recorded = _json(_read(ref, destination, path.stat().st_size))
    inventory_ref = recorded["source_inventory_ref"]
    _require(
        inventory_ref["path"] == str(destination / "source_inventory.json"), "Canonical captured inventory required"
    )
    captured = _json(_read(inventory_ref, destination, recorded["budgets"]["max_bytes"]))
    budgets, inputs = captured["budgets"], captured["input_refs"]
    sources, *originals = _sources(inputs["manifest_ref"], inputs["config_ref"], inputs["osmesa_ref"], **budgets)
    _require(captured == sources, "Source inventory differs from the captured pre-copy identity")
    expected, payloads = _layout(destination, sources, *originals)
    _require(recorded == expected, "Snapshot metadata or three/five path mapping differs")
    files, directories, excluded = _walk(destination, exclude_cache=False)
    expected_files = {destination / name for name in payloads} | {
        destination / item["relative_path"] for item in sources["source_inventory"]
    }
    _require(
        set(files) == expected_files
        and set(directories) == _directories(destination, sources, payloads)
        and not excluded,
        "Snapshot has missing or additional files/directories",
    )
    for name, raw in payloads.items():
        _require(
            (destination / name).read_bytes() == raw,
            "Original bytes or migrated configuration differ from exact path-only transformation",
        )
    for item in expected["file_inventory"]:
        _ref(item["snapshot_ref"], destination)
        _require(_regular(Path(item["snapshot_ref"]["path"])) == item["size_bytes"], "Copied size mismatch")
    # Explicit inverse check avoids treating a self-consistent rehash as proof of
    # a changed non-path parameter. The original input refs stay independent.
    restored_manifest = _json(payloads["runtime_manifest.json"])
    restored_config = _json(payloads["config/config.yaml"])
    for key, change in expected["path_mappings"]["runtime_config"].items():
        restored_manifest["runtime_config"][key] = change["original_path"]
    for key, change in expected["path_mappings"]["libero_config"].items():
        restored_config[key] = change["original_path"]
    _require(
        restored_manifest == originals[2] and restored_config == originals[3], "Inverse configuration migration differs"
    )
    return recorded
