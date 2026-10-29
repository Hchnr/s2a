"""Keep current-source checks separate from registered historical byte proofs."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

from s2a.infrastructure.current_layout import BASELINE_COMMIT, CATALOG_REF, current_source_path
from s2a.infrastructure.repository_paths import repo_path, repository_root
from s2a.infrastructure.source_identity import SourceIdentityLocator

__all__ = [
    "current_module_name",
    "current_source_path",
    "historical_import_closure",
    "historical_source_bytes",
    "historical_source_ref",
]


def current_module_name(old_path):
    """Spell the module at one verified current source path without importing it."""
    path = current_source_path(old_path).relative_to(repository_root()).with_suffix("")
    parts = path.parts[1:] if path.parts[0] == "src" else path.parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def historical_source_bytes(ref):
    """Read one exact baseline identity; these bytes confer no execution rights."""
    root = repository_root()
    locator = SourceIdentityLocator(
        root=root,
        baseline_commit=BASELINE_COMMIT,
        catalog_ref=CATALOG_REF,
        cache_directory=root / "outputs/repository_test_historical_source_cache_20261029",
    )
    resolved = locator.resolve(ref)
    data = Path(resolved.storage_ref["path"]).read_bytes()
    locator.finish()
    return data


def _historical_refs():
    raw = repo_path(CATALOG_REF["path"], root=repository_root()).read_bytes()
    if hashlib.sha256(raw).hexdigest() != CATALOG_REF["sha256"]:
        raise ValueError("Registered historical catalog bytes changed")
    return {row["old_path"]: dict(path=row["old_path"], sha256=row["sha256"]) for row in json.loads(raw)["entries"]}


def historical_source_ref(old_path):
    """Name one explicit identity from the approved historical catalog."""
    refs = _historical_refs()
    if old_path not in refs:
        raise ValueError("Unregistered historical source: " + old_path)
    return refs[old_path]


def historical_import_closure(entrypoints, directory):
    """Stage only registered static source dependencies for archival CLI help."""
    root = repository_root()
    directory = repo_path(Path(directory).relative_to(root), root=root)
    if not directory.is_relative_to(root / "outputs"):
        raise ValueError("Archival CLI fixture must stay inside repository outputs")
    directory.mkdir(parents=True, exist_ok=False)
    entries = {
        relative: ref
        for relative, ref in _historical_refs().items()
        if relative.startswith(("src/", "scripts/")) and relative.endswith(".py")
    }
    modules = {}
    for relative in entries:
        parts = Path(relative).with_suffix("").parts
        if parts[0] == "src":
            parts = parts[1:]
        if parts[-1] == "__init__":
            parts = parts[:-1]
        module = ".".join(parts)
        modules[module] = relative
        if relative.startswith("scripts/") and len(Path(relative).parts) == 2:
            modules[Path(relative).stem] = relative
    locator = SourceIdentityLocator(
        root=root,
        baseline_commit=BASELINE_COMMIT,
        catalog_ref=CATALOG_REF,
        cache_directory=root / "outputs/repository_test_historical_source_cache_20261029",
    )
    pending = set(entrypoints) | {"src/s2a/__init__.py"}
    records = {}

    def add_module(module):
        parts = module.split(".")
        for length in range(1, len(parts) + 1):
            relative = modules.get(".".join(parts[:length]))
            if relative is not None:
                pending.add(relative)

    while pending:
        relative = min(pending)
        pending.remove(relative)
        if relative in records:
            continue
        if relative not in entries:
            raise ValueError("Unregistered historical import source: " + relative)
        original_ref = entries[relative]
        resolved = locator.resolve(original_ref)
        data = Path(resolved.storage_ref["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != original_ref["sha256"]:
            raise ValueError("Historical CLI source changed while reading")
        target = repo_path((directory / relative).relative_to(root), root=root)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
        records[relative] = dict(original_ref=original_ref, staged_path=str(target), storage_ref=resolved.storage_ref)
        parts = Path(relative).with_suffix("").parts
        if parts[0] == "src":
            parts = parts[1:]
        package = ".".join(parts[:-1])
        for node in ast.walk(ast.parse(data, filename=relative)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add_module(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    module = importlib.util.resolve_name("." * node.level + module, package)
                add_module(module)
                for alias in node.names:
                    if alias.name != "*":
                        add_module(module + "." + alias.name)
    locator.finish()
    for record in records.values():
        if hashlib.sha256(Path(record["staged_path"]).read_bytes()).hexdigest() != record["original_ref"]["sha256"]:
            raise ValueError("Staged archival CLI source drift")
    receipt = dict(
        identity_scope="isolated_archival_help_only_no_runtime_or_control_qualification",
        catalog_ref=CATALOG_REF,
        baseline_commit=BASELINE_COMMIT,
        entrypoints=list(entrypoints),
        source_count=len(records),
        sources=records,
        git_operations=locator.git_operations,
    )
    with (directory / "closure.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return receipt
