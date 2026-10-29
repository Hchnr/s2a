"""Locate registered original source bytes; never admit them for execution.

References retain their declared path and SHA256. Current source, migration
targets and a local Git blob cache are separate physical locations. No source
is imported, no historical workspace is accessed and no file API is patched.
"""

import copy
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from s2a.infrastructure.repository_paths import (
    FORBIDDEN_HISTORICAL_ROOT,
    _absolute_without_links,
    repo_path,
    repository_root,
)

CHUNK_SIZE = 1024 * 1024
METADATA_BYTE_LIMIT = 32 * 1024 * 1024
IDENTITY_SCOPE = "byte_identity_only"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_OID = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_AREAS = {"src", "scripts", "tests", "configs", "plans", "reports", "docs"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _reference(ref):
    _require(
        isinstance(ref, dict)
        and set(ref) == {"path", "sha256"}
        and isinstance(ref["path"], str)
        and bool(ref["path"])
        and isinstance(ref["sha256"], str)
        and _SHA256.fullmatch(ref["sha256"]),
        "Exact path/SHA256 reference required",
    )
    return copy.deepcopy(ref)


def _relative(path):
    _require(isinstance(path, str) and bool(path), "String relative path required")
    result = Path(path)
    _require(
        not result.is_absolute() and bool(result.parts) and ".." not in result.parts and result.as_posix() == path,
        "Canonical repository-relative path required",
    )
    _require(result.parts[0] in _AREAS, "Source path is outside registered repository areas")
    return result


def _stream_digest(path):
    path = _absolute_without_links(path)
    before = path.lstat()
    _require(stat.S_ISREG(before.st_mode), "Regular source file required")
    digest, observed = hashlib.sha256(), 0
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        _require(stat.S_ISREG(opened.st_mode), "Regular opened source required")
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
            observed += len(chunk)
        finished = os.fstat(stream.fileno())
    after = path.lstat()

    def identity(details):
        return details.st_dev, details.st_ino, details.st_size, details.st_mtime_ns

    _require(
        identity(before) == identity(opened) == identity(finished) == identity(after) and observed == before.st_size,
        "Source changed during streaming SHA256 verification",
    )
    return digest.hexdigest(), observed


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON value is forbidden: {value}")


@dataclass(frozen=True)
class ResolvedSourceIdentity:
    """A byte identity receipt; it confers no runtime or control qualification."""

    declared_ref: dict
    storage_ref: dict
    migration_ref: dict | None
    catalog_ref: dict
    resolution_kind: str
    origin: dict
    identity_scope: str = IDENTITY_SCOPE


class SourceIdentityLocator:
    """Resolve only catalogued identities against explicit, verified locations."""

    def __init__(self, *, root, baseline_commit, catalog_ref, migration_refs=(), cache_directory):
        self.root = _absolute_without_links(root)
        _require(repository_root(anchor=self.root) == self.root, "Explicit S2A repository root required")
        _require(
            isinstance(baseline_commit, str) and _GIT_OID.fullmatch(baseline_commit),
            "Full Git baseline commit required",
        )
        self.baseline_commit = baseline_commit
        self.catalog_ref = _reference(catalog_ref)
        self.migration_refs = tuple(_reference(ref) for ref in migration_refs)
        self.cache_directory = self._storage_path(cache_directory)
        _require(
            self.cache_directory.is_relative_to(self.root / "outputs")
            and self.cache_directory != self.root / "outputs",
            "Dedicated cache directory must be inside repository outputs",
        )
        self._consumed = {}
        self._metadata = {}
        self._git_operations = []
        self._verified_git_entries = set()
        self._entries = self._catalog(self._read_metadata(self.catalog_ref))
        self._moves = self._migrations()
        top_level = self._git(["rev-parse", "--show-toplevel"]).decode().strip()
        _require(top_level == str(self.root), "Explicit root must be the local Git worktree root")
        resolved = self._git(["rev-parse", "--verify", f"{baseline_commit}^{{commit}}"]).decode().strip()
        _require(resolved == baseline_commit, "Baseline must identify the exact local commit")

    @property
    def git_operations(self):
        """Actual subprocess wait/reap receipts, including failed Git commands."""
        return copy.deepcopy(self._git_operations)

    def _storage_path(self, value):
        path = Path(value)
        path = _absolute_without_links(path) if path.is_absolute() else repo_path(path, root=self.root)
        _require(path.is_relative_to(self.root) and path != self.root, "Storage path escaped repository root")
        _require(not path.is_relative_to(self.root / ".git"), "Git metadata is not a source storage location")
        return path

    def _track(self, path, expected, *, size=None):
        observed, actual_size = _stream_digest(path)
        _require(observed == expected, f"Storage SHA256 mismatch: {path}")
        _require(size is None or actual_size == size, f"Storage byte count mismatch: {path}")
        _require(self._consumed.get(str(path), expected) == expected, "Conflicting storage identity")
        self._consumed[str(path)] = expected
        return dict(path=str(path), sha256=observed)

    def _read_metadata(self, ref):
        path = self._storage_path(ref["path"])
        self._track(path, ref["sha256"])
        self._metadata[str(path)] = ref["sha256"]
        _require(path.stat().st_size <= METADATA_BYTE_LIMIT, "Metadata exceeds byte budget")
        with path.open("rb") as stream:
            raw = stream.read(METADATA_BYTE_LIMIT + 1)
        _require(len(raw) <= METADATA_BYTE_LIMIT, "Metadata exceeds byte budget")
        _require(hashlib.sha256(raw).hexdigest() == ref["sha256"], "Metadata changed before consumption")
        payload = json.loads(raw, object_pairs_hook=_json_object, parse_constant=_invalid_constant)
        _require(isinstance(payload, dict), "Metadata object required")
        return payload

    def _catalog(self, catalog):
        _require(
            type(catalog.get("schema_version")) is int
            and catalog["schema_version"] == 1
            and catalog.get("baseline_commit") == self.baseline_commit
            and isinstance(catalog.get("entries"), list)
            and bool(catalog["entries"]),
            "Registered baseline catalog schema required",
        )
        entries = {}
        for entry in catalog["entries"]:
            _require(
                isinstance(entry, dict)
                and set(entry) == {"old_path", "sha256", "size_bytes", "git_blob_oid", "git_mode", "baseline_commit"},
                "Exact baseline entry schema required",
            )
            relative = _relative(entry["old_path"]).as_posix()
            _require(
                isinstance(entry["sha256"], str)
                and _SHA256.fullmatch(entry["sha256"])
                and type(entry["size_bytes"]) is int
                and entry["size_bytes"] >= 0
                and isinstance(entry["git_blob_oid"], str)
                and _GIT_OID.fullmatch(entry["git_blob_oid"])
                and entry["git_mode"] in {"100644", "100755"}
                and entry["baseline_commit"] == self.baseline_commit,
                "Malformed baseline source identity",
            )
            _require(relative not in entries, "Duplicate or ambiguous baseline path")
            entries[relative] = copy.deepcopy(entry)
        _require(
            type(catalog.get("file_count")) is int
            and catalog["file_count"] == len(entries)
            and type(catalog.get("total_size_bytes")) is int
            and catalog["total_size_bytes"] == sum(entry["size_bytes"] for entry in entries.values()),
            "Baseline catalog counts differ from its complete entries",
        )
        return entries

    def _migrations(self):
        moves, destinations, manifests = {}, set(), set()
        for ref in self.migration_refs:
            path = str(self._storage_path(ref["path"]))
            _require(path not in manifests, "Duplicate migration manifest")
            manifests.add(path)
            manifest = self._read_metadata(ref)
            _require(
                manifest.get("catalog_ref") == self.catalog_ref and isinstance(manifest.get("moves"), list),
                "Migration manifest must bind the exact catalog reference",
            )
            for move in manifest["moves"]:
                _require(
                    isinstance(move, dict)
                    and set(move) == {"old_path", "original_sha256", "new_path", "current_sha256"},
                    "Exact migration move schema required",
                )
                old = _relative(move["old_path"]).as_posix()
                new = _relative(move["new_path"]).as_posix()
                _require(
                    old in self._entries and move["original_sha256"] == self._entries[old]["sha256"],
                    "Migration original identity is not registered in baseline",
                )
                _require(
                    isinstance(move["current_sha256"], str) and _SHA256.fullmatch(move["current_sha256"]),
                    "Migration current SHA256 required",
                )
                _require(old not in moves and new not in destinations, "Duplicate or ambiguous migration move")
                moves[old] = dict(move=copy.deepcopy(move), manifest_ref=ref)
                destinations.add(new)
        return moves

    def _git(self, args, *, destination=None):
        command = ["git", "--no-optional-locks", "--literal-pathspecs", "-C", str(self.root), *args]
        environment = dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_NO_REPLACE_OBJECTS="1", GIT_TERMINAL_PROMPT="0")
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE if destination is None else destination,
            stderr=subprocess.PIPE,
            env=environment,
        )
        receipt = dict(
            command=command, pid=process.pid, actual_wait=False, reaped=False, returncode=None, signal_sent=False
        )
        self._git_operations.append(receipt)
        try:
            stdout, stderr = process.communicate(timeout=30)
        except BaseException:
            if process.poll() is None:
                process.kill()
                receipt["signal_sent"] = True
            process.wait()
            receipt.update(actual_wait=True, reaped=True, returncode=process.returncode)
            raise
        receipt.update(actual_wait=True, reaped=True, returncode=process.returncode)
        _require(
            process.returncode == 0, f"Local Git identity command failed: {stderr.decode(errors='replace').strip()}"
        )
        return stdout

    def _verify_git_entry(self, entry):
        old = entry["old_path"]
        if old not in self._verified_git_entries:
            expected = f"{entry['git_mode']} blob {entry['git_blob_oid']}\t{old}".encode()
            observed = self._git(["ls-tree", "-z", self.baseline_commit, "--", old])
            _require(observed == expected + b"\0", "Catalog Git blob differs from registered commit/path")
            self._verified_git_entries.add(old)

    def _cache(self, entry):
        digest = entry["sha256"]
        target = self._storage_path(self.cache_directory / digest[:2] / digest)
        if target.exists():
            return self._track(target, digest, size=entry["size_bytes"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target = self._storage_path(target)
        descriptor, temporary = tempfile.mkstemp(prefix=".source-", dir=target.parent)
        temporary = Path(temporary)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                self._git(["cat-file", "blob", entry["git_blob_oid"]], destination=stream)
                stream.flush()
                os.fsync(stream.fileno())
            observed, size = _stream_digest(temporary)
            _require(
                observed == digest and size == entry["size_bytes"], "Git blob bytes differ from catalog SHA256/size"
            )
            try:
                os.link(temporary, target)
            except FileExistsError:
                pass
            return self._track(target, digest, size=entry["size_bytes"])
        finally:
            temporary.unlink(missing_ok=True)

    def _declared_relative(self, ref):
        declared = Path(ref["path"])
        _require(".." not in declared.parts and str(declared) == ref["path"], "Canonical declaration required")
        if declared.is_absolute():
            if declared.is_relative_to(self.root):
                declared = declared.relative_to(self.root)
            elif declared.is_relative_to(FORBIDDEN_HISTORICAL_ROOT):
                declared = declared.relative_to(FORBIDDEN_HISTORICAL_ROOT)
            else:
                raise ValueError("Declared source is outside registered repository roots")
        return _relative(declared.as_posix()).as_posix()

    def resolve(self, declared_ref):
        """Locate exact registered bytes without importing or qualifying source."""
        declared_ref = _reference(declared_ref)
        relative = self._declared_relative(declared_ref)
        entry = self._entries.get(relative)
        _require(entry is not None and declared_ref["sha256"] == entry["sha256"], "Unknown historical source identity")
        self._recheck(self._metadata)
        self._verify_git_entry(entry)
        migration = self._moves.get(relative)
        migrated = None
        if migration is not None:
            move = migration["move"]
            target = self._storage_path(move["new_path"])
            # A declared current target is a tamper guard even when original
            # bytes are still available elsewhere; never hide drift by fallback.
            migrated = self._track(target, move["current_sha256"])
        active = self._storage_path(relative)
        if active.is_file() and _stream_digest(active) == (entry["sha256"], entry["size_bytes"]):
            storage, kind = self._track(active, entry["sha256"], size=entry["size_bytes"]), "active"
        elif migrated is not None and migrated["sha256"] == entry["sha256"]:
            storage, kind = migrated, "migrated"
            self._track(Path(storage["path"]), entry["sha256"], size=entry["size_bytes"])
        else:
            storage, kind = self._cache(entry), "git_cache"
        return ResolvedSourceIdentity(
            declared_ref=declared_ref,
            storage_ref=storage,
            migration_ref=None if migration is None else copy.deepcopy(migration["manifest_ref"]),
            catalog_ref=copy.deepcopy(self.catalog_ref),
            resolution_kind=kind,
            origin=copy.deepcopy(entry),
        )

    def finish(self):
        """Recheck consumed metadata, current targets and source/cache bytes."""
        self._recheck(self._consumed)

    @staticmethod
    def _recheck(mapping):
        for path, expected in mapping.items():
            observed, _ = _stream_digest(Path(path))
            _require(observed == expected, f"Consumed source or manifest changed: {path}")
