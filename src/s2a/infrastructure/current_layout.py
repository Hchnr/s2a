"""Explicit current source locations, separate from historical byte resolution."""

import copy
import hashlib
import json
from pathlib import Path

from s2a.infrastructure.repository_paths import _absolute_without_links, repo_path, repository_root
from s2a.infrastructure.source_identity import (
    METADATA_BYTE_LIMIT,
    _invalid_constant,
    _json_object,
    _reference,
    _relative,
    _stream_digest,
)

BASELINE_COMMIT = "7b2712850dea54090d61e564380f88825ad8b8f6"
CATALOG_REF = dict(
    path="docs/migrations/baseline_20261029.json",
    sha256="1a7d3737696e5c6cb1c4d22bc572257dd5994493a01bcd991228df9f9eb23897",
)
CURRENT_MIGRATION_PATH = "docs/migrations/repository_layout_20261029.json"
# Root inserts the approved final manifest reference after physical moves and
# import rewrites. This new module is outside the baseline mapping, so its pin
# does not introduce a source/manifest fingerprint cycle.
CURRENT_MIGRATION_REF = dict(
    path=CURRENT_MIGRATION_PATH,
    sha256="53654d7b7edc8cd919d04adf018b7c039c5b4d0aba96c562f4e620ed93828115",
)
_ROOT_METADATA = {"README.md", "AGENTS.md", "pyproject.toml", ".gitignore"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def approved_migration_ref():
    """Return an approved literal pin; never approve bytes by self-hashing them."""
    _require(CURRENT_MIGRATION_REF is not None, "Final current manifest must be explicitly approved and pinned")
    return _reference(CURRENT_MIGRATION_REF)


class CurrentLayout:
    """Consume one hash-bound current map without searching or importing sources."""

    def __init__(self, *, root=None, migration_ref=None, baseline_commit=BASELINE_COMMIT, catalog_ref=CATALOG_REF):
        self.root = repository_root() if root is None else repository_root(anchor=root)
        if root is not None:
            _require(self.root == _absolute_without_links(root), "Explicit repository root required")
        self.migration_ref = approved_migration_ref() if migration_ref is None else _reference(migration_ref)
        self.catalog_ref = _reference(catalog_ref)
        self.baseline_commit = baseline_commit
        self._metadata = {}
        self._consumed = {}
        manifest = self._read_metadata(self.migration_ref)
        _require(
            type(manifest.get("schema_version")) is int
            and manifest["schema_version"] == 1
            and manifest.get("baseline_commit") == self.baseline_commit
            and manifest.get("catalog_ref") == self.catalog_ref
            and isinstance(manifest.get("moves"), list),
            "Approved cumulative current manifest required",
        )
        self.script_namespace = manifest.get("script_namespace")
        additional = manifest.get("additional_catalog_refs", [])
        _require(isinstance(additional, list), "Explicit supplemental catalogs required")
        self.additional_catalog_refs = tuple(_reference(ref) for ref in additional)
        self._catalog_entries = {}
        for index, ref in enumerate((self.catalog_ref, *self.additional_catalog_refs)):
            catalog = self._read_metadata(ref)
            _require(
                type(catalog.get("schema_version")) is int
                and catalog["schema_version"] == 1
                and catalog.get("baseline_commit") == self.baseline_commit
                and isinstance(catalog.get("entries"), list),
                "Registered baseline source catalog required",
            )
            for entry in catalog["entries"]:
                _require(isinstance(entry, dict), "Source catalog entry required")
                old_path = _relative(entry.get("old_path")).as_posix()
                _require(old_path not in self._catalog_entries, "Duplicate or ambiguous historical source identity")
                _require(not index or old_path in _ROOT_METADATA, "Supplemental catalog may contain only root metadata")
                _require(
                    entry.get("baseline_commit") == self.baseline_commit
                    and entry.get("git_mode") in {"100644", "100755"},
                    "Regular original source identity required",
                )
                self._catalog_entries[old_path] = _reference(dict(path=old_path, sha256=entry.get("sha256")))
        self._entries = {}
        targets = set()
        for move in manifest["moves"]:
            _require(isinstance(move, dict), "Current mapping entry required")
            old_path = _relative(move.get("old_path")).as_posix()
            new_path = _relative(move.get("new_path")).as_posix()
            _require(old_path not in self._entries, "Duplicate current historical identity")
            _require(new_path not in targets, "Duplicate current physical target")
            original = _reference(dict(path=old_path, sha256=move.get("original_sha256")))
            _require(self._catalog_entries.get(old_path) == original, "Original current-map/catalog identity drift")
            current = _reference(dict(path=new_path, sha256=move.get("current_sha256")))
            self._entries[old_path] = dict(original_ref=original, current_ref=current)
            targets.add(new_path)
        self.finish()

    def _read_metadata(self, ref):
        path = Path(ref["path"])
        if path.is_absolute():
            _require(path.is_relative_to(self.root), "Current identity metadata escaped repository")
            path = path.relative_to(self.root)
        path = repo_path(path, root=self.root)
        _require(str(path) not in self._metadata, "Duplicate current identity metadata reference")
        digest, size = _stream_digest(path)
        _require(digest == ref["sha256"] and size <= METADATA_BYTE_LIMIT, "Approved metadata SHA/size mismatch")
        raw = path.read_bytes()
        _require(
            len(raw) <= METADATA_BYTE_LIMIT and hashlib.sha256(raw).hexdigest() == ref["sha256"],
            "Current identity metadata changed while reading",
        )
        result = json.loads(raw, object_pairs_hook=_json_object, parse_constant=_invalid_constant)
        _require(isinstance(result, dict), "Current identity metadata object required")
        self._metadata[str(path)] = dict(path=str(path), sha256=ref["sha256"])
        return result

    def entry(self, old_path):
        """Return declared identities without reading an absent source file."""
        old_path = _relative(old_path).as_posix()
        _require(old_path in self._entries, f"Unknown current source identity: {old_path}")
        return copy.deepcopy(self._entries[old_path])

    def source_path(self, old_path):
        """Verify the one declared current path and its SHA; no fallback search."""
        current = self.entry(old_path)["current_ref"]
        path = repo_path(current["path"], root=self.root)
        _require(_stream_digest(path)[0] == current["sha256"], "Current physical source SHA drift")
        ref = dict(path=str(path), sha256=current["sha256"])
        _require(self._consumed.get(str(path), ref) == ref, "Conflicting current physical source identity")
        self._consumed[str(path)] = ref
        return path

    def finish(self):
        """Recheck approved metadata and all current sources consumed so far."""
        for ref in (*self._metadata.values(), *self._consumed.values()):
            _require(_stream_digest(Path(ref["path"]))[0] == ref["sha256"], "Approved current identity bytes changed")

    @property
    def source_refs(self):
        return copy.deepcopy(list(self._consumed.values()))


def current_source_path(
    old_path, *, root=None, migration_ref=None, baseline_commit=BASELINE_COMMIT, catalog_ref=CATALOG_REF
):
    """Resolve one explicit current identity; historical identities stay separate."""
    layout = CurrentLayout(
        root=root, migration_ref=migration_ref, baseline_commit=baseline_commit, catalog_ref=catalog_ref
    )
    path = layout.source_path(old_path)
    layout.finish()
    return path
