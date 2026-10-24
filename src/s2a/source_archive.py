"""Verify historical source bytes without repairing or importing missing sources."""

from collections.abc import Mapping
from pathlib import Path

from s2a.artifacts import sha256


def _entries(mapping, name):
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError(f"{name} must be a nonempty path-to-SHA256 mapping")
    entries = []
    seen = set()
    for original_path, expected in mapping.items():
        path = Path(original_path)
        if not path.is_absolute():
            raise ValueError(f"{name} path must be absolute: {path}")
        normalized = path.resolve()
        if normalized in seen:
            raise ValueError(f"duplicate normalized {name} path: {path}")
        seen.add(normalized)
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError(f"invalid SHA256 for {name} path: {path}")
        entries.append((path, normalized, expected))
    return entries


def verify_archived_sources(mapping, snapshots, *, missing_prefix, hash_function=sha256):
    """Verify every live source and snapshot, admitting narrowly scoped absence.

    A missing source strictly beneath ``missing_prefix`` may be represented by
    an independently hashed snapshot with the exact expected SHA256. Existing
    source mismatches always fail, even when a snapshot has the historical
    bytes. Every declared snapshot must exist and match its digest, including
    snapshots not selected for fallback. No files are created or imported.

    Archive fallback proves availability of historical bytes only. It does not
    establish that a missing live source exists or that a model can run.
    """
    sources = _entries(mapping, "source")
    archives = _entries(snapshots, "snapshot")
    prefix = Path(missing_prefix)
    if not prefix.is_absolute():
        raise ValueError("missing_prefix must be absolute")
    prefix = prefix.resolve()
    if prefix.exists() and not prefix.is_dir():
        raise ValueError("missing_prefix must identify a directory")

    by_digest = {}
    for path, normalized, expected in archives:
        if not path.is_file():
            raise ValueError(f"snapshot file missing: {path}")
        if hash_function(path) != expected:
            raise ValueError(f"snapshot SHA256 mismatch: {path}")
        by_digest.setdefault(expected, []).append((path, normalized))

    missing = []
    live_count = 0
    for path, normalized, expected in sources:
        if path.exists() or path.is_symlink():
            if not path.is_file():
                raise ValueError(f"live source is not a readable file: {path}")
            if hash_function(path) != expected:
                raise ValueError(f"live source SHA256 mismatch: {path}")
            live_count += 1
            continue
        if normalized == prefix or not normalized.is_relative_to(prefix):
            raise ValueError(f"missing source outside allowed prefix: {path}")
        candidates = by_digest.get(expected, ())
        if not candidates:
            raise ValueError(f"no exact verified snapshot for missing source: {path}")
        archive, _ = min(candidates, key=lambda item: str(item[1]))
        missing.append(
            dict(
                original_path=str(path),
                expected_sha256=expected,
                archive_ref=dict(path=str(archive), sha256=expected),
                live_source_present=False,
            )
        )
    return dict(
        verified_live_source_count=live_count,
        verified_snapshot_count=len(archives),
        missing_sources=missing,
        historical_bytes_only=True,
    )
