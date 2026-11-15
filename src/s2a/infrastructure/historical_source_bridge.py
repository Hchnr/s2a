"""Verify explicitly pinned historical Git blobs beside current source files."""

import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

from s2a.infrastructure.artifacts import REPO

MANIFEST = REPO / "configs/tracking/joint_control/historical_solver_source_bridge_20261116.json"
MANIFEST_SHA256 = "76e99de6807e020a2bc8a8d68991b86e1358c91247dfee733dcee5c3dcaed8b5"
SCHEMA = "s2a.historical-solver-source-bridge.v1"


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def load_manifest():
    try:
        current = reference(MANIFEST)
    except OSError as error:
        raise ValueError("The pinned historical source bridge manifest is unavailable") from error
    if current["sha256"] != MANIFEST_SHA256:
        raise ValueError("The pinned historical source bridge manifest changed")
    data = json.loads(MANIFEST.read_text())
    if set(data) != {"schema", "bridges"} or data["schema"] != SCHEMA or not isinstance(data["bridges"], list):
        raise ValueError("The pinned historical source bridge manifest is malformed")
    return data


def source_refs():
    """Bind both the bridge implementation and its pinned manifest to new runs."""
    load_manifest()
    return [reference(Path(__file__)), reference(MANIFEST)]


def git_blob(root, revision, path):
    try:
        return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=root, stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("The pinned historical Git blob is unavailable") from error


def compare_sources(recorded, current, manifest, *, root, extensions, read_blob, require_extensions):
    """Reject everything except exact references and explicitly pinned bridges."""
    root = Path(root).resolve()
    if type(require_extensions) is not bool or set(manifest) != {"schema", "bridges"} or manifest["schema"] != SCHEMA:
        raise ValueError("A valid manifest and explicit extension policy are required")
    if not isinstance(recorded, list) or not isinstance(current, list) or not isinstance(manifest["bridges"], list):
        raise ValueError("Ordered source-reference lists and bridge rows are required")
    rows = {}
    for row in manifest["bridges"]:
        if not isinstance(row, dict) or set(row) != {
            "path",
            "historical_sha256",
            "historical_commit",
            "current_sha256",
        }:
            raise ValueError("A bridge must pin its path and both source versions")
        if any(not isinstance(row[key], str) for key in row):
            raise ValueError("Bridge paths, commits and hashes must be strings")
        rel = Path(row["path"])
        path = (root / rel).resolve()
        if rel.is_absolute() or ".." in rel.parts or not path.is_relative_to(root) or str(path) in rows:
            raise ValueError("Each bridge path must uniquely belong to the current repository")
        if not all(re.fullmatch(r"[0-9a-f]{64}", row[key]) for key in ("historical_sha256", "current_sha256")):
            raise ValueError("Both bridge hashes must be full lowercase SHA-256 values")
        if not re.fullmatch(r"[0-9a-f]{40}", row["historical_commit"]):
            raise ValueError("The historical Git commit must be a full object id")
        rows[str(path)] = row
    extension_paths = {str(Path(path).resolve()) for path in extensions}
    if len(extension_paths) != len(extensions):
        raise ValueError("Bridge extensions must be unique")
    if any(not isinstance(ref, dict) or set(ref) != {"path", "sha256"} for ref in recorded + current):
        raise ValueError("Canonical source references are required")
    added = [ref for ref in current if ref.get("path") in extension_paths]
    counts = Counter(ref["path"] for ref in added)
    if added and (set(counts) != extension_paths or len(set(counts.values())) != 1):
        raise ValueError("Bridge source extensions must appear in complete matching groups")
    if require_extensions and not added:
        raise ValueError("New registrations must include the bridge source extensions")
    for ref in added:
        if reference(ref["path"]) != ref:
            raise ValueError("A bridge source extension changed")
    prior_extensions = [ref for ref in recorded if ref["path"] in extension_paths]
    prior_counts = Counter(ref["path"] for ref in prior_extensions)
    if prior_extensions and (set(prior_counts) != extension_paths or len(set(prior_counts.values())) != 1):
        raise ValueError("Historical bridge extensions must appear in complete matching groups")
    filtered = current if prior_extensions else [ref for ref in current if ref["path"] not in extension_paths]
    if len(recorded) != len(filtered):
        raise ValueError("Historical source-reference count changed")
    bridges, current_bytes, historical_blobs = [], {}, {}
    for old, new in zip(recorded, filtered, strict=True):
        if (
            not isinstance(old, dict)
            or not isinstance(new, dict)
            or set(old) != {"path", "sha256"}
            or set(new) != set(old)
        ):
            raise ValueError("Canonical source references are required")
        if old["path"] != new["path"]:
            raise ValueError("Historical source-reference paths or order changed")
        if new["path"] not in current_bytes:
            current_bytes[new["path"]] = reference(new["path"])
        if current_bytes[new["path"]] != new:
            raise ValueError("The current source bytes changed")
        if old == new:
            continue
        row = rows.get(old["path"])
        if row is None or old["sha256"] != row["historical_sha256"] or new["sha256"] != row["current_sha256"]:
            raise ValueError("An unregistered historical source difference was found")
        identity = (row["historical_commit"], row["path"])
        if identity not in historical_blobs:
            historical_blobs[identity] = read_blob(root, *identity)
        blob = historical_blobs[identity]
        if hashlib.sha256(blob).hexdigest() != row["historical_sha256"]:
            raise ValueError("The historical Git blob does not match its original SHA-256")
        bridges.append(dict(row))
    return dict(matched_sources=len(recorded), bridges=bridges, extensions=added)


def verify(recorded, current, *, require_extensions):
    manifest = load_manifest()
    return compare_sources(
        recorded,
        current,
        manifest,
        root=REPO,
        extensions=(Path(__file__), MANIFEST),
        read_blob=git_blob,
        require_extensions=require_extensions,
    )
